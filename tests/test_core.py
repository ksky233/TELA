import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import DatasetWideForecast, encode_wind_direction
from layers import TELAAttentionBlock
from models.TELA import Model
from run import get_args
from tests.synthetic import write_synthetic_csv
from utils.checkpoints import load_checkpoint
from utils.results import result_file_path, save_json_result


def model_args(**overrides):
    args = vars(get_args(["--no-use-gpu"])).copy()
    args.update(enc_in=16, target_index=15)
    args.update(overrides)
    return SimpleNamespace(**args)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_conference_defaults(self):
        args = get_args(["--no-use-gpu"])
        self.assertEqual((args.batch_size, args.train_epochs, args.patience), (512, 50, 10))
        self.assertEqual((args.loss, args.mae_loss_weight), ("mse_mae", 0.5))
        self.assertEqual(args.wind_dir_encoding, "sincos")
        self.assertFalse(args.use_revin)
        self.assertFalse(args.use_future_exog)
        self.assertTrue(Path(args.root_path).is_absolute())

    def test_tag_requires_experiment(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            get_args(["--turned", "seed2026"])

    def test_aliases(self):
        args = get_args(["--exp-name", "test", "--tuned", "a", "--seed", "2027", "--bs", "8"])
        self.assertEqual((args.turned, args.random_seed, args.batch_size), ("a", 2027, 8))

    def test_forward_shape_and_parameter_count(self):
        model = Model(model_args()).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 96, 16))
        self.assertEqual(tuple(output.shape), (2, 16, 1))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 339217)
        self.assertTrue(torch.isfinite(output).all())

    def test_main_model_ignores_future_values(self):
        model = Model(model_args(d_model=16, d_ff=32, n_heads=2, e_layers=1)).eval()
        x = torch.randn(2, 96, 16)
        with torch.no_grad():
            a = model(x, x_dec=torch.randn(2, 64, 16))
            b = model(x, x_dec=torch.randn(2, 64, 16) * 10000)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_token_grid_flattening(self):
        model = Model(model_args(d_model=16, d_ff=32, n_heads=2)).eval()
        with torch.no_grad():
            sequence, grid = model.history_tokenizer(torch.randn(2, 96, 16))
        self.assertEqual(tuple(grid.shape), (2, 16, 6, 16))
        torch.testing.assert_close(sequence, grid.reshape(2, 96, 16))

    def make_block(self, **overrides):
        kwargs = dict(d_model=16, n_heads=2, d_ff=32, n_vars=3,
                      patch_num=2, pred_len=4, target_index=2, top_k=3)
        kwargs.update(overrides)
        return TELAAttentionBlock(**kwargs)

    def test_temporal_mask(self):
        block = self.make_block()
        self.assertTrue(block.temporal_mask[0, 1])
        self.assertFalse(block.temporal_mask[0, 2])
        self.assertFalse(block.temporal_mask[0, 0])
        self.assertFalse(block.temporal_mask[6:, :].any())

    def test_exogenous_mask(self):
        block = self.make_block()
        self.assertTrue(block.channel_mask[0, 2])
        self.assertFalse(block.channel_mask[0, 3])
        self.assertFalse(block.channel_mask[0, 0])

    def test_lag_mask_excludes_power(self):
        block = self.make_block()
        self.assertTrue(block.wind_lag_mask[:, :4].all())
        self.assertFalse(block.wind_lag_mask[:, 4:].any())

    def test_topk_and_row_normalization(self):
        block = self.make_block().eval()
        _, attention = block(torch.randn(2, 10, 16))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 2, 10))
        self.assertTrue(((attention > 0).sum(dim=-1) == 3).all())

    def test_disabled_bias_stays_zero(self):
        block = self.make_block(use_temporal_bias=False, use_exog_bias=False, use_wind_lag_bias=False)
        with torch.no_grad():
            block.temporal_bias.fill_(1)
            block.channel_bias.fill_(1)
            block.wind_lag_bias.fill_(1)
        self.assertEqual(torch.count_nonzero(block._relation_bias(torch.float32, "cpu")), 0)

    def test_all_biases_receive_gradients(self):
        block = self.make_block(top_k=0)
        output, _ = block(torch.randn(2, 10, 16))
        output[..., 0].square().mean().backward()
        for name in ("temporal_bias", "channel_bias", "wind_lag_bias"):
            gradient = getattr(block, name).grad
            self.assertIsNotNone(gradient)
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_wind_encoding(self):
        frame = pd.DataFrame({"wind_dir_10m": [0.0, 90.0], "power": [2.0, 3.0]})
        encoded = encode_wind_direction(frame, "sincos", "date", "power")
        self.assertEqual(list(encoded), ["wind_dir_10m_sin", "wind_dir_10m_cos", "power"])
        np.testing.assert_allclose(encoded.wind_dir_10m_sin, [0, 1], atol=1e-6)
        np.testing.assert_allclose(encoded.wind_dir_10m_cos, [1, 0], atol=1e-6)
        np.testing.assert_array_equal(encoded.power, frame.power)
        self.assertIn("wind_dir_10m", frame)

    def test_train_only_scaler_and_split_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = write_synthetic_csv(Path(directory) / "data.csv")
            kwargs = dict(root_path=directory, data_path="data.csv", size=(16, 8, 4), wind_dir_encoding="sincos")
            train = DatasetWideForecast(**kwargs, flag="train")
            val = DatasetWideForecast(**kwargs, flag="val")
            test = DatasetWideForecast(**kwargs, flag="test")
            train_end = int(len(raw) * 0.7)
            val_end = train_end + int(len(raw) * 0.1)
            np.testing.assert_allclose(train.scaler.mean[0, -1], raw.power[:train_end].to_numpy(np.float32).mean(), rtol=1e-6)
            np.testing.assert_array_equal(train.scaler.mean, val.scaler.mean)
            np.testing.assert_array_equal(train.scaler.std, test.scaler.std)
            _, target, _, _ = test[0]
            restored = test.inverse_transform(target[-4:, -1:].numpy())
            np.testing.assert_allclose(restored[:, 0], raw.power[val_end:val_end + 4], rtol=1e-6)
            self.assertEqual(len(test), len(raw) - val_end - 4 + 1)

    def test_single_target_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            write_synthetic_csv(Path(directory) / "data.csv")
            data = DatasetWideForecast(directory, "data.csv", features="S", size=(16, 8, 4))
            self.assertEqual(tuple(data[0][0].shape), (16, 1))
            self.assertEqual(data.target_index, 0)

    def test_checkpoint_round_trip(self):
        model = Model(model_args(d_model=16, d_ff=32, n_heads=2, e_layers=1)).eval()
        other = Model(model_args(d_model=16, d_ff=32, n_heads=2, e_layers=1)).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(model.state_dict(), path)
            load_checkpoint(other, path)
        x = torch.randn(2, 96, 16)
        with torch.no_grad():
            torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)

    def test_checkpoint_rejects_missing_key(self):
        model = Model(model_args(d_model=16, d_ff=32, n_heads=2, e_layers=1))
        state = model.state_dict()
        del state["query_type"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, path)

    def test_checkpoint_allows_only_zero_legacy_parameter(self):
        model = Model(model_args(d_model=16, d_ff=32, n_heads=2, e_layers=1))
        state = model.state_dict()
        state["blocks.0.query_power_bias"] = torch.zeros(2, 16, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save(state, path)
            with self.assertWarns(UserWarning):
                load_checkpoint(model, path)
            state["blocks.0.query_power_bias"].fill_(1)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, path)

    def test_result_naming_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(exp_name="trial", turned="a", result_path=directory)
            self.assertEqual(Path(result_file_path(args)).name, "turnd-a.json")
            save_json_result(args, {"metrics": {"mse": 2}})
            save_json_result(args, {"metrics": {"mse": 1}})
            payload = json.loads(Path(result_file_path(args)).read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics"]["mse"], 1)
            args.turned = ""
            self.assertEqual(Path(result_file_path(args)).name, "metres.json")


if __name__ == "__main__":
    unittest.main()
