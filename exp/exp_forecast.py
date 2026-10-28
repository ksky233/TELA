import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider import data_provider
from exp.exp_basic import ExpBasic
from utils.checkpoints import load_checkpoint
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class MSEMAELoss(nn.Module):
    def __init__(self, mae_weight: float = 0.3):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        return self.mse(pred, true) + self.mae_weight * self.mae(pred, true)


class RampAwareMSEMAELoss(nn.Module):
    def __init__(
        self,
        mae_weight: float = 0.5,
        ramp_weight: float = 1.0,
        focus_weight: float = 0.0,
        sign_weight: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mae_weight = mae_weight
        self.ramp_weight = ramp_weight
        self.focus_weight = focus_weight
        self.sign_weight = sign_weight
        self.eps = eps
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def _ramp_weights(self, true_diff):
        if self.focus_weight <= 0.0:
            return 1.0
        scale = true_diff.detach().abs().mean(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        weights = 1.0 + self.focus_weight * true_diff.detach().abs() / scale
        return weights.clamp(max=10.0)

    def forward(self, pred, true, history_target=None):
        point_loss = self.mse(pred, true) + self.mae_weight * self.mae(pred, true)
        if pred.size(1) <= 1 and history_target is None:
            return point_loss

        if history_target is not None:
            pred_base = torch.cat([history_target, pred], dim=1)
            true_base = torch.cat([history_target, true], dim=1)
        else:
            pred_base = pred
            true_base = true
        pred_diff = pred_base[:, 1:, :] - pred_base[:, :-1, :]
        true_diff = true_base[:, 1:, :] - true_base[:, :-1, :]
        diff_error = pred_diff - true_diff
        weights = self._ramp_weights(true_diff)
        ramp_loss = (weights * diff_error.pow(2)).mean()
        ramp_loss = ramp_loss + self.mae_weight * (weights * diff_error.abs()).mean()

        if self.sign_weight > 0.0:
            sign_loss = (weights * torch.relu(-pred_diff * true_diff)).mean()
            ramp_loss = ramp_loss + self.sign_weight * sign_loss

        return point_loss + self.ramp_weight * ramp_loss


class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("quantiles", quantiles)

    def forward(self, pred, true):
        error = true - pred
        quantiles = self.quantiles.to(pred.device)
        loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        return loss.mean()


class WeightAverager:
    def __init__(self, model, args):
        self.decay = getattr(args, "ema_decay", 0.995)
        self.method = "ema"
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self._init_shadow(model)

    @property
    def enabled(self):
        return True

    @property
    def ready(self):
        return self.enabled and self.num_updates > 0

    def _named_params(self, model):
        return [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    def _init_shadow(self, model):
        self.shadow = {
            name: param.detach().clone()
            for name, param in self._named_params(model)
        }

    def update_after_step(self, model):
        if not self.shadow:
            self._init_shadow(model)
        with torch.no_grad():
            for name, param in self._named_params(model):
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        self.num_updates += 1

    def update_after_epoch(self, model, epoch):
        return

    def store(self, model):
        self.backup = {
            name: param.detach().clone()
            for name, param in self._named_params(model)
        }

    def copy_to(self, model):
        if not self.ready:
            return
        with torch.no_grad():
            for name, param in self._named_params(model):
                param.copy_(self.shadow[name])

    def restore(self, model):
        if not self.backup:
            return
        with torch.no_grad():
            for name, param in self._named_params(model):
                param.copy_(self.backup[name])
        self.backup = {}


class ExpForecast(ExpBasic):
    def model_parameter_summary(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "total_params_m": float(total_params / 1_000_000),
            "trainable_params_m": float(trainable_params / 1_000_000),
        }

    def _reset_cuda_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def _cuda_peak_memory_summary(self, prefix):
        if self.device.type != "cuda":
            return {
                f"{prefix}_peak_allocated_mb": 0.0,
                f"{prefix}_peak_reserved_mb": 0.0,
            }
        torch.cuda.synchronize(self.device)
        return {
            f"{prefix}_peak_allocated_mb": float(torch.cuda.max_memory_allocated(self.device) / 1024**2),
            f"{prefix}_peak_reserved_mb": float(torch.cuda.max_memory_reserved(self.device) / 1024**2),
        }

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        if getattr(self.args, "output_mode", "point") == "quantile":
            return PinballLoss(self.args.quantiles)
        if self.args.loss == "mae":
            return nn.L1Loss()
        if self.args.loss == "mse":
            return nn.MSELoss()
        if self.args.loss == "mse_mae":
            return MSEMAELoss(mae_weight=self.args.mae_loss_weight)
        if self.args.loss == "ramp_mse_mae":
            return RampAwareMSEMAELoss(
                mae_weight=self.args.mae_loss_weight,
                ramp_weight=self.args.ramp_loss_weight,
                focus_weight=self.args.ramp_focus_weight,
                sign_weight=self.args.ramp_sign_weight,
            )
        raise ValueError(f"Unsupported loss: {self.args.loss}")

    def _target_slice(self):
        return -1 if self.args.features == "MS" else 0

    def _is_quantile(self):
        return getattr(self.args, "output_mode", "point") == "quantile"

    def _prepare_output_target(self, outputs, batch_y):
        f_dim = self._target_slice()
        target = batch_y[:, -self.args.pred_len :, f_dim:]
        if self._is_quantile():
            outputs = outputs[:, -self.args.pred_len :, :]
            outputs, _ = torch.sort(outputs, dim=-1)
            return outputs, target
        return outputs[:, -self.args.pred_len :, f_dim:], target

    def _criterion_loss(self, criterion, outputs, target, batch_x=None):
        if isinstance(criterion, RampAwareMSEMAELoss) and batch_x is not None:
            history_target = batch_x[:, -1:, self.args.target_index : self.args.target_index + 1]
            return criterion(outputs, target, history_target=history_target)
        return criterion(outputs, target)

    def _p50_from_quantiles(self, preds):
        quantiles = np.asarray(self.args.quantiles, dtype=np.float64)
        median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
        return preds[..., median_idx : median_idx + 1]

    def _quantile_metrics(self, preds, trues, suffix=""):
        quantiles = np.asarray(self.args.quantiles, dtype=np.float64)
        pred_q = np.asarray(preds, dtype=np.float64)
        true = np.asarray(trues, dtype=np.float64)
        error = true - pred_q
        q = quantiles.reshape(1, 1, -1)
        pinball = np.maximum(q * error, (q - 1.0) * error)

        result = {
            f"pinball_loss{suffix}": float(pinball.mean()),
            f"crps{suffix}": float(2.0 * pinball.mean()),
            f"quantile_crossing_rate{suffix}": float(np.mean(np.diff(pred_q, axis=-1) < 0.0)),
        }

        for lo, hi, name in [(0.2, 0.8, "60"), (0.1, 0.9, "80"), (0.05, 0.95, "90")]:
            lo_idx = np.where(np.isclose(quantiles, lo))[0]
            hi_idx = np.where(np.isclose(quantiles, hi))[0]
            if len(lo_idx) == 0 or len(hi_idx) == 0:
                continue
            lower = pred_q[..., int(lo_idx[0]) : int(lo_idx[0]) + 1]
            upper = pred_q[..., int(hi_idx[0]) : int(hi_idx[0]) + 1]
            result[f"cov{name}{suffix}"] = float(np.mean((true >= lower) & (true <= upper)))
            result[f"width{name}{suffix}"] = float(np.mean(upper - lower))
        return result

    @staticmethod
    def _unpack_model_output(model_output):
        if isinstance(model_output, tuple):
            output = model_output[0]
            aux_loss = model_output[1] if len(model_output) > 1 else None
            return output, aux_loss
        return model_output, None

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, _ = self._unpack_model_output(
                    self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                )
                outputs, batch_y = self._prepare_output_target(outputs, batch_y)
                loss = self._criterion_loss(criterion, outputs, batch_y, batch_x)
                total_loss.append(loss.item())
        self.model.train()
        return float(np.average(total_loss))

    def vali_with_metrics(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, _ = self._unpack_model_output(
                    self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                )
                outputs, batch_y = self._prepare_output_target(outputs, batch_y)
                loss = self._criterion_loss(criterion, outputs, batch_y, batch_x)
                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        self.model.train()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        result = {"loss": float(np.average(total_loss))}
        point_preds = self._p50_from_quantiles(preds) if self._is_quantile() else preds
        result.update({f"{key}_norm": value for key, value in metric(point_preds, trues).items()})
        if self._is_quantile():
            result.update(self._quantile_metrics(preds, trues, suffix="_norm"))
        return result

    def _selection_score(self, val_result, best_val_mse_norm=None):
        metric_name = getattr(self.args, "select_metric", "val_loss")
        if metric_name == "val_loss":
            return val_result["loss"]
        if metric_name == "val_mae":
            return val_result["mae_norm"]
        if metric_name == "val_mse":
            return val_result["mse_norm"]
        if metric_name == "val_wape":
            return val_result["wape_norm"]
        if metric_name == "val_composite":
            return val_result["mae_norm"] + val_result["mse_norm"]
        raise ValueError(f"Unsupported select_metric: {metric_name}")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.amp.GradScaler("cuda") if self.args.use_amp else None
        averager = WeightAverager(self.model, self.args)

        best_val_loss = np.inf
        best_select_score = np.inf
        best_val_mse_norm = np.inf
        best_epoch = 0
        start_time = time.time()
        epoch_times = []
        self._reset_cuda_peak_memory()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            self.model.train()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                with torch.amp.autocast("cuda", enabled=self.args.use_amp):
                    outputs, aux_loss = self._unpack_model_output(
                        self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    )
                    outputs, target = self._prepare_output_target(outputs, batch_y)
                    loss = self._criterion_loss(criterion, outputs, target, batch_x)
                    if aux_loss is not None:
                        loss = loss + aux_loss

                train_loss.append(loss.item())
                if scaler is None:
                    loss.backward()
                    model_optim.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                averager.update_after_step(self.model)

            train_loss = float(np.average(train_loss))
            averager.update_after_epoch(self.model, epoch + 1)
            if averager.ready:
                averager.store(self.model)
                averager.copy_to(self.model)

            val_result = self.vali_with_metrics(vali_data, vali_loader, criterion)
            test_result = self.vali_with_metrics(test_data, test_loader, criterion)
            vali_loss = val_result["loss"]
            test_loss = test_result["loss"]

            if vali_loss < best_val_loss:
                best_val_loss = vali_loss
            best_val_mse_norm = min(best_val_mse_norm, val_result["mse_norm"])
            select_score = self._selection_score(val_result, best_val_mse_norm)
            if select_score < best_select_score:
                best_select_score = select_score
                best_epoch = epoch + 1

            epoch_seconds = time.time() - epoch_time
            epoch_times.append(epoch_seconds)
            print(
                "Epoch: {0}, Steps: {1}, Train Loss: {2:.7f}, Vali Loss: {3:.7f}, Test Loss: {4:.7f}, Select({5}): {6:.7f}, Val MAE: {7:.7f}, Val MSE: {8:.7f}, Time: {9:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                    getattr(self.args, "select_metric", "val_loss"),
                    select_score,
                    val_result["mae_norm"],
                    val_result["mse_norm"],
                    epoch_seconds,
                )
            )
            early_stopping(select_score, self.model, path)
            if averager.ready:
                averager.restore(self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        load_checkpoint(self.model, best_model_path, self.device)
        train_seconds = time.time() - start_time
        return {
            **self.model_parameter_summary(),
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_mse_norm": float(best_val_mse_norm),
            "select_metric": getattr(self.args, "select_metric", "val_loss"),
            "best_select_score": float(best_select_score),
            "avg_method": getattr(self.args, "avg_method", "none"),
            "avg_updates": int(averager.num_updates),
            "epochs_ran": int(len(epoch_times)),
            "avg_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "train_seconds": float(train_seconds),
            "checkpoint": best_model_path,
            "train_size": len(train_data),
            "val_size": len(vali_data),
            "test_size": len(test_data),
            **self._cuda_peak_memory_summary("train"),
        }

    def test(self, setting, load=False):
        test_data, test_loader = self._get_data(flag="test")
        if load:
            checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            load_checkpoint(self.model, checkpoint, self.device)

        preds = []
        trues = []
        self.model.eval()
        self._reset_cuda_peak_memory()
        infer_start = time.time()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, _ = self._unpack_model_output(
                    self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                )
                outputs, batch_y = self._prepare_output_target(outputs, batch_y)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_seconds = time.time() - infer_start

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        point_preds = self._p50_from_quantiles(preds) if self._is_quantile() else preds
        norm_metrics = metric(point_preds, trues)
        result = {f"{key}_norm": value for key, value in norm_metrics.items()}
        if self._is_quantile():
            result.update(self._quantile_metrics(preds, trues, suffix="_norm"))

        if self.args.inverse:
            pred_inv = test_data.inverse_transform(point_preds)
            true_inv = test_data.inverse_transform(trues)
            result.update(metric(pred_inv, true_inv))
            if self._is_quantile():
                pred_q_inv = test_data.inverse_transform(preds)
                result.update(self._quantile_metrics(pred_q_inv, true_inv, suffix=""))

        result["samples"] = int(preds.shape[0])
        result["pred_shape"] = list(preds.shape)
        self.test_summary = {
            "inference_seconds": float(inference_seconds),
            "inference_ms_per_sample": float(inference_seconds * 1000.0 / max(int(preds.shape[0]), 1)),
            "test_batches": int(len(test_loader)),
            **self._cuda_peak_memory_summary("inference"),
        }
        return result
