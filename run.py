import argparse
import json
import os
import sys

import torch

from exp.exp_forecast import ExpForecast
from utils.audit import collect_audit
from utils.results import save_json_result
from utils.tools import set_seed, time_now


def _bool_flag(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false", help=f"disable {help_text}")
    parser.set_defaults(**{name.replace("-", "_"): default})


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="TELA forecasting runner")

    parser.add_argument("--exp-name", type=str, default="", help="experiment name under result/")
    parser.add_argument("--turned", "--tuned", dest="turned", type=str, default="", help="tuning tag; requires --exp-name")
    parser.add_argument("--is-training", type=int, default=1)

    parser.add_argument("--model", type=str, default="TELA")
    parser.add_argument("--root-path", type=str, default="datasets/wind")
    parser.add_argument("--data-path", type=str, default="grid_wind_site1_99MW.csv")
    parser.add_argument("--time-col", type=str, default="date")
    parser.add_argument("--target", type=str, default="power")
    parser.add_argument("--features", type=str, default="MS", choices=["S", "MS"])
    parser.add_argument("--freq", type=str, default="15min")
    parser.add_argument("--wind-dir-encoding", type=str, default="sincos", choices=["none", "sincos"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--data-limit", type=int, default=0, help="use first N rows for smoke tests; 0 means all")

    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=16)

    parser.add_argument("--enc-in", type=int, default=0, help="auto-filled from data when 0")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--head-type",
        type=str,
        default="persistence",
        choices=["persistence", "direct", "gated_shape", "cumulative_delta"],
        help="prediction head: persistence residual, direct shape, gated persistence-shape blend, or cumulative deltas",
    )
    parser.add_argument("--head-gate-init", type=float, default=0.85, help="initial persistence gate for gated_shape")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=16)

    parser.add_argument("--top-k", "--graph-top-k", dest="graph_top_k", type=int, default=32, help="0 means dense attention")
    parser.add_argument("--rel-bias-weight", type=float, default=1.0)
    parser.add_argument("--wind-lag-window", type=int, default=0, help="0 means all history patches")
    _bool_flag(parser, "use-temporal-bias", True, "use temporal relation bias")
    _bool_flag(parser, "use-exog-bias", True, "use exogenous relation bias")
    _bool_flag(parser, "use-wind-lag-bias", True, "use wind-lag relation bias")
    _bool_flag(parser, "use-future-exog", False, "condition future queries on observed future exogenous variables")

    parser.add_argument("--batch-size", "--bs", dest="batch_size", type=int, default=512)
    parser.add_argument("--train-epochs", "--ep", dest="train_epochs", type=int, default=50)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", type=str, default="mse_mae", choices=["mse", "mae", "mse_mae", "ramp_mse_mae"])
    parser.add_argument("--mae-loss-weight", type=float, default=0.5)
    parser.add_argument("--ramp-loss-weight", type=float, default=1.0)
    parser.add_argument("--ramp-focus-weight", type=float, default=0.0)
    parser.add_argument("--ramp-sign-weight", type=float, default=0.0)
    parser.add_argument("--output-mode", type=str, default="point", choices=["point", "quantile"])
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.05,0.1,0.2,0.5,0.8,0.9,0.95",
        help="comma-separated quantiles used when --output-mode quantile",
    )
    parser.add_argument(
        "--select-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_mae", "val_mse", "val_wape", "val_composite"],
        help="validation metric used for checkpoint selection and early stopping",
    )
    parser.add_argument("--lradj", type=str, default="none", choices=["none", "type1", "cosine"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-seed", "--random_seed", "--seed", dest="random_seed", type=int, default=2026)

    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--result-path", type=str, default="result")

    _bool_flag(parser, "use-gpu", True, "use gpu")
    parser.add_argument("--gpu", type=int, default=0)
    _bool_flag(parser, "use-amp", False, "use automatic mixed precision")
    _bool_flag(parser, "inverse", True, "save metrics in original scale")
    _bool_flag(parser, "use-revin", False, "use RevIN normalization")

    args = parser.parse_args(argv)
    if args.turned and not args.exp_name:
        parser.error("--turned requires --exp-name")
    args.quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]
    if args.output_mode == "quantile":
        if not args.quantiles:
            parser.error("--output-mode quantile requires --quantiles")
        if any(q <= 0.0 or q >= 1.0 for q in args.quantiles):
            parser.error("--quantiles must be between 0 and 1")
        args.quantiles = sorted(args.quantiles)

    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    args.use_amp = bool(args.use_amp and args.use_gpu)
    args.avg_method = "ema"
    args.ema_decay = 0.995

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.root_path):
        args.root_path = os.path.join(base_dir, args.root_path)
    if not os.path.isabs(args.checkpoints):
        args.checkpoints = os.path.join(base_dir, args.checkpoints)
    if not os.path.isabs(args.result_path):
        args.result_path = os.path.join(base_dir, args.result_path)

    return args


def fill_data_dependent_args(args):
    import pandas as pd
    from data_provider.data_loader import encode_wind_direction

    file_path = args.data_path
    if not os.path.isabs(file_path):
        file_path = os.path.join(args.root_path, args.data_path)
    df = pd.read_csv(file_path, nrows=16)
    df = encode_wind_direction(df, args.wind_dir_encoding, args.time_col, args.target)
    numeric_cols = [
        col for col in df.columns if col != args.time_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if args.target not in numeric_cols:
        raise ValueError(f"target column '{args.target}' must be numeric")
    if args.features == "S":
        args.enc_in = 1
        cols_data = [args.target]
    else:
        cols_data = [col for col in numeric_cols if col != args.target] + [args.target]
        args.enc_in = len(cols_data)

    args.feature_names = cols_data
    args.target_index = cols_data.index(args.target)
    return args


def main():
    args = fill_data_dependent_args(get_args())
    set_seed(args.random_seed)

    setting_name = args.exp_name or "default"
    if args.turned:
        setting_name = f"{setting_name}_turnd_{args.turned}"

    print(f"[{time_now()}] Start experiment: {setting_name}")
    print(f"Args: {json.dumps(vars(args), ensure_ascii=False, default=str)}")

    exp = ExpForecast(args)
    train_summary = {}
    if args.is_training:
        train_summary = exp.train(setting_name)
        metrics = exp.test(setting_name, load=False)
    else:
        metrics = exp.test(setting_name, load=True)

    efficiency = exp.model_parameter_summary()
    for key in [
        "epochs_ran",
        "avg_epoch_seconds",
        "train_seconds",
        "train_peak_allocated_mb",
        "train_peak_reserved_mb",
    ]:
        if key in train_summary:
            efficiency[key] = train_summary[key]
    efficiency.update(getattr(exp, "test_summary", {}))

    result_path = save_json_result(
        args,
        {
            "setting": setting_name,
            "model": args.model,
            "metrics": metrics,
            "train_summary": train_summary,
            "efficiency": efficiency,
            "argv": [sys.executable, *sys.argv],
            "command_line": " ".join([sys.executable, *sys.argv]),
            "args": vars(args),
            "audit": collect_audit(args),
        },
    )

    print(f"[{time_now()}] Result saved: {result_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
