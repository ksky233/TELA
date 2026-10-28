"""Export chronological multi-horizon test predictions without shifting curves."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_attention import build_model, load_run  # noqa: E402
from data_provider import data_provider  # noqa: E402
from utils.audit import sha256_file  # noqa: E402
from utils.metrics import metric  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    payload, args = load_run(cli.result_file)
    if args.output_mode != "point":
        parser.error("This export entry uses point forecasts; choose a point-mode result JSON")
    if cli.data_root:
        args.root_path = str(cli.data_root.resolve())
    checkpoint = cli.checkpoint or Path(payload.get("train_summary", {}).get("checkpoint") or
                                        Path(args.checkpoints) / payload["setting"] / "checkpoint.pth")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    output = cli.output_dir or cli.result_file.resolve().parent / "forecast_exports"
    if output.exists() and any(output.iterdir()) and not cli.overwrite:
        parser.error(f"Output directory is not empty: {output}; use --overwrite to replace exports")
    output.mkdir(parents=True, exist_ok=True)
    args.use_gpu = not cli.cpu and torch.cuda.is_available()
    args.gpu = cli.gpu
    args.batch_size = cli.batch_size
    args.num_workers = 0
    device = torch.device(f"cuda:{cli.gpu}" if args.use_gpu else "cpu")
    model = build_model(args, checkpoint, device)
    dataset, loader = data_provider(args, "test")
    predictions, truths = [], []
    with torch.no_grad():
        for x, y, x_mark, y_mark in loader:
            pred = model(x.float().to(device), x_mark.float().to(device),
                         y.float().to(device), y_mark.float().to(device))
            predictions.append(pred[:, -args.pred_len:, :].cpu().numpy())
            truths.append(y[:, -args.pred_len:, args.target_index:args.target_index + 1].numpy())
    pred_norm, true_norm = np.concatenate(predictions), np.concatenate(truths)
    pred_original = dataset.inverse_transform(pred_norm)
    true_original = dataset.inverse_transform(true_norm)
    for name, values in (("pred_norm", pred_norm), ("true_norm", true_norm),
                         ("pred_original", pred_original), ("true_original", true_original)):
        np.save(output / f"{name}.npy", values)
    with (output / "horizon_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["horizon", "mse_norm", "mae_norm", "rmse_norm",
                                                    "mse_original", "mae_original", "rmse_original"])
        writer.writeheader()
        for h in range(args.pred_len):
            row = {"horizon": h + 1}
            for label, pred, true in (("norm", pred_norm, true_norm),
                                       ("original", pred_original, true_original)):
                values = metric(pred[:, h], true[:, h])
                row.update({f"{key}_{label}": values[key] for key in ("mse", "mae", "rmse")})
            writer.writerow(row)
    summary = {
        "result_file": str(cli.result_file.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "shape": list(pred_norm.shape),
        "ordering": "Chronological test origins at stride 1; horizon 1 is the next target step.",
        "normalization": "Per-channel mean/std fitted on the training partition only.",
        "target_mean": float(dataset.scaler.mean[0, dataset.target_index]),
        "target_std": float(dataset.scaler.std[0, dataset.target_index]),
        "metrics_norm": metric(pred_norm, true_norm),
        "metrics_original": metric(pred_original, true_original),
    }
    (output / "export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved test predictions and horizon-wise metrics: {output}")


if __name__ == "__main__":
    main()
