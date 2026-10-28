import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
TELA_DIR = THIS_DIR.parent
if str(TELA_DIR) not in sys.path:
    sys.path.insert(0, str(TELA_DIR))

from analyze_attention import build_model, load_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Export learned TELA relation bias parameters.")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def calc_patch_num(seq_len, patch_len, stride):
    remainder = (seq_len - patch_len) % stride
    pad_right = (stride - remainder) % stride
    return (seq_len + pad_right - patch_len) // stride + 1


def patch_labels(seq_len, patch_len, stride, minutes_per_step=15):
    labels = []
    for patch_idx in range(calc_patch_num(seq_len, patch_len, stride)):
        start_step = patch_idx * stride - seq_len
        end_step = patch_idx * stride + patch_len - seq_len
        start_hour = start_step * minutes_per_step / 60.0
        end_hour = end_step * minutes_per_step / 60.0
        end_text = "now" if abs(end_hour) < 1e-8 else f"{end_hour:g}h"
        labels.append(f"{start_hour:g}~{end_text}")
    return labels


def savefig(path, dpi):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def draw_heatmap(values, xlabels, ylabels, title, xlabel, ylabel, path, dpi, cmap="coolwarm"):
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    vmax = max(vmax, 1e-8)
    plt.figure(figsize=(max(5.6, len(xlabels) * 0.7), max(4.4, len(ylabels) * 0.35)))
    ax = plt.gca()
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    savefig(path, dpi)


def draw_bar(values, labels, title, ylabel, path, dpi):
    colors = ["#4477AA" if value >= 0 else "#CC6677" for value in values]
    plt.figure(figsize=(max(5.2, len(values) * 0.62), 4.2))
    ax = plt.gca()
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    savefig(path, dpi)


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def summarize_block(block, layer_idx, run_args, output_dir, dpi):
    rows = []
    figures = []
    rel_scale = float(getattr(block, "base_rel_bias_weight", getattr(block, "rel_bias_weight", 1.0)))

    temporal = tensor_to_numpy(block.temporal_bias).reshape(block.n_heads, -1).mean(axis=1) * rel_scale
    channel_raw = tensor_to_numpy(block.channel_bias).reshape(block.n_heads, -1) * rel_scale
    channel = channel_raw.mean(axis=1)

    wind_lag_raw = tensor_to_numpy(block.wind_lag_bias) * rel_scale
    if wind_lag_raw.ndim == 4:
        wind_lag = wind_lag_raw.mean(axis=2)
    else:
        wind_lag = wind_lag_raw

    head_labels = [f"H{idx}" for idx in range(block.n_heads)]
    patch_names = patch_labels(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    horizon_names = [f"h{idx + 1}" for idx in range(run_args.pred_len)]

    for head in range(block.n_heads):
        rows.append(
            {
                "layer": layer_idx,
                "head": head,
                "temporal_bias": float(temporal[head]),
                "channel_bias": float(channel[head]),
                "wind_lag_bias_mean": float(wind_lag[head].mean()),
                "wind_lag_bias_min": float(wind_lag[head].min()),
                "wind_lag_bias_max": float(wind_lag[head].max()),
            }
        )

    temporal_path = os.path.join(output_dir, f"relation_bias_temporal_layer{layer_idx}.png")
    channel_path = os.path.join(output_dir, f"relation_bias_channel_layer{layer_idx}.png")
    wind_mean_path = os.path.join(output_dir, f"relation_bias_wind_lag_head_mean_layer{layer_idx}.png")
    wind_horizon_lag_path = os.path.join(output_dir, f"relation_bias_wind_lag_horizon_patch_layer{layer_idx}.png")

    draw_bar(temporal, head_labels, f"Layer {layer_idx}: temporal relation bias by head", "Score bias", temporal_path, dpi)
    draw_bar(channel, head_labels, f"Layer {layer_idx}: exogenous/channel relation bias by head", "Score bias", channel_path, dpi)
    draw_bar(
        wind_lag.mean(axis=(1, 2)),
        head_labels,
        f"Layer {layer_idx}: wind-lag relation bias by head",
        "Mean score bias",
        wind_mean_path,
        dpi,
    )
    draw_heatmap(
        wind_lag.mean(axis=0),
        patch_names,
        horizon_names,
        f"Layer {layer_idx}: mean wind-lag bias over heads",
        "Historical lag patch",
        "Forecast horizon",
        wind_horizon_lag_path,
        dpi,
    )
    figures.extend(
        [
            os.path.basename(temporal_path),
            os.path.basename(channel_path),
            os.path.basename(wind_mean_path),
            os.path.basename(wind_horizon_lag_path),
        ]
    )

    for head in range(block.n_heads):
        path = os.path.join(output_dir, f"relation_bias_wind_lag_layer{layer_idx}_head{head}.png")
        draw_heatmap(
            wind_lag[head],
            patch_names,
            horizon_names,
            f"Layer {layer_idx} Head {head}: wind-lag bias",
            "Historical lag patch",
            "Forecast horizon",
            path,
            dpi,
        )
        figures.append(os.path.basename(path))

    return rows, figures


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    payload, run_args = load_run(args.result_file)
    checkpoint = args.checkpoint or payload.get("train_summary", {}).get("checkpoint", "")
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    run_args.use_gpu = bool((not args.no_use_gpu) and torch.cuda.is_available())
    run_args.gpu = args.gpu
    device = torch.device(f"cuda:{args.gpu}" if run_args.use_gpu else "cpu")
    model = build_model(run_args, checkpoint, device)
    output_dir = args.output_dir
    if not output_dir:
        result_dir = os.path.dirname(os.path.abspath(args.result_file))
        output_dir = os.path.join(result_dir, "attention_analysis", "relation_bias")
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    figures = []
    for layer_idx, block in enumerate(model.blocks):
        rows, layer_figures = summarize_block(block, layer_idx, run_args, output_dir, args.dpi)
        all_rows.extend(rows)
        figures.extend(layer_figures)

    csv_path = os.path.join(output_dir, "relation_bias_summary.csv")
    write_csv(csv_path, all_rows)
    manifest = {
        "result_file": os.path.abspath(args.result_file),
        "checkpoint": os.path.abspath(checkpoint),
        "output_dir": os.path.abspath(output_dir),
        "relation_bias_weight": float(getattr(run_args, "rel_bias_weight", 1.0)),
        "use_temporal_bias": bool(getattr(run_args, "use_temporal_bias", True)),
        "use_exog_bias": bool(getattr(run_args, "use_exog_bias", True)),
        "use_wind_lag_bias": bool(getattr(run_args, "use_wind_lag_bias", True)),
        "figures": sorted(figures),
        "summary_csv": os.path.basename(csv_path),
    }
    with open(os.path.join(output_dir, "relation_bias_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved relation bias analysis to: {output_dir}")


if __name__ == "__main__":
    main()
