import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_attention import collect_attention_matrices  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
TELA_DIR = THIS_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Plot physically grouped TELA attention heatmaps.")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
    )
    parser.add_argument("--analysis-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="include the target power variable in variable/lag heatmaps; disabled by default",
    )
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


def query_history_cube(matrix, n_vars, patch_num, pred_len):
    hist_n = n_vars * patch_num
    qh = matrix[hist_n : hist_n + pred_len, :hist_n]
    return qh.reshape(pred_len, n_vars, patch_num)


def selected_variable_indices(run_args, include_target):
    n_vars = int(run_args.enc_in)
    target_index = int(run_args.target_index)
    if include_target:
        return list(range(n_vars))
    return [idx for idx in range(n_vars) if idx != target_index]


def selected_variable_names(run_args, include_target):
    names = list(run_args.feature_names)
    return [names[idx] for idx in selected_variable_indices(run_args, include_target)]


def normalize_total(values):
    denom = float(values.sum())
    if denom <= 0:
        return values
    return values / denom


def normalize_rows(values):
    denom = values.sum(axis=1, keepdims=True)
    return np.divide(values, np.maximum(denom, 1e-12))


def draw_heatmap(ax, values, title, xlabel, ylabel, xticklabels, yticklabels, cmap, vmin=None, vmax=None):
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)
    return image


def plot_variable_lag(matrices, run_args, output_dir, dpi, include_target):
    n_vars = int(run_args.enc_in)
    patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    labels = patch_labels(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    variable_indices = selected_variable_indices(run_args, include_target)
    feature_names = selected_variable_names(run_args, include_target)
    source_name = "variable" if include_target else "exogenous variable"

    layer_maps = {}
    for layer, matrix in sorted(matrices.items()):
        cube = query_history_cube(matrix, n_vars, patch_num, run_args.pred_len)
        layer_maps[layer] = normalize_total(cube.sum(axis=0)[variable_indices, :])

    vmax = max(float(values.max()) for values in layer_maps.values())
    fig, axes = plt.subplots(
        1,
        len(layer_maps),
        figsize=(6.4 * len(layer_maps), 5.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (layer, values) in zip(axes[0], sorted(layer_maps.items())):
        image = draw_heatmap(
            ax,
            values,
            f"Layer {layer}: future query -> {source_name}-lag",
            "Historical lag patch",
            "Historical variable" if include_target else "Historical exogenous variable",
            labels,
            feature_names,
            "YlOrRd",
            0.0,
            vmax,
        )
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    savefig(os.path.join(output_dir, "heatmap_query_variable_lag.png"), dpi)


def plot_exogenous_variable_lag(matrices, run_args, output_dir, dpi):
    n_vars = int(run_args.enc_in)
    patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    labels = patch_labels(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    feature_names = list(run_args.feature_names)
    target_index = int(run_args.target_index)
    exog_indices = [idx for idx in range(n_vars) if idx != target_index]
    exog_names = [feature_names[idx] for idx in exog_indices]

    layer_maps = {}
    for layer, matrix in sorted(matrices.items()):
        cube = query_history_cube(matrix, n_vars, patch_num, run_args.pred_len)
        values = cube.sum(axis=0)[exog_indices, :]
        layer_maps[layer] = normalize_total(values)

    fig, axes = plt.subplots(
        1,
        len(layer_maps),
        figsize=(6.4 * len(layer_maps), 5.5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (layer, values) in zip(axes[0], sorted(layer_maps.items())):
        image = draw_heatmap(
            ax,
            values,
            f"Layer {layer}: future query -> exogenous variable-lag",
            "Historical lag patch",
            "Historical exogenous variable",
            labels,
            exog_names,
            "YlGnBu",
            0.0,
            float(values.max()),
        )
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    savefig(os.path.join(output_dir, "heatmap_query_exogenous_variable_lag.png"), dpi)


def plot_horizon_variable(matrices, run_args, output_dir, dpi, include_target):
    n_vars = int(run_args.enc_in)
    patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    variable_indices = selected_variable_indices(run_args, include_target)
    feature_names = selected_variable_names(run_args, include_target)
    source_name = "variable" if include_target else "exogenous variable"
    horizon_labels = [f"h{i + 1}" for i in range(int(run_args.pred_len))]

    for layer, matrix in sorted(matrices.items()):
        cube = query_history_cube(matrix, n_vars, patch_num, run_args.pred_len)
        values = normalize_rows(cube[:, variable_indices, :].sum(axis=2))
        plt.figure(figsize=(8.4, 6.2))
        ax = plt.gca()
        image = draw_heatmap(
            ax,
            values,
            f"Layer {layer}: horizon-wise {source_name} selection",
            "Historical variable" if include_target else "Historical exogenous variable",
            "Forecast horizon",
            feature_names,
            horizon_labels,
            "Blues",
        )
        ax.tick_params(axis="x", labelsize=8)
        plt.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
        savefig(os.path.join(output_dir, f"heatmap_horizon_variable_layer{layer}.png"), dpi)


def plot_horizon_lag(matrices, run_args, output_dir, dpi, include_target):
    n_vars = int(run_args.enc_in)
    patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    variable_indices = selected_variable_indices(run_args, include_target)
    source_name = "all variables" if include_target else "exogenous variables"
    lag_labels = patch_labels(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    horizon_labels = [f"h{i + 1}" for i in range(int(run_args.pred_len))]

    for layer, matrix in sorted(matrices.items()):
        cube = query_history_cube(matrix, n_vars, patch_num, run_args.pred_len)
        values = normalize_rows(cube[:, variable_indices, :].sum(axis=1))
        plt.figure(figsize=(5.4, 6.2))
        ax = plt.gca()
        image = draw_heatmap(
            ax,
            values,
            f"Layer {layer}: horizon-wise lag selection over {source_name}",
            "Historical lag patch",
            "Forecast horizon",
            lag_labels,
            horizon_labels,
            "viridis",
        )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        savefig(os.path.join(output_dir, f"heatmap_horizon_lag_layer{layer}.png"), dpi)


def plot_history_variable_interaction(matrices, run_args, output_dir, dpi, include_target):
    n_vars = int(run_args.enc_in)
    patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    hist_n = n_vars * patch_num
    variable_indices = selected_variable_indices(run_args, include_target)
    feature_names = selected_variable_names(run_args, include_target)
    source_name = "variable-to-variable" if include_target else "exogenous-to-exogenous"

    for layer, matrix in sorted(matrices.items()):
        hist = matrix[:hist_n, :hist_n].reshape(n_vars, patch_num, n_vars, patch_num)
        values = hist[np.ix_(variable_indices, range(patch_num), variable_indices, range(patch_num))]
        values = values.sum(axis=(1, 3))
        values = normalize_rows(values)
        plt.figure(figsize=(7.4, 6.3))
        ax = plt.gca()
        image = draw_heatmap(
            ax,
            values,
            f"Layer {layer}: historical {source_name} attention",
            "Key variable" if include_target else "Key exogenous variable",
            "Query variable" if include_target else "Query exogenous variable",
            feature_names,
            feature_names,
            "cividis",
        )
        ax.tick_params(axis="both", labelsize=8)
        plt.colorbar(image, ax=ax, fraction=0.036, pad=0.02)
        savefig(os.path.join(output_dir, f"heatmap_history_variable_interaction_layer{layer}.png"), dpi)


def read_relation_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_relation_lift_heatmap(analysis_dir, output_dir, dpi, include_target):
    rows = read_relation_rows(os.path.join(analysis_dir, "relation_attention.csv"))
    if not rows:
        return
    keep_relations = ["history_temporal_same_var", "history_exogenous_same_patch", "query_to_exogenous_lag"]
    labels = ["Hist. temporal", "Hist. exogenous", "Query-exogenous"]
    if include_target:
        keep_relations.insert(2, "query_to_target_lookback")
        labels.insert(2, "Query-target")
    layers = sorted({int(row["layer"]) for row in rows})
    row_map = {(int(row["layer"]), row["relation"]): float(row["mass_density_lift"]) for row in rows}
    values = np.array([[row_map.get((layer, rel), np.nan) for layer in layers] for rel in keep_relations])
    plt.figure(figsize=(4.8, 4.2))
    ax = plt.gca()
    image = draw_heatmap(
        ax,
        values,
        "Attention concentration over relation density",
        "Layer",
        "Relation",
        [str(layer) for layer in layers],
        labels,
        "rocket_r" if "rocket_r" in plt.colormaps() else "magma",
        0.0,
        max(1.0, float(np.nanmax(values))),
    )
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    savefig(os.path.join(output_dir, "heatmap_relation_lift.png"), dpi)


def main():
    args = parse_args()
    result_dir = os.path.dirname(os.path.abspath(args.result_file))
    analysis_dir = args.analysis_dir or os.path.join(result_dir, "attention_analysis")
    default_leaf = "heatmaps_with_target" if args.include_target else "heatmaps_exog_only"
    output_dir = args.output_dir or os.path.join(analysis_dir, default_leaf)
    os.makedirs(output_dir, exist_ok=True)

    matrices, run_args = collect_attention_matrices(
        args.result_file,
        args.split,
        args.max_batches,
        args.batch_size,
        args.num_workers,
        args.gpu,
        args.no_use_gpu,
    )
    plot_variable_lag(matrices, run_args, output_dir, args.dpi, args.include_target)
    plot_horizon_variable(matrices, run_args, output_dir, args.dpi, args.include_target)
    plot_horizon_lag(matrices, run_args, output_dir, args.dpi, args.include_target)
    plot_history_variable_interaction(matrices, run_args, output_dir, args.dpi, args.include_target)
    plot_relation_lift_heatmap(analysis_dir, output_dir, args.dpi, args.include_target)

    manifest = {
        "result_file": os.path.abspath(args.result_file),
        "analysis_dir": os.path.abspath(analysis_dir),
        "output_dir": os.path.abspath(output_dir),
        "max_batches": args.max_batches,
        "include_target": bool(args.include_target),
        "layers": sorted(int(layer) for layer in matrices),
        "figures": sorted(name for name in os.listdir(output_dir) if name.endswith(".png")),
    }
    with open(os.path.join(output_dir, "heatmap_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved grouped attention heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()
