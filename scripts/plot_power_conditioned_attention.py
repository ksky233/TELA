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
from data_provider import data_provider  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Plot power-conditioned exogenous attention heatmaps.")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--title-size", type=float, default=14.0)
    parser.add_argument("--label-size", type=float, default=13.0)
    parser.add_argument("--tick-size", type=float, default=11.0)
    parser.add_argument("--colorbar-tick-size", type=float, default=10.5)
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


def row_normalize(values):
    denom = values.sum(axis=1, keepdims=True)
    return np.divide(values, np.maximum(denom, 1e-12))


def physical_variable_name(name):
    for suffix in ("_sin", "_cos"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def aggregate_physical_variables(values, names):
    groups = []
    group_index = {}
    for idx, name in enumerate(names):
        physical_name = physical_variable_name(name)
        if physical_name not in group_index:
            group_index[physical_name] = len(groups)
            groups.append(physical_name)

    grouped = np.zeros((values.shape[0], len(groups)), dtype=np.float64)
    for idx, name in enumerate(names):
        grouped[:, group_index[physical_variable_name(name)]] += values[:, idx]
    return grouped, groups


def display_physical_variable_name(name):
    mapping = {
        "wind_speed_10m": "WS10",
        "wind_dir_10m": "WD10",
        "wind_speed_30m": "WS30",
        "wind_dir_30m": "WD30",
        "wind_speed_50m": "WS50",
        "wind_dir_50m": "WD50",
        "wind_speed_hub": "WS hub",
        "wind_dir_hub": "WD hub",
        "temperature": "Temp.",
        "pressure": "Pres.",
        "humidity": "Hum.",
    }
    return mapping.get(name, name.replace("_", " "))


def display_physical_variable_names(names):
    return [display_physical_variable_name(name) for name in names]


class PowerAttentionCollector:
    def __init__(self, model, run_args):
        self.model = model
        self.run_args = run_args
        self.handles = []
        self.batch_layers = []
        self.samples = {}
        self.power_values = []
        self.n_vars = int(run_args.enc_in)
        self.patch_num = calc_patch_num(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
        self.hist_n = self.n_vars * self.patch_num
        self.var_ids = np.repeat(np.arange(self.n_vars), self.patch_num)
        self.patch_ids = np.tile(np.arange(self.patch_num), self.n_vars)
        self.target_index = int(run_args.target_index)
        self.exog_indices = [idx for idx in range(self.n_vars) if idx != self.target_index]

    def __enter__(self):
        for layer_idx, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer_idx):
        def hook(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) > 1):
                return
            attn = output[1].detach().float().cpu()
            query_to_hist = attn[:, :, self.hist_n :, : self.hist_n].mean(dim=(1, 2)).numpy()
            var_mass = []
            lag_mass = []
            exog_cols = self.var_ids != self.target_index
            for sample_attn in query_to_hist:
                per_var = [
                    float(sample_attn[self.var_ids == var_idx].sum())
                    for var_idx in self.exog_indices
                ]
                per_lag = [
                    float(sample_attn[(self.patch_ids == patch_idx) & exog_cols].sum())
                    for patch_idx in range(self.patch_num)
                ]
                var_mass.append(per_var)
                lag_mass.append(per_lag)
            self.batch_layers.append(
                {
                    "layer": layer_idx,
                    "variable_mass": np.asarray(var_mass, dtype=np.float64),
                    "lag_mass": np.asarray(lag_mass, dtype=np.float64),
                    "exog_total_mass": np.asarray(var_mass, dtype=np.float64).sum(axis=1),
                }
            )

        return hook

    def start_batch(self):
        self.batch_layers = []

    def finish_batch(self, batch_power):
        self.power_values.append(batch_power)
        for item in self.batch_layers:
            layer = item["layer"]
            if layer not in self.samples:
                self.samples[layer] = {
                    "variable_mass": [],
                    "lag_mass": [],
                    "exog_total_mass": [],
                }
            self.samples[layer]["variable_mass"].append(item["variable_mass"])
            self.samples[layer]["lag_mass"].append(item["lag_mass"])
            self.samples[layer]["exog_total_mass"].append(item["exog_total_mass"])
        self.batch_layers = []

    def results(self):
        power = np.concatenate(self.power_values, axis=0)
        out = {}
        for layer, chunks in self.samples.items():
            out[layer] = {
                "variable_mass": np.concatenate(chunks["variable_mass"], axis=0),
                "lag_mass": np.concatenate(chunks["lag_mass"], axis=0),
                "exog_total_mass": np.concatenate(chunks["exog_total_mass"], axis=0),
            }
        return power, out


def inverse_last_power(dataset, batch_x, target_index):
    last_norm = batch_x[:, -1, target_index].detach().cpu().numpy().reshape(-1, 1)
    if hasattr(dataset, "inverse_transform"):
        return dataset.inverse_transform(last_norm).reshape(-1)
    return last_norm.reshape(-1)


def collect(result_file, split, max_batches, batch_size, num_workers, gpu, no_use_gpu):
    payload, run_args = load_run(result_file)
    checkpoint = payload.get("train_summary", {}).get("checkpoint", "")
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    run_args.batch_size = batch_size
    run_args.num_workers = num_workers
    run_args.use_gpu = bool((not no_use_gpu) and torch.cuda.is_available())
    run_args.gpu = gpu
    device = torch.device(f"cuda:{gpu}" if run_args.use_gpu else "cpu")
    model = build_model(run_args, checkpoint, device)
    dataset, loader = data_provider(run_args, split)

    with PowerAttentionCollector(model, run_args) as collector:
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                collector.start_batch()
                power = inverse_last_power(dataset, batch_x, run_args.target_index)
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                collector.finish_batch(power)
    return collector.results(), run_args


def make_power_bins(power, n_bins):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(power, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(power.min()), float(power.max()), n_bins + 1)
    bin_ids = np.digitize(power, edges[1:-1], right=True)
    labels = []
    for idx in range(len(edges) - 1):
        labels.append(f"{edges[idx]:.1f}-{edges[idx + 1]:.1f}")
    return bin_ids, labels, edges


def aggregate_by_power_bin(power, layer_payload, n_bins):
    bin_ids, labels, edges = make_power_bins(power, n_bins)
    actual_bins = len(labels)
    variable = np.zeros((actual_bins, layer_payload["variable_mass"].shape[1]), dtype=np.float64)
    lag = np.zeros((actual_bins, layer_payload["lag_mass"].shape[1]), dtype=np.float64)
    total = np.zeros(actual_bins, dtype=np.float64)
    counts = np.zeros(actual_bins, dtype=np.int64)
    for bin_idx in range(actual_bins):
        mask = bin_ids == bin_idx
        counts[bin_idx] = int(mask.sum())
        if not mask.any():
            continue
        variable[bin_idx] = layer_payload["variable_mass"][mask].mean(axis=0)
        lag[bin_idx] = layer_payload["lag_mass"][mask].mean(axis=0)
        total[bin_idx] = layer_payload["exog_total_mass"][mask].mean()
    return {
        "variable": variable,
        "lag": lag,
        "total": total,
        "counts": counts,
        "labels": labels,
        "edges": edges,
    }


def draw_heatmap(
    values,
    xlabels,
    ylabels,
    title,
    xlabel,
    ylabel,
    output_path,
    dpi,
    title_size=14.0,
    label_size=13.0,
    tick_size=11.0,
    colorbar_tick_size=10.5,
):
    plt.figure(figsize=(max(8.2, len(xlabels) * 0.52), max(4.2, len(ylabels) * 0.62)))
    ax = plt.gca()
    image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(float(values.max()), 1e-12))
    if title:
        ax.set_title(title, fontsize=title_size, pad=8)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=tick_size)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=tick_size)
    cbar = plt.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)
    savefig(output_path, dpi)


def draw_combined_heatmaps(variable_values, lag_values, exog_names, lag_labels, power_labels, output_dir, dpi, args):
    variable_path = os.path.join(output_dir, "heatmap_power_bin_exog_variable_merged.png")
    physical_variable_path = os.path.join(
        output_dir,
        "heatmap_power_bin_exog_physical_variable_merged.png",
    )
    lag_path = os.path.join(output_dir, "heatmap_power_bin_exog_lag_merged.png")
    physical_values, physical_names = aggregate_physical_variables(variable_values, exog_names)
    physical_values = row_normalize(physical_values)
    draw_heatmap(
        variable_values,
        exog_names,
        power_labels,
        "Merged layers: exogenous variable weights across power levels",
        "Historical exogenous variable",
        "Observed power bin",
        variable_path,
        dpi,
        args.title_size,
        args.label_size,
        args.tick_size,
        args.colorbar_tick_size,
    )
    draw_heatmap(
        physical_values,
        display_physical_variable_names(physical_names),
        power_labels,
        "Physical exogenous variables",
        "Historical exogenous variable",
        "Observed power bin",
        physical_variable_path,
        dpi,
        args.title_size,
        args.label_size,
        args.tick_size,
        args.colorbar_tick_size,
    )
    draw_heatmap(
        lag_values,
        lag_labels,
        power_labels,
        "Merged layers: exogenous lag weights across power levels",
        "Historical lag patch",
        "Observed power bin",
        lag_path,
        dpi,
        args.title_size,
        args.label_size,
        args.tick_size,
        args.colorbar_tick_size,
    )
    return [
        os.path.basename(variable_path),
        os.path.basename(physical_variable_path),
        os.path.basename(lag_path),
    ]


def write_summary_csv(path, layer, labels, counts, total):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "power_bin", "sample_count", "mean_exog_attention_mass"],
        )
        if f.tell() == 0:
            writer.writeheader()
        for label, count, value in zip(labels, counts, total):
            writer.writerow(
                {
                    "layer": layer,
                    "power_bin": label,
                    "sample_count": int(count),
                    "mean_exog_attention_mass": float(value),
                }
            )


def main():
    args = parse_args()
    result_dir = os.path.dirname(os.path.abspath(args.result_file))
    output_dir = args.output_dir or os.path.join(
        result_dir,
        "attention_analysis",
        "power_conditioned_exog",
    )
    os.makedirs(output_dir, exist_ok=True)

    (power, layer_results), run_args = collect(
        args.result_file,
        args.split,
        args.max_batches,
        args.batch_size,
        args.num_workers,
        args.gpu,
        args.no_use_gpu,
    )
    exog_names = [
        name
        for idx, name in enumerate(run_args.feature_names)
        if idx != int(run_args.target_index)
    ]
    lag_labels = patch_labels(run_args.seq_len, run_args.patch_len, run_args.patch_stride)
    summary_csv = os.path.join(output_dir, "power_bin_attention_summary.csv")
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    manifest = {
        "result_file": os.path.abspath(args.result_file),
        "output_dir": os.path.abspath(output_dir),
        "bins": args.bins,
        "max_batches": args.max_batches,
        "figures": [],
    }
    merged_variables = []
    merged_lags = []
    merged_labels = None
    for layer, payload in sorted(layer_results.items()):
        agg = aggregate_by_power_bin(power, payload, args.bins)
        variable_norm = row_normalize(agg["variable"])
        lag_norm = row_normalize(agg["lag"])
        ylabels = [f"{label} MW" for label in agg["labels"]]
        merged_variables.append(variable_norm)
        merged_lags.append(lag_norm)
        merged_labels = ylabels

        variable_path = os.path.join(output_dir, f"heatmap_power_bin_exog_variable_layer{layer}.png")
        physical_variable_path = os.path.join(
            output_dir,
            f"heatmap_power_bin_exog_physical_variable_layer{layer}.png",
        )
        lag_path = os.path.join(output_dir, f"heatmap_power_bin_exog_lag_layer{layer}.png")
        physical_variable_norm, physical_names = aggregate_physical_variables(
            variable_norm,
            exog_names,
        )
        physical_variable_norm = row_normalize(physical_variable_norm)
        draw_heatmap(
            variable_norm,
            exog_names,
            ylabels,
            f"Layer {layer}: exogenous variable weights across power levels",
            "Historical exogenous variable",
            "Observed power bin",
            variable_path,
            args.dpi,
            args.title_size,
            args.label_size,
            args.tick_size,
            args.colorbar_tick_size,
        )
        draw_heatmap(
            physical_variable_norm,
            display_physical_variable_names(physical_names),
            ylabels,
            f"Layer {layer}: physical exogenous variables",
            "Historical exogenous variable",
            "Observed power bin",
            physical_variable_path,
            args.dpi,
            args.title_size,
            args.label_size,
            args.tick_size,
            args.colorbar_tick_size,
        )
        draw_heatmap(
            lag_norm,
            lag_labels,
            ylabels,
            f"Layer {layer}: exogenous lag weights across power levels",
            "Historical lag patch",
            "Observed power bin",
            lag_path,
            args.dpi,
            args.title_size,
            args.label_size,
            args.tick_size,
            args.colorbar_tick_size,
        )
        write_summary_csv(summary_csv, layer, agg["labels"], agg["counts"], agg["total"])
        manifest["figures"].extend(
            [
                os.path.basename(variable_path),
                os.path.basename(physical_variable_path),
                os.path.basename(lag_path),
            ]
        )

    if merged_variables and merged_lags and merged_labels is not None:
        merged_variable = np.mean(np.stack(merged_variables, axis=0), axis=0)
        merged_lag = np.mean(np.stack(merged_lags, axis=0), axis=0)
        manifest["figures"].extend(
            draw_combined_heatmaps(
                merged_variable,
                merged_lag,
                exog_names,
                lag_labels,
                merged_labels,
                output_dir,
                args.dpi,
                args,
            )
        )

    with open(os.path.join(output_dir, "power_conditioned_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved power-conditioned attention heatmaps to: {output_dir}")


if __name__ == "__main__":
    main()
