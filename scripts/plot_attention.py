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


RELATION_LABELS = {
    "history_temporal_same_var": "History temporal",
    "history_exogenous_same_patch": "History exogenous",
    "query_to_target_lookback": "Query-target lookback",
    "query_to_exogenous_lag": "Query-exogenous lag",
    "query_to_query_horizon": "Query-query",
    "query_to_history_all": "Query-history all",
    "other": "Other",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot TELA attention analysis figures.")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to a TELA result JSON.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="",
        help="Directory containing attention CSV/JSON files. Defaults to result_dir/attention_analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for figures. Defaults to analysis_dir/figures.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true", help="Only plot CSV summaries.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    return float(row[key])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def savefig(path, dpi):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_relation_summary(rows, output_dir, dpi):
    selected = [
        "history_temporal_same_var",
        "history_exogenous_same_patch",
        "query_to_target_lookback",
        "query_to_exogenous_lag",
        "query_to_query_horizon",
        "other",
    ]
    layers = sorted({int(row["layer"]) for row in rows})
    row_map = {(int(row["layer"]), row["relation"]): row for row in rows}
    x = np.arange(len(selected))
    width = 0.34 if len(layers) <= 2 else 0.8 / max(len(layers), 1)

    for metric, ylabel, filename, baseline in [
        ("mass_density_lift", "Attention lift over edge density", "relation_lift.png", 1.0),
        ("attention_mass", "Mean attention mass", "relation_mass.png", None),
    ]:
        plt.figure(figsize=(10.5, 4.8))
        for idx, layer in enumerate(layers):
            values = [
                as_float(row_map[(layer, rel)], metric) if (layer, rel) in row_map else 0.0
                for rel in selected
            ]
            offset = (idx - (len(layers) - 1) / 2) * width
            plt.bar(x + offset, values, width=width, label=f"Layer {layer}")
        if baseline is not None:
            plt.axhline(baseline, color="#777777", linewidth=1.0, linestyle="--")
        plt.xticks(x, [RELATION_LABELS[rel] for rel in selected], rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        plt.grid(axis="y", alpha=0.25)
        savefig(os.path.join(output_dir, filename), dpi)


def plot_query_variable(rows, output_dir, dpi):
    layers = sorted({int(row["layer"]) for row in rows})
    for layer in layers:
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        layer_rows = sorted(layer_rows, key=lambda row: as_float(row, "query_history_attention_mass"))
        labels = [row["variable"] for row in layer_rows]
        values = [as_float(row, "query_history_attention_mass") for row in layer_rows]
        colors = ["#d95f02" if int(row["is_target"]) else "#1b9e77" for row in layer_rows]
        plt.figure(figsize=(8.6, 6.2))
        plt.barh(np.arange(len(labels)), values, color=colors)
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel("Query-to-history attention mass")
        plt.title(f"Layer {layer}: variable attention from future queries")
        plt.grid(axis="x", alpha=0.25)
        savefig(os.path.join(output_dir, f"query_variable_layer{layer}.png"), dpi)


def plot_query_patch(rows, output_dir, dpi):
    layers = sorted({int(row["layer"]) for row in rows})
    plt.figure(figsize=(7.2, 4.4))
    for layer in layers:
        layer_rows = sorted(
            [row for row in rows if int(row["layer"]) == layer],
            key=lambda row: int(row["patch_index"]),
        )
        patches = [int(row["patch_index"]) for row in layer_rows]
        values = [as_float(row, "query_history_attention_mass") for row in layer_rows]
        plt.plot(patches, values, marker="o", linewidth=2.0, label=f"Layer {layer}")
    plt.xlabel("Historical patch index")
    plt.ylabel("Query-to-history attention mass")
    plt.xticks(sorted({int(row["patch_index"]) for row in rows}))
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    savefig(os.path.join(output_dir, "query_patch_attention.png"), dpi)


class MatrixCollector:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.sums = {}
        self.counts = {}

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
            attn = output[1].detach().float()
            batch_size = attn.size(0)
            matrix = attn.mean(dim=1).sum(dim=0).cpu().numpy()
            if layer_idx not in self.sums:
                self.sums[layer_idx] = matrix
                self.counts[layer_idx] = batch_size
            else:
                self.sums[layer_idx] += matrix
                self.counts[layer_idx] += batch_size

        return hook

    def averages(self):
        return {layer: self.sums[layer] / max(self.counts[layer], 1) for layer in self.sums}


def collect_attention_matrices(result_file, split, max_batches, batch_size, num_workers, gpu, no_use_gpu):
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
    _, loader = data_provider(run_args, split)
    with MatrixCollector(model) as collector:
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    return collector.averages(), run_args


def add_boundaries(ax, n_vars, patch_num, hist_n):
    ax.axhline(hist_n - 0.5, color="white", linewidth=1.1)
    ax.axvline(hist_n - 0.5, color="white", linewidth=1.1)
    for var_idx in range(1, n_vars):
        pos = var_idx * patch_num - 0.5
        ax.axhline(pos, color="white", linewidth=0.25, alpha=0.45)
        ax.axvline(pos, color="white", linewidth=0.25, alpha=0.45)


def plot_attention_matrices(matrices, run_args, output_dir, dpi):
    n_vars = int(run_args.enc_in)
    patch_num = int((run_args.seq_len - run_args.patch_len) // run_args.patch_stride + 1)
    hist_n = n_vars * patch_num
    feature_names = list(run_args.feature_names)

    for layer, matrix in sorted(matrices.items()):
        vmax = float(np.quantile(matrix, 0.995))
        plt.figure(figsize=(7.2, 6.4))
        ax = plt.gca()
        image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(vmax, 1e-8), aspect="auto")
        add_boundaries(ax, n_vars, patch_num, hist_n)
        ax.set_title(f"Layer {layer}: average attention matrix")
        ax.set_xlabel("Key node")
        ax.set_ylabel("Query node")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        savefig(os.path.join(output_dir, f"attention_heatmap_layer{layer}.png"), dpi)

        qh = matrix[hist_n:, :hist_n]
        vmax_qh = float(np.quantile(qh, 0.995))
        plt.figure(figsize=(8.4, 4.0))
        ax = plt.gca()
        image = ax.imshow(qh, cmap="viridis", vmin=0.0, vmax=max(vmax_qh, 1e-8), aspect="auto")
        for var_idx in range(1, n_vars):
            ax.axvline(var_idx * patch_num - 0.5, color="white", linewidth=0.5, alpha=0.65)
        ax.set_title(f"Layer {layer}: future query to historical nodes")
        ax.set_xlabel("Historical variable-patch node")
        ax.set_ylabel("Forecast horizon")
        ax.set_yticks([0, max(run_args.pred_len // 2 - 1, 0), run_args.pred_len - 1])
        ax.set_yticklabels(["h1", f"h{max(run_args.pred_len // 2, 1)}", f"h{run_args.pred_len}"])
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        savefig(os.path.join(output_dir, f"query_history_heatmap_layer{layer}.png"), dpi)

        var_patch = qh.mean(axis=0).reshape(n_vars, patch_num)
        vmax_vp = float(np.quantile(var_patch, 0.98))
        plt.figure(figsize=(6.8, max(4.8, n_vars * 0.28)))
        ax = plt.gca()
        image = ax.imshow(var_patch, cmap="YlOrRd", vmin=0.0, vmax=max(vmax_vp, 1e-8), aspect="auto")
        ax.set_title(f"Layer {layer}: query attention by variable and patch")
        ax.set_xlabel("Historical patch")
        ax.set_ylabel("Variable")
        ax.set_xticks(np.arange(patch_num))
        ax.set_yticks(np.arange(n_vars))
        ax.set_yticklabels(feature_names, fontsize=8)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        savefig(os.path.join(output_dir, f"query_variable_patch_layer{layer}.png"), dpi)


def main():
    args = parse_args()
    result_dir = os.path.dirname(os.path.abspath(args.result_file))
    analysis_dir = args.analysis_dir or os.path.join(result_dir, "attention_analysis")
    output_dir = args.output_dir or os.path.join(analysis_dir, "figures")
    ensure_dir(output_dir)

    relation_path = os.path.join(analysis_dir, "relation_attention.csv")
    variable_path = os.path.join(analysis_dir, "query_variable_attention.csv")
    patch_path = os.path.join(analysis_dir, "query_patch_attention.csv")
    if not os.path.exists(relation_path):
        raise FileNotFoundError(f"missing relation csv: {relation_path}")

    plot_relation_summary(read_csv_rows(relation_path), output_dir, args.dpi)
    if os.path.exists(variable_path):
        plot_query_variable(read_csv_rows(variable_path), output_dir, args.dpi)
    if os.path.exists(patch_path):
        plot_query_patch(read_csv_rows(patch_path), output_dir, args.dpi)

    matrix_layers = []
    if not args.skip_matrix:
        matrices, run_args = collect_attention_matrices(
            args.result_file,
            args.split,
            args.max_batches,
            args.batch_size,
            args.num_workers,
            args.gpu,
            args.no_use_gpu,
        )
        plot_attention_matrices(matrices, run_args, output_dir, args.dpi)
        matrix_layers = sorted(int(layer) for layer in matrices)

    manifest = {
        "result_file": os.path.abspath(args.result_file),
        "analysis_dir": os.path.abspath(analysis_dir),
        "output_dir": os.path.abspath(output_dir),
        "matrix_layers": matrix_layers,
        "max_batches": args.max_batches,
        "figures": sorted(
            name for name in os.listdir(output_dir) if name.lower().endswith((".png", ".pdf"))
        ),
    }
    with open(os.path.join(output_dir, "figure_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved attention figures to: {output_dir}")


if __name__ == "__main__":
    main()
