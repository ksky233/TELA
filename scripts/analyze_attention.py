import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TELA_DIR = THIS_DIR.parent
if str(TELA_DIR) not in sys.path:
    sys.path.insert(0, str(TELA_DIR))

from data_provider import data_provider  # noqa: E402
from utils.checkpoints import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze TELA attention relation patterns.")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="Path to a TELA result JSON containing args and checkpoint metadata.",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Override checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-use-gpu", action="store_true")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for attention analysis files.")
    return parser.parse_args()


def load_run(result_file):
    with open(result_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    run_args = dict(payload["args"])
    run_args.setdefault("turned", "")
    run_args.setdefault("use_amp", False)
    run_args.setdefault("inverse", True)
    run_args.setdefault("use_revin", False)
    run_args.setdefault("use_temporal_bias", True)
    run_args.setdefault("use_exog_bias", True)
    run_args.setdefault("use_wind_lag_bias", True)
    run_args.setdefault("use_future_exog", False)
    run_args.setdefault("output_mode", "point")
    run_args.setdefault("quantiles", [])
    return payload, SimpleNamespace(**run_args)


def build_model(args, checkpoint, device):
    module = importlib.import_module(f"models.{args.model}")
    model = module.Model(args).float().to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    return model


def build_relation_masks(args):
    n_vars = args.enc_in
    patch_num = (args.seq_len + ((args.patch_stride - (args.seq_len - args.patch_len) % args.patch_stride) % args.patch_stride) - args.patch_len) // args.patch_stride + 1
    pred_len = args.pred_len
    target_index = args.target_index
    hist_n = n_vars * patch_num
    total_n = hist_n + pred_len

    var_ids = torch.arange(n_vars).repeat_interleave(patch_num)
    patch_ids = torch.arange(patch_num).repeat(n_vars)
    target_mask = var_ids == target_index
    exog_mask = var_ids != target_index

    masks = {}
    masks["history_temporal_same_var"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    masks["history_temporal_same_var"][:hist_n, :hist_n] = (
        (var_ids[:, None] == var_ids[None, :]) & (patch_ids[:, None] != patch_ids[None, :])
    )

    masks["history_exogenous_same_patch"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    masks["history_exogenous_same_patch"][:hist_n, :hist_n] = (
        (patch_ids[:, None] == patch_ids[None, :]) & (var_ids[:, None] != var_ids[None, :])
    )

    masks["query_to_target_lookback"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    masks["query_to_target_lookback"][hist_n:total_n, :hist_n] = target_mask.view(1, hist_n)

    masks["query_to_exogenous_lag"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    masks["query_to_exogenous_lag"][hist_n:total_n, :hist_n] = exog_mask.view(1, hist_n)

    masks["query_to_history_all"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    masks["query_to_history_all"][hist_n:total_n, :hist_n] = True

    masks["query_to_query_horizon"] = torch.zeros(total_n, total_n, dtype=torch.bool)
    query_query = torch.ones(pred_len, pred_len, dtype=torch.bool)
    query_query.fill_diagonal_(False)
    masks["query_to_query_horizon"][hist_n:total_n, hist_n:total_n] = query_query

    masks["other"] = torch.ones(total_n, total_n, dtype=torch.bool)
    union = torch.zeros(total_n, total_n, dtype=torch.bool)
    for name, mask in masks.items():
        if name != "other":
            union |= mask
    masks["other"] = masks["other"] & ~union
    return masks, var_ids, patch_ids


class AttentionCollector:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.current = []
        self.records = []

    def __enter__(self):
        for layer_idx, block in enumerate(self.model.blocks):
            handle = block.register_forward_hook(self._hook(layer_idx))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer_idx):
        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) > 1:
                self.current.append((layer_idx, output[1].detach().cpu()))

        return hook

    def start_batch(self):
        self.current = []

    def finish_batch(self):
        self.records.extend(self.current)
        self.current = []


def masked_mass(attn, mask):
    mask = mask.to(attn.device).view(1, 1, mask.size(0), mask.size(1))
    return (attn * mask).sum(dim=-1).mean().item()


def masked_density(mask):
    return float(mask.float().mean().item())


def analyze_records(records, masks, feature_names, target_index, var_ids, patch_ids):
    grouped = {}
    for layer_idx, attn in records:
        grouped.setdefault(layer_idx, []).append(attn)

    relation_rows = []
    variable_rows = []
    patch_rows = []
    for layer_idx, chunks in grouped.items():
        attn = torch.cat(chunks, dim=0)
        layer_summary = {"layer": layer_idx, "samples": int(attn.size(0))}
        for name, mask in masks.items():
            mass = masked_mass(attn, mask)
            density = masked_density(mask)
            lift = mass / density if density > 0 else 0.0
            row = {
                "layer": layer_idx,
                "relation": name,
                "attention_mass": mass,
                "edge_density": density,
                "mass_density_lift": lift,
            }
            relation_rows.append(row)
            layer_summary[name] = row

        hist_n = int(var_ids.numel())
        query_rows = slice(hist_n, hist_n + attn.size(-1) - hist_n)
        query_to_hist = attn[:, :, query_rows, :hist_n]
        query_to_hist_mean = query_to_hist.mean(dim=(0, 1, 2))
        for var_idx, name in enumerate(feature_names):
            cols = torch.where(var_ids == var_idx)[0]
            mass = query_to_hist_mean[cols].sum().item()
            variable_rows.append(
                {
                    "layer": layer_idx,
                    "variable_index": var_idx,
                    "variable": name,
                    "is_target": int(var_idx == target_index),
                    "query_history_attention_mass": mass,
                }
            )

        for patch_idx in range(int(patch_ids.max().item()) + 1):
            cols = torch.where(patch_ids == patch_idx)[0]
            mass = query_to_hist_mean[cols].sum().item()
            patch_rows.append(
                {
                    "layer": layer_idx,
                    "patch_index": patch_idx,
                    "query_history_attention_mass": mass,
                }
            )

    return relation_rows, variable_rows, patch_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cli = parse_args()
    payload, run_args = load_run(cli.result_file)
    run_args.batch_size = cli.batch_size
    run_args.num_workers = cli.num_workers
    run_args.use_gpu = bool((not cli.no_use_gpu) and torch.cuda.is_available())
    run_args.gpu = cli.gpu

    checkpoint = cli.checkpoint or payload.get("train_summary", {}).get("checkpoint", "")
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_dir = cli.output_dir
    if not output_dir:
        result_dir = os.path.dirname(os.path.abspath(cli.result_file))
        output_dir = os.path.join(result_dir, "attention_analysis")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{cli.gpu}" if run_args.use_gpu else "cpu")
    model = build_model(run_args, checkpoint, device)
    _, loader = data_provider(run_args, cli.split)
    masks, var_ids, patch_ids = build_relation_masks(run_args)

    with AttentionCollector(model) as collector:
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if cli.max_batches > 0 and batch_idx >= cli.max_batches:
                    break
                collector.start_batch()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                _ = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                collector.finish_batch()

    relation_rows, variable_rows, patch_rows = analyze_records(
        collector.records,
        masks,
        run_args.feature_names,
        run_args.target_index,
        var_ids,
        patch_ids,
    )

    write_csv(os.path.join(output_dir, "relation_attention.csv"), relation_rows)
    write_csv(os.path.join(output_dir, "query_variable_attention.csv"), variable_rows)
    write_csv(os.path.join(output_dir, "query_patch_attention.csv"), patch_rows)

    summary = {
        "source_result_file": os.path.abspath(cli.result_file),
        "checkpoint": checkpoint,
        "split": cli.split,
        "max_batches": cli.max_batches,
        "feature_names": run_args.feature_names,
        "target_index": run_args.target_index,
        "relation_attention": relation_rows,
        "query_variable_attention": variable_rows,
        "query_patch_attention": patch_rows,
    }
    with open(os.path.join(output_dir, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved attention analysis to: {output_dir}")


if __name__ == "__main__":
    main()
