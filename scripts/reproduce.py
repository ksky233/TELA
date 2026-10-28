"""Build and run the conference protocol without machine-specific paths."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITES = {
    "site1": "grid_wind_site1_99MW.csv",
    "site2": "grid_wind_site2_200MW.csv",
    "site4": "grid_wind_site4_66MW.csv",
}
VARIANTS = {
    "full": {},
    "no_temporal": {"use_temporal_bias": False},
    "no_exogenous": {"use_exog_bias": False},
    "no_lag": {"use_wind_lag_bias": False},
    "no_relation_bias": {
        "use_temporal_bias": False,
        "use_exog_bias": False,
        "use_wind_lag_bias": False,
    },
    "dense": {"top_k": 0},
    "raw_direction": {"wind_dir_encoding": "none"},
    "topk16": {"top_k": 16},
    "topk64": {"top_k": 64},
    "patch8": {"patch_len": 8, "patch_stride": 8},
    "patch32": {"patch_len": 32, "patch_stride": 32},
    "blocks1": {"e_layers": 1},
    "blocks3": {"e_layers": 3},
    "heads2": {"n_heads": 2},
    "heads8": {"n_heads": 8},
}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def format_command(command):
    if sys.platform == "win32":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def build_command(args, site, seed, variant):
    config = json.loads((ROOT / "configs" / "conference.json").read_text(encoding="utf-8"))
    config.update(VARIANTS[variant])
    config.update({
        "exp_name": f"tela_{site}_{variant}",
        "turned": f"seed{seed}",
        "random_seed": seed,
        "root_path": str(resolve_path(args.data_root)),
        "data_path": SITES[site],
        "checkpoints": str(resolve_path(args.checkpoints)),
        "result_path": str(resolve_path(args.results) / "evaluation") if args.evaluate
        else str(resolve_path(args.results)),
        "is_training": 0 if args.evaluate else 1,
        "use_gpu": not args.cpu,
        "gpu": args.gpu,
    })
    command = [sys.executable, str(ROOT / "run.py")]
    for key, value in config.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            command.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            command.extend([f"--{flag}", str(value)])
    checkpoint = resolve_path(args.checkpoints) / f"{config['exp_name']}_turnd_seed{seed}" / "checkpoint.pth"
    result = Path(config["result_path"]) / config["exp_name"] / f"turnd-seed{seed}.json"
    return command, checkpoint, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", choices=tuple(SITES), default=["site1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["full"])
    parser.add_argument("--data-root", default="datasets/wind")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--results", default="result")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing matching outputs")
    args = parser.parse_args()

    jobs = [build_command(args, site, seed, variant)
            for site in dict.fromkeys(args.sites)
            for seed in dict.fromkeys(args.seeds)
            for variant in dict.fromkeys(args.variants)]
    if not args.dry_run:
        for site in args.sites:
            data = resolve_path(args.data_root) / SITES[site]
            if not data.is_file():
                parser.error(f"Data file not found: {data}; see datasets/README.md")
        for _, checkpoint, result in jobs:
            if args.evaluate and not checkpoint.is_file():
                parser.error(f"Checkpoint not found: {checkpoint}")
            existing = result.exists() or (not args.evaluate and checkpoint.exists())
            if existing and not args.overwrite:
                parser.error(f"Existing experiment outputs: {result.parent}; use --overwrite to replace")

    for command, _, _ in jobs:
        print(format_command(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
