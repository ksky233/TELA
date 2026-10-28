"""Validate the prepared wind CSV schema and report reference fingerprints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.audit import sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets" / "wind")
    parser.add_argument("--sites", nargs="+", choices=["site1", "site2", "site4"], default=["site1", "site2", "site4"])
    parser.add_argument("--strict-hash", action="store_true", help="also require identical CSV bytes")
    cli = parser.parse_args()
    manifest = json.loads((ROOT / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    for name in cli.sites:
        site = manifest["sites"][name]
        path = cli.data_root / site["filename"]
        if not path.is_file():
            parser.error(f"Missing CSV: {path}")
        frame = pd.read_csv(path)
        expected = manifest["common_columns"] + site["trailing_columns"]
        if frame.columns.tolist() != expected:
            parser.error(f"Column order mismatch: {path.name}; expected {expected}")
        dates = pd.to_datetime(frame[manifest["time_column"]], errors="raise")
        if len(frame) != manifest["rows"] or dates.isna().any():
            parser.error(f"Row count or timestamp mismatch: {path.name}")
        if str(dates.iloc[0]) != manifest["start"] or str(dates.iloc[-1]) != manifest["end"]:
            parser.error(f"Time range mismatch: {path.name}")
        if not dates.diff().iloc[1:].eq(pd.Timedelta(manifest["frequency"])).all():
            parser.error(f"Nonuniform or unsorted timestamps: {path.name}")
        if not np.isfinite(frame[expected[1:]].to_numpy(dtype=np.float64)).all():
            parser.error(f"Non-finite numeric values: {path.name}")
        digest = sha256_file(path)
        match = digest == site["sha256"]
        if cli.strict_hash and not match:
            parser.error(f"Fingerprint mismatch: {path.name}")
        print(json.dumps({"site": name, "rows": len(frame), "schema_valid": True,
                          "reference_bytes_match": match, "sha256": digest}))


if __name__ == "__main__":
    main()
