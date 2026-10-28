"""Deterministic synthetic inputs for software checks, not research results."""

import numpy as np
import pandas as pd


def write_synthetic_csv(path, rows=320):
    rng = np.random.default_rng(2026)
    t = np.arange(rows, dtype=np.float32)
    wind = 7.0 + 2.0 * np.sin(t / 13.0)
    frame = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="15min"),
        "wind_speed_10m": wind,
        "wind_dir_10m": (t * 3.0) % 360.0,
        "temperature": 15.0 + np.cos(t / 30.0),
        "power": np.clip(0.08 * wind**3 + rng.normal(0.0, 0.5, rows), 0.0, 99.0),
    })
    frame.to_csv(path, index=False)
    return frame
