import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def _time_features(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates)
    minute_of_day = (dt.dt.hour * 60 + dt.dt.minute).astype(np.float32)
    day_of_week = dt.dt.dayofweek.astype(np.float32)
    day_of_year = dt.dt.dayofyear.astype(np.float32)
    month = dt.dt.month.astype(np.float32)

    return np.stack(
        [
            minute_of_day / 1440.0 - 0.5,
            day_of_week / 6.0 - 0.5,
            day_of_year / 366.0 - 0.5,
            month / 12.0 - 0.5,
        ],
        axis=1,
    ).astype(np.float32)


def encode_wind_direction(df: pd.DataFrame, mode: str, time_col: str, target: str) -> pd.DataFrame:
    if mode == "none":
        return df
    if mode != "sincos":
        raise ValueError("wind_dir_encoding must be one of: none, sincos")

    df = df.copy()
    direction_cols = [
        col
        for col in df.columns
        if col not in {time_col, target}
        and pd.api.types.is_numeric_dtype(df[col])
        and ("wind_dir" in col.lower() or "wind_direction" in col.lower())
    ]
    for col in direction_cols:
        radians = np.deg2rad(df[col].astype(np.float32) % 360.0)
        insert_at = df.columns.get_loc(col)
        df.insert(insert_at, f"{col}_sin", np.sin(radians).astype(np.float32))
        df.insert(insert_at + 1, f"{col}_cos", np.cos(radians).astype(np.float32))
        df.drop(columns=[col], inplace=True)
    return df


class DatasetWideForecast(Dataset):
    """Wide-table forecasting dataset.

    The input CSV is expected to contain one timestamp column plus numeric feature
    columns. For `features=MS`, all numeric columns are used as encoder inputs
    and the target column is predicted.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",
        size: Tuple[int, int, int] = (96, 48, 16),
        features: str = "MS",
        target: str = "power",
        time_col: str = "date",
        freq: str = "15min",
        wind_dir_encoding: str = "none",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        data_limit: int = 0,
    ):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")

        self.seq_len, self.label_len, self.pred_len = size
        self.features = features
        self.target = target
        self.time_col = time_col
        self.freq = freq
        self.wind_dir_encoding = wind_dir_encoding
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.data_limit = data_limit

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        file_path = self.data_path
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        df_raw = pd.read_csv(file_path)
        if self.time_col not in df_raw.columns:
            raise ValueError(f"time column '{self.time_col}' is not in {file_path}")
        if self.target not in df_raw.columns:
            raise ValueError(f"target column '{self.target}' is not in {file_path}")

        df_raw = encode_wind_direction(df_raw, self.wind_dir_encoding, self.time_col, self.target)

        if self.data_limit and self.data_limit > 0:
            df_raw = df_raw.iloc[: self.data_limit].copy()

        numeric_cols = [
            col
            for col in df_raw.columns
            if col != self.time_col and pd.api.types.is_numeric_dtype(df_raw[col])
        ]
        if self.target not in numeric_cols:
            raise ValueError(f"target column '{self.target}' must be numeric")

        feature_cols = [col for col in numeric_cols if col != self.target] + [self.target]
        if self.features == "S":
            cols_data = [self.target]
        elif self.features in {"M", "MS"}:
            cols_data = feature_cols
        else:
            raise ValueError("features must be one of: S, M, MS")

        n = len(df_raw)
        if n < self.seq_len + self.pred_len + 1:
            raise ValueError(
                f"Dataset is too short: {n}, needs at least {self.seq_len + self.pred_len + 1}"
            )

        num_train = int(n * self.train_ratio)
        num_val = int(n * self.val_ratio)
        num_test = n - num_train - num_val
        if min(num_train, num_val, num_test) <= 0:
            raise ValueError("Invalid split ratios; train/val/test must all be non-empty")

        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]
        border1 = max(border1s[self.set_type], 0)
        border2 = border2s[self.set_type]

        data = df_raw[cols_data].astype(np.float32).values
        self.scaler = StandardScaler()
        self.scaler.fit(data[border1s[0] : border2s[0]])
        data = self.scaler.transform(data).astype(np.float32)

        df_stamp = df_raw[[self.time_col]].iloc[border1:border2].copy()
        data_stamp = _time_features(df_stamp[self.time_col])

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.cols_data = cols_data
        self.target_index = cols_data.index(self.target)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if data.shape[-1] == len(self.cols_data):
            return self.scaler.inverse_transform(data)

        target_mean = self.scaler.mean[..., self.target_index]
        target_std = self.scaler.std[..., self.target_index]
        return data * target_std + target_mean
