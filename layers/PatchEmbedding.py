import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(-2)]


class PatchEmbedding(nn.Module):
    """Patch embedding for wide-table time series.

    Input:  [B, L, C]
    Output: [B, C, P, D]
    """

    def __init__(self, d_model: int, patch_len: int, stride: int = None, dropout: float = 0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("PatchEmbedding expects input with shape [B, L, C]")
        x = x.permute(0, 2, 1)
        batch_size, n_vars, input_len = x.shape
        if input_len < self.patch_len:
            raise ValueError(f"input_len={input_len} is shorter than patch_len={self.patch_len}")

        remainder = (input_len - self.patch_len) % self.stride
        pad_right = (self.stride - remainder) % self.stride
        if pad_right > 0:
            x = F.pad(x, (0, pad_right), mode="replicate")

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.value_embedding(x)
        x = x + self.position_embedding(x).unsqueeze(1)
        return self.dropout(x), n_vars
