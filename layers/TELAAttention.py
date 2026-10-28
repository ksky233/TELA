import math

import torch
import torch.nn as nn


class TELAAttentionBlock(nn.Module):
    """Unified attention with temporal, exogenous, and wind-lag score bias."""

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        n_vars,
        patch_num,
        pred_len,
        target_index,
        top_k=0,
        rel_bias_weight=1.0,
        wind_lag_window=0,
        use_temporal_bias=True,
        use_exog_bias=True,
        use_wind_lag_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_vars = n_vars
        self.patch_num = patch_num
        self.pred_len = pred_len
        self.target_index = target_index
        self.num_history = n_vars * patch_num
        self.num_nodes = self.num_history + pred_len
        self.top_k = top_k
        self.base_rel_bias_weight = float(rel_bias_weight)
        self.rel_bias_weight = float(rel_bias_weight)
        self.wind_lag_window = wind_lag_window
        self.use_temporal_bias = use_temporal_bias
        self.use_exog_bias = use_exog_bias
        self.use_wind_lag_bias = use_wind_lag_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.temporal_bias = nn.Parameter(torch.zeros(n_heads, 1))
        self.channel_bias = nn.Parameter(torch.zeros(n_heads, 1))
        self.wind_lag_bias = nn.Parameter(torch.zeros(n_heads, pred_len, patch_num))

        masks = self._build_masks()
        for name, mask in masks.items():
            self.register_buffer(name, mask, persistent=False)

    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        q = self.q_proj(x).view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, num_nodes, self.n_heads, self.head_dim)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        scores = scores + self._relation_bias(scores.dtype, scores.device)

        if self._use_sparse_attention(num_nodes):
            scores = self._sparsify_scores(scores)

        attn = torch.softmax(scores, dim=-1)
        message = torch.einsum("bhij,bjhd->bihd", self.dropout(attn), value).reshape(
            batch_size, num_nodes, self.d_model
        )
        x = self.norm1(x + self.dropout(self.out_proj(message)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn

    def _use_sparse_attention(self, num_nodes):
        return bool(self.top_k and 0 < self.top_k < num_nodes)

    def _sparsify_scores(self, scores):
        keep = self._topk_keep_mask(scores, self.top_k)
        return scores.masked_fill(~keep, torch.finfo(scores.dtype).min)

    @staticmethod
    def _topk_keep_mask(scores, top_k):
        k = min(int(top_k), scores.size(-1))
        if k <= 0:
            return torch.ones_like(scores, dtype=torch.bool)
        _, indices = torch.topk(scores, k, dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool)
        return keep.scatter(-1, indices, True)

    def _relation_bias(self, dtype, device):
        bias = torch.zeros(
            1,
            self.n_heads,
            self.num_nodes,
            self.num_nodes,
            dtype=dtype,
            device=device,
        )
        if self.use_temporal_bias:
            bias = bias + self._temporal_relation_bias(dtype, device)
        if self.use_exog_bias:
            bias = bias + self._exogenous_relation_bias(dtype, device)
        if self.use_wind_lag_bias:
            bias = bias + self._lag_relation_bias(dtype, device)
        return self.rel_bias_weight * bias

    def _temporal_relation_bias(self, dtype, device):
        temporal = self.temporal_mask.to(device=device).view(1, 1, self.num_nodes, self.num_nodes)
        return temporal * self.temporal_bias.to(device=device, dtype=dtype).view(1, self.n_heads, 1, 1)

    def _exogenous_relation_bias(self, dtype, device):
        channel = self.channel_mask.to(device=device).view(1, 1, self.num_nodes, self.num_nodes)
        return channel * self.channel_bias.to(device=device, dtype=dtype).view(1, self.n_heads, 1, 1)

    def _lag_relation_bias(self, dtype, device):
        wind_mask = self.wind_lag_mask.to(device=device).view(1, 1, self.pred_len, self.num_history)
        wind_bias = torch.zeros(
            1,
            self.n_heads,
            self.pred_len,
            self.num_history,
            dtype=dtype,
            device=device,
        )
        wind_lag = self.wind_lag_bias.to(device=device, dtype=dtype)
        for patch_idx in range(self.patch_num):
            cols = slice(patch_idx, self.num_history, self.patch_num)
            wind_bias[:, :, :, cols] = wind_lag[:, :, patch_idx].view(1, self.n_heads, self.pred_len, 1)

        full_bias = torch.zeros(
            1,
            self.n_heads,
            self.num_nodes,
            self.num_nodes,
            dtype=dtype,
            device=device,
        )
        query_rows = slice(self.num_history, self.num_nodes)
        history_cols = slice(0, self.num_history)
        full_bias[:, :, query_rows, history_cols] = wind_bias * wind_mask
        return full_bias

    def _build_masks(self):
        var_ids = torch.arange(self.n_vars).repeat_interleave(self.patch_num)
        patch_ids = torch.arange(self.patch_num).repeat(self.n_vars)
        self.register_buffer("var_ids", var_ids, persistent=False)
        self.register_buffer("patch_ids", patch_ids, persistent=False)

        hist_n = self.num_history
        total_n = self.num_nodes

        temporal = torch.zeros(total_n, total_n, dtype=torch.bool)
        temporal[:hist_n, :hist_n] = (var_ids[:, None] == var_ids[None, :]) & (
            patch_ids[:, None] != patch_ids[None, :]
        )

        channel = torch.zeros(total_n, total_n, dtype=torch.bool)
        channel[:hist_n, :hist_n] = (patch_ids[:, None] == patch_ids[None, :]) & (
            var_ids[:, None] != var_ids[None, :]
        )

        wind_lag = torch.zeros(self.pred_len, hist_n, dtype=torch.bool)
        allowed_patches = self._allowed_lag_patches()
        exog_mask = var_ids != self.target_index
        for patch_idx in allowed_patches:
            patch_mask = patch_ids == patch_idx
            wind_lag[:, patch_mask & exog_mask] = True

        return {
            "temporal_mask": temporal,
            "channel_mask": channel,
            "wind_lag_mask": wind_lag,
        }

    def _allowed_lag_patches(self):
        if self.wind_lag_window <= 0 or self.wind_lag_window >= self.patch_num:
            return list(range(self.patch_num))
        start = self.patch_num - self.wind_lag_window
        return list(range(start, self.patch_num))
