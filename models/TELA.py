import math

import torch
import torch.nn as nn

from layers import (
    CumulativeDeltaHead,
    DirectShapeHead,
    FutureQuery,
    GatedShapeHead,
    PersistenceResidualHead,
    QuantilePersistenceResidualHead,
    RevIN,
    TELAAttentionBlock,
    VariableTimeTokenizer,
)


class Model(nn.Module):
    """TELA: unified temporal-exogenous-lag attention for wind power forecasting."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.target_index = getattr(configs, "target_index", self.enc_in - 1)
        self.d_model = configs.d_model
        self.patch_len = configs.patch_len
        self.patch_stride = getattr(configs, "patch_stride", self.patch_len)
        self.patch_num = self._calc_patch_num(self.seq_len, self.patch_len, self.patch_stride)
        self.use_revin = getattr(configs, "use_revin", True)
        self.output_mode = getattr(configs, "output_mode", "point")
        self.quantiles = list(getattr(configs, "quantiles", []))
        self.head_type = getattr(configs, "head_type", "persistence")
        self.use_future_exog = getattr(configs, "use_future_exog", False)
        if self.use_future_exog and self.enc_in <= 1:
            raise ValueError("use_future_exog=True requires at least one exogenous variable")

        self.revin = RevIN(self.enc_in) if self.use_revin else None
        self.history_tokenizer = VariableTimeTokenizer(
            n_vars=self.enc_in,
            d_model=self.d_model,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            patch_num=self.patch_num,
            dropout=configs.dropout,
        )
        self.query_type = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.future_query = FutureQuery(self.pred_len, self.d_model, dropout=configs.dropout)
        if self.use_future_exog:
            exog_dim = max(self.enc_in - 1, 1)
            self.future_exog_encoder = nn.Sequential(
                nn.LayerNorm(exog_dim),
                nn.Linear(exog_dim, self.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
            exog_indices = [idx for idx in range(self.enc_in) if idx != self.target_index]
            self.register_buffer(
                "future_exog_indices",
                torch.tensor(exog_indices, dtype=torch.long),
                persistent=False,
            )
        else:
            self.future_exog_encoder = None

        n_heads = getattr(configs, "n_heads", 4)
        d_ff = getattr(configs, "d_ff", self.d_model * 4)
        e_layers = getattr(configs, "e_layers", 2)
        self.blocks = nn.ModuleList(
            [
                TELAAttentionBlock(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    n_vars=self.enc_in,
                    patch_num=self.patch_num,
                    pred_len=self.pred_len,
                    target_index=self.target_index,
                    top_k=getattr(configs, "graph_top_k", 0),
                    rel_bias_weight=getattr(configs, "rel_bias_weight", 1.0),
                    wind_lag_window=getattr(configs, "wind_lag_window", 0),
                    use_temporal_bias=getattr(configs, "use_temporal_bias", True),
                    use_exog_bias=getattr(configs, "use_exog_bias", True),
                    use_wind_lag_bias=getattr(configs, "use_wind_lag_bias", True),
                    dropout=configs.dropout,
                )
                for _ in range(e_layers)
            ]
        )

        if self.output_mode == "quantile":
            if not self.quantiles:
                raise ValueError("quantile output mode requires non-empty quantiles")
            self.prediction_head = QuantilePersistenceResidualHead(
                self.d_model,
                d_ff,
                num_quantiles=len(self.quantiles),
                dropout=configs.dropout,
            )
        else:
            if self.head_type == "persistence":
                self.prediction_head = PersistenceResidualHead(self.d_model, d_ff, dropout=configs.dropout)
            elif self.head_type == "direct":
                self.prediction_head = DirectShapeHead(self.d_model, d_ff, dropout=configs.dropout)
            elif self.head_type == "cumulative_delta":
                self.prediction_head = CumulativeDeltaHead(self.d_model, d_ff, dropout=configs.dropout)
            elif self.head_type == "gated_shape":
                self.prediction_head = GatedShapeHead(
                    self.d_model,
                    d_ff,
                    pred_len=self.pred_len,
                    dropout=configs.dropout,
                    gate_init=getattr(configs, "head_gate_init", 0.85),
                )
            else:
                raise ValueError(f"Unsupported head_type: {self.head_type}")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        batch_size, _, n_vars = x_enc.shape
        if n_vars != self.enc_in:
            raise ValueError(f"Expected enc_in={self.enc_in}, got {n_vars}")

        if self.use_revin:
            x = self.revin(x_enc, mode="norm")
        else:
            x = x_enc

        history_tokens, _ = self.history_tokenizer(x)
        query_tokens = self.future_query(history_tokens) + self.query_type
        if self.use_future_exog:
            if x_dec is None:
                raise ValueError("use_future_exog=True requires x_dec with future exogenous variables")
            future_exog = x_dec[:, -self.pred_len :, :].to(device=x.device, dtype=x.dtype)
            future_exog = future_exog.index_select(dim=-1, index=self.future_exog_indices.to(x.device))
            query_tokens = query_tokens + self.future_exog_encoder(future_exog)
        tokens = torch.cat([history_tokens, query_tokens], dim=1)

        for block in self.blocks:
            tokens, _ = block(tokens)

        query_tokens = tokens[:, -self.pred_len :, :]
        output = self.prediction_head(query_tokens, x, self.target_index)

        if self.use_revin:
            output = self.revin(output, mode="denorm", target_index=self.target_index)
        return output

    @staticmethod
    def _calc_patch_num(seq_len, patch_len, stride):
        remainder = (seq_len - patch_len) % stride
        pad_right = (stride - remainder) % stride
        return math.floor((seq_len + pad_right - patch_len) / stride) + 1
