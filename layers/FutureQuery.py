import torch
import torch.nn as nn


class FutureQuery(nn.Module):
    """Learnable horizon queries conditioned on historical patch tokens."""

    def __init__(self, pred_len: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.pred_len = pred_len
        self.query = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history_tokens):
        batch_size = history_tokens.size(0)
        context = history_tokens.mean(dim=1)
        query = self.query.expand(batch_size, -1, -1)
        query = query + self.context_proj(context).unsqueeze(1)
        return self.norm(query)
