import torch.nn as nn
import torch


class PersistenceResidualHead(nn.Module):
    """Predict future deviations from the last observed target value."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, query_tokens, x, target_index):
        residual = self.readout(query_tokens)
        last_target = x[:, -1:, target_index : target_index + 1].repeat(1, query_tokens.size(1), 1)
        return last_target + residual


class DirectShapeHead(nn.Module):
    """Predict the future target curve directly from horizon query tokens."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, query_tokens, x, target_index):
        return self.readout(query_tokens)


class GatedShapeHead(nn.Module):
    """Blend a learnable horizon-wise persistence gate with a direct shape head."""

    def __init__(self, d_model, d_ff, pred_len, dropout=0.0, gate_init=0.85):
        super().__init__()
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        init_logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
        self.persistence_gate = nn.Parameter(torch.full((1, pred_len, 1), float(init_logit)))

    def forward(self, query_tokens, x, target_index):
        shape = self.readout(query_tokens)
        last_target = x[:, -1:, target_index : target_index + 1].repeat(1, query_tokens.size(1), 1)
        gate = torch.sigmoid(self.persistence_gate).to(dtype=query_tokens.dtype, device=query_tokens.device)
        return gate * last_target + (1.0 - gate) * shape


class CumulativeDeltaHead(nn.Module):
    """Predict horizon-wise target increments and integrate them from the last target value."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, query_tokens, x, target_index):
        deltas = self.readout(query_tokens)
        last_target = x[:, -1:, target_index : target_index + 1]
        return last_target + torch.cumsum(deltas, dim=1)


class QuantilePersistenceResidualHead(nn.Module):
    """Predict quantile-specific future deviations from the last target value."""

    def __init__(self, d_model, d_ff, num_quantiles, dropout=0.0):
        super().__init__()
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_quantiles),
        )

    def forward(self, query_tokens, x, target_index):
        residual = self.readout(query_tokens)
        last_target = x[:, -1:, target_index].unsqueeze(-1).repeat(1, query_tokens.size(1), residual.size(-1))
        return last_target + residual
