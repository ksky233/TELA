import torch
import torch.nn as nn

from .PatchEmbedding import PatchEmbedding


class VariableTimeTokenizer(nn.Module):
    """Tokenize wide-table history into variable-time patch tokens."""

    def __init__(self, n_vars, d_model, patch_len, patch_stride, patch_num, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.patch_num = patch_num
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=patch_stride,
            dropout=dropout,
        )
        self.variable_embedding = nn.Embedding(n_vars, d_model)
        self.history_type = nn.Parameter(torch.zeros(1, 1, 1, d_model))

    def forward(self, x):
        batch_size, _, n_vars = x.shape
        if n_vars != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {n_vars}")

        patch_grid, _ = self.patch_embedding(x)
        if patch_grid.size(2) != self.patch_num:
            raise RuntimeError(
                f"Patch number mismatch: expected {self.patch_num}, got {patch_grid.size(2)}"
            )

        var_ids = torch.arange(self.n_vars, device=x.device)
        var_emb = self.variable_embedding(var_ids).view(1, self.n_vars, 1, -1)
        history_grid = patch_grid + var_emb + self.history_type
        history_tokens = history_grid.reshape(batch_size, -1, history_grid.size(-1))
        return history_tokens, history_grid
