import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible instance normalization for non-stationary time series."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        self._mean = None
        self._stdev = None

    def forward(self, x, mode: str, target_index: int = None):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            var = torch.var(x, dim=1, keepdim=True, unbiased=False)
            self._stdev = torch.sqrt(var + self.eps).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        if mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("RevIN denorm called before norm")

            mean = self._mean
            stdev = self._stdev
            weight = self.affine_weight if self.affine else None
            bias = self.affine_bias if self.affine else None
            if target_index is not None:
                mean = mean[..., target_index : target_index + 1]
                stdev = stdev[..., target_index : target_index + 1]
                if self.affine:
                    weight = weight[..., target_index : target_index + 1]
                    bias = bias[..., target_index : target_index + 1]

            if self.affine:
                x = (x - bias) / (weight + self.eps * self.eps)
            return x * stdev + mean

        raise ValueError("mode must be 'norm' or 'denorm'")
