"""Numerically stable normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm with FP32 statistics and a learnable scale."""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the final dimension while preserving the input dtype."""

        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = x_float * inverse_rms
        return (normalized * self.weight.float()).to(input_dtype)


class IdentityNorm(nn.Module):
    """Identity layer with the same call signature as normalization modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
