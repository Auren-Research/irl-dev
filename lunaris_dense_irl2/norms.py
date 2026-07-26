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
        variance = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return normalized.to(input_dtype) * self.weight.to(input_dtype)


class IdentityNorm(nn.Module):
    """Identity layer with the same call signature as normalization modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
