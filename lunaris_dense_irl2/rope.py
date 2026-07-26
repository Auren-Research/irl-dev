"""Rotary position embeddings with optional position interpolation."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs by ninety degrees."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """Compute RoPE cosine and sine values from absolute positions."""

    def __init__(
        self,
        head_dim: int,
        theta: float,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if theta <= 0.0:
            raise ValueError("theta must be positive")
        if scaling_factor <= 0.0:
            raise ValueError("scaling_factor must be positive")

        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.head_dim = int(head_dim)
        self.scaling_factor = float(scaling_factor)
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcastable cosine and sine tensors."""

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.dim() != 2:
            raise ValueError(
                "position_ids must have shape [sequence] or [batch, sequence]"
            )

        positions = position_ids.float() / self.scaling_factor
        phase = torch.einsum("bs,d->bsd", positions, self.inverse_frequencies.float())
        phase = torch.repeat_interleave(phase, repeats=2, dim=-1)
        cos = phase.cos().unsqueeze(1).to(dtype)
        sin = phase.sin().unsqueeze(1).to(dtype)
        return cos, sin


def apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one RoPE transform to query and key tensors."""

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )
