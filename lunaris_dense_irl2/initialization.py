"""Centralized parameter initialization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import LunarisDenseConfig


def initialize_module(module: nn.Module, config: LunarisDenseConfig) -> None:
    """Apply the common normal initialization to matrix parameters."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()


def rescale_residual_projections(
    blocks: nn.ModuleList,
    *,
    n_layers: int,
) -> None:
    """Scale residual output matrices according to total model depth."""

    scale = 1.0 / math.sqrt(2.0 * n_layers)
    with torch.no_grad():
        for block in blocks:
            block.attention.o_proj.weight.mul_(scale)
            block.irl.ffn.down_proj.weight.mul_(scale)
