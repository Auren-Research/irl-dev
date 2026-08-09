"""Ordinary causal Transformer layers surrounding the IRL-3 computer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lunaris_dense_irl2.attention import GroupedQueryAttention
from lunaris_dense_irl2.cache import LayerKVCache
from lunaris_dense_irl2.irl2 import SwiGLU
from lunaris_dense_irl2.norms import IdentityNorm, RMSNorm

from .config import LunarisIRL3Config


class LunarisIRL3TransformerBlock(nn.Module):
    """A fixed-depth attention/FFN block used as anchor prelude or coda."""

    def __init__(self, config: LunarisIRL3Config, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config, layer_index)
        self.attention_post_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.sandwich_norm
            else IdentityNorm()
        )
        self.attention_scale = nn.Parameter(torch.ones(config.d_model))

        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden_dim)
        self.ffn_post_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.sandwich_norm
            else IdentityNorm()
        )
        self.ffn_scale = nn.Parameter(torch.ones(config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Optional[LayerKVCache],
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.attention_norm(hidden_states),
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        attention_output = self.attention_post_norm(attention_output)
        hidden_states = hidden_states + attention_output * self.attention_scale.to(
            attention_output.dtype
        )

        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        ffn_output = self.ffn_post_norm(ffn_output)
        return hidden_states + self.dropout(ffn_output) * self.ffn_scale.to(
            ffn_output.dtype
        )
