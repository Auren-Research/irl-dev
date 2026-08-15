"""Ordinary Transformer blocks used before and after the IRL Core."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lunaris_dense_irl2.attention import GroupedQueryAttention
from lunaris_dense_irl2.cache import LayerKVCache
from lunaris_dense_irl2.irl2 import SwiGLU
from lunaris_dense_irl2.norms import IdentityNorm, RMSNorm

from .config import LunarisIRLCoreConfig


class TransformerBlock(nn.Module):
    """Pre-norm causal GQA followed by a pre-norm SwiGLU residual."""

    def __init__(self, config: LunarisIRLCoreConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config, self.layer_index)
        self.attention_post_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.sandwich_norm
            else IdentityNorm()
        )
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden_dim)
        self.ffn_dropout = nn.Dropout(config.dropout)

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
        """Apply one ordinary causal Transformer block."""

        attended = self.attention(
            self.attention_norm(hidden_states),
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        hidden_states = hidden_states + self.attention_post_norm(attended)
        transformed = self.ffn(self.ffn_norm(hidden_states))
        return hidden_states + self.ffn_dropout(transformed)
