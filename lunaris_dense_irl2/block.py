"""One dense Transformer block centered on IRL-2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import GroupedQueryAttention
from .cache import LayerKVCache
from .config import LunarisDenseConfig
from .diagnostics import IRLDiagnostics
from .irl2 import IRL2Cell
from .norms import IdentityNorm, RMSNorm


class LunarisDenseBlock(nn.Module):
    """Apply attention once and recurrent dense refinement afterwards."""

    def __init__(self, config: LunarisDenseConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)

        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config, layer_index)
        self.attention_post_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.sandwich_norm
            else IdentityNorm()
        )
        self.attention_layer_scale = nn.Parameter(torch.ones(config.d_model))

        self.irl_anchor_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.irl = IRL2Cell(config)

    def _run_irl(
        self,
        anchor: torch.Tensor,
        *,
        n_iters: Optional[int],
        return_stats: bool,
        return_consistency: bool,
    ) -> Tuple[torch.Tensor, Optional[IRLDiagnostics], torch.Tensor]:
        """Run IRL directly or checkpoint its tensor-only path."""

        should_checkpoint = (
            self.training
            and self.config.use_gradient_checkpointing
            and self.config.gradient_checkpointing_policy == "irl"
            and not return_stats
        )
        if not should_checkpoint:
            return self.irl(
                anchor,
                n_iters=n_iters,
                return_stats=return_stats,
                return_consistency=return_consistency,
            )

        def checkpointed_irl(
            value: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            correction, _, consistency = self.irl(
                value,
                n_iters=n_iters,
                return_stats=False,
                return_consistency=return_consistency,
            )
            return correction, consistency

        correction, consistency = checkpoint(
            checkpointed_irl,
            anchor,
            use_reentrant=False,
        )
        return correction, None, consistency

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
        n_iters: Optional[int],
        return_irl_stats: bool,
        return_irl_consistency: bool,
    ) -> Tuple[torch.Tensor, Optional[IRLDiagnostics], torch.Tensor]:
        """Run a complete dense block."""

        normalized = self.attention_norm(hidden_states)
        attention_output = self.attention(
            normalized,
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        attention_output = self.attention_post_norm(attention_output)
        attention_output = attention_output * self.attention_layer_scale.to(
            attention_output.dtype
        )
        attended_states = hidden_states + attention_output

        anchor = self.irl_anchor_norm(attended_states)
        correction, diagnostics, consistency = self._run_irl(
            anchor,
            n_iters=n_iters,
            return_stats=return_irl_stats,
            return_consistency=return_irl_consistency,
        )
        return attended_states + correction, diagnostics, consistency
