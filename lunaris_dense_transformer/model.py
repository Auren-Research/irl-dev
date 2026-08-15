"""Conventional decoder-only Transformer used by the controlled benchmark."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from lunaris_dense_irl2.cache import LayerKVCache
from lunaris_dense_irl2.initialization import initialize_module
from lunaris_dense_irl2.model import LunarisDenseForCausalLM
from lunaris_dense_irl2.norms import RMSNorm
from lunaris_dense_irl2.rope import RotaryEmbedding
from lunaris_dense_irl_core.block import TransformerBlock

from .config import LunarisTransformerConfig


class _BenchmarkTransformerBlock(TransformerBlock):
    """Adapt an ordinary block to the shared training model interface."""

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
        n_iters: Optional[int] = None,
        return_irl_stats: bool = False,
        return_irl_consistency: bool = False,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """Run one non-recurrent attention-plus-SwiGLU block."""

        del n_iters, return_irl_stats, return_irl_consistency
        output = super().forward(
            hidden_states,
            position_ids=position_ids,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        return output, None, output.new_zeros((), dtype=torch.float32)


class LunarisTransformerForCausalLM(LunarisDenseForCausalLM):
    """Decoder-only baseline with no recurrent or auxiliary trainable state."""

    def __init__(self, config: LunarisTransformerConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.regional_compile_enabled = True

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                _BenchmarkTransformerBlock(config, layer_index)
                for layer_index in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_predictor = None

        self.global_rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta_global,
            scaling_factor=config.rope_scaling_factor,
        )
        self.local_rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta_local,
            scaling_factor=config.rope_scaling_factor,
        )

        self.apply(lambda module: initialize_module(module, config))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.scale_residual_projections:
            self._scale_residual_projections()

    def _scale_residual_projections(self) -> None:
        """Use the same depth-aware scaling as the qualified IRL Core."""

        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attention.o_proj.weight.mul_(scale)
                block.ffn.down_proj.weight.mul_(scale)

    def get_recurrent_num_parameters(self, *, trainable_only: bool = True) -> int:
        """Return zero because this architecture has no recurrent component."""

        del trainable_only
        return 0

    def effective_attention_calls(self, n_iters: Optional[int] = None) -> int:
        """Return the number of ordinary attention calls per forward."""

        del n_iters
        return self.config.n_layers

    def save_pretrained(self, directory: str | Path) -> None:
        """Save a versioned state dict and architecture configuration."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.config.save_json(destination / "config.json")
        torch.save(
            {
                "architecture": self.config.architecture_name,
                "format_version": self.config.checkpoint_format_version,
                "state_dict": self.state_dict(),
            },
            destination / "model.pt",
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "LunarisTransformerForCausalLM":
        """Load a version-compatible portable checkpoint."""

        source = Path(directory)
        config = LunarisTransformerConfig.from_json(source / "config.json")
        payload = torch.load(
            source / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        if payload.get("architecture") != config.architecture_name:
            raise ValueError("Checkpoint architecture does not match the configuration")
        if payload.get("format_version") != config.checkpoint_format_version:
            raise ValueError("Unsupported checkpoint format version")
        model = cls(config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Use the shared cache-aware autoregressive generator."""

        from lunaris_dense_irl2.generation import generate

        return generate(self, input_ids, **kwargs)
