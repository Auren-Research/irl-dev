"""The single-state immutable-anchor IRL Core."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lunaris_dense_irl2.attention import build_attention_mask
from lunaris_dense_irl2.cache import LayerKVCache
from lunaris_dense_irl2.irl2 import SwiGLU
from lunaris_dense_irl2.norms import RMSNorm
from lunaris_dense_irl2.rope import rotate_half

from .config import LunarisIRLCoreConfig
from .diagnostics import IRLCoreDiagnostics


class IRLCore(nn.Module):
    """Repeatedly query one immutable causal anchor with a shared state."""

    def __init__(self, config: LunarisIRLCoreConfig) -> None:
        super().__init__()
        self.train_iters = int(config.irl_train_iters)
        self.eval_iters = int(config.irl_eval_iters)
        self.lambda_min = float(config.irl_lambda_min)
        self.lambda_max = float(config.irl_lambda_max)
        self.track_stats_by_default = bool(config.track_irl_stats)
        self.use_gradient_checkpointing = bool(config.use_gradient_checkpointing)
        self.gradient_checkpointing_policy = config.gradient_checkpointing_policy

        self.d_model = int(config.d_model)
        self.n_heads = int(config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.head_dim = int(config.head_dim)

        self.anchor_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.state_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.update_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps)
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps)
            if config.qk_norm
            else nn.Identity()
        )
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden_dim)
        self.attention_dropout = float(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lambda_logit = nn.Parameter(
            torch.full(
                (config.d_model,),
                float(config.lambda_logit()),
                dtype=torch.float32,
            )
        )

    @property
    def damping(self) -> torch.Tensor:
        """Return the learned channel-wise damping vector."""

        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(
            self.lambda_logit
        )

    @staticmethod
    def _apply_rope(
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to one projected query or key tensor."""

        return value * cos + rotate_half(value) * sin

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        """Return a scalar FP32 RMS."""

        return value.float().square().mean().sqrt()

    def _resolve_iterations(self, n_iters: Optional[int]) -> int:
        """Resolve the fixed train/eval depth or an explicit diagnostic override."""

        if n_iters is None:
            return self.train_iters if self.training else self.eval_iters
        if int(n_iters) <= 0:
            raise ValueError("n_iters must be positive")
        return int(n_iters)

    def _project_anchor(
        self,
        anchor: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project immutable anchor keys and values exactly once."""

        batch_size, sequence_length, _ = anchor.shape
        normalized_anchor = self.anchor_norm(anchor)
        key = self.k_proj(normalized_anchor).view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )
        value = self.v_proj(normalized_anchor).view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        return self._apply_rope(key, rope_cos, rope_sin), value

    def _project_query(
        self,
        recurrent_input: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Project one evolving state-conditioned query."""

        batch_size, sequence_length, _ = recurrent_input.shape
        query = self.q_proj(recurrent_input).view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        return self._apply_rope(query, rope_cos, rope_sin)

    def _native_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Run the same native grouped SDPA path as ordinary attention."""

        dropout_p = self.attention_dropout if self.training else 0.0
        if self.n_heads == self.n_kv_heads:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        try:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )
        except TypeError:
            repeats = self.n_heads // self.n_kv_heads
            return F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(repeats, dim=1),
                value.repeat_interleave(repeats, dim=1),
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def _read(
        self,
        recurrent_input: torch.Tensor,
        anchor_key: torch.Tensor,
        anchor_value: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        sdpa_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Read cached immutable evidence with one evolving query."""

        query = self._project_query(
            recurrent_input,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        attended = self._native_sdpa(
            query,
            anchor_key,
            anchor_value,
            attention_mask=sdpa_mask,
            is_causal=is_causal,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                recurrent_input.size(0),
                recurrent_input.size(1),
                self.d_model,
            )
        )
        return self.output_dropout(self.o_proj(attended))

    def _step(
        self,
        state: torch.Tensor,
        anchor: torch.Tensor,
        anchor_key: torch.Tensor,
        anchor_value: torch.Tensor,
        damping: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        *,
        sdpa_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply one shared recurrent transition."""

        recurrent_input = self.state_norm(anchor + state)
        read = self._read(
            recurrent_input,
            anchor_key,
            anchor_value,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            sdpa_mask=sdpa_mask,
            is_causal=is_causal,
        )
        nonlinear_update = self.ffn(self.update_norm(recurrent_input + read))
        target = read + nonlinear_update
        new_state = (1.0 - damping) * state + damping * target
        return new_state, read

    def forward(
        self,
        anchor: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Optional[LayerKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        n_iters: Optional[int] = None,
        return_stats: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[IRLCoreDiagnostics]]:
        """Return ``A + S_T`` after shared fixed-depth recurrent computation."""

        iterations = self._resolve_iterations(n_iters)
        track_stats = (
            self.track_stats_by_default if return_stats is None else bool(return_stats)
        )
        if self.training and self.use_gradient_checkpointing and track_stats:
            raise ValueError(
                "IRL diagnostics must be disabled while gradient "
                "checkpointing is active"
            )

        current_key, current_value = self._project_anchor(
            anchor,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        past_key: Optional[torch.Tensor] = None
        past_value: Optional[torch.Tensor] = None
        past_positions: Optional[torch.Tensor] = None
        if layer_cache is not None:
            past_key, past_value, past_positions = layer_cache.get()

        past_length = 0 if past_key is None else past_key.size(2)
        if past_key is not None and past_value is not None:
            if past_positions is None:
                raise RuntimeError("Cached anchor keys are missing positions")
            anchor_key = torch.cat((past_key, current_key), dim=2)
            anchor_value = torch.cat((past_value, current_value), dim=2)
            key_positions = torch.cat(
                (
                    past_positions.unsqueeze(0).expand(anchor.size(0), -1),
                    position_ids,
                ),
                dim=-1,
            )
        else:
            anchor_key = current_key
            anchor_value = current_value
            key_positions = position_ids

        fast_causal_path = (
            layer_cache is None
            and attention_mask is None
            and document_ids is None
            and anchor.size(1) > 1
        )
        if fast_causal_path:
            sdpa_mask = None
            is_causal = True
        else:
            sdpa_mask = build_attention_mask(
                query_positions=position_ids,
                key_positions=key_positions,
                sliding_window=None,
                attention_mask=attention_mask,
                document_ids=document_ids,
                past_length=past_length,
            )
            is_causal = False

        state = torch.zeros_like(anchor)
        damping = self.damping.to(dtype=anchor.dtype)
        use_step_checkpointing = (
            self.training
            and self.use_gradient_checkpointing
            and self.gradient_checkpointing_policy == "core"
            and not track_stats
        )

        state_rms: List[torch.Tensor] = []
        update_rms: List[torch.Tensor] = []
        relative_update_norm: List[torch.Tensor] = []
        read_rms: List[torch.Tensor] = []
        correction_anchor_ratio: List[torch.Tensor] = []
        adjacent_update_cosine: List[torch.Tensor] = []
        previous_update: Optional[torch.Tensor] = None
        anchor_rms = (
            self._rms(anchor).clamp_min(1e-12) if track_stats else None
        )

        for _ in range(iterations):
            if use_step_checkpointing:

                def checkpointed_step(
                    state_value: torch.Tensor,
                    anchor_value_tensor: torch.Tensor,
                    key_value: torch.Tensor,
                    value_value: torch.Tensor,
                    damping_value: torch.Tensor,
                    cos_value: torch.Tensor,
                    sin_value: torch.Tensor,
                ) -> torch.Tensor:
                    next_state, _ = self._step(
                        state_value,
                        anchor_value_tensor,
                        key_value,
                        value_value,
                        damping_value,
                        cos_value,
                        sin_value,
                        sdpa_mask=sdpa_mask,
                        is_causal=is_causal,
                    )
                    return next_state

                state = checkpoint(
                    checkpointed_step,
                    state,
                    anchor,
                    anchor_key,
                    anchor_value,
                    damping,
                    rope_cos,
                    rope_sin,
                    use_reentrant=False,
                )
                continue

            new_state, read = self._step(
                state,
                anchor,
                anchor_key,
                anchor_value,
                damping,
                rope_cos,
                rope_sin,
                sdpa_mask=sdpa_mask,
                is_causal=is_causal,
            )
            if track_stats:
                if anchor_rms is None:
                    raise RuntimeError("Diagnostic anchor RMS was not initialized")
                update = new_state - state
                update_norm = update.float().norm(dim=-1).mean()
                state_norm = new_state.float().norm(dim=-1).mean()
                state_rms.append(self._rms(new_state))
                update_rms.append(self._rms(update))
                relative_update_norm.append(
                    update_norm / state_norm.clamp_min(1e-12)
                )
                read_rms.append(self._rms(read))
                correction_anchor_ratio.append(self._rms(new_state) / anchor_rms)
                if previous_update is not None:
                    adjacent_update_cosine.append(
                        F.cosine_similarity(
                            update.float(),
                            previous_update.float(),
                            dim=-1,
                            eps=1e-12,
                        ).mean()
                    )
                previous_update = update
            state = new_state

        if layer_cache is not None:
            layer_cache.append(
                current_key.detach(),
                current_value.detach(),
                position_ids,
            )

        if not track_stats:
            return anchor + state, None

        empty = anchor.new_empty((0,), dtype=torch.float32)
        damping_stats = damping.detach().float()
        diagnostics = IRLCoreDiagnostics(
            state_rms=torch.stack(state_rms).detach(),
            update_rms=torch.stack(update_rms).detach(),
            relative_update_norm=torch.stack(relative_update_norm).detach(),
            read_rms=torch.stack(read_rms).detach(),
            correction_anchor_ratio=torch.stack(correction_anchor_ratio).detach(),
            adjacent_update_cosine=(
                torch.stack(adjacent_update_cosine).detach()
                if adjacent_update_cosine
                else empty
            ),
            damping_min=damping_stats.min(),
            damping_mean=damping_stats.mean(),
            damping_max=damping_stats.max(),
            iterations=iterations,
        )
        return anchor + state, diagnostics
