"""IRL-3: a causal stateful latent recurrent computer.

Teacher-forced training executes one small computer per causal prefix in
parallel. Cached decoding executes the same program only for the newest token.
The recurrent state separates fast workspace ``Z``, slow strategy ``H``, and
writable working memory ``M`` while immutable anchor K/V are reused at every
reasoning step.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lunaris_dense_irl2.cache import LayerKVCache
from lunaris_dense_irl2.irl2 import SwiGLU
from lunaris_dense_irl2.norms import RMSNorm

from .config import LunarisIRL3Config
from .diagnostics import IRL3Diagnostics, IRL3State


def _rms(x: torch.Tensor, dimensions: Tuple[int, ...]) -> torch.Tensor:
    """Return a zero-preserving, finite-gradient FP32 RMS reduction.

    A literal ``sqrt(mean(x**2))`` has an undefined derivative at an exactly
    unchanged state. That case is routine for the slow state between clock
    ticks, so an unstabilized diagnostic can poison the next phase embedding
    even while every forward activation is finite.
    """

    epsilon = 1e-12
    mean_square = x.float().square().mean(dim=dimensions, keepdim=True)
    return torch.sqrt(mean_square + epsilon) - math.sqrt(epsilon)


class StateAttention(nn.Module):
    """Multi-head attention over small state-slot arrays.

    Inputs use ``[batch, prefix, slots, state_dim]``. Prefixes never interact;
    this invariant is what makes the latent computer causal under parallel
    teacher forcing.
    """

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        self.dimension = config.state_dim
        self.n_heads = config.irl3_state_heads
        self.head_dim = config.state_head_dim
        self.dropout_p = float(config.attention_dropout)

        self.q_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.q_proj = nn.Linear(self.dimension, self.dimension, bias=False)
        self.k_proj = nn.Linear(self.dimension, self.dimension, bias=False)
        self.v_proj = nn.Linear(self.dimension, self.dimension, bias=False)
        self.o_proj = nn.Linear(self.dimension, self.dimension, bias=False)

    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
    ) -> torch.Tensor:
        if query_states.dim() != 4 or key_value_states.dim() != 4:
            raise ValueError("State attention expects [batch, prefix, slots, dim]")
        if query_states.shape[:2] != key_value_states.shape[:2]:
            raise ValueError("State attention prefixes must match")

        batch_size, prefixes, query_slots, _ = query_states.shape
        key_slots = key_value_states.size(2)
        query = self.q_proj(self.q_norm(query_states)).view(
            batch_size * prefixes,
            query_slots,
            self.n_heads,
            self.head_dim,
        )
        key_value = self.kv_norm(key_value_states)
        key = self.k_proj(key_value).view(
            batch_size * prefixes,
            key_slots,
            self.n_heads,
            self.head_dim,
        )
        value = self.v_proj(key_value).view(
            batch_size * prefixes,
            key_slots,
            self.n_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, prefixes, query_slots, self.dimension)
        )
        return self.o_proj(attended)


class AnchorReader(nn.Module):
    """Reuse fixed causal evidence K/V while evolving latent queries."""

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        self.dimension = config.state_dim
        self.n_heads = config.irl3_state_heads
        self.head_dim = config.state_head_dim
        self.window = config.irl3_anchor_window
        self.dropout_p = float(config.attention_dropout)

        self.anchor_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.query_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.query_proj = nn.Linear(self.dimension, self.dimension, bias=False)
        self.key_proj = nn.Linear(config.d_model, self.dimension, bias=False)
        self.value_proj = nn.Linear(config.d_model, self.dimension, bias=False)
        self.output_proj = nn.Linear(self.dimension, self.dimension, bias=False)
        self.q_head_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_head_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_anchor(
        self,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = anchor.shape
        normalized = self.anchor_norm(anchor)
        key = self.key_proj(normalized).view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        value = self.value_proj(normalized).view_as(key)
        key = self.k_head_norm(key)
        return key, value

    @staticmethod
    def _window_view(value: torch.Tensor, window: int) -> torch.Tensor:
        """Return trailing windows as ``[B, K, H, W, D]`` without copies."""

        padded = F.pad(value, (0, 0, 0, 0, window - 1, 0))
        return padded.unfold(1, window, 1).permute(0, 1, 2, 4, 3)

    def forward(
        self,
        latents: torch.Tensor,
        evidence_key: torch.Tensor,
        evidence_value: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
        return_entropy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_length, latent_slots, _ = latents.shape
        if evidence_key.shape != evidence_value.shape:
            raise ValueError("Anchor evidence K/V shapes must match")
        if evidence_key.dim() != 4 or evidence_key.size(0) != batch_size:
            raise ValueError(
                "Anchor evidence must have shape [batch, token, head, dim]"
            )
        key_length = evidence_key.size(1)
        if key_length < query_length:
            raise ValueError("Anchor evidence cannot be shorter than its queries")

        if attention_mask is None:
            valid = torch.ones(
                batch_size,
                key_length,
                device=latents.device,
                dtype=torch.bool,
            )
        else:
            if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            if attention_mask.size(1) != key_length:
                raise ValueError("attention_mask must cover the complete anchor")
            valid = attention_mask.to(torch.bool)

        window = min(self.window, key_length)
        key_windows = self._window_view(evidence_key, window)[:, -query_length:]
        value_windows = self._window_view(evidence_value, window)[:, -query_length:]
        valid_windows = F.pad(valid, (window - 1, 0), value=False).unfold(1, window, 1)[
            :, -query_length:
        ]

        if document_ids is not None:
            if key_length != query_length:
                raise ValueError("document_ids are not supported with anchor cache")
            if document_ids.shape != (batch_size, query_length):
                raise ValueError("document_ids must match the current token sequence")
            padded_docs = F.pad(document_ids, (window - 1, 0), value=-1)
            doc_windows = padded_docs.unfold(1, window, 1)[:, -query_length:]
            valid_windows = valid_windows & doc_windows.eq(document_ids.unsqueeze(-1))

        query = self.query_proj(self.query_norm(latents)).view(
            batch_size,
            query_length,
            latent_slots,
            self.n_heads,
            self.head_dim,
        )
        query = self.q_head_norm(query)
        scores = torch.einsum("bqrhd,bqhwd->bqrhw", query, key_windows)
        scores = scores.float() / math.sqrt(self.head_dim)
        allowed = valid_windows[:, :, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities)
        if self.training and self.dropout_p > 0.0:
            probabilities = F.dropout(probabilities, p=self.dropout_p)
        attended = torch.einsum(
            "bqrhw,bqhwd->bqrhd",
            probabilities.to(value_windows.dtype),
            value_windows,
        ).reshape(batch_size, query_length, latent_slots, self.dimension)
        output = self.output_proj(attended)

        entropy = output.new_zeros((), dtype=torch.float32)
        if return_entropy:
            log_probabilities = probabilities.clamp_min(1e-9).log()
            entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
            entropy = entropy / math.log(max(2, window))

        return output, entropy


class PhaseEncoder(nn.Module):
    """Continuous computation-phase and budget conditioning."""

    feature_count = 12

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        hidden = max(32, config.state_dim // 2)
        self.maximum_steps = config.irl3_max_steps
        self.slow_interval = config.irl3_slow_update_interval
        self.input = nn.Linear(self.feature_count, hidden, bias=True)
        self.output = nn.Linear(hidden, config.state_dim, bias=False)

    def forward(
        self,
        *,
        step: int,
        budget: int,
        metrics: Dict[str, torch.Tensor],
        survival: torch.Tensor,
    ) -> torch.Tensor:
        reference = metrics["z"]
        shape = reference.shape

        def constant(value: float) -> torch.Tensor:
            return reference.new_full(shape, float(value))

        progress = step / max(1, budget - 1)
        remaining = (budget - step - 1) / max(1, budget)
        absolute = (step + 1) / self.maximum_steps
        budget_scale = math.log1p(budget) / math.log1p(self.maximum_steps)
        clock = (step % self.slow_interval) / max(1, self.slow_interval - 1)
        features = [
            constant(progress),
            constant(remaining),
            constant(absolute),
            constant(budget_scale),
            constant(math.sin(math.pi * progress)),
            constant(math.cos(math.pi * progress)),
            constant(clock),
            torch.log1p(metrics["z"].clamp_min(0.0)),
            torch.log1p(metrics["h"].clamp_min(0.0)),
            torch.log1p(metrics["m"].clamp_min(0.0)),
            torch.log1p(metrics["memory_read"].clamp_min(0.0)),
            survival.squeeze(-1).to(reference.dtype),
        ]
        encoded = torch.cat(features, dim=-1)
        encoded = encoded.to(dtype=self.input.weight.dtype)
        return self.output(F.silu(self.input(encoded)))


@dataclass
class IRL3ComputerResult:
    correction: torch.Tensor
    state: IRL3State
    ponder_loss: torch.Tensor
    equilibrium_loss: torch.Tensor
    diagnostics: Optional[IRL3Diagnostics]


class IRL3Computer(nn.Module):
    """Shared neural program implementing read-think-write-strategy-halt."""

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        self.config = config
        self.dimension = config.state_dim
        self.train_steps = config.irl3_train_steps
        self.eval_steps = config.irl3_eval_steps
        self.maximum_steps = config.irl3_max_steps
        self.minimum_steps = config.irl3_min_steps
        self.gradient_horizon = config.irl3_gradient_horizon
        self.slow_interval = config.irl3_slow_update_interval
        self.halt_threshold = float(config.irl3_halt_threshold)
        self.active_threshold = float(config.irl3_active_gate_threshold)
        self.memory_equilibrium_weight = float(config.irl3_equilibrium_memory_weight)
        self.track_stats_by_default = bool(config.track_irl3_stats)

        self.z_slots = nn.Parameter(
            torch.empty(config.irl3_num_latents, self.dimension)
        )
        self.h_slots = nn.Parameter(torch.empty(config.irl3_slow_slots, self.dimension))
        self.m_slots = nn.Parameter(
            torch.empty(config.irl3_memory_slots, self.dimension)
        )
        self.z_seed = nn.Linear(config.d_model, self.dimension, bias=False)
        self.h_seed = nn.Linear(config.d_model, self.dimension, bias=False)
        self.m_seed = nn.Linear(config.d_model, self.dimension, bias=False)
        self.z_init_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.h_init_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.m_init_norm = RMSNorm(self.dimension, config.rms_norm_eps)

        self.anchor_reader = AnchorReader(config)
        self.memory_reader = StateAttention(config)
        self.strategy_reader = StateAttention(config)
        self.phase_encoder = PhaseEncoder(config)

        self.fast_state_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.fast_read_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.fast_delta_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.fast_exit_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.read_projection = nn.Linear(self.dimension, self.dimension, bias=False)
        self.read_gate = nn.Linear(2 * self.dimension, self.dimension)
        self.fast_attention = StateAttention(config)
        self.fast_ffn_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.fast_ffn = SwiGLU(self.dimension, config.state_ffn_dim)
        self.fast_update_gate = nn.Linear(3 * self.dimension, self.dimension)
        self.z_prior_projection = nn.Linear(self.dimension, self.dimension, bias=False)

        self.memory_writer = StateAttention(config)
        self.memory_state_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.memory_exit_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.memory_erase = nn.Linear(self.dimension, self.dimension)
        self.memory_add = nn.Linear(self.dimension, self.dimension)
        self.memory_write_gate = nn.Linear(self.dimension, 1)
        self.m_prior_projection = nn.Linear(self.dimension, self.dimension, bias=False)

        self.slow_reader = StateAttention(config)
        self.slow_attention = StateAttention(config)
        self.slow_state_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.slow_delta_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.slow_exit_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.slow_ffn_norm = RMSNorm(self.dimension, config.rms_norm_eps)
        self.slow_ffn = SwiGLU(self.dimension, config.state_ffn_dim)
        self.slow_update_gate = nn.Linear(3 * self.dimension, self.dimension)
        self.h_prior_projection = nn.Linear(self.dimension, self.dimension, bias=False)

        halt_hidden = max(32, self.dimension // 2)
        self.halt_hidden = nn.Linear(4 * self.dimension + 1, halt_hidden)
        self.halt_output = nn.Linear(halt_hidden, 1)

        self.write_query_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.write_query = nn.Linear(config.d_model, self.dimension, bias=False)
        self.write_attention = StateAttention(config)
        self.write_output = nn.Linear(self.dimension, config.d_model, bias=False)
        self.write_gate = nn.Linear(2 * self.dimension, config.d_model)

        self.fast_scale = nn.Parameter(
            torch.full((self.dimension,), float(config.irl3_update_scale_init))
        )
        self.slow_scale = nn.Parameter(
            torch.full((self.dimension,), float(config.irl3_update_scale_init))
        )
        self.writeback_scale = nn.Parameter(
            torch.full((config.d_model,), float(config.irl3_writeback_scale_init))
        )

        nn.init.normal_(self.z_slots, std=config.initializer_range)
        nn.init.normal_(self.h_slots, std=config.initializer_range)
        nn.init.normal_(self.m_slots, std=config.initializer_range)
        nn.init.constant_(
            self.memory_write_gate.bias, float(config.irl3_memory_gate_init)
        )
        nn.init.constant_(self.halt_output.bias, float(config.irl3_halt_bias_init))
        nn.init.constant_(self.write_gate.bias, float(config.irl3_writeback_gate_init))

    def _resolve_depth(self, compute_depth: Optional[int]) -> int:
        if compute_depth is None:
            return self.train_steps if self.training else self.eval_steps
        depth = int(compute_depth)
        if not (self.minimum_steps <= depth <= self.maximum_steps):
            raise ValueError(
                f"compute_depth must be in [{self.minimum_steps}, {self.maximum_steps}]"
            )
        return depth

    def _resolve_halt_mode(self, halt_mode: Optional[str]) -> str:
        if halt_mode is None:
            return "act" if self.training else self.config.irl3_eval_halt_mode
        mode = halt_mode.lower()
        if mode not in {"fixed", "act", "adaptive"}:
            raise ValueError("halt_mode must be fixed, act, or adaptive")
        if self.training and mode == "adaptive":
            raise ValueError("adaptive hard halting is evaluation-only")
        return mode

    def _conditioned_state(
        self,
        anchor: torch.Tensor,
        persistent_state: Optional[IRL3State],
    ) -> Tuple[IRL3State, IRL3State]:
        batch_size, prefixes, _ = anchor.shape
        z = self.z_seed(anchor).unsqueeze(2) + self.z_slots.view(
            1, 1, -1, self.dimension
        )
        h = self.h_seed(anchor).unsqueeze(2) + self.h_slots.view(
            1, 1, -1, self.dimension
        )
        m = self.m_seed(anchor).unsqueeze(2) + self.m_slots.view(
            1, 1, -1, self.dimension
        )
        z = self.z_init_norm(z)
        h = self.h_init_norm(h)
        m = self.m_init_norm(m)
        seeds = IRL3State(z=z, h=h, m=m)

        if persistent_state is None:
            return seeds, seeds
        if prefixes != 1:
            raise ValueError(
                "A persistent reasoning state can initialize only one new prefix"
            )
        expected = {
            "z": (batch_size, self.config.irl3_num_latents, self.dimension),
            "h": (batch_size, self.config.irl3_slow_slots, self.dimension),
            "m": (batch_size, self.config.irl3_memory_slots, self.dimension),
        }
        for name, shape in expected.items():
            value = getattr(persistent_state, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Persistent {name} state must have shape {shape}, "
                    f"got {tuple(value.shape)}"
                )

        def fuse(
            seed: torch.Tensor,
            carried: torch.Tensor,
            projection: nn.Linear,
        ) -> torch.Tensor:
            carried = carried.to(device=seed.device, dtype=seed.dtype).unsqueeze(1)
            # Persistent fusion has no parameters used only in streaming mode.
            # It reuses prior projections trained at every recurrent step and
            # assigns more weight to a carried slot when its direction agrees
            # with the new problem-conditioned seed. This keeps ordinary
            # pretraining checkpoints complete while gradients can still flow
            # through token-time state whenever streaming training is added.
            seed_direction = F.normalize(seed.float(), dim=-1)
            carried_direction = F.normalize(carried.float(), dim=-1)
            agreement = (seed_direction * carried_direction).sum(dim=-1, keepdim=True)
            gate = torch.sigmoid(agreement).to(seed.dtype)
            projected = projection(carried)
            return seed + gate * (projected - seed)

        return (
            IRL3State(
                z=fuse(z, persistent_state.z, self.z_prior_projection),
                h=fuse(h, persistent_state.h, self.h_prior_projection),
                m=fuse(m, persistent_state.m, self.m_prior_projection),
            ),
            seeds,
        )

    def _halt_probability(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
        phase: torch.Tensor,
        joint_residual: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        pooled = torch.cat(
            (
                z.mean(dim=2),
                h.mean(dim=2),
                m.mean(dim=2),
                phase,
                joint_residual,
            ),
            dim=-1,
        )
        pooled = pooled.to(dtype=self.halt_hidden.weight.dtype)
        probability = torch.sigmoid(self.halt_output(F.silu(self.halt_hidden(pooled))))
        if step + 1 < self.minimum_steps:
            probability = probability * 0.0
        return probability.unsqueeze(-1)

    def _step(
        self,
        state: IRL3State,
        seeds: IRL3State,
        *,
        phase: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
        step: int,
        active: Optional[torch.Tensor],
        track_stats: bool,
    ) -> Tuple[IRL3State, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        z, h, m = state.z, state.h, state.m
        anchor_read, anchor_entropy = self.anchor_reader(
            z,
            current_key,
            current_value,
            attention_mask=attention_mask,
            document_ids=document_ids,
            return_entropy=track_stats,
        )
        memory_read = self.memory_reader(z, m)
        strategy_read = self.strategy_reader(z, h)

        phase_z = phase.unsqueeze(2).expand_as(z)
        prior_z = self.z_prior_projection(seeds.z)
        read = self.fast_read_norm(
            anchor_read + memory_read + strategy_read + phase_z + prior_z
        )
        normalized_z = self.fast_state_norm(z)
        read_gate = torch.sigmoid(
            self.read_gate(torch.cat((normalized_z, read), dim=-1))
        )
        integrated = z + read_gate * self.read_projection(read)
        attention_delta = self.fast_attention(integrated, integrated)
        fast_mid = integrated + attention_delta
        ffn_delta = self.fast_ffn(self.fast_ffn_norm(fast_mid))
        delta_z = attention_delta + ffn_delta
        normalized_delta_z = self.fast_delta_norm(delta_z)
        update_gate_z = torch.sigmoid(
            self.fast_update_gate(
                torch.cat((normalized_z, normalized_delta_z, phase_z), dim=-1)
            )
        )
        candidate_z = self.fast_exit_norm(z + self.fast_scale.to(z.dtype) * delta_z)
        next_z = z + update_gate_z * (candidate_z - z)

        memory_proposal = self.memory_writer(m, next_z)
        phase_m = phase.unsqueeze(2).expand_as(m)
        strategy_for_memory = h.mean(dim=2, keepdim=True).expand_as(m)
        memory_features = self.memory_state_norm(
            m
            + memory_proposal
            + strategy_for_memory
            + phase_m
            + self.m_prior_projection(seeds.m)
        )
        erase = torch.sigmoid(self.memory_erase(memory_features))
        add = torch.tanh(self.memory_add(memory_features))
        write_gate = torch.sigmoid(self.memory_write_gate(memory_features))
        candidate_m = self.memory_exit_norm((1.0 - erase) * m + add)
        next_m = m + write_gate * (candidate_m - m)

        slow_updated = (step + 1) % self.slow_interval == 0
        if slow_updated:
            slow_sources = torch.cat((next_z, next_m), dim=2)
            slow_read = self.slow_reader(h, slow_sources)
            slow_attention = self.slow_attention(h + slow_read, h + slow_read)
            slow_mid = h + slow_read + slow_attention
            slow_ffn = self.slow_ffn(self.slow_ffn_norm(slow_mid))
            delta_h = (
                slow_read + slow_attention + slow_ffn + self.h_prior_projection(seeds.h)
            )
            phase_h = phase.unsqueeze(2).expand_as(h)
            update_gate_h = torch.sigmoid(
                self.slow_update_gate(
                    torch.cat(
                        (
                            self.slow_state_norm(h),
                            self.slow_delta_norm(delta_h),
                            phase_h,
                        ),
                        dim=-1,
                    )
                )
            )
            candidate_h = self.slow_exit_norm(h + self.slow_scale.to(h.dtype) * delta_h)
            next_h = h + update_gate_h * (candidate_h - h)
        else:
            next_h = h
            update_gate_h = torch.zeros_like(h)

        if active is not None:
            next_z = torch.where(active, next_z, z)
            next_h = torch.where(active, next_h, h)
            next_m = torch.where(active, next_m, m)

        z_update = _rms(next_z - z, (-2, -1)).squeeze(-2)
        h_update = _rms(next_h - h, (-2, -1)).squeeze(-2)
        m_update = _rms(next_m - m, (-2, -1)).squeeze(-2)
        joint_residual = (z_update + h_update + m_update) / 3.0
        metrics = {
            "z": z_update,
            "h": h_update,
            "m": m_update,
            "memory_read": _rms(memory_read, (-2, -1)).squeeze(-2),
            "anchor_read": _rms(anchor_read, (-2, -1)).squeeze(-2),
            "memory_write": _rms(next_m - m, (-2, -1)).squeeze(-2),
            "memory_utilization": write_gate.float()
            .mean(dim=(-2, -1), keepdim=False)
            .unsqueeze(-1),
            "active_fraction": update_gate_z.float()
            .mean(dim=-1)
            .gt(self.active_threshold)
            .float()
            .mean(dim=-1, keepdim=True),
            "z_rms": _rms(next_z, (-2, -1)).squeeze(-2),
            "h_rms": _rms(next_h, (-2, -1)).squeeze(-2),
            "m_rms": _rms(next_m, (-2, -1)).squeeze(-2),
        }
        next_state = IRL3State(next_z, next_h, next_m)
        halt_probability = self._halt_probability(
            next_z,
            next_h,
            next_m,
            phase,
            joint_residual,
            step=step,
        )
        return next_state, halt_probability, metrics, anchor_entropy

    def forward(
        self,
        anchor: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        anchor_cache: Optional[LayerKVCache] = None,
        persistent_state: Optional[IRL3State] = None,
        compute_depth: Optional[int] = None,
        halt_mode: Optional[str] = None,
        gradient_horizon: Optional[int] = None,
        return_stats: Optional[bool] = None,
    ) -> IRL3ComputerResult:
        if anchor.dim() != 3:
            raise ValueError("anchor must have shape [batch, sequence, d_model]")
        depth = self._resolve_depth(compute_depth)
        mode = self._resolve_halt_mode(halt_mode)
        track_stats = (
            self.track_stats_by_default if return_stats is None else bool(return_stats)
        )
        if anchor_cache is not None and document_ids is not None:
            raise ValueError("Packed documents are not supported with anchor cache")

        horizon = (
            self.gradient_horizon if gradient_horizon is None else int(gradient_horizon)
        )
        if horizon <= 0:
            raise ValueError("gradient_horizon must be positive")
        horizon = min(horizon, depth)
        no_grad_steps = depth - horizon if self.training else 0

        state, seeds = self._conditioned_state(anchor, persistent_state)
        current_key, current_value = self.anchor_reader.project_anchor(anchor)
        batch_size, prefixes, _ = anchor.shape
        evidence_key = current_key
        evidence_value = current_value
        evidence_attention_mask = attention_mask
        if anchor_cache is not None:
            cached_key, cached_value, _ = anchor_cache.get()
            past_length = 0
            if cached_key is not None and cached_value is not None:
                past_length = cached_key.size(2)
                evidence_key = torch.cat(
                    (cached_key.transpose(1, 2), current_key), dim=1
                )
                evidence_value = torch.cat(
                    (cached_value.transpose(1, 2), current_value), dim=1
                )
            if attention_mask is not None and attention_mask.size(1) == prefixes:
                past_valid = torch.ones(
                    batch_size,
                    past_length,
                    device=anchor.device,
                    dtype=attention_mask.dtype,
                )
                evidence_attention_mask = torch.cat((past_valid, attention_mask), dim=1)
            anchor_cache.append(
                current_key.detach().transpose(1, 2),
                current_value.detach().transpose(1, 2),
                position_ids,
            )
        probability_shape = (batch_size, prefixes, 1, 1)
        survival = anchor.new_ones(probability_shape, dtype=torch.float32)
        cumulative = anchor.new_zeros(probability_shape, dtype=torch.float32)
        accumulated_z = torch.zeros_like(state.z)
        accumulated_h = torch.zeros_like(state.h)
        accumulated_m = torch.zeros_like(state.m)
        expected_steps = anchor.new_zeros(probability_shape, dtype=torch.float32)
        active = (
            torch.ones(probability_shape, device=anchor.device, dtype=torch.bool)
            if mode == "adaptive"
            else None
        )
        step_counts = torch.full(
            (batch_size, prefixes),
            depth,
            device=anchor.device,
            dtype=torch.long,
        )

        zero_metric = anchor.new_zeros((batch_size, prefixes, 1), dtype=torch.float32)
        phase_metrics = {
            "z": zero_metric,
            "h": zero_metric,
            "m": zero_metric,
            "memory_read": zero_metric,
        }
        previous_joint: Optional[torch.Tensor] = None
        previous_state = state
        halt_probabilities: List[torch.Tensor] = []
        halt_weights: List[torch.Tensor] = []
        active_fractions: List[torch.Tensor] = []
        anchor_read_rms: List[torch.Tensor] = []
        anchor_entropies: List[torch.Tensor] = []
        memory_read_rms: List[torch.Tensor] = []
        memory_write_rms: List[torch.Tensor] = []
        memory_utilization: List[torch.Tensor] = []
        z_update_rms: List[torch.Tensor] = []
        h_update_rms: List[torch.Tensor] = []
        m_update_rms: List[torch.Tensor] = []
        z_rms: List[torch.Tensor] = []
        h_rms: List[torch.Tensor] = []
        m_rms: List[torch.Tensor] = []
        joint_residuals: List[torch.Tensor] = []
        contraction_ratios: List[torch.Tensor] = []
        iterations_used = 0

        for step in range(depth):
            context = torch.no_grad() if step < no_grad_steps else nullcontext()
            with context:
                phase = self.phase_encoder(
                    step=step,
                    budget=depth,
                    metrics=phase_metrics,
                    survival=survival,
                )
                previous_state = state
                state, halt_probability, metrics, anchor_entropy = self._step(
                    state,
                    seeds,
                    phase=phase,
                    current_key=evidence_key,
                    current_value=evidence_value,
                    attention_mask=evidence_attention_mask,
                    document_ids=document_ids,
                    step=step,
                    active=active,
                    track_stats=track_stats,
                )

            phase_metrics = {
                "z": metrics["z"],
                "h": metrics["h"],
                "m": metrics["m"],
                "memory_read": metrics["memory_read"],
            }
            iterations_used += 1
            hazard = halt_probability.float()
            if active is not None:
                hazard = torch.where(active, hazard, torch.zeros_like(hazard))
            is_last = step + 1 == depth

            if mode == "adaptive" and not is_last:
                proposed_mass = cumulative + survival * hazard
                newly_halted = active & proposed_mass.ge(self.halt_threshold)
                normal_weight = survival * hazard
                weight = torch.where(newly_halted, survival, normal_weight)
                expected_steps = expected_steps + weight * float(step + 1)
                cumulative = cumulative + weight
                survival = torch.where(
                    newly_halted,
                    torch.zeros_like(survival),
                    survival * (1.0 - hazard),
                )
                newly_halted_2d = newly_halted.squeeze(-1).squeeze(-1)
                step_counts = torch.where(
                    newly_halted_2d,
                    torch.full_like(step_counts, step + 1),
                    step_counts,
                )
                active = active & ~newly_halted
            else:
                weight = survival if is_last else survival * hazard
                expected_steps = expected_steps + weight * float(step + 1)
                cumulative = cumulative + weight
                survival = (
                    torch.zeros_like(survival)
                    if is_last
                    else (survival * (1.0 - hazard))
                )

            accumulated_z = accumulated_z + weight.to(state.z.dtype) * state.z
            accumulated_h = accumulated_h + weight.to(state.h.dtype) * state.h
            accumulated_m = accumulated_m + weight.to(state.m.dtype) * state.m

            halt_probabilities.append(hazard.mean())
            halt_weights.append(weight.mean())
            if track_stats:
                active_fractions.append(metrics["active_fraction"].mean())
                anchor_read_rms.append(metrics["anchor_read"].mean())
                anchor_entropies.append(anchor_entropy)
                memory_read_rms.append(metrics["memory_read"].mean())
                memory_write_rms.append(metrics["memory_write"].mean())
                memory_utilization.append(metrics["memory_utilization"].mean())
                z_update_rms.append(metrics["z"].mean())
                h_update_rms.append(metrics["h"].mean())
                m_update_rms.append(metrics["m"].mean())
                z_rms.append(metrics["z_rms"].mean())
                h_rms.append(metrics["h_rms"].mean())
                m_rms.append(metrics["m_rms"].mean())
                joint = (metrics["z"] + metrics["h"] + metrics["m"]) / 3.0
                joint_mean = joint.mean()
                joint_residuals.append(joint_mean)
                if previous_joint is not None:
                    contraction_ratios.append(
                        joint_mean / previous_joint.clamp_min(1e-12)
                    )
                previous_joint = joint_mean

            if active is not None and not bool(active.any()):
                break

        if mode == "fixed":
            final_state = state
        else:
            # Any numerical remainder is assigned to the last available state.
            if bool(survival.gt(0).any()):
                accumulated_z = accumulated_z + survival.to(state.z.dtype) * state.z
                accumulated_h = accumulated_h + survival.to(state.h.dtype) * state.h
                accumulated_m = accumulated_m + survival.to(state.m.dtype) * state.m
                expected_steps = expected_steps + survival * float(iterations_used)
            final_state = IRL3State(accumulated_z, accumulated_h, accumulated_m)

        z_denominator = final_state.z.float().square().mean().clamp_min(1e-6)
        h_denominator = final_state.h.float().square().mean().clamp_min(1e-6)
        m_denominator = final_state.m.float().square().mean().clamp_min(1e-6)
        equilibrium_loss = (
            (state.z - previous_state.z).float().square().mean() / z_denominator
            + (state.h - previous_state.h).float().square().mean() / h_denominator
            + self.memory_equilibrium_weight
            * (state.m - previous_state.m).float().square().mean()
            / m_denominator
        )
        ponder_loss = expected_steps.mean()

        write_query = self.write_query(self.write_query_norm(anchor)).unsqueeze(2)
        solved_slots = torch.cat((final_state.z, final_state.h, final_state.m), dim=2)
        solution = self.write_attention(write_query, solved_slots).squeeze(2)
        gate_input = torch.cat((write_query.squeeze(2), solution), dim=-1)
        write_gate = torch.sigmoid(self.write_gate(gate_input))
        correction = (
            self.write_output(solution)
            * write_gate
            * self.writeback_scale.to(solution.dtype)
        )

        diagnostics: Optional[IRL3Diagnostics] = None
        if track_stats:
            empty = anchor.new_empty(0, dtype=torch.float32)

            def stack(values: List[torch.Tensor]) -> torch.Tensor:
                return torch.stack(values) if values else empty

            diagnostics = IRL3Diagnostics(
                halt_probabilities=stack(halt_probabilities),
                halt_weights=stack(halt_weights),
                active_latent_fraction=stack(active_fractions),
                anchor_read_rms=stack(anchor_read_rms),
                anchor_attention_entropy=stack(anchor_entropies),
                memory_read_rms=stack(memory_read_rms),
                memory_write_rms=stack(memory_write_rms),
                memory_utilization=stack(memory_utilization),
                z_update_rms=stack(z_update_rms),
                h_update_rms=stack(h_update_rms),
                m_update_rms=stack(m_update_rms),
                z_rms=stack(z_rms),
                h_rms=stack(h_rms),
                m_rms=stack(m_rms),
                joint_residual=stack(joint_residuals),
                contraction_ratios=stack(contraction_ratios),
                writeback_rms=correction.float().square().mean().sqrt(),
                expected_steps=expected_steps.squeeze(-1).squeeze(-1),
                step_counts=step_counts,
                iterations=iterations_used,
                converged_fraction=float(
                    step_counts.lt(depth).float().mean().detach().cpu()
                ),
                halt_mode=mode,
            )

        return IRL3ComputerResult(
            correction=correction,
            state=final_state,
            ponder_loss=ponder_loss,
            equilibrium_loss=equilibrium_loss,
            diagnostics=diagnostics,
        )
