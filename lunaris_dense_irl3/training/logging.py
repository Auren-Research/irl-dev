"""IRL-3 cost estimates plus the shared structured metric logger."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from lunaris_dense_irl2.training.logging import (
    MetricLogger,
    memory_metrics,
    perplexity,
)

from ..config import LunarisIRL3Config


def _linear_flops(rows: int, input_dim: int, output_dim: int) -> float:
    """Count a dense multiply-add as two FLOPs for each output element."""

    return 2.0 * rows * input_dim * output_dim


def _state_attention_flops(
    query_slots: int,
    key_value_slots: int,
    state_dim: int,
) -> float:
    """Count Q/K/V/O projections plus QK and probability-value products."""

    projections = (
        _linear_flops(query_slots, state_dim, state_dim)
        + 2.0 * _linear_flops(key_value_slots, state_dim, state_dim)
        + _linear_flops(query_slots, state_dim, state_dim)
    )
    mixing = 4.0 * query_slots * key_value_slots * state_dim
    return projections + mixing


def _recurrent_flops_per_step(
    model: LunarisIRL3Config,
    anchor_window: int,
) -> Tuple[float, float]:
    """Return always-executed and slow-clock FLOPs for one causal prefix."""

    s = model.state_dim
    f = model.state_ffn_dim
    z = model.irl3_num_latents
    m = model.irl3_memory_slots
    h = model.irl3_slow_slots

    phase_hidden = max(32, s // 2)
    phase_encoder = _linear_flops(1, 12, phase_hidden) + _linear_flops(
        1, phase_hidden, s
    )

    anchor_reader = (
        2.0 * _linear_flops(z, s, s)
        + 4.0 * z * anchor_window * s
    )
    state_reads_and_think = (
        _state_attention_flops(z, m, s)
        + _state_attention_flops(z, h, s)
        + _state_attention_flops(z, z, s)
        + _state_attention_flops(m, z, s)
    )
    fast_dense = (
        _linear_flops(z, s, s)  # seed prior
        + _linear_flops(z, s, s)  # read projection
        + _linear_flops(z, 2 * s, s)  # read gate
        + 6.0 * z * s * f  # SwiGLU up/gate/down
        + _linear_flops(z, 3 * s, s)  # update gate
    )
    memory_dense = (
        3.0 * _linear_flops(m, s, s)  # prior, erase, and add
        + _linear_flops(m, s, 1)  # write gate
    )
    halt_hidden = max(32, s // 2)
    halt = _linear_flops(1, 4 * s + 1, halt_hidden) + _linear_flops(
        1, halt_hidden, 1
    )
    fast_step = (
        phase_encoder
        + anchor_reader
        + state_reads_and_think
        + fast_dense
        + memory_dense
        + halt
    )

    slow_attention = _state_attention_flops(h, z + m, s) + (
        _state_attention_flops(h, h, s)
    )
    slow_dense = (
        6.0 * h * s * f
        + _linear_flops(h, s, s)  # seed prior
        + _linear_flops(h, 3 * s, s)  # update gate
    )
    return fast_step, slow_attention + slow_dense


def estimate_forward_flops_per_token(
    model: LunarisIRL3Config,
    sequence_length: int,
    *,
    recurrent_depth: Optional[int] = None,
    mtp_active_heads: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate dense forward FLOPs per observed input token.

    The estimate follows the implemented tensor rows. It counts dense linear
    multiply-adds, attention QK products, and attention probability-value
    products. It intentionally omits embedding lookups, normalization,
    activation functions, softmax/masking, reductions, and elementwise gates.
    Those operations and kernel launch effects matter to latency, especially
    for small state arrays, so qualification wall time remains authoritative.
    """

    sequence_length = int(sequence_length)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    depth = model.irl3_train_steps if recurrent_depth is None else int(recurrent_depth)
    if depth <= 0:
        raise ValueError("recurrent_depth must be positive")
    active_heads = (
        model.mtp_num_heads if mtp_active_heads is None else int(mtp_active_heads)
    )
    if not (0 <= active_heads <= model.mtp_num_heads):
        raise ValueError("mtp_active_heads must be in [0, model.mtp_num_heads]")

    d = model.d_model
    s = model.state_dim
    kv = model.kv_dim
    transformer = 0.0
    for layer_index in range(model.n_layers):
        visible = (
            min(sequence_length, model.local_window)
            if model.attention_type(layer_index) == "local"
            else sequence_length
        )
        projections = 2.0 * d * (2 * kv + 2 * d)
        ffn = 6.0 * d * model.ffn_hidden_dim
        mixing = 4.0 * visible * d
        transformer += projections + ffn + mixing

    state_initialization = 3.0 * _linear_flops(1, d, s)
    anchor_kv_projection = 2.0 * _linear_flops(1, d, s)
    anchor_window = min(sequence_length, model.irl3_anchor_window)
    fast_step, slow_tick = _recurrent_flops_per_step(model, anchor_window)
    recurrent_fast = depth * fast_step
    recurrent_slow = (depth // model.irl3_slow_update_interval) * slow_tick

    solved_slots = (
        model.irl3_num_latents
        + model.irl3_memory_slots
        + model.irl3_slow_slots
    )
    writeback = (
        _linear_flops(1, d, s)
        + _state_attention_flops(1, solved_slots, s)
        + _linear_flops(1, s, d)
        + _linear_flops(1, 2 * s, d)
    )

    next_token_fraction = max(0, sequence_length - 1) / sequence_length
    lm_head = next_token_fraction * _linear_flops(1, d, model.vocab_size)
    mtp = 0.0
    if active_heads > 0:
        mtp_hidden = model.mtp_ffn_hidden_dim or d
        mtp_dense = (
            4.0 * d * d  # hidden and teacher projections
            + 6.0 * d * mtp_hidden  # SwiGLU
            + 4.0 * d * d  # 2d -> d update gate
            + 2.0 * d * model.vocab_size
        )
        for horizon_index in range(active_heads):
            horizon = horizon_index + 2
            active_fraction = max(0, sequence_length - horizon) / sequence_length
            mtp += active_fraction * mtp_dense

    components = {
        "transformer": transformer,
        "state_initialization": state_initialization,
        "anchor_kv_projection": anchor_kv_projection,
        "recurrent_fast": recurrent_fast,
        "recurrent_slow": recurrent_slow,
        "writeback": writeback,
        "lm_head": lm_head,
        "mtp": mtp,
    }
    components["total"] = sum(components.values())
    return components


def estimate_training_flops_breakdown(
    model: LunarisIRL3Config,
    sequence_length: int,
    *,
    recurrent_depth: Optional[int] = None,
    gradient_horizon: Optional[int] = None,
    mtp_active_heads: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate executed forward/backward FLOPs per observed training token.

    Backward is approximated as twice the corresponding differentiable dense
    forward work. Only the configured recurrent credit horizon receives that
    multiplier. If gradient checkpointing is active, the selected region's
    extra forward recomputation is reported separately.
    """

    depth = model.irl3_train_steps if recurrent_depth is None else int(recurrent_depth)
    horizon = (
        model.irl3_gradient_horizon
        if gradient_horizon is None
        else int(gradient_horizon)
    )
    if depth <= 0 or horizon <= 0:
        raise ValueError("recurrent_depth and gradient_horizon must be positive")
    horizon = min(depth, horizon)
    forward = estimate_forward_flops_per_token(
        model,
        sequence_length,
        recurrent_depth=depth,
        mtp_active_heads=mtp_active_heads,
    )

    anchor_window = min(int(sequence_length), model.irl3_anchor_window)
    fast_step, slow_tick = _recurrent_flops_per_step(model, anchor_window)
    interval = model.irl3_slow_update_interval
    differentiable_slow_ticks = depth // interval - (depth - horizon) // interval
    differentiable_forward = (
        forward["total"]
        - forward["recurrent_fast"]
        - forward["recurrent_slow"]
        + horizon * fast_step
        + differentiable_slow_ticks * slow_tick
    )
    backward_estimate = 2.0 * differentiable_forward

    checkpoint_recompute = 0.0
    if model.use_gradient_checkpointing:
        if model.gradient_checkpointing_policy == "computer":
            checkpoint_recompute = (
                forward["state_initialization"]
                + forward["anchor_kv_projection"]
                + forward["recurrent_fast"]
                + forward["recurrent_slow"]
                + forward["writeback"]
            )
        elif model.gradient_checkpointing_policy == "block":
            checkpoint_recompute = forward["transformer"]

    result = {
        **{
            f"forward_{name}": value
            for name, value in forward.items()
            if name != "total"
        },
        "forward_total": forward["total"],
        "differentiable_forward_equivalent": differentiable_forward,
        "backward_estimate": backward_estimate,
        "checkpoint_recompute": checkpoint_recompute,
    }
    result["training_total"] = (
        result["forward_total"]
        + result["backward_estimate"]
        + result["checkpoint_recompute"]
    )
    return result


def estimate_training_flops_per_token(
    model: LunarisIRL3Config,
    sequence_length: int,
    *,
    recurrent_depth: Optional[int] = None,
    gradient_horizon: Optional[int] = None,
    mtp_active_heads: Optional[int] = None,
) -> float:
    """Return the total from :func:`estimate_training_flops_breakdown`."""

    return estimate_training_flops_breakdown(
        model,
        sequence_length,
        recurrent_depth=recurrent_depth,
        gradient_horizon=gradient_horizon,
        mtp_active_heads=mtp_active_heads,
    )["training_total"]


__all__ = [
    "MetricLogger",
    "estimate_forward_flops_per_token",
    "estimate_training_flops_breakdown",
    "estimate_training_flops_per_token",
    "memory_metrics",
    "perplexity",
]
