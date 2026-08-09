"""IRL-3 cost estimates plus the shared structured metric logger."""

from __future__ import annotations

from lunaris_dense_irl2.training.logging import (
    MetricLogger,
    memory_metrics,
    perplexity,
)

from ..config import LunarisIRL3Config


def estimate_training_flops_per_token(
    model: LunarisIRL3Config,
    sequence_length: int,
) -> float:
    """Estimate forward-plus-backward FLOPs at configured train depth.

    This deliberately remains an analytic estimate. Qualification also records
    measured throughput because kernels and small-slot attention dominate the
    difference between nominal FLOPs and wall-clock efficiency.
    """

    d = model.d_model
    s = model.state_dim
    kv = model.kv_dim
    token_projection = 2.0 * d * (2 * kv + 2 * d)
    token_ffn = 6.0 * d * model.ffn_hidden_dim
    token_mixing = 0.0
    for layer_index in range(model.n_layers):
        visible = (
            min(sequence_length, model.local_window)
            if model.attention_type(layer_index) == "local"
            else sequence_length
        )
        token_mixing += 4.0 * visible * d
    transformer_forward = model.n_layers * (token_projection + token_ffn) + token_mixing

    anchor_projection = 4.0 * d * s
    state_attention_projection = 8.0 * s * s
    recurrent_dense = (
        4.0 * state_attention_projection + 6.0 * s * model.state_ffn_dim + 18.0 * s * s
    )
    recurrent_slow = (
        2.0 * state_attention_projection + 6.0 * s * model.state_ffn_dim + 8.0 * s * s
    ) / model.irl3_slow_update_interval
    anchor_mixing = (
        4.0
        * min(sequence_length, model.irl3_anchor_window)
        * model.irl3_num_latents
        * s
    )
    recurrent_forward = anchor_projection + model.irl3_train_steps * (
        recurrent_dense + recurrent_slow + anchor_mixing
    )
    writeback_forward = 8.0 * s * s + 2.0 * s * d
    output_forward = 2.0 * d * model.vocab_size
    mtp_forward = 0.0
    if model.mtp_num_heads > 0:
        mtp_hidden = model.mtp_ffn_hidden_dim or model.d_model
        mtp_forward = model.mtp_num_heads * (
            6.0 * d * d + 6.0 * d * mtp_hidden + 2.0 * d * model.vocab_size
        )
    return 3.0 * (
        transformer_forward
        + recurrent_forward
        + writeback_forward
        + output_forward
        + mtp_forward
    )


__all__ = [
    "MetricLogger",
    "estimate_training_flops_per_token",
    "memory_metrics",
    "perplexity",
]
