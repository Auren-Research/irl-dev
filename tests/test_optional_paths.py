"""Tests for optional stability and memory paths."""

from __future__ import annotations

import pytest
import torch

from lunaris_dense_irl2 import LunarisDenseConfig, LunarisDenseForCausalLM


@pytest.mark.parametrize("policy", ["block", "irl"])
def test_gradient_checkpointing_preserves_gradients(policy: str) -> None:
    torch.manual_seed(8)
    config = LunarisDenseConfig.debug(
        vocab_size=193,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=96,
        max_seq_len=32,
        local_window=4,
        use_gradient_checkpointing=True,
        gradient_checkpointing_policy=policy,
        sandwich_norm=True,
        attention_output_gate=True,
    )
    model = LunarisDenseForCausalLM(config).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_global_halting_stops_the_recurrence() -> None:
    torch.manual_seed(9)
    config = LunarisDenseConfig.debug(
        vocab_size=193,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=96,
        max_seq_len=32,
        local_window=4,
        irl_eval_iters=6,
        irl_halt_tol=1e6,
        track_irl_stats=True,
    )
    model = LunarisDenseForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        output = model(input_ids)

    assert output.irl_diagnostics is not None
    assert all(item.iterations == 1 for item in output.irl_diagnostics)
    assert all(item.converged for item in output.irl_diagnostics)
