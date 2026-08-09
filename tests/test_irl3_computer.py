"""State-transition and credit-assignment tests for the IRL-3 computer."""

from __future__ import annotations

from types import MethodType

import torch

from lunaris_dense_irl3 import IRL3State, LunarisIRL3Config
from lunaris_dense_irl3.irl3 import IRL3Computer


def tiny_config(**overrides: object) -> LunarisIRL3Config:
    values = {
        "vocab_size": 97,
        "d_model": 48,
        "n_layers": 2,
        "irl3_coda_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 72,
        "max_seq_len": 32,
        "attention_pattern": "LG",
        "local_window": 8,
        "irl3_state_dim": 32,
        "irl3_state_heads": 4,
        "irl3_state_ffn_dim": 64,
        "irl3_num_latents": 3,
        "irl3_memory_slots": 2,
        "irl3_slow_slots": 1,
        "irl3_anchor_window": 8,
        "irl3_train_steps": 4,
        "irl3_eval_steps": 4,
        "irl3_min_steps": 2,
        "irl3_max_steps": 8,
        "irl3_gradient_horizon": 4,
        "irl3_slow_update_interval": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    values.update(overrides)
    return LunarisIRL3Config(**values)


def test_shared_program_supports_multiple_compute_depths() -> None:
    torch.manual_seed(30)
    config = tiny_config()
    computer = IRL3Computer(config).eval()
    anchor = torch.randn(2, 6, config.d_model)
    position_ids = torch.arange(6).unsqueeze(0).expand(2, -1)
    parameter_ids = {id(parameter) for parameter in computer.parameters()}

    with torch.no_grad():
        depth_two = computer(
            anchor,
            position_ids=position_ids,
            compute_depth=2,
            halt_mode="fixed",
            return_stats=True,
        )
        depth_six = computer(
            anchor,
            position_ids=position_ids,
            compute_depth=6,
            halt_mode="fixed",
            return_stats=True,
        )

    assert parameter_ids == {id(parameter) for parameter in computer.parameters()}
    assert depth_two.correction.shape == depth_six.correction.shape == anchor.shape
    assert depth_two.state.z.shape == (2, 6, 3, config.state_dim)
    assert depth_two.state.h.shape == (2, 6, 1, config.state_dim)
    assert depth_two.state.m.shape == (2, 6, 2, config.state_dim)
    assert not torch.allclose(depth_two.correction, depth_six.correction)
    assert depth_six.diagnostics is not None
    assert depth_six.diagnostics.z_update_rms.shape == (6,)
    assert depth_six.diagnostics.h_update_rms[0].item() == 0.0
    assert depth_six.diagnostics.h_update_rms[1].item() > 0.0


def test_read_think_memory_strategy_and_halting_receive_gradients() -> None:
    torch.manual_seed(31)
    config = tiny_config()
    computer = IRL3Computer(config).train()
    anchor = torch.randn(2, 5, config.d_model, requires_grad=True)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    result = computer(
        anchor,
        position_ids=position_ids,
        compute_depth=4,
        halt_mode="act",
        return_stats=True,
    )
    objective = (
        result.correction.square().mean()
        + 0.01 * result.ponder_loss
        + 0.01 * result.equilibrium_loss
        + 0.001 * result.state.z.square().mean()
        + 0.001 * result.state.h.square().mean()
        + 0.001 * result.state.m.square().mean()
    )
    objective.backward()

    assert anchor.grad is not None and anchor.grad.abs().sum() > 0
    required_parameters = [
        "anchor_reader.query_proj.weight",
        "memory_reader.q_proj.weight",
        "strategy_reader.q_proj.weight",
        "fast_attention.q_proj.weight",
        "fast_ffn.up_proj.weight",
        "memory_writer.q_proj.weight",
        "memory_erase.weight",
        "memory_add.weight",
        "memory_write_gate.weight",
        "slow_reader.q_proj.weight",
        "slow_attention.q_proj.weight",
        "slow_ffn.up_proj.weight",
        "halt_output.weight",
        "write_attention.q_proj.weight",
        "write_output.weight",
    ]
    parameters = dict(computer.named_parameters())
    for name in required_parameters:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.anchor_read_rms.min() > 0
    assert diagnostics.memory_read_rms.min() > 0
    assert diagnostics.memory_write_rms.min() > 0
    assert diagnostics.memory_utilization.min() > 0
    assert diagnostics.writeback_rms > 0
    torch.testing.assert_close(
        diagnostics.halt_weights.sum(),
        torch.tensor(1.0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_persistent_state_is_fused_differentiably() -> None:
    torch.manual_seed(32)
    config = tiny_config()
    computer = IRL3Computer(config).train()
    anchor = torch.randn(2, 1, config.d_model)
    persistent = IRL3State(
        z=torch.randn(2, config.irl3_num_latents, config.state_dim, requires_grad=True),
        h=torch.randn(2, config.irl3_slow_slots, config.state_dim, requires_grad=True),
        m=torch.randn(
            2, config.irl3_memory_slots, config.state_dim, requires_grad=True
        ),
    )

    result = computer(
        anchor,
        position_ids=torch.zeros(2, 1, dtype=torch.long),
        persistent_state=persistent,
        compute_depth=2,
        halt_mode="fixed",
    )
    result.correction.square().mean().backward()

    for name in ("z", "h", "m"):
        gradient = getattr(persistent, name).grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name


def test_adaptive_halting_freezes_individual_prefixes() -> None:
    torch.manual_seed(33)
    config = tiny_config(irl3_eval_steps=7, irl3_halt_threshold=0.5)
    computer = IRL3Computer(config).eval()

    def staggered_halt(
        self: IRL3Computer,
        z: torch.Tensor,
        h: torch.Tensor,
        m: torch.Tensor,
        phase: torch.Tensor,
        joint_residual: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        del self, h, m, phase, joint_residual
        target = 2 + torch.arange(z.size(1), device=z.device).remainder(3)
        probability = (step + 1 >= target).to(z.dtype)
        return (
            probability.unsqueeze(0).expand(z.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        )

    computer._halt_probability = MethodType(  # type: ignore[method-assign]
        staggered_halt, computer
    )
    with torch.no_grad():
        result = computer(
            torch.randn(2, 5, config.d_model),
            position_ids=torch.arange(5).unsqueeze(0).expand(2, -1),
            compute_depth=7,
            halt_mode="adaptive",
            return_stats=True,
        )

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.iterations == 4
    assert diagnostics.step_counts[0].tolist() == [2, 3, 4, 2, 3]
    assert diagnostics.converged_fraction == 1.0
