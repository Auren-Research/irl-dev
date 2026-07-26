"""IRL-2 recurrence and gradient tests."""

from __future__ import annotations

import torch

from lunaris_dense_irl2 import LunarisDenseConfig
from lunaris_dense_irl2.irl2 import IRL2Cell


def test_irl_reuses_one_parameter_set_for_any_depth() -> None:
    torch.manual_seed(1)
    config = LunarisDenseConfig.debug(
        d_model=64, n_heads=4, n_kv_heads=2, ffn_hidden_dim=96
    )
    cell = IRL2Cell(config)
    anchor = torch.randn(2, 5, config.d_model)

    parameter_ids_before = {id(parameter) for parameter in cell.parameters()}
    output_t1, _, _ = cell(anchor, n_iters=1)
    output_t4, diagnostics, _ = cell(anchor, n_iters=4, return_stats=True)
    parameter_ids_after = {id(parameter) for parameter in cell.parameters()}

    assert parameter_ids_before == parameter_ids_after
    assert output_t1.shape == output_t4.shape == anchor.shape
    assert not torch.allclose(output_t1, output_t4)
    assert diagnostics is not None
    assert diagnostics.residual_norms.shape == (4,)
    assert diagnostics.contraction_ratios.shape == (3,)


def test_irl_full_backprop_reaches_every_parameter() -> None:
    torch.manual_seed(2)
    config = LunarisDenseConfig.debug(
        d_model=64, n_heads=4, n_kv_heads=2, ffn_hidden_dim=96
    )
    cell = IRL2Cell(config)
    anchor = torch.randn(2, 4, config.d_model, requires_grad=True)

    output, _, consistency = cell(
        anchor,
        n_iters=3,
        return_consistency=True,
    )
    assert consistency.isfinite()
    assert consistency.item() >= 0.0
    output.square().mean().backward()

    assert anchor.grad is not None
    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None, name
