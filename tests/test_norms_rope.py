"""Numerical equivalence tests for normalization and rotary embeddings."""

from __future__ import annotations

import torch

from lunaris_dense_irl2.norms import RMSNorm
from lunaris_dense_irl2.rope import RotaryEmbedding, apply_rotary_embedding


def _reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = x.dtype
    x_float = x.float()
    inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    normalized = x_float * inverse_rms
    return (normalized * weight.float()).to(input_dtype)


def _reference_rope_values(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    scaling_factor: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = position_ids.float() / scaling_factor
    phase = torch.einsum("bs,d->bsd", positions, frequencies)
    phase = torch.repeat_interleave(phase, repeats=2, dim=-1)
    return phase.cos().unsqueeze(1).to(dtype), phase.sin().unsqueeze(1).to(dtype)


def _reference_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def test_rms_norm_matches_reference_outputs_and_gradients() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        torch.manual_seed(17)
        x_actual = torch.randn(2, 3, 16, dtype=dtype, requires_grad=True)
        x_expected = x_actual.detach().clone().requires_grad_(True)
        norm = RMSNorm(16, eps=1e-6)
        weight_expected = norm.weight.detach().clone().requires_grad_(True)

        actual = norm(x_actual)
        expected = _reference_rms_norm(x_expected, weight_expected, norm.eps)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(
            x_actual.grad,
            x_expected.grad,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            norm.weight.grad,
            weight_expected.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_compact_rope_matches_expanded_reference() -> None:
    head_dim = 16
    theta = 1_000_000.0
    scaling_factor = 2.0
    rope = RotaryEmbedding(head_dim, theta, scaling_factor=scaling_factor)
    position_ids = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])

    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        compact_cos, compact_sin = rope(position_ids, dtype=dtype)
        expanded_cos, expanded_sin = _reference_rope_values(
            position_ids,
            head_dim=head_dim,
            theta=theta,
            scaling_factor=scaling_factor,
            dtype=dtype,
        )
        torch.testing.assert_close(
            compact_cos.repeat_interleave(2, dim=-1),
            expanded_cos,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            compact_sin.repeat_interleave(2, dim=-1),
            expanded_sin,
            rtol=0.0,
            atol=0.0,
        )


def test_compact_rotary_application_matches_reference() -> None:
    torch.manual_seed(23)
    rope = RotaryEmbedding(16, 10_000.0)
    positions = torch.arange(7).unsqueeze(0).expand(2, -1)
    compact_cos, compact_sin = rope(positions, dtype=torch.float32)
    expanded_cos = compact_cos.repeat_interleave(2, dim=-1)
    expanded_sin = compact_sin.repeat_interleave(2, dim=-1)
    query = torch.randn(2, 4, 7, 16)
    key = torch.randn(2, 2, 7, 16)

    actual_query, actual_key = apply_rotary_embedding(
        query,
        key,
        compact_cos,
        compact_sin,
    )
    expected_query = _reference_apply_rope(
        query,
        expanded_cos,
        expanded_sin,
    )
    expected_key = _reference_apply_rope(
        key,
        expanded_cos,
        expanded_sin,
    )
    torch.testing.assert_close(actual_query, expected_query, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_key, expected_key, rtol=0.0, atol=0.0)
