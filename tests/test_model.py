"""End-to-end model tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lunaris_dense_irl2 import LunarisDenseConfig, LunarisDenseForCausalLM
from lunaris_dense_irl2.attention import flash_attention_available
from lunaris_dense_irl2.irl2 import IRL2Cell


def tiny_config(**overrides: object) -> LunarisDenseConfig:
    values = {
        "vocab_size": 127,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 96,
        "max_seq_len": 32,
        "attention_pattern": "LG",
        "local_window": 4,
        "irl_train_iters": 2,
        "irl_eval_iters": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    values.update(overrides)
    return LunarisDenseConfig(**values)


def test_shapes_loss_and_gradients() -> None:
    torch.manual_seed(3)
    config = tiny_config()
    model = LunarisDenseForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids, labels=input_ids, return_irl_stats=True)
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.irl_diagnostics is not None
    assert len(output.irl_diagnostics) == config.n_layers
    assert all(
        diagnostics.iterations == config.irl_train_iters
        for diagnostics in output.irl_diagnostics
    )

    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_recurrence_reuses_one_parameter_set_at_configured_depth() -> None:
    torch.manual_seed(1)
    config = tiny_config()
    cell = IRL2Cell(config)
    anchor = torch.randn(2, 5, config.d_model)
    parameter_ids = {id(parameter) for parameter in cell.parameters()}

    output_t1, _, _ = cell(anchor, n_iters=1)
    output_t4, diagnostics, _ = cell(anchor, n_iters=4, return_stats=True)

    assert parameter_ids == {id(parameter) for parameter in cell.parameters()}
    assert output_t1.shape == output_t4.shape == anchor.shape
    assert not torch.allclose(output_t1, output_t4)
    assert diagnostics is not None
    assert diagnostics.residual_norms.shape == (4,)


def test_cached_logits_match_full_forward() -> None:
    torch.manual_seed(4)
    config = tiny_config()
    model = LunarisDenseForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full_logits = model(input_ids).logits
        cache = model.init_cache(batch_size=1)
        cached_logits = []
        for position in range(input_ids.size(1)):
            output = model(
                input_ids[:, position : position + 1],
                cache=cache,
                logits_to_keep=1,
            )
            cached_logits.append(output.logits)
        incremental_logits = torch.cat(cached_logits, dim=1)

    torch.testing.assert_close(incremental_logits, full_logits, rtol=1e-4, atol=1e-5)


def test_packed_documents_are_isolated() -> None:
    torch.manual_seed(5)
    config = tiny_config(attention_pattern="G")
    model = LunarisDenseForCausalLM(config).eval()

    first = torch.tensor([[1, 2, 10, 11]])
    second = torch.tensor([[8, 9, 10, 11]])
    position_ids = torch.tensor([[0, 1, 0, 1]])
    document_ids = torch.tensor([[0, 0, 1, 1]])

    with torch.no_grad():
        logits_a = model(
            first,
            position_ids=position_ids,
            document_ids=document_ids,
        ).logits
        logits_b = model(
            second,
            position_ids=position_ids,
            document_ids=document_ids,
        ).logits

    torch.testing.assert_close(logits_a[:, 2:], logits_b[:, 2:], rtol=1e-5, atol=1e-6)


def test_no_future_leakage() -> None:
    torch.manual_seed(11)
    model = LunarisDenseForCausalLM(tiny_config(attention_pattern="G")).eval()
    first = torch.tensor([[5, 7, 11, 13, 17, 19]])
    second = first.clone()
    second[:, -1] = 23

    with torch.no_grad():
        logits_a = model(first).logits
        logits_b = model(second).logits

    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1], rtol=0.0, atol=1e-6)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(6)
    config = tiny_config()
    model = LunarisDenseForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        expected = model(input_ids).logits
        repeated = model(input_ids).logits
    torch.testing.assert_close(repeated, expected, rtol=0.0, atol=0.0)
    model.save_pretrained(tmp_path)
    restored = LunarisDenseForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids).logits

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_version_one_projection_weights_load_strictly() -> None:
    """Separate projection checkpoints migrate to fused production weights."""

    torch.manual_seed(61)
    model = LunarisDenseForCausalLM(tiny_config())
    expected = model.state_dict()
    legacy = {name: value.clone() for name, value in expected.items()}
    for layer_index in range(model.config.n_layers):
        attention_prefix = f"blocks.{layer_index}.attention."
        qkv = legacy.pop(attention_prefix + "qkv_proj.weight")
        q_width = model.config.n_heads * model.config.head_dim
        kv_width = model.config.kv_dim
        query, key, value = qkv.split((q_width, kv_width, kv_width), dim=0)
        legacy[attention_prefix + "q_proj.weight"] = query
        legacy[attention_prefix + "k_proj.weight"] = key
        legacy[attention_prefix + "v_proj.weight"] = value

        ffn_prefix = f"blocks.{layer_index}.irl.ffn."
        gate_up = legacy.pop(ffn_prefix + "gate_up_proj.weight")
        gate, up = gate_up.chunk(2, dim=0)
        legacy[ffn_prefix + "gate_proj.weight"] = gate
        legacy[ffn_prefix + "up_proj.weight"] = up

    restored = LunarisDenseForCausalLM(tiny_config())
    restored.load_state_dict(legacy, strict=True)
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0.0, atol=0.0)


def test_generation_uses_cache() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = LunarisDenseForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (2, 4))

    generated = model.generate(
        prompt,
        max_new_tokens=3,
        temperature=0.0,
    )
    assert generated.shape == (2, 7)


def test_bf16_forward_backward() -> None:
    torch.manual_seed(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tiny_config()
    model = LunarisDenseForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (1, 7), device=device)
    try:
        output = model(input_ids, labels=input_ids, return_logits=False)
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
    except RuntimeError as exc:
        if device.type == "cpu" and "BFloat16" in str(exc):
            pytest.skip(f"This CPU build lacks BF16 support: {exc}")
        raise
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(
    not torch.cuda.is_available() or not flash_attention_available(),
    reason="FlashAttention accelerator backend is unavailable",
)
def test_flash_local_attention_matches_sdpa_numerically() -> None:
    """The production local window agrees with the dense SDPA reference."""

    torch.manual_seed(120)
    flash_model = LunarisDenseForCausalLM(
        tiny_config(
            attention_pattern="L",
            attention_backend="flash",
            local_window=4,
        )
    ).cuda()
    sdpa_model = LunarisDenseForCausalLM(
        tiny_config(
            attention_pattern="L",
            attention_backend="sdpa",
            local_window=4,
        )
    ).cuda()
    sdpa_model.load_state_dict(flash_model.state_dict())
    tokens = torch.randint(0, flash_model.config.vocab_size, (2, 16), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        flash_output = flash_model(tokens, labels=tokens, return_logits=False)
        sdpa_output = sdpa_model(tokens, labels=tokens, return_logits=False)
    assert flash_output.loss is not None and sdpa_output.loss is not None
    torch.testing.assert_close(
        flash_output.loss,
        sdpa_output.loss,
        rtol=2e-3,
        atol=2e-3,
    )

    flash_output.loss.backward()
    sdpa_output.loss.backward()
    flash_gradient = flash_model.blocks[0].attention.qkv_proj.weight.grad
    sdpa_gradient = sdpa_model.blocks[0].attention.qkv_proj.weight.grad
    assert flash_gradient is not None and sdpa_gradient is not None
    cosine = torch.nn.functional.cosine_similarity(
        flash_gradient.flatten().float(),
        sdpa_gradient.flatten().float(),
        dim=0,
    )
    assert float(cosine) > 0.999


@pytest.mark.parametrize("policy", ["block", "irl"])
def test_gradient_checkpointing_preserves_gradients(policy: str) -> None:
    torch.manual_seed(8)
    config = tiny_config(
        use_gradient_checkpointing=True,
        gradient_checkpointing_policy=policy,
    )
    model = LunarisDenseForCausalLM(config).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 7))

    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    output.loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


def test_mtp_irl_objective_reaches_all_auxiliary_parameters() -> None:
    torch.manual_seed(10)
    config = tiny_config(
        mtp_num_heads=2,
        mtp_irl_iters=2,
        mtp_ffn_hidden_dim=48,
    )
    model = LunarisDenseForCausalLM(config).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))

    output = model(
        input_ids,
        labels=input_ids,
        return_logits=False,
        loss_chunk_size=7,
        mtp_loss_weight=0.2,
        mtp_active_heads=2,
        mtp_horizon_decay=0.7,
        irl_consistency_coefficient=0.01,
    )

    assert output.loss is not None
    assert output.mtp_loss is not None
    assert output.mtp_loss.item() > 0.0
    assert output.mtp_losses is not None
    assert len(output.mtp_losses) == 2
    assert output.irl_consistency_loss is not None
    assert output.irl_consistency_loss.item() >= 0.0

    output.loss.backward()
    for name, parameter in model.mtp_predictor.named_parameters():
        assert parameter.grad is not None, name


def test_packed_targets_do_not_cross_document_boundaries() -> None:
    config = tiny_config()
    model = LunarisDenseForCausalLM(config)
    labels = torch.tensor([[11, 12, 21, 22, 23]])
    document_ids = torch.tensor([[0, 0, 1, 1, 1]])

    next_token_targets = model._prepare_targets(
        labels,
        horizon=1,
        attention_mask=None,
        document_ids=document_ids,
    )
    horizon_two_targets = model._prepare_targets(
        labels,
        horizon=2,
        attention_mask=None,
        document_ids=document_ids,
    )

    assert next_token_targets.tolist() == [[12, -100, 22, 23]]
    assert horizon_two_targets.tolist() == [[-100, -100, 23]]
