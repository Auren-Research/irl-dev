"""End-to-end causal LM tests for IRL-3."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lunaris_dense_irl3 import LunarisIRL3Config, LunarisIRL3ForCausalLM


def tiny_config(**overrides: object) -> LunarisIRL3Config:
    values = {
        "vocab_size": 101,
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


def test_finite_loss_backward_and_auxiliary_objectives() -> None:
    torch.manual_seed(34)
    config = tiny_config(mtp_num_heads=2, mtp_ffn_hidden_dim=48)
    model = LunarisIRL3ForCausalLM(config).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 9))
    output = model(
        input_ids,
        labels=input_ids,
        return_logits=False,
        loss_chunk_size=7,
        loss_backend="chunked",
        z_loss_coefficient=1e-4,
        mtp_loss_weight=0.1,
        mtp_active_heads=2,
        ponder_coefficient=1e-3,
        equilibrium_coefficient=1e-3,
        halt_mode="act",
        return_irl3_stats=True,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.cross_entropy is not None and torch.isfinite(output.cross_entropy)
    assert output.z_loss is not None and torch.isfinite(output.z_loss)
    assert output.mtp_loss is not None and output.mtp_loss > 0
    assert output.mtp_losses is not None and len(output.mtp_losses) == 2
    assert output.ponder_loss is not None and output.ponder_loss >= 2
    assert output.equilibrium_loss is not None
    assert output.irl3_diagnostics is not None
    output.loss.backward()

    for prefix in (
        "computer.anchor_reader",
        "computer.fast_attention",
        "computer.memory_writer",
        "computer.slow_reader",
        "computer.halt_output",
        "computer.write_output",
        "mtp_predictor",
    ):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients, prefix
        assert any(
            gradient is not None and gradient.abs().sum() > 0 for gradient in gradients
        ), prefix


def test_no_future_token_leakage() -> None:
    torch.manual_seed(35)
    model = LunarisIRL3ForCausalLM(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 70, 71, 72]])

    with torch.no_grad():
        logits_first = model(first, compute_depth=4).logits
        logits_second = model(second, compute_depth=4).logits

    assert logits_first is not None and logits_second is not None
    torch.testing.assert_close(
        logits_first[:, :3], logits_second[:, :3], rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(logits_first[:, 3:], logits_second[:, 3:])


def test_packed_documents_isolate_anchor_and_transformer_attention() -> None:
    torch.manual_seed(36)
    model = LunarisIRL3ForCausalLM(
        tiny_config(attention_pattern="G", irl3_anchor_window=16)
    ).eval()
    first = torch.tensor([[1, 2, 10, 11]])
    second = torch.tensor([[8, 9, 10, 11]])
    positions = torch.tensor([[0, 1, 0, 1]])
    documents = torch.tensor([[0, 0, 1, 1]])

    with torch.no_grad():
        logits_first = model(
            first, position_ids=positions, document_ids=documents
        ).logits
        logits_second = model(
            second, position_ids=positions, document_ids=documents
        ).logits

    assert logits_first is not None and logits_second is not None
    torch.testing.assert_close(
        logits_first[:, 2:], logits_second[:, 2:], rtol=1e-5, atol=1e-6
    )


def test_cached_decoding_matches_full_forward() -> None:
    torch.manual_seed(37)
    config = tiny_config()
    model = LunarisIRL3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full = model(input_ids, compute_depth=3).logits
        cache = model.init_cache(batch_size=1)
        pieces = []
        for position in range(input_ids.size(1)):
            piece = model(
                input_ids[:, position : position + 1],
                cache=cache,
                compute_depth=3,
                logits_to_keep=1,
            ).logits
            pieces.append(piece)
        incremental = torch.cat(pieces, dim=1)

    assert full is not None
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-5)
    assert cache.seen_tokens == input_ids.size(1)
    assert cache.anchor.length == config.irl3_anchor_window


def test_depth_override_changes_computation_without_parameters() -> None:
    torch.manual_seed(38)
    config = tiny_config()
    model = LunarisIRL3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 7))
    count = model.get_num_parameters()
    with torch.no_grad():
        shallow = model(input_ids, compute_depth=2).logits
        deep = model(input_ids, compute_depth=7).logits
    assert count == model.get_num_parameters()
    assert shallow is not None and deep is not None
    assert not torch.allclose(shallow, deep)


def test_persistent_generation_state_and_generation_path() -> None:
    torch.manual_seed(39)
    config = tiny_config(irl3_persist_state=True)
    model = LunarisIRL3ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (2, 4))
    cache = model.init_cache(batch_size=2)

    with torch.no_grad():
        output = model(
            prompt,
            cache=cache,
            compute_depth=2,
            persist_reasoning_state=True,
            return_reasoning_state=True,
        )
    assert output.reasoning_state is not None
    assert cache.reasoning_state is not None
    assert output.reasoning_state.z.shape == (
        2,
        config.irl3_num_latents,
        config.state_dim,
    )

    generated = model.generate(
        prompt,
        max_new_tokens=3,
        temperature=0.0,
        compute_depth=2,
        persist_reasoning_state=True,
    )
    assert generated.shape == (2, 7)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(40)
    config = tiny_config()
    model = LunarisIRL3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        expected = model(input_ids, compute_depth=3).logits
    model.save_pretrained(tmp_path)
    restored = LunarisIRL3ForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids, compute_depth=3).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("policy", ["block", "computer"])
def test_gradient_checkpointing(policy: str) -> None:
    torch.manual_seed(41)
    config = tiny_config(
        use_gradient_checkpointing=True,
        gradient_checkpointing_policy=policy,
    )
    model = LunarisIRL3ForCausalLM(config).train()
    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    output = model(input_ids, labels=input_ids, halt_mode="act")
    assert output.loss is not None
    output.loss.backward()
    assert model.computer.write_output.weight.grad is not None
    assert model.blocks[0].attention.o_proj.weight.grad is not None


def test_bfloat16_forward_backward_when_supported() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    torch.manual_seed(42)
    config = tiny_config()
    model = LunarisIRL3ForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (1, 6), device=device)
    try:
        output = model(input_ids, labels=input_ids, halt_mode="act")
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
    except RuntimeError as exc:
        if device.type == "cpu" and "bfloat16" in str(exc).lower():
            pytest.skip(f"CPU build lacks BF16 support: {exc}")
        raise
    assert model.computer.fast_ffn.up_proj.weight.grad is not None
