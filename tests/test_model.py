"""End-to-end model tests."""

from __future__ import annotations

from pathlib import Path

import torch

from lunaris_dense_irl2 import LunarisDenseConfig, LunarisDenseForCausalLM


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
    assert output.loss is not None
    assert output.irl_diagnostics is not None
    assert len(output.irl_diagnostics) == config.n_layers

    output.loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name


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


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(6)
    config = tiny_config()
    model = LunarisDenseForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.no_grad():
        expected = model(input_ids).logits
    model.save_pretrained(tmp_path)
    restored = LunarisDenseForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(input_ids).logits

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


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
