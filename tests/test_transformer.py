"""Correctness and control tests for the three-model benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lunaris_dense_irl2.training.architecture_benchmark import (
    architecture_report,
)
from lunaris_dense_irl2.training.config import TrainConfig
from lunaris_dense_irl_core import LunarisIRLCoreConfig
from lunaris_dense_transformer import (
    LunarisTransformerConfig,
    LunarisTransformerForCausalLM,
)


def tiny_config(**overrides: object) -> LunarisTransformerConfig:
    """Return a tiny conventional baseline profile."""

    values = {
        "vocab_size": 127,
        "d_model": 64,
        "n_layers": 3,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 96,
        "max_seq_len": 32,
        "attention_pattern": "LG",
        "local_window": 8,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    values.update(overrides)
    return LunarisTransformerConfig(**values)


def test_transformer_forward_backward_has_no_recurrent_parameters() -> None:
    torch.manual_seed(201)
    model = LunarisTransformerForCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    output = model(
        tokens,
        labels=tokens,
        return_logits=False,
        loss_chunk_size=5,
        loss_backend="chunked",
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert model.get_recurrent_num_parameters() == 0
    assert not any("irl" in name.lower() for name, _ in model.named_parameters())


def test_transformer_cache_determinism_and_save_load(tmp_path: Path) -> None:
    torch.manual_seed(202)
    model = LunarisTransformerForCausalLM(tiny_config()).eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        expected = model(tokens).logits
        repeated = model(tokens).logits
        cache = model.init_cache(batch_size=1)
        pieces = [
            model(
                tokens[:, index : index + 1],
                cache=cache,
                logits_to_keep=1,
            ).logits
            for index in range(tokens.size(1))
        ]
    torch.testing.assert_close(expected, repeated, rtol=0.0, atol=0.0)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=1e-4, atol=1e-5)

    model.save_pretrained(tmp_path)
    restored = LunarisTransformerForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_transformer_packed_document_isolation() -> None:
    torch.manual_seed(203)
    model = LunarisTransformerForCausalLM(tiny_config(attention_pattern="G")).eval()
    first = torch.tensor([[1, 2, 10, 11]])
    second = torch.tensor([[8, 9, 10, 11]])
    position_ids = torch.tensor([[0, 1, 0, 1]])
    document_ids = torch.tensor([[0, 0, 1, 1]])
    with torch.no_grad():
        left = model(
            first,
            position_ids=position_ids,
            document_ids=document_ids,
        ).logits
        right = model(
            second,
            position_ids=position_ids,
            document_ids=document_ids,
        ).logits
    torch.testing.assert_close(left[:, 2:], right[:, 2:], rtol=1e-5, atol=1e-6)


def test_benchmark_profiles_are_parameter_matched_and_frozen() -> None:
    paths = {
        "transformer": "training_profiles/benchmark_transformer_135m_1x_mi300x.yaml",
        "irl2": "training_profiles/benchmark_irl2_135m_1x_mi300x.yaml",
        "irl_core": "training_profiles/benchmark_irl_core_135m_1x_mi300x.yaml",
    }
    expected = {
        "transformer": 132_147_968,
        "irl2": 132_197_136,
        "irl_core": 132_149_504,
    }
    reports = {
        name: architecture_report(TrainConfig.from_yaml(path))
        for name, path in paths.items()
    }
    totals = {name: report["total_parameters"] for name, report in reports.items()}
    assert totals == expected
    spread = (max(expected.values()) - min(expected.values())) / min(expected.values())
    assert spread < 0.01
    assert reports["transformer"]["recurrent_specific_parameters"] == 0
    assert reports["irl2"]["recurrent_depth"] == 2
    assert reports["irl_core"]["recurrent_depth"] == 6
    assert reports["transformer"]["effective_attention_calls_per_forward"] == 16
    assert reports["irl2"]["effective_attention_calls_per_forward"] == 16
    assert reports["irl_core"]["effective_attention_calls_per_forward"] == 21
    qualified = LunarisIRLCoreConfig.from_yaml("profiles/irl_core_135m.yaml")
    benchmark = LunarisIRLCoreConfig.from_yaml("profiles/benchmark_irl_core_135m.yaml")
    assert benchmark.to_dict() == qualified.to_dict()


def test_transformer_bf16_forward_backward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LunarisTransformerForCausalLM(tiny_config()).to(
        device=device,
        dtype=torch.bfloat16,
    )
    tokens = torch.randint(0, model.config.vocab_size, (1, 7), device=device)
    try:
        output = model(
            tokens,
            labels=tokens,
            return_logits=False,
            loss_chunk_size=4,
        )
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
    except RuntimeError as exc:
        if device.type == "cpu" and "BFloat16" in str(exc):
            pytest.skip(f"This CPU build lacks BF16 support: {exc}")
        raise
    assert model.blocks[0].attention.qkv_proj.weight.grad is not None
