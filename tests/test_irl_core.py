"""Correctness tests for the simplified immutable-anchor IRL Core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lunaris_dense_irl2.training.export_checkpoint import export_checkpoint
from lunaris_dense_irl2.training.train import train
from lunaris_dense_irl_core import (
    LunarisIRLCoreConfig,
    LunarisIRLCoreForCausalLM,
)


def tiny_core_config(**overrides: object) -> LunarisIRLCoreConfig:
    """Return a small, fast configuration with the production recurrence."""

    values = {
        "vocab_size": 127,
        "d_model": 32,
        "n_prelude_layers": 1,
        "n_coda_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 48,
        "max_seq_len": 32,
        "attention_pattern": "G",
        "local_window": 8,
        "irl_train_iters": 6,
        "irl_eval_iters": 6,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    values.update(overrides)
    return LunarisIRLCoreConfig(**values)


def test_forward_backward_has_finite_nonzero_core_gradients() -> None:
    torch.manual_seed(101)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).train()
    tokens = torch.randint(0, config.vocab_size, (2, 8))

    output = model(tokens, labels=tokens, return_irl_stats=True)
    assert output.logits is not None
    assert output.logits.shape == (2, 8, config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.irl_diagnostics is not None
    diagnostics = output.irl_diagnostics[0]
    assert diagnostics.iterations == 6
    assert diagnostics.state_rms.shape == (6,)
    assert diagnostics.adjacent_update_cosine.shape == (5,)

    output.loss.backward()
    gradients = []
    for name, parameter in model.irl_core.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        gradients.append(parameter.grad.detach().abs().sum())
    assert torch.stack(gradients).sum().item() > 0.0


def test_recurrence_shares_one_parameter_set_and_supports_depth_override() -> None:
    torch.manual_seed(102)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 7))
    parameter_ids = {id(parameter) for parameter in model.irl_core.parameters()}

    with torch.no_grad():
        t1 = model(tokens, n_iters=1, return_irl_stats=True)
        t6 = model(tokens, n_iters=6, return_irl_stats=True)

    assert parameter_ids == {id(parameter) for parameter in model.irl_core.parameters()}
    assert t1.irl_diagnostics is not None
    assert t6.irl_diagnostics is not None
    assert t1.irl_diagnostics[0].iterations == 1
    assert t6.irl_diagnostics[0].iterations == 6
    assert not torch.allclose(t1.logits, t6.logits)
    assert not any(
        isinstance(module, torch.nn.ModuleList)
        for module in model.irl_core.modules()
    )


def test_anchor_kv_are_projected_once_while_queries_change_each_step() -> None:
    torch.manual_seed(103)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    calls = {"q": 0, "k": 0, "v": 0}

    def count(name: str):
        def hook(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
            calls[name] += 1

        return hook

    handles = [
        model.irl_core.q_proj.register_forward_hook(count("q")),
        model.irl_core.k_proj.register_forward_hook(count("k")),
        model.irl_core.v_proj.register_forward_hook(count("v")),
    ]
    try:
        with torch.no_grad():
            model(tokens, n_iters=6)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"q": 6, "k": 1, "v": 1}


def test_no_future_leakage() -> None:
    torch.manual_seed(104)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
    first = torch.tensor([[5, 7, 11, 13, 17, 19]])
    second = first.clone()
    second[:, -1] = 23

    with torch.no_grad():
        logits_a = model(first).logits
        logits_b = model(second).logits

    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1], rtol=0.0, atol=1e-6)


def test_packed_documents_are_isolated() -> None:
    torch.manual_seed(105)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
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


def test_cache_and_generation_match_full_forward() -> None:
    torch.manual_seed(106)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        full_logits = model(tokens).logits
        cache = model.init_cache(batch_size=1)
        pieces = [
            model(
                tokens[:, index : index + 1],
                cache=cache,
                logits_to_keep=1,
            ).logits
            for index in range(tokens.size(1))
        ]
        cached_logits = torch.cat(pieces, dim=1)
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-4, atol=1e-5)

    generated = model.generate(tokens[:, :3], max_new_tokens=2, temperature=0.0)
    assert generated.shape == (1, 5)


def test_deterministic_evaluation_and_save_load(tmp_path: Path) -> None:
    torch.manual_seed(107)
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        expected = model(tokens).logits
        repeated = model(tokens).logits
    torch.testing.assert_close(repeated, expected, rtol=0.0, atol=0.0)

    model.save_pretrained(tmp_path)
    restored = LunarisIRLCoreForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = restored(tokens).logits
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("policy", ["core", "block"])
def test_gradient_checkpointing_preserves_core_gradients(policy: str) -> None:
    torch.manual_seed(108)
    config = tiny_core_config(
        irl_train_iters=2,
        use_gradient_checkpointing=True,
        gradient_checkpointing_policy=policy,
    )
    model = LunarisIRLCoreForCausalLM(config).train()
    tokens = torch.randint(0, config.vocab_size, (2, 7))
    output = model(tokens, labels=tokens)
    assert output.loss is not None
    output.loss.backward()
    for name, parameter in model.irl_core.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_bf16_forward_backward() -> None:
    torch.manual_seed(109)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tiny_core_config(irl_train_iters=2)
    model = LunarisIRLCoreForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    tokens = torch.randint(0, config.vocab_size, (1, 7), device=device)
    try:
        output = model(tokens, labels=tokens, return_logits=False, loss_chunk_size=4)
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
    except RuntimeError as exc:
        if device.type == "cpu" and "BFloat16" in str(exc):
            pytest.skip(f"This CPU build lacks BF16 support: {exc}")
        raise
    assert model.irl_core.q_proj.weight.grad is not None
    assert torch.isfinite(model.irl_core.q_proj.weight.grad).all()


def test_core_parameter_count_has_no_per_step_duplication() -> None:
    config = tiny_core_config()
    model = LunarisIRLCoreForCausalLM(config)
    expected_core_parameters = (
        2 * config.d_model * config.d_model
        + 2 * config.d_model * config.kv_dim
        + 3 * config.d_model * config.ffn_hidden_dim
        + 4 * config.d_model
        + 2 * config.head_dim
    )
    assert model.get_irl_num_parameters() == expected_core_parameters
    assert model.irl_core.lambda_logit.numel() == config.d_model
    assert model.effective_attention_calls() == config.n_ordinary_layers + 6


@pytest.mark.parametrize(
    ("profile_name", "expected_parameters", "expected_core_parameters"),
    [
        ("irl_core_135m.yaml", 132_149_504, 5_115_008),
        ("irl_core_500m.yaml", 501_813_504, 20_060_352),
    ],
)
def test_production_profile_parameter_counts(
    profile_name: str,
    expected_parameters: int,
    expected_core_parameters: int,
) -> None:
    profile = Path(__file__).parents[1] / "profiles" / profile_name
    config = LunarisIRLCoreConfig.from_yaml(profile)
    with torch.device("meta"):
        model = LunarisIRLCoreForCausalLM(config)
    assert model.get_num_parameters(trainable_only=False) == expected_parameters
    assert model.get_num_parameters(trainable_only=True) == expected_parameters
    assert (
        model.get_irl_num_parameters(trainable_only=False)
        == expected_core_parameters
    )


def _training_payload(
    data_root: Path,
    checkpoint_root: Path,
    max_steps: int,
    resume: str,
) -> dict:
    """Build a tiny fixed-depth core training profile."""

    return {
        "model": {
            "architecture_name": "lunaris_dense_irl_core",
            "vocab_size": 67,
            "d_model": 24,
            "n_prelude_layers": 1,
            "n_coda_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 32,
            "max_seq_len": 16,
            "attention_pattern": "G",
            "local_window": 8,
            "irl_train_iters": 2,
            "irl_eval_iters": 2,
        },
        "data": {
            "train_dir": str(data_root / "train"),
            "validation_dir": None,
            "sequence_length": 8,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "mmap_cache_size": 2,
            "shuffle": True,
            "seed": 3,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-3,
            "fused": "false",
        },
        "schedule": {
            "name": "cosine",
            "warmup_steps": 0,
            "min_lr_ratio": 0.1,
        },
        "parallel": {
            "strategy": "none",
            "precision": "fp32",
            "compile": False,
            "regional_compile": False,
        },
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "loss_chunk_size": 4,
            "z_loss_coefficient": 1e-4,
        },
        "objectives": {
            "irl_start_iters": 2,
            "irl_ramp_steps": 0,
            "irl_consistency_weight": 0.0,
            "irl_consistency_warmup_steps": 0,
            "irl_consistency_ramp_steps": 0,
            "mtp_loss_weight": 0.0,
            "mtp_warmup_steps": 0,
            "mtp_ramp_steps": 0,
            "mtp_horizon_interval": 0,
        },
        "validation": {"interval": 0, "batches": 0, "at_start": False},
        "checkpoint": {
            "directory": str(checkpoint_root),
            "interval": 1,
            "keep_last": 2,
            "async_save": False,
            "resume": resume,
            "save_final": True,
        },
        "logging": {
            "interval": 1,
            "jsonl_path": str(checkpoint_root / "metrics.jsonl"),
            "profile_steps": 0,
        },
    }


def test_training_checkpoint_resume_and_export(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train_directory = data_root / "train"
    train_directory.mkdir(parents=True)
    generator = np.random.default_rng(110)
    np.save(
        train_directory / "tokens.npy",
        generator.integers(0, 67, 256, dtype=np.uint32),
    )
    checkpoint_root = tmp_path / "checkpoints"
    config_path = tmp_path / "train.yaml"

    config_path.write_text(
        yaml.safe_dump(_training_payload(data_root, checkpoint_root, 1, "none")),
        encoding="utf-8",
    )
    train(config_path)
    assert (checkpoint_root / "step_00000001" / "_SUCCESS").exists()

    config_path.write_text(
        yaml.safe_dump(_training_payload(data_root, checkpoint_root, 2, "auto")),
        encoding="utf-8",
    )
    train(config_path)
    second_checkpoint = checkpoint_root / "step_00000002"
    assert (second_checkpoint / "_SUCCESS").exists()
    metrics = json.loads(
        (checkpoint_root / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert "pre_clip_gradient_norm" in metrics
    assert "clipped_step_fraction" in metrics

    export_directory = tmp_path / "exported"
    export_checkpoint(second_checkpoint, export_directory)
    restored = LunarisIRLCoreForCausalLM.from_pretrained(export_directory)
    assert restored.config.architecture_name == "lunaris_dense_irl_core"
