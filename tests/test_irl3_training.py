"""Training curricula, preflight, and resumable checkpoint tests for IRL-3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lunaris_dense_irl3 import LunarisIRL3Config, LunarisIRL3ForCausalLM
from lunaris_dense_irl3.training.config import ObjectiveConfig, TrainConfig
from lunaris_dense_irl3.training.export_checkpoint import export_checkpoint
from lunaris_dense_irl3.training.logging import (
    _state_attention_flops,
    estimate_forward_flops_per_token,
    estimate_training_flops_breakdown,
)
from lunaris_dense_irl3.training.preflight import build_preflight_report
from lunaris_dense_irl3.training.schedules import auxiliary_schedule
from lunaris_dense_irl3.training.train import train


def test_chunked_loss_matches_materialized_loss() -> None:
    torch.manual_seed(50)
    config = LunarisIRL3Config.debug(
        vocab_size=97,
        d_model=32,
        n_layers=2,
        irl3_coda_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=48,
        max_seq_len=16,
        local_window=8,
        irl3_state_dim=24,
        irl3_state_heads=4,
        irl3_state_ffn_dim=40,
        irl3_num_latents=2,
        irl3_memory_slots=2,
        irl3_anchor_window=4,
        irl3_train_steps=2,
        irl3_eval_steps=2,
        irl3_min_steps=2,
        irl3_gradient_horizon=2,
        mtp_num_heads=0,
    )
    materialized = LunarisIRL3ForCausalLM(config)
    chunked = LunarisIRL3ForCausalLM(config)
    chunked.load_state_dict(materialized.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 9))

    full = materialized(
        tokens,
        labels=tokens,
        halt_mode="act",
        z_loss_coefficient=1e-4,
    )
    compact = chunked(
        tokens,
        labels=tokens,
        halt_mode="act",
        return_logits=False,
        loss_chunk_size=5,
        loss_backend="chunked",
        z_loss_coefficient=1e-4,
    )
    assert full.loss is not None and compact.loss is not None
    torch.testing.assert_close(compact.loss, full.loss, rtol=1e-5, atol=1e-6)


def test_compute_and_credit_curriculum_is_deterministic() -> None:
    model = LunarisIRL3Config.debug(
        irl3_train_steps=8,
        irl3_eval_steps=8,
        irl3_max_steps=12,
        irl3_gradient_horizon=6,
        mtp_num_heads=2,
    )
    objectives = ObjectiveConfig(
        compute_start_steps=2,
        compute_ramp_steps=60,
        depth_sampling="log_uniform",
        depth_sampling_seed=11,
        gradient_horizon_start=2,
        gradient_horizon_ramp_steps=60,
        ponder_weight=0.01,
        ponder_warmup_steps=20,
        ponder_ramp_steps=20,
        equilibrium_weight=0.02,
        equilibrium_warmup_steps=10,
        equilibrium_ramp_steps=20,
        mtp_loss_weight=0.2,
        mtp_warmup_steps=10,
        mtp_ramp_steps=20,
        mtp_horizon_interval=20,
    )
    objectives.validate(model)

    start = auxiliary_schedule(0, model, objectives)
    middle = auxiliary_schedule(30, model, objectives)
    repeated = auxiliary_schedule(30, model, objectives)
    end = auxiliary_schedule(100, model, objectives)

    assert start.compute_depth == start.depth_ceiling == 2
    assert start.gradient_horizon == 2
    assert start.ponder_weight == 0.0
    assert start.mtp_loss_weight == 0.0
    assert middle == repeated
    assert 2 <= middle.compute_depth <= middle.depth_ceiling <= 8
    assert middle.gradient_horizon <= middle.compute_depth
    assert end.depth_ceiling == 8
    assert end.gradient_horizon == min(6, end.compute_depth)
    assert end.ponder_weight == 0.01
    assert end.equilibrium_weight == 0.02
    assert end.mtp_active_heads == 2


def test_training_halt_mode_is_explicit_and_rejects_adaptive() -> None:
    model = LunarisIRL3Config.debug(mtp_num_heads=0)
    fixed = ObjectiveConfig(mtp_loss_weight=0.0, training_halt_mode="fixed")
    fixed.validate(model)
    assert fixed.training_halt_mode == "fixed"

    invalid = ObjectiveConfig(mtp_loss_weight=0.0, training_halt_mode="adaptive")
    with pytest.raises(ValueError, match="training_halt_mode"):
        invalid.validate(model)


def test_flops_account_for_slot_rows_slow_ticks_mtp_and_checkpointing() -> None:
    config = LunarisIRL3Config.debug(
        irl3_num_latents=3,
        irl3_memory_slots=2,
        irl3_slow_slots=1,
        irl3_slow_update_interval=3,
        irl3_train_steps=6,
        irl3_gradient_horizon=2,
        mtp_num_heads=2,
        mtp_ffn_hidden_dim=80,
        use_gradient_checkpointing=True,
        gradient_checkpointing_policy="computer",
    )
    s = config.state_dim
    assert _state_attention_flops(3, 2, s) == (
        4.0 * (3 + 2) * s * s + 4.0 * 3 * 2 * s
    )

    depth_five = estimate_forward_flops_per_token(
        config,
        32,
        recurrent_depth=5,
        mtp_active_heads=2,
    )
    depth_six = estimate_forward_flops_per_token(
        config,
        32,
        recurrent_depth=6,
        mtp_active_heads=2,
    )
    depth_four = estimate_forward_flops_per_token(
        config,
        32,
        recurrent_depth=4,
        mtp_active_heads=2,
    )
    fast_step = depth_five["recurrent_fast"] / 5
    slow_tick = depth_six["recurrent_slow"] / 2
    assert depth_five["recurrent_slow"] == depth_four["recurrent_slow"]
    assert depth_six["total"] - depth_five["total"] == fast_step + slow_tick
    assert depth_six["mtp"] > 0

    training = estimate_training_flops_breakdown(
        config,
        32,
        recurrent_depth=6,
        gradient_horizon=2,
        mtp_active_heads=2,
    )
    assert training["checkpoint_recompute"] == (
        training["forward_state_initialization"]
        + training["forward_anchor_kv_projection"]
        + training["forward_recurrent_fast"]
        + training["forward_recurrent_slow"]
        + training["forward_writeback"]
    )
    assert training["training_total"] > training["forward_total"]


def _payload(
    data_root: Path,
    checkpoint_root: Path,
    *,
    max_steps: int,
    resume: str,
) -> dict:
    return {
        "model": {
            "vocab_size": 67,
            "d_model": 24,
            "n_layers": 1,
            "irl3_coda_layers": 0,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 32,
            "max_seq_len": 16,
            "attention_pattern": "G",
            "local_window": 8,
            "irl3_state_dim": 16,
            "irl3_state_heads": 4,
            "irl3_state_ffn_dim": 24,
            "irl3_num_latents": 2,
            "irl3_memory_slots": 2,
            "irl3_slow_slots": 1,
            "irl3_anchor_window": 4,
            "irl3_train_steps": 2,
            "irl3_eval_steps": 2,
            "irl3_min_steps": 2,
            "irl3_max_steps": 4,
            "irl3_gradient_horizon": 2,
        },
        "data": {
            "train_dir": str(data_root / "train"),
            "validation_dir": str(data_root / "validation"),
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
            "compute_start_steps": 2,
            "compute_ramp_steps": 0,
            "depth_sampling": "fixed",
            "gradient_horizon_start": 2,
            "gradient_horizon_ramp_steps": 0,
            "ponder_weight": 0.0,
            "equilibrium_weight": 0.001,
            "equilibrium_warmup_steps": 0,
            "equilibrium_ramp_steps": 0,
            "mtp_loss_weight": 0.0,
        },
        "validation": {"interval": 1, "batches": 1, "at_start": False},
        "checkpoint": {
            "directory": str(checkpoint_root),
            "interval": 1,
            "keep_last": 3,
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


def test_cpu_training_checkpoint_resume_export_and_preflight(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for split, seed in (("train", 51), ("validation", 52)):
        directory = data_root / split
        directory.mkdir(parents=True)
        generator = np.random.default_rng(seed)
        np.save(
            directory / "tokens.npy",
            generator.integers(0, 67, 256, dtype=np.uint32),
        )
    checkpoint_root = tmp_path / "checkpoints"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _payload(data_root, checkpoint_root, max_steps=1, resume="none")
        ),
        encoding="utf-8",
    )
    train(config_path)
    assert (checkpoint_root / "step_00000001" / "_SUCCESS").exists()

    config_path.write_text(
        yaml.safe_dump(
            _payload(data_root, checkpoint_root, max_steps=2, resume="auto")
        ),
        encoding="utf-8",
    )
    train(config_path)
    checkpoint = checkpoint_root / "step_00000002"
    assert (checkpoint / "_SUCCESS").exists()

    export_directory = tmp_path / "exported"
    export_checkpoint(checkpoint, export_directory)
    restored = LunarisIRL3ForCausalLM.from_pretrained(export_directory)
    assert restored.config.architecture_name == "lunaris_dense_irl3"

    resolved = TrainConfig.from_yaml(config_path)
    resolved.parallel.strategy = "auto"
    report = build_preflight_report(resolved, world_size=2)
    assert report["architecture"]["parameters"] > 0
    assert report["architecture"]["fast_latents"] == 2
    assert report["topology"]["strategy"] == "fsdp2"
    assert report["budget"]["tokens_per_step"] == 32
    assert report["analytic_flops_per_observed_token"]["training_total"] > 0
