"""Tests for scalable training components and the CPU integration path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lunaris_dense_irl2 import LunarisDenseConfig, LunarisDenseForCausalLM
from lunaris_dense_irl2.training.config import (
    ObjectiveConfig,
    OptimizerConfig,
    ScheduleConfig,
    TrainConfig,
)
from lunaris_dense_irl2.training.data import (
    DistributedAffineBatchSampler,
    MemoryMappedTokenDataset,
)
from lunaris_dense_irl2.training.distributed import fsdp_mesh_shape
from lunaris_dense_irl2.training.export_checkpoint import export_checkpoint
from lunaris_dense_irl2.training.optimizers import (
    MONA,
    build_optimizers,
    restore_base_learning_rates,
)
from lunaris_dense_irl2.training.preflight import build_preflight_report
from lunaris_dense_irl2.training.schedules import (
    apply_learning_rate,
    auxiliary_schedule,
    lr_multiplier,
)
from lunaris_dense_irl2.training.train import train


def test_chunked_loss_matches_materialized_loss() -> None:
    torch.manual_seed(31)
    config = LunarisDenseConfig.debug(
        vocab_size=97,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=48,
        max_seq_len=16,
    )
    materialized = LunarisDenseForCausalLM(config)
    chunked = LunarisDenseForCausalLM(config)
    chunked.load_state_dict(materialized.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 9))

    full_output = materialized(
        tokens,
        labels=tokens,
        z_loss_coefficient=1e-4,
    )
    chunked_output = chunked(
        tokens,
        labels=tokens,
        return_logits=False,
        loss_chunk_size=5,
        z_loss_coefficient=1e-4,
    )
    assert full_output.loss is not None
    assert chunked_output.loss is not None
    assert chunked_output.logits is None
    torch.testing.assert_close(
        chunked_output.loss,
        full_output.loss,
        rtol=1e-5,
        atol=1e-6,
    )

    full_output.loss.backward()
    chunked_output.loss.backward()
    for full_parameter, chunked_parameter in zip(
        materialized.parameters(),
        chunked.parameters(),
    ):
        torch.testing.assert_close(
            chunked_parameter.grad,
            full_parameter.grad,
            rtol=2e-4,
            atol=2e-5,
        )


def test_affine_sampler_is_disjoint_and_resumable() -> None:
    sampler_rank_0 = DistributedAffineBatchSampler(
        103,
        batch_size=3,
        rank=0,
        world_size=2,
        seed=19,
        shuffle=True,
        infinite=False,
        max_batches=5,
    )
    sampler_rank_1 = DistributedAffineBatchSampler(
        103,
        batch_size=3,
        rank=1,
        world_size=2,
        seed=19,
        shuffle=True,
        infinite=False,
        max_batches=5,
    )
    rank_0_batches = list(sampler_rank_0)
    rank_1_batches = list(sampler_rank_1)
    rank_0_indices = {index for batch in rank_0_batches for index in batch}
    rank_1_indices = {index for batch in rank_1_batches for index in batch}
    assert rank_0_indices.isdisjoint(rank_1_indices)

    resumed = DistributedAffineBatchSampler(
        103,
        batch_size=3,
        rank=0,
        world_size=2,
        seed=19,
        shuffle=True,
        infinite=True,
        start_global_sample=18,
        max_batches=2,
    )
    assert list(resumed) == rank_0_batches[3:5]


def test_affine_sampler_resume_survives_world_size_change() -> None:
    first_rank_0 = DistributedAffineBatchSampler(
        101,
        batch_size=2,
        rank=0,
        world_size=2,
        seed=23,
        shuffle=True,
        infinite=True,
        max_batches=3,
    )
    first_rank_1 = DistributedAffineBatchSampler(
        101,
        batch_size=2,
        rank=1,
        world_size=2,
        seed=23,
        shuffle=True,
        infinite=True,
        max_batches=3,
    )
    consumed = [
        index
        for rank_0, rank_1 in zip(first_rank_0, first_rank_1)
        for index in (*rank_0, *rank_1)
    ]
    assert len(consumed) == 12

    resumed = DistributedAffineBatchSampler(
        101,
        batch_size=2,
        rank=0,
        world_size=1,
        seed=23,
        shuffle=True,
        infinite=True,
        start_global_sample=12,
        max_batches=2,
    )
    reference = DistributedAffineBatchSampler(
        101,
        batch_size=2,
        rank=0,
        world_size=1,
        seed=23,
        shuffle=True,
        infinite=True,
        max_batches=8,
    )
    assert list(resumed) == list(reference)[6:8]


def test_fsdp_and_hsdp_mesh_shapes() -> None:
    assert fsdp_mesh_shape(8, 1) == (8,)
    assert fsdp_mesh_shape(16, 2) == (2, 8)
    with pytest.raises(ValueError, match="divisible"):
        fsdp_mesh_shape(8, 3)


def test_memory_mapped_dataset_respects_shard_boundaries(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.arange(19, dtype=np.uint32))
    np.save(tmp_path / "b.npy", np.arange(100, 123, dtype=np.uint32))
    dataset = MemoryMappedTokenDataset(tmp_path, sequence_length=8)

    assert len(dataset) == 4
    torch.testing.assert_close(dataset[0], torch.arange(8))
    torch.testing.assert_close(dataset[2], torch.arange(100, 108))


def test_wsd_schedule_has_expected_phases() -> None:
    config = ScheduleConfig(
        name="wsd",
        warmup_steps=2,
        stable_steps=3,
        decay_steps=2,
        min_lr_ratio=0.1,
    )
    config.validate(max_steps=7)
    assert lr_multiplier(0, config, 7) == 0.5
    assert lr_multiplier(1, config, 7) == 1.0
    assert lr_multiplier(4, config, 7) == 1.0
    assert lr_multiplier(7, config, 7) == 0.1


def test_auxiliary_curriculum_stages_irl_and_mtp() -> None:
    model = LunarisDenseConfig.debug(
        irl_train_iters=3,
        mtp_num_heads=3,
    )
    config = ObjectiveConfig(
        irl_start_iters=1,
        irl_ramp_steps=90,
        irl_consistency_weight=0.01,
        irl_consistency_warmup_steps=30,
        irl_consistency_ramp_steps=30,
        mtp_loss_weight=0.2,
        mtp_warmup_steps=10,
        mtp_ramp_steps=20,
        mtp_horizon_interval=15,
    )
    config.validate(model)

    start = auxiliary_schedule(0, model, config)
    middle = auxiliary_schedule(45, model, config)
    end = auxiliary_schedule(100, model, config)

    assert start.irl_iters == 1
    assert start.mtp_active_heads == 1
    assert start.mtp_loss_weight == 0.0
    assert middle.irl_iters == 2
    assert middle.mtp_active_heads == 3
    assert middle.mtp_loss_weight == 0.2
    assert end.irl_iters == 3
    assert end.irl_consistency_weight == 0.01


def test_mona_updates_matrix_parameters() -> None:
    parameter = torch.nn.Parameter(torch.randn(8, 4))
    optimizer = MONA(
        [parameter],
        lr=0.01,
        momentum=0.95,
        acceleration_beta=0.95,
        acceleration_alpha=-0.1,
        weight_decay=0.1,
        ns_steps=3,
    )
    before = parameter.detach().clone()
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert not torch.equal(parameter, before)


def test_resume_restores_scheduler_base_learning_rate() -> None:
    model = LunarisDenseForCausalLM(
        LunarisDenseConfig.debug(
            vocab_size=67,
            d_model=24,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden_dim=32,
            max_seq_len=16,
        )
    )
    config = OptimizerConfig(
        name="adamw",
        learning_rate=1e-3,
        fused="false",
    )
    config.validate()
    optimizers = build_optimizers(model, config)

    for group in optimizers.optimizers[0].param_groups:
        group["lr"] = 1e-5
        group["initial_lr"] = 1e-5
    restore_base_learning_rates(optimizers, config)
    apply_learning_rate(optimizers.optimizers, 0.5)

    for group in optimizers.optimizers[0].param_groups:
        assert group["initial_lr"] == 1e-3
        assert group["lr"] == 5e-4


def _debug_training_payload(
    data_root: Path,
    checkpoint_root: Path,
    *,
    max_steps: int,
    resume: str,
) -> dict:
    """Build a tiny end-to-end training configuration."""

    return {
        "model": {
            "vocab_size": 67,
            "d_model": 24,
            "n_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 32,
            "max_seq_len": 16,
            "attention_pattern": "G",
            "local_window": 8,
            "irl_train_iters": 1,
            "irl_eval_iters": 1,
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
        },
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": max_steps,
            "loss_chunk_size": 4,
            "z_loss_coefficient": 1e-4,
        },
        "validation": {
            "interval": 1,
            "batches": 1,
            "at_start": False,
        },
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


def test_cpu_training_and_checkpoint_resume(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for split, seed in (("train", 41), ("validation", 42)):
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
            _debug_training_payload(
                data_root,
                checkpoint_root,
                max_steps=2,
                resume="none",
            )
        ),
        encoding="utf-8",
    )
    train(config_path, stop_after_step=1)
    assert (checkpoint_root / "step_00000001" / "_SUCCESS").exists()

    config_path.write_text(
        yaml.safe_dump(
            _debug_training_payload(
                data_root,
                checkpoint_root,
                max_steps=2,
                resume="auto",
            )
        ),
        encoding="utf-8",
    )
    train(config_path, stop_after_step=2)
    assert (checkpoint_root / "step_00000002" / "_SUCCESS").exists()

    export_directory = tmp_path / "exported"
    export_checkpoint(
        checkpoint_root / "step_00000002",
        export_directory,
    )
    restored = LunarisDenseForCausalLM.from_pretrained(export_directory)
    assert restored.config.vocab_size == 67

    resolved = TrainConfig.from_yaml(config_path)
    assert resolved.training.max_steps == 2


def test_preflight_uses_meta_model_and_reports_budget(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    train_directory = data_root / "train"
    validation_directory = data_root / "validation"
    train_directory.mkdir(parents=True)
    validation_directory.mkdir(parents=True)
    np.save(
        train_directory / "tokens.npy",
        np.arange(256, dtype=np.uint32) % 67,
    )
    np.save(
        validation_directory / "tokens.npy",
        np.arange(256, dtype=np.uint32) % 67,
    )
    checkpoint_root = tmp_path / "checkpoints"
    config_path = tmp_path / "preflight.yaml"
    config_path.write_text(
        yaml.safe_dump(
            _debug_training_payload(
                data_root,
                checkpoint_root,
                max_steps=7,
                resume="none",
            )
        ),
        encoding="utf-8",
    )

    config = TrainConfig.from_yaml(config_path)
    config.parallel.strategy = "auto"
    report = build_preflight_report(config, world_size=2)
    assert report["architecture"]["parameters"] > 0
    assert report["topology"]["strategy"] == "fsdp2"
    assert report["budget"]["tokens_per_step"] == 32
    assert report["budget"]["planned_tokens"] == 224
    assert report["dataset"]["tokens"] == 256
