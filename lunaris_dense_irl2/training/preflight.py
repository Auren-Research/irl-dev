"""Allocation-free preflight checks for future large pretraining runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch

from ..model import LunarisDenseForCausalLM
from .config import TrainConfig
from .data import MemoryMappedTokenDataset
from .distributed import fsdp_mesh_shape


def _configuration_fingerprint(config: TrainConfig) -> str:
    """Return a stable identifier for the fully resolved configuration."""

    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_parameter_count(config: TrainConfig) -> int:
    """Count parameters on the meta device without allocating model storage."""

    with torch.device("meta"):
        model = LunarisDenseForCausalLM(config.model)
    return sum(parameter.numel() for parameter in model.parameters())


def _dataset_report(config: TrainConfig) -> tuple[Dict[str, Any], list[str]]:
    """Inspect token shard headers without reading shard payloads."""

    warnings: list[str] = []
    directory = Path(config.data.train_dir)
    if not directory.is_dir():
        warnings.append(
            f"Training data directory is not available on this host: {directory}"
        )
        return {
            "available": False,
            "directory": str(directory),
        }, warnings

    dataset = MemoryMappedTokenDataset(
        directory,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=config.data.mmap_cache_size,
    )
    manifest = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in dataset.shards
    ]
    manifest_digest = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "available": True,
        "directory": str(directory.resolve()),
        "shards": len(dataset.shards),
        "tokens": dataset.total_tokens,
        "complete_sequences": dataset.total_samples,
        "manifest_sha256": manifest_digest,
    }, warnings


def build_preflight_report(
    config: TrainConfig,
    *,
    world_size: int,
) -> Dict[str, Any]:
    """Build a machine-readable run plan without allocating model tensors."""

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    parameter_count = _model_parameter_count(config)
    global_batch = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * world_size
    )
    tokens_per_step = global_batch * int(config.data.sequence_length)
    max_steps = config.training.max_steps
    if config.training.max_tokens is not None:
        max_steps = min(
            max_steps,
            math.ceil(config.training.max_tokens / tokens_per_step),
        )
    planned_tokens = max_steps * tokens_per_step

    strategy = config.parallel.strategy
    if world_size == 1:
        strategy = "none"
    elif strategy == "auto":
        strategy = "fsdp2"
    elif strategy == "none":
        raise ValueError(
            "parallel.strategy=none cannot be planned with world_size > 1"
        )

    mesh_shape: tuple[int, ...] = (world_size,)
    shard_group_size = 1
    if strategy == "fsdp2":
        mesh_shape = fsdp_mesh_shape(
            world_size,
            config.parallel.fsdp_replicate_group_size,
        )
        shard_group_size = mesh_shape[-1]

    # This is deliberately a lower bound: activations, compiler workspaces,
    # temporary logits, communication buffers, and allocator fragmentation
    # depend on the accelerator and are not estimated without a real run.
    persistent_bytes_per_parameter = 16
    persistent_state_gib = (
        parameter_count
        * persistent_bytes_per_parameter
        / shard_group_size
        / float(1024**3)
    )

    dataset, warnings = _dataset_report(config)
    if strategy == "ddp" and parameter_count >= 300_000_000:
        warnings.append(
            "DDP replicates parameters, gradients, and optimizer state on every "
            "GPU; qualify FSDP2 before scaling this model."
        )
    if config.optimizer.name != "adamw":
        warnings.append(
            "The selected matrix optimizer remains experimental; AdamW is the "
            "quality-preserving production baseline."
        )
    if config.parallel.compile:
        warnings.append(
            "Whole-model compilation is not the MI300X-validated path; regional "
            "compilation is the current default."
        )
    if config.parallel.precision == "fp16":
        warnings.append(
            "FP16 requires loss scaling; BF16 is preferred on MI300X and modern "
            "NVIDIA datacenter GPUs."
        )

    return {
        "schema_version": 1,
        "configuration_sha256": _configuration_fingerprint(config),
        "architecture": {
            "name": config.model.architecture_name,
            "parameters": parameter_count,
            "irl_train_iters": config.model.irl_train_iters,
            "irl_eval_iters": config.model.irl_eval_iters,
            "mtp_heads": config.model.mtp_num_heads,
        },
        "topology": {
            "world_size": world_size,
            "strategy": strategy,
            "mesh_shape": list(mesh_shape),
            "replicate_group_size": (
                mesh_shape[0] if len(mesh_shape) == 2 else 1
            ),
            "shard_group_size": shard_group_size,
        },
        "budget": {
            "micro_batch_size": config.training.micro_batch_size,
            "gradient_accumulation_steps": (
                config.training.gradient_accumulation_steps
            ),
            "global_batch_sequences": global_batch,
            "tokens_per_step": tokens_per_step,
            "optimizer_steps": max_steps,
            "planned_tokens": planned_tokens,
        },
        "memory_lower_bound": {
            "persistent_state_gib_per_rank": persistent_state_gib,
            "bytes_per_parameter_assumption": persistent_bytes_per_parameter,
            "excludes": [
                "activations",
                "temporary vocabulary projections",
                "communication buffers",
                "compiler workspaces",
                "allocator fragmentation",
            ],
        },
        "dataset": dataset,
        "runtime": {
            "precision": config.parallel.precision,
            "whole_model_compile": config.parallel.compile,
            "regional_compile": config.parallel.regional_compile,
            "deterministic": config.parallel.deterministic,
            "checkpoint_async": config.checkpoint.async_save,
        },
        "warnings": warnings,
    }


def main() -> None:
    """Validate a YAML run and print or save its preflight report."""

    parser = argparse.ArgumentParser(
        description="Inspect a Lunaris pretraining plan without allocating it."
    )
    parser.add_argument("--config", required=True, help="Training YAML path")
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Planned total accelerator process count",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report destination",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Fail if the configured training shards are unavailable",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    report = build_preflight_report(config, world_size=args.world_size)
    if args.require_data and not report["dataset"]["available"]:
        raise FileNotFoundError(report["dataset"]["directory"])

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
