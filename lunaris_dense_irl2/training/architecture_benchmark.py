# ruff: noqa: E501
"""Controlled Transformer/IRL-2/IRL Core benchmark utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.distributed.checkpoint as dcp

from .config import TrainConfig
from .data import (
    DevicePrefetcher,
    DistributedAffineBatchSampler,
    MemoryMappedTokenDataset,
    build_dataloader,
)
from .distributed import autocast_context
from .logging import estimate_training_flops_per_token, memory_metrics
from .model_factory import build_model
from .optimizers import GradientClipper, build_optimizers
from .schedules import auxiliary_schedule

MODEL_ORDER = ("transformer", "irl2", "irl_core")
DEPTHS = (1, 2, 4, 6, 8, 12)
MILESTONE_STEPS = (3_052, 7_629, 15_258)
MAX_BENCHMARK_TOKENS = 500_000_000


def sha256_bytes(value: bytes) -> str:
    """Return a hexadecimal SHA-256 digest."""

    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a compact benchmark input or artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    """Return the exact source commit when running inside a checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _unique_named_parameters(model: torch.nn.Module) -> Iterable[tuple[str, Any]]:
    """Yield each tied parameter exactly once."""

    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        yield name, parameter


def parameter_breakdown(model: torch.nn.Module) -> Dict[str, int]:
    """Partition unique model parameters into benchmark-reporting categories."""

    architecture_name = model.config.architecture_name
    counts = {
        "embedding_parameters": 0,
        "ordinary_transformer_block_parameters": 0,
        "recurrent_specific_parameters": 0,
        "other_parameters": 0,
    }
    trainable = 0
    total = 0
    for name, parameter in _unique_named_parameters(model):
        size = parameter.numel()
        total += size
        if parameter.requires_grad:
            trainable += size
        if name.startswith("token_embedding."):
            counts["embedding_parameters"] += size
        elif architecture_name == "lunaris_dense_irl_core" and name.startswith(
            "irl_core."
        ):
            counts["recurrent_specific_parameters"] += size
        elif architecture_name == "lunaris_dense_irl2" and (
            ".irl." in name or ".irl_anchor_norm." in name
        ):
            counts["recurrent_specific_parameters"] += size
        elif (
            name.startswith("blocks.")
            or name.startswith("prelude_blocks.")
            or name.startswith("coda_blocks.")
        ):
            counts["ordinary_transformer_block_parameters"] += size
        else:
            counts["other_parameters"] += size
    counts["total_parameters"] = total
    counts["trainable_parameters"] = trainable
    if sum(counts[key] for key in counts if key.endswith("_parameters") and key not in {"total_parameters", "trainable_parameters"}) != total:
        raise RuntimeError("Parameter categories do not sum to the total")
    return counts


def architecture_report(config: TrainConfig) -> Dict[str, Any]:
    """Construct one model on meta and report exact controlled properties."""

    with torch.device("meta"):
        model = build_model(config.model)
    counts = parameter_breakdown(model)
    architecture_name = config.model.architecture_name
    if architecture_name == "lunaris_dense_transformer":
        model_name = "transformer"
        recurrent_modules = 0
        recurrent_depth = 0
        attention_calls = config.model.n_layers
    elif architecture_name == "lunaris_dense_irl2":
        model_name = "irl2"
        recurrent_modules = config.model.n_layers
        recurrent_depth = config.model.irl_train_iters
        attention_calls = config.model.n_layers
    elif architecture_name == "lunaris_dense_irl_core":
        model_name = "irl_core"
        recurrent_modules = 1
        recurrent_depth = config.model.irl_train_iters
        attention_calls = (
            config.model.n_ordinary_layers + config.model.irl_train_iters
        )
    else:
        raise ValueError(f"Unknown benchmark architecture: {architecture_name}")
    report: Dict[str, Any] = {
        "model": model_name,
        "architecture_name": architecture_name,
        **counts,
        "recurrent_modules": recurrent_modules,
        "recurrent_depth": recurrent_depth,
        "effective_attention_calls_per_forward": attention_calls,
        "estimated_training_flops_per_token": estimate_training_flops_per_token(
            config.model,
            int(config.data.sequence_length),
        ),
    }
    return report


def _sampler_digest(config: TrainConfig, dataset_size: int) -> Dict[str, Any]:
    """Fingerprint a deterministic prefix of the absolute training stream."""

    sampler = DistributedAffineBatchSampler(
        dataset_size,
        batch_size=config.training.micro_batch_size,
        rank=0,
        world_size=1,
        seed=config.data.seed,
        shuffle=config.data.shuffle,
        infinite=True,
        max_batches=256,
    )
    digest = hashlib.sha256()
    first_batch: Optional[list[int]] = None
    last_batch: Optional[list[int]] = None
    samples = 0
    for batch in sampler:
        if first_batch is None:
            first_batch = list(batch)
        last_batch = list(batch)
        samples += len(batch)
        for index in batch:
            digest.update(int(index).to_bytes(8, "little", signed=False))
    return {
        "sha256": digest.hexdigest(),
        "samples_hashed": samples,
        "first_batch_indices": first_batch,
        "last_batch_indices": last_batch,
    }


def build_preflight(
    config_paths: Dict[str, Path],
    *,
    data_manifest_path: Path,
) -> Dict[str, Any]:
    """Verify parameter, optimizer, data-order, and token-budget controls."""

    if tuple(config_paths) != MODEL_ORDER:
        raise ValueError(f"Expected config order {MODEL_ORDER}, got {tuple(config_paths)}")
    configs = {name: TrainConfig.from_yaml(path) for name, path in config_paths.items()}
    reports = {name: architecture_report(config) for name, config in configs.items()}

    model_controls = (
        "vocab_size",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "ffn_hidden_dim",
        "max_seq_len",
        "attention_pattern",
        "local_window",
        "attention_backend",
        "qk_norm",
        "rms_norm_eps",
        "tie_word_embeddings",
        "rope_theta_global",
        "rope_theta_local",
    )
    reference = configs["transformer"]
    for name, config in configs.items():
        for field in model_controls:
            if getattr(config.model, field) != getattr(reference.model, field):
                raise ValueError(f"Model control {field} differs for {name}")
        for section in ("data", "optimizer", "schedule", "parallel", "training"):
            left = asdict(getattr(config, section))
            right = asdict(getattr(reference, section))
            if left != right:
                raise ValueError(f"Controlled training section {section} differs for {name}")
        if config.model.mtp_num_heads != 0:
            raise ValueError(f"MTP must be disabled for {name}")

    if configs["irl2"].model.irl_train_iters != 2:
        raise ValueError("The frozen IRL-2 benchmark profile must train at T=2")
    if configs["irl_core"].model.irl_train_iters != 6:
        raise ValueError("The qualified IRL Core benchmark profile must train at T=6")

    global_batch = (
        reference.training.micro_batch_size
        * reference.training.gradient_accumulation_steps
    )
    tokens_per_step = global_batch * int(reference.data.sequence_length)
    planned_tokens = reference.training.max_steps * tokens_per_step
    if planned_tokens > MAX_BENCHMARK_TOKENS:
        raise ValueError(
            f"Planned trajectory exceeds 500M tokens: {planned_tokens:,}"
        )
    milestones = [
        {
            "step": step,
            "tokens": step * tokens_per_step,
        }
        for step in MILESTONE_STEPS
    ]
    if milestones[-1]["step"] != reference.training.max_steps:
        raise ValueError("The final milestone must equal training.max_steps")

    totals = [report["total_parameters"] for report in reports.values()]
    parameter_spread_fraction = (max(totals) - min(totals)) / min(totals)
    if parameter_spread_fraction > 0.01:
        raise ValueError("Benchmark parameter counts differ by more than 1%")

    manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    configured_split_paths = {
        "train": Path(reference.data.train_dir).resolve(),
        "dev": Path(reference.data.validation_dir or "").resolve(),
    }
    split_ranges = []
    for split in ("train", "dev", "holdout"):
        values = manifest["splits"][split]
        split_path = Path(values["path"]).resolve()
        if split in configured_split_paths and split_path.parent != configured_split_paths[split]:
            raise ValueError(
                f"Configured {split} directory does not match the frozen manifest"
            )
        if file_sha256(split_path) != values["sha256"]:
            raise ValueError(f"Frozen {split} shard digest does not match")
        split_ranges.append((values["start"], values["end_exclusive"], split))
    for left, right in zip(split_ranges, split_ranges[1:]):
        if left[1] > right[0]:
            raise ValueError(f"Data ranges overlap: {left[2]} and {right[2]}")

    train_dataset = MemoryMappedTokenDataset(
        reference.data.train_dir,
        sequence_length=int(reference.data.sequence_length),
        shard_glob=reference.data.shard_glob,
        mmap_cache_size=reference.data.mmap_cache_size,
    )
    sampler_reports = {
        name: _sampler_digest(config, len(train_dataset))
        for name, config in configs.items()
    }
    sampler_hashes = {value["sha256"] for value in sampler_reports.values()}
    if len(sampler_hashes) != 1:
        raise ValueError("Training token-order fingerprints differ")

    config_hashes = {
        name: file_sha256(path.resolve()) for name, path in config_paths.items()
    }
    return {
        "schema_version": 1,
        "source_commit": git_commit(),
        "created_unix_time": time.time(),
        "config_paths": {name: str(path.resolve()) for name, path in config_paths.items()},
        "config_sha256": config_hashes,
        "data_manifest_path": str(data_manifest_path.resolve()),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "model_reports": reports,
        "parameter_spread_fraction": parameter_spread_fraction,
        "global_batch_sequences": global_batch,
        "tokens_per_optimizer_step": tokens_per_step,
        "planned_tokens_per_model": planned_tokens,
        "milestones": milestones,
        "training_seed": reference.training.seed,
        "data_seed": reference.data.seed,
        "sampler_fingerprints": sampler_reports,
        "controls_verified": True,
    }


def _load_checkpoint_model(
    config: TrainConfig,
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load only model tensors from a resumable DCP checkpoint."""

    if not (checkpoint / "_SUCCESS").exists():
        raise FileNotFoundError(f"Incomplete benchmark checkpoint: {checkpoint}")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    model = build_model(config.model)
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=checkpoint / "dcp", no_dist=True)
    model.load_state_dict(state["model"], strict=True)
    # Local FlexAttention is only memory-efficient through its compiled path;
    # disabling this here materializes full LxL score tensors at sequence 4096.
    model.set_regional_compile(config.parallel.regional_compile)
    return model.to(device), metadata


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    config: TrainConfig,
    directory: Path,
    *,
    device: torch.device,
    batch_size: int,
    n_iters: int,
) -> Dict[str, Any]:
    """Return token-weighted CE and paired fixed-batch observations."""

    dataset = MemoryMappedTokenDataset(
        directory,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=config.data.mmap_cache_size,
    )
    batches = len(dataset) // batch_size
    if batches <= 1:
        raise ValueError("Evaluation split must contain at least two full batches")
    loader = build_dataloader(
        dataset,
        config.data,
        batch_size=batch_size,
        rank=0,
        world_size=1,
        training=False,
        max_batches=batches,
    )
    prefetcher = DevicePrefetcher(loader, device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    ce_weighted_sum = 0.0
    predicted_tokens_total = 0
    batch_cross_entropy: list[float] = []
    for _ in range(batches):
        tokens = prefetcher.next()
        predicted_tokens = tokens.size(0) * (tokens.size(1) - 1)
        with autocast_context(device, config.parallel.precision):
            output = model(
                tokens,
                labels=tokens,
                return_logits=False,
                return_irl_stats=False,
                loss_chunk_size=config.training.loss_chunk_size,
                z_loss_coefficient=0.0,
                loss_backend=config.training.loss_backend,
                n_iters=n_iters,
                mtp_loss_weight=0.0,
                mtp_active_heads=0,
                irl_consistency_coefficient=0.0,
            )
        if output.cross_entropy is None:
            raise RuntimeError("The model did not return evaluation CE")
        value = float(output.cross_entropy.detach().float().item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite benchmark evaluation CE")
        batch_cross_entropy.append(value)
        ce_weighted_sum += value * predicted_tokens
        predicted_tokens_total += predicted_tokens
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result: Dict[str, Any] = {
        "cross_entropy": ce_weighted_sum / predicted_tokens_total,
        "perplexity": math.exp(ce_weighted_sum / predicted_tokens_total),
        "n_iters": n_iters,
        "batches": batches,
        "sequences": batches * batch_size,
        "predicted_tokens": predicted_tokens_total,
        "elapsed_seconds": elapsed,
        "tokens_per_second": predicted_tokens_total / elapsed,
        "batch_cross_entropy": batch_cross_entropy,
    }
    result.update(memory_metrics(device))
    return result


@torch.no_grad()
def _determinism_check(
    model: torch.nn.Module,
    config: TrainConfig,
    directory: Path,
    *,
    device: torch.device,
    n_iters: int,
) -> Dict[str, Any]:
    """Repeat exactly one fixed evaluation example."""

    dataset = MemoryMappedTokenDataset(
        directory,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=1,
    )
    tokens = dataset[0].unsqueeze(0).to(device)
    values = []
    for _ in range(2):
        with autocast_context(device, config.parallel.precision):
            output = model(
                tokens,
                labels=tokens,
                return_logits=False,
                return_irl_stats=False,
                loss_chunk_size=config.training.loss_chunk_size,
                z_loss_coefficient=0.0,
                loss_backend=config.training.loss_backend,
                n_iters=n_iters,
            )
        if output.cross_entropy is None:
            raise RuntimeError("Missing determinism-check CE")
        values.append(float(output.cross_entropy.detach().float().item()))
    return {
        "values": values,
        "absolute_difference": abs(values[1] - values[0]),
        "exact_match": values[0] == values[1],
    }


@torch.no_grad()
def _core_diagnostics(
    model: torch.nn.Module,
    config: TrainConfig,
    directory: Path,
    *,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Capture the qualified lightweight Core state summaries on one example."""

    if config.model.architecture_name != "lunaris_dense_irl_core":
        return None
    dataset = MemoryMappedTokenDataset(
        directory,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=1,
    )
    tokens = dataset[0].unsqueeze(0).to(device)
    with autocast_context(device, config.parallel.precision):
        output = model(
            tokens,
            n_iters=config.model.irl_eval_iters,
            return_irl_stats=True,
            logits_to_keep=1,
        )
    if not output.irl_diagnostics:
        raise RuntimeError("IRL Core diagnostics were requested but not returned")
    diagnostics = output.irl_diagnostics[0].as_dict()
    numeric_values = [
        value
        for key, values in diagnostics.items()
        if key != "iterations"
        for value in (values if isinstance(values, list) else [values])
    ]
    diagnostics["all_finite"] = all(math.isfinite(float(value)) for value in numeric_values)
    return diagnostics


def evaluate_checkpoint(
    config_path: Path,
    checkpoint: Path,
    *,
    model_name: str,
    dev_dir: Path,
    holdout_dir: Path,
    output_path: Path,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate one frozen milestone checkpoint without modifying it."""

    config = TrainConfig.from_yaml(config_path)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)
    model, metadata = _load_checkpoint_model(config, checkpoint, device)
    architecture_name = config.model.architecture_name
    depth_values = DEPTHS if architecture_name != "lunaris_dense_transformer" else (1,)
    dev_by_depth: Dict[str, Any] = {}
    for depth in depth_values:
        dev_by_depth[str(depth)] = evaluate_split(
            model,
            config,
            dev_dir,
            device=device,
            batch_size=batch_size,
            n_iters=depth,
        )
    default_depth = config.model.irl_eval_iters
    holdout = evaluate_split(
        model,
        config,
        holdout_dir,
        device=device,
        batch_size=batch_size,
        n_iters=default_depth,
    )
    training_state = metadata["training_state"]
    result = {
        "schema_version": 1,
        "source_commit": git_commit(),
        "model": model_name,
        "architecture_name": architecture_name,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": int(training_state["step"]),
        "training_tokens": int(training_state["tokens_seen"]),
        "default_eval_depth": default_depth,
        "dev": dev_by_depth[str(default_depth)],
        "holdout": holdout,
        "depth_curve_dev": dev_by_depth,
        "determinism": _determinism_check(
            model,
            config,
            dev_dir,
            device=device,
            n_iters=default_depth,
        ),
        "irl_core_diagnostics": _core_diagnostics(
            model,
            config,
            dev_dir,
            device=device,
        ),
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _median(values: Sequence[float]) -> float:
    """Return a numeric median and reject empty timing lists."""

    if not values:
        raise ValueError("Cannot summarize an empty timing sequence")
    return float(statistics.median(values))


def qualify_throughput(
    config_path: Path,
    *,
    output_path: Path,
    steps: int = 4,
) -> Dict[str, Any]:
    """Run a short real-data BF16 forward/backward timing qualification."""

    if steps < 3:
        raise ValueError("At least three qualification steps are required")
    if not torch.cuda.is_available():
        raise RuntimeError("MI300X throughput qualification requires an accelerator")
    config = TrainConfig.from_yaml(config_path)
    device = torch.device("cuda", 0)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    dataset = MemoryMappedTokenDataset(
        config.data.train_dir,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=1,
    )
    tokens = torch.stack(
        [dataset[index] for index in range(config.training.micro_batch_size)]
    ).to(device)

    allocation_started = time.perf_counter()
    model = build_model(config.model).to(device)
    model.set_regional_compile(config.parallel.regional_compile)
    if config.parallel.compile:
        model.compile(mode=config.parallel.compile_mode, dynamic=False)
    optimizers = build_optimizers(model, config.optimizer)
    clipper = GradientClipper(config.training)
    allocation_seconds = time.perf_counter() - allocation_started
    auxiliary = auxiliary_schedule(0, config.model, config.objectives)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[Dict[str, float]] = []
    recurrent_gradients: list[float] = []

    for step in range(steps):
        optimizers.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        total_started = time.perf_counter()
        forward_started = total_started
        with autocast_context(device, config.parallel.precision):
            output = model(
                tokens,
                labels=tokens,
                return_logits=False,
                return_irl_stats=False,
                loss_chunk_size=config.training.loss_chunk_size,
                z_loss_coefficient=config.training.z_loss_coefficient,
                loss_backend=config.training.loss_backend,
                n_iters=auxiliary.irl_iters,
                mtp_loss_weight=0.0,
                mtp_active_heads=0,
                irl_consistency_coefficient=0.0,
            )
        if output.loss is None:
            raise RuntimeError("Qualification model did not return a loss")
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started

        backward_started = time.perf_counter()
        output.loss.backward()
        torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started

        optimizer_started = time.perf_counter()
        grad_norm = clipper.clip(model, error_if_nonfinite=True)
        optimizers.step()
        torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        total_seconds = time.perf_counter() - total_started
        timings.append(
            {
                "step": step + 1,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "total_seconds": total_seconds,
                "loss": float(output.loss.detach().float().item()),
                "pre_clip_gradient_norm": float(grad_norm.detach().float().item()),
            }
        )
        recurrent_total = 0.0
        for name, parameter in model.named_parameters():
            is_recurrent = (
                config.model.architecture_name == "lunaris_dense_irl_core"
                and name.startswith("irl_core.")
            ) or (
                config.model.architecture_name == "lunaris_dense_irl2"
                and (".irl." in name or ".irl_anchor_norm." in name)
            )
            if is_recurrent and parameter.grad is not None:
                recurrent_total += float(parameter.grad.detach().float().square().sum().item())
        recurrent_gradients.append(math.sqrt(recurrent_total))

    steady = timings[1:]
    median_total = _median([value["total_seconds"] for value in steady])
    tokens_per_step = tokens.numel()
    report = {
        "schema_version": 1,
        "source_commit": git_commit(),
        "architecture_name": config.model.architecture_name,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "steps": timings,
        "allocation_seconds": allocation_seconds,
        "compilation_and_cold_start_seconds_estimate": max(
            0.0,
            timings[0]["total_seconds"] - median_total,
        ),
        "median_steady_step_seconds": median_total,
        "median_steady_forward_seconds": _median(
            [value["forward_seconds"] for value in steady]
        ),
        "median_steady_backward_seconds": _median(
            [value["backward_seconds"] for value in steady]
        ),
        "median_steady_optimizer_seconds": _median(
            [value["optimizer_seconds"] for value in steady]
        ),
        "tokens_per_step": tokens_per_step,
        "steady_tokens_per_second": tokens_per_step / median_total,
        "projected_trajectory_hours": (
            config.training.max_steps * tokens_per_step / (tokens_per_step / median_total)
        )
        / 3600.0,
        "recurrent_gradient_norms": recurrent_gradients,
        "all_losses_finite": all(math.isfinite(value["loss"]) for value in timings),
        **memory_metrics(device),
        "torch": {
            "version": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _config_mapping(paths: Sequence[str]) -> Dict[str, Path]:
    """Map three ordered CLI paths to stable benchmark model names."""

    if len(paths) != len(MODEL_ORDER):
        raise ValueError("Exactly three --config values are required")
    return {name: Path(path) for name, path in zip(MODEL_ORDER, paths)}


def main() -> None:
    """Run preflight, GPU qualification, or one milestone evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", action="append", required=True)
    preflight_parser.add_argument("--data-manifest", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)

    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--config", type=Path, required=True)
    qualify_parser.add_argument("--output", type=Path, required=True)
    qualify_parser.add_argument("--steps", type=int, default=4)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--model-name", choices=MODEL_ORDER, required=True)
    evaluate_parser.add_argument("--dev-dir", type=Path, required=True)
    evaluate_parser.add_argument("--holdout-dir", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    if args.command == "preflight":
        report = build_preflight(
            _config_mapping(args.config),
            data_manifest_path=args.data_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif args.command == "qualify":
        qualify_throughput(args.config, output_path=args.output, steps=args.steps)
    else:
        evaluate_checkpoint(
            args.config,
            args.checkpoint,
            model_name=args.model_name,
            dev_dir=args.dev_dir,
            holdout_dir=args.holdout_dir,
            output_path=args.output,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
