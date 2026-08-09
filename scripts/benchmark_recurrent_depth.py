#!/usr/bin/env python3
"""Measure loss, latency, memory, and recurrence diagnostics by compute budget."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from lunaris_dense_irl2 import LunarisDenseForCausalLM
from lunaris_dense_irl2.training.data import MemoryMappedTokenDataset
from lunaris_dense_irl3 import LunarisIRL3ForCausalLM
from lunaris_dense_irl3.training.logging import estimate_forward_flops_per_token


def _parse_depths(value: str) -> list[int]:
    depths = [int(item) for item in value.split(",") if item.strip()]
    if not depths or min(depths) <= 0:
        raise argparse.ArgumentTypeError("depths must be positive comma-separated ints")
    return sorted(set(depths))


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    floating = value.detach().float()
    return {
        "mean": float(floating.mean().cpu()),
        "minimum": float(floating.min().cpu()),
        "maximum": float(floating.max().cpu()),
    }


def _irl3_diagnostics(output: Any) -> dict[str, Any]:
    diagnostics = output.irl3_diagnostics
    if diagnostics is None:
        return {}
    tensor_names = (
        "halt_probabilities",
        "halt_weights",
        "active_latent_fraction",
        "anchor_read_rms",
        "anchor_attention_entropy",
        "memory_read_rms",
        "memory_write_rms",
        "memory_utilization",
        "z_update_rms",
        "h_update_rms",
        "m_update_rms",
        "z_rms",
        "h_rms",
        "m_rms",
        "slow_update_mask",
        "fast_residual",
        "joint_residual",
        "fast_contraction_ratios",
        "contraction_ratios",
        "equilibrium_z_residual",
        "equilibrium_h_residual",
        "equilibrium_m_residual",
        "last_h_update_rms",
        "writeback_rms",
        "writeback_to_anchor_ratio",
        "expected_steps",
        "step_counts",
    )
    result = {
        name: _tensor_summary(getattr(diagnostics, name))
        for name in tensor_names
        if getattr(diagnostics, name).numel() > 0
    }
    result.update(
        iterations=diagnostics.iterations,
        slow_updates=diagnostics.slow_updates,
        last_slow_update_step=diagnostics.last_slow_update_step,
        converged_fraction=diagnostics.converged_fraction,
        halt_mode=diagnostics.halt_mode,
    )
    counts = diagnostics.step_counts.detach().cpu().flatten().tolist()
    result["step_count_histogram"] = {
        str(depth): counts.count(depth) for depth in sorted(set(counts))
    }
    return result


def _irl2_diagnostics(output: Any) -> dict[str, Any]:
    if not output.irl_diagnostics:
        return {}
    return {
        "layers": [item.as_dict() for item in output.irl_diagnostics],
        "maximum_iterations": max(item.iterations for item in output.irl_diagnostics),
    }


def _batches(
    dataset: MemoryMappedTokenDataset,
    *,
    batch_size: int,
    count: int,
) -> list[torch.Tensor]:
    required = batch_size * count
    if len(dataset) < required:
        raise ValueError(
            f"Validation dataset has {len(dataset)} samples; {required} required"
        )
    return [
        torch.stack([dataset[index] for index in range(start, start + batch_size)])
        for start in range(0, required, batch_size)
    ]


@torch.inference_mode()
def benchmark(
    *,
    architecture: str,
    checkpoint: Path,
    data_directory: Path,
    depths: Iterable[int],
    sequence_length: int,
    batch_size: int,
    batches: int,
    repeats: int,
    precision: str,
    halt_mode: str,
    disable_writeback: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Recurrent-depth qualification requires a CUDA/ROCm device")
    device = torch.device("cuda")
    dtype = _dtype(precision)
    if architecture != "irl3" and disable_writeback:
        raise ValueError("Write-back ablation is available only for IRL-3")
    if architecture == "irl3":
        model: Any = LunarisIRL3ForCausalLM.from_pretrained(checkpoint)
    else:
        model = LunarisDenseForCausalLM.from_pretrained(checkpoint)
    model = model.to(device=device, dtype=dtype).eval()
    model.set_regional_compile(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dataset = MemoryMappedTokenDataset(
        data_directory,
        sequence_length=sequence_length,
        mmap_cache_size=2,
    )
    host_batches = _batches(dataset, batch_size=batch_size, count=batches)
    device_batches = [tokens.to(device, non_blocking=False) for tokens in host_batches]

    records: list[dict[str, Any]] = []
    for depth in depths:
        forward_kwargs: dict[str, Any] = {
            "labels": device_batches[0],
            "return_logits": False,
            "loss_chunk_size": 1024,
            "loss_backend": "chunked",
            "mtp_active_heads": 0,
        }
        if architecture == "irl3":
            forward_kwargs.update(
                compute_depth=depth,
                halt_mode=halt_mode,
                return_irl3_stats=False,
                disable_irl3_writeback=disable_writeback,
            )
        else:
            forward_kwargs.update(n_iters=depth, return_irl_stats=False)
        model(device_batches[0], **forward_kwargs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_samples: list[float] = []
        cross_entropies: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            for tokens in device_batches:
                kwargs = dict(forward_kwargs)
                kwargs["labels"] = tokens
                output = model(tokens, **kwargs)
                if output.cross_entropy is None:
                    raise RuntimeError("Benchmark forward did not return CE")
                cross_entropies.append(float(output.cross_entropy.float().cpu()))
            torch.cuda.synchronize(device)
            elapsed_samples.append(time.perf_counter() - start)
        median_elapsed = statistics.median(elapsed_samples)
        timed_peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        torch.cuda.reset_peak_memory_stats(device)
        diagnostic_kwargs = dict(forward_kwargs)
        diagnostic_kwargs["labels"] = device_batches[0]
        if architecture == "irl3":
            diagnostic_kwargs["return_irl3_stats"] = True
        else:
            diagnostic_kwargs["return_irl_stats"] = True
        diagnostic_output = model(device_batches[0], **diagnostic_kwargs)
        torch.cuda.synchronize(device)
        diagnostic_peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        predicted_tokens = repeats * batches * batch_size * (sequence_length - 1)
        diagnostics = (
            _irl3_diagnostics(diagnostic_output)
            if architecture == "irl3"
            else _irl2_diagnostics(diagnostic_output)
        )
        analytic_budget_flops: float | None = None
        analytic_diagnostic_flops: float | None = None
        if architecture == "irl3":
            analytic_budget_flops = estimate_forward_flops_per_token(
                model.config,
                sequence_length,
                recurrent_depth=depth,
                mtp_active_heads=0,
            )["total"]
            diagnostic_depth = int(diagnostics.get("iterations", depth))
            analytic_diagnostic_flops = estimate_forward_flops_per_token(
                model.config,
                sequence_length,
                recurrent_depth=diagnostic_depth,
                mtp_active_heads=0,
            )["total"]
        record: dict[str, Any] = {
            "depth": depth,
            "halt_mode": halt_mode if architecture == "irl3" else "irl2",
            "evaluation_variant": (
                "writeback_off"
                if disable_writeback
                else (halt_mode if architecture == "irl3" else "irl2")
            ),
            "writeback_disabled": bool(disable_writeback),
            "cross_entropy_mean": statistics.fmean(cross_entropies),
            "cross_entropy_samples": cross_entropies,
            "elapsed_seconds_samples": elapsed_samples,
            "median_elapsed_seconds": median_elapsed,
            "median_batch_latency_milliseconds": (
                1000.0 * median_elapsed / batches
            ),
            "median_tokens_per_second": predicted_tokens / (repeats * median_elapsed),
            "peak_memory_gib": timed_peak_memory,
            "diagnostic_peak_memory_gib": diagnostic_peak_memory,
            "analytic_budget_forward_flops_per_token": analytic_budget_flops,
            "analytic_diagnostic_forward_flops_per_token": (
                analytic_diagnostic_flops
            ),
            "diagnostics": diagnostics,
        }
        records.append(record)
    return {
        "schema_version": 2,
        "architecture": architecture,
        "checkpoint": str(checkpoint.resolve()),
        "parameters": parameter_count,
        "device": torch.cuda.get_device_name(device),
        "precision": precision,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "batches": batches,
        "repeats": repeats,
        "halt_mode": halt_mode if architecture == "irl3" else "irl2",
        "writeback_disabled": bool(disable_writeback),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("irl3", "irl2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--depths", type=_parse_depths, default=[2, 4, 6, 8])
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--halt-mode", choices=("fixed", "act", "adaptive"), default="fixed"
    )
    parser.add_argument(
        "--disable-writeback",
        action="store_true",
        help="Evaluation-only IRL-3 correction ablation on the same checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sequence_length, args.batch_size, args.batches, args.repeats) <= 0:
        raise ValueError(
            "sequence length, batch size, batches, and repeats are positive"
        )
    result = benchmark(
        architecture=args.architecture,
        checkpoint=args.checkpoint,
        data_directory=args.data,
        depths=args.depths,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        batches=args.batches,
        repeats=args.repeats,
        precision=args.precision,
        halt_mode=args.halt_mode,
        disable_writeback=args.disable_writeback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
