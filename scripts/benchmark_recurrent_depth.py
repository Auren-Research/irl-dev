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


def _sample_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sample")
    return {
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _paired_tensor_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Compared recurrent tensors must match: {candidate.shape} != "
            f"{reference.shape}"
        )
    candidate_fp32 = candidate.detach().float()
    reference_fp32 = reference.detach().float()
    reference_norm = torch.linalg.vector_norm(reference_fp32).clamp_min(1e-12)
    candidate_norm = torch.linalg.vector_norm(candidate_fp32).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(candidate_fp32 - reference_fp32)
    relative_l2 = relative_l2 / reference_norm
    cosine = (candidate_fp32 * reference_fp32).sum()
    cosine = cosine / (candidate_norm * reference_norm)
    return float(relative_l2.cpu()), float(cosine.cpu())


@torch.inference_mode()
def _irl3_state_comparisons(
    model: LunarisIRL3ForCausalLM,
    device_batches: list[torch.Tensor],
    *,
    depths: list[int],
    reference_depth: int,
    comparison_batches: int,
) -> dict[int, dict[str, Any]]:
    """Compare complete fixed-budget recurrent solutions on paired prefixes."""

    if comparison_batches <= 0 or comparison_batches > len(device_batches):
        raise ValueError("state comparison batches must be in [1, benchmark batches]")
    samples: dict[int, dict[str, list[float]]] = {
        depth: {
            "z_relative_l2": [],
            "h_relative_l2": [],
            "m_relative_l2": [],
            "correction_relative_l2": [],
            "z_cosine_similarity": [],
            "h_cosine_similarity": [],
            "m_cosine_similarity": [],
            "correction_cosine_similarity": [],
        }
        for depth in depths
    }
    captured: dict[str, Any] = {}

    def capture_result(_module: Any, _inputs: Any, output: Any) -> None:
        captured["result"] = output

    handle = model.computer.register_forward_hook(capture_result)

    def solve(tokens: torch.Tensor, depth: int) -> Any:
        captured.clear()
        model(
            tokens,
            compute_depth=depth,
            halt_mode="fixed",
            return_irl3_stats=False,
            logits_to_keep=1,
        )
        if "result" not in captured:
            raise RuntimeError("IRL-3 computer hook did not capture a result")
        return captured["result"]

    try:
        for tokens in device_batches[:comparison_batches]:
            reference = solve(tokens, reference_depth)
            for depth in depths:
                candidate = (
                    reference if depth == reference_depth else solve(tokens, depth)
                )
                pairs = {
                    "z": (candidate.state.z, reference.state.z),
                    "h": (candidate.state.h, reference.state.h),
                    "m": (candidate.state.m, reference.state.m),
                    "correction": (candidate.correction, reference.correction),
                }
                for name, (candidate_tensor, reference_tensor) in pairs.items():
                    relative_l2, cosine = _paired_tensor_metrics(
                        candidate_tensor,
                        reference_tensor,
                    )
                    samples[depth][f"{name}_relative_l2"].append(relative_l2)
                    samples[depth][f"{name}_cosine_similarity"].append(cosine)
    finally:
        handle.remove()

    return {
        depth: {name: _sample_summary(values) for name, values in metrics.items()}
        for depth, metrics in samples.items()
    }


def _paired_ce_summary(
    candidate: list[float],
    reference: list[float],
) -> dict[str, float]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Paired CE samples must be non-empty and equal length")
    deltas = [value - baseline for value, baseline in zip(candidate, reference)]
    mean = statistics.fmean(deltas)
    standard_error = (
        statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else 0.0
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence_95_low": mean - 1.96 * standard_error,
        "confidence_95_high": mean + 1.96 * standard_error,
        "minimum": min(deltas),
        "maximum": max(deltas),
    }


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
    halt_threshold: float | None = None,
    state_reference_depth: int | None = None,
    state_comparison_batches: int = 1,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Recurrent-depth qualification requires a CUDA/ROCm device")
    device = torch.device("cuda")
    dtype = _dtype(precision)
    if architecture != "irl3" and disable_writeback:
        raise ValueError("Write-back ablation is available only for IRL-3")
    if halt_threshold is not None and architecture != "irl3":
        raise ValueError("Halt-threshold override is available only for IRL-3")
    if halt_threshold is not None and not 0.0 < halt_threshold < 1.0:
        raise ValueError("halt_threshold must be in (0, 1)")
    if architecture == "irl3":
        model: Any = LunarisIRL3ForCausalLM.from_pretrained(checkpoint)
        if halt_threshold is not None:
            model.computer.halt_threshold = float(halt_threshold)
        resolved_halt_threshold: float | None = model.computer.halt_threshold
    else:
        model = LunarisDenseForCausalLM.from_pretrained(checkpoint)
        resolved_halt_threshold = None
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

    resolved_depths = list(depths)
    if state_reference_depth is not None:
        if architecture != "irl3" or halt_mode != "fixed":
            raise ValueError(
                "State comparisons require architecture=irl3 and halt_mode=fixed"
            )
        if state_comparison_batches > batches:
            raise ValueError("state comparison batches cannot exceed benchmark batches")

    records: list[dict[str, Any]] = []
    for depth in resolved_depths:
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
            "halt_threshold": resolved_halt_threshold,
            "cross_entropy_mean": statistics.fmean(cross_entropies),
            "cross_entropy_samples": cross_entropies,
            "elapsed_seconds_samples": elapsed_samples,
            "median_elapsed_seconds": median_elapsed,
            "median_batch_latency_milliseconds": (1000.0 * median_elapsed / batches),
            "median_tokens_per_second": predicted_tokens / (repeats * median_elapsed),
            "peak_memory_gib": timed_peak_memory,
            "diagnostic_peak_memory_gib": diagnostic_peak_memory,
            "analytic_budget_forward_flops_per_token": analytic_budget_flops,
            "analytic_diagnostic_forward_flops_per_token": (analytic_diagnostic_flops),
            "diagnostics": diagnostics,
        }
        records.append(record)
    state_comparisons: dict[int, dict[str, Any]] = {}
    if state_reference_depth is not None:
        state_comparisons = _irl3_state_comparisons(
            model,
            device_batches,
            depths=resolved_depths,
            reference_depth=state_reference_depth,
            comparison_batches=state_comparison_batches,
        )
        reference_record = next(
            (record for record in records if record["depth"] == state_reference_depth),
            None,
        )
        if reference_record is None:
            raise ValueError("state reference depth must be included in --depths")
        reference_ce = reference_record["cross_entropy_samples"]
        for record in records:
            depth = int(record["depth"])
            record["state_comparison_to_reference"] = state_comparisons[depth]
            record["paired_cross_entropy_delta_to_reference"] = _paired_ce_summary(
                record["cross_entropy_samples"],
                reference_ce,
            )
    return {
        "schema_version": 3 if state_reference_depth is not None else 2,
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
        "halt_threshold": resolved_halt_threshold,
        "writeback_disabled": bool(disable_writeback),
        "evaluated_predicted_tokens": (
            repeats * batches * batch_size * (sequence_length - 1)
        ),
        "state_reference_depth": state_reference_depth,
        "state_comparison_batches": (
            state_comparison_batches if state_reference_depth is not None else 0
        ),
        "state_comparison_prefixes": (
            state_comparison_batches * batch_size * sequence_length
            if state_reference_depth is not None
            else 0
        ),
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
        "--halt-threshold",
        type=float,
        help="Evaluation-only IRL-3 adaptive cumulative-halt threshold override",
    )
    parser.add_argument(
        "--disable-writeback",
        action="store_true",
        help="Evaluation-only IRL-3 correction ablation on the same checkpoint",
    )
    parser.add_argument(
        "--state-reference-depth",
        type=int,
        help="Compare fixed-budget Z/H/M/corrections against this depth",
    )
    parser.add_argument(
        "--state-comparison-batches",
        type=int,
        default=1,
        help="Number of paired batches used for full-state comparisons",
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
        halt_threshold=args.halt_threshold,
        state_reference_depth=args.state_reference_depth,
        state_comparison_batches=args.state_comparison_batches,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
