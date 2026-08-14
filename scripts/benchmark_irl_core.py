#!/usr/bin/env python3
"""BF16 correctness, training, memory, and throughput probe for IRL Core."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from lunaris_dense_irl2.training.logging import estimate_training_flops_per_token
from lunaris_dense_irl_core import LunarisIRLCoreConfig, LunarisIRLCoreForCausalLM


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator work when timing a CUDA/ROCm device."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_step(
    model: LunarisIRLCoreForCausalLM,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    n_iters: int,
    loss_chunk_size: int,
    update: bool,
) -> Dict[str, float]:
    """Run and time one terminal-state LM optimization step."""

    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    start = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        output = model(
            tokens,
            labels=tokens,
            n_iters=n_iters,
            return_logits=False,
            loss_chunk_size=loss_chunk_size,
            loss_backend="chunked",
            z_loss_coefficient=1e-4,
        )
    if output.loss is None:
        raise RuntimeError("Model did not return a training loss")
    synchronize(device)
    forward_end = time.perf_counter()
    output.loss.backward()
    synchronize(device)
    backward_end = time.perf_counter()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    core_gradient = model.irl_core.q_proj.weight.grad
    if core_gradient is None or not bool(torch.isfinite(core_gradient).all()):
        raise RuntimeError("IRL Core query projection has invalid gradients")
    if update:
        optimizer.step()
    synchronize(device)
    end = time.perf_counter()
    return {
        "loss": float(output.loss.detach()),
        "pre_clip_gradient_norm": float(grad_norm.detach()),
        "core_query_gradient_rms": float(
            core_gradient.detach().float().square().mean().sqrt()
        ),
        "forward_seconds": forward_end - start,
        "backward_seconds": backward_end - forward_end,
        "optimizer_seconds": end - backward_end,
        "step_seconds": end - start,
    }


@torch.no_grad()
def recurrent_diagnostics(
    model: LunarisIRLCoreForCausalLM,
    tokens: torch.Tensor,
    *,
    depths: List[int],
    device: torch.device,
    loss_chunk_size: int,
) -> Dict[str, Any]:
    """Collect one small post-training depth and state-health diagnostic."""

    model.eval()
    depth_losses: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    for depth in depths:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(
                tokens,
                labels=tokens,
                n_iters=depth,
                return_irl_stats=depth == model.config.irl_eval_iters,
                return_logits=False,
                loss_chunk_size=loss_chunk_size,
                loss_backend="chunked",
            )
        if output.loss is None:
            raise RuntimeError("Diagnostic evaluation did not return a loss")
        depth_losses[str(depth)] = float(output.loss.detach())
        if output.irl_diagnostics is not None:
            diagnostics = output.irl_diagnostics[0].as_dict()
    model.train()
    return {"loss_by_depth": depth_losses, "recurrent": diagnostics}


def checkpoint_resume_smoke(
    model: LunarisIRLCoreForCausalLM,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    n_iters: int,
    loss_chunk_size: int,
) -> Dict[str, Any]:
    """Round-trip model and optimizer state, then complete another update."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1,
        },
        checkpoint_path,
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    resumed = timed_step(
        model,
        optimizer,
        tokens,
        device=device,
        n_iters=n_iters,
        loss_chunk_size=loss_chunk_size,
        update=True,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "restored_step": int(payload["step"]),
        "resume_step_loss": resumed["loss"],
        "resume_step_seconds": resumed["step_seconds"],
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--loss-chunk-size", type=int, default=512)
    parser.add_argument("--n-iters", type=int, default=6)
    parser.add_argument("--depths", default="")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--whole-model-compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the requested profile and write a compact JSON summary."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA or ROCm accelerator")
    device = torch.device("cuda")
    config = LunarisIRLCoreConfig.from_yaml(args.profile)
    if args.sequence_length > config.max_seq_len:
        raise ValueError("sequence-length exceeds the model profile")

    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    model = LunarisIRLCoreForCausalLM(config).to(device)
    model.set_regional_compile(not args.whole_model_compile)
    if args.whole_model_compile:
        model.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
    model.train()
    optimizer = model.configure_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        fused=True,
    )
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    warmup: List[Dict[str, float]] = []
    for _ in range(args.warmup_steps):
        warmup.append(
            timed_step(
                model,
                optimizer,
                tokens,
                device=device,
                n_iters=args.n_iters,
                loss_chunk_size=args.loss_chunk_size,
                update=True,
            )
        )

    measured = [
        timed_step(
            model,
            optimizer,
            tokens,
            device=device,
            n_iters=args.n_iters,
            loss_chunk_size=args.loss_chunk_size,
            update=True,
        )
        for _ in range(args.steps)
    ]
    step_seconds = [item["step_seconds"] for item in measured]
    gradient_norms = [item["pre_clip_gradient_norm"] for item in measured]
    median_step = statistics.median(step_seconds)
    tokens_per_step = args.batch_size * args.sequence_length
    flops_per_token = estimate_training_flops_per_token(
        config,
        args.sequence_length,
    )
    first_step = warmup[0]["step_seconds"] if warmup else measured[0]["step_seconds"]
    divisor = float(1024**3)
    result: Dict[str, Any] = {
        "profile": str(Path(args.profile).resolve()),
        "architecture": config.architecture_name,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "device": torch.cuda.get_device_name(device),
        "parameters": model.get_num_parameters(trainable_only=False),
        "trainable_parameters": model.get_num_parameters(trainable_only=True),
        "irl_parameters": model.get_irl_num_parameters(trainable_only=False),
        "recurrent_depth": args.n_iters,
        "effective_attention_calls": model.effective_attention_calls(args.n_iters),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "tokens_per_step": tokens_per_step,
        "measured_steps": args.steps,
        "losses": [item["loss"] for item in measured],
        "loss_decreased": measured[-1]["loss"] < measured[0]["loss"],
        "pre_clip_gradient_norms": gradient_norms,
        "clipped_step_fraction": sum(value > 1.0 for value in gradient_norms)
        / len(gradient_norms),
        "core_query_gradient_rms": [
            item["core_query_gradient_rms"] for item in measured
        ],
        "median_step_seconds": median_step,
        "mean_step_seconds": statistics.mean(step_seconds),
        "tokens_per_second": tokens_per_step / statistics.mean(step_seconds),
        "median_forward_seconds": statistics.median(
            item["forward_seconds"] for item in measured
        ),
        "median_backward_seconds": statistics.median(
            item["backward_seconds"] for item in measured
        ),
        "median_optimizer_seconds": statistics.median(
            item["optimizer_seconds"] for item in measured
        ),
        "first_step_seconds": first_step,
        "compilation_time_estimate_seconds": max(0.0, first_step - median_step),
        "allocated_hbm_gib": torch.cuda.memory_allocated(device) / divisor,
        "reserved_hbm_gib": torch.cuda.memory_reserved(device) / divisor,
        "peak_allocated_hbm_gib": torch.cuda.max_memory_allocated(device) / divisor,
        "estimated_training_flops_per_token": flops_per_token,
        "estimated_tflops": (
            tokens_per_step / statistics.mean(step_seconds) * flops_per_token / 1e12
        ),
        "whole_model_compile": bool(args.whole_model_compile),
        "regional_compile": not args.whole_model_compile,
    }

    depths = [int(value) for value in args.depths.split(",") if value]
    if depths:
        result["diagnostics"] = recurrent_diagnostics(
            model,
            tokens,
            depths=depths,
            device=device,
            loss_chunk_size=args.loss_chunk_size,
        )
    if args.checkpoint_path:
        result["checkpoint_resume"] = checkpoint_resume_smoke(
            model,
            optimizer,
            Path(args.checkpoint_path),
            tokens,
            device=device,
            n_iters=args.n_iters,
            loss_chunk_size=args.loss_chunk_size,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
