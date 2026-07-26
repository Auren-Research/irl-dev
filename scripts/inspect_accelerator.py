#!/usr/bin/env python3
"""Run a small ROCm/PyTorch health probe and save machine-readable evidence."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def synchronize() -> None:
    """Synchronize the active accelerator when available."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def accelerator_report() -> dict[str, Any]:
    """Collect runtime metadata and execute BF16 matmul and SDPA probes."""

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_hip_runtime": torch.version.hip,
        "accelerator_available": torch.cuda.is_available(),
        "accelerator_count": torch.cuda.device_count(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA or ROCm accelerator")
    if torch.version.hip is None:
        raise RuntimeError(
            "Expected a ROCm PyTorch build, but torch.version.hip is null"
        )

    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    report["device"] = {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability(device)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }

    torch.manual_seed(1337)
    left = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    right = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    synchronize()
    started = time.perf_counter()
    product = left @ right
    synchronize()
    report["bf16_matmul"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "finite": bool(torch.isfinite(product).all().item()),
        "mean": float(product.float().mean().item()),
    }

    query = torch.randn(
        2,
        8,
        512,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        2,
        2,
        512,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    value = torch.randn(
        2,
        2,
        512,
        64,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    synchronize()
    started = time.perf_counter()
    attention = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=True,
        enable_gqa=True,
    )
    attention.float().square().mean().backward()
    synchronize()
    report["sdpa_gqa_backward"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "finite": bool(torch.isfinite(attention).all().item()),
        "query_gradient_finite": bool(torch.isfinite(query.grad).all().item()),
    }

    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

        report["flex_attention_importable"] = True
    except Exception as exc:
        report["flex_attention_importable"] = False
        report["flex_attention_import_error"] = repr(exc)

    report["memory"] = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    return report


def main() -> None:
    """Write the report and echo it to standard output."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qualification_artifacts/accelerator_report.json"),
    )
    args = parser.parse_args()
    report = accelerator_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
