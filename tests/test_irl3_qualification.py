"""Qualification aggregation tests for the falsification-oriented controls."""

from __future__ import annotations

import pytest
import torch

from scripts.benchmark_recurrent_depth import (
    _paired_ce_summary,
    _paired_tensor_metrics,
)
from scripts.build_artifact_chunk_manifest import build_manifest
from scripts.summarize_irl3_qualification import build_summary, markdown


def _report(
    mode: str,
    depths: list[int],
    *,
    checkpoint: str = "/checkpoint/irl3",
    writeback_disabled: bool = False,
) -> dict:
    records = []
    for depth in depths:
        ce = 4.0 - 0.01 * depth + (0.2 if writeback_disabled else 0.0)
        if mode == "adaptive":
            ce += 0.03
        records.append(
            {
                "depth": depth,
                "halt_mode": mode,
                "evaluation_variant": (
                    "writeback_off" if writeback_disabled else mode
                ),
                "writeback_disabled": writeback_disabled,
                "cross_entropy_mean": ce,
                "median_elapsed_seconds": 1.0,
                "median_batch_latency_milliseconds": 500.0 + depth,
                "median_tokens_per_second": 1000.0 / depth,
                "peak_memory_gib": 12.0 + depth / 10,
                "diagnostic_peak_memory_gib": 13.0,
                "analytic_budget_forward_flops_per_token": 1e8 * depth,
                "analytic_diagnostic_forward_flops_per_token": 1e8 * depth,
                "diagnostics": {
                    "expected_steps": {
                        "mean": depth - 0.5,
                        "minimum": 2.0,
                        "maximum": float(depth),
                    },
                    "halt_probabilities": {
                        "mean": 0.2,
                        "minimum": 0.1,
                        "maximum": 0.4,
                    },
                    "writeback_rms": {
                        "mean": 0.1,
                        "minimum": 0.1,
                        "maximum": 0.1,
                    },
                    "writeback_to_anchor_ratio": {
                        "mean": 0.05,
                        "minimum": 0.05,
                        "maximum": 0.05,
                    },
                    "z_update_rms": {"mean": 0.2},
                    "h_update_rms": {"mean": 0.1},
                    "m_update_rms": {"mean": 0.15},
                    "memory_write_rms": {"mean": 0.15},
                    "memory_utilization": {"mean": 0.25},
                    "equilibrium_h_residual": {"mean": 0.01},
                    "fast_contraction_ratios": {"mean": 0.9},
                    "iterations": depth,
                    "slow_updates": depth // 2,
                    "last_slow_update_step": depth,
                    "converged_fraction": 0.5 if mode == "adaptive" else 0.0,
                },
            }
        )
    return {
        "architecture": "irl3",
        "checkpoint": checkpoint,
        "parameters": 135_000_000,
        "precision": "bf16",
        "sequence_length": 4096,
        "batch_size": 1,
        "batches": 2,
        "repeats": 1,
        "halt_mode": mode,
        "writeback_disabled": writeback_disabled,
        "records": records,
    }


def test_summary_exposes_soft_hard_and_writeback_comparisons() -> None:
    fixed = _report("fixed", [2, 4, 6, 8, 12])
    act = _report("act", [4, 8, 12])
    adaptive = _report("adaptive", [4, 8, 12])
    off = _report("fixed", [6], writeback_disabled=True)
    summary = build_summary(
        irl3_metrics=[],
        irl3_fixed=fixed,
        irl3_act=act,
        irl3_adaptive=adaptive,
        irl3_writeback_off=off,
        irl2_metrics=[],
        irl2_depth=None,
        accelerator=None,
    )

    comparison = summary["irl3_soft_vs_hard_halting"]
    assert comparison is not None and len(comparison["records"]) == 3
    assert comparison["records"][0][
        "adaptive_minus_act_cross_entropy"
    ] == pytest.approx(0.03)
    ablation = summary["irl3_writeback_ablation"]
    assert ablation is not None
    assert ablation["records"][0]["disabled_minus_enabled_cross_entropy"] > 0
    report = markdown(summary)
    assert "OFF (write-back disabled)" in report
    assert "ACT-soft versus adaptive-hard" in report


def test_summary_rejects_mixed_irl3_checkpoints() -> None:
    with pytest.raises(ValueError, match="same checkpoint"):
        build_summary(
            irl3_metrics=[],
            irl3_fixed=_report("fixed", [4]),
            irl3_act=_report("act", [4], checkpoint="/checkpoint/other"),
            irl3_adaptive=_report("adaptive", [4]),
            irl3_writeback_off=_report(
                "fixed", [4], writeback_disabled=True
            ),
            irl2_metrics=[],
            irl2_depth=None,
            accelerator=None,
        )


def test_paired_depth_statistics_preserve_sign_and_uncertainty() -> None:
    summary = _paired_ce_summary(
        [2.0, 3.25, 3.5, 5.0],
        [2.0, 3.0, 4.0, 4.5],
    )
    assert summary["mean"] == pytest.approx(0.0625)
    assert summary["standard_error"] > 0.0
    assert summary["confidence_95_low"] < summary["mean"]
    assert summary["confidence_95_high"] > summary["mean"]


def test_recurrent_state_comparison_reports_relative_l2_and_cosine() -> None:
    reference = torch.tensor([3.0, 4.0])
    relative_l2, cosine = _paired_tensor_metrics(reference, reference)
    assert relative_l2 == pytest.approx(0.0)
    assert cosine == pytest.approx(1.0)

    candidate = torch.tensor([-3.0, -4.0])
    relative_l2, cosine = _paired_tensor_metrics(candidate, reference)
    assert relative_l2 == pytest.approx(2.0)
    assert cosine == pytest.approx(-1.0)


def test_artifact_chunk_manifest_is_lossless_and_verifiable(tmp_path) -> None:
    payload = bytes(range(251)) * 41
    archive = tmp_path / "artifacts.tar"
    archive.write_bytes(payload)
    chunks = tmp_path / "chunks"
    manifest = build_manifest(archive, chunks, chunk_bytes=1024)

    reconstructed = b"".join(
        (chunks / item["name"]).read_bytes() for item in manifest["chunks"]
    )
    assert reconstructed == payload
    assert manifest["archive_bytes"] == len(payload)
    assert len(manifest["archive_sha256"]) == 64
