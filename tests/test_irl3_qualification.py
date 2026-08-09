"""Qualification aggregation tests for the falsification-oriented controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.benchmark_recurrent_depth import (
    _paired_ce_summary,
    _paired_tensor_metrics,
)
from scripts.build_artifact_chunk_manifest import build_manifest
from scripts.generate_irl3_recurrent_utility_configs import (
    POLICIES,
    generate_stage_a,
)
from scripts.prepare_irl3_recurrent_utility_data import _range, _validate_ranges
from scripts.summarize_irl3_qualification import build_summary, markdown
from scripts.summarize_irl3_recurrent_utility import _curve


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
                "evaluation_variant": ("writeback_off" if writeback_disabled else mode),
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
            irl3_writeback_off=_report("fixed", [4], writeback_disabled=True),
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


def test_recurrent_utility_policy_matrix_changes_only_intended_controls(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "training_profiles").mkdir(parents=True)
    (repo_root / "profiles").mkdir()
    source = (
        Path(__file__).parents[1]
        / "training_profiles/irl3_135m_1x_mi300x_deep_compute_continuation.yaml"
    )
    profile = Path(__file__).parents[1] / "profiles/irl3_135m_mi300x_qualification.yaml"
    (repo_root / "training_profiles" / source.name).write_bytes(source.read_bytes())
    (repo_root / "profiles" / profile.name).write_bytes(profile.read_bytes())
    starting = tmp_path / "step_00001000"
    starting.mkdir()

    generated = generate_stage_a(repo_root, tmp_path / "artifacts", starting)

    assert set(generated) == set(POLICIES)
    control = yaml.safe_load(Path(generated["control"]["config"]).read_text())
    fixed = yaml.safe_load(Path(generated["fixed_depth_only"]["config"]).read_text())
    full = yaml.safe_load(Path(generated["full_credit_only"]["config"]).read_text())
    zero = yaml.safe_load(Path(generated["zero_ponder_only"]["config"]).read_text())
    assert control["objectives"]["depth_sampling"] == "log_uniform"
    assert control["model"]["irl3_gradient_horizon"] == 4
    assert control["objectives"]["ponder_weight"] == pytest.approx(1e-3)
    assert fixed["objectives"]["depth_sampling"] == "fixed"
    assert fixed["model"]["irl3_gradient_horizon"] == 4
    assert full["objectives"]["depth_sampling"] == "log_uniform"
    assert full["model"]["irl3_gradient_horizon"] == 6
    assert zero["objectives"]["ponder_weight"] == 0.0


def test_recurrent_utility_data_ranges_reject_overlap() -> None:
    _validate_ranges(
        100,
        [
            _range("train", 0, 50),
            _range("dev", 60, 10),
            _range("holdout", 80, 20),
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        _validate_ranges(
            100,
            [_range("train", 0, 70), _range("dev", 60, 10)],
        )


def test_recurrent_utility_curve_classifies_significant_depths(tmp_path) -> None:
    records = []
    for depth, delta in ((2, 0.0), (4, -2e-4), (8, 3e-4)):
        records.append(
            {
                "depth": depth,
                "cross_entropy_mean": 5.0 + delta,
                "median_batch_latency_milliseconds": float(depth),
                "median_tokens_per_second": 1000.0,
                "peak_memory_gib": 3.0,
                "diagnostics": {},
                "state_comparison_to_reference": {},
                "paired_cross_entropy_delta_to_reference": {
                    "mean": delta,
                    "standard_error": 1e-5,
                    "confidence_95_low": delta - 2e-5,
                    "confidence_95_high": delta + 2e-5,
                },
            }
        )
    path = tmp_path / "curve.json"
    path.write_text(
        json.dumps({"evaluated_predicted_tokens": 1000, "records": records})
    )
    result = _curve(path)
    assert result["best_depth"] == 4
    assert result["detectable_gain_depths"] == [4]
    assert result["detectable_harm_depths"] == [8]
