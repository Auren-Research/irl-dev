#!/usr/bin/env python3
"""Summarize the step-1000 and deep-compute continuation mechanism study."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(record: dict[str, Any], name: str) -> float | None:
    value = record.get("diagnostics", {}).get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return None


def _comparison_mean(record: dict[str, Any], name: str) -> float | None:
    value = record.get("state_comparison_to_reference", {}).get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (float, int)) and math.isfinite(float(value)):
        return float(value)
    return None


def _paired(values: list[float], reference: list[float]) -> dict[str, float]:
    if not values or len(values) != len(reference):
        raise ValueError("Paired samples must be non-empty and equal length")
    deltas = [float(value) - float(base) for value, base in zip(values, reference)]
    mean = statistics.fmean(deltas)
    standard_error = (
        statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence_95_low": mean - 1.96 * standard_error,
        "confidence_95_high": mean + 1.96 * standard_error,
    }


def _fixed_summary(report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for record in sorted(report["records"], key=lambda item: int(item["depth"])):
        paired = record.get("paired_cross_entropy_delta_to_reference")
        rows.append(
            {
                "depth": int(record["depth"]),
                "cross_entropy": float(record["cross_entropy_mean"]),
                "paired_cross_entropy_delta_to_depth_2": paired,
                "latency_milliseconds": float(
                    record["median_batch_latency_milliseconds"]
                ),
                "tokens_per_second": float(record["median_tokens_per_second"]),
                "peak_memory_gib": float(record["peak_memory_gib"]),
                "writeback_rms": _mean(record, "writeback_rms"),
                "writeback_to_anchor_ratio": _mean(record, "writeback_to_anchor_ratio"),
                "z_rms": _mean(record, "z_rms"),
                "h_rms": _mean(record, "h_rms"),
                "m_rms": _mean(record, "m_rms"),
                "z_update_rms": _mean(record, "z_update_rms"),
                "h_update_rms": _mean(record, "h_update_rms"),
                "m_update_rms": _mean(record, "m_update_rms"),
                "last_meaningful_h_residual": _mean(record, "equilibrium_h_residual"),
                "memory_read_rms": _mean(record, "memory_read_rms"),
                "memory_write_rms": _mean(record, "memory_write_rms"),
                "memory_utilization": _mean(record, "memory_utilization"),
                "anchor_attention_entropy": _mean(record, "anchor_attention_entropy"),
                "fast_residual": _mean(record, "fast_residual"),
                "fast_contraction": _mean(record, "fast_contraction_ratios"),
                "slow_updates": record.get("diagnostics", {}).get("slow_updates"),
                "z_relative_l2_to_depth_2": _comparison_mean(record, "z_relative_l2"),
                "h_relative_l2_to_depth_2": _comparison_mean(record, "h_relative_l2"),
                "m_relative_l2_to_depth_2": _comparison_mean(record, "m_relative_l2"),
                "correction_relative_l2_to_depth_2": _comparison_mean(
                    record, "correction_relative_l2"
                ),
                "z_cosine_to_depth_2": _comparison_mean(record, "z_cosine_similarity"),
                "correction_cosine_to_depth_2": _comparison_mean(
                    record, "correction_cosine_similarity"
                ),
            }
        )
    detectable_gains = [
        row["depth"]
        for row in rows[1:]
        if row["paired_cross_entropy_delta_to_depth_2"]
        and row["paired_cross_entropy_delta_to_depth_2"]["confidence_95_high"] < 0
    ]
    detectable_degradations = [
        row["depth"]
        for row in rows[1:]
        if row["paired_cross_entropy_delta_to_depth_2"]
        and row["paired_cross_entropy_delta_to_depth_2"]["confidence_95_low"] > 0
    ]
    return {
        "checkpoint": report["checkpoint"],
        "evaluated_predicted_tokens": report.get("evaluated_predicted_tokens"),
        "state_comparison_prefixes": report.get("state_comparison_prefixes"),
        "records": rows,
        "statistically_detectable_gain_depths": detectable_gains,
        "statistically_detectable_degradation_depths": detectable_degradations,
    }


def _training_summary(
    records: list[dict[str, Any]], starting_step: int
) -> dict[str, Any] | None:
    if not records:
        return None
    records = sorted(records, key=lambda item: int(item["step"]))
    warm = records[max(1, len(records) // 10) :]
    elapsed = [float(item["elapsed_seconds"]) for item in records]
    validation_curve = []
    last_validation: float | None = None
    for item in records:
        value = item.get("validation_cross_entropy")
        if value is None or float(value) == last_validation:
            continue
        last_validation = float(value)
        validation_curve.append(
            {
                "step": int(item["step"]),
                "cross_entropy": last_validation,
                "total_loss": float(item["validation_loss"]),
                "equilibrium_loss": float(item["validation_equilibrium_loss"]),
            }
        )
    return {
        "starting_step": starting_step,
        "first_logged_step": int(records[0]["step"]),
        "final_logged_step": int(records[-1]["step"]),
        "additional_steps": int(records[-1]["step"]) - starting_step,
        "final_tokens_seen": int(records[-1]["tokens_seen"]),
        "logged_elapsed_seconds": max(elapsed) - min(elapsed),
        "median_warm_tokens_per_second": statistics.median(
            float(item["tokens_per_second"]) for item in warm
        ),
        "peak_memory_gib": max(float(item["memory_peak_gib"]) for item in records),
        "gradient_norm_maximum": max(float(item["gradient_norm"]) for item in records),
        "cross_entropy_first_logged": float(records[0]["cross_entropy"]),
        "cross_entropy_final_logged": float(records[-1]["cross_entropy"]),
        "equilibrium_loss_first_logged": float(records[0]["equilibrium_loss"]),
        "equilibrium_loss_final_logged": float(records[-1]["equilibrium_loss"]),
        "validation_curve": validation_curve,
        "all_finite": all(
            math.isfinite(float(item[name]))
            for item in records
            for name in ("loss", "cross_entropy", "gradient_norm")
        ),
        "compute_depths": sorted({int(item["irl3_compute_depth"]) for item in records}),
        "gradient_horizons": sorted(
            {int(item["irl3_gradient_horizon"]) for item in records}
        ),
        "ponder_weights": sorted({float(item["ponder_weight"]) for item in records}),
        "equilibrium_weights": sorted(
            {float(item["equilibrium_weight"]) for item in records}
        ),
        "mtp_loss_weights": sorted(
            {float(item["mtp_loss_weight"]) for item in records}
        ),
    }


def _writeback_summary(
    fixed: dict[str, Any], off: dict[str, Any] | None
) -> dict[str, Any] | None:
    if off is None:
        return None
    enabled = {int(item["depth"]): item for item in fixed["records"]}
    rows = []
    for disabled in off["records"]:
        depth = int(disabled["depth"])
        if depth not in enabled:
            continue
        on = enabled[depth]
        disabled_samples = disabled["cross_entropy_samples"]
        enabled_samples = on["cross_entropy_samples"]
        if len(disabled_samples) == len(enabled_samples):
            paired = _paired(disabled_samples, enabled_samples)
            paired["paired"] = True
        else:
            paired = {
                "mean": float(disabled["cross_entropy_mean"])
                - float(on["cross_entropy_mean"]),
                "standard_error": None,
                "confidence_95_low": None,
                "confidence_95_high": None,
                "paired": False,
            }
        rows.append(
            {
                "depth": depth,
                "enabled_cross_entropy": float(on["cross_entropy_mean"]),
                "disabled_cross_entropy": float(disabled["cross_entropy_mean"]),
                "disabled_minus_enabled": paired,
                "correction_rms": _mean(on, "writeback_rms"),
                "correction_to_anchor_ratio": _mean(on, "writeback_to_anchor_ratio"),
            }
        )
    return {"records": rows}


def _halting_summary(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    return [
        {
            "budget": int(item["depth"]),
            "cross_entropy": float(item["cross_entropy_mean"]),
            "expected_steps": _mean(item, "expected_steps"),
            "step_counts": item.get("diagnostics", {}).get("step_count_histogram"),
            "halt_probability": _mean(item, "halt_probabilities"),
            "latency_milliseconds": float(item["median_batch_latency_milliseconds"]),
            "tokens_per_second": float(item["median_tokens_per_second"]),
            "peak_memory_gib": float(item["peak_memory_gib"]),
        }
        for item in sorted(report["records"], key=lambda row: int(row["depth"]))
    ]


def _halt_threshold_summary(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        record = report["records"][0]
        diagnostics = record["diagnostics"]
        rows.append(
            {
                "threshold": float(report["halt_threshold"]),
                "budget": int(record["depth"]),
                "cross_entropy": float(record["cross_entropy_mean"]),
                "iterations": int(diagnostics["iterations"]),
                "expected_steps": _mean(record, "expected_steps"),
                "diagnostic_step_histogram": diagnostics["step_count_histogram"],
                "latency_milliseconds": float(
                    record["median_batch_latency_milliseconds"]
                ),
                "writeback_to_anchor_ratio": _mean(record, "writeback_to_anchor_ratio"),
            }
        )
    return sorted(rows, key=lambda item: item["threshold"])


def build_summary(
    *,
    baseline_fixed: dict[str, Any],
    deep_fixed: dict[str, Any],
    deep_metrics: list[dict[str, Any]],
    baseline_control_fixed: dict[str, Any] | None,
    baseline_writeback_off: dict[str, Any] | None,
    deep_writeback_off: dict[str, Any] | None,
    baseline_act: dict[str, Any] | None,
    baseline_adaptive: dict[str, Any] | None,
    deep_act: dict[str, Any] | None,
    deep_adaptive: dict[str, Any] | None,
    halt_threshold_reports: list[dict[str, Any]],
    starting_step: int,
) -> dict[str, Any]:
    baseline = _fixed_summary(baseline_fixed)
    deep = _fixed_summary(deep_fixed)
    baseline_gain = bool(baseline["statistically_detectable_gain_depths"])
    deep_gain = bool(deep["statistically_detectable_gain_depths"])
    if not baseline_gain and deep_gain:
        policy_result = "depth_utility_emerged_after_deep_compute_continuation"
    elif not baseline_gain and not deep_gain:
        policy_result = "no_detectable_depth_utility_before_or_after_continuation"
    elif baseline_gain and deep_gain:
        policy_result = "depth_utility_detectable_before_and_after_continuation"
    else:
        policy_result = "baseline_depth_utility_not_retained_after_continuation"
    baseline_deepest = baseline["records"][-1]
    deep_deepest = deep["records"][-1]
    deep_reference = deep["records"][0]
    deep_best = min(deep["records"], key=lambda item: item["cross_entropy"])
    depth_gain = deep_reference["cross_entropy"] - deep_best["cross_entropy"]
    additional_steps = deep_best["depth"] - deep_reference["depth"]
    break_even_ponder = depth_gain / additional_steps if additional_steps > 0 else None
    return {
        "schema_version": 1,
        "scope": "irl3_recurrent_time_mechanism_study",
        "baseline_1000": {
            "fixed_depth": baseline,
            "writeback": _writeback_summary(
                baseline_control_fixed or baseline_fixed,
                baseline_writeback_off,
            ),
            "act_soft": _halting_summary(baseline_act),
            "adaptive_hard": _halting_summary(baseline_adaptive),
        },
        "deep_compute_continuation": {
            "training": _training_summary(deep_metrics, starting_step),
            "fixed_depth": deep,
            "writeback": _writeback_summary(deep_fixed, deep_writeback_off),
            "act_soft": _halting_summary(deep_act),
            "adaptive_hard": _halting_summary(deep_adaptive),
            "halt_threshold_control": _halt_threshold_summary(halt_threshold_reports),
        },
        "mechanistic_tests": {
            "training_policy_result": policy_result,
            "baseline_case_b_later_state_change_without_ce_gain": (
                not baseline_gain
                and (baseline_deepest["z_relative_l2_to_depth_2"] or 0.0) > 0.05
                and (baseline_deepest["correction_relative_l2_to_depth_2"] or 0.0)
                > 0.03
            ),
            "deep_case_b_later_state_change_without_ce_gain": (
                not deep_gain
                and (deep_deepest["z_relative_l2_to_depth_2"] or 0.0) > 0.05
                and (deep_deepest["correction_relative_l2_to_depth_2"] or 0.0) > 0.03
            ),
            "equilibrium_pressure_isolated": False,
            "ordinary_lm_objective_isolated": False,
            "threshold_override_moves_mean_depth": any(
                (row["expected_steps"] or 0.0) > 2.01
                for row in _halt_threshold_summary(halt_threshold_reports)
            ),
            "best_measured_fixed_depth": deep_best["depth"],
            "best_measured_fixed_depth_ce_gain_vs_depth_2": depth_gain,
            "fixed_ce_break_even_ponder_per_extra_step": break_even_ponder,
            "original_ponder_weight": 0.001,
            "original_ponder_to_fixed_ce_break_even_ratio": (
                0.001 / break_even_ponder
                if break_even_ponder is not None and break_even_ponder > 0
                else None
            ),
        },
        "interpretation_limits": [
            "Paired CE confidence intervals quantify sampling error on one fixed "
            "validation range; they do not include checkpoint or data-seed variance.",
            "The deep continuation changes shallow-budget exposure, recurrent "
            "credit horizon, and ponder pressure together, so it tests the combined "
            "training policy rather than attributing effects among those three "
            "factors.",
            "Baseline write-back uses the original same-checkpoint control sweep; "
            "the higher-statistics baseline sweep is used for recurrent-depth and "
            "state comparisons.",
            "Equilibrium regularization and model architecture are unchanged, so "
            "this comparison does not isolate either one.",
            "ACT/adaptive results after fixed-mode continuation diagnose transfer "
            "of the existing halt head; the halt head was not directly trained "
            "during the continuation.",
            "The reported ponder break-even is a scale comparison using fixed-depth "
            "CE only; ACT state marginalization and auxiliary objectives make it an "
            "approximation rather than an optimized stopping threshold.",
        ],
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_1000"]
    deep = summary["deep_compute_continuation"]
    lines = [
        "# IRL-3 recurrent-time mechanism study",
        "",
        "## Measured",
        "",
    ]
    training = deep["training"]
    if training:
        lines.extend(
            [
                "### Deep-compute continuation",
                "",
                f"- Steps: {training['starting_step']} -> "
                f"{training['final_logged_step']} "
                f"({training['additional_steps']} additional)",
                "- Fixed depth / full credit horizon: "
                f"{training['compute_depths']} / {training['gradient_horizons']}",
                "- Ponder / equilibrium / MTP weights: "
                f"{training['ponder_weights']} / "
                f"{training['equilibrium_weights']} / "
                f"{training['mtp_loss_weights']}",
                "- Median warm throughput: "
                f"{_fmt(training['median_warm_tokens_per_second'], 1)} tok/s",
                f"- Peak allocated HBM: {_fmt(training['peak_memory_gib'], 2)} GiB",
                f"- Finite metrics: {training['all_finite']}",
                "",
                "| validation step | CE | total objective | equilibrium loss |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in training["validation_curve"]:
            lines.append(
                f"| {row['step']} | {_fmt(row['cross_entropy'], 8)} | "
                f"{_fmt(row['total_loss'], 8)} | "
                f"{_fmt(row['equilibrium_loss'], 8)} |"
            )
        lines.append("")
    if deep["halt_threshold_control"]:
        lines.extend(
            [
                "### Adaptive halt-threshold control",
                "",
                "| threshold | budget | CE | iterations | expected steps | "
                "diagnostic hard histogram | latency ms | correction/anchor |",
                "|---:|---:|---:|---:|---:|---|---:|---:|",
            ]
        )
        for row in deep["halt_threshold_control"]:
            lines.append(
                f"| {row['threshold']:.8f} | {row['budget']} | "
                f"{_fmt(row['cross_entropy'], 8)} | {row['iterations']} | "
                f"{_fmt(row['expected_steps'], 6)} | "
                f"`{row['diagnostic_step_histogram']}` | "
                f"{_fmt(row['latency_milliseconds'], 2)} | "
                f"{_fmt(row['writeback_to_anchor_ratio'], 6)} |"
            )
        lines.extend(
            [
                "",
                "The histogram is for the first 4,096-prefix diagnostic batch; "
                "CE and latency use all configured evaluation batches.",
                "",
            ]
        )
    for title, block in (
        ("Baseline step-1000 fixed-depth sweep", baseline["fixed_depth"]),
        ("Deep-compute fixed-depth sweep", deep["fixed_depth"]),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| T | CE | paired ΔCE vs T2 | 95% CI | Z Δ/T2 | M Δ/T2 | "
                "correction Δ/T2 | correction cosine | correction/anchor | "
                "latency ms |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in block["records"]:
            paired = row["paired_cross_entropy_delta_to_depth_2"] or {}
            ci = (
                f"[{_fmt(paired.get('confidence_95_low'), 7)}, "
                f"{_fmt(paired.get('confidence_95_high'), 7)}]"
            )
            lines.append(
                f"| {row['depth']} | {_fmt(row['cross_entropy'], 8)} | "
                f"{_fmt(paired.get('mean'), 8)} | {ci} | "
                f"{_fmt(row['z_relative_l2_to_depth_2'], 5)} | "
                f"{_fmt(row['m_relative_l2_to_depth_2'], 5)} | "
                f"{_fmt(row['correction_relative_l2_to_depth_2'], 5)} | "
                f"{_fmt(row['correction_cosine_to_depth_2'], 6)} | "
                f"{_fmt(row['writeback_to_anchor_ratio'], 6)} | "
                f"{_fmt(row['latency_milliseconds'], 2)} |"
            )
        lines.extend(
            [
                "",
                "Detectable gain depths: "
                f"{block['statistically_detectable_gain_depths']}",
                "Detectable degradation depths: "
                f"{block['statistically_detectable_degradation_depths']}",
                "",
            ]
        )
    for title, block in (
        ("Baseline recurrent dynamics", baseline["fixed_depth"]),
        ("Deep-compute recurrent dynamics", deep["fixed_depth"]),
    ):
        selected = [row for row in block["records"] if row["depth"] in {2, 6, 8, 32}]
        lines.extend(
            [
                f"### {title}",
                "",
                "| T | Z/H/M RMS | Z/H/M update RMS | memory read/write/util | "
                "H residual | fast residual/contraction | anchor entropy | "
                "slow ticks |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['depth']} | {_fmt(row['z_rms'], 5)} / "
                f"{_fmt(row['h_rms'], 5)} / {_fmt(row['m_rms'], 5)} | "
                f"{_fmt(row['z_update_rms'], 6)} / "
                f"{_fmt(row['h_update_rms'], 6)} / "
                f"{_fmt(row['m_update_rms'], 6)} | "
                f"{_fmt(row['memory_read_rms'], 6)} / "
                f"{_fmt(row['memory_write_rms'], 6)} / "
                f"{_fmt(row['memory_utilization'], 6)} | "
                f"{_fmt(row['last_meaningful_h_residual'], 8)} | "
                f"{_fmt(row['fast_residual'], 6)} / "
                f"{_fmt(row['fast_contraction'], 6)} | "
                f"{_fmt(row['anchor_attention_entropy'], 6)} | "
                f"{row['slow_updates']} |"
            )
        lines.append("")
    for title, block in (
        ("Baseline write-back", baseline["writeback"]),
        ("Deep-compute write-back", deep["writeback"]),
    ):
        if not block:
            continue
        lines.extend([f"### {title}", ""])
        for row in block["records"]:
            delta = row["disabled_minus_enabled"]
            lines.append(
                f"- T={row['depth']}: OFF−ON CE {_fmt(delta['mean'], 8)} "
                f"(95% CI [{_fmt(delta['confidence_95_low'], 8)}, "
                f"{_fmt(delta['confidence_95_high'], 8)}]); correction/anchor "
                f"{_fmt(row['correction_to_anchor_ratio'], 6)}."
            )
        lines.append("")
    for title, block in (
        ("Baseline", baseline),
        ("Deep-compute", deep),
    ):
        if not block["act_soft"] and not block["adaptive_hard"]:
            continue
        lines.extend(
            [
                f"### {title} halting",
                "",
                "| mode | budget | CE | expected steps | hard histogram | latency ms |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for mode, rows in (
            ("ACT-soft", block["act_soft"]),
            ("adaptive-hard", block["adaptive_hard"]),
        ):
            for row in rows:
                lines.append(
                    f"| {mode} | {row['budget']} | {_fmt(row['cross_entropy'], 8)} | "
                    f"{_fmt(row['expected_steps'], 4)} | `{row['step_counts']}` | "
                    f"{_fmt(row['latency_milliseconds'], 2)} |"
                )
        lines.append("")
    tests = summary["mechanistic_tests"]
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- Training-policy result: `{tests['training_policy_result']}`.",
            "- Baseline is Case B (later states change without a detected CE "
            "gain): "
            f"{tests['baseline_case_b_later_state_change_without_ce_gain']}.",
            "- Deep continuation is Case B by the same operational test: "
            f"{tests['deep_case_b_later_state_change_without_ce_gain']}.",
            "- Raising the adaptive threshold materially moves mean depth: "
            f"{tests['threshold_override_moves_mean_depth']}.",
            "- Best measured fixed budget / CE gain over T=2: "
            f"T={tests['best_measured_fixed_depth']} / "
            f"{_fmt(tests['best_measured_fixed_depth_ce_gain_vs_depth_2'], 8)}.",
            "- Fixed-CE break-even ponder cost per extra step: "
            f"{_fmt(tests['fixed_ce_break_even_ponder_per_extra_step'], 8)}; "
            "the original 0.001 coefficient is approximately "
            f"{_fmt(tests['original_ponder_to_fixed_ce_break_even_ratio'], 1)}x "
            "larger.",
            "- This experiment can support or reject the combined deep-compute "
            "training policy; it cannot separate shallow-budget curriculum, "
            "truncated credit, and ponder pressure from one another.",
            "",
            "## Speculation and unresolved mechanisms",
            "",
            "- Equilibrium pressure was held constant and therefore was not isolated.",
            "- Whether ordinary next-token prediction supplies enough incentive "
            "for iterative computation was not isolated.",
            "- A fixed-mode continuation does not directly retrain the halt head; "
            "halting results test transfer, not a newly optimized stopping policy.",
            "- Raising the cumulative halt threshold alone tests saturation, but "
            "does not retrain the halt projection or change its minimum-depth "
            "training history.",
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in summary["interpretation_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-fixed", type=Path, required=True)
    parser.add_argument("--baseline-control-fixed", type=Path)
    parser.add_argument("--deep-fixed", type=Path, required=True)
    parser.add_argument("--deep-metrics", type=Path, required=True)
    parser.add_argument("--baseline-writeback-off", type=Path)
    parser.add_argument("--deep-writeback-off", type=Path)
    parser.add_argument("--baseline-act", type=Path)
    parser.add_argument("--baseline-adaptive", type=Path)
    parser.add_argument("--deep-act", type=Path)
    parser.add_argument("--deep-adaptive", type=Path)
    parser.add_argument("--halt-threshold-directory", type=Path)
    parser.add_argument("--starting-step", type=int, default=1000)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    baseline_fixed = _json(args.baseline_fixed)
    deep_fixed = _json(args.deep_fixed)
    if baseline_fixed is None or deep_fixed is None:
        raise FileNotFoundError("Both fixed-depth reports are required")
    summary = build_summary(
        baseline_fixed=baseline_fixed,
        deep_fixed=deep_fixed,
        deep_metrics=_jsonl(args.deep_metrics),
        baseline_control_fixed=_json(args.baseline_control_fixed),
        baseline_writeback_off=_json(args.baseline_writeback_off),
        deep_writeback_off=_json(args.deep_writeback_off),
        baseline_act=_json(args.baseline_act),
        baseline_adaptive=_json(args.baseline_adaptive),
        deep_act=_json(args.deep_act),
        deep_adaptive=_json(args.deep_adaptive),
        halt_threshold_reports=[
            report
            for path in sorted(
                (args.halt_threshold_directory or Path("/nonexistent")).glob(
                    "threshold_*.json"
                )
            )
            if (report := _json(path)) is not None
        ],
        starting_step=args.starting_step,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
