#!/usr/bin/env python3
"""Aggregate IRL-3 policy-attribution and data-scaling depth sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return None


def _record(payload: dict[str, Any], depth: int) -> dict[str, Any]:
    return next(item for item in payload["records"] if int(item["depth"]) == depth)


def _window_mean(values: list[float], *, first: bool, size: int = 10) -> float | None:
    if not values:
        return None
    window = values[:size] if first else values[-size:]
    return statistics.fmean(window)


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _curve(path: Path) -> dict[str, Any]:
    payload = _load(path)
    records: list[dict[str, Any]] = []
    for item in payload["records"]:
        paired = item.get("paired_cross_entropy_delta_to_reference", {})
        diagnostics = item.get("diagnostics", {})
        comparison = item.get("state_comparison_to_reference", {})
        records.append(
            {
                "depth": int(item["depth"]),
                "cross_entropy": float(item["cross_entropy_mean"]),
                "delta_ce_to_t2": float(paired.get("mean", 0.0)),
                "delta_ce_standard_error": float(paired.get("standard_error", 0.0)),
                "delta_ce_ci95_low": float(paired.get("confidence_95_low", 0.0)),
                "delta_ce_ci95_high": float(paired.get("confidence_95_high", 0.0)),
                "latency_ms": float(item["median_batch_latency_milliseconds"]),
                "tokens_per_second": float(item["median_tokens_per_second"]),
                "peak_memory_gib": float(item["peak_memory_gib"]),
                "correction_anchor_ratio": _mean(
                    diagnostics, "writeback_to_anchor_ratio"
                ),
                "correction_rms": _mean(diagnostics, "writeback_rms"),
                "z_rms": _mean(diagnostics, "z_rms"),
                "h_rms": _mean(diagnostics, "h_rms"),
                "m_rms": _mean(diagnostics, "m_rms"),
                "z_update_rms": _mean(diagnostics, "z_update_rms"),
                "h_update_rms": _mean(diagnostics, "h_update_rms"),
                "m_update_rms": _mean(diagnostics, "m_update_rms"),
                "memory_read_rms": _mean(diagnostics, "memory_read_rms"),
                "memory_write_rms": _mean(diagnostics, "memory_write_rms"),
                "memory_utilization": _mean(diagnostics, "memory_utilization"),
                "last_h_update_rms": _mean(diagnostics, "last_h_update_rms"),
                "fast_residual": _mean(diagnostics, "fast_residual"),
                "fast_contraction": _mean(diagnostics, "fast_contraction_ratios"),
                "anchor_attention_entropy": _mean(
                    diagnostics, "anchor_attention_entropy"
                ),
                "slow_updates": int(diagnostics.get("slow_updates", 0)),
                "z_relative_l2_to_t2": _mean(comparison, "z_relative_l2"),
                "h_relative_l2_to_t2": _mean(comparison, "h_relative_l2"),
                "m_relative_l2_to_t2": _mean(comparison, "m_relative_l2"),
                "correction_relative_l2_to_t2": _mean(
                    comparison, "correction_relative_l2"
                ),
                "correction_cosine_to_t2": _mean(
                    comparison, "correction_cosine_similarity"
                ),
            }
        )
    records.sort(key=lambda item: item["depth"])
    best = min(records, key=lambda item: item["cross_entropy"])
    detectable_gain = [
        item["depth"]
        for item in records
        if item["depth"] != 2 and item["delta_ce_ci95_high"] < 0.0
    ]
    detectable_harm = [
        item["depth"]
        for item in records
        if item["depth"] != 2 and item["delta_ce_ci95_low"] > 0.0
    ]
    return {
        "path": str(path),
        "evaluated_predicted_tokens": int(payload["evaluated_predicted_tokens"]),
        "records": records,
        "best_depth": int(best["depth"]),
        "best_delta_ce": float(best["delta_ce_to_t2"]),
        "detectable_gain_depths": detectable_gain,
        "detectable_harm_depths": detectable_harm,
        "harmful_depth_boundary": min(detectable_harm) if detectable_harm else None,
    }


def _training_summary(
    root: Path,
    *,
    start_step_exclusive: int | None = None,
    end_step_inclusive: int | None = None,
    gradient_clip_threshold: float = 1.0,
) -> dict[str, Any]:
    metrics_path = root / "metrics.jsonl"
    metrics: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                metrics.append(json.loads(line))
    if start_step_exclusive is not None:
        metrics = [item for item in metrics if int(item["step"]) > start_step_exclusive]
    if end_step_inclusive is not None:
        metrics = [item for item in metrics if int(item["step"]) <= end_step_inclusive]
    warm_metrics = metrics[5:] if len(metrics) > 5 else metrics
    throughputs = [
        float(item["tokens_per_second"])
        for item in warm_metrics
        if item.get("tokens_per_second") is not None
        and math.isfinite(float(item["tokens_per_second"]))
    ]
    cross_entropies = [
        float(item["cross_entropy"])
        for item in metrics
        if item.get("cross_entropy") is not None
        and math.isfinite(float(item["cross_entropy"]))
    ]
    losses = [
        float(item["loss"])
        for item in metrics
        if item.get("loss") is not None and math.isfinite(float(item["loss"]))
    ]
    equilibrium_losses = [
        float(item["equilibrium_loss"])
        for item in metrics
        if item.get("equilibrium_loss") is not None
        and math.isfinite(float(item["equilibrium_loss"]))
    ]
    gradient_norms = [
        float(item["gradient_norm"])
        for item in metrics
        if item.get("gradient_norm") is not None
        and math.isfinite(float(item["gradient_norm"]))
    ]
    analytic_flops = next(
        (
            float(item["analytic_training_flops_per_token"])
            for item in metrics
            if item.get("analytic_training_flops_per_token") is not None
        ),
        None,
    )
    elapsed_path = root / "training_elapsed_seconds.txt"
    elapsed_text = (
        elapsed_path.read_text(encoding="utf-8").strip()
        if elapsed_path.exists()
        else ""
    )
    return {
        "metrics_path": str(metrics_path),
        "logged_points": len(metrics),
        "observed_start_step": int(metrics[0]["step"]) if metrics else None,
        "observed_end_step": int(metrics[-1]["step"]) if metrics else None,
        "observed_start_tokens": int(metrics[0]["tokens_seen"]) if metrics else None,
        "observed_end_tokens": int(metrics[-1]["tokens_seen"]) if metrics else None,
        "final_step": int(metrics[-1]["step"]) if metrics else None,
        "final_tokens": int(metrics[-1]["tokens_seen"]) if metrics else None,
        "final_training_ce": float(metrics[-1]["cross_entropy"]) if metrics else None,
        "median_warm_tokens_per_second": (
            statistics.median(throughputs) if throughputs else None
        ),
        "median_analytic_training_tflops": (
            statistics.median(throughputs) * analytic_flops / 1e12
            if throughputs and analytic_flops is not None
            else None
        ),
        "cross_entropy_first_window_mean": _window_mean(cross_entropies, first=True),
        "cross_entropy_last_window_mean": _window_mean(cross_entropies, first=False),
        "loss_first_window_mean": _window_mean(losses, first=True),
        "loss_last_window_mean": _window_mean(losses, first=False),
        "equilibrium_loss_first_window_mean": _window_mean(
            equilibrium_losses, first=True
        ),
        "equilibrium_loss_last_window_mean": _window_mean(
            equilibrium_losses, first=False
        ),
        "gradient_clip_threshold": gradient_clip_threshold,
        "gradient_norm_median": (
            statistics.median(gradient_norms) if gradient_norms else None
        ),
        "gradient_norm_p95": _quantile(gradient_norms, 0.95),
        "gradient_norm_maximum": max(gradient_norms) if gradient_norms else None,
        "gradient_clip_fraction": (
            sum(value > gradient_clip_threshold for value in gradient_norms)
            / len(gradient_norms)
            if gradient_norms
            else None
        ),
        "peak_allocated_gib": (
            max(float(item.get("memory_peak_gib", 0.0)) for item in metrics)
            if metrics
            else None
        ),
        "peak_reserved_gib": (
            max(float(item.get("memory_reserved_gib", 0.0)) for item in metrics)
            if metrics
            else None
        ),
        "analytic_training_flops_per_token": analytic_flops,
        "elapsed_seconds": int(elapsed_text)
        if elapsed_text.isdigit()
        else elapsed_text,
        "finite": all(
            math.isfinite(float(value))
            for item in metrics
            for key in ("loss", "cross_entropy", "gradient_norm")
            if (value := item.get(key)) is not None
        ),
    }


def _stage_a(artifact_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition, metadata in manifest.get("stage_a", {}).items():
        root = artifact_root / "stage_a" / "conditions" / condition
        splits = {}
        for split in ("mechanism_dev", "confirmation_holdout"):
            path = root / f"{split}_depth_sweep.json"
            if path.exists():
                splits[split] = _curve(path)
        result[condition] = {
            "policy": metadata["policy"],
            "target_step": metadata["target_step"],
            "target_tokens": metadata["target_tokens"],
            "training": _training_summary(root),
            "splits": splits,
        }
    return result


def _stage_b(artifact_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    stage_b = manifest.get("stage_b")
    if not isinstance(stage_b, dict):
        return {}
    result: dict[str, Any] = {
        "policy_name": stage_b["policy_name"],
        "policy": stage_b["policy"],
        "milestones": {},
    }
    checkpoint_metadata = stage_b.get("starting_checkpoint_identity", {}).get(
        "metadata", {}
    )
    previous_step = int(checkpoint_metadata.get("training_state", {}).get("step", 0))
    for label, metadata in stage_b["milestones"].items():
        root = artifact_root / "stage_b" / "milestones" / label
        splits = {}
        for split in ("mechanism_dev", "confirmation_holdout"):
            path = root / f"{split}_depth_sweep.json"
            if path.exists():
                splits[split] = _curve(path)
        if splits:
            result["milestones"][label] = {
                "step": metadata["step"],
                "tokens": metadata["tokens"],
                "elapsed_seconds": (
                    int((root / "training_elapsed_seconds.txt").read_text().strip())
                    if (root / "training_elapsed_seconds.txt").exists()
                    else None
                ),
                "training_segment": _training_summary(
                    artifact_root / "stage_b",
                    start_step_exclusive=previous_step,
                    end_step_inclusive=int(metadata["step"]),
                ),
                "splits": splits,
            }
        previous_step = int(metadata["step"])
    result["training"] = _training_summary(artifact_root / "stage_b")
    return result


def _rows(summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for condition, item in summary["stage_a"].items():
        for split, curve in item["splits"].items():
            for record in curve["records"]:
                yield {
                    "stage": "A",
                    "run": condition,
                    "training_tokens": item["target_tokens"],
                    "split": split,
                    **record,
                }
    stage_b = summary.get("stage_b", {})
    for label, item in stage_b.get("milestones", {}).items():
        for split, curve in item["splits"].items():
            for record in curve["records"]:
                yield {
                    "stage": "B",
                    "run": label,
                    "training_tokens": item["tokens"],
                    "split": split,
                    **record,
                }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scaling_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        training = item.get("training_segment", {})
        for split, curve in item["splits"].items():
            by_depth = {record["depth"]: record for record in curve["records"]}
            reference = by_depth[2]
            best = by_depth[curve["best_depth"]]
            rows.append(
                {
                    "milestone": label,
                    "training_tokens": item["tokens"],
                    "split": split,
                    "t2_cross_entropy": reference["cross_entropy"],
                    "best_depth": curve["best_depth"],
                    "best_delta_ce": curve["best_delta_ce"],
                    "best_delta_ce_ci95_low": best["delta_ce_ci95_low"],
                    "best_delta_ce_ci95_high": best["delta_ce_ci95_high"],
                    "delta_ce_t8": by_depth.get(8, {}).get("delta_ce_to_t2"),
                    "delta_ce_t12": by_depth.get(12, {}).get("delta_ce_to_t2"),
                    "harmful_depth_boundary": curve["harmful_depth_boundary"],
                    "correction_anchor_t2": reference["correction_anchor_ratio"],
                    "correction_anchor_best": best["correction_anchor_ratio"],
                    "z_distance_best_to_t2": best["z_relative_l2_to_t2"],
                    "h_distance_best_to_t2": best["h_relative_l2_to_t2"],
                    "m_distance_best_to_t2": best["m_relative_l2_to_t2"],
                    "correction_distance_best_to_t2": best[
                        "correction_relative_l2_to_t2"
                    ],
                    "correction_cosine_best_to_t2": best["correction_cosine_to_t2"],
                    "median_training_tokens_per_second": training.get(
                        "median_warm_tokens_per_second"
                    ),
                    "gradient_clip_fraction": training.get("gradient_clip_fraction"),
                    "gradient_norm_p95": training.get("gradient_norm_p95"),
                    "training_runtime_seconds": item.get("elapsed_seconds"),
                }
            )
    return rows


def _svg_lines(
    path: Path,
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    x_label: str,
    y_label: str,
) -> None:
    if not series or not any(points for _, points in series):
        return
    width, height = 960, 560
    left, right, top, bottom = 90, 30, 55, 75
    all_points = [point for _, points in series for point in points]
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="30" text-anchor="middle" '
            f'font-size="20">{title}</text>'
        ),
        (
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
            f'y2="{height - bottom}" stroke="#111"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}" stroke="#111"/>'
        ),
    ]
    if y_min <= 0.0 <= y_max:
        parts.append(
            f'<line x1="{left}" y1="{sy(0)}" x2="{width - right}" '
            f'y2="{sy(0)}" stroke="#999" stroke-dasharray="5 4"/>'
        )
    for index, (label, points) in enumerate(series):
        color = colors[index % len(colors)]
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}"/>'
            )
        legend_y = top + 18 * index
        parts.append(
            f'<text x="{width - right - 170}" y="{legend_y}" fill="{color}" '
            f'font-size="13">{label}</text>'
        )
    parts.extend(
        [
            (
                f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" '
                f'font-size="15">{x_label}</text>'
            ),
            (
                f'<text x="20" y="{height / 2}" text-anchor="middle" '
                f'transform="rotate(-90 20 {height / 2})" font-size="15">'
                f"{y_label}</text>"
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _plots(artifact_root: Path, summary: dict[str, Any]) -> None:
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    stage_a_series = []
    for condition, item in summary["stage_a"].items():
        curve = item["splits"].get("confirmation_holdout")
        if curve:
            stage_a_series.append(
                (
                    condition,
                    [
                        (record["depth"], 1e6 * record["delta_ce_to_t2"])
                        for record in curve["records"]
                    ],
                )
            )
    _svg_lines(
        plot_root / "stage_a_delta_ce_vs_depth.svg",
        "Stage A confirmation holdout",
        stage_a_series,
        "fixed recurrent depth",
        "delta CE vs T=2 (micro-nats)",
    )

    stage_b_series = []
    stage_b_ce_series = []
    delta8_points: list[tuple[float, float]] = []
    optimal_depth_points: list[tuple[float, float]] = []
    correction_points: list[tuple[float, float]] = []
    training_ce_points: list[tuple[float, float]] = []
    clip_fraction_points: list[tuple[float, float]] = []
    final_holdout: dict[str, Any] | None = None
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        curve = item["splits"].get("confirmation_holdout")
        if not curve:
            continue
        final_holdout = curve
        stage_b_series.append(
            (
                label,
                [
                    (record["depth"], 1e6 * record["delta_ce_to_t2"])
                    for record in curve["records"]
                ],
            )
        )
        stage_b_ce_series.append(
            (
                label,
                [
                    (record["depth"], record["cross_entropy"])
                    for record in curve["records"]
                ],
            )
        )
        by_depth = {record["depth"]: record for record in curve["records"]}
        tokens = float(item["tokens"])
        if 8 in by_depth:
            delta8_points.append((tokens, 1e6 * by_depth[8]["delta_ce_to_t2"]))
        optimal_depth_points.append((tokens, float(curve["best_depth"])))
        correction_points.append(
            (tokens, 100.0 * by_depth[2]["correction_anchor_ratio"])
        )
        training = item.get("training_segment", {})
        if training.get("cross_entropy_last_window_mean") is not None:
            training_ce_points.append(
                (tokens, training["cross_entropy_last_window_mean"])
            )
        if training.get("gradient_clip_fraction") is not None:
            clip_fraction_points.append(
                (tokens, 100.0 * training["gradient_clip_fraction"])
            )

    _svg_lines(
        plot_root / "stage_b_delta_ce_vs_depth.svg",
        "Stage B confirmation holdout",
        stage_b_series,
        "fixed recurrent depth",
        "delta CE vs T=2 (micro-nats)",
    )
    _svg_lines(
        plot_root / "stage_b_ce_vs_depth.svg",
        "Stage B confirmation holdout cross-entropy",
        stage_b_ce_series,
        "fixed recurrent depth",
        "cross-entropy (nats)",
    )
    _svg_lines(
        plot_root / "stage_b_delta_ce8_vs_tokens.svg",
        "T=8 recurrent utility versus training tokens",
        [("holdout", delta8_points)],
        "cumulative training tokens",
        "delta CE(T=8 vs T=2), micro-nats",
    )
    _svg_lines(
        plot_root / "stage_b_optimal_depth_vs_tokens.svg",
        "Optimal evaluated depth versus training tokens",
        [("holdout", optimal_depth_points)],
        "cumulative training tokens",
        "best evaluated depth",
    )
    _svg_lines(
        plot_root / "stage_b_correction_anchor_vs_tokens.svg",
        "Write-back correction / anchor versus training tokens",
        [("T=2 holdout", correction_points)],
        "cumulative training tokens",
        "correction / anchor RMS (%)",
    )
    _svg_lines(
        plot_root / "stage_b_training_ce_vs_tokens.svg",
        "Smoothed terminal training CE versus training tokens",
        [("last 10 logged points", training_ce_points)],
        "cumulative training tokens",
        "training cross-entropy (nats)",
    )
    _svg_lines(
        plot_root / "stage_b_clip_fraction_vs_tokens.svg",
        "Gradient clipping frequency by training segment",
        [("pre-clip norm > 1", clip_fraction_points)],
        "cumulative training tokens",
        "logged batches clipped (%)",
    )
    if final_holdout is not None:
        distances = [
            (
                label,
                [
                    (record["depth"], record[key])
                    for record in final_holdout["records"]
                    if record[key] is not None
                ],
            )
            for label, key in (
                ("Z", "z_relative_l2_to_t2"),
                ("H", "h_relative_l2_to_t2"),
                ("M", "m_relative_l2_to_t2"),
                ("correction", "correction_relative_l2_to_t2"),
            )
        ]
        _svg_lines(
            plot_root / "stage_b_final_state_distance_vs_depth.svg",
            "Final-checkpoint state distance from T=2",
            distances,
            "fixed recurrent depth",
            "relative L2 distance",
        )


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# IRL-3 recurrent-utility study",
        "",
        f"Experiment execution revision: `{summary['execution_revision']}`. "
        f"Manifest-generation revision: `{summary['source_revision']}`. "
        f"Post-hoc analysis revision: "
        f"`{summary['analysis_revision'] or 'uncommitted'}`.",
        "",
        "All depth comparisons use the same checkpoint within a row and paired "
        "token losses on deterministic, non-overlapping evaluation data. The "
        "mechanism/dev and confirmation-holdout sweeps each contain 819,000 "
        "predicted tokens. Confidence intervals are paired 95% intervals.",
        "",
        "## Stage A: training-policy attribution",
        "",
        "D changes sampled shallow/ACT-soft training to fixed-depth-6 terminal-state "
        "training. C changes recurrent credit from horizon 4 to full horizon 6. "
        "P removes the 1e-3 ponder coefficient.",
        "",
    ]
    lines.append(
        "| condition | D | C | P | tokens | runtime s | tok/s | holdout T* | "
        "best delta CE [95% CI] | significant gains | harmful boundary |"
    )
    lines.append("|---|:---:|:---:|:---:|---:|---:|---:|---:|---|---|---:|")
    for condition, item in summary["stage_a"].items():
        curve = item["splits"].get("confirmation_holdout")
        training = item["training"]
        if not curve:
            continue
        policy = item["policy"]
        best = next(
            record
            for record in curve["records"]
            if record["depth"] == curve["best_depth"]
        )
        elapsed = (
            "reused" if condition == "positive_control" else training["elapsed_seconds"]
        )
        throughput = (
            "reused"
            if condition == "positive_control"
            else f"{training['median_warm_tokens_per_second']:.0f}"
        )
        lines.append(
            f"| {condition} | {int(policy['fixed_depth'])} | "
            f"{int(policy['full_credit'])} | {int(policy['ponder_weight'] == 0.0)} | "
            f"{item['target_tokens']} | {elapsed} | {throughput} | "
            f"{curve['best_depth']} | {curve['best_delta_ce']:+.8f} "
            f"[{best['delta_ce_ci95_low']:+.8f}, {best['delta_ce_ci95_high']:+.8f}] | "
            f"{curve['detectable_gain_depths']} | {curve['harmful_depth_boundary']} |"
        )

    lines.extend(["", "### Stage A holdout depth curves", ""])
    lines.append(
        "| condition | T2 CE | T4 delta [CI] | T6 delta [CI] | T8 delta [CI] | "
        "T12 delta [CI] | T16 delta [CI] |"
    )
    lines.append("|---|---:|---|---|---|---|---|")
    for condition, item in summary["stage_a"].items():
        curve = item["splits"].get("confirmation_holdout")
        if not curve:
            continue
        by_depth = {record["depth"]: record for record in curve["records"]}

        def cell(depth: int) -> str:
            record = by_depth[depth]
            return (
                f"{record['delta_ce_to_t2']:+.8f} "
                f"[{record['delta_ce_ci95_low']:+.8f}, "
                f"{record['delta_ce_ci95_high']:+.8f}]"
            )

        lines.append(
            f"| {condition} | {by_depth[2]['cross_entropy']:.8f} | "
            f"{cell(4)} | {cell(6)} | {cell(8)} | {cell(12)} | {cell(16)} |"
        )

    lines.extend(
        [
            "",
            "**Stage A inference.** D is sufficient and is the only single-factor "
            "condition that produces a replicated useful-depth curve. C alone and "
            "P alone are not sufficient. D+C+P reproduces utility but is weaker than "
            "D alone, so full credit and/or zero ponder are not required and may "
            "interact negatively in the fixed-depth regime. D still conflates fixed "
            "budget exposure with fixed terminal-state supervision versus ACT-soft "
            "state marginalization; this study does not separate those two subfactors.",
            "",
            "## Stage B: moderate data scaling",
            "",
        ]
    )
    lines.append(
        "| milestone | tokens | T2 CE | holdout T* | best delta CE [95% CI] | "
        "delta CE8 | delta CE12 | harmful boundary | correction/anchor T2 |"
    )
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        curve = item["splits"].get("confirmation_holdout")
        if not curve:
            continue
        by_depth = {record["depth"]: record for record in curve["records"]}
        best = by_depth[curve["best_depth"]]
        delta8 = by_depth.get(8, {}).get("delta_ce_to_t2", float("nan"))
        delta12 = by_depth.get(12, {}).get("delta_ce_to_t2", float("nan"))
        lines.append(
            f"| {label} | {item['tokens']} | {by_depth[2]['cross_entropy']:.8f} | "
            f"{curve['best_depth']} | {curve['best_delta_ce']:+.8f} "
            f"[{best['delta_ce_ci95_low']:+.8f}, {best['delta_ce_ci95_high']:+.8f}] | "
            f"{delta8:+.8f} | {delta12:+.8f} | "
            f"{curve['harmful_depth_boundary']} | "
            f"{100.0 * by_depth[2]['correction_anchor_ratio']:.3f}% |"
        )

    lines.extend(["", "### Stage B training and hardware behavior", ""])
    lines.append(
        "| segment end | runtime h | tok/s | analytic TFLOP/s | terminal train CE | "
        "clip fraction | grad median / p95 / max | peak allocated / reserved GiB |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        training = item.get("training_segment", {})
        lines.append(
            f"| {label} | {item['elapsed_seconds'] / 3600:.3f} | "
            f"{training['median_warm_tokens_per_second']:.0f} | "
            f"{training['median_analytic_training_tflops']:.2f} | "
            f"{training['cross_entropy_last_window_mean']:.5f} | "
            f"{100.0 * training['gradient_clip_fraction']:.1f}% | "
            f"{training['gradient_norm_median']:.3f} / "
            f"{training['gradient_norm_p95']:.3f} / "
            f"{training['gradient_norm_maximum']:.3f} | "
            f"{training['peak_allocated_gib']:.2f} / "
            f"{training['peak_reserved_gib']:.2f} |"
        )

    lines.extend(["", "### Stage B recurrent dynamics", ""])
    lines.append(
        "| milestone | depth | corr/anchor | Z/H/M update RMS | memory util | "
        "Z/H/M distance to T2 | correction distance / cosine |"
    )
    lines.append("|---|---:|---:|---|---:|---|---|")
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        curve = item["splits"].get("confirmation_holdout")
        if not curve:
            continue
        by_depth = {record["depth"]: record for record in curve["records"]}
        for depth in sorted({2, curve["best_depth"], max(by_depth)}):
            record = by_depth[depth]
            lines.append(
                f"| {label} | {depth} | "
                f"{100.0 * record['correction_anchor_ratio']:.3f}% | "
                f"{record['z_update_rms']:.4f} / {record['h_update_rms']:.4f} / "
                f"{record['m_update_rms']:.4f} | "
                f"{100.0 * record['memory_utilization']:.2f}% | "
                f"{record['z_relative_l2_to_t2']:.3f} / "
                f"{record['h_relative_l2_to_t2']:.3f} / "
                f"{record['m_relative_l2_to_t2']:.3f} | "
                f"{record['correction_relative_l2_to_t2']:.3f} / "
                f"{record['correction_cosine_to_t2']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Measured",
            "",
            "- Fixed-depth terminal-state exposure is sufficient for recurrent-time "
            "utility; full credit alone and zero ponder alone are not.",
            "- Peak holdout utility grows from the 49M fixed-depth-only checkpoint "
            "through 100M, 200M, 350M, and 500M. At 500M, T6 improves CE by "
            "0.00151873 nats relative to T2.",
            "- The final useful evaluated window is T4-T12. T16, T24, and T32 are "
            "significantly harmful. The optimum remains near T6 (T6/T8 are nearly "
            "tied at 200M).",
            "- Z/H/M and the correction continue changing at harmful depths. At the "
            "500M holdout T32, Z/H/M relative distances are about "
            "0.720/0.471/0.510 and correction cosine to T2 is about 0.715.",
            "- Correction/anchor at T2 rises from about 5.47% at the 49M "
            "fixed-depth-only checkpoint to 8.85% at 500M. Prior qualification "
            "context (not remeasured here) found that disabling write-back worsened "
            "CE by about 0.001998 at the earlier positive-control checkpoint.",
            "- Training is stable in BF16 but becomes increasingly gradient-clip "
            "limited under the constant 6e-5 continuation schedule.",
            "",
            "### Inference",
            "",
            "- The original shallow sampled/ACT-soft policy compresses useful work "
            "into early recurrence. Giving a fixed terminal depth teaches later "
            "states to affect prediction.",
            "- More data strengthens peak recurrent utility and makes the recurrent "
            "computer more active, but it does not expand the useful window beyond "
            "T12. Later state change is computation-active but prediction-harmful, "
            "rather than an early fixed-point collapse.",
            "- These results justify a bounded parameter-scaling pilot, but not an "
            "unqualified large-scale pretraining commitment. Replication and better "
            "control of the useful-depth objective should precede a major run.",
            "",
            "### Speculation and next experiment",
            "",
            "The highest-value next attribution is to split D into (1) fixed-budget "
            "exposure and (2) fixed terminal-state supervision versus ACT-soft "
            "marginalization, using two matched continuations from the same "
            "checkpoint. "
            "Then train the winning policy on a depth support concentrated on "
            "T6/T8/T12, with a learning-rate schedule that reduces persistent "
            "clipping, "
            "and test whether the T16 overthinking boundary moves. Only after that "
            "replicates should a roughly 500M-parameter pilot be considered; a 1B "
            "parameter commitment is premature.",
            "",
            "Raw paired intervals and all diagnostics are in `summary.json`, "
            "`depth_curves.csv`, and `scaling_summary.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(artifact_root: Path) -> dict[str, Any]:
    manifest = _load(artifact_root / "experiment_manifest.json")
    summary = {
        "format_version": 1,
        "source_revision": manifest["source_revision"],
        "execution_revision": manifest.get(
            "experiment_execution_revision", manifest["source_revision"]
        ),
        "analysis_revision": manifest.get("analysis_revision"),
        "stage_a": _stage_a(artifact_root, manifest),
        "stage_b": _stage_b(artifact_root, manifest),
    }
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(_rows(summary))
    _write_csv(artifact_root / "depth_curves.csv", rows)
    _write_csv(artifact_root / "scaling_summary.csv", _scaling_rows(summary))
    (artifact_root / "REPORT.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    _plots(artifact_root, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("qualification_artifacts/irl3_recurrent_utility"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = summarize(args.artifact_root.resolve())
    print(
        json.dumps(
            {"stage_a": list(output["stage_a"]), "stage_b": list(output["stage_b"])}
        )
    )
