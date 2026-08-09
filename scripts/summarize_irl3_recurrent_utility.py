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


def _training_summary(root: Path) -> dict[str, Any]:
    metrics_path = root / "metrics.jsonl"
    metrics: list[dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                metrics.append(json.loads(line))
    throughputs = [
        float(item["tokens_per_second"])
        for item in metrics[5:]
        if item.get("tokens_per_second") is not None
        and math.isfinite(float(item["tokens_per_second"]))
    ]
    elapsed_path = root / "training_elapsed_seconds.txt"
    elapsed_text = (
        elapsed_path.read_text(encoding="utf-8").strip()
        if elapsed_path.exists()
        else ""
    )
    return {
        "metrics_path": str(metrics_path),
        "final_step": int(metrics[-1]["step"]) if metrics else None,
        "final_tokens": int(metrics[-1]["tokens_seen"]) if metrics else None,
        "final_training_ce": float(metrics[-1]["cross_entropy"]) if metrics else None,
        "median_warm_tokens_per_second": (
            statistics.median(throughputs) if throughputs else None
        ),
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
                "splits": splits,
            }
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
        "ΔCE vs T=2 (micro-nats)",
    )
    stage_b_series = []
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        curve = item["splits"].get("confirmation_holdout")
        if curve:
            stage_b_series.append(
                (
                    label,
                    [
                        (record["depth"], 1e6 * record["delta_ce_to_t2"])
                        for record in curve["records"]
                    ],
                )
            )
    _svg_lines(
        plot_root / "stage_b_delta_ce_vs_depth.svg",
        "Stage B confirmation holdout",
        stage_b_series,
        "fixed recurrent depth",
        "ΔCE vs T=2 (micro-nats)",
    )


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# IRL-3 recurrent-utility study", "", "## Stage A", ""]
    lines.append(
        "| condition | holdout T* | best ΔCE | significant gains | "
        "harmful boundary | tok/s | runtime s |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for condition, item in summary["stage_a"].items():
        curve = item["splits"].get("confirmation_holdout")
        training = item["training"]
        if not curve:
            continue
        lines.append(
            f"| {condition} | {curve['best_depth']} | {curve['best_delta_ce']:.8f} | "
            f"{curve['detectable_gain_depths']} | {curve['harmful_depth_boundary']} | "
            f"{training['median_warm_tokens_per_second']} | "
            f"{training['elapsed_seconds']} |"
        )
    lines.extend(["", "## Stage B", ""])
    lines.append(
        "| milestone | tokens | holdout T* | ΔCE8 | ΔCE12 | harmful boundary |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, item in summary.get("stage_b", {}).get("milestones", {}).items():
        curve = item["splits"].get("confirmation_holdout")
        if not curve:
            continue
        by_depth = {record["depth"]: record for record in curve["records"]}
        delta8 = by_depth.get(8, {}).get("delta_ce_to_t2", float("nan"))
        delta12 = by_depth.get(12, {}).get("delta_ce_to_t2", float("nan"))
        lines.append(
            f"| {label} | {item['tokens']} | {curve['best_depth']} | "
            f"{delta8:.8f} | {delta12:.8f} | {curve['harmful_depth_boundary']} |"
        )
    lines.extend(
        [
            "",
            "Tiny paired effects should be interpreted with their confidence "
            "intervals in `summary.json` and `depth_curves.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(artifact_root: Path) -> dict[str, Any]:
    manifest = _load(artifact_root / "experiment_manifest.json")
    summary = {
        "format_version": 1,
        "source_revision": manifest["source_revision"],
        "stage_a": _stage_a(artifact_root, manifest),
        "stage_b": _stage_b(artifact_root, manifest),
    }
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = list(_rows(summary))
    _write_csv(artifact_root / "depth_curves.csv", rows)
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
