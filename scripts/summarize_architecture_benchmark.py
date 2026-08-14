#!/usr/bin/env python3
# ruff: noqa: E501
"""Summarize controlled architecture benchmark evidence and render SVG plots."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

MODEL_ORDER = ("transformer", "irl2", "irl_core")
MODEL_LABELS = {
    "transformer": "Transformer",
    "irl2": "IRL-2",
    "irl_core": "IRL Core",
}
COLORS = {
    "transformer": "#3b82f6",
    "irl2": "#f59e0b",
    "irl_core": "#10b981",
}
PAIRWISE = (
    ("irl_core", "transformer", "Core-Transformer"),
    ("irl2", "transformer", "IRL2-Transformer"),
    ("irl_core", "irl2", "Core-IRL2"),
)


def read_json(path: Path) -> dict[str, Any]:
    """Load one benchmark JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records."""

    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write stable machine-readable rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_ci(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Return a paired batch-mean difference and normal 95% interval."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Paired CE arrays must have the same length >= 2")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    mean = statistics.fmean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    half_width = statistics.NormalDist().inv_cdf(0.975) * standard_error
    return {
        "delta_ce": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "paired_batches": len(differences),
        "standard_error": standard_error,
    }


def _interpolate_tokens(
    trajectory: Sequence[tuple[float, float]],
    target_ce: float,
) -> float | None:
    """Linearly interpolate only inside an observed CE segment."""

    points = sorted(trajectory)
    for (left_tokens, left_ce), (right_tokens, right_ce) in zip(points, points[1:]):
        low, high = sorted((left_ce, right_ce))
        if low <= target_ce <= high and left_ce != right_ce:
            fraction = (target_ce - left_ce) / (right_ce - left_ce)
            return left_tokens + fraction * (right_tokens - left_tokens)
        if target_ce == left_ce:
            return left_tokens
    if points and target_ce == points[-1][1]:
        return points[-1][0]
    return None


def _trajectory_area(points: Sequence[tuple[float, float]]) -> float:
    """Return mean CE over the measured token interval by trapezoid rule."""

    ordered = sorted(points)
    if len(ordered) < 2:
        return float("nan")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:])
    )
    return area / (ordered[-1][0] - ordered[0][0])


def _scale(
    value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    """Map one numeric coordinate into SVG space."""

    if source_max == source_min:
        return (target_min + target_max) / 2.0
    fraction = (value - source_min) / (source_max - source_min)
    return target_min + fraction * (target_max - target_min)


def line_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, str, Sequence[tuple[float, float]]]],
    x_formatter: Callable[[float], str] = lambda value: f"{value:g}",
    y_formatter: Callable[[float], str] = lambda value: f"{value:.4f}",
    zero_line: bool = False,
) -> None:
    """Render a compact dependency-free SVG line chart."""

    width, height = 900, 540
    left, right, top, bottom = 90, 30, 55, 75
    points = [point for _, _, values in series for point in values]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_padding = max(1e-6, (y_max - y_min) * 0.12)
    y_min -= y_padding
    y_max += y_padding
    if zero_line:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    plot_right, plot_bottom = width - right, height - bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5
        x_value = x_min + fraction * (x_max - x_min)
        x = _scale(x_value, x_min, x_max, left, plot_right)
        elements.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(x_formatter(x_value))}</text>'
        )
        y_value = y_min + fraction * (y_max - y_min)
        y = _scale(y_value, y_min, y_max, plot_bottom, top)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(y_formatter(y_value))}</text>'
        )
    if zero_line and y_min <= 0.0 <= y_max:
        y = _scale(0.0, y_min, y_max, plot_bottom, top)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#111827" stroke-dasharray="5 4"/>'
        )
    for label, color, values in series:
        coordinates = " ".join(
            f"{_scale(x, x_min, x_max, left, plot_right):.2f},{_scale(y, y_min, y_max, plot_bottom, top):.2f}"
            for x, y in values
        )
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x_value, y_value in values:
            x = _scale(x_value, x_min, x_max, left, plot_right)
            y = _scale(y_value, y_min, y_max, plot_bottom, top)
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        legend_y = top + 20 * len([item for item in elements if "legend-swatch" in item])
        elements.append(
            f'<line class="legend-swatch" x1="{plot_right - 145}" y1="{legend_y}" x2="{plot_right - 120}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{plot_right - 112}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="#111827"/>',
            f'<text x="{(left + plot_right) / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{(top + plot_bottom) / 2}" text-anchor="middle" transform="rotate(-90 20 {(top + plot_bottom) / 2})" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def bar_svg(
    path: Path,
    *,
    title: str,
    y_label: str,
    values: Sequence[tuple[str, str, float]],
    formatter: Callable[[float], str],
) -> None:
    """Render a compact dependency-free SVG bar chart."""

    width, height = 760, 500
    left, right, top, bottom = 90, 30, 55, 80
    plot_right, plot_bottom = width - right, height - bottom
    maximum = max(value for _, _, value in values) * 1.12
    bar_space = (plot_right - left) / len(values)
    bar_width = bar_space * 0.58
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]
    for index in range(6):
        value = maximum * index / 5
        y = _scale(value, 0.0, maximum, plot_bottom, top)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{html.escape(formatter(value))}</text>')
    for index, (label, color, value) in enumerate(values):
        x = left + index * bar_space + (bar_space - bar_width) / 2
        y = _scale(value, 0.0, maximum, plot_bottom, top)
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{plot_bottom - y:.2f}" fill="{color}"/>')
        elements.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(formatter(value))}</text>')
        elements.append(f'<text x="{x + bar_width / 2:.2f}" y="{plot_bottom + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
    elements.extend(
        [
            f'<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="#111827"/>',
            f'<text x="20" y="{(top + plot_bottom) / 2}" text-anchor="middle" transform="rotate(-90 20 {(top + plot_bottom) / 2})" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _training_stage_seconds(events: Sequence[dict[str, Any]]) -> dict[tuple[str, int], float]:
    """Accumulate completed training subprocess wall time by milestone."""

    cumulative = {model: 0.0 for model in MODEL_ORDER}
    result: dict[tuple[str, int], float] = {}
    finished = [
        event
        for event in events
        if event.get("event") == "stage_finished"
        and event.get("stage") == "train"
        and event.get("returncode") == 0
    ]
    for event in finished:
        model = event["model"]
        cumulative[model] += float(event["elapsed_seconds"])
        result[(model, int(event["milestone_step"]))] = cumulative[model]
    return result


def summarize(root: Path) -> dict[str, Any]:
    """Create all compact tables, decisions, plots, and a markdown report."""

    preflight = read_json(root / "preflight.json")
    qualifications = {
        model: read_json(root / "qualification" / f"{model}.json")
        for model in MODEL_ORDER
    }
    evaluations: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((root / "evaluation").glob("*.json")):
        record = read_json(path)
        evaluations[(record["model"], int(record["checkpoint_step"]))] = record
    expected = {
        (model, int(milestone["step"]))
        for model in MODEL_ORDER
        for milestone in preflight["milestones"]
    }
    missing = expected - set(evaluations)
    if missing:
        raise RuntimeError(f"Missing milestone evaluations: {sorted(missing)}")

    training_records: dict[str, list[dict[str, Any]]] = {}
    training_curve_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        records = read_jsonl(root / "training" / f"{model}.jsonl")
        training_records[model] = records
        training_curve_rows.extend({"model": model, **record} for record in records)
    write_csv(root / "tables" / "training_curves.csv", training_curve_rows)

    events = read_jsonl(root / "orchestration" / "events.jsonl")
    wall_seconds = _training_stage_seconds(events)
    report_by_model = preflight["model_reports"]
    milestone_rows: list[dict[str, Any]] = []
    clipping_rows: list[dict[str, Any]] = []
    hardware_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        logs = training_records[model]
        cumulative_clipped_steps = 0.0
        previous_step = 0
        clipped_by_step: dict[int, float] = {}
        for record in logs:
            step = int(record["step"])
            delta_steps = step - previous_step
            if delta_steps <= 0:
                continue
            cumulative_clipped_steps += float(record["clipped_step_fraction"]) * delta_steps
            clipped_by_step[step] = cumulative_clipped_steps / step
            previous_step = step

        elapsed_seen = float("inf")
        steady_rates = []
        for record in logs:
            elapsed = float(record["elapsed_seconds"])
            if elapsed < elapsed_seen:
                elapsed_seen = elapsed
                continue
            elapsed_seen = elapsed
            rate = float(record["tokens_per_second"])
            if rate > 0.0:
                steady_rates.append(rate)
        training_tps = statistics.median(steady_rates) if steady_rates else float("nan")
        final_step = int(preflight["milestones"][-1]["step"])
        final_eval = evaluations[(model, final_step)]
        final_wall = wall_seconds[(model, final_step)]
        final_tokens = int(final_eval["training_tokens"])
        model_logs_to_final = [record for record in logs if int(record["step"]) <= final_step]
        max_peak = max(float(record["memory_peak_gib"]) for record in model_logs_to_final)
        max_reserved = max(float(record["memory_reserved_gib"]) for record in model_logs_to_final)
        max_allocated = max(float(record["memory_allocated_gib"]) for record in model_logs_to_final)
        qualification = qualifications[model]
        hardware_rows.append(
            {
                "model": model,
                "training_tokens_per_second_median": training_tps,
                "end_to_end_tokens_per_second": final_tokens / final_wall,
                "training_wall_seconds": final_wall,
                "peak_allocated_hbm_gib": max_peak,
                "max_logged_allocated_hbm_gib": max_allocated,
                "max_logged_reserved_hbm_gib": max_reserved,
                "qualification_steady_tokens_per_second": qualification["steady_tokens_per_second"],
                "compilation_cold_start_seconds_estimate": qualification["compilation_and_cold_start_seconds_estimate"],
                "median_forward_seconds": qualification["median_steady_forward_seconds"],
                "median_backward_seconds": qualification["median_steady_backward_seconds"],
                "median_optimizer_seconds": qualification["median_steady_optimizer_seconds"],
            }
        )

        for milestone in preflight["milestones"]:
            step = int(milestone["step"])
            evaluation = evaluations[(model, step)]
            matching_logs = [record for record in logs if int(record["step"]) == step]
            if not matching_logs:
                raise RuntimeError(f"No exact training metric for {model} step {step}")
            train_metric = matching_logs[-1]
            milestone_rows.append(
                {
                    "model": model,
                    "step": step,
                    "training_tokens": evaluation["training_tokens"],
                    "train_cross_entropy": train_metric["cross_entropy"],
                    "dev_cross_entropy": evaluation["dev"]["cross_entropy"],
                    "holdout_cross_entropy": evaluation["holdout"]["cross_entropy"],
                    "parameters": report_by_model[model]["total_parameters"],
                    "learning_rate": train_metric["learning_rate"],
                    "cumulative_clipped_step_fraction": clipped_by_step[step],
                    "training_wall_seconds": wall_seconds[(model, step)],
                    "estimated_training_flops_per_token": report_by_model[model]["estimated_training_flops_per_token"],
                    "cumulative_estimated_training_flops": evaluation["training_tokens"] * report_by_model[model]["estimated_training_flops_per_token"],
                }
            )
            clipping_rows.append(
                {
                    "model": model,
                    "step": step,
                    "training_tokens": evaluation["training_tokens"],
                    "cumulative_clipped_step_fraction": clipped_by_step[step],
                    "last_window_pre_clip_gradient_norm": train_metric["pre_clip_gradient_norm"],
                }
            )
            for depth, values in evaluation["depth_curve_dev"].items():
                depth_rows.append(
                    {
                        "model": model,
                        "step": step,
                        "training_tokens": evaluation["training_tokens"],
                        "depth": int(depth),
                        "dev_cross_entropy": values["cross_entropy"],
                    }
                )

    write_csv(root / "tables" / "milestone_ce.csv", milestone_rows)
    write_csv(root / "tables" / "depth_curves.csv", depth_rows)
    write_csv(root / "tables" / "hardware_metrics.csv", hardware_rows)
    write_csv(root / "tables" / "clipping_statistics.csv", clipping_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for milestone in preflight["milestones"]:
        step = int(milestone["step"])
        for left, right, label in PAIRWISE:
            left_values = evaluations[(left, step)]["holdout"]["batch_cross_entropy"]
            right_values = evaluations[(right, step)]["holdout"]["batch_cross_entropy"]
            interval = paired_ci(left_values, right_values)
            pairwise_rows.append(
                {
                    "comparison": label,
                    "left_model": left,
                    "right_model": right,
                    "step": step,
                    "training_tokens": evaluations[(left, step)]["training_tokens"],
                    **interval,
                    "significant_at_95pct": interval["ci95_high"] < 0.0 or interval["ci95_low"] > 0.0,
                }
            )
    write_csv(root / "tables" / "pairwise_comparisons.csv", pairwise_rows)

    final_step = int(preflight["milestones"][-1]["step"])
    final_pairs = {row["comparison"]: row for row in pairwise_rows if row["step"] == final_step}
    trajectories = {
        model: [
            (float(row["training_tokens"]), float(row["holdout_cross_entropy"]))
            for row in milestone_rows
            if row["model"] == model
        ]
        for model in MODEL_ORDER
    }
    transformer_target = trajectories["transformer"][-1][1]
    token_efficiency: dict[str, Any] = {}
    for model in ("irl2", "irl_core"):
        estimated_tokens = _interpolate_tokens(trajectories[model], transformer_target)
        token_efficiency[model] = {
            "target_ce": transformer_target,
            "estimated_tokens": estimated_tokens,
            "token_saving_fraction": (
                None
                if estimated_tokens is None
                else 1.0 - estimated_tokens / trajectories["transformer"][-1][0]
            ),
            "interpolation_supported": estimated_tokens is not None,
        }

    depth_lookup = {
        (row["model"], row["step"], row["depth"]): row["dev_cross_entropy"]
        for row in depth_rows
    }
    recurrent_utility = []
    for milestone in preflight["milestones"]:
        step = int(milestone["step"])
        recurrent_utility.append(
            {
                "step": step,
                "training_tokens": milestone["tokens"],
                "ce6_minus_ce1": depth_lookup[("irl_core", step, 6)] - depth_lookup[("irl_core", step, 1)],
                "ce6_minus_ce2": depth_lookup[("irl_core", step, 6)] - depth_lookup[("irl_core", step, 2)],
            }
        )

    parameter_ratio = report_by_model["irl_core"]["total_parameters"] / report_by_model["transformer"]["total_parameters"]
    flop_ratio = report_by_model["irl_core"]["estimated_training_flops_per_token"] / report_by_model["transformer"]["estimated_training_flops_per_token"]
    final_hardware = {row["model"]: row for row in hardware_rows}
    wall_ratio = final_hardware["irl_core"]["training_wall_seconds"] / final_hardware["transformer"]["training_wall_seconds"]
    core_transformer = final_pairs["Core-Transformer"]
    core_irl2 = final_pairs["Core-IRL2"]
    irl2_transformer = final_pairs["IRL2-Transformer"]

    def comparison_outcome(values: dict[str, Any]) -> str:
        if values["ci95_high"] < 0.0:
            return "left_better"
        if values["ci95_low"] > 0.0:
            return "right_better"
        return "not_resolved"

    core_outcome = comparison_outcome(core_transformer)
    irl2_outcome = comparison_outcome(irl2_transformer)
    core_irl2_outcome = comparison_outcome(core_irl2)
    utility_emerged = (
        recurrent_utility[-1]["ce6_minus_ce1"] < 0.0
        and recurrent_utility[-1]["ce6_minus_ce2"] < 0.0
        and recurrent_utility[-1]["ce6_minus_ce1"] < recurrent_utility[0]["ce6_minus_ce1"]
    )
    if core_outcome == "left_better" and core_irl2_outcome == "left_better" and utility_emerged:
        scaling_decision = "supports_500m_candidate"
    elif core_outcome == "left_better":
        scaling_decision = "qualified_support_with_compute_caveat"
    else:
        scaling_decision = "do_not_proceed_directly"

    summary = {
        "schema_version": 1,
        "source_commit": preflight["source_commit"],
        "final_step": final_step,
        "final_training_tokens": evaluations[("transformer", final_step)]["training_tokens"],
        "parameter_ratio_core_to_transformer": parameter_ratio,
        "flops_per_token_ratio_core_to_transformer": flop_ratio,
        "wall_clock_ratio_core_to_transformer": wall_ratio,
        "final_pairwise": final_pairs,
        "outcomes": {
            "core_vs_transformer": core_outcome,
            "irl2_vs_transformer": irl2_outcome,
            "core_vs_irl2": core_irl2_outcome,
            "core_recurrent_utility_emerged": utility_emerged,
            "scaling_decision": scaling_decision,
        },
        "token_efficiency": token_efficiency,
        "trajectory_mean_holdout_ce": {
            model: _trajectory_area(points) for model, points in trajectories.items()
        },
        "recurrent_utility": recurrent_utility,
        "parameter_counts": report_by_model,
        "hardware": final_hardware,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "parameter_counts.json").write_text(
        json.dumps(report_by_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "timing.json").write_text(
        json.dumps(final_hardware, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_root = root / "plots"
    line_svg(
        plot_root / "validation_ce_vs_training_tokens.svg",
        title="Confirmation CE vs training tokens",
        x_label="Training tokens (millions)",
        y_label="Cross-entropy",
        series=[
            (
                MODEL_LABELS[model],
                COLORS[model],
                [(tokens / 1e6, ce) for tokens, ce in trajectories[model]],
            )
            for model in MODEL_ORDER
        ],
    )
    difference_series = []
    for model, label in (("irl2", "IRL-2 - Transformer"), ("irl_core", "IRL Core - Transformer")):
        values = [
            (row["training_tokens"] / 1e6, row["delta_ce"])
            for row in pairwise_rows
            if row["left_model"] == model and row["right_model"] == "transformer"
        ]
        difference_series.append((label, COLORS[model], values))
    line_svg(
        plot_root / "ce_difference_vs_transformer.svg",
        title="Paired confirmation CE difference vs Transformer",
        x_label="Training tokens (millions)",
        y_label="Delta CE (negative is better)",
        series=difference_series,
        zero_line=True,
        y_formatter=lambda value: f"{value:+.4f}",
    )
    final_ce = {
        row["model"]: row["holdout_cross_entropy"]
        for row in milestone_rows
        if row["step"] == final_step
    }
    bar_svg(
        plot_root / "final_500m_ce.svg",
        title="Final confirmation CE",
        y_label="Cross-entropy",
        values=[(MODEL_LABELS[model], COLORS[model], final_ce[model]) for model in MODEL_ORDER],
        formatter=lambda value: f"{value:.4f}",
    )
    line_svg(
        plot_root / "irl_core_ce_vs_depth.svg",
        title="IRL Core dev CE vs recurrent depth",
        x_label="Recurrent depth T",
        y_label="Cross-entropy",
        series=[
            (
                f"{int(milestone['tokens']) / 1e6:.0f}M tokens",
                color,
                [
                    (row["depth"], row["dev_cross_entropy"])
                    for row in depth_rows
                    if row["model"] == "irl_core" and row["step"] == milestone["step"]
                ],
            )
            for milestone, color in zip(preflight["milestones"], ("#94a3b8", "#0ea5e9", "#10b981"))
        ],
        x_formatter=lambda value: f"{value:.0f}",
    )
    bar_svg(
        plot_root / "throughput_vs_architecture.svg",
        title="Measured training throughput",
        y_label="Tokens / second",
        values=[
            (
                MODEL_LABELS[model],
                COLORS[model],
                final_hardware[model]["training_tokens_per_second_median"],
            )
            for model in MODEL_ORDER
        ],
        formatter=lambda value: f"{value / 1000:.1f}k",
    )
    line_svg(
        plot_root / "quality_vs_cumulative_flops.svg",
        title="Confirmation CE vs estimated cumulative training FLOPs",
        x_label="Estimated training FLOPs (1e20)",
        y_label="Cross-entropy",
        series=[
            (
                MODEL_LABELS[model],
                COLORS[model],
                [
                    (
                        row["cumulative_estimated_training_flops"] / 1e20,
                        row["holdout_cross_entropy"],
                    )
                    for row in milestone_rows
                    if row["model"] == model
                ],
            )
            for model in MODEL_ORDER
        ],
    )

    central = []
    for model in MODEL_ORDER:
        rows = sorted(
            (row for row in milestone_rows if row["model"] == model),
            key=lambda row: row["step"],
        )
        central.append(
            "| {label} | {params:,} | {ce100:.6f} | {ce250:.6f} | "
            "{ce500:.6f} | {tps:,.0f} | {flops:.3e} |".format(
                label=MODEL_LABELS[model],
                params=rows[-1]["parameters"],
                ce100=rows[0]["holdout_cross_entropy"],
                ce250=rows[1]["holdout_cross_entropy"],
                ce500=rows[2]["holdout_cross_entropy"],
                tps=final_hardware[model]["training_tokens_per_second_median"],
                flops=rows[-1]["estimated_training_flops_per_token"],
            )
        )
    report = f"""# 135M Controlled Architecture Benchmark

Source commit used for all trajectories: `{preflight['source_commit']}`.

| Model | Params | ~100M CE | ~250M CE | ~500M CE | tok/s | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(central)}

The CE values above are from the frozen confirmation holdout. This benchmark is parameter- and token-matched, not FLOP-matched.

## Final paired comparisons

| Comparison | Delta CE | Paired 95% CI | Batches |
| --- | ---: | ---: | ---: |
{chr(10).join(f"| {label} | {values['delta_ce']:+.6f} | [{values['ci95_low']:+.6f}, {values['ci95_high']:+.6f}] | {values['paired_batches']} |" for label, values in final_pairs.items())}

## Decision

- IRL Core vs Transformer: `{core_outcome}`.
- IRL-2 vs Transformer: `{irl2_outcome}`.
- IRL Core vs IRL-2: `{core_irl2_outcome}`.
- Core recurrent-time utility emerged by the declared criterion: `{utility_emerged}`.
- Core requires `{flop_ratio:.3f}x` estimated training FLOPs/token and `{wall_ratio:.3f}x` measured training wall time relative to the Transformer.
- Scaling decision: `{scaling_decision}`.

The machine-readable summary, paired batch observations, depth curves, clipping data, timings, and plots in this directory are the authoritative evidence. Linear token-efficiency interpolation is reported only when the observed 100M–500M curve brackets the Transformer-at-500M target; no out-of-range extrapolation is used.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the result tree to summarize."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("qualification_artifacts/irl_core_architecture_benchmark"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args().root.resolve())
