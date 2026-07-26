#!/usr/bin/env python3
"""Summarize a Lunaris qualification JSONL log into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without third-party packages."""

    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def finite_values(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    """Collect finite numeric values for one metric key."""

    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def cumulative_elapsed_seconds(records: Iterable[dict[str, Any]]) -> float:
    """Sum elapsed-time segments when a resumed process resets its timer."""

    total = 0.0
    previous = 0.0
    observed = False
    for record in records:
        value = record.get("elapsed_seconds")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        current = max(0.0, float(value))
        if observed and current < previous:
            total += previous
        previous = current
        observed = True
    return total + previous if observed else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact statistical and correctness summary."""

    if not records:
        raise ValueError("The metrics file contains no records")
    elapsed_seconds = cumulative_elapsed_seconds(records)
    records = sorted(records, key=lambda item: int(item["step"]))
    warm_records = records[max(1, len(records) // 5) :]
    throughput = finite_values(warm_records, "tokens_per_second")
    losses = finite_values(records, "loss")
    cross_entropy = finite_values(records, "cross_entropy")
    validation = finite_values(records, "validation_cross_entropy")
    gradients = finite_values(records, "gradient_norm")
    mtp_losses = finite_values(records, "mtp_loss")
    consistency = finite_values(records, "irl_consistency_loss")
    memory_peak = finite_values(records, "memory_peak_gib")

    finite_required = losses + cross_entropy + gradients + mtp_losses + consistency
    all_finite = bool(finite_required) and all(
        math.isfinite(value) for value in finite_required
    )
    return {
        "records": len(records),
        "first_step": int(records[0]["step"]),
        "final_step": int(records[-1]["step"]),
        "tokens_seen": int(records[-1].get("tokens_seen", 0)),
        "elapsed_seconds": elapsed_seconds,
        "all_required_metrics_finite": all_finite,
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "delta": losses[-1] - losses[0],
        },
        "cross_entropy": {
            "first": cross_entropy[0],
            "last": cross_entropy[-1],
            "delta": cross_entropy[-1] - cross_entropy[0],
        },
        "validation_cross_entropy": {
            "first": validation[0] if validation else None,
            "last": validation[-1] if validation else None,
            "delta": (
                validation[-1] - validation[0] if len(validation) >= 2 else None
            ),
        },
        "throughput_tokens_per_second_after_warmup": {
            "mean": statistics.fmean(throughput) if throughput else None,
            "median": statistics.median(throughput) if throughput else None,
            "p90": percentile(throughput, 0.90) if throughput else None,
        },
        "gradient_norm": {
            "median": statistics.median(gradients),
            "maximum": max(gradients),
        },
        "mtp_loss": {
            "first": mtp_losses[0],
            "last": mtp_losses[-1],
        },
        "irl_consistency_loss": {
            "first": consistency[0],
            "last": consistency[-1],
        },
        "maximum_irl_iterations": max(
            int(record.get("irl_train_iters", 0)) for record in records
        ),
        "maximum_active_mtp_heads": max(
            int(record.get("mtp_active_heads", 0)) for record in records
        ),
        "peak_memory_gib": max(memory_peak) if memory_peak else None,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render a human-readable qualification report."""

    throughput = summary["throughput_tokens_per_second_after_warmup"]
    validation = summary["validation_cross_entropy"]
    lines = [
        "# Lunaris MI300X Qualification Summary",
        "",
        f"- Final step: {summary['final_step']:,}",
        f"- Tokens seen: {summary['tokens_seen']:,}",
        f"- Elapsed training time: {summary['elapsed_seconds']:.2f} s",
        f"- Required metrics finite: {summary['all_required_metrics_finite']}",
        f"- Training CE: {summary['cross_entropy']['first']:.6f} → "
        f"{summary['cross_entropy']['last']:.6f}",
        f"- Validation CE: {validation['first']} → {validation['last']}",
        f"- Median throughput after warmup: {throughput['median']} tokens/s",
        f"- Peak accelerator memory: {summary['peak_memory_gib']} GiB",
        f"- Maximum IRL iterations exercised: "
        f"{summary['maximum_irl_iterations']}",
        f"- Maximum MTP heads exercised: "
        f"{summary['maximum_active_mtp_heads']}",
        "",
        "The report is descriptive evidence from a short qualification run, "
        "not a claim of converged model quality.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Parse one metrics file and write JSON plus Markdown summaries."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("qualification_artifacts/summary.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("qualification_artifacts/SUMMARY.md"),
    )
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize(records)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = markdown(summary)
    args.markdown_output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
