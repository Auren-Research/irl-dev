#!/usr/bin/env python3
"""Build machine-readable and concise human IRL-3 qualification reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _training_summary(records: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not records:
        return None
    records = sorted(records, key=lambda item: int(item["step"]))
    ce = _finite(records, "cross_entropy")
    losses = _finite(records, "loss")
    gradients = _finite(records, "gradient_norm")
    throughput = _finite(records[max(1, len(records) // 5) :], "tokens_per_second")
    memory = _finite(records, "memory_peak_gib")
    required = ce + losses + gradients
    return {
        "records": len(records),
        "first_step": int(records[0]["step"]),
        "final_step": int(records[-1]["step"]),
        "tokens_seen": int(records[-1].get("tokens_seen", 0)),
        "all_required_metrics_finite": bool(required)
        and all(math.isfinite(value) for value in required),
        "cross_entropy_first": ce[0] if ce else None,
        "cross_entropy_last": ce[-1] if ce else None,
        "cross_entropy_delta": ce[-1] - ce[0] if len(ce) >= 2 else None,
        "gradient_norm_maximum": max(gradients) if gradients else None,
        "throughput_tokens_per_second_median": (
            statistics.median(throughput) if throughput else None
        ),
        "peak_memory_gib": max(memory) if memory else None,
        "maximum_compute_depth": max(
            int(record.get("irl3_compute_depth", record.get("irl_train_iters", 0)))
            for record in records
        ),
    }


def _depth_summary(report: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if report is None:
        return None
    records = report.get("records", [])
    if not records:
        return None
    shallow = records[0]
    deep = records[-1]
    return {
        "architecture": report.get("architecture"),
        "parameters": report.get("parameters"),
        "precision": report.get("precision"),
        "sequence_length": report.get("sequence_length"),
        "records": [
            {
                "depth": item["depth"],
                "halt_mode": item["halt_mode"],
                "cross_entropy_mean": item["cross_entropy_mean"],
                "median_tokens_per_second": item["median_tokens_per_second"],
                "peak_memory_gib": item["peak_memory_gib"],
                "expected_steps": item.get("diagnostics", {}).get("expected_steps"),
                "converged_fraction": item.get("diagnostics", {}).get(
                    "converged_fraction"
                ),
            }
            for item in records
        ],
        "deep_minus_shallow_cross_entropy": (
            float(deep["cross_entropy_mean"]) - float(shallow["cross_entropy_mean"])
        ),
        "deep_to_shallow_throughput_ratio": (
            float(deep["median_tokens_per_second"])
            / max(float(shallow["median_tokens_per_second"]), 1e-12)
        ),
    }


def build_summary(
    *,
    irl3_metrics: list[dict[str, Any]],
    irl3_fixed: Optional[dict[str, Any]],
    irl3_adaptive: Optional[dict[str, Any]],
    irl2_metrics: list[dict[str, Any]],
    irl2_depth: Optional[dict[str, Any]],
    accelerator: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "qualification_scope": "architecture_and_hardware_not_convergence",
        "accelerator": accelerator,
        "irl3_training": _training_summary(irl3_metrics),
        "irl3_fixed_depth_scaling": _depth_summary(irl3_fixed),
        "irl3_adaptive_halting": _depth_summary(irl3_adaptive),
        "irl2_small_matched_baseline_training": _training_summary(irl2_metrics),
        "irl2_depth_scaling": _depth_summary(irl2_depth),
        "interpretation_constraints": [
            "The run is too short to establish converged language-model quality.",
            "Depth comparisons reuse one checkpoint and isolate inference compute.",
            "The IRL-2 baseline is parameter/sequence matched but has a "
            "smaller token budget.",
            "A non-improving short-run depth curve is evidence for iteration, "
            "not disproof.",
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    train = summary["irl3_training"]
    fixed = summary["irl3_fixed_depth_scaling"]
    adaptive = summary["irl3_adaptive_halting"]
    baseline = summary["irl2_small_matched_baseline_training"]
    lines = [
        "# IRL-3 single-MI300X qualification",
        "",
        "This is an architecture/hardware qualification, not a "
        "converged-quality claim.",
        "",
    ]
    if train:
        lines.extend(
            [
                "## Training path",
                "",
                f"- Final step: {train['final_step']}",
                f"- Tokens: {train['tokens_seen']:,}",
                f"- Metrics finite: {train['all_required_metrics_finite']}",
                f"- CE: {train['cross_entropy_first']} -> "
                f"{train['cross_entropy_last']}",
                "- Median warm throughput: "
                f"{train['throughput_tokens_per_second_median']} tok/s",
                f"- Peak HBM: {train['peak_memory_gib']} GiB",
                "",
            ]
        )
    if fixed:
        lines.extend(["## Fixed test-time compute", ""])
        for item in fixed["records"]:
            lines.append(
                f"- depth {item['depth']}: CE {item['cross_entropy_mean']:.6f}, "
                f"{item['median_tokens_per_second']:.1f} tok/s, "
                f"{item['peak_memory_gib']:.2f} GiB"
            )
        lines.extend(
            [
                "- Deep-minus-shallow CE: "
                f"{fixed['deep_minus_shallow_cross_entropy']:.6f}",
                "",
            ]
        )
    if adaptive:
        lines.extend(["## Learned halting probe", ""])
        for item in adaptive["records"]:
            lines.append(
                f"- budget {item['depth']}: expected steps {item['expected_steps']}, "
                f"converged fraction {item['converged_fraction']}"
            )
        lines.append("")
    if baseline:
        lines.extend(
            [
                "## Small IRL-2 baseline",
                "",
                f"- Steps/tokens: {baseline['final_step']} / "
                f"{baseline['tokens_seen']:,}",
                f"- CE: {baseline['cross_entropy_first']} -> "
                f"{baseline['cross_entropy_last']}",
                "- Token budget is smaller than IRL-3; interpret only as a "
                "path sanity check.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            *[f"- {item}" for item in summary["interpretation_constraints"]],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--irl3-metrics", type=Path, required=True)
    parser.add_argument("--irl3-fixed", type=Path, required=True)
    parser.add_argument("--irl3-adaptive", type=Path)
    parser.add_argument("--irl2-metrics", type=Path)
    parser.add_argument("--irl2-depth", type=Path)
    parser.add_argument("--accelerator", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        irl3_metrics=_load_jsonl(args.irl3_metrics),
        irl3_fixed=_load_json(args.irl3_fixed),
        irl3_adaptive=_load_json(args.irl3_adaptive),
        irl2_metrics=_load_jsonl(args.irl2_metrics),
        irl2_depth=_load_json(args.irl2_depth),
        accelerator=_load_json(args.accelerator),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown(summary)
    args.markdown_output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
