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
    warm_records = records[max(1, len(records) // 5) :]
    throughput = _finite(warm_records, "tokens_per_second")
    memory = _finite(records, "memory_peak_gib")
    analytic_flops = _finite(records, "analytic_training_flops_per_token")
    estimated_tflops = _finite(records, "estimated_tflops")
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
        "analytic_training_flops_per_token_last": (
            analytic_flops[-1] if analytic_flops else None
        ),
        "estimated_tflops_median": (
            statistics.median(estimated_tflops) if estimated_tflops else None
        ),
        "maximum_compute_depth": max(
            int(record.get("irl3_compute_depth", record.get("irl_train_iters", 0)))
            for record in records
        ),
    }


_DIAGNOSTIC_NAMES = (
    "halt_probabilities",
    "halt_weights",
    "active_latent_fraction",
    "anchor_read_rms",
    "anchor_attention_entropy",
    "memory_read_rms",
    "memory_write_rms",
    "memory_utilization",
    "z_update_rms",
    "h_update_rms",
    "m_update_rms",
    "z_rms",
    "h_rms",
    "m_rms",
    "slow_update_mask",
    "fast_residual",
    "joint_residual",
    "fast_contraction_ratios",
    "contraction_ratios",
    "equilibrium_z_residual",
    "equilibrium_h_residual",
    "equilibrium_m_residual",
    "last_h_update_rms",
    "writeback_rms",
    "writeback_to_anchor_ratio",
    "expected_steps",
    "step_counts",
)


def _compact_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("diagnostics", {})
    result = {
        name: diagnostics[name]
        for name in _DIAGNOSTIC_NAMES
        if name in diagnostics
    }
    for name in (
        "iterations",
        "slow_updates",
        "last_slow_update_step",
        "converged_fraction",
        "step_count_histogram",
    ):
        if name in diagnostics:
            result[name] = diagnostics[name]
    return result


def _record_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": int(item["depth"]),
        "halt_mode": item["halt_mode"],
        "evaluation_variant": item.get("evaluation_variant", item["halt_mode"]),
        "writeback_disabled": bool(item.get("writeback_disabled", False)),
        "cross_entropy_mean": float(item["cross_entropy_mean"]),
        "median_elapsed_seconds": item.get("median_elapsed_seconds"),
        "median_batch_latency_milliseconds": item.get(
            "median_batch_latency_milliseconds"
        ),
        "median_tokens_per_second": float(item["median_tokens_per_second"]),
        "peak_memory_gib": float(item["peak_memory_gib"]),
        "diagnostic_peak_memory_gib": item.get("diagnostic_peak_memory_gib"),
        "analytic_budget_forward_flops_per_token": item.get(
            "analytic_budget_forward_flops_per_token"
        ),
        "analytic_diagnostic_forward_flops_per_token": item.get(
            "analytic_diagnostic_forward_flops_per_token"
        ),
        "diagnostics": _compact_diagnostics(item),
    }


def _depth_summary(report: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if report is None:
        return None
    records = sorted(report.get("records", []), key=lambda item: int(item["depth"]))
    if not records:
        return None
    summarized = [_record_summary(item) for item in records]
    shallow = summarized[0]
    deep = summarized[-1]
    return {
        "architecture": report.get("architecture"),
        "checkpoint": report.get("checkpoint"),
        "parameters": report.get("parameters"),
        "precision": report.get("precision"),
        "sequence_length": report.get("sequence_length"),
        "batch_size": report.get("batch_size"),
        "batches": report.get("batches"),
        "repeats": report.get("repeats"),
        "halt_mode": report.get("halt_mode"),
        "writeback_disabled": bool(report.get("writeback_disabled", False)),
        "records": summarized,
        "deep_minus_shallow_cross_entropy": (
            deep["cross_entropy_mean"] - shallow["cross_entropy_mean"]
        ),
        "deep_to_shallow_throughput_ratio": (
            deep["median_tokens_per_second"]
            / max(shallow["median_tokens_per_second"], 1e-12)
        ),
    }


def _diagnostic_mean(record: dict[str, Any], name: str) -> Optional[float]:
    value = record.get("diagnostics", {}).get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _halting_comparison(
    act: Optional[dict[str, Any]],
    adaptive: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if act is None or adaptive is None:
        return None
    act_by_depth = {item["depth"]: item for item in act["records"]}
    adaptive_by_depth = {item["depth"]: item for item in adaptive["records"]}
    records = []
    for depth in sorted(set(act_by_depth) & set(adaptive_by_depth)):
        soft = act_by_depth[depth]
        hard = adaptive_by_depth[depth]
        soft_latency = soft.get("median_batch_latency_milliseconds")
        hard_latency = hard.get("median_batch_latency_milliseconds")
        records.append(
            {
                "depth": depth,
                "act_soft_cross_entropy": soft["cross_entropy_mean"],
                "adaptive_hard_cross_entropy": hard["cross_entropy_mean"],
                "adaptive_minus_act_cross_entropy": (
                    hard["cross_entropy_mean"] - soft["cross_entropy_mean"]
                ),
                "act_soft_latency_milliseconds": soft_latency,
                "adaptive_hard_latency_milliseconds": hard_latency,
                "adaptive_to_act_latency_ratio": (
                    None
                    if soft_latency is None or hard_latency is None
                    else hard_latency / max(soft_latency, 1e-12)
                ),
                "act_soft_tokens_per_second": soft["median_tokens_per_second"],
                "adaptive_hard_tokens_per_second": hard[
                    "median_tokens_per_second"
                ],
                "act_soft_peak_memory_gib": soft["peak_memory_gib"],
                "adaptive_hard_peak_memory_gib": hard["peak_memory_gib"],
                "act_soft_expected_steps": _diagnostic_mean(
                    soft, "expected_steps"
                ),
                "adaptive_hard_expected_steps": _diagnostic_mean(
                    hard, "expected_steps"
                ),
                "adaptive_hard_converged_fraction": hard.get(
                    "diagnostics", {}
                ).get("converged_fraction"),
                "act_soft_halt_probability": _diagnostic_mean(
                    soft, "halt_probabilities"
                ),
                "adaptive_hard_halt_probability": _diagnostic_mean(
                    hard, "halt_probabilities"
                ),
            }
        )
    return {"records": records} if records else None


def _writeback_comparison(
    fixed: Optional[dict[str, Any]],
    writeback_off: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if fixed is None or writeback_off is None:
        return None
    fixed_by_depth = {item["depth"]: item for item in fixed["records"]}
    records = []
    for disabled in writeback_off["records"]:
        enabled = fixed_by_depth.get(disabled["depth"])
        if enabled is None:
            continue
        records.append(
            {
                "depth": disabled["depth"],
                "writeback_enabled_cross_entropy": enabled["cross_entropy_mean"],
                "writeback_disabled_cross_entropy": disabled[
                    "cross_entropy_mean"
                ],
                "disabled_minus_enabled_cross_entropy": (
                    disabled["cross_entropy_mean"] - enabled["cross_entropy_mean"]
                ),
                "writeback_enabled_latency_milliseconds": enabled.get(
                    "median_batch_latency_milliseconds"
                ),
                "writeback_disabled_latency_milliseconds": disabled.get(
                    "median_batch_latency_milliseconds"
                ),
                "writeback_disabled_tokens_per_second": disabled[
                    "median_tokens_per_second"
                ],
                "writeback_disabled_peak_memory_gib": disabled["peak_memory_gib"],
                "would_be_writeback_rms": _diagnostic_mean(
                    disabled, "writeback_rms"
                ),
                "would_be_correction_to_anchor_ratio": _diagnostic_mean(
                    disabled, "writeback_to_anchor_ratio"
                ),
            }
        )
    return {"records": records} if records else None


def _validate_shared_checkpoint(*reports: Optional[dict[str, Any]]) -> Optional[str]:
    checkpoints = {
        str(report["checkpoint"])
        for report in reports
        if report is not None and report.get("checkpoint") is not None
    }
    if len(checkpoints) > 1:
        raise ValueError(
            "All IRL-3 qualification variants must use the same checkpoint: "
            f"{sorted(checkpoints)}"
        )
    return next(iter(checkpoints), None)


def build_summary(
    *,
    irl3_metrics: list[dict[str, Any]],
    irl3_fixed: Optional[dict[str, Any]],
    irl3_act: Optional[dict[str, Any]],
    irl3_adaptive: Optional[dict[str, Any]],
    irl3_writeback_off: Optional[dict[str, Any]],
    irl2_metrics: list[dict[str, Any]],
    irl2_depth: Optional[dict[str, Any]],
    accelerator: Optional[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = _validate_shared_checkpoint(
        irl3_fixed,
        irl3_act,
        irl3_adaptive,
        irl3_writeback_off,
    )
    fixed = _depth_summary(irl3_fixed)
    act = _depth_summary(irl3_act)
    adaptive = _depth_summary(irl3_adaptive)
    writeback_off = _depth_summary(irl3_writeback_off)
    return {
        "schema_version": 2,
        "qualification_scope": "architecture_and_hardware_not_convergence",
        "accelerator": accelerator,
        "irl3_shared_checkpoint": checkpoint,
        "irl3_training": _training_summary(irl3_metrics),
        "irl3_fixed_depth_scaling": fixed,
        "irl3_act_soft": act,
        "irl3_adaptive_hard": adaptive,
        "irl3_soft_vs_hard_halting": _halting_comparison(act, adaptive),
        "irl3_writeback_off": writeback_off,
        "irl3_writeback_ablation": _writeback_comparison(fixed, writeback_off),
        "irl2_small_matched_baseline_training": _training_summary(irl2_metrics),
        "irl2_depth_scaling": _depth_summary(irl2_depth),
        "interpretation_constraints": [
            "The run is too short to establish converged language-model quality.",
            "Every IRL-3 inference variant reuses exactly one checkpoint.",
            "Write-back OFF still executes the recurrent computer but does not "
            "apply its correction to the token stream.",
            "ACT-soft averages recurrent states, whereas adaptive-hard freezes "
            "prefix states; their downstream nonlinear coda need not agree.",
            "Analytic FLOPs count dense matmuls and are not a substitute for "
            "measured latency or HBM.",
            "The IRL-2 baseline is parameter/sequence matched but has a smaller "
            "token budget.",
            "A non-improving short-run depth curve is evidence for iteration, "
            "not disproof.",
        ],
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    train = summary["irl3_training"]
    fixed = summary["irl3_fixed_depth_scaling"]
    halting = summary["irl3_soft_vs_hard_halting"]
    writeback = summary["irl3_writeback_ablation"]
    baseline = summary["irl2_small_matched_baseline_training"]
    baseline_depth = summary["irl2_depth_scaling"]
    lines = [
        "# IRL-3 single-MI300X qualification",
        "",
        "This is an architecture/hardware qualification, not a "
        "converged-quality claim.",
        "",
        f"Shared IRL-3 checkpoint: `{summary['irl3_shared_checkpoint']}`",
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
                "- Final analytic training FLOPs/observed token: "
                f"{train['analytic_training_flops_per_token_last']}",
                "",
            ]
        )
    if fixed:
        lines.extend(
            [
                "## Fixed test-time compute and write-back control",
                "",
                "| Variant | Budget | CE | Latency ms/batch | tok/s | Peak GiB | "
                "correction/anchor |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if writeback:
            for item in writeback["records"]:
                lines.append(
                    "| OFF (write-back disabled) | "
                    f"{item['depth']} | "
                    f"{_fmt(item['writeback_disabled_cross_entropy'], 6)} | "
                    f"{_fmt(item['writeback_disabled_latency_milliseconds'], 2)} | "
                    f"{_fmt(item['writeback_disabled_tokens_per_second'], 1)} | "
                    f"{_fmt(item['writeback_disabled_peak_memory_gib'], 2)} | "
                    f"{_fmt(item['would_be_correction_to_anchor_ratio'], 6)} |"
                )
        for item in fixed["records"]:
            lines.append(
                f"| fixed | {item['depth']} | "
                f"{_fmt(item['cross_entropy_mean'], 6)} | "
                f"{_fmt(item['median_batch_latency_milliseconds'], 2)} | "
                f"{_fmt(item['median_tokens_per_second'], 1)} | "
                f"{_fmt(item['peak_memory_gib'], 2)} | "
                f"{_fmt(_diagnostic_mean(item, 'writeback_to_anchor_ratio'), 6)} |"
            )
        lines.extend(
            [
                "",
                "- Deep-minus-shallow CE: "
                f"{fixed['deep_minus_shallow_cross_entropy']:.6f}",
            ]
        )
        if writeback:
            for item in writeback["records"]:
                lines.append(
                    "- Write-back OFF minus ON CE at budget "
                    f"{item['depth']}: "
                    f"{item['disabled_minus_enabled_cross_entropy']:.6f} "
                    "(positive means applying write-back helped)."
                )
        lines.append("")
    if halting:
        lines.extend(
            [
                "## ACT-soft versus adaptive-hard",
                "",
                "| Budget | ACT CE | Adaptive CE | Hard-soft CE | ACT steps | "
                "Adaptive steps | ACT ms | Adaptive ms | ACT tok/s | "
                "Adaptive tok/s | ACT/Adaptive GiB |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in halting["records"]:
            lines.append(
                f"| {item['depth']} | "
                f"{_fmt(item['act_soft_cross_entropy'], 6)} | "
                f"{_fmt(item['adaptive_hard_cross_entropy'], 6)} | "
                f"{_fmt(item['adaptive_minus_act_cross_entropy'], 6)} | "
                f"{_fmt(item['act_soft_expected_steps'], 3)} | "
                f"{_fmt(item['adaptive_hard_expected_steps'], 3)} | "
                f"{_fmt(item['act_soft_latency_milliseconds'], 2)} | "
                f"{_fmt(item['adaptive_hard_latency_milliseconds'], 2)} | "
                f"{_fmt(item['act_soft_tokens_per_second'], 1)} | "
                f"{_fmt(item['adaptive_hard_tokens_per_second'], 1)} | "
                f"{_fmt(item['act_soft_peak_memory_gib'], 2)} / "
                f"{_fmt(item['adaptive_hard_peak_memory_gib'], 2)} |"
            )
        lines.extend(
            [
                "",
                "The CE delta is reported rather than assumed negligible: "
                "soft state averaging and hard prefix freezing feed a nonlinear coda.",
                "",
            ]
        )
    if fixed:
        reference = next(
            (item for item in fixed["records"] if item["depth"] == 6),
            fixed["records"][-1],
        )
        diagnostics = reference["diagnostics"]
        lines.extend(
            [
                "## Recurrent dynamics at the reference fixed budget",
                "",
                f"- Budget: {reference['depth']}",
                f"- Z/H/M update RMS means: "
                f"{_fmt(_diagnostic_mean(reference, 'z_update_rms'), 6)} / "
                f"{_fmt(_diagnostic_mean(reference, 'h_update_rms'), 6)} / "
                f"{_fmt(_diagnostic_mean(reference, 'm_update_rms'), 6)}",
                "- Memory write RMS/utilization: "
                f"{_fmt(_diagnostic_mean(reference, 'memory_write_rms'), 6)} / "
                f"{_fmt(_diagnostic_mean(reference, 'memory_utilization'), 6)}",
                f"- Slow updates / last tick: {diagnostics.get('slow_updates')} / "
                f"{diagnostics.get('last_slow_update_step')}",
                "- Last meaningful H equilibrium residual: "
                f"{_fmt(_diagnostic_mean(reference, 'equilibrium_h_residual'), 6)}",
                "- Fast contraction mean (clock-independent): "
                f"{_fmt(_diagnostic_mean(reference, 'fast_contraction_ratios'), 6)}",
                "",
            ]
        )
    if baseline:
        lines.extend(
            [
                "## Guaranteed small IRL-2 baseline",
                "",
                f"- Steps/tokens: {baseline['final_step']} / "
                f"{baseline['tokens_seen']:,}",
                f"- CE: {baseline['cross_entropy_first']} -> "
                f"{baseline['cross_entropy_last']}",
                "- Token budget is smaller than IRL-3; interpret only as a "
                "matched short-run control.",
            ]
        )
        if baseline_depth:
            lines.append(
                "- Deep-minus-shallow IRL-2 CE: "
                f"{baseline_depth['deep_minus_shallow_cross_entropy']:.6f}"
            )
        lines.append("")
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
    parser.add_argument("--irl3-act", type=Path, required=True)
    parser.add_argument("--irl3-adaptive", type=Path, required=True)
    parser.add_argument("--irl3-writeback-off", type=Path, required=True)
    parser.add_argument("--irl2-metrics", type=Path)
    parser.add_argument("--irl2-depth", type=Path)
    parser.add_argument("--accelerator", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(
        irl3_metrics=_load_jsonl(args.irl3_metrics),
        irl3_fixed=_load_json(args.irl3_fixed),
        irl3_act=_load_json(args.irl3_act),
        irl3_adaptive=_load_json(args.irl3_adaptive),
        irl3_writeback_off=_load_json(args.irl3_writeback_off),
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
