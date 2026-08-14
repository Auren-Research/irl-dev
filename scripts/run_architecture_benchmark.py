#!/usr/bin/env python3
"""Run the three frozen 135M trajectories and milestone evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from lunaris_dense_irl2.training.architecture_benchmark import (
    MILESTONE_STEPS,
    MODEL_ORDER,
    file_sha256,
    git_commit,
)
from lunaris_dense_irl2.training.config import TrainConfig

CONFIG_PATHS = {
    "transformer": Path(
        "training_profiles/benchmark_transformer_135m_1x_mi300x.yaml"
    ),
    "irl2": Path("training_profiles/benchmark_irl2_135m_1x_mi300x.yaml"),
    "irl_core": Path(
        "training_profiles/benchmark_irl_core_135m_1x_mi300x.yaml"
    ),
}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one durable orchestration event."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run(command: Sequence[str], events_path: Path, **metadata: Any) -> float:
    """Run one foreground stage and record wall-clock status."""

    started_unix = time.time()
    started = time.perf_counter()
    _append_jsonl(
        events_path,
        {
            "event": "stage_started",
            "started_unix_time": started_unix,
            "command": list(command),
            **metadata,
        },
    )
    completed = subprocess.run(command, check=False)
    seconds = time.perf_counter() - started
    _append_jsonl(
        events_path,
        {
            "event": "stage_finished",
            "started_unix_time": started_unix,
            "finished_unix_time": time.time(),
            "elapsed_seconds": seconds,
            "returncode": completed.returncode,
            "command": list(command),
            **metadata,
        },
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return seconds


def _verify_frozen_inputs(
    preflight_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """Fail before compute if provenance or a frozen profile changed."""

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    current_commit = git_commit()
    if current_commit != expected_commit:
        raise RuntimeError(
            f"Benchmark expected commit {expected_commit}, found {current_commit}"
        )
    if preflight.get("source_commit") != expected_commit:
        raise RuntimeError("Preflight was generated from a different commit")
    if not preflight.get("controls_verified"):
        raise RuntimeError("Preflight did not verify the controlled comparison")
    for model_name, config_path in CONFIG_PATHS.items():
        expected_hash = preflight["config_sha256"][model_name]
        actual_hash = file_sha256(config_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frozen config changed for {model_name}")
    return preflight


def _verify_runtime_budget(
    qualification_root: Path,
    *,
    max_projected_hours: float,
) -> dict[str, Any]:
    """Sum independent steady-state projections before long training starts."""

    reports = {}
    total_hours = 0.0
    for model_name in MODEL_ORDER:
        path = qualification_root / f"{model_name}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[model_name] = report
        total_hours += float(report["projected_trajectory_hours"])
    # Reserve 30 minutes for milestone checkpointing and bounded evaluations.
    projected_total_hours = total_hours + 0.5
    if projected_total_hours > max_projected_hours:
        raise RuntimeError(
            f"Projected benchmark time is {projected_total_hours:.2f}h, "
            f"above the configured {max_projected_hours:.2f}h ceiling"
        )
    return {
        "training_hours": total_hours,
        "evaluation_and_checkpoint_allowance_hours": 0.5,
        "total_hours": projected_total_hours,
        "models": reports,
    }


def run(args: argparse.Namespace) -> None:
    """Interleave architectures at 100M, 250M, and 500M milestones."""

    result_root = args.result_root.resolve()
    events_path = result_root / "orchestration" / "events.jsonl"
    preflight = _verify_frozen_inputs(args.preflight, args.expected_commit)
    budget = _verify_runtime_budget(
        args.qualification_root,
        max_projected_hours=args.max_projected_hours,
    )
    freeze_manifest = {
        "schema_version": 1,
        "source_commit": args.expected_commit,
        "started_unix_time": time.time(),
        "config_sha256": preflight["config_sha256"],
        "data_manifest_sha256": preflight["data_manifest_sha256"],
        "milestones": preflight["milestones"],
        "runtime_projection": budget,
        "model_order_within_each_milestone": list(MODEL_ORDER),
    }
    freeze_path = result_root / "frozen_run.json"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        for key in ("source_commit", "config_sha256", "data_manifest_sha256"):
            if existing[key] != freeze_manifest[key]:
                raise RuntimeError(f"Existing frozen run differs in {key}")
    else:
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for milestone_step in MILESTONE_STEPS:
        milestone_tokens = next(
            value["tokens"]
            for value in preflight["milestones"]
            if value["step"] == milestone_step
        )
        for model_name in MODEL_ORDER:
            config_path = CONFIG_PATHS[model_name]
            config = TrainConfig.from_yaml(config_path)
            checkpoint = (
                Path(config.checkpoint.directory)
                / f"step_{milestone_step:08d}"
            )
            if not (checkpoint / "_SUCCESS").exists():
                _run(
                    [
                        sys.executable,
                        "-m",
                        "lunaris_dense_irl2.training.train",
                        "--config",
                        str(config_path),
                        "--resume",
                        "auto",
                        "--stop-after-step",
                        str(milestone_step),
                    ],
                    events_path,
                    stage="train",
                    model=model_name,
                    milestone_step=milestone_step,
                    milestone_tokens=milestone_tokens,
                )
            if not (checkpoint / "_SUCCESS").exists():
                raise RuntimeError(f"Training did not create {checkpoint}")

            output = (
                result_root
                / "evaluation"
                / f"{model_name}_step_{milestone_step:08d}.json"
            )
            if args.force_evaluation or not output.exists():
                _run(
                    [
                        sys.executable,
                        "-m",
                        "lunaris_dense_irl2.training.architecture_benchmark",
                        "evaluate",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(checkpoint),
                        "--model-name",
                        model_name,
                        "--dev-dir",
                        str(args.dev_dir),
                        "--holdout-dir",
                        str(args.holdout_dir),
                        "--output",
                        str(output),
                        "--batch-size",
                        str(args.evaluation_batch_size),
                    ],
                    events_path,
                    stage="evaluate",
                    model=model_name,
                    milestone_step=milestone_step,
                    milestone_tokens=milestone_tokens,
                )

    _append_jsonl(
        events_path,
        {
            "event": "benchmark_finished",
            "finished_unix_time": time.time(),
            "source_commit": args.expected_commit,
        },
    )


def parse_args() -> argparse.Namespace:
    """Parse orchestration controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path(
            "qualification_artifacts/irl_core_architecture_benchmark/preflight.json"
        ),
    )
    parser.add_argument(
        "--qualification-root",
        type=Path,
        default=Path(
            "qualification_artifacts/irl_core_architecture_benchmark/qualification"
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("qualification_artifacts/irl_core_architecture_benchmark"),
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=Path("data/architecture_benchmark/dev"),
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=Path("data/architecture_benchmark/holdout"),
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--max-projected-hours", type=float, default=7.0)
    parser.add_argument("--force-evaluation", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
