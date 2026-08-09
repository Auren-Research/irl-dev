#!/usr/bin/env python3
"""Generate matched IRL-3 recurrent-utility training configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Policy:
    fixed_depth: bool
    full_credit: bool
    ponder_weight: float

    @property
    def halt_mode(self) -> str:
        # Fixed-depth exposure includes training from the terminal fixed state.
        # Sampled-depth controls retain the original ACT state marginalization.
        return "fixed" if self.fixed_depth else "act"


POLICIES = {
    "control": Policy(False, False, 1e-3),
    "fixed_depth_only": Policy(True, False, 1e-3),
    "full_credit_only": Policy(False, True, 1e-3),
    "zero_ponder_only": Policy(False, False, 0.0),
    "positive_control": Policy(True, True, 0.0),
}

TOKENS_PER_STEP = 4 * 4096
STAGE_A_FINAL_STEP = 3000
STAGE_B_MILESTONES = {
    "100m": 6104,
    "200m": 12208,
    "350m": 21363,
    "500m": 30518,
}


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
    ).strip()


def _load_base(repo_root: Path) -> dict[str, Any]:
    path = (
        repo_root
        / "training_profiles/irl3_135m_1x_mi300x_deep_compute_continuation.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid base configuration: {path}")
    payload["model_profile"] = str(
        (repo_root / "profiles/irl3_135m_mi300x_qualification.yaml").resolve()
    )
    return payload


def _apply_policy(config: dict[str, Any], policy: Policy) -> None:
    model = config["model"]
    objectives = config["objectives"]
    model["irl3_train_steps"] = 6
    model["irl3_gradient_horizon"] = 6 if policy.full_credit else 4
    model["irl3_eval_steps"] = 6
    model["irl3_eval_halt_mode"] = "fixed"
    objectives.update(
        {
            "compute_start_steps": 6 if policy.fixed_depth else 2,
            "compute_ramp_steps": 0,
            "depth_sampling": "fixed" if policy.fixed_depth else "log_uniform",
            "depth_sampling_seed": 17,
            "training_halt_mode": policy.halt_mode,
            "gradient_horizon_start": 6 if policy.full_credit else 4,
            "gradient_horizon_ramp_steps": 0,
            "ponder_weight": policy.ponder_weight,
            "ponder_warmup_steps": 0,
            "ponder_ramp_steps": 0,
            "equilibrium_weight": 0.001,
            "equilibrium_warmup_steps": 0,
            "equilibrium_ramp_steps": 0,
            "mtp_loss_weight": 0.2,
            "mtp_warmup_steps": 0,
            "mtp_ramp_steps": 0,
            "mtp_horizon_interval": 40,
            "mtp_horizon_decay": 0.7,
        }
    )


def _common_config(repo_root: Path) -> dict[str, Any]:
    config = _load_base(repo_root)
    config["optimizer"].update(
        {
            "learning_rate": 6e-5,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "fused": "auto",
        }
    )
    config["schedule"] = {
        "name": "constant",
        "warmup_steps": 0,
        "min_lr_ratio": 1.0,
    }
    config["validation"] = {"interval": 0, "batches": 0, "at_start": False}
    config["checkpoint"].update(
        {
            "async_save": False,
            "resume": "none",
            "save_final": True,
            "min_free_space_gib": 20.0,
        }
    )
    return config


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def generate_stage_a(
    repo_root: Path,
    artifact_root: Path,
    starting_checkpoint: Path,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for name, policy in POLICIES.items():
        condition_root = artifact_root / "stage_a" / "conditions" / name
        config = copy.deepcopy(_common_config(repo_root))
        _apply_policy(config, policy)
        config["data"].update(
            {
                "train_dir": str((repo_root / "data/qualification/train").resolve()),
                "validation_dir": str(
                    (repo_root / "data/irl3_recurrent_utility/mechanism_dev").resolve()
                ),
                "seed": 1337,
                "shuffle": True,
            }
        )
        config["training"].update(
            {
                "max_steps": STAGE_A_FINAL_STEP,
                "max_tokens": STAGE_A_FINAL_STEP * TOKENS_PER_STEP,
                "seed": 1337,
            }
        )
        config["checkpoint"].update(
            {
                "directory": str((condition_root / "checkpoints").resolve()),
                "interval": 1000,
                "keep_last": 2,
            }
        )
        config["logging"].update(
            {"jsonl_path": str((condition_root / "metrics.jsonl").resolve())}
        )
        config_path = condition_root / "config.yaml"
        _write_yaml(config_path, config)
        conditions[name] = {
            "policy": asdict(policy) | {"training_halt_mode": policy.halt_mode},
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "starting_checkpoint": str(starting_checkpoint),
            "target_step": STAGE_A_FINAL_STEP,
            "target_tokens": STAGE_A_FINAL_STEP * TOKENS_PER_STEP,
        }
    return conditions


def generate_stage_b(
    repo_root: Path,
    artifact_root: Path,
    starting_checkpoint: Path,
    policy_name: str,
) -> dict[str, Any]:
    policy = POLICIES[policy_name]
    config = copy.deepcopy(_common_config(repo_root))
    _apply_policy(config, policy)
    if config["objectives"]["training_halt_mode"] != "fixed":
        raise ValueError(
            "Stage B forbids ACT/adaptive training; select a fixed-depth policy"
        )
    stage_root = artifact_root / "stage_b"
    config["data"].update(
        {
            "train_dir": str(
                (repo_root / "data/irl3_recurrent_utility/train").resolve()
            ),
            "validation_dir": str(
                (repo_root / "data/irl3_recurrent_utility/mechanism_dev").resolve()
            ),
            "seed": 1337,
            "shuffle": True,
        }
    )
    final_step = STAGE_B_MILESTONES["500m"]
    config["training"].update(
        {
            "max_steps": final_step,
            "max_tokens": final_step * TOKENS_PER_STEP,
            "seed": 1337,
        }
    )
    config["checkpoint"].update(
        {
            "directory": str((stage_root / "checkpoints").resolve()),
            "interval": 0,
            "keep_last": len(STAGE_B_MILESTONES),
        }
    )
    config["logging"].update(
        {"jsonl_path": str((stage_root / "metrics.jsonl").resolve())}
    )
    config_path = stage_root / "config.yaml"
    _write_yaml(config_path, config)
    return {
        "policy_name": policy_name,
        "policy": asdict(policy) | {"training_halt_mode": policy.halt_mode},
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "starting_checkpoint": str(starting_checkpoint),
        "milestones": {
            label: {"step": step, "tokens": step * TOKENS_PER_STEP}
            for label, step in STAGE_B_MILESTONES.items()
        },
    }


def checkpoint_identity(path: Path) -> dict[str, Any]:
    required = [
        path / "_SUCCESS",
        path / "metadata.json",
        path / "dcp/.metadata",
        path / "dcp/__0_0.distcp",
        path / "rank_00000.pt",
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {path}: {missing}")
    return {
        "path": str(path),
        "files": {
            str(item.relative_to(path)): {
                "bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in required
        },
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("qualification_artifacts/irl3_recurrent_utility"),
    )
    parser.add_argument("--starting-checkpoint", type=Path, required=True)
    parser.add_argument("--positive-checkpoint", type=Path)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--stage-b-policy", choices=tuple(POLICIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_root = args.artifact_root.resolve()
    starting_checkpoint = args.starting_checkpoint.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_root / "experiment_manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "format_version": 1,
            "source_revision": _git(repo_root, "rev-parse", "HEAD"),
            "source_status": _git(repo_root, "status", "--short"),
            "tokens_per_step": TOKENS_PER_STEP,
            "starting_checkpoint_identity": checkpoint_identity(starting_checkpoint),
        }
    if args.positive_checkpoint is not None:
        positive = args.positive_checkpoint.resolve()
        if "positive_checkpoint_identity" not in manifest:
            manifest["positive_checkpoint_identity"] = checkpoint_identity(positive)
    if args.stage == "a":
        manifest["stage_a"] = generate_stage_a(
            repo_root,
            artifact_root,
            starting_checkpoint,
        )
    else:
        if args.stage_b_policy is None:
            raise ValueError("--stage-b-policy is required for Stage B")
        manifest["stage_b"] = generate_stage_b(
            repo_root,
            artifact_root,
            starting_checkpoint,
            args.stage_b_policy,
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
