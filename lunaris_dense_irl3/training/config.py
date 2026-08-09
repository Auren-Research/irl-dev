"""Strict, resumable training configuration for Lunaris IRL-3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, TypeVar

from lunaris_dense_irl2.training.config import (
    DataConfig,
    LoggingConfig,
    OptimizerConfig,
    ParallelConfig,
    ScheduleConfig,
    TrainingConfig,
    ValidationConfig,
)

from ..config import LunarisIRL3Config

T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Dict[str, Any], section: str) -> T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown fields in `{section}`: {unknown}")
    return cls(**values)


@dataclass
class ObjectiveConfig:
    """Curricula for compute, credit assignment, and auxiliary objectives."""

    compute_start_steps: int = 2
    compute_ramp_steps: int = 10_000
    depth_sampling: str = "log_uniform"
    depth_sampling_seed: int = 17
    training_halt_mode: str = "act"

    gradient_horizon_start: int = 2
    gradient_horizon_ramp_steps: int = 10_000

    ponder_weight: float = 1e-3
    ponder_warmup_steps: int = 4_000
    ponder_ramp_steps: int = 6_000
    equilibrium_weight: float = 1e-3
    equilibrium_warmup_steps: int = 1_000
    equilibrium_ramp_steps: int = 4_000

    mtp_loss_weight: float = 0.0
    mtp_warmup_steps: int = 2_000
    mtp_ramp_steps: int = 8_000
    mtp_horizon_interval: int = 4_000
    mtp_horizon_decay: float = 0.7

    def validate(self, model: LunarisIRL3Config) -> None:
        self.compute_start_steps = int(self.compute_start_steps)
        if not (
            model.irl3_min_steps <= self.compute_start_steps <= model.irl3_train_steps
        ):
            raise ValueError(
                "objectives.compute_start_steps must be between the model's "
                "minimum and train recurrence depths"
            )
        self.gradient_horizon_start = int(self.gradient_horizon_start)
        if not (1 <= self.gradient_horizon_start <= model.irl3_gradient_horizon):
            raise ValueError(
                "objectives.gradient_horizon_start must be in "
                "[1, model.irl3_gradient_horizon]"
            )
        self.depth_sampling = self.depth_sampling.lower()
        if self.depth_sampling not in {"fixed", "uniform", "log_uniform"}:
            raise ValueError(
                "objectives.depth_sampling must be fixed, uniform, or log_uniform"
            )
        self.depth_sampling_seed = int(self.depth_sampling_seed)
        self.training_halt_mode = self.training_halt_mode.lower()
        if self.training_halt_mode not in {"act", "fixed"}:
            raise ValueError(
                "objectives.training_halt_mode must be act or fixed"
            )
        nonnegative_steps = {
            "compute_ramp_steps": self.compute_ramp_steps,
            "gradient_horizon_ramp_steps": self.gradient_horizon_ramp_steps,
            "ponder_warmup_steps": self.ponder_warmup_steps,
            "ponder_ramp_steps": self.ponder_ramp_steps,
            "equilibrium_warmup_steps": self.equilibrium_warmup_steps,
            "equilibrium_ramp_steps": self.equilibrium_ramp_steps,
            "mtp_warmup_steps": self.mtp_warmup_steps,
            "mtp_ramp_steps": self.mtp_ramp_steps,
            "mtp_horizon_interval": self.mtp_horizon_interval,
        }
        for name, value in nonnegative_steps.items():
            if int(value) < 0:
                raise ValueError(f"objectives.{name} cannot be negative")
            setattr(self, name, int(value))
        for name in (
            "ponder_weight",
            "equilibrium_weight",
            "mtp_loss_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"objectives.{name} cannot be negative")
        if self.mtp_horizon_decay <= 0.0:
            raise ValueError("objectives.mtp_horizon_decay must be positive")
        if self.mtp_loss_weight > 0.0 and model.mtp_num_heads <= 0:
            raise ValueError(
                "objectives.mtp_loss_weight requires model.mtp_num_heads > 0"
            )
        if model.mtp_num_heads > 0 and self.mtp_loss_weight <= 0.0:
            raise ValueError(
                "A training model with MTP heads requires a positive "
                "objectives.mtp_loss_weight"
            )


@dataclass
class CheckpointConfig:
    """Distributed checkpoint controls with an IRL-3-specific default root."""

    directory: str = "checkpoints/lunaris_dense_irl3"
    interval: int = 1_000
    keep_last: int = 3
    async_save: bool = True
    resume: str = "auto"
    save_final: bool = True
    min_free_space_gib: float = 0.0

    def validate(self) -> None:
        if self.interval < 0:
            raise ValueError("checkpoint.interval cannot be negative")
        if self.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be positive")
        if self.min_free_space_gib < 0.0:
            raise ValueError("checkpoint.min_free_space_gib cannot be negative")


@dataclass
class TrainConfig:
    """Complete IRL-3 pretraining configuration."""

    model: LunarisIRL3Config = field(default_factory=LunarisIRL3Config)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    objectives: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        self.data.validate(self.model)  # type: ignore[arg-type]
        self.optimizer.validate()
        self.parallel.validate()
        self.training.validate()
        self.objectives.validate(self.model)
        self.validation.validate()
        self.checkpoint.validate()
        self.logging.validate()
        self.schedule.validate(self.training.max_steps)
        if self.model.track_irl3_stats and self.model.use_gradient_checkpointing:
            raise ValueError(
                "model.track_irl3_stats must be false with gradient checkpointing"
            )
        if (
            self.training.loss_backend == "native"
            and self.training.z_loss_coefficient > 0.0
        ):
            raise ValueError(
                "training.loss_backend=native requires z_loss_coefficient=0"
            )
        if self.optimizer.name == "mona" and self.parallel.strategy == "fsdp2":
            raise ValueError("MONA is not supported with FSDP2 parameter sharding")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for training configs") from exc
        config_path = Path(path).resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("The training YAML root must be a mapping")
        model_profile = payload.pop("model_profile", None)
        model_values = payload.pop("model", {})
        if not isinstance(model_values, dict):
            raise ValueError("`model` must be a mapping")
        if model_profile is not None:
            profile_path = Path(model_profile)
            if not profile_path.is_absolute():
                profile_path = config_path.parent / profile_path
            profile_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            if not isinstance(profile_payload, dict):
                raise ValueError("The model profile must contain a mapping")
            profile_payload.update(model_values)
            model_values = profile_payload
        section_types = {
            "data": DataConfig,
            "optimizer": OptimizerConfig,
            "schedule": ScheduleConfig,
            "parallel": ParallelConfig,
            "training": TrainingConfig,
            "objectives": ObjectiveConfig,
            "validation": ValidationConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
        }
        unknown = sorted(set(payload) - set(section_types))
        if unknown:
            raise ValueError(f"Unknown training config sections: {unknown}")
        config = cls(model=LunarisIRL3Config.from_dict(model_values))
        for name, section_type in section_types.items():
            values = payload.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"`{name}` must be a mapping")
            setattr(config, name, _strict_dataclass(section_type, values, name))
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
