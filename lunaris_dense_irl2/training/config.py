"""Validated configuration for large-scale Lunaris training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from ..config import LunarisDenseConfig
from .model_factory import model_config_from_dict

T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Dict[str, Any], section: str) -> T:
    """Construct a dataclass while rejecting unknown configuration fields."""

    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown fields in `{section}`: {unknown}")
    return cls(**values)


@dataclass
class DataConfig:
    """Memory-mapped token shard settings."""

    train_dir: str = "data/train"
    validation_dir: Optional[str] = "data/validation"
    shard_glob: str = "*.npy"
    sequence_length: Optional[int] = None
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    mmap_cache_size: int = 8
    shuffle: bool = True
    seed: int = 1337

    def validate(self, model: Any) -> None:
        """Validate data settings against the model context length."""

        if self.sequence_length is None:
            self.sequence_length = model.max_seq_len
        self.sequence_length = int(self.sequence_length)
        if self.sequence_length < 2:
            raise ValueError("data.sequence_length must be at least 2")
        if self.sequence_length > model.max_seq_len:
            raise ValueError("data.sequence_length cannot exceed model.max_seq_len")
        if (
            model.mtp_num_heads > 0
            and self.sequence_length <= model.mtp_num_heads + 1
        ):
            raise ValueError(
                "data.sequence_length must exceed the largest configured "
                "MTP horizon"
            )
        if self.num_workers < 0:
            raise ValueError("data.num_workers cannot be negative")
        if self.prefetch_factor <= 0:
            raise ValueError("data.prefetch_factor must be positive")
        if self.mmap_cache_size <= 0:
            raise ValueError("data.mmap_cache_size must be positive")


@dataclass
class OptimizerConfig:
    """Optimizer settings with a conservative AdamW default."""

    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    fused: str = "auto"

    # Experimental MONA settings.
    mona_momentum: float = 0.95
    mona_acceleration_beta: float = 0.95
    mona_acceleration_alpha: float = -0.1
    mona_ns_steps: int = 5
    mona_lr: float = 0.02

    def validate(self) -> None:
        """Validate optimizer ranges and normalize its name."""

        self.name = self.name.lower()
        if self.name not in {"adamw", "mona"}:
            raise ValueError("optimizer.name must be `adamw` or `mona`")
        if self.learning_rate <= 0.0 or self.mona_lr <= 0.0:
            raise ValueError("Optimizer learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay cannot be negative")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("optimizer.epsilon must be positive")
        if self.fused not in {"auto", "true", "false"}:
            raise ValueError("optimizer.fused must be auto, true, or false")
        if not (0.0 <= self.mona_momentum < 1.0):
            raise ValueError("optimizer.mona_momentum must be in [0, 1)")
        if not (0.0 <= self.mona_acceleration_beta < 1.0):
            raise ValueError("optimizer.mona_acceleration_beta must be in [0, 1)")
        if self.mona_ns_steps <= 0:
            raise ValueError("optimizer.mona_ns_steps must be positive")


@dataclass
class ScheduleConfig:
    """Learning-rate schedule settings."""

    name: str = "wsd"
    warmup_steps: int = 2_000
    stable_steps: Optional[int] = None
    decay_steps: Optional[int] = None
    min_lr_ratio: float = 0.1
    decay_shape: str = "cosine"

    def validate(self, max_steps: int) -> None:
        """Resolve schedule phases and validate their total length."""

        self.name = self.name.lower()
        if self.name not in {"wsd", "cosine", "constant"}:
            raise ValueError("schedule.name must be wsd, cosine, or constant")
        if self.warmup_steps < 0:
            raise ValueError("schedule.warmup_steps cannot be negative")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ValueError("schedule.min_lr_ratio must be in [0, 1]")
        if self.decay_shape not in {"cosine", "linear"}:
            raise ValueError("schedule.decay_shape must be cosine or linear")

        remaining = max(0, max_steps - self.warmup_steps)
        if self.name == "wsd":
            if self.decay_steps is None:
                self.decay_steps = max(1, int(0.1 * max_steps))
            if self.stable_steps is None:
                self.stable_steps = max(0, remaining - self.decay_steps)
            if self.warmup_steps + self.stable_steps + self.decay_steps > max_steps:
                raise ValueError("WSD phases exceed training.max_steps")
        else:
            self.stable_steps = 0
            self.decay_steps = remaining


@dataclass
class ParallelConfig:
    """Hardware-neutral PyTorch distributed settings."""

    strategy: str = "auto"
    precision: str = "bf16"
    reduce_dtype: str = "fp32"
    compile: bool = False
    regional_compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    reshard_after_forward: bool = True
    fsdp_replicate_group_size: int = 1
    ddp_static_graph: bool = False
    deterministic: bool = False
    deterministic_warn_only: bool = False
    timeout_minutes: int = 30

    def validate(self) -> None:
        """Validate distributed and precision choices."""

        self.strategy = self.strategy.lower()
        if self.strategy not in {"auto", "none", "ddp", "fsdp2"}:
            raise ValueError("parallel.strategy must be auto, none, ddp, or fsdp2")
        self.precision = self.precision.lower()
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("parallel.precision must be bf16, fp16, or fp32")
        self.reduce_dtype = self.reduce_dtype.lower()
        if self.reduce_dtype not in {"bf16", "fp32"}:
            raise ValueError("parallel.reduce_dtype must be bf16 or fp32")
        self.fsdp_replicate_group_size = int(self.fsdp_replicate_group_size)
        if self.fsdp_replicate_group_size <= 0:
            raise ValueError(
                "parallel.fsdp_replicate_group_size must be positive"
            )
        if self.compile and self.regional_compile:
            raise ValueError(
                "Whole-model and regional compilation are mutually exclusive; "
                "keep regional_compile=true for the validated Lunaris path"
            )
        if self.timeout_minutes <= 0:
            raise ValueError("parallel.timeout_minutes must be positive")


@dataclass
class TrainingConfig:
    """Core optimization loop settings."""

    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100_000
    max_tokens: Optional[int] = None
    seed: int = 1337
    gradient_clip: float = 1.0
    gradient_clip_mode: str = "global"
    adaptive_clip_beta: float = 0.98
    adaptive_clip_factor: float = 2.0
    loss_chunk_size: int = 4_096
    loss_backend: str = "auto"
    z_loss_coefficient: float = 0.0
    detect_anomaly: bool = False
    fail_on_nonfinite: bool = True

    def validate(self) -> None:
        """Validate loop controls."""

        integer_positive = {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
        }
        for name, value in integer_positive.items():
            if int(value) <= 0:
                raise ValueError(f"training.{name} must be positive")
            setattr(self, name, int(value))
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("training.max_tokens must be positive")
        if self.gradient_clip < 0.0:
            raise ValueError("training.gradient_clip cannot be negative")
        self.gradient_clip_mode = self.gradient_clip_mode.lower()
        if self.gradient_clip_mode not in {"global", "adaptive_group", "none"}:
            raise ValueError(
                "training.gradient_clip_mode must be global, adaptive_group, or none"
            )
        if not (0.0 <= self.adaptive_clip_beta < 1.0):
            raise ValueError("training.adaptive_clip_beta must be in [0, 1)")
        if self.adaptive_clip_factor <= 0.0:
            raise ValueError("training.adaptive_clip_factor must be positive")
        if self.loss_chunk_size < 0:
            raise ValueError("training.loss_chunk_size cannot be negative")
        self.loss_backend = self.loss_backend.lower()
        if self.loss_backend not in {"auto", "chunked", "native"}:
            raise ValueError(
                "training.loss_backend must be auto, chunked, or native"
            )
        if self.z_loss_coefficient < 0.0:
            raise ValueError("training.z_loss_coefficient cannot be negative")


@dataclass
class ObjectiveConfig:
    """Curricula for recurrent depth and auxiliary training objectives."""

    irl_start_iters: int = 1
    irl_ramp_steps: int = 10_000
    irl_consistency_weight: float = 0.01
    irl_consistency_warmup_steps: int = 5_000
    irl_consistency_ramp_steps: int = 5_000

    mtp_loss_weight: float = 0.0
    mtp_warmup_steps: int = 2_000
    mtp_ramp_steps: int = 8_000
    mtp_horizon_interval: int = 4_000
    mtp_horizon_decay: float = 0.7

    def validate(self, model: Any) -> None:
        """Validate auxiliary losses against the instantiated architecture."""

        self.irl_start_iters = int(self.irl_start_iters)
        if not (1 <= self.irl_start_iters <= model.irl_train_iters):
            raise ValueError(
                "objectives.irl_start_iters must be in [1, model.irl_train_iters]"
            )
        nonnegative_steps = {
            "irl_ramp_steps": self.irl_ramp_steps,
            "irl_consistency_warmup_steps": self.irl_consistency_warmup_steps,
            "irl_consistency_ramp_steps": self.irl_consistency_ramp_steps,
            "mtp_warmup_steps": self.mtp_warmup_steps,
            "mtp_ramp_steps": self.mtp_ramp_steps,
            "mtp_horizon_interval": self.mtp_horizon_interval,
        }
        for name, value in nonnegative_steps.items():
            if int(value) < 0:
                raise ValueError(f"objectives.{name} cannot be negative")
            setattr(self, name, int(value))
        if self.irl_consistency_weight < 0.0:
            raise ValueError("objectives.irl_consistency_weight cannot be negative")
        if self.mtp_loss_weight < 0.0:
            raise ValueError("objectives.mtp_loss_weight cannot be negative")
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
        if model.architecture_name == "lunaris_dense_transformer":
            if self.irl_start_iters != 1 or self.irl_ramp_steps != 0:
                raise ValueError(
                    "The Transformer baseline requires fixed compatibility "
                    "objectives.irl_start_iters=1 and irl_ramp_steps=0"
                )
            if self.irl_consistency_weight != 0.0:
                raise ValueError(
                    "The Transformer baseline requires "
                    "objectives.irl_consistency_weight=0"
                )
            if self.mtp_loss_weight != 0.0:
                raise ValueError(
                    "The Transformer baseline requires objectives.mtp_loss_weight=0"
                )
        elif model.architecture_name == "lunaris_dense_irl_core":
            if self.irl_start_iters != model.irl_train_iters:
                raise ValueError(
                    "IRL Core requires objectives.irl_start_iters to equal "
                    "model.irl_train_iters (fixed-depth training)"
                )
            if self.irl_ramp_steps != 0:
                raise ValueError(
                    "IRL Core requires objectives.irl_ramp_steps=0"
                )
            if self.irl_consistency_weight != 0.0:
                raise ValueError(
                    "IRL Core requires objectives.irl_consistency_weight=0"
                )
            if self.mtp_loss_weight != 0.0:
                raise ValueError("IRL Core requires objectives.mtp_loss_weight=0")


@dataclass
class ValidationConfig:
    """Periodic validation controls."""

    interval: int = 500
    batches: int = 50
    at_start: bool = False

    def validate(self) -> None:
        """Validate evaluation intervals."""

        if self.interval < 0 or self.batches < 0:
            raise ValueError("Validation intervals and batches cannot be negative")


@dataclass
class CheckpointConfig:
    """Distributed checkpoint controls."""

    directory: str = "checkpoints/lunaris_dense_irl2"
    interval: int = 1_000
    keep_last: int = 3
    async_save: bool = True
    resume: str = "auto"
    save_final: bool = True
    min_free_space_gib: float = 0.0

    def validate(self) -> None:
        """Validate checkpoint retention and resume mode."""

        if self.interval < 0:
            raise ValueError("checkpoint.interval cannot be negative")
        if self.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be positive")
        if self.min_free_space_gib < 0.0:
            raise ValueError("checkpoint.min_free_space_gib cannot be negative")


@dataclass
class LoggingConfig:
    """Console, JSONL, W&B, and profiler settings."""

    interval: int = 10
    jsonl_path: Optional[str] = "metrics.jsonl"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    profile_steps: int = 0
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_directory: str = "traces"
    peak_tflops_per_gpu: Optional[float] = None

    def validate(self) -> None:
        """Validate logging and profiler settings."""

        if self.interval <= 0:
            raise ValueError("logging.interval must be positive")
        if min(self.profile_steps, self.profile_wait, self.profile_warmup) < 0:
            raise ValueError("Profiler values cannot be negative")
        if self.peak_tflops_per_gpu is not None and self.peak_tflops_per_gpu <= 0:
            raise ValueError("logging.peak_tflops_per_gpu must be positive")


@dataclass
class TrainConfig:
    """Complete large-scale training configuration."""

    model: Any = field(default_factory=LunarisDenseConfig)
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
        """Validate all sections and derive token-based step limits."""

        self.data.validate(self.model)
        self.optimizer.validate()
        self.parallel.validate()
        self.training.validate()
        self.objectives.validate(self.model)
        self.validation.validate()
        self.checkpoint.validate()
        self.logging.validate()

        self.schedule.validate(self.training.max_steps)

        if self.model.track_irl_stats and self.model.use_gradient_checkpointing:
            raise ValueError(
                "model.track_irl_stats must be false with gradient checkpointing"
            )
        if (
            self.training.loss_backend == "native"
            and self.training.z_loss_coefficient > 0.0
        ):
            raise ValueError(
                "training.loss_backend=native requires z_loss_coefficient=0"
            )
        if self.optimizer.name == "mona" and self.parallel.strategy == "fsdp2":
            raise ValueError(
                "MONA requires full gradient matrices and is not supported with "
                "FSDP2 parameter sharding; use DDP or AdamW"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load a strict YAML config with an optional external model profile."""

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
        unknown_sections = sorted(set(payload) - set(section_types))
        if unknown_sections:
            raise ValueError(f"Unknown training config sections: {unknown_sections}")

        config = cls(model=model_config_from_dict(model_values))
        for name, section_type in section_types.items():
            values = payload.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"`{name}` must be a mapping")
            setattr(config, name, _strict_dataclass(section_type, values, name))
        config.validate()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fully resolved configuration."""

        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Write the resolved configuration as JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
