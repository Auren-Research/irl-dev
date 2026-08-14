"""Configuration for the simplified Lunaris IRL Core architecture."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LunarisIRLCoreConfig:
    """Validated configuration for Prelude -> IRL Core -> Coda models."""

    # Model dimensions.
    vocab_size: int = 65_536
    d_model: int = 768
    n_prelude_layers: int = 13
    n_coda_layers: int = 2
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden_dim: int = 1_536
    max_seq_len: int = 4_096

    # Transformer behavior.
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    qk_norm: bool = True
    sandwich_norm: bool = False
    attention_output_gate: bool = False
    initializer_range: float = 0.02
    scale_residual_projections: bool = True

    # Ordinary-block attention layout. The IRL Core read is always global.
    attention_pattern: str = "LLLG"
    local_window: int = 1_024
    attention_backend: str = "auto"
    flex_block_size: int = 128
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    rope_scaling_factor: float = 1.0

    # Fixed-depth IRL Core recurrence.
    irl_train_iters: int = 6
    irl_eval_iters: int = 6
    irl_lambda_min: float = 0.2
    irl_lambda_max: float = 1.0
    irl_lambda_init: float = 0.9
    track_irl_stats: bool = False

    # Kept only so the shared training configuration remains compatible.
    # The simplified architecture intentionally does not instantiate MTP heads.
    mtp_num_heads: int = 0

    # Runtime behavior.
    use_cache: bool = True
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_policy: str = "core"
    final_logit_softcap: Optional[float] = None
    loss_ignore_index: int = -100

    # Token metadata.
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Checkpoint compatibility.
    architecture_name: str = "lunaris_dense_irl_core"
    checkpoint_format_version: int = 1

    def __post_init__(self) -> None:
        """Normalize public values and reject incompatible configurations."""

        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "max_seq_len": self.max_seq_len,
            "local_window": self.local_window,
            "irl_train_iters": self.irl_train_iters,
            "irl_eval_iters": self.irl_eval_iters,
            "flex_block_size": self.flex_block_size,
        }
        for name, value in positive_integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            setattr(self, name, int(value))

        for name in ("n_prelude_layers", "n_coda_layers"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            setattr(self, name, value)
        if self.n_ordinary_layers <= 0:
            raise ValueError("At least one prelude or coda layer is required")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.rms_norm_eps <= 0.0:
            raise ValueError("rms_norm_eps must be positive")
        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")

        self.attention_pattern = self.attention_pattern.upper().replace(" ", "")
        if not self.attention_pattern:
            raise ValueError("attention_pattern cannot be empty")
        invalid_attention_types = set(self.attention_pattern) - {"G", "L"}
        if invalid_attention_types:
            raise ValueError(
                "attention_pattern only accepts 'G' (global) and 'L' (local); "
                f"found {sorted(invalid_attention_types)}"
            )
        if "L" in self.attention_pattern and self.local_window > self.max_seq_len:
            self.local_window = self.max_seq_len
        self.attention_backend = self.attention_backend.lower()
        if self.attention_backend not in {"auto", "sdpa", "flex"}:
            raise ValueError("attention_backend must be auto, sdpa, or flex")
        if self.flex_block_size & (self.flex_block_size - 1):
            raise ValueError("flex_block_size must be a power of two")

        if self.rope_theta_global <= 0.0 or self.rope_theta_local <= 0.0:
            raise ValueError("RoPE theta values must be positive")
        if self.rope_scaling_factor <= 0.0:
            raise ValueError("rope_scaling_factor must be positive")
        if not (0.0 < self.irl_lambda_min < self.irl_lambda_max <= 1.0):
            raise ValueError("Expected 0 < irl_lambda_min < irl_lambda_max <= 1")
        if not (self.irl_lambda_min < self.irl_lambda_init < self.irl_lambda_max):
            raise ValueError(
                "irl_lambda_init must lie strictly inside the configured interval"
            )

        self.mtp_num_heads = int(self.mtp_num_heads)
        if self.mtp_num_heads != 0:
            raise ValueError("IRL Core intentionally requires mtp_num_heads=0")
        self.gradient_checkpointing_policy = (
            self.gradient_checkpointing_policy.lower()
        )
        if self.gradient_checkpointing_policy not in {"none", "block", "core"}:
            raise ValueError(
                "gradient_checkpointing_policy must be none, block, or core"
            )
        if self.final_logit_softcap is not None and self.final_logit_softcap <= 0.0:
            raise ValueError("final_logit_softcap must be positive when provided")
        if self.architecture_name != "lunaris_dense_irl_core":
            raise ValueError("architecture_name must be lunaris_dense_irl_core")
        if self.checkpoint_format_version <= 0:
            raise ValueError("checkpoint_format_version must be positive")

    @property
    def head_dim(self) -> int:
        """Return the per-head attention dimension."""

        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        """Return the combined key/value projection width."""

        return self.n_kv_heads * self.head_dim

    @property
    def n_ordinary_layers(self) -> int:
        """Return the total number of non-recurrent Transformer blocks."""

        return self.n_prelude_layers + self.n_coda_layers

    @property
    def core_layer_index(self) -> int:
        """Return the IRL Core's index in the model-wide KV cache."""

        return self.n_prelude_layers

    @property
    def n_layers(self) -> int:
        """Return the number of attention-bearing modules, including the core."""

        return self.n_ordinary_layers + 1

    def attention_type(self, layer_index: int) -> str:
        """Resolve attention type for a model-wide attention-layer index."""

        layer_index = int(layer_index)
        if not (0 <= layer_index < self.n_layers):
            raise IndexError(f"attention layer index out of range: {layer_index}")
        if layer_index == self.core_layer_index:
            return "global"
        ordinary_index = (
            layer_index
            if layer_index < self.core_layer_index
            else layer_index - 1
        )
        code = self.attention_pattern[
            ordinary_index % len(self.attention_pattern)
        ]
        return "local" if code == "L" else "global"

    def rope_theta(self, layer_index: int) -> float:
        """Resolve the RoPE base for a model-wide attention-layer index."""

        if self.attention_type(layer_index) == "local":
            return self.rope_theta_local
        return self.rope_theta_global

    def lambda_logit(self) -> float:
        """Return the unconstrained initialization for every damping channel."""

        normalized = (self.irl_lambda_init - self.irl_lambda_min) / (
            self.irl_lambda_max - self.irl_lambda_min
        )
        return math.log(normalized / (1.0 - normalized))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to plain Python values."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "LunarisIRLCoreConfig":
        """Build a config while rejecting unknown fields."""

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")
        return cls(**values)

    def save_json(self, path: str | Path) -> None:
        """Write a stable JSON configuration file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "LunarisIRLCoreConfig":
        """Load a JSON profile."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LunarisIRLCoreConfig":
        """Load a strict YAML profile."""

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML profiles") from exc
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("A configuration profile must contain a YAML mapping")
        return cls.from_dict(values)

    @classmethod
    def debug(cls, **overrides: Any) -> "LunarisIRLCoreConfig":
        """Return a tiny profile with production-identical execution paths."""

        values: Dict[str, Any] = {
            "vocab_size": 257,
            "d_model": 64,
            "n_prelude_layers": 2,
            "n_coda_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 96,
            "max_seq_len": 64,
            "attention_pattern": "LG",
            "local_window": 16,
            "irl_train_iters": 6,
            "irl_eval_iters": 6,
        }
        values.update(overrides)
        return cls(**values)
