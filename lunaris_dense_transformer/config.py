"""Configuration for the controlled conventional Transformer baseline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LunarisTransformerConfig:
    """Validated decoder-only Transformer configuration with no recurrence."""

    vocab_size: int = 65_536
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_hidden_dim: int = 1_536
    max_seq_len: int = 4_096

    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    qk_norm: bool = True
    sandwich_norm: bool = False
    attention_output_gate: bool = False
    initializer_range: float = 0.02
    scale_residual_projections: bool = True

    attention_pattern: str = "LLLG"
    local_window: int = 1_024
    attention_backend: str = "auto"
    flex_block_size: int = 128
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    rope_scaling_factor: float = 1.0

    use_cache: bool = True
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_policy: str = "block"
    final_logit_softcap: Optional[float] = None
    loss_ignore_index: int = -100

    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    architecture_name: str = "lunaris_dense_transformer"
    checkpoint_format_version: int = 1

    def __post_init__(self) -> None:
        """Normalize public values and reject incompatible configurations."""

        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "max_seq_len": self.max_seq_len,
            "local_window": self.local_window,
            "flex_block_size": self.flex_block_size,
        }
        for name, value in positive_integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            setattr(self, name, int(value))

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

        self.gradient_checkpointing_policy = (
            self.gradient_checkpointing_policy.lower()
        )
        if self.gradient_checkpointing_policy not in {"none", "block"}:
            raise ValueError(
                "gradient_checkpointing_policy must be none or block"
            )
        if self.final_logit_softcap is not None and self.final_logit_softcap <= 0.0:
            raise ValueError("final_logit_softcap must be positive when provided")
        if self.architecture_name != "lunaris_dense_transformer":
            raise ValueError("architecture_name must be lunaris_dense_transformer")
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
    def irl_train_iters(self) -> int:
        """Expose one compatibility iteration without instantiating recurrence."""

        return 1

    @property
    def irl_eval_iters(self) -> int:
        """Expose one compatibility iteration without instantiating recurrence."""

        return 1

    @property
    def mtp_num_heads(self) -> int:
        """The controlled baseline has no MTP parameters."""

        return 0

    @property
    def mtp_irl_iters(self) -> int:
        """Compatibility value unused because MTP is absent."""

        return 1

    @property
    def mtp_ffn_hidden_dim(self) -> None:
        """Compatibility value unused because MTP is absent."""

        return None

    @property
    def track_irl_stats(self) -> bool:
        """The conventional baseline has no recurrent statistics."""

        return False

    def attention_type(self, layer_index: int) -> str:
        """Resolve the attention type for an ordinary Transformer layer."""

        layer_index = int(layer_index)
        if not (0 <= layer_index < self.n_layers):
            raise IndexError(f"attention layer index out of range: {layer_index}")
        code = self.attention_pattern[layer_index % len(self.attention_pattern)]
        return "local" if code == "L" else "global"

    def rope_theta(self, layer_index: int) -> float:
        """Resolve the RoPE base for one layer."""

        if self.attention_type(layer_index) == "local":
            return self.rope_theta_local
        return self.rope_theta_global

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this configuration to plain Python values."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "LunarisTransformerConfig":
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
    def from_json(cls, path: str | Path) -> "LunarisTransformerConfig":
        """Load a strict JSON profile."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LunarisTransformerConfig":
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
    def debug(cls, **overrides: Any) -> "LunarisTransformerConfig":
        """Return a tiny profile with production-identical execution paths."""

        values: Dict[str, Any] = {
            "vocab_size": 257,
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 96,
            "max_seq_len": 64,
            "attention_pattern": "LG",
            "local_window": 16,
        }
        values.update(overrides)
        return cls(**values)
