"""Validated configuration for the Lunaris Dense IRL-3 architecture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LunarisIRL3Config:
    """Configuration for a causal LM with one stateful latent computer.

    ``n_layers`` counts the ordinary Transformer layers surrounding the
    recurrent computer. ``irl3_coda_layers`` of those layers execute after
    latent write-back; all preceding layers construct the immutable anchor.
    """

    # Token model dimensions.
    vocab_size: int = 48_000
    d_model: int = 768
    n_layers: int = 12
    irl3_coda_layers: int = 2
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

    # Attention layout for the ordinary Transformer layers.
    attention_pattern: str = "G"
    local_window: int = 1_024
    attention_backend: str = "auto"
    flex_block_size: int = 128
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    rope_scaling_factor: float = 1.0

    # IRL-3 state capacity.
    irl3_state_dim: Optional[int] = None
    irl3_state_heads: int = 4
    irl3_state_ffn_dim: Optional[int] = None
    irl3_num_latents: int = 4
    irl3_memory_slots: int = 4
    irl3_slow_slots: int = 1
    irl3_anchor_window: int = 128

    # IRL-3 recurrent schedule and train-time credit assignment.
    irl3_train_steps: int = 4
    irl3_eval_steps: int = 6
    irl3_max_steps: int = 64
    irl3_min_steps: int = 2
    irl3_slow_update_interval: int = 2
    irl3_gradient_horizon: int = 4

    # Recurrent dynamics and learned halting.
    irl3_update_scale_init: float = 0.25
    irl3_writeback_scale_init: float = 0.1
    irl3_writeback_gate_init: float = 0.0
    irl3_memory_gate_init: float = -1.5
    irl3_halt_bias_init: float = -3.0
    irl3_halt_threshold: float = 0.95
    irl3_eval_halt_mode: str = "adaptive"
    irl3_active_gate_threshold: float = 0.1
    irl3_equilibrium_memory_weight: float = 0.1
    irl3_persist_state: bool = False
    track_irl3_stats: bool = False

    # Multi-token prediction from the final recurrent representation.
    mtp_num_heads: int = 0
    mtp_ffn_hidden_dim: Optional[int] = None

    # Runtime behavior.
    use_cache: bool = True
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_policy: str = "block"
    final_logit_softcap: Optional[float] = None
    loss_ignore_index: int = -100

    # Token metadata.
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Checkpoint identity.
    architecture_name: str = "lunaris_dense_irl3"
    checkpoint_format_version: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "max_seq_len": self.max_seq_len,
            "local_window": self.local_window,
            "flex_block_size": self.flex_block_size,
            "irl3_state_heads": self.irl3_state_heads,
            "irl3_num_latents": self.irl3_num_latents,
            "irl3_memory_slots": self.irl3_memory_slots,
            "irl3_slow_slots": self.irl3_slow_slots,
            "irl3_anchor_window": self.irl3_anchor_window,
            "irl3_train_steps": self.irl3_train_steps,
            "irl3_eval_steps": self.irl3_eval_steps,
            "irl3_max_steps": self.irl3_max_steps,
            "irl3_min_steps": self.irl3_min_steps,
            "irl3_slow_update_interval": self.irl3_slow_update_interval,
            "irl3_gradient_horizon": self.irl3_gradient_horizon,
        }
        for name, value in integer_fields.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
            setattr(self, name, int(value))

        self.irl3_coda_layers = int(self.irl3_coda_layers)
        if not (0 <= self.irl3_coda_layers < self.n_layers):
            raise ValueError("irl3_coda_layers must be in [0, n_layers)")

        if self.irl3_state_dim is None:
            self.irl3_state_dim = self.d_model
        self.irl3_state_dim = int(self.irl3_state_dim)
        if self.irl3_state_dim <= 0:
            raise ValueError("irl3_state_dim must be positive")
        if self.irl3_state_ffn_dim is None:
            self.irl3_state_ffn_dim = 2 * self.irl3_state_dim
        self.irl3_state_ffn_dim = int(self.irl3_state_ffn_dim)
        if self.irl3_state_ffn_dim <= 0:
            raise ValueError("irl3_state_ffn_dim must be positive")

        self.mtp_num_heads = int(self.mtp_num_heads)
        if self.mtp_num_heads < 0:
            raise ValueError("mtp_num_heads cannot be negative")
        if self.mtp_ffn_hidden_dim is not None:
            self.mtp_ffn_hidden_dim = int(self.mtp_ffn_hidden_dim)
            if self.mtp_ffn_hidden_dim <= 0:
                raise ValueError("mtp_ffn_hidden_dim must be positive when provided")
        if self.mtp_num_heads + 1 >= self.max_seq_len:
            raise ValueError("MTP horizons must be shorter than max_seq_len")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("The token-attention head dimension must be even for RoPE")
        if self.state_dim % self.irl3_state_heads != 0:
            raise ValueError("irl3_state_dim must be divisible by irl3_state_heads")

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
        invalid = set(self.attention_pattern) - {"G", "L"}
        if invalid:
            raise ValueError(
                f"attention_pattern only accepts 'G' and 'L'; found {sorted(invalid)}"
            )
        if "L" in self.attention_pattern and self.local_window > self.max_seq_len:
            self.local_window = self.max_seq_len
        self.irl3_anchor_window = min(self.irl3_anchor_window, self.max_seq_len)

        self.attention_backend = self.attention_backend.lower()
        if self.attention_backend not in {"auto", "sdpa", "flex"}:
            raise ValueError("attention_backend must be auto, sdpa, or flex")
        if self.flex_block_size & (self.flex_block_size - 1):
            raise ValueError("flex_block_size must be a power of two")
        if self.rope_theta_global <= 0.0 or self.rope_theta_local <= 0.0:
            raise ValueError("RoPE theta values must be positive")
        if self.rope_scaling_factor <= 0.0:
            raise ValueError("rope_scaling_factor must be positive")

        if not (self.irl3_min_steps <= self.irl3_train_steps <= self.irl3_max_steps):
            raise ValueError(
                "Expected irl3_min_steps <= irl3_train_steps <= irl3_max_steps"
            )
        if not (self.irl3_min_steps <= self.irl3_eval_steps <= self.irl3_max_steps):
            raise ValueError(
                "Expected irl3_min_steps <= irl3_eval_steps <= irl3_max_steps"
            )
        if self.irl3_gradient_horizon > self.irl3_max_steps:
            raise ValueError("irl3_gradient_horizon cannot exceed irl3_max_steps")
        if self.irl3_update_scale_init <= 0.0:
            raise ValueError("irl3_update_scale_init must be positive")
        if self.irl3_writeback_scale_init <= 0.0:
            raise ValueError("irl3_writeback_scale_init must be positive")
        if not (0.0 < self.irl3_halt_threshold < 1.0):
            raise ValueError("irl3_halt_threshold must be in (0, 1)")
        if not (0.0 <= self.irl3_active_gate_threshold <= 1.0):
            raise ValueError("irl3_active_gate_threshold must be in [0, 1]")
        if not (0.0 <= self.irl3_equilibrium_memory_weight <= 1.0):
            raise ValueError("irl3_equilibrium_memory_weight must be in [0, 1]")
        self.irl3_eval_halt_mode = self.irl3_eval_halt_mode.lower()
        if self.irl3_eval_halt_mode not in {"fixed", "act", "adaptive"}:
            raise ValueError("irl3_eval_halt_mode must be fixed, act, or adaptive")

        if self.gradient_checkpointing_policy not in {
            "none",
            "block",
            "computer",
        }:
            raise ValueError(
                "gradient_checkpointing_policy must be none, block, or computer"
            )
        if self.final_logit_softcap is not None and self.final_logit_softcap <= 0:
            raise ValueError("final_logit_softcap must be positive when provided")
        if self.checkpoint_format_version <= 0:
            raise ValueError("checkpoint_format_version must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def state_dim(self) -> int:
        if self.irl3_state_dim is None:
            raise RuntimeError("irl3_state_dim was not resolved")
        return int(self.irl3_state_dim)

    @property
    def state_head_dim(self) -> int:
        return self.state_dim // self.irl3_state_heads

    @property
    def state_ffn_dim(self) -> int:
        if self.irl3_state_ffn_dim is None:
            raise RuntimeError("irl3_state_ffn_dim was not resolved")
        return int(self.irl3_state_ffn_dim)

    @property
    def anchor_layers(self) -> int:
        return self.n_layers - self.irl3_coda_layers

    def attention_type(self, layer_index: int) -> str:
        code = self.attention_pattern[layer_index % len(self.attention_pattern)]
        return "local" if code == "L" else "global"

    def rope_theta(self, layer_index: int) -> float:
        if self.attention_type(layer_index) == "local":
            return self.rope_theta_local
        return self.rope_theta_global

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "LunarisIRL3Config":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {unknown}")
        return cls(**values)

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "LunarisIRL3Config":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LunarisIRL3Config":
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML profiles") from exc
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("A configuration profile must contain a YAML mapping")
        return cls.from_dict(values)

    @classmethod
    def debug(cls, **overrides: Any) -> "LunarisIRL3Config":
        values: Dict[str, Any] = {
            "vocab_size": 512,
            "d_model": 96,
            "n_layers": 4,
            "irl3_coda_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 2,
            "ffn_hidden_dim": 192,
            "max_seq_len": 128,
            "attention_pattern": "LG",
            "local_window": 32,
            "irl3_state_dim": 64,
            "irl3_state_heads": 4,
            "irl3_state_ffn_dim": 128,
            "irl3_num_latents": 3,
            "irl3_memory_slots": 3,
            "irl3_slow_slots": 1,
            "irl3_anchor_window": 16,
            "irl3_train_steps": 4,
            "irl3_eval_steps": 4,
            "irl3_max_steps": 12,
            "irl3_min_steps": 2,
            "irl3_gradient_horizon": 4,
        }
        values.update(overrides)
        return cls(**values)
