"""Token KV, immutable-anchor KV, and optional recurrent-state caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from lunaris_dense_irl2.cache import LayerKVCache

from .config import LunarisIRL3Config
from .diagnostics import IRL3State


@dataclass
class IRL3Cache:
    """Cache for autoregressive token time and recurrent reasoning time."""

    layers: List[LayerKVCache]
    anchor: LayerKVCache
    max_batch_size: int
    max_seq_len: int
    seen_tokens: int = 0
    reasoning_state: Optional[IRL3State] = None

    @classmethod
    def from_config(
        cls,
        config: LunarisIRL3Config,
        *,
        batch_size: int,
    ) -> "IRL3Cache":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        layers: List[LayerKVCache] = []
        for layer_index in range(config.n_layers):
            is_local = config.attention_type(layer_index) == "local"
            capacity = config.local_window if is_local else config.max_seq_len
            layers.append(
                LayerKVCache(
                    capacity=capacity,
                    is_local=is_local,
                    batch_size=batch_size,
                    n_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                )
            )

        anchor_capacity = min(config.irl3_anchor_window, config.max_seq_len)
        anchor = LayerKVCache(
            capacity=anchor_capacity,
            is_local=anchor_capacity < config.max_seq_len,
            batch_size=batch_size,
            n_kv_heads=config.irl3_state_heads,
            head_dim=config.state_head_dim,
        )
        return cls(
            layers=layers,
            anchor=anchor,
            max_batch_size=batch_size,
            max_seq_len=config.max_seq_len,
        )

    def layer(self, index: int) -> LayerKVCache:
        return self.layers[index]

    def advance(self, token_count: int) -> None:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        next_position = self.seen_tokens + int(token_count)
        if next_position > self.max_seq_len:
            raise ValueError(
                f"Cache position {next_position} exceeds max_seq_len={self.max_seq_len}"
            )
        self.seen_tokens = next_position

    def reset(self) -> None:
        for layer_cache in self.layers:
            layer_cache.reset()
        self.anchor.reset()
        self.reasoning_state = None
        self.seen_tokens = 0
