"""Preallocated global and ring-buffer local KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .config import LunarisDenseConfig


class LayerKVCache:
    """Store one layer's key/value states in chronological order."""

    def __init__(
        self,
        *,
        capacity: int,
        is_local: bool,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.capacity = int(capacity)
        self.is_local = bool(is_local)
        self.batch_size = int(batch_size)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)

        self.key: Optional[torch.Tensor] = None
        self.value: Optional[torch.Tensor] = None
        self.positions: Optional[torch.Tensor] = None
        self.length = 0
        self.write_index = 0

    def _allocate(self, reference: torch.Tensor) -> None:
        """Allocate cache storage on the first update."""

        shape = (
            self.batch_size,
            self.n_kv_heads,
            self.capacity,
            self.head_dim,
        )
        self.key = torch.empty(shape, device=reference.device, dtype=reference.dtype)
        self.value = torch.empty_like(self.key)
        self.positions = torch.empty(
            self.capacity,
            device=reference.device,
            dtype=torch.long,
        )

    def _validate(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Validate update tensor shapes and cache placement."""

        if key.shape != value.shape:
            raise ValueError("Key and value tensors must have identical shapes")
        if key.dim() != 4:
            raise ValueError(
                "KV tensors must have shape [batch, kv_heads, sequence, head_dim]"
            )
        expected = (self.batch_size, self.n_kv_heads, self.head_dim)
        actual = (key.size(0), key.size(1), key.size(3))
        if actual != expected:
            raise ValueError(f"Expected KV dimensions {expected}, got {actual}")
        if self.key is not None:
            if key.device != self.key.device:
                raise ValueError("Cache and KV tensors must be on the same device")
            if key.dtype != self.key.dtype:
                raise ValueError("Cache and KV tensors must use the same dtype")

    def get(
        self,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return cached states ordered from oldest to newest."""

        if self.length == 0 or self.key is None or self.value is None:
            return None, None, None
        if self.positions is None:
            raise RuntimeError("Cache positions are missing")

        if self.is_local and self.length == self.capacity:
            indices = (
                torch.arange(self.capacity, device=self.key.device) + self.write_index
            ) % self.capacity
            return (
                self.key.index_select(2, indices),
                self.value.index_select(2, indices),
                self.positions.index_select(0, indices),
            )

        return (
            self.key[:, :, : self.length],
            self.value[:, :, : self.length],
            self.positions[: self.length],
        )

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Append new states, wrapping only for local attention."""

        self._validate(key, value)
        if self.key is None:
            self._allocate(key)
        if self.key is None or self.value is None or self.positions is None:
            raise RuntimeError("Cache allocation failed")

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape != (self.batch_size, key.size(2)):
            raise ValueError(
                "position_ids must match the cached batch and sequence dimensions"
            )
        if self.batch_size > 1:
            shared_positions = position_ids[:1].expand_as(position_ids)
            if not torch.equal(position_ids, shared_positions):
                raise ValueError(
                    "Cached generation requires identical positions across the batch"
                )

        sequence_length = key.size(2)
        positions = position_ids[0].to(device=key.device, dtype=torch.long)

        if not self.is_local:
            end = self.length + sequence_length
            if end > self.capacity:
                raise ValueError(
                    "Global cache capacity exceeded: "
                    f"requested {end}, capacity {self.capacity}"
                )
            self.key[:, :, self.length : end].copy_(key)
            self.value[:, :, self.length : end].copy_(value)
            self.positions[self.length : end].copy_(positions)
            self.length = end
            self.write_index = self.length
            return

        if sequence_length >= self.capacity:
            self.key.copy_(key[:, :, -self.capacity :])
            self.value.copy_(value[:, :, -self.capacity :])
            self.positions.copy_(positions[-self.capacity :])
            self.length = self.capacity
            self.write_index = 0
            return

        slots = (
            torch.arange(sequence_length, device=key.device) + self.write_index
        ) % self.capacity
        self.key.index_copy_(2, slots, key)
        self.value.index_copy_(2, slots, value)
        self.positions.index_copy_(0, slots, positions)
        self.write_index = int((self.write_index + sequence_length) % self.capacity)
        self.length = min(self.capacity, self.length + sequence_length)

    def reset(self) -> None:
        """Clear logical cache state without reallocating memory."""

        self.length = 0
        self.write_index = 0


@dataclass
class KVCache:
    """Model-wide collection of layer caches."""

    layers: List[LayerKVCache]
    max_batch_size: int
    max_seq_len: int
    seen_tokens: int = 0

    @classmethod
    def from_config(
        cls,
        config: LunarisDenseConfig,
        *,
        batch_size: int,
    ) -> "KVCache":
        """Create lazy cache descriptors for every model layer."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        layer_caches: List[LayerKVCache] = []
        for layer_index in range(config.n_layers):
            is_local = config.attention_type(layer_index) == "local"
            capacity = config.local_window if is_local else config.max_seq_len
            layer_caches.append(
                LayerKVCache(
                    capacity=capacity,
                    is_local=is_local,
                    batch_size=batch_size,
                    n_kv_heads=config.n_kv_heads,
                    head_dim=config.head_dim,
                )
            )
        return cls(
            layers=layer_caches,
            max_batch_size=batch_size,
            max_seq_len=config.max_seq_len,
        )

    def layer(self, index: int) -> LayerKVCache:
        """Return one layer cache."""

        return self.layers[index]

    def advance(self, token_count: int) -> None:
        """Advance the shared absolute position after a complete forward pass."""

        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        next_position = self.seen_tokens + int(token_count)
        if next_position > self.max_seq_len:
            raise ValueError(
                f"Cache position {next_position} exceeds max_seq_len={self.max_seq_len}"
            )
        self.seen_tokens = next_position

    def reset(self) -> None:
        """Reset every layer while preserving allocated storage."""

        for layer_cache in self.layers:
            layer_cache.reset()
        self.seen_tokens = 0
