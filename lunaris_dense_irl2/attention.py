"""Grouped-query causal attention with QK-Norm and local/global masking."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import LayerKVCache
from .config import LunarisDenseConfig
from .norms import RMSNorm
from .rope import apply_rotary_embedding

try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _FLEX_ATTENTION_AVAILABLE = True
except (ImportError, OSError):
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    _FLEX_ATTENTION_AVAILABLE = False

try:
    from flash_attn import flash_attn_func

    _FLASH_ATTENTION_AVAILABLE = True
except (ImportError, OSError):
    flash_attn_func = None  # type: ignore[assignment]
    _FLASH_ATTENTION_AVAILABLE = False


def flash_attention_available() -> bool:
    """Return whether the optional FlashAttention training backend imported."""

    return _FLASH_ATTENTION_AVAILABLE


@lru_cache(maxsize=1)
def _compiled_flex_attention() -> Any:
    """Compile FlexAttention once and reuse its regional graph across layers."""

    if flex_attention is None:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")
    return torch.compile(
        flex_attention,
        dynamic=False,
        fullgraph=True,
    )


@lru_cache(maxsize=64)
def _cached_local_block_mask(
    sequence_length: int,
    sliding_window: int,
    device_name: str,
    block_size: int,
) -> Any:
    """Create one reusable causal sliding-window BlockMask per shape/device."""

    if create_block_mask is None:
        raise RuntimeError("FlexAttention is not available in this PyTorch build")

    def causal_local_mask(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        causal = query_index >= key_index
        in_window = query_index - key_index < sliding_window
        return causal & in_window

    return create_block_mask(
        causal_local_mask,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device_name,
        BLOCK_SIZE=block_size,
        _compile=True,
    )


def _expand_token_mask(
    token_mask: torch.Tensor,
    *,
    batch_size: int,
    key_length: int,
    current_length: int,
    past_length: int,
) -> torch.Tensor:
    """Normalize a 2D token mask to include cached keys."""

    if token_mask.dim() != 2 or token_mask.size(0) != batch_size:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    valid = token_mask.to(torch.bool)
    if valid.size(1) == key_length:
        return valid
    if valid.size(1) == current_length and past_length > 0:
        prefix = torch.ones(
            batch_size,
            past_length,
            device=valid.device,
            dtype=torch.bool,
        )
        return torch.cat((prefix, valid), dim=-1)
    raise ValueError(
        f"attention_mask length must be {current_length} or {key_length}, "
        f"got {valid.size(1)}"
    )


def build_attention_mask(
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sliding_window: Optional[int],
    attention_mask: Optional[torch.Tensor],
    document_ids: Optional[torch.Tensor],
    past_length: int,
) -> torch.Tensor:
    """Build a boolean mask where True entries are visible to attention."""

    if query_positions.dim() != 2 or key_positions.dim() != 2:
        raise ValueError("Position tensors must have shape [batch, sequence]")
    batch_size, query_length = query_positions.shape
    key_length = key_positions.size(1)

    query = query_positions.unsqueeze(-1)
    key = key_positions.unsqueeze(-2)
    allowed = key <= query
    if sliding_window is not None:
        allowed = allowed & (key >= (query - sliding_window + 1))

    if attention_mask is not None:
        valid_keys = _expand_token_mask(
            attention_mask,
            batch_size=batch_size,
            key_length=key_length,
            current_length=query_length,
            past_length=past_length,
        )
        allowed = allowed & valid_keys.unsqueeze(1)

    if document_ids is not None:
        if past_length > 0:
            raise ValueError("document_ids are not supported with KV cache")
        if document_ids.shape != (batch_size, query_length):
            raise ValueError("document_ids must have shape [batch, sequence]")
        same_document = document_ids.unsqueeze(-1) == document_ids.unsqueeze(-2)
        allowed = allowed & same_document

    return allowed.unsqueeze(1)


class GroupedQueryAttention(nn.Module):
    """Causal GQA using native SDPA and a compact KV projection."""

    def __init__(self, config: LunarisDenseConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.attention_type = config.attention_type(layer_index)
        self.sliding_window = (
            config.local_window if self.attention_type == "local" else None
        )
        self.attention_backend = config.attention_backend
        self.flex_block_size = int(config.flex_block_size)
        self.regional_compile_enabled = True

        self.q_width = config.n_heads * config.head_dim
        self.kv_width = config.kv_dim
        self.qkv_proj = nn.Linear(
            config.d_model,
            self.q_width + 2 * self.kv_width,
            bias=False,
        )
        self.o_proj = nn.Linear(
            config.n_heads * config.head_dim,
            config.d_model,
            bias=False,
        )

        self.q_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps)
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps)
            if config.qk_norm
            else nn.Identity()
        )
        self.output_gate = (
            nn.Linear(config.d_model, config.n_heads, bias=False)
            if config.attention_output_gate
            else None
        )
        self.dropout_p = float(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Upgrade separate Q/K/V projections from version-one checkpoints."""

        fused_key = prefix + "qkv_proj.weight"
        source_keys = [
            prefix + "q_proj.weight",
            prefix + "k_proj.weight",
            prefix + "v_proj.weight",
        ]
        if fused_key not in state_dict and all(
            key in state_dict for key in source_keys
        ):
            state_dict[fused_key] = torch.cat(
                tuple(state_dict.pop(key) for key in source_keys),
                dim=0,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _use_flex_attention(
        self,
        query: torch.Tensor,
        *,
        layer_cache: Optional[LayerKVCache],
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> bool:
        """Select sparse FlexAttention only for its safe fused training path."""

        if (
            self.sliding_window is None
            or self.attention_backend not in {"auto", "flex"}
        ):
            return False
        supported = (
            _FLEX_ATTENTION_AVAILABLE
            and query.device.type == "cuda"
            and layer_cache is None
            and attention_mask is None
            and document_ids is None
            and query.size(2) > 1
            and (not self.training or self.dropout_p == 0.0)
        )
        if self.attention_backend == "flex" and not supported:
            raise RuntimeError(
                "attention_backend=flex requires a CUDA/ROCm device, "
                "FlexAttention support, no explicit masks/cache, and zero "
                "attention dropout during training"
            )
        return supported

    def _use_flash_attention(
        self,
        query: torch.Tensor,
        *,
        layer_cache: Optional[LayerKVCache],
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> bool:
        """Select FlashAttention for unmasked fixed-length training batches."""

        if self.attention_backend != "flash":
            return False
        supported = (
            _FLASH_ATTENTION_AVAILABLE
            and query.device.type == "cuda"
            and layer_cache is None
            and attention_mask is None
            and document_ids is None
            and query.size(2) > 1
            and (not self.training or self.dropout_p == 0.0)
        )
        if self.attention_backend == "flash" and not supported:
            fast_path_requested = (
                query.device.type == "cuda"
                and layer_cache is None
                and attention_mask is None
                and document_ids is None
                and query.size(2) > 1
                and (not self.training or self.dropout_p == 0.0)
            )
            if fast_path_requested and not _FLASH_ATTENTION_AVAILABLE:
                raise RuntimeError(
                    "attention_backend=flash requires the flash-attn package"
                )
        return supported

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal FlashAttention, including the exact local window."""

        if flash_attn_func is None:
            raise RuntimeError("FlashAttention is unavailable")
        window_size = (
            (-1, -1)
            if self.sliding_window is None
            else (self.sliding_window - 1, 0)
        )
        query = query.to(value.dtype)
        key = key.to(value.dtype)
        attended = flash_attn_func(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            causal=True,
            window_size=window_size,
        )
        return attended.transpose(1, 2)

    def _flex_local_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run block-sparse causal local attention without a dense LxL mask."""

        if flex_attention is None or self.sliding_window is None:
            raise RuntimeError("FlexAttention local path is unavailable")
        block_mask = _cached_local_block_mask(
            query.size(2),
            self.sliding_window,
            str(query.device),
            self.flex_block_size,
        )
        regional_flex_attention = (
            _compiled_flex_attention()
            if self.regional_compile_enabled
            else flex_attention
        )
        return regional_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )

    def _native_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """Use native grouped SDPA with a compatibility fallback."""

        dropout_p = self.dropout_p if self.training else 0.0
        if self.n_heads == self.n_kv_heads:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        try:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )
        except TypeError:
            repeats = self.n_heads // self.n_kv_heads
            expanded_key = key.repeat_interleave(repeats, dim=1)
            expanded_value = value.repeat_interleave(repeats, dim=1)
            return F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Optional[LayerKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention and append current keys/values to the cache."""

        batch_size, sequence_length, _ = hidden_states.shape

        query, key, value = self.qkv_proj(hidden_states).split(
            (self.q_width, self.kv_width, self.kv_width),
            dim=-1,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.n_heads,
            self.head_dim,
        )
        key = key.view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )
        value = value.view(
            batch_size,
            sequence_length,
            self.n_kv_heads,
            self.head_dim,
        )

        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, rope_cos, rope_sin)

        past_key: Optional[torch.Tensor] = None
        past_value: Optional[torch.Tensor] = None
        past_positions: Optional[torch.Tensor] = None
        if layer_cache is not None:
            past_key, past_value, past_positions = layer_cache.get()

        past_length = 0 if past_key is None else past_key.size(2)
        if past_key is not None and past_value is not None:
            combined_key = torch.cat((past_key, key), dim=2)
            combined_value = torch.cat((past_value, value), dim=2)
            if past_positions is None:
                raise RuntimeError("Cached keys are missing absolute positions")
            key_positions = torch.cat(
                (
                    past_positions.unsqueeze(0).expand(batch_size, -1),
                    position_ids,
                ),
                dim=-1,
            )
        else:
            combined_key = key
            combined_value = value
            key_positions = position_ids

        use_flash_attention = self._use_flash_attention(
            query,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        use_flex_attention = self._use_flex_attention(
            query,
            layer_cache=layer_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        fast_causal_path = (
            layer_cache is None
            and self.sliding_window is None
            and attention_mask is None
            and document_ids is None
            and sequence_length > 1
        )
        if use_flash_attention:
            attended = self._flash_attention(
                query,
                combined_key,
                combined_value,
            )
        elif use_flex_attention:
            attended = self._flex_local_attention(
                query,
                combined_key,
                combined_value,
            )
        elif fast_causal_path:
            sdpa_mask = None
            is_causal = True
        else:
            sdpa_mask = build_attention_mask(
                query_positions=position_ids,
                key_positions=key_positions,
                sliding_window=self.sliding_window,
                attention_mask=attention_mask,
                document_ids=document_ids,
                past_length=past_length,
            )
            is_causal = False

        if not use_flash_attention and not use_flex_attention:
            attended = self._native_sdpa(
                query,
                combined_key,
                combined_value,
                attention_mask=sdpa_mask,
                is_causal=is_causal,
            )

        if self.output_gate is not None:
            gate = torch.sigmoid(self.output_gate(hidden_states))
            attended = attended * gate.transpose(1, 2).unsqueeze(-1).to(attended.dtype)

        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.n_heads * self.head_dim,
            )
        )
        output = self.output_dropout(self.o_proj(attended))

        if layer_cache is not None:
            layer_cache.append(key.detach(), value.detach(), position_ids)

        return output
