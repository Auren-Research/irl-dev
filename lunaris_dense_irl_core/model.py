"""Complete Prelude -> IRL Core -> Coda causal language model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lunaris_dense_irl2.cache import KVCache
from lunaris_dense_irl2.initialization import initialize_module
from lunaris_dense_irl2.model import (
    _compiled_linear_loss_chunk,
    _linear_loss_chunk_sums,
)
from lunaris_dense_irl2.norms import RMSNorm
from lunaris_dense_irl2.rope import RotaryEmbedding

from .block import TransformerBlock
from .config import LunarisIRLCoreConfig
from .core import IRLCore
from .diagnostics import IRLCoreDiagnostics, LunarisIRLCoreCausalLMOutput


class LunarisIRLCoreForCausalLM(nn.Module):
    """Decoder-only LM with one shared recurrent relational core."""

    def __init__(self, config: LunarisIRLCoreConfig) -> None:
        super().__init__()
        self.config = config
        self.regional_compile_enabled = True

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.prelude_blocks = nn.ModuleList(
            [
                TransformerBlock(config, layer_index)
                for layer_index in range(config.n_prelude_layers)
            ]
        )
        self.irl_core = IRLCore(config)
        self.coda_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    config.core_layer_index + 1 + coda_index,
                )
                for coda_index in range(config.n_coda_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.global_rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta_global,
            scaling_factor=config.rope_scaling_factor,
        )
        self.local_rope = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta_local,
            scaling_factor=config.rope_scaling_factor,
        )

        self.apply(lambda module: initialize_module(module, config))
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.scale_residual_projections:
            self._scale_residual_projections()

    @property
    def blocks(self) -> List[nn.Module]:
        """Expose attention-bearing modules for shared FSDP wrapping."""

        return [*self.prelude_blocks, self.irl_core, *self.coda_blocks]

    def _scale_residual_projections(self) -> None:
        """Apply the existing depth-aware residual projection scaling."""

        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        with torch.no_grad():
            for block in (*self.prelude_blocks, *self.coda_blocks):
                block.attention.o_proj.weight.mul_(scale)
                block.ffn.down_proj.weight.mul_(scale)
            self.irl_core.o_proj.weight.mul_(scale)
            self.irl_core.ffn.down_proj.weight.mul_(scale)

    def set_regional_compile(self, enabled: bool) -> None:
        """Enable or disable bounded compilation in reused attention/loss paths."""

        self.regional_compile_enabled = bool(enabled)
        for module in (*self.prelude_blocks, *self.coda_blocks):
            module.attention.regional_compile_enabled = bool(enabled)

    def get_num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count all unique model parameters."""

        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in self.parameters())

    def get_irl_num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count parameters belonging to the one recurrent core."""

        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in self.irl_core.parameters()
                if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in self.irl_core.parameters())

    def effective_attention_calls(self, n_iters: Optional[int] = None) -> int:
        """Return ordinary attention calls plus recurrent anchor reads."""

        iterations = (
            self.config.irl_train_iters if n_iters is None else int(n_iters)
        )
        if iterations <= 0:
            raise ValueError("n_iters must be positive")
        return self.config.n_ordinary_layers + iterations

    def init_cache(self, *, batch_size: int) -> KVCache:
        """Create a cache for prelude, core-anchor, and coda attention."""

        return KVCache.from_config(self.config, batch_size=batch_size)

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cache: Optional[KVCache],
    ) -> torch.Tensor:
        """Resolve absolute positions for training, prefill, or decoding."""

        batch_size, sequence_length = input_ids.shape
        explicit_position_ids = position_ids is not None
        if position_ids is None:
            start = 0 if cache is None else cache.seen_tokens
            positions = torch.arange(
                start,
                start + sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            )
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if position_ids.shape != (batch_size, sequence_length):
            raise ValueError("position_ids must match input_ids shape")
        if explicit_position_ids and not torch.compiler.is_compiling():
            if bool(position_ids.lt(0).any()):
                raise ValueError("position_ids cannot be negative")
            if bool(position_ids.ge(self.config.max_seq_len).any()):
                raise ValueError(
                    "position_ids must be smaller than "
                    f"max_seq_len={self.config.max_seq_len}"
                )
        return position_ids

    def _rope_for_layer(
        self,
        layer_index: int,
        *,
        global_cos: torch.Tensor,
        global_sin: torch.Tensor,
        local_cos: Optional[torch.Tensor],
        local_sin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select already-computed global or local RoPE values."""

        if self.config.attention_type(layer_index) == "global":
            return global_cos, global_sin
        if local_cos is None or local_sin is None:
            raise RuntimeError("Local RoPE values were not initialized")
        return local_cos, local_sin

    def _run_transformer_block(
        self,
        block: TransformerBlock,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Optional[Any],
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run or checkpoint one ordinary Transformer block."""

        should_checkpoint = (
            self.training
            and self.config.use_gradient_checkpointing
            and self.config.gradient_checkpointing_policy == "block"
            and layer_cache is None
        )
        if not should_checkpoint:
            return block(
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )

        def block_forward(value: torch.Tensor) -> torch.Tensor:
            return block(
                value,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=None,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )

        return checkpoint(block_forward, hidden_states, use_reentrant=False)

    def _run_core(
        self,
        anchor: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Optional[Any],
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
        n_iters: Optional[int],
        return_stats: bool,
    ) -> Tuple[torch.Tensor, Optional[IRLCoreDiagnostics]]:
        """Run the core directly or checkpoint its complete tensor-only path."""

        should_checkpoint = (
            self.training
            and self.config.use_gradient_checkpointing
            and self.config.gradient_checkpointing_policy == "block"
            and layer_cache is None
            and not return_stats
        )
        if not should_checkpoint:
            return self.irl_core(
                anchor,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
                document_ids=document_ids,
                n_iters=n_iters,
                return_stats=return_stats,
            )

        def core_forward(value: torch.Tensor) -> torch.Tensor:
            output, _ = self.irl_core(
                value,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=None,
                attention_mask=attention_mask,
                document_ids=document_ids,
                n_iters=n_iters,
                return_stats=False,
            )
            return output

        return checkpoint(core_forward, anchor, use_reentrant=False), None

    def _apply_logit_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured smooth logit bound."""

        if self.config.final_logit_softcap is None:
            return logits
        cap = float(self.config.final_logit_softcap)
        return torch.tanh(logits / cap) * cap

    def _prepare_targets(
        self,
        labels: torch.Tensor,
        *,
        horizon: int,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Align labels and suppress invalid or cross-document predictions."""

        targets = labels[:, horizon:]
        if attention_mask is None and document_ids is None:
            return targets
        valid = torch.ones_like(targets, dtype=torch.bool)
        if attention_mask is not None:
            if attention_mask.shape != labels.shape:
                raise ValueError("attention_mask must have the same shape as labels")
            token_valid = attention_mask.to(torch.bool)
            valid = valid & token_valid[:, :-horizon] & token_valid[:, horizon:]
        if document_ids is not None:
            if document_ids.shape != labels.shape:
                raise ValueError("document_ids must have the same shape as labels")
            valid = valid & document_ids[:, :-horizon].eq(document_ids[:, horizon:])
        return targets.masked_fill(~valid, self.config.loss_ignore_index)

    def _native_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        *,
        chunk_size: int,
    ) -> Optional[torch.Tensor]:
        """Use the version-gated fused linear cross entropy when available."""

        linear_cross_entropy = getattr(F, "linear_cross_entropy", None)
        options_type = getattr(nn, "LinearCrossEntropyOptions", None)
        if (
            linear_cross_entropy is None
            or options_type is None
            or chunk_size <= 0
            or self.config.final_logit_softcap is not None
        ):
            return None
        options = options_type(batch_chunk_size=chunk_size)
        loss_sum = linear_cross_entropy(
            hidden_states,
            self.lm_head.weight,
            labels,
            reduction="sum",
            ignore_index=self.config.loss_ignore_index,
            options=options,
        )
        denominator = labels.ne(self.config.loss_ignore_index).sum().to(
            dtype=torch.float32
        )
        return loss_sum / denominator.clamp_min(1.0)

    def _chunked_token_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        *,
        chunk_size: int,
        z_loss_coefficient: float,
        loss_backend: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aligned loss without retaining full vocabulary logits."""

        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_labels = labels.reshape(-1)
        valid_tokens = flat_labels.ne(self.config.loss_ignore_index).sum()

        native_loss = None
        if z_loss_coefficient == 0.0 and loss_backend in {"auto", "native"}:
            native_loss = self._native_linear_cross_entropy(
                flat_hidden,
                flat_labels,
                chunk_size=chunk_size,
            )
        if native_loss is not None:
            cross_entropy = native_loss.float()
            z_loss = cross_entropy.new_zeros(())
            return cross_entropy, cross_entropy, z_loss
        if loss_backend == "native":
            raise RuntimeError(
                "Native linear_cross_entropy requires a supporting PyTorch "
                "build, a positive chunk size, no logit softcap, and zero z-loss"
            )

        rows_per_chunk = chunk_size if chunk_size > 0 else flat_hidden.size(0)
        ce_sum = hidden_states.new_zeros((), dtype=torch.float32)
        z_sum = hidden_states.new_zeros((), dtype=torch.float32)
        use_regional_compile = (
            self.regional_compile_enabled
            and flat_hidden.device.type == "cuda"
            and chunk_size > 0
        )
        if use_regional_compile:
            padding = (-flat_hidden.size(0)) % rows_per_chunk
            if padding:
                flat_hidden = F.pad(flat_hidden, (0, 0, 0, padding))
                flat_labels = F.pad(
                    flat_labels,
                    (0, padding),
                    value=self.config.loss_ignore_index,
                )
            chunk_loss = _compiled_linear_loss_chunk()
        else:
            chunk_loss = _linear_loss_chunk_sums
        logit_softcap = (
            0.0
            if self.config.final_logit_softcap is None
            else float(self.config.final_logit_softcap)
        )

        for start in range(0, flat_hidden.size(0), rows_per_chunk):
            end = min(start + rows_per_chunk, flat_hidden.size(0))
            chunk_ce, chunk_z = chunk_loss(
                flat_hidden[start:end],
                self.lm_head.weight,
                flat_labels[start:end],
                self.config.loss_ignore_index,
                float(z_loss_coefficient),
                logit_softcap,
            )
            ce_sum = ce_sum + chunk_ce
            z_sum = z_sum + chunk_z

        denominator = valid_tokens.to(dtype=torch.float32).clamp_min(1.0)
        cross_entropy = ce_sum / denominator
        z_loss = z_sum / denominator
        total_loss = cross_entropy + float(z_loss_coefficient) * z_loss
        return total_loss, cross_entropy, z_loss

    def _token_loss_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        z_loss_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aligned loss from materialized logits."""

        aligned_logits = logits.contiguous().float()
        aligned_labels = labels.contiguous()
        valid = aligned_labels.ne(self.config.loss_ignore_index)
        denominator = valid.sum().to(dtype=torch.float32).clamp_min(1.0)
        cross_entropy = F.cross_entropy(
            aligned_logits.view(-1, aligned_logits.size(-1)),
            aligned_labels.view(-1),
            ignore_index=self.config.loss_ignore_index,
            reduction="sum",
        ) / denominator
        z_loss = aligned_logits.new_zeros(())
        if z_loss_coefficient > 0.0:
            log_partition = torch.logsumexp(aligned_logits, dim=-1)
            z_loss = log_partition.square().masked_select(valid).sum() / denominator
        total_loss = cross_entropy + float(z_loss_coefficient) * z_loss
        return total_loss, cross_entropy, z_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        n_iters: Optional[int] = None,
        return_irl_stats: Optional[bool] = None,
        logits_to_keep: int = 0,
        return_logits: bool = True,
        loss_chunk_size: int = 0,
        z_loss_coefficient: float = 0.0,
        loss_backend: str = "auto",
        mtp_loss_weight: float = 0.0,
        mtp_active_heads: Optional[int] = None,
        mtp_horizon_decay: float = 0.7,
        irl_consistency_coefficient: float = 0.0,
    ) -> LunarisIRLCoreCausalLMOutput:
        """Run terminal-state causal LM training or cache-aware inference."""

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor")
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")
        if cache is not None:
            if self.training:
                raise ValueError("KV cache is only supported in evaluation mode")
            if input_ids.size(0) != cache.max_batch_size:
                raise ValueError("Input batch size does not match the KV cache")
            if document_ids is not None:
                raise ValueError("Packed document IDs are not supported with KV cache")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if logits_to_keep < 0 or loss_chunk_size < 0:
            raise ValueError("logits_to_keep and loss_chunk_size cannot be negative")
        if z_loss_coefficient < 0.0:
            raise ValueError("z_loss_coefficient cannot be negative")
        loss_backend = loss_backend.lower()
        if loss_backend not in {"auto", "chunked", "native"}:
            raise ValueError("loss_backend must be auto, chunked, or native")
        if mtp_loss_weight != 0.0 or (mtp_active_heads not in {None, 0}):
            raise ValueError("MTP is disabled for the simplified IRL Core")
        if mtp_horizon_decay <= 0.0:
            raise ValueError("mtp_horizon_decay must be positive")
        if irl_consistency_coefficient != 0.0:
            raise ValueError("IRL consistency loss is disabled for IRL Core")
        if not return_logits and labels is None:
            raise ValueError("return_logits=False requires labels")

        collect_stats = (
            self.config.track_irl_stats
            if return_irl_stats is None
            else bool(return_irl_stats)
        )
        position_ids = self._resolve_position_ids(input_ids, position_ids, cache)
        token_embeddings = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(token_embeddings)
        global_cos, global_sin = self.global_rope(
            position_ids,
            dtype=hidden_states.dtype,
        )
        local_cos: Optional[torch.Tensor] = None
        local_sin: Optional[torch.Tensor] = None
        if "L" in self.config.attention_pattern:
            local_cos, local_sin = self.local_rope(
                position_ids,
                dtype=hidden_states.dtype,
            )

        for layer_index, block in enumerate(self.prelude_blocks):
            rope_cos, rope_sin = self._rope_for_layer(
                layer_index,
                global_cos=global_cos,
                global_sin=global_sin,
                local_cos=local_cos,
                local_sin=local_sin,
            )
            layer_cache = None if cache is None else cache.layer(layer_index)
            hidden_states = self._run_transformer_block(
                block,
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )

        core_cache = (
            None if cache is None else cache.layer(self.config.core_layer_index)
        )
        hidden_states, diagnostics = self._run_core(
            hidden_states,
            position_ids=position_ids,
            rope_cos=global_cos,
            rope_sin=global_sin,
            layer_cache=core_cache,
            attention_mask=attention_mask,
            document_ids=document_ids,
            n_iters=n_iters,
            return_stats=collect_stats,
        )

        for coda_index, block in enumerate(self.coda_blocks):
            layer_index = self.config.core_layer_index + 1 + coda_index
            rope_cos, rope_sin = self._rope_for_layer(
                layer_index,
                global_cos=global_cos,
                global_sin=global_sin,
                local_cos=local_cos,
                local_sin=local_sin,
            )
            layer_cache = None if cache is None else cache.layer(layer_index)
            hidden_states = self._run_transformer_block(
                block,
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )

        if cache is not None:
            cache.advance(input_ids.size(1))

        hidden_states = self.final_norm(hidden_states)
        logits: Optional[torch.Tensor] = None
        if return_logits:
            if labels is not None:
                logits_to_keep = 0
            hidden_for_logits = (
                hidden_states[:, -logits_to_keep:]
                if logits_to_keep > 0
                else hidden_states
            )
            logits = self._apply_logit_softcap(self.lm_head(hidden_for_logits))

        loss: Optional[torch.Tensor] = None
        cross_entropy: Optional[torch.Tensor] = None
        z_loss: Optional[torch.Tensor] = None
        mtp_loss: Optional[torch.Tensor] = None
        irl_consistency_loss: Optional[torch.Tensor] = None
        mtp_losses: Optional[List[torch.Tensor]] = None
        if labels is not None:
            if input_ids.size(1) < 2:
                raise ValueError(
                    "At least two tokens are required to compute shifted LM loss"
                )
            main_targets = self._prepare_targets(
                labels,
                horizon=1,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
            if logits is not None:
                loss, cross_entropy, z_loss = self._token_loss_from_logits(
                    logits[:, :-1],
                    main_targets,
                    z_loss_coefficient=z_loss_coefficient,
                )
            else:
                loss, cross_entropy, z_loss = self._chunked_token_loss(
                    hidden_states[:, :-1],
                    main_targets,
                    chunk_size=loss_chunk_size,
                    z_loss_coefficient=z_loss_coefficient,
                    loss_backend=loss_backend,
                )
            mtp_loss = hidden_states.new_zeros((), dtype=torch.float32)
            mtp_losses = []
            irl_consistency_loss = hidden_states.new_zeros((), dtype=torch.float32)

        return LunarisIRLCoreCausalLMOutput(
            logits=logits,
            loss=loss,
            cross_entropy=cross_entropy,
            z_loss=z_loss,
            mtp_loss=mtp_loss,
            mtp_losses=mtp_losses,
            irl_consistency_loss=irl_consistency_loss,
            cache=cache,
            irl_diagnostics=[diagnostics] if diagnostics is not None else None,
        )

    def configure_optimizer(
        self,
        *,
        learning_rate: float,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: Optional[bool] = None,
    ) -> torch.optim.AdamW:
        """Create conventional AdamW decay and no-decay groups."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        decay_parameters: List[nn.Parameter] = []
        no_decay_parameters: List[nn.Parameter] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            target = decay_parameters if parameter.dim() >= 2 else no_decay_parameters
            target.append(parameter)
        optimizer_kwargs: Dict[str, Any] = {
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        }
        if fused is not None:
            optimizer_kwargs["fused"] = bool(fused)
        return torch.optim.AdamW(
            [
                {"params": decay_parameters, "weight_decay": weight_decay},
                {"params": no_decay_parameters, "weight_decay": 0.0},
            ],
            **optimizer_kwargs,
        )

    def save_pretrained(self, directory: str | Path) -> None:
        """Save a versioned state dict and architecture configuration."""

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.config.save_json(destination / "config.json")
        torch.save(
            {
                "architecture": self.config.architecture_name,
                "format_version": self.config.checkpoint_format_version,
                "state_dict": self.state_dict(),
            },
            destination / "model.pt",
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "LunarisIRLCoreForCausalLM":
        """Load a version-compatible portable checkpoint."""

        source = Path(directory)
        config = LunarisIRLCoreConfig.from_json(source / "config.json")
        payload = torch.load(
            source / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        if payload.get("architecture") != config.architecture_name:
            raise ValueError("Checkpoint architecture does not match the configuration")
        if payload.get("format_version") != config.checkpoint_format_version:
            raise ValueError("Unsupported checkpoint format version")
        model = cls(config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Use the existing cache-aware autoregressive generator."""

        from lunaris_dense_irl2.generation import generate

        return generate(self, input_ids, **kwargs)
