"""Complete Lunaris Dense IRL-3 causal language model."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lunaris_dense_irl2.initialization import initialize_module
from lunaris_dense_irl2.norms import RMSNorm
from lunaris_dense_irl2.rope import RotaryEmbedding

from .block import LunarisIRL3TransformerBlock
from .cache import IRL3Cache
from .config import LunarisIRL3Config
from .diagnostics import (
    IRL3State,
    LunarisIRL3CausalLMOutput,
)
from .irl3 import IRL3Computer, IRL3ComputerResult
from .mtp import IRL3MultiTokenPredictor


def _linear_loss_chunk_sums(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_coefficient: float,
    logit_softcap: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden_states, weight)
    if logit_softcap > 0.0:
        logits = torch.tanh(logits / logit_softcap) * logit_softcap
    logits = logits.float()
    valid = labels.ne(ignore_index)
    safe_labels = labels.masked_fill(~valid, 0)
    log_partition = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    ce_sum = ((log_partition - target_logits) * valid).sum()
    z_sum = log_partition.new_zeros(())
    if z_loss_coefficient > 0.0:
        z_sum = (log_partition.square() * valid).sum()
    return ce_sum, z_sum


@lru_cache(maxsize=1)
def _compiled_linear_loss_chunk() -> Any:
    return torch.compile(_linear_loss_chunk_sums, dynamic=False, fullgraph=True)


class LunarisIRL3ForCausalLM(nn.Module):
    """Decoder-only language model with a stateful latent recurrent computer."""

    def __init__(self, config: LunarisIRL3Config) -> None:
        super().__init__()
        self.config = config
        self.regional_compile_enabled = True

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                LunarisIRL3TransformerBlock(config, layer_index)
                for layer_index in range(config.n_layers)
            ]
        )
        self.computer = IRL3Computer(config)
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_predictor = (
            IRL3MultiTokenPredictor(config) if config.mtp_num_heads > 0 else None
        )
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
        self._restore_special_initialization()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.scale_residual_projections:
            self._rescale_residual_projections()

    def _restore_special_initialization(self) -> None:
        """Restore gate biases intentionally different from generic linear init."""

        nn.init.constant_(
            self.computer.memory_write_gate.bias,
            float(self.config.irl3_memory_gate_init),
        )
        nn.init.constant_(
            self.computer.halt_output.bias,
            float(self.config.irl3_halt_bias_init),
        )
        nn.init.constant_(
            self.computer.write_gate.bias,
            float(self.config.irl3_writeback_gate_init),
        )

    def _rescale_residual_projections(self) -> None:
        fixed_scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        recurrent_scale = 1.0 / math.sqrt(float(self.config.irl3_train_steps))
        with torch.no_grad():
            for block in self.blocks:
                block.attention.o_proj.weight.mul_(fixed_scale)
                block.ffn.down_proj.weight.mul_(fixed_scale)
            self.computer.fast_ffn.down_proj.weight.mul_(recurrent_scale)
            self.computer.slow_ffn.down_proj.weight.mul_(recurrent_scale)
            self.computer.anchor_reader.output_proj.weight.mul_(recurrent_scale)
            for module in self.computer.modules():
                if module.__class__.__name__ == "StateAttention":
                    module.o_proj.weight.mul_(recurrent_scale)

    def set_regional_compile(self, enabled: bool) -> None:
        self.regional_compile_enabled = bool(enabled)
        for module in self.modules():
            if hasattr(module, "regional_compile_enabled"):
                module.regional_compile_enabled = bool(enabled)

    def get_num_parameters(self, *, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def init_cache(self, *, batch_size: int) -> IRL3Cache:
        return IRL3Cache.from_config(self.config, batch_size=batch_size)

    def _resolve_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        cache: Optional[IRL3Cache],
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        explicit = position_ids is not None
        if position_ids is None:
            start = 0 if cache is None else cache.seen_tokens
            if start + sequence_length > self.config.max_seq_len:
                raise ValueError("Input positions exceed max_seq_len")
            positions = torch.arange(
                start,
                start + sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            )
            position_ids = positions.unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids shape")
        if explicit and not torch.compiler.is_compiling():
            if bool(position_ids.lt(0).any()):
                raise ValueError("position_ids cannot be negative")
            if bool(position_ids.ge(self.config.max_seq_len).any()):
                raise ValueError("position_ids exceed max_seq_len")
        return position_ids

    def _rope_for_layer(
        self,
        layer_index: int,
        global_rope: Tuple[torch.Tensor, torch.Tensor],
        local_rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.attention_type(layer_index) == "local":
            if local_rope is None:
                raise RuntimeError("Local RoPE was not initialized")
            return local_rope
        return global_rope

    def _run_block(
        self,
        block: LunarisIRL3TransformerBlock,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        layer_cache: Any,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        should_checkpoint = (
            self.training
            and self.config.use_gradient_checkpointing
            and self.config.gradient_checkpointing_policy == "block"
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
        if layer_cache is not None:
            raise ValueError("Checkpointed blocks cannot update a cache")

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

    def _run_computer(
        self,
        anchor: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
        cache: Optional[IRL3Cache],
        persistent_state: Optional[IRL3State],
        compute_depth: Optional[int],
        halt_mode: Optional[str],
        gradient_horizon: Optional[int],
        return_stats: bool,
    ) -> IRL3ComputerResult:
        should_checkpoint = (
            self.training
            and self.config.use_gradient_checkpointing
            and self.config.gradient_checkpointing_policy == "computer"
        )
        if should_checkpoint and return_stats:
            raise ValueError("IRL-3 diagnostics are disabled with checkpointing")
        if should_checkpoint and (cache is not None or persistent_state is not None):
            raise ValueError("Checkpointed IRL-3 execution cannot use persistent state")
        if not should_checkpoint:
            return self.computer(
                anchor,
                position_ids=position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
                anchor_cache=None if cache is None else cache.anchor,
                persistent_state=persistent_state,
                compute_depth=compute_depth,
                halt_mode=halt_mode,
                gradient_horizon=gradient_horizon,
                return_stats=return_stats,
            )

        def computer_forward(
            value: torch.Tensor,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            result = self.computer(
                value,
                position_ids=position_ids,
                attention_mask=attention_mask,
                document_ids=document_ids,
                compute_depth=compute_depth,
                halt_mode=halt_mode,
                gradient_horizon=gradient_horizon,
                return_stats=False,
            )
            return (
                result.correction,
                result.state.z,
                result.state.h,
                result.state.m,
                result.ponder_loss,
                result.equilibrium_loss,
            )

        correction, z, h, m, ponder, equilibrium = checkpoint(
            computer_forward,
            anchor,
            use_reentrant=False,
        )
        return IRL3ComputerResult(
            correction=correction,
            state=IRL3State(z, h, m),
            ponder_loss=ponder,
            equilibrium_loss=equilibrium,
            diagnostics=None,
        )

    def _apply_logit_softcap(self, logits: torch.Tensor) -> torch.Tensor:
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
        denominator = labels.ne(self.config.loss_ignore_index).sum().float()
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
        flat_hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_labels = labels.reshape(-1)
        valid_tokens = flat_labels.ne(self.config.loss_ignore_index).sum()
        native = None
        if z_loss_coefficient == 0.0 and loss_backend in {"auto", "native"}:
            native = self._native_linear_cross_entropy(
                flat_hidden, flat_labels, chunk_size=chunk_size
            )
        if native is not None:
            cross_entropy = native.float()
            zero = cross_entropy.new_zeros(())
            return cross_entropy, cross_entropy, zero
        if loss_backend == "native":
            raise RuntimeError("Native linear cross entropy is unavailable")

        rows_per_chunk = chunk_size if chunk_size > 0 else flat_hidden.size(0)
        ce_sum = hidden_states.new_zeros((), dtype=torch.float32)
        z_sum = hidden_states.new_zeros((), dtype=torch.float32)
        use_compile = (
            self.regional_compile_enabled
            and flat_hidden.device.type == "cuda"
            and chunk_size > 0
        )
        if use_compile:
            # Keep every invocation of the full-graph regional kernel at the
            # same static shape. Ignored padding does not change either sum or
            # the denominator, which was measured before padding.
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
        denominator = valid_tokens.float().clamp_min(1.0)
        cross_entropy = ce_sum / denominator
        z_loss = z_sum / denominator
        return (
            cross_entropy + float(z_loss_coefficient) * z_loss,
            cross_entropy,
            z_loss,
        )

    def _token_loss_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        z_loss_coefficient: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logits.contiguous().float()
        labels = labels.contiguous()
        valid = labels.ne(self.config.loss_ignore_index)
        denominator = valid.sum().float().clamp_min(1.0)
        cross_entropy = (
            F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.loss_ignore_index,
                reduction="sum",
            )
            / denominator
        )
        z_loss = logits.new_zeros(())
        if z_loss_coefficient > 0.0:
            log_partition = torch.logsumexp(logits, dim=-1)
            z_loss = log_partition.square().masked_select(valid).sum() / denominator
        return (
            cross_entropy + float(z_loss_coefficient) * z_loss,
            cross_entropy,
            z_loss,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[IRL3Cache] = None,
        compute_depth: Optional[int] = None,
        halt_mode: Optional[str] = None,
        gradient_horizon: Optional[int] = None,
        reasoning_state: Optional[IRL3State] = None,
        persist_reasoning_state: Optional[bool] = None,
        return_reasoning_state: bool = False,
        return_irl3_stats: Optional[bool] = None,
        disable_irl3_writeback: bool = False,
        logits_to_keep: int = 0,
        return_logits: bool = True,
        loss_chunk_size: int = 0,
        z_loss_coefficient: float = 0.0,
        loss_backend: str = "auto",
        mtp_loss_weight: float = 0.0,
        mtp_active_heads: Optional[int] = None,
        mtp_horizon_decay: float = 0.7,
        ponder_coefficient: float = 0.0,
        equilibrium_coefficient: float = 0.0,
    ) -> LunarisIRL3CausalLMOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an integer tensor")
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")
        if disable_irl3_writeback and self.training:
            raise ValueError("IRL-3 write-back ablation is evaluation-only")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if cache is not None:
            if self.training:
                raise ValueError("Cache is evaluation-only")
            if input_ids.size(0) != cache.max_batch_size:
                raise ValueError("Input batch size does not match the cache")
            if document_ids is not None:
                raise ValueError("Packed documents are unsupported with cache")
        if (
            min(
                logits_to_keep,
                loss_chunk_size,
            )
            < 0
        ):
            raise ValueError("logits_to_keep and loss_chunk_size cannot be negative")
        if (
            min(
                z_loss_coefficient,
                mtp_loss_weight,
                ponder_coefficient,
                equilibrium_coefficient,
            )
            < 0.0
        ):
            raise ValueError("Loss coefficients cannot be negative")
        if mtp_horizon_decay <= 0.0:
            raise ValueError("mtp_horizon_decay must be positive")
        if mtp_active_heads is not None and int(mtp_active_heads) < 0:
            raise ValueError("mtp_active_heads cannot be negative")
        loss_backend = loss_backend.lower()
        if loss_backend not in {"auto", "chunked", "native"}:
            raise ValueError("loss_backend must be auto, chunked, or native")
        if mtp_loss_weight > 0.0 and self.mtp_predictor is None:
            raise ValueError("mtp_loss_weight requires mtp_num_heads > 0")
        if labels is None and (
            mtp_loss_weight > 0.0
            or ponder_coefficient > 0.0
            or equilibrium_coefficient > 0.0
        ):
            raise ValueError("Auxiliary objectives require labels")
        if not return_logits and labels is None:
            raise ValueError("return_logits=False requires labels")

        collect_stats = (
            self.config.track_irl3_stats
            if return_irl3_stats is None
            else bool(return_irl3_stats)
        )
        persist = (
            self.config.irl3_persist_state
            if persist_reasoning_state is None
            else bool(persist_reasoning_state)
        )
        if persist and cache is None and reasoning_state is None and self.training:
            raise ValueError("Persistent reasoning state is evaluation-only")
        if reasoning_state is None and persist and cache is not None:
            reasoning_state = cache.reasoning_state
        if compute_depth is not None and halt_mode is None:
            # An explicit budget is an experiment control, not a router request.
            halt_mode = "fixed"

        position_ids = self._resolve_position_ids(input_ids, position_ids, cache)
        token_embeddings = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(token_embeddings)
        global_rope = self.global_rope(position_ids, dtype=hidden_states.dtype)
        local_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if "L" in self.config.attention_pattern:
            local_rope = self.local_rope(position_ids, dtype=hidden_states.dtype)

        for layer_index in range(self.config.anchor_layers):
            rope_cos, rope_sin = self._rope_for_layer(
                layer_index, global_rope, local_rope
            )
            hidden_states = self._run_block(
                self.blocks[layer_index],
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=None if cache is None else cache.layer(layer_index),
                attention_mask=attention_mask,
                document_ids=document_ids,
            )

        anchor = hidden_states
        computer_result = self._run_computer(
            anchor,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            cache=cache,
            persistent_state=reasoning_state,
            compute_depth=compute_depth,
            halt_mode=halt_mode,
            gradient_horizon=gradient_horizon,
            return_stats=collect_stats,
        )
        hidden_states = (
            anchor
            if disable_irl3_writeback
            else anchor + computer_result.correction
        )

        for layer_index in range(self.config.anchor_layers, self.config.n_layers):
            rope_cos, rope_sin = self._rope_for_layer(
                layer_index, global_rope, local_rope
            )
            hidden_states = self._run_block(
                self.blocks[layer_index],
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=None if cache is None else cache.layer(layer_index),
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        if cache is not None:
            cache.advance(input_ids.size(1))

        public_state = IRL3State(
            computer_result.state.z[:, -1],
            computer_result.state.h[:, -1],
            computer_result.state.m[:, -1],
        )
        if persist and cache is not None:
            cache.reasoning_state = public_state.detached()

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
        mtp_losses: Optional[List[torch.Tensor]] = None
        if labels is not None:
            if input_ids.size(1) < 2:
                raise ValueError("At least two tokens are required for LM loss")
            targets = self._prepare_targets(
                labels,
                horizon=1,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
            if logits is not None:
                loss, cross_entropy, z_loss = self._token_loss_from_logits(
                    logits[:, :-1],
                    targets,
                    z_loss_coefficient=z_loss_coefficient,
                )
            else:
                loss, cross_entropy, z_loss = self._chunked_token_loss(
                    hidden_states[:, :-1],
                    targets,
                    chunk_size=loss_chunk_size,
                    z_loss_coefficient=z_loss_coefficient,
                    loss_backend=loss_backend,
                )

            mtp_losses = []
            mtp_loss = hidden_states.new_zeros((), dtype=torch.float32)
            active_heads = (
                self.config.mtp_num_heads
                if mtp_active_heads is None
                else min(int(mtp_active_heads), self.config.mtp_num_heads)
            )
            if self.mtp_predictor is not None and active_heads > 0:
                predictions = self.mtp_predictor(
                    hidden_states,
                    token_embeddings,
                    active_heads=active_heads,
                )
                weighted = hidden_states.new_zeros((), dtype=torch.float32)
                weight_sum = 0.0
                for horizon_index, prediction in enumerate(predictions):
                    horizon = horizon_index + 2
                    horizon_targets = self._prepare_targets(
                        labels,
                        horizon=horizon,
                        attention_mask=attention_mask,
                        document_ids=document_ids,
                    )
                    _, horizon_loss, _ = self._chunked_token_loss(
                        prediction,
                        horizon_targets,
                        chunk_size=loss_chunk_size,
                        z_loss_coefficient=0.0,
                        loss_backend=loss_backend,
                    )
                    mtp_losses.append(horizon_loss)
                    horizon_weight = float(mtp_horizon_decay) ** horizon_index
                    weighted = weighted + horizon_weight * horizon_loss
                    weight_sum += horizon_weight
                if weight_sum > 0.0:
                    mtp_loss = weighted / weight_sum
                    loss = loss + float(mtp_loss_weight) * mtp_loss
            loss = (
                loss
                + float(ponder_coefficient) * computer_result.ponder_loss
                + float(equilibrium_coefficient) * computer_result.equilibrium_loss
            )

        return LunarisIRL3CausalLMOutput(
            logits=logits,
            loss=loss,
            cross_entropy=cross_entropy,
            z_loss=z_loss,
            mtp_loss=mtp_loss,
            mtp_losses=mtp_losses,
            ponder_loss=computer_result.ponder_loss,
            equilibrium_loss=computer_result.equilibrium_loss,
            cache=cache,
            reasoning_state=(
                public_state if return_reasoning_state or persist else None
            ),
            irl3_diagnostics=computer_result.diagnostics,
            writeback_disabled=bool(disable_irl3_writeback),
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
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        decay: List[nn.Parameter] = []
        no_decay: List[nn.Parameter] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.dim() >= 2 else no_decay).append(parameter)
        kwargs: Dict[str, Any] = {
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        }
        if fused is not None:
            kwargs["fused"] = bool(fused)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **kwargs,
        )

    def save_pretrained(self, directory: str | Path) -> None:
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
    ) -> "LunarisIRL3ForCausalLM":
        source = Path(directory)
        config = LunarisIRL3Config.from_json(source / "config.json")
        payload = torch.load(
            source / "model.pt",
            map_location=map_location,
            weights_only=True,
        )
        if payload.get("architecture") != config.architecture_name:
            raise ValueError("Checkpoint architecture does not match config")
        if payload.get("format_version") != config.checkpoint_format_version:
            raise ValueError("Unsupported checkpoint format version")
        model = cls(config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        from .generation import generate

        return generate(self, input_ids, **kwargs)
