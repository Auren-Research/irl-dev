"""Complete Lunaris Dense IRL-2 causal language model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .block import LunarisDenseBlock
from .cache import KVCache
from .config import LunarisDenseConfig
from .diagnostics import IRLDiagnostics, LunarisCausalLMOutput
from .initialization import initialize_module, rescale_residual_projections
from .mtp import IRLMultiTokenPredictor
from .norms import RMSNorm
from .rope import RotaryEmbedding


def _linear_loss_chunk_sums(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    z_loss_coefficient: float,
    logit_softcap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CE and z-loss sums from one reusable linear-projection chunk."""

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
    """Compile one regional CE kernel instead of the complete model graph."""

    return torch.compile(
        _linear_loss_chunk_sums,
        dynamic=False,
        fullgraph=True,
    )


class LunarisDenseForCausalLM(nn.Module):
    """Dense decoder-only language model with IRL-2 in every block."""

    def __init__(self, config: LunarisDenseConfig) -> None:
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
                LunarisDenseBlock(config, layer_index)
                for layer_index in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_predictor = (
            IRLMultiTokenPredictor(config) if config.mtp_num_heads > 0 else None
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
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        if config.scale_residual_projections:
            rescale_residual_projections(self.blocks, n_layers=config.n_layers)

    def set_regional_compile(self, enabled: bool) -> None:
        """Enable or disable the bounded regional compilation paths."""

        self.regional_compile_enabled = bool(enabled)
        for module in self.modules():
            if hasattr(module, "regional_compile_enabled"):
                module.regional_compile_enabled = bool(enabled)

    def get_num_parameters(self, *, trainable_only: bool = True) -> int:
        """Count model parameters."""

        parameters = self.parameters()
        if trainable_only:
            return sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            )
        return sum(parameter.numel() for parameter in parameters)

    def init_cache(self, *, batch_size: int) -> KVCache:
        """Create a model-compatible lazy KV cache."""

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
                    f"position_ids must be smaller than "
                    f"max_seq_len={self.config.max_seq_len}"
                )
        return position_ids

    def _run_block_checkpointed(
        self,
        block: LunarisDenseBlock,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        document_ids: Optional[torch.Tensor],
        n_iters: Optional[int],
        return_irl_consistency: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Checkpoint a complete training block without cache or diagnostics."""

        def block_forward(
            value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output, _, consistency = block(
                value,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=None,
                attention_mask=attention_mask,
                document_ids=document_ids,
                n_iters=n_iters,
                return_irl_stats=False,
                return_irl_consistency=return_irl_consistency,
            )
            return output, consistency

        return checkpoint(block_forward, hidden_states, use_reentrant=False)

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
            valid = valid & document_ids[:, :-horizon].eq(
                document_ids[:, horizon:]
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aligned token loss without retaining full vocabulary logits."""

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
            chunk_labels = flat_labels[start:end]
            chunk_ce, chunk_z = chunk_loss(
                flat_hidden[start:end],
                self.lm_head.weight,
                chunk_labels,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aligned token loss from already materialized logits."""

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
    ) -> LunarisCausalLMOutput:
        """Run causal language modeling, packed training, or cached decoding."""

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
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep cannot be negative")
        if loss_chunk_size < 0:
            raise ValueError("loss_chunk_size cannot be negative")
        if z_loss_coefficient < 0.0:
            raise ValueError("z_loss_coefficient cannot be negative")
        loss_backend = loss_backend.lower()
        if loss_backend not in {"auto", "chunked", "native"}:
            raise ValueError("loss_backend must be auto, chunked, or native")
        if mtp_loss_weight < 0.0:
            raise ValueError("mtp_loss_weight cannot be negative")
        if mtp_active_heads is not None and mtp_active_heads < 0:
            raise ValueError("mtp_active_heads cannot be negative")
        if mtp_horizon_decay <= 0.0:
            raise ValueError("mtp_horizon_decay must be positive")
        if irl_consistency_coefficient < 0.0:
            raise ValueError("irl_consistency_coefficient cannot be negative")
        if mtp_loss_weight > 0.0 and self.mtp_predictor is None:
            raise ValueError("mtp_loss_weight requires model.mtp_num_heads > 0")
        if labels is None and (
            mtp_loss_weight > 0.0 or irl_consistency_coefficient > 0.0
        ):
            raise ValueError("Auxiliary training losses require labels")
        if not return_logits and labels is None:
            raise ValueError("return_logits=False requires labels")

        collect_stats = (
            self.config.track_irl_stats
            if return_irl_stats is None
            else bool(return_irl_stats)
        )
        if self.training and self.config.use_gradient_checkpointing and collect_stats:
            raise ValueError(
                "IRL diagnostics must be disabled while "
                "gradient checkpointing is active"
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

        diagnostics: List[IRLDiagnostics] = []
        consistency_losses: List[torch.Tensor] = []
        return_irl_consistency = irl_consistency_coefficient > 0.0
        for layer_index, block in enumerate(self.blocks):
            is_local = self.config.attention_type(layer_index) == "local"
            if is_local:
                if local_cos is None or local_sin is None:
                    raise RuntimeError("Local RoPE values were not initialized")
                rope_cos, rope_sin = local_cos, local_sin
            else:
                rope_cos, rope_sin = global_cos, global_sin

            checkpoint_block = (
                self.training
                and self.config.use_gradient_checkpointing
                and self.config.gradient_checkpointing_policy == "block"
                and not collect_stats
            )
            if checkpoint_block:
                hidden_states, layer_consistency = self._run_block_checkpointed(
                    block,
                    hidden_states,
                    position_ids=position_ids,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    n_iters=n_iters,
                    return_irl_consistency=return_irl_consistency,
                )
                if return_irl_consistency:
                    consistency_losses.append(layer_consistency)
                continue

            layer_cache = None if cache is None else cache.layer(layer_index)
            hidden_states, layer_diagnostics, layer_consistency = block(
                hidden_states,
                position_ids=position_ids,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                layer_cache=layer_cache,
                attention_mask=attention_mask,
                document_ids=document_ids,
                n_iters=n_iters,
                return_irl_stats=collect_stats,
                return_irl_consistency=return_irl_consistency,
            )
            if return_irl_consistency:
                consistency_losses.append(layer_consistency)
            if layer_diagnostics is not None:
                diagnostics.append(layer_diagnostics)

        if cache is not None:
            cache.advance(input_ids.size(1))

        hidden_states = self.final_norm(hidden_states)
        logits: Optional[torch.Tensor] = None
        if return_logits:
            if labels is not None:
                logits_to_keep = 0
            if logits_to_keep > 0:
                hidden_for_logits = hidden_states[:, -logits_to_keep:]
            else:
                hidden_for_logits = hidden_states
            logits = self._apply_logit_softcap(self.lm_head(hidden_for_logits))

        loss: Optional[torch.Tensor] = None
        cross_entropy: Optional[torch.Tensor] = None
        z_loss: Optional[torch.Tensor] = None
        mtp_loss: Optional[torch.Tensor] = None
        mtp_losses: Optional[List[torch.Tensor]] = None
        irl_consistency_loss: Optional[torch.Tensor] = None
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

            mtp_losses = []
            mtp_loss = hidden_states.new_zeros((), dtype=torch.float32)
            active_mtp_heads = (
                self.config.mtp_num_heads
                if mtp_active_heads is None
                else min(int(mtp_active_heads), self.config.mtp_num_heads)
            )
            if self.mtp_predictor is not None and active_mtp_heads > 0:
                prediction_states, mtp_consistency = self.mtp_predictor(
                    hidden_states,
                    token_embeddings,
                    active_heads=active_mtp_heads,
                    return_consistency=return_irl_consistency,
                )
                if return_irl_consistency and self.config.mtp_irl_iters > 1:
                    consistency_losses.extend(mtp_consistency)
                weighted_mtp = hidden_states.new_zeros((), dtype=torch.float32)
                weight_sum = 0.0
                for horizon_index, prediction_state in enumerate(prediction_states):
                    horizon = horizon_index + 2
                    horizon_targets = self._prepare_targets(
                        labels,
                        horizon=horizon,
                        attention_mask=attention_mask,
                        document_ids=document_ids,
                    )
                    _, horizon_loss, _ = self._chunked_token_loss(
                        prediction_state,
                        horizon_targets,
                        chunk_size=loss_chunk_size,
                        z_loss_coefficient=0.0,
                        loss_backend=loss_backend,
                    )
                    mtp_losses.append(horizon_loss)
                    horizon_weight = float(mtp_horizon_decay) ** horizon_index
                    weighted_mtp = weighted_mtp + horizon_weight * horizon_loss
                    weight_sum += horizon_weight
                if weight_sum > 0.0:
                    mtp_loss = weighted_mtp / weight_sum
                    loss = loss + float(mtp_loss_weight) * mtp_loss

            irl_consistency_loss = (
                torch.stack(consistency_losses).mean()
                if consistency_losses
                else hidden_states.new_zeros((), dtype=torch.float32)
            )
            if irl_consistency_coefficient > 0.0:
                loss = (
                    loss
                    + float(irl_consistency_coefficient) * irl_consistency_loss
                )

        return LunarisCausalLMOutput(
            logits=logits,
            loss=loss,
            cross_entropy=cross_entropy,
            z_loss=z_loss,
            mtp_loss=mtp_loss,
            mtp_losses=mtp_losses,
            irl_consistency_loss=irl_consistency_loss,
            cache=cache,
            irl_diagnostics=diagnostics if collect_stats else None,
        )

    def configure_optimizer(
        self,
        *,
        learning_rate: float,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: Optional[bool] = None,
    ) -> torch.optim.AdamW:
        """Create AdamW groups without external optimizer helpers."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        decay_parameters: List[nn.Parameter] = []
        no_decay_parameters: List[nn.Parameter] = []
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dim() >= 2:
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)

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
        """Save a versioned state dict and JSON config."""

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
    ) -> "LunarisDenseForCausalLM":
        """Load a version-compatible checkpoint."""

        source = Path(directory)
        config = LunarisDenseConfig.from_json(source / "config.json")
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
        """Delegate generation to the standalone cache-aware implementation."""

        from .generation import generate

        return generate(self, input_ids, **kwargs)
