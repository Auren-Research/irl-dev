"""Anchored IRL-2 recurrent refinement."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LunarisDenseConfig
from .diagnostics import IRLDiagnostics
from .norms import RMSNorm


class SwiGLU(nn.Module):
    """Bias-free SwiGLU projection used by the recurrent map."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate_up_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.hidden_dim, dim=-1)
        return self.down_proj(F.silu(gate) * up)

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
        """Upgrade separate gate/up projections from version-one checkpoints."""

        fused_key = prefix + "gate_up_proj.weight"
        gate_key = prefix + "gate_proj.weight"
        up_key = prefix + "up_proj.weight"
        if (
            fused_key not in state_dict
            and gate_key in state_dict
            and up_key in state_dict
        ):
            state_dict[fused_key] = torch.cat(
                (state_dict.pop(gate_key), state_dict.pop(up_key)),
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


class IRL2Cell(nn.Module):
    """Refine a residual correction with one shared nonlinear map."""

    def __init__(
        self,
        config: LunarisDenseConfig,
        *,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.train_iters = int(config.irl_train_iters)
        self.eval_iters = int(config.irl_eval_iters)
        self.lambda_min = float(config.irl_lambda_min)
        self.lambda_max = float(config.irl_lambda_max)
        self.default_halt_tol = float(config.irl_halt_tol)
        self.halt_percentile = float(config.irl_halt_percentile)
        self.track_stats_by_default = bool(config.track_irl_stats)

        self.state_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        resolved_hidden_dim = (
            config.ffn_hidden_dim if hidden_dim is None else int(hidden_dim)
        )
        if resolved_hidden_dim <= 0:
            raise ValueError("IRL hidden_dim must be positive")
        self.ffn = SwiGLU(config.d_model, resolved_hidden_dim)

        self.anchor_gate = nn.Parameter(
            torch.full((config.d_model,), float(config.irl_anchor_gate_init))
        )
        self.layer_scale = nn.Parameter(
            torch.full((config.d_model,), float(config.irl_layer_scale_init))
        )
        self.lambda_logit = nn.Parameter(
            torch.tensor(config.lambda_logit(), dtype=torch.float32)
        )

    @property
    def damping(self) -> torch.Tensor:
        """Return damping constrained to the configured open interval."""

        return self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(
            self.lambda_logit
        )

    def _target(self, state: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Evaluate the anchored fixed-point target."""

        recurrent_input = self.state_norm(anchor + state)
        recurrent_input = recurrent_input + self.anchor_gate.to(anchor.dtype) * anchor
        target = self.ffn(recurrent_input)
        return target * self.layer_scale.to(target.dtype)

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        """Compute one scalar RMS statistic in FP32."""

        return x.float().square().mean().sqrt()

    def _resolve_iterations(self, n_iters: Optional[int]) -> int:
        """Resolve train/eval defaults and validate an explicit override."""

        if n_iters is None:
            return self.train_iters if self.training else self.eval_iters
        if int(n_iters) <= 0:
            raise ValueError("n_iters must be positive")
        return int(n_iters)

    def forward(
        self,
        anchor: torch.Tensor,
        *,
        n_iters: Optional[int] = None,
        halt_tol: Optional[float] = None,
        return_stats: Optional[bool] = None,
        return_consistency: bool = False,
    ) -> Tuple[torch.Tensor, Optional[IRLDiagnostics], torch.Tensor]:
        """Run shared-weight refinement and optionally collect diagnostics."""

        iterations = self._resolve_iterations(n_iters)
        track_stats = (
            self.track_stats_by_default if return_stats is None else bool(return_stats)
        )
        tolerance = self.default_halt_tol if halt_tol is None else float(halt_tol)
        if tolerance < 0.0:
            raise ValueError("halt_tol cannot be negative")
        halting_enabled = (not self.training) and tolerance > 0.0

        damping = self.damping.to(anchor.dtype)
        state = torch.zeros_like(anchor)
        previous_delta: Optional[torch.Tensor] = None

        residual_norms: List[torch.Tensor] = []
        contraction_ratios: List[torch.Tensor] = []
        state_rms: List[torch.Tensor] = []
        update_rms: List[torch.Tensor] = []
        update_cosine: List[torch.Tensor] = []
        consistency_losses: List[torch.Tensor] = []
        converged = False
        iterations_used = 0

        for iteration_index in range(iterations):
            target = self._target(state, anchor)
            new_state = (1.0 - damping) * state + damping * target
            delta = new_state - state
            iterations_used += 1

            if return_consistency and iteration_index > 0:
                teacher = new_state.detach()
                numerator = (state - teacher).float().square().mean()
                denominator = teacher.float().square().mean().clamp_min(1e-6)
                consistency_losses.append(numerator / denominator)

            if track_stats:
                delta_norm = delta.float().norm(dim=-1).mean()
                residual_norms.append(delta_norm)
                state_rms.append(self._rms(new_state))
                update_rms.append(self._rms(delta))
                if previous_delta is not None:
                    previous_norm = previous_delta.float().norm(dim=-1).mean()
                    contraction_ratios.append(
                        delta_norm / previous_norm.clamp_min(1e-12)
                    )
                    cosine = F.cosine_similarity(
                        delta.float(),
                        previous_delta.float(),
                        dim=-1,
                        eps=1e-12,
                    ).mean()
                    update_cosine.append(cosine)

            if halting_enabled:
                relative_step = delta.float().norm(dim=-1) / (
                    new_state.float().norm(dim=-1) + 1e-6
                )
                if self.halt_percentile >= 1.0:
                    halt_metric = relative_step.max()
                else:
                    halt_metric = torch.quantile(
                        relative_step.flatten(),
                        self.halt_percentile,
                    )
                if bool(halt_metric <= tolerance):
                    state = new_state
                    converged = True
                    previous_delta = delta
                    break

            state = new_state
            previous_delta = delta

        consistency = (
            torch.stack(consistency_losses).mean()
            if consistency_losses
            else anchor.new_zeros((), dtype=torch.float32)
        )
        if not track_stats:
            return state, None, consistency

        device = anchor.device
        empty = torch.empty(0, device=device, dtype=torch.float32)
        diagnostics = IRLDiagnostics(
            residual_norms=torch.stack(residual_norms) if residual_norms else empty,
            contraction_ratios=(
                torch.stack(contraction_ratios) if contraction_ratios else empty
            ),
            state_rms=torch.stack(state_rms) if state_rms else empty,
            update_rms=torch.stack(update_rms) if update_rms else empty,
            update_cosine=torch.stack(update_cosine) if update_cosine else empty,
            damping=self.damping.detach(),
            layer_scale_mean=self.layer_scale.detach().float().mean(),
            layer_scale_max=self.layer_scale.detach().float().abs().max(),
            iterations=iterations_used,
            converged=converged,
        )
        return state, diagnostics, consistency
