"""Typed model outputs and optional IRL-2 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .cache import KVCache


@dataclass
class IRLDiagnostics:
    """Summary of one IRL-2 execution."""

    residual_norms: torch.Tensor
    contraction_ratios: torch.Tensor
    state_rms: torch.Tensor
    update_rms: torch.Tensor
    update_cosine: torch.Tensor
    damping: torch.Tensor
    layer_scale_mean: torch.Tensor
    layer_scale_max: torch.Tensor
    iterations: int
    converged: bool

    def detached_cpu(self) -> "IRLDiagnostics":
        """Return a detached CPU copy suitable for logging."""

        return IRLDiagnostics(
            residual_norms=self.residual_norms.detach().cpu(),
            contraction_ratios=self.contraction_ratios.detach().cpu(),
            state_rms=self.state_rms.detach().cpu(),
            update_rms=self.update_rms.detach().cpu(),
            update_cosine=self.update_cosine.detach().cpu(),
            damping=self.damping.detach().cpu(),
            layer_scale_mean=self.layer_scale_mean.detach().cpu(),
            layer_scale_max=self.layer_scale_max.detach().cpu(),
            iterations=self.iterations,
            converged=self.converged,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert diagnostics into a serialization-friendly dictionary."""

        cpu = self.detached_cpu()
        return {
            "residual_norms": cpu.residual_norms.tolist(),
            "contraction_ratios": cpu.contraction_ratios.tolist(),
            "state_rms": cpu.state_rms.tolist(),
            "update_rms": cpu.update_rms.tolist(),
            "update_cosine": cpu.update_cosine.tolist(),
            "damping": float(cpu.damping),
            "layer_scale_mean": float(cpu.layer_scale_mean),
            "layer_scale_max": float(cpu.layer_scale_max),
            "iterations": cpu.iterations,
            "converged": cpu.converged,
        }


@dataclass
class LunarisCausalLMOutput:
    """Output of the causal language model."""

    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_losses: Optional[List[torch.Tensor]] = None
    irl_consistency_loss: Optional[torch.Tensor] = None
    cache: Optional["KVCache"] = None
    irl_diagnostics: Optional[List[IRLDiagnostics]] = None
