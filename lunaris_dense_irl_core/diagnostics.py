"""Typed outputs and lightweight diagnostics for Lunaris IRL Core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from lunaris_dense_irl2.cache import KVCache


@dataclass
class IRLCoreDiagnostics:
    """Per-step summaries from one recurrent-core execution."""

    state_rms: torch.Tensor
    update_rms: torch.Tensor
    relative_update_norm: torch.Tensor
    read_rms: torch.Tensor
    correction_anchor_ratio: torch.Tensor
    adjacent_update_cosine: torch.Tensor
    damping_min: torch.Tensor
    damping_mean: torch.Tensor
    damping_max: torch.Tensor
    iterations: int

    def detached_cpu(self) -> "IRLCoreDiagnostics":
        """Return a detached CPU copy suitable for logging or JSON export."""

        return IRLCoreDiagnostics(
            state_rms=self.state_rms.detach().cpu(),
            update_rms=self.update_rms.detach().cpu(),
            relative_update_norm=self.relative_update_norm.detach().cpu(),
            read_rms=self.read_rms.detach().cpu(),
            correction_anchor_ratio=self.correction_anchor_ratio.detach().cpu(),
            adjacent_update_cosine=self.adjacent_update_cosine.detach().cpu(),
            damping_min=self.damping_min.detach().cpu(),
            damping_mean=self.damping_mean.detach().cpu(),
            damping_max=self.damping_max.detach().cpu(),
            iterations=self.iterations,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert diagnostics into serialization-friendly values."""

        cpu = self.detached_cpu()
        return {
            "state_rms": cpu.state_rms.tolist(),
            "update_rms": cpu.update_rms.tolist(),
            "relative_update_norm": cpu.relative_update_norm.tolist(),
            "read_rms": cpu.read_rms.tolist(),
            "correction_anchor_ratio": cpu.correction_anchor_ratio.tolist(),
            "adjacent_update_cosine": cpu.adjacent_update_cosine.tolist(),
            "damping_min": float(cpu.damping_min),
            "damping_mean": float(cpu.damping_mean),
            "damping_max": float(cpu.damping_max),
            "iterations": cpu.iterations,
        }


@dataclass
class LunarisIRLCoreCausalLMOutput:
    """Output of the simplified recurrent causal language model."""

    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_losses: Optional[List[torch.Tensor]] = None
    irl_consistency_loss: Optional[torch.Tensor] = None
    cache: Optional["KVCache"] = None
    irl_diagnostics: Optional[List[IRLCoreDiagnostics]] = None
