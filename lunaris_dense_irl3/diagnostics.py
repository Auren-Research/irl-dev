"""Typed state, diagnostics, and model outputs for IRL-3."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from .cache import IRL3Cache


@dataclass
class IRL3State:
    """Persistent joint state of the latent recurrent computer.

    Public persistent states omit the teacher-forcing prefix dimension and use
    shapes ``[batch, slots, state_dim]`` for ``z``, ``h``, and ``m``.
    """

    z: torch.Tensor
    h: torch.Tensor
    m: torch.Tensor

    def detached(self) -> "IRL3State":
        return IRL3State(self.z.detach(), self.h.detach(), self.m.detach())

    def to(self, *args: Any, **kwargs: Any) -> "IRL3State":
        return IRL3State(
            self.z.to(*args, **kwargs),
            self.h.to(*args, **kwargs),
            self.m.to(*args, **kwargs),
        )


@dataclass
class IRL3Diagnostics:
    """Low-cost reductions from one execution of the recurrent computer."""

    halt_probabilities: torch.Tensor
    halt_weights: torch.Tensor
    active_latent_fraction: torch.Tensor
    anchor_read_rms: torch.Tensor
    anchor_attention_entropy: torch.Tensor
    memory_read_rms: torch.Tensor
    memory_write_rms: torch.Tensor
    memory_utilization: torch.Tensor
    z_update_rms: torch.Tensor
    h_update_rms: torch.Tensor
    m_update_rms: torch.Tensor
    z_rms: torch.Tensor
    h_rms: torch.Tensor
    m_rms: torch.Tensor
    slow_update_mask: torch.Tensor
    fast_residual: torch.Tensor
    joint_residual: torch.Tensor
    fast_contraction_ratios: torch.Tensor
    contraction_ratios: torch.Tensor
    equilibrium_z_residual: torch.Tensor
    equilibrium_h_residual: torch.Tensor
    equilibrium_m_residual: torch.Tensor
    last_h_update_rms: torch.Tensor
    writeback_rms: torch.Tensor
    writeback_to_anchor_ratio: torch.Tensor
    expected_steps: torch.Tensor
    step_counts: torch.Tensor
    iterations: int
    slow_updates: int
    last_slow_update_step: int
    converged_fraction: float
    halt_mode: str

    def detached_cpu(self) -> "IRL3Diagnostics":
        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.detach().cpu() if torch.is_tensor(value) else value
            )
        return IRL3Diagnostics(**values)

    def as_dict(self) -> Dict[str, Any]:
        cpu = self.detached_cpu()
        result: Dict[str, Any] = {}
        for field in fields(cpu):
            value = getattr(cpu, field.name)
            if torch.is_tensor(value):
                result[field.name] = (
                    float(value) if value.numel() == 1 else value.tolist()
                )
            else:
                result[field.name] = value
        return result


@dataclass
class LunarisIRL3CausalLMOutput:
    """Output of the IRL-3 causal language model."""

    logits: Optional[torch.Tensor]
    loss: Optional[torch.Tensor] = None
    cross_entropy: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_losses: Optional[List[torch.Tensor]] = None
    ponder_loss: Optional[torch.Tensor] = None
    equilibrium_loss: Optional[torch.Tensor] = None
    cache: Optional["IRL3Cache"] = None
    reasoning_state: Optional[IRL3State] = None
    irl3_diagnostics: Optional[IRL3Diagnostics] = None
    writeback_disabled: bool = False
