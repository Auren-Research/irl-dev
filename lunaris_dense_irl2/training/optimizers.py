"""Cross-vendor optimizers and gradient stabilization."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .config import OptimizerConfig, TrainingConfig


def zeroth_power_via_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Approximate the matrix polar factor with quintic Newton-Schulz steps."""

    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization requires a matrix")
    compute_dtype = torch.bfloat16 if matrix.device.type == "cuda" else torch.float32
    value = matrix.to(dtype=compute_dtype)
    transposed = value.size(0) > value.size(1)
    if transposed:
        value = value.transpose(0, 1)
    value = value / value.norm().clamp_min(epsilon)

    coefficient_a = 3.4445
    coefficient_b = -4.7750
    coefficient_c = 2.0315
    for _ in range(steps):
        gram = value @ value.transpose(0, 1)
        polynomial = coefficient_b * gram + coefficient_c * (gram @ gram)
        value = coefficient_a * value + polynomial @ value

    if transposed:
        value = value.transpose(0, 1)
    return value.to(dtype=matrix.dtype)


class MONA(Optimizer):
    """Experimental MONA matrix optimizer with decoupled weight decay.

    This follows the 2026 MONA gradient-difference acceleration and the
    scalable Muon update normalization. It is intentionally restricted to
    replicated full matrices; FSDP2 shards do not preserve the matrix polar
    factor.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        acceleration_beta: float,
        acceleration_alpha: float,
        weight_decay: float,
        ns_steps: int,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "acceleration_beta": acceleration_beta,
            "acceleration_alpha": acceleration_alpha,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[no-untyped-def]
        """Apply one accelerated orthogonalized update."""

        loss = None if closure is None else closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            momentum = float(group["momentum"])
            acceleration_beta = float(group["acceleration_beta"])
            acceleration_alpha = float(group["acceleration_alpha"])
            weight_decay = float(group["weight_decay"])
            ns_steps = int(group["ns_steps"])

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise RuntimeError("MONA only accepts matrix parameters")

                gradient = parameter.grad.detach().float()
                state = self.state[parameter]
                if not state:
                    state["previous_gradient"] = gradient.clone()
                    state["acceleration"] = torch.zeros_like(gradient)
                    state["momentum_buffer"] = torch.zeros_like(gradient)

                previous = state["previous_gradient"]
                acceleration = state["acceleration"]
                momentum_buffer = state["momentum_buffer"]

                gradient_difference = gradient - previous
                acceleration.mul_(acceleration_beta).add_(
                    gradient_difference,
                    alpha=1.0 - acceleration_beta,
                )
                accelerated_gradient = gradient.add(
                    acceleration,
                    alpha=acceleration_alpha,
                )
                momentum_buffer.mul_(momentum).add_(accelerated_gradient)

                update = zeroth_power_via_newton_schulz(
                    momentum_buffer,
                    steps=ns_steps,
                )
                rows, columns = parameter.shape
                update_scale = math.sqrt(max(1.0, rows / max(1, columns)))
                if weight_decay > 0.0:
                    parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(
                    update.to(dtype=parameter.dtype),
                    alpha=-learning_rate * update_scale,
                )
                previous.copy_(gradient)
        return loss


class OptimizerBundle:
    """Present one step API over one or more native PyTorch optimizers."""

    def __init__(self, optimizers: List[Optimizer]) -> None:
        if not optimizers:
            raise ValueError("At least one optimizer is required")
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear all optimizer gradients."""

        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        """Step every disjoint optimizer."""

        for optimizer in self.optimizers:
            optimizer.step()


def restore_base_learning_rates(
    optimizers: OptimizerBundle,
    config: OptimizerConfig,
) -> None:
    """Restore immutable schedule bases after loading optimizer state."""

    if config.name == "adamw":
        base_learning_rates = [config.learning_rate]
    else:
        base_learning_rates = [config.mona_lr, config.learning_rate]
    if len(optimizers.optimizers) != len(base_learning_rates):
        raise RuntimeError("Optimizer bundle does not match its configuration")
    for optimizer, base_learning_rate in zip(
        optimizers.optimizers,
        base_learning_rates,
    ):
        for group in optimizer.param_groups:
            group["initial_lr"] = float(base_learning_rate)


def _resolve_fused(
    setting: str,
    parameters: List[nn.Parameter],
) -> bool:
    """Resolve fused AdamW without assuming an accelerator vendor."""

    if setting == "false":
        return False
    if setting == "true":
        return True
    return bool(parameters) and all(
        parameter.device.type == "cuda" for parameter in parameters
    )


def _adamw_groups(
    named_parameters: List[Tuple[str, nn.Parameter]],
) -> List[Dict[str, object]]:
    """Separate matrix weight decay from biases, scales, and norms."""

    decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for _, parameter in named_parameters
        if parameter.requires_grad and parameter.ndim < 2
    ]
    return [
        {"params": decay, "weight_decay": 0.0},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizers(
    model: nn.Module,
    config: OptimizerConfig,
) -> OptimizerBundle:
    """Build AdamW or the MONA/AdamW hybrid after parallelization."""

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    all_parameters = [parameter for _, parameter in named_parameters]

    if config.name == "adamw":
        groups = _adamw_groups(named_parameters)
        groups[0]["weight_decay"] = config.weight_decay
        adamw = torch.optim.AdamW(
            groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            fused=_resolve_fused(config.fused, all_parameters),
        )
        bundle = OptimizerBundle([adamw])
        restore_base_learning_rates(bundle, config)
        return bundle

    mona_named: List[Tuple[str, nn.Parameter]] = []
    adamw_named: List[Tuple[str, nn.Parameter]] = []
    for name, parameter in named_parameters:
        is_embedding_or_head = (
            "token_embedding" in name
            or "horizon_embeddings" in name
            or "lm_head" in name
        )
        if parameter.ndim == 2 and not is_embedding_or_head:
            mona_named.append((name, parameter))
        else:
            adamw_named.append((name, parameter))

    mona = MONA(
        [parameter for _, parameter in mona_named],
        lr=config.mona_lr,
        momentum=config.mona_momentum,
        acceleration_beta=config.mona_acceleration_beta,
        acceleration_alpha=config.mona_acceleration_alpha,
        weight_decay=config.weight_decay,
        ns_steps=config.mona_ns_steps,
    )
    adamw_parameters = [parameter for _, parameter in adamw_named]
    adamw_groups = _adamw_groups(adamw_named)
    adamw_groups[0]["weight_decay"] = config.weight_decay
    adamw = torch.optim.AdamW(
        adamw_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        fused=_resolve_fused(config.fused, adamw_parameters),
    )
    bundle = OptimizerBundle([mona, adamw])
    restore_base_learning_rates(bundle, config)
    return bundle


def _gradient_group(name: str) -> str:
    """Map parameters to coarse functional groups for adaptive clipping."""

    if "attention" in name:
        return "attention"
    if "mtp_predictor" in name:
        return "mtp"
    if ".irl." in name:
        return "irl"
    if "embedding" in name or "lm_head" in name:
        return "embedding_and_head"
    return "norms_and_scales"


class GradientClipper:
    """Global clipping or an AGGC-inspired adaptive group clipping policy."""

    def __init__(self, config: TrainingConfig) -> None:
        self.mode = config.gradient_clip_mode
        self.max_norm = float(config.gradient_clip)
        self.beta = float(config.adaptive_clip_beta)
        self.factor = float(config.adaptive_clip_factor)
        self.ema_norms: Dict[str, float] = {}

    def clip(
        self,
        model: nn.Module,
        *,
        error_if_nonfinite: bool,
    ) -> torch.Tensor:
        """Clip gradients and return their pre-clipping norm."""

        parameters = [
            parameter for parameter in model.parameters() if parameter.grad is not None
        ]
        if not parameters:
            return torch.zeros((), device=next(model.parameters()).device)

        if self.mode in {"global", "none"}:
            limit = self.max_norm if self.mode == "global" else float("inf")
            return torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=limit,
                error_if_nonfinite=error_if_nonfinite,
                foreach=None,
            )

        groups: Dict[str, List[nn.Parameter]] = defaultdict(list)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                groups[_gradient_group(name)].append(parameter)

        squared_total = torch.zeros(
            (),
            device=parameters[0].grad.device,
            dtype=torch.float32,
        )
        for name, group_parameters in groups.items():
            group_norm = torch.nn.utils.clip_grad_norm_(
                group_parameters,
                max_norm=float("inf"),
                error_if_nonfinite=error_if_nonfinite,
                foreach=None,
            )
            norm_value = float(group_norm.detach().float().item())
            previous = self.ema_norms.get(
                name,
                self.max_norm / max(self.factor, 1e-12),
            )
            threshold = max(self.max_norm, self.factor * previous)
            if norm_value > threshold:
                scale = threshold / max(norm_value, 1e-12)
                for parameter in group_parameters:
                    parameter.grad.mul_(scale)
            self.ema_norms[name] = self.beta * previous + (1.0 - self.beta) * norm_value
            squared_total.add_(group_norm.detach().float().square())
        return squared_total.sqrt()
