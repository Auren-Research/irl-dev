"""Stateless learning-rate schedules for resumable pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .config import ObjectiveConfig, ScheduleConfig


@dataclass(frozen=True)
class AuxiliarySchedule:
    """Resolved recurrent depth and auxiliary coefficients for one step."""

    irl_iters: int
    mtp_active_heads: int
    mtp_loss_weight: float
    irl_consistency_weight: float


def _linear_ramp(step: int, start: int, duration: int) -> float:
    """Return a deterministic zero-to-one ramp."""

    if step < start:
        return 0.0
    if duration <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - start) / duration))


def auxiliary_schedule(
    step: int,
    model: Any,
    config: ObjectiveConfig,
) -> AuxiliarySchedule:
    """Resolve the coupled IRL-depth and MTP curriculum."""

    step = max(0, int(step))
    if model.architecture_name == "lunaris_dense_transformer":
        return AuxiliarySchedule(
            irl_iters=1,
            mtp_active_heads=0,
            mtp_loss_weight=0.0,
            irl_consistency_weight=0.0,
        )
    if model.architecture_name == "lunaris_dense_irl_core":
        return AuxiliarySchedule(
            irl_iters=model.irl_train_iters,
            mtp_active_heads=0,
            mtp_loss_weight=0.0,
            irl_consistency_weight=0.0,
        )
    depth_levels = model.irl_train_iters - config.irl_start_iters
    if depth_levels <= 0 or config.irl_ramp_steps <= 0:
        irl_iters = model.irl_train_iters
    else:
        progress = min(1.0, step / config.irl_ramp_steps)
        completed_levels = min(
            depth_levels,
            int(progress * (depth_levels + 1)),
        )
        irl_iters = config.irl_start_iters + completed_levels

    mtp_ramp = _linear_ramp(
        step,
        config.mtp_warmup_steps,
        config.mtp_ramp_steps,
    )
    mtp_loss_weight = config.mtp_loss_weight * mtp_ramp
    if config.mtp_loss_weight <= 0.0 or model.mtp_num_heads <= 0:
        active_heads = 0
    elif config.mtp_horizon_interval <= 0:
        active_heads = model.mtp_num_heads
    else:
        # Keep one head in the zero-weight warmup graph so DDP observes every
        # shared MTP parameter from the first update.
        elapsed = max(0, step - config.mtp_warmup_steps)
        active_heads = min(
            model.mtp_num_heads,
            1 + elapsed // config.mtp_horizon_interval,
        )

    consistency_ramp = _linear_ramp(
        step,
        config.irl_consistency_warmup_steps,
        config.irl_consistency_ramp_steps,
    )
    consistency_weight = config.irl_consistency_weight * consistency_ramp
    if irl_iters <= 1:
        consistency_weight = 0.0

    return AuxiliarySchedule(
        irl_iters=irl_iters,
        mtp_active_heads=active_heads,
        mtp_loss_weight=mtp_loss_weight,
        irl_consistency_weight=consistency_weight,
    )


def _decay_value(progress: float, shape: str) -> float:
    """Evaluate a normalized linear or cosine decay."""

    progress = min(1.0, max(0.0, progress))
    if shape == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_multiplier(step: int, config: ScheduleConfig, max_steps: int) -> float:
    """Return the LR multiplier for the next optimizer update."""

    step = max(0, int(step))
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return float(step + 1) / float(config.warmup_steps)

    if config.name == "constant":
        return 1.0

    if config.name == "cosine":
        decay_start = config.warmup_steps
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        value = _decay_value(progress, "cosine")
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * value

    stable_steps = int(config.stable_steps or 0)
    decay_steps = int(config.decay_steps or 0)
    decay_start = config.warmup_steps + stable_steps
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, decay_steps)
    value = _decay_value(progress, config.decay_shape)
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * value


def apply_learning_rate(
    optimizers: Iterable[torch.optim.Optimizer],
    multiplier: float,
) -> None:
    """Apply a multiplier while preserving each parameter group's base LR."""

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            base_lr = float(group.setdefault("initial_lr", group["lr"]))
            group["lr"] = base_lr * float(multiplier)
