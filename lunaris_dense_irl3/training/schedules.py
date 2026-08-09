"""Stateless compute, auxiliary-loss, and learning-rate schedules."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from lunaris_dense_irl2.training.schedules import (
    apply_learning_rate,
    lr_multiplier,
)

from ..config import LunarisIRL3Config
from .config import ObjectiveConfig


@dataclass(frozen=True)
class AuxiliarySchedule:
    """Resolved IRL-3 training controls for one optimizer step."""

    compute_depth: int
    depth_ceiling: int
    gradient_horizon: int
    mtp_active_heads: int
    mtp_loss_weight: float
    ponder_weight: float
    equilibrium_weight: float


def _linear_ramp(step: int, start: int, duration: int) -> float:
    if step < start:
        return 0.0
    if duration <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - start) / duration))


def _integer_curriculum(start: int, target: int, step: int, ramp: int) -> int:
    if target <= start or ramp <= 0:
        return target
    progress = min(1.0, max(0.0, step / ramp))
    return min(target, start + int(math.floor(progress * (target - start) + 1e-9)))


def _sample_depth(
    *,
    minimum: int,
    maximum: int,
    distribution: str,
    seed: int,
    step: int,
) -> int:
    if distribution == "fixed" or maximum <= minimum:
        return maximum
    generator = random.Random(seed + 1_000_003 * int(step))
    if distribution == "uniform":
        return generator.randint(minimum, maximum)
    log_sample = generator.uniform(math.log(minimum), math.log(maximum + 1.0))
    return min(maximum, max(minimum, int(math.floor(math.exp(log_sample)))))


def auxiliary_schedule(
    step: int,
    model: LunarisIRL3Config,
    config: ObjectiveConfig,
) -> AuxiliarySchedule:
    """Resolve a synchronized depth/credit curriculum on every rank."""

    ceiling = _integer_curriculum(
        config.compute_start_steps,
        model.irl3_train_steps,
        step,
        config.compute_ramp_steps,
    )
    depth = _sample_depth(
        minimum=config.compute_start_steps,
        maximum=ceiling,
        distribution=config.depth_sampling,
        seed=config.depth_sampling_seed,
        step=step,
    )
    credit_ceiling = _integer_curriculum(
        config.gradient_horizon_start,
        model.irl3_gradient_horizon,
        step,
        config.gradient_horizon_ramp_steps,
    )
    gradient_horizon = min(depth, credit_ceiling)

    ponder_weight = config.ponder_weight * _linear_ramp(
        step, config.ponder_warmup_steps, config.ponder_ramp_steps
    )
    equilibrium_weight = config.equilibrium_weight * _linear_ramp(
        step,
        config.equilibrium_warmup_steps,
        config.equilibrium_ramp_steps,
    )
    mtp_weight = config.mtp_loss_weight * _linear_ramp(
        step, config.mtp_warmup_steps, config.mtp_ramp_steps
    )
    if config.mtp_loss_weight <= 0.0 or model.mtp_num_heads <= 0:
        active_heads = 0
    elif config.mtp_horizon_interval <= 0:
        active_heads = model.mtp_num_heads
    else:
        elapsed = max(0, step - config.mtp_warmup_steps)
        active_heads = min(
            model.mtp_num_heads,
            1 + elapsed // config.mtp_horizon_interval,
        )
    return AuxiliarySchedule(
        compute_depth=depth,
        depth_ceiling=ceiling,
        gradient_horizon=gradient_horizon,
        mtp_active_heads=active_heads,
        mtp_loss_weight=mtp_weight,
        ponder_weight=ponder_weight,
        equilibrium_weight=equilibrium_weight,
    )


__all__ = [
    "AuxiliarySchedule",
    "apply_learning_rate",
    "auxiliary_schedule",
    "lr_multiplier",
]
