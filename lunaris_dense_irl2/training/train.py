"""Production-oriented pretraining entrypoint for Lunaris Dense IRL-2."""

from __future__ import annotations

import argparse
import math
import random
import signal
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .checkpointing import CheckpointManager, TrainingState
from .config import TrainConfig
from .data import (
    DevicePrefetcher,
    MemoryMappedTokenDataset,
    build_dataloader,
)
from .distributed import (
    DistributedContext,
    autocast_context,
    compile_model,
    configure_runtime,
    maybe_no_sync,
    parallelize_model,
    resolve_strategy,
    set_gradient_sync,
)
from .logging import (
    MetricLogger,
    estimate_training_flops_per_token,
    memory_metrics,
    perplexity,
)
from .model_factory import build_model
from .optimizers import (
    GradientClipper,
    OptimizerBundle,
    build_optimizers,
    restore_base_learning_rates,
)
from .schedules import (
    apply_learning_rate,
    auxiliary_schedule,
    lr_multiplier,
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU, CUDA, and ROCm generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_grad_scaler(
    config: TrainConfig,
    context: DistributedContext,
) -> Optional[torch.amp.GradScaler]:
    """Create a scaler only for FP16 accelerator training."""

    if config.parallel.precision != "fp16" or context.device.type != "cuda":
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def optimizer_step(
    optimizers: OptimizerBundle,
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    """Step all optimizers with one shared loss scale."""

    if scaler is None:
        optimizers.step()
        return
    for optimizer in optimizers.optimizers:
        scaler.step(optimizer)
    scaler.update()


def unscale_gradients(
    optimizers: OptimizerBundle,
    scaler: Optional[torch.amp.GradScaler],
) -> None:
    """Unscale gradients before clipping."""

    if scaler is None:
        return
    for optimizer in optimizers.optimizers:
        scaler.unscale_(optimizer)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[torch.Tensor],
    config: TrainConfig,
    context: DistributedContext,
) -> Dict[str, float]:
    """Compute token-weighted validation loss across all ranks."""

    model.eval()
    prefetcher = DevicePrefetcher(loader, context.device)
    ce_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    z_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    mtp_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    consistency_sum = torch.zeros((), device=context.device, dtype=torch.float64)
    token_sum = torch.zeros((), device=context.device, dtype=torch.float64)

    for _ in range(config.validation.batches):
        try:
            tokens = prefetcher.next()
        except StopIteration:
            break
        predicted_tokens = tokens.size(0) * (tokens.size(1) - 1)
        with autocast_context(context.device, config.parallel.precision):
            output = model(
                tokens,
                labels=tokens,
                return_logits=False,
                return_irl_stats=False,
                loss_chunk_size=config.training.loss_chunk_size,
                z_loss_coefficient=config.training.z_loss_coefficient,
                loss_backend=config.training.loss_backend,
                n_iters=config.model.irl_eval_iters,
                mtp_loss_weight=config.objectives.mtp_loss_weight,
                mtp_active_heads=config.model.mtp_num_heads,
                mtp_horizon_decay=config.objectives.mtp_horizon_decay,
                irl_consistency_coefficient=(
                    config.objectives.irl_consistency_weight
                ),
            )
        if (
            output.cross_entropy is None
            or output.z_loss is None
            or output.mtp_loss is None
            or output.irl_consistency_loss is None
        ):
            raise RuntimeError("The model did not return validation losses")
        ce_sum.add_(output.cross_entropy.detach().double() * predicted_tokens)
        z_sum.add_(output.z_loss.detach().double() * predicted_tokens)
        mtp_sum.add_(output.mtp_loss.detach().double() * predicted_tokens)
        consistency_sum.add_(
            output.irl_consistency_loss.detach().double() * predicted_tokens
        )
        token_sum.add_(predicted_tokens)

    totals = context.reduce_sum(
        torch.stack((ce_sum, z_sum, mtp_sum, consistency_sum, token_sum))
    )
    denominator = totals[4].clamp_min(1.0)
    cross_entropy = float((totals[0] / denominator).item())
    z_loss = float((totals[1] / denominator).item())
    mtp_loss = float((totals[2] / denominator).item())
    consistency_loss = float((totals[3] / denominator).item())
    model.train()
    return {
        "validation_cross_entropy": cross_entropy,
        "validation_z_loss": z_loss,
        "validation_mtp_loss": mtp_loss,
        "validation_irl_consistency_loss": consistency_loss,
        "validation_loss": (
            cross_entropy
            + config.training.z_loss_coefficient * z_loss
            + config.objectives.mtp_loss_weight * mtp_loss
            + config.objectives.irl_consistency_weight * consistency_loss
        ),
        "validation_perplexity": perplexity(cross_entropy),
    }


def build_profiler(
    config: TrainConfig,
    context: DistributedContext,
):
    """Create an optional per-rank PyTorch profiler context."""

    if config.logging.profile_steps <= 0:
        return nullcontext(None)

    trace_directory = Path(config.logging.profile_directory)
    trace_directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if context.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_index = 0

    def export_trace(profiler: torch.profiler.profile) -> None:
        nonlocal trace_index
        destination = trace_directory / (
            f"rank_{context.rank:05d}_trace_{trace_index:03d}.json"
        )
        profiler.export_chrome_trace(str(destination))
        trace_index += 1

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.logging.profile_wait,
            warmup=config.logging.profile_warmup,
            active=config.logging.profile_steps,
            repeat=1,
        ),
        on_trace_ready=export_trace,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )


def _resolve_runtime_steps(
    config: TrainConfig,
    context: DistributedContext,
) -> int:
    """Resolve a token budget against the actual data-parallel world size."""

    max_steps = config.training.max_steps
    if config.training.max_tokens is None:
        return max_steps

    tokens_per_step = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * int(config.data.sequence_length)
        * context.world_size
    )
    token_limited_steps = math.ceil(config.training.max_tokens / tokens_per_step)
    max_steps = min(max_steps, token_limited_steps)
    config.training.max_steps = max_steps

    if config.schedule.name == "wsd":
        available = max(0, max_steps - config.schedule.warmup_steps)
        decay_steps = min(int(config.schedule.decay_steps or 0), available)
        config.schedule.decay_steps = decay_steps
        config.schedule.stable_steps = max(0, available - decay_steps)
    return max_steps


def _fit_schedule_to_steps(config: TrainConfig, max_steps: int) -> None:
    """Shrink schedule phases for an explicit short-run override."""

    config.schedule.warmup_steps = min(config.schedule.warmup_steps, max_steps)
    if config.schedule.name == "wsd":
        available = max(0, max_steps - config.schedule.warmup_steps)
        decay_steps = min(int(config.schedule.decay_steps or 0), available)
        config.schedule.decay_steps = decay_steps
        config.schedule.stable_steps = max(0, available - decay_steps)
    else:
        config.schedule.decay_steps = max(
            0,
            max_steps - config.schedule.warmup_steps,
        )


def _print_startup(
    config: TrainConfig,
    context: DistributedContext,
    strategy: str,
    model: nn.Module,
    train_dataset: MemoryMappedTokenDataset,
    max_steps: int,
) -> None:
    """Print one concise resolved-run summary."""

    if not context.is_master:
        return
    parameters = sum(parameter.numel() for parameter in model.parameters())
    global_batch = (
        config.training.micro_batch_size
        * config.training.gradient_accumulation_steps
        * context.world_size
    )
    tokens_per_step = global_batch * int(config.data.sequence_length)
    print("=" * 88)
    print("LUNARIS DENSE IRL-2 :: LARGE-SCALE PRETRAINING")
    print("=" * 88)
    print(
        f"accelerator={context.accelerator} world={context.world_size} "
        f"strategy={strategy} precision={config.parallel.precision}"
    )
    print(
        f"parameters={parameters:,} sequence={config.data.sequence_length:,} "
        f"global_batch={global_batch:,} tokens/step={tokens_per_step:,}"
    )
    print(
        f"dataset={train_dataset.total_tokens:,} tokens in "
        f"{len(train_dataset.shards):,} shards max_steps={max_steps:,}"
    )
    print(
        f"optimizer={config.optimizer.name} schedule={config.schedule.name} "
        f"compile={config.parallel.compile}"
    )
    print("=" * 88, flush=True)


def train(
    config_path: str | Path,
    *,
    max_steps_override: Optional[int] = None,
    resume_override: Optional[str] = None,
) -> None:
    """Run resumable single-node or multi-node pretraining."""

    config = TrainConfig.from_yaml(config_path)
    if resume_override is not None:
        config.checkpoint.resume = resume_override
    if max_steps_override is not None:
        config.training.max_steps = int(max_steps_override)
        config.training.max_tokens = None
        _fit_schedule_to_steps(config, config.training.max_steps)
        config.schedule.validate(config.training.max_steps)

    context = DistributedContext.initialize(config.parallel)
    configure_runtime(context, config.parallel)
    strategy = resolve_strategy(config.parallel, context)
    if config.optimizer.name == "mona" and strategy == "fsdp2":
        raise ValueError("MONA is only supported with replicated parameters")
    if config.training.gradient_clip_mode == "adaptive_group" and strategy == "fsdp2":
        raise ValueError(
            "Adaptive group clipping requires full replicated parameter groups"
        )
    max_steps = _resolve_runtime_steps(config, context)
    _fit_schedule_to_steps(config, max_steps)

    # Every rank must construct identical weights before DDP/FSDP2 setup.
    seed_everything(config.training.seed)
    train_dataset = MemoryMappedTokenDataset(
        config.data.train_dir,
        sequence_length=int(config.data.sequence_length),
        shard_glob=config.data.shard_glob,
        mmap_cache_size=config.data.mmap_cache_size,
    )
    validation_dataset: Optional[MemoryMappedTokenDataset] = None
    if config.data.validation_dir is not None:
        validation_path = Path(config.data.validation_dir)
        if validation_path.is_dir():
            validation_dataset = MemoryMappedTokenDataset(
                validation_path,
                sequence_length=int(config.data.sequence_length),
                shard_glob=config.data.shard_glob,
                mmap_cache_size=config.data.mmap_cache_size,
            )

    architecture = build_model(config.model)
    architecture.set_regional_compile(config.parallel.regional_compile)
    compile_model(architecture, config.parallel, context)
    model = parallelize_model(
        architecture,
        config.parallel,
        context,
        strategy,
    )
    optimizers = build_optimizers(model, config.optimizer)
    scaler = create_grad_scaler(config, context)
    clipper = GradientClipper(config.training)
    checkpoints = CheckpointManager(config.checkpoint, config, context)

    state = TrainingState()
    resume_path = checkpoints.resolve_resume_path()
    if resume_path is not None:
        if context.is_master:
            print(f"[checkpoint] loading {resume_path}", flush=True)
        state = checkpoints.load(
            resume_path,
            model,
            optimizers,
            scaler,
        )
        restore_base_learning_rates(optimizers, config.optimizer)

    # New runs use rank-local dropout streams after identical initialization.
    if resume_path is None:
        seed_everything(config.training.seed + context.rank)
    elif not checkpoints.restored_rng:
        seed_everything(
            config.training.seed
            + context.rank
            + 1_000_003 * state.step
        )
        if context.is_master:
            print(
                "[checkpoint] topology changed; model/optimizer/data cursor "
                "were resharded, and rank-local RNG streams were reseeded",
                flush=True,
            )
    train_loader = build_dataloader(
        train_dataset,
        config.data,
        batch_size=config.training.micro_batch_size,
        rank=context.rank,
        world_size=context.world_size,
        training=True,
        start_global_sample=state.global_samples_seen,
    )
    train_prefetcher = DevicePrefetcher(train_loader, context.device)

    validation_loader = None
    if validation_dataset is not None and config.validation.batches > 0:
        validation_loader = build_dataloader(
            validation_dataset,
            config.data,
            batch_size=config.training.micro_batch_size,
            rank=context.rank,
            world_size=context.world_size,
            training=False,
            max_batches=config.validation.batches,
        )

    logger = MetricLogger(config.logging, config, context)
    logger.last_tokens = state.tokens_seen
    flops_per_token = estimate_training_flops_per_token(
        config.model,
        int(config.data.sequence_length),
    )
    _print_startup(
        config,
        context,
        strategy,
        model,
        train_dataset,
        max_steps,
    )

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        if context.is_master:
            print(
                f"[signal] received {signum}; checkpointing after this step",
                flush=True,
            )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    optimizers.zero_grad(set_to_none=True)
    model.train()
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    window_loss = torch.zeros((), device=context.device)
    window_ce = torch.zeros((), device=context.device)
    window_z = torch.zeros((), device=context.device)
    window_mtp = torch.zeros((), device=context.device)
    window_consistency = torch.zeros((), device=context.device)
    window_clipped_steps = torch.zeros((), device=context.device)
    window_steps = 0
    latest_grad_norm = torch.zeros((), device=context.device)
    latest_validation: Dict[str, float] = {}
    last_checkpoint_step = state.step

    if validation_loader is not None and config.validation.at_start and state.step == 0:
        latest_validation = run_validation(
            model,
            validation_loader,
            config,
            context,
        )

    try:
        with build_profiler(config, context) as profiler:
            while state.step < max_steps and not stop_requested:
                multiplier = lr_multiplier(
                    state.step,
                    config.schedule,
                    max_steps,
                )
                apply_learning_rate(optimizers.optimizers, multiplier)
                auxiliary = auxiliary_schedule(
                    state.step,
                    config.model,
                    config.objectives,
                )
                step_loss = torch.zeros((), device=context.device)
                step_ce = torch.zeros((), device=context.device)
                step_z = torch.zeros((), device=context.device)
                step_mtp = torch.zeros((), device=context.device)
                step_consistency = torch.zeros((), device=context.device)

                for micro_step in range(config.training.gradient_accumulation_steps):
                    synchronize = (
                        micro_step == config.training.gradient_accumulation_steps - 1
                    )
                    set_gradient_sync(model, strategy, synchronize)
                    tokens = train_prefetcher.next()
                    with maybe_no_sync(model, strategy, synchronize):
                        with autocast_context(
                            context.device,
                            config.parallel.precision,
                        ):
                            output = model(
                                tokens,
                                labels=tokens,
                                return_logits=False,
                                return_irl_stats=False,
                                loss_chunk_size=config.training.loss_chunk_size,
                                z_loss_coefficient=(config.training.z_loss_coefficient),
                                loss_backend=config.training.loss_backend,
                                n_iters=auxiliary.irl_iters,
                                mtp_loss_weight=auxiliary.mtp_loss_weight,
                                mtp_active_heads=auxiliary.mtp_active_heads,
                                mtp_horizon_decay=(
                                    config.objectives.mtp_horizon_decay
                                ),
                                irl_consistency_coefficient=(
                                    auxiliary.irl_consistency_weight
                                ),
                            )
                            if (
                                output.loss is None
                                or output.cross_entropy is None
                                or output.z_loss is None
                                or output.mtp_loss is None
                                or output.irl_consistency_loss is None
                            ):
                                raise RuntimeError(
                                    "The model did not return training losses"
                                )
                            scaled_loss = (
                                output.loss
                                / config.training.gradient_accumulation_steps
                            )

                        if scaler is None:
                            scaled_loss.backward()
                        else:
                            scaler.scale(scaled_loss).backward()

                    accumulation_scale = (
                        1.0 / config.training.gradient_accumulation_steps
                    )
                    step_loss.add_(
                        output.loss.detach(),
                        alpha=accumulation_scale,
                    )
                    step_ce.add_(
                        output.cross_entropy.detach(),
                        alpha=accumulation_scale,
                    )
                    step_z.add_(
                        output.z_loss.detach(),
                        alpha=accumulation_scale,
                    )
                    step_mtp.add_(
                        output.mtp_loss.detach(),
                        alpha=accumulation_scale,
                    )
                    step_consistency.add_(
                        output.irl_consistency_loss.detach(),
                        alpha=accumulation_scale,
                    )
                    state.microbatches_consumed += 1
                    state.global_samples_seen += tokens.size(0) * context.world_size
                    state.tokens_seen += tokens.numel() * context.world_size

                unscale_gradients(optimizers, scaler)
                latest_grad_norm = clipper.clip(
                    model,
                    error_if_nonfinite=config.training.fail_on_nonfinite,
                )
                if clipper.last_was_clipped is not None:
                    window_clipped_steps.add_(clipper.last_was_clipped)
                optimizer_step(optimizers, scaler)
                optimizers.zero_grad(set_to_none=True)
                state.step += 1

                window_loss.add_(step_loss)
                window_ce.add_(step_ce)
                window_z.add_(step_z)
                window_mtp.add_(step_mtp)
                window_consistency.add_(step_consistency)
                window_steps += 1

                if profiler is not None:
                    profiler.step()

                should_validate = (
                    validation_loader is not None
                    and config.validation.interval > 0
                    and state.step % config.validation.interval == 0
                )
                if should_validate:
                    latest_validation = run_validation(
                        model,
                        validation_loader,
                        config,
                        context,
                    )
                    state.best_validation_loss = min(
                        state.best_validation_loss,
                        latest_validation["validation_loss"],
                    )

                should_log = state.step % config.logging.interval == 0
                if should_log:
                    averaged = torch.stack(
                        (
                            window_loss / max(1, window_steps),
                            window_ce / max(1, window_steps),
                            window_z / max(1, window_steps),
                            window_mtp / max(1, window_steps),
                            window_consistency / max(1, window_steps),
                            latest_grad_norm.float(),
                            window_clipped_steps / max(1, window_steps),
                        )
                    )
                    averaged = context.reduce_mean(averaged)
                    current_lr = float(optimizers.optimizers[0].param_groups[0]["lr"])
                    metrics: Dict[str, Any] = {
                        "step": state.step,
                        "tokens_seen": state.tokens_seen,
                        "loss": float(averaged[0].item()),
                        "cross_entropy": float(averaged[1].item()),
                        "z_loss": float(averaged[2].item()),
                        "mtp_loss": float(averaged[3].item()),
                        "irl_consistency_loss": float(averaged[4].item()),
                        "perplexity": perplexity(float(averaged[1].item())),
                        "learning_rate": current_lr,
                        "gradient_norm": float(averaged[5].item()),
                        "pre_clip_gradient_norm": float(averaged[5].item()),
                        "clipped_step_fraction": float(averaged[6].item()),
                        "irl_train_iters": auxiliary.irl_iters,
                        "mtp_active_heads": auxiliary.mtp_active_heads,
                        "mtp_loss_weight": auxiliary.mtp_loss_weight,
                        "irl_consistency_weight": (
                            auxiliary.irl_consistency_weight
                        ),
                        "world_size": context.world_size,
                    }
                    metrics.update(
                        logger.throughput(
                            tokens_seen=state.tokens_seen,
                            flops_per_token=flops_per_token,
                        )
                    )
                    metrics.update(memory_metrics(context.device))
                    metrics.update(latest_validation)
                    logger.log(metrics)
                    window_loss.zero_()
                    window_ce.zero_()
                    window_z.zero_()
                    window_mtp.zero_()
                    window_consistency.zero_()
                    window_clipped_steps.zero_()
                    window_steps = 0

                should_checkpoint = (
                    config.checkpoint.interval > 0
                    and state.step % config.checkpoint.interval == 0
                )
                if should_checkpoint:
                    path = checkpoints.save(
                        state,
                        model,
                        optimizers,
                        scaler,
                    )
                    last_checkpoint_step = state.step
                    if context.is_master:
                        mode = (
                            "started asynchronous save"
                            if config.checkpoint.async_save and context.is_distributed
                            else "saved"
                        )
                        print(f"[checkpoint] {mode}: {path}", flush=True)

        if config.checkpoint.save_final and state.step != last_checkpoint_step:
            checkpoints.save(state, model, optimizers, scaler)
            last_checkpoint_step = state.step
        checkpoints.wait()
    finally:
        try:
            checkpoints.wait()
        finally:
            logger.close(
                {
                    "final_step": state.step,
                    "tokens_seen": state.tokens_seen,
                    "best_validation_loss": state.best_validation_loss,
                    "stopped_by_signal": stop_requested,
                }
            )
            context.close()


def main() -> None:
    """Parse command-line arguments and start training."""

    parser = argparse.ArgumentParser(
        description="Train Lunaris Dense IRL-2 with DDP or FSDP2."
    )
    parser.add_argument("--config", required=True, help="Training YAML path")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional short-run override for smoke testing",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Override checkpoint.resume with auto, none, or a path",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate and print the resolved config without allocating a model",
    )
    args = parser.parse_args()

    if args.validate_config:
        config = TrainConfig.from_yaml(args.config)
        if args.resume is not None:
            config.checkpoint.resume = args.resume
        print(config.to_dict())
        return

    train(
        args.config,
        max_steps_override=args.max_steps,
        resume_override=args.resume,
    )


if __name__ == "__main__":
    main()
