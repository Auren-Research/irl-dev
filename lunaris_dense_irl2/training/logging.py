"""Low-overhead console, JSONL, and optional W&B logging."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import LoggingConfig, TrainConfig
from .distributed import DistributedContext


def estimate_training_flops_per_token(
    model: Any,
    sequence_length: int,
) -> float:
    """Estimate forward-plus-backward FLOPs for the dense IRL architecture."""

    d_model = model.d_model
    kv_width = model.kv_dim
    attention_projection = 2.0 * d_model * (2 * kv_width + 2 * d_model)
    if model.architecture_name == "lunaris_dense_irl_core":
        ffn_projection = 6.0 * d_model * model.ffn_hidden_dim
        ordinary_forward = model.n_ordinary_layers * (
            attention_projection + ffn_projection
        )
        anchor_kv_projection = 4.0 * d_model * kv_width
        recurrent_projection = model.irl_train_iters * (
            4.0 * d_model * d_model + ffn_projection
        )
        attention_mixing = 0.0
        for layer_index in range(model.n_layers):
            if layer_index == model.core_layer_index:
                continue
            visible = (
                min(sequence_length, model.local_window)
                if model.attention_type(layer_index) == "local"
                else sequence_length
            )
            attention_mixing += 4.0 * visible * d_model
        attention_mixing += (
            model.irl_train_iters * 4.0 * sequence_length * d_model
        )
        output_forward = 2.0 * d_model * model.vocab_size
        return 3.0 * (
            ordinary_forward
            + anchor_kv_projection
            + recurrent_projection
            + attention_mixing
            + output_forward
        )

    irl_projection = 6.0 * d_model * model.ffn_hidden_dim * model.irl_train_iters

    attention_mixing = 0.0
    for layer_index in range(model.n_layers):
        visible = (
            min(sequence_length, model.local_window)
            if model.attention_type(layer_index) == "local"
            else sequence_length
        )
        attention_mixing += 4.0 * visible * d_model

    block_forward = (
        model.n_layers * (attention_projection + irl_projection) + attention_mixing
    )
    output_forward = 2.0 * d_model * model.vocab_size
    mtp_forward = 0.0
    if model.mtp_num_heads > 0:
        mtp_hidden = model.mtp_ffn_hidden_dim or model.d_model
        mtp_fusion = 4.0 * d_model * d_model
        mtp_irl = 6.0 * d_model * mtp_hidden * model.mtp_irl_iters
        mtp_output = 2.0 * d_model * model.vocab_size
        mtp_forward = model.mtp_num_heads * (
            mtp_fusion + mtp_irl + mtp_output
        )
    return 3.0 * (block_forward + output_forward + mtp_forward)


class MetricLogger:
    """Write synchronized training metrics from rank zero."""

    def __init__(
        self,
        config: LoggingConfig,
        train_config: TrainConfig,
        context: DistributedContext,
    ) -> None:
        self.config = config
        self.context = context
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.last_tokens = 0
        self.jsonl_path = (
            Path(config.jsonl_path) if config.jsonl_path is not None else None
        )
        self.wandb: Optional[Any] = None
        if not context.is_master:
            return

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if config.wandb_project is not None:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name,
                    config=train_config.to_dict(),
                )
            except Exception as exc:
                print(f"[warning] W&B initialization failed: {exc}", flush=True)
                self.wandb = None

    def log(self, metrics: Dict[str, Any]) -> None:
        """Emit one compact console line and persist structured metrics."""

        if not self.context.is_master:
            return
        serializable = {
            key: float(value) if isinstance(value, torch.Tensor) else value
            for key, value in metrics.items()
        }
        step = int(serializable["step"])
        loss = float(serializable["loss"])
        lr = float(serializable["learning_rate"])
        tokens_per_second = float(serializable["tokens_per_second"])
        grad_norm = float(serializable["gradient_norm"])
        clipped_fraction = float(serializable.get("clipped_step_fraction", 0.0))
        memory = float(serializable.get("memory_allocated_gib", 0.0))
        validation = serializable.get("validation_loss")
        validation_text = "" if validation is None else f" val={float(validation):.4f}"
        print(
            f"step={step:,} loss={loss:.4f}{validation_text} "
            f"lr={lr:.3e} grad={grad_norm:.3f} "
            f"clip={clipped_fraction:.1%} tok/s={tokens_per_second:,.0f} "
            f"mem={memory:.1f}GiB",
            flush=True,
        )

        if self.jsonl_path is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(serializable, sort_keys=True) + "\n")
        if self.wandb is not None:
            self.wandb.log(serializable, step=step)

    def throughput(
        self,
        *,
        tokens_seen: int,
        flops_per_token: float,
    ) -> Dict[str, float]:
        """Measure interval throughput and optional estimated MFU."""

        now = time.monotonic()
        elapsed = max(1e-9, now - self.last_time)
        token_delta = max(0, tokens_seen - self.last_tokens)
        tokens_per_second = token_delta / elapsed
        self.last_time = now
        self.last_tokens = tokens_seen

        metrics = {
            "tokens_per_second": tokens_per_second,
            "elapsed_seconds": now - self.start_time,
        }
        if self.config.peak_tflops_per_gpu is not None:
            achieved = tokens_per_second * flops_per_token / 1e12
            peak = self.config.peak_tflops_per_gpu * self.context.world_size
            metrics["estimated_tflops"] = achieved
            metrics["estimated_mfu"] = achieved / peak
        return metrics

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Finalize the optional remote logger."""

        if not self.context.is_master or self.wandb is None:
            return
        if summary is not None and self.wandb.run is not None:
            for key, value in summary.items():
                self.wandb.run.summary[key] = value
        self.wandb.finish()


def memory_metrics(device: torch.device) -> Dict[str, float]:
    """Return portable accelerator memory counters."""

    if device.type != "cuda":
        return {
            "memory_allocated_gib": 0.0,
            "memory_reserved_gib": 0.0,
            "memory_peak_gib": 0.0,
        }
    divisor = float(1024**3)
    return {
        "memory_allocated_gib": torch.cuda.memory_allocated(device) / divisor,
        "memory_reserved_gib": torch.cuda.memory_reserved(device) / divisor,
        "memory_peak_gib": torch.cuda.max_memory_allocated(device) / divisor,
    }


def perplexity(loss: float) -> float:
    """Convert a cross-entropy value to a numerically safe perplexity."""

    return math.exp(loss) if loss < 20.0 else float("inf")
