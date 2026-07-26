"""Reshardable distributed checkpoints with exact per-rank RNG state."""

from __future__ import annotations

import json
import random
import shutil
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)

from .config import CheckpointConfig, TrainConfig
from .distributed import DistributedContext
from .optimizers import OptimizerBundle


@dataclass
class TrainingState:
    """Small replicated state stored beside sharded tensors."""

    step: int = 0
    microbatches_consumed: int = 0
    global_samples_seen: int = 0
    tokens_seen: int = 0
    best_validation_loss: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to JSON-compatible values."""

        return {
            "step": self.step,
            "microbatches_consumed": self.microbatches_consumed,
            "global_samples_seen": self.global_samples_seen,
            "tokens_seen": self.tokens_seen,
            "best_validation_loss": self.best_validation_loss,
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TrainingState":
        """Restore state from checkpoint metadata."""

        return cls(
            step=int(values.get("step", 0)),
            microbatches_consumed=int(values.get("microbatches_consumed", 0)),
            global_samples_seen=int(values.get("global_samples_seen", 0)),
            tokens_seen=int(values.get("tokens_seen", 0)),
            best_validation_loss=float(
                values.get("best_validation_loss", float("inf"))
            ),
        )


def capture_rng_state() -> Dict[str, Any]:
    """Capture process-local random number generator state."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["accelerator"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore process-local random number generator state."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "accelerator" in state:
        torch.cuda.set_rng_state(state["accelerator"])


class CheckpointManager:
    """Save and restore DDP/FSDP2 checkpoints through PyTorch DCP."""

    def __init__(
        self,
        config: CheckpointConfig,
        train_config: TrainConfig,
        context: DistributedContext,
    ) -> None:
        self.config = config
        self.train_config = train_config
        self.context = context
        self.root = Path(config.directory)
        self._pending: Optional[tuple[Path, TrainingState, Future[Any]]] = None
        self.resume_world_size: Optional[int] = None
        self.restored_rng = False
        if context.is_master:
            self.root.mkdir(parents=True, exist_ok=True)
        context.barrier()

    @staticmethod
    def _step_from_path(path: Path) -> int:
        """Parse a numbered checkpoint directory."""

        try:
            return int(path.name.removeprefix("step_"))
        except ValueError:
            return -1

    def completed_checkpoints(self) -> list[Path]:
        """List complete checkpoints in ascending step order."""

        return sorted(
            (
                path
                for path in self.root.glob("step_*")
                if path.is_dir() and (path / "_SUCCESS").exists()
            ),
            key=self._step_from_path,
        )

    def resolve_resume_path(self) -> Optional[Path]:
        """Resolve none, auto, or an explicit checkpoint path."""

        resume = self.config.resume
        if resume.lower() in {"none", "false", "off"}:
            return None
        if resume.lower() == "auto":
            checkpoints = self.completed_checkpoints()
            return checkpoints[-1] if checkpoints else None
        path = Path(resume)
        if not path.is_absolute():
            path = self.root / path
        if not (path / "_SUCCESS").exists():
            raise FileNotFoundError(f"Incomplete or missing checkpoint: {path}")
        return path

    def _tensor_state(
        self,
        model: nn.Module,
        optimizers: OptimizerBundle,
    ) -> Dict[str, Any]:
        """Build canonical state dicts that DCP can reshard on load."""

        model_state, optimizer_state = get_state_dict(
            model,
            optimizers.optimizers,
        )
        return {
            "model": model_state,
            "optimizer": optimizer_state,
        }

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizers: OptimizerBundle,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> TrainingState:
        """Load tensors, optimizer state, metadata, and local RNG state."""

        state = self._tensor_state(model, optimizers)
        dcp.load(
            state,
            checkpoint_id=path / "dcp",
            no_dist=not self.context.is_distributed,
        )
        set_state_dict(
            model,
            optimizers.optimizers,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )

        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.resume_world_size = int(metadata.get("world_size_at_save", 1))
        rank_state_path = path / f"rank_{self.context.rank:05d}.pt"
        same_topology = self.resume_world_size == self.context.world_size
        if same_topology and rank_state_path.exists():
            local_state = torch.load(
                rank_state_path,
                map_location="cpu",
                weights_only=False,
            )
            restore_rng_state(local_state["rng"])
            if (
                scaler is not None
                and "scaler" in local_state
                and local_state["scaler"] is not None
            ):
                scaler.load_state_dict(local_state["scaler"])
            self.restored_rng = True
        state = TrainingState.from_dict(metadata["training_state"])
        if state.global_samples_seen == 0 and state.tokens_seen > 0:
            sequence_length = int(self.train_config.data.sequence_length)
            state.global_samples_seen = state.tokens_seen // sequence_length
        return state

    def _write_completion_files(
        self,
        path: Path,
        state: TrainingState,
    ) -> None:
        """Publish metadata only after every DCP writer has completed."""

        self.context.barrier()
        if self.context.is_master:
            metadata = {
                "format_version": 2,
                "training_state": state.to_dict(),
                "train_config": self.train_config.to_dict(),
                "world_size_at_save": self.context.world_size,
            }
            (path / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (path / "_SUCCESS").write_text("complete\n", encoding="utf-8")
        self.context.barrier()
        self._prune()

    def wait(self) -> None:
        """Finalize any asynchronous checkpoint."""

        if self._pending is None:
            return
        path, state, future = self._pending
        future.result()
        self._write_completion_files(path, state)
        self._pending = None

    def save(
        self,
        state: TrainingState,
        model: nn.Module,
        optimizers: OptimizerBundle,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Path:
        """Start or complete a sharded checkpoint save."""

        self.wait()
        if self.config.min_free_space_gib > 0.0:
            free_gib = shutil.disk_usage(self.root).free / float(1024**3)
            free_tensor = torch.tensor(
                free_gib,
                device=self.context.device,
                dtype=torch.float64,
            )
            minimum_free_gib = float(
                self.context.reduce_min(free_tensor).cpu().item()
            )
            if minimum_free_gib < self.config.min_free_space_gib:
                raise OSError(
                    f"Checkpoint filesystem has at least one rank with only "
                    f"{minimum_free_gib:.2f} GiB free, "
                    f"below checkpoint.min_free_space_gib="
                    f"{self.config.min_free_space_gib:.2f}"
                )
        path = self.root / f"step_{state.step:08d}"
        if self.context.is_master:
            if path.exists():
                if (path / "_SUCCESS").exists():
                    raise FileExistsError(
                        f"Refusing to overwrite complete checkpoint: {path}"
                    )
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        self.context.barrier()

        torch.save(
            {
                "rng": capture_rng_state(),
                "rank": self.context.rank,
                "scaler": None if scaler is None else scaler.state_dict(),
            },
            path / f"rank_{self.context.rank:05d}.pt",
        )
        tensor_state = self._tensor_state(model, optimizers)

        if self.config.async_save and self.context.is_distributed:
            future = dcp.async_save(
                tensor_state,
                checkpoint_id=path / "dcp",
            )
            self._pending = (path, TrainingState(**state.to_dict()), future)
        else:
            dcp.save(
                tensor_state,
                checkpoint_id=path / "dcp",
                no_dist=not self.context.is_distributed,
            )
            self._write_completion_files(path, state)
        return path

    def _prune(self) -> None:
        """Retain only the configured number of completed checkpoints."""

        self.context.barrier()
        if self.context.is_master:
            checkpoints = self.completed_checkpoints()
            for path in checkpoints[: -self.config.keep_last]:
                shutil.rmtree(path)
        self.context.barrier()
