"""NCCL/RCCL-neutral distributed runtime and model parallelization."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..model import LunarisDenseForCausalLM
from .config import ParallelConfig


@dataclass
class DistributedContext:
    """Process topology initialized from torchrun environment variables."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: Optional[str]
    initialized_here: bool

    @property
    def is_distributed(self) -> bool:
        """Return whether multiple data-parallel processes are active."""

        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        """Return whether this process owns user-facing logs."""

        return self.rank == 0

    @property
    def accelerator(self) -> str:
        """Identify the current accelerator software stack."""

        if self.device.type != "cuda":
            return "cpu"
        if torch.version.hip is not None:
            return f"amd-rocm-{torch.version.hip}"
        return f"nvidia-cuda-{torch.version.cuda}"

    @classmethod
    def initialize(cls, config: ParallelConfig) -> "DistributedContext":
        """Initialize a torchrun job using NCCL, RCCL, or Gloo."""

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo" if world_size > 1 else None

        initialized_here = False
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=config.timeout_minutes),
            )
            initialized_here = True

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            backend=backend,
            initialized_here=initialized_here,
        )

    def barrier(self) -> None:
        """Synchronize ranks when a process group exists."""

        if self.is_distributed:
            dist.barrier()

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a scalar tensor and return its global mean."""

        output = tensor.detach().clone()
        if self.is_distributed:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            output.div_(self.world_size)
        return output

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor and return its global sum."""

        output = tensor.detach().clone()
        if self.is_distributed:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce a tensor and return its global minimum."""

        output = tensor.detach().clone()
        if self.is_distributed:
            dist.all_reduce(output, op=dist.ReduceOp.MIN)
        return output

    def close(self) -> None:
        """Destroy only the process group created by this context."""

        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def resolve_strategy(config: ParallelConfig, context: DistributedContext) -> str:
    """Resolve the automatic parallel strategy for the current world size."""

    if not context.is_distributed:
        return "none"
    if config.strategy == "auto":
        return "fsdp2"
    if config.strategy == "none":
        raise ValueError("parallel.strategy=none cannot be used with WORLD_SIZE > 1")
    return config.strategy


def precision_dtype(name: str) -> Optional[torch.dtype]:
    """Map a precision name to its autocast dtype."""

    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def autocast_context(
    device: torch.device,
    precision: str,
) -> AbstractContextManager[Any]:
    """Create the appropriate mixed-precision context."""

    dtype = precision_dtype(precision)
    if dtype is None or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def configure_runtime(
    context: DistributedContext,
    config: ParallelConfig,
) -> None:
    """Enable safe backend-level throughput settings."""

    torch.set_float32_matmul_precision("high")
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(
            True,
            warn_only=config.deterministic_warn_only,
        )
    if context.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not config.deterministic


def fsdp_mesh_shape(
    world_size: int,
    replicate_group_size: int,
) -> tuple[int, ...]:
    """Resolve a 1D FSDP or 2D HSDP mesh without initializing collectives."""

    world_size = int(world_size)
    replicate_group_size = int(replicate_group_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if replicate_group_size <= 0:
        raise ValueError("replicate_group_size must be positive")
    if world_size % replicate_group_size != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by "
            "parallel.fsdp_replicate_group_size"
        )
    if replicate_group_size == 1:
        return (world_size,)
    shard_group_size = world_size // replicate_group_size
    if shard_group_size <= 1:
        raise ValueError(
            "HSDP requires at least two ranks in each FSDP shard group"
        )
    return (replicate_group_size, shard_group_size)


def compile_model(
    model: LunarisDenseForCausalLM,
    config: ParallelConfig,
    context: DistributedContext,
) -> None:
    """Compile the inner model before applying a distributed wrapper."""

    if not config.compile or context.device.type != "cuda":
        return
    model.compile(mode=config.compile_mode, dynamic=False)


def parallelize_model(
    model: LunarisDenseForCausalLM,
    config: ParallelConfig,
    context: DistributedContext,
    strategy: str,
) -> nn.Module:
    """Move and parallelize the model with DDP or composable FSDP2."""

    if strategy == "none":
        return model.to(context.device)

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        model = model.to(context.device)
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=config.ddp_static_graph,
        )

    if strategy == "fsdp2":
        if context.device.type != "cuda":
            raise RuntimeError("FSDP2 training requires a CUDA or ROCm accelerator")
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        param_dtype = precision_dtype(config.precision)
        reduce_dtype = (
            torch.bfloat16 if config.reduce_dtype == "bf16" else torch.float32
        )
        policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=param_dtype,
            cast_forward_inputs=True,
        )
        mesh_shape = fsdp_mesh_shape(
            context.world_size,
            config.fsdp_replicate_group_size,
        )
        mesh_names = (
            ("replicate", "shard")
            if len(mesh_shape) == 2
            else ("shard",)
        )
        mesh = init_device_mesh(
            "cuda",
            mesh_shape,
            mesh_dim_names=mesh_names,
        )
        for block in model.blocks:
            fully_shard(
                block,
                mesh=mesh,
                reshard_after_forward=config.reshard_after_forward,
                mp_policy=policy,
            )
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=config.reshard_after_forward,
            mp_policy=policy,
        )
        return model

    raise ValueError(f"Unsupported parallel strategy: {strategy}")


def set_gradient_sync(model: nn.Module, strategy: str, enabled: bool) -> None:
    """Control gradient synchronization for accumulation micro-batches."""

    if strategy == "fsdp2":
        model.set_requires_gradient_sync(enabled)  # type: ignore[attr-defined]


def maybe_no_sync(
    model: nn.Module,
    strategy: str,
    enabled: bool,
) -> AbstractContextManager[Any]:
    """Return the DDP no-sync context when synchronization is disabled."""

    if strategy == "ddp" and not enabled:
        return model.no_sync()  # type: ignore[no-any-return, attr-defined]
    return nullcontext()


def unwrap_model(model: nn.Module, strategy: str) -> LunarisDenseForCausalLM:
    """Return the architecture module without a DDP wrapper."""

    if strategy == "ddp":
        return model.module  # type: ignore[no-any-return, union-attr]
    return model  # type: ignore[return-value]
