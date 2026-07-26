"""Scalable memory-mapped token data loading without global permutations."""

from __future__ import annotations

import bisect
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import DataConfig


class MemoryMappedTokenDataset(Dataset[torch.Tensor]):
    """Expose fixed-length samples from independent one-dimensional NPY shards."""

    def __init__(
        self,
        directory: str | Path,
        *,
        sequence_length: int,
        shard_glob: str = "*.npy",
        mmap_cache_size: int = 8,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.sequence_length = int(sequence_length)
        self.mmap_cache_size = int(mmap_cache_size)
        self.shards = sorted(self.directory.glob(shard_glob))
        if not self.shards:
            raise ValueError(
                f"No token shards matching {shard_glob!r} in {self.directory}"
            )

        self.shard_lengths: List[int] = []
        self.sample_counts: List[int] = []
        for path in self.shards:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1:
                raise ValueError(f"Token shard must be one-dimensional: {path}")
            if not np.issubdtype(array.dtype, np.integer):
                raise TypeError(f"Token shard must use an integer dtype: {path}")
            length = int(array.shape[0])
            self.shard_lengths.append(length)
            self.sample_counts.append(length // self.sequence_length)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

        self.cumulative_samples = np.cumsum(self.sample_counts).tolist()
        self.total_samples = int(sum(self.sample_counts))
        self.total_tokens = int(sum(self.shard_lengths))
        if self.total_samples <= 0:
            raise ValueError(
                "The dataset has no complete samples at the configured sequence length"
            )

        self._mmap_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of complete, non-overlapping samples."""

        return self.total_samples

    def _open_shard(self, shard_index: int) -> np.ndarray:
        """Open one shard lazily and retain a small per-worker LRU cache."""

        cached = self._mmap_cache.pop(shard_index, None)
        if cached is not None:
            self._mmap_cache[shard_index] = cached
            return cached

        array = np.load(
            self.shards[shard_index],
            mmap_mode="r",
            allow_pickle=False,
        )
        self._mmap_cache[shard_index] = array
        while len(self._mmap_cache) > self.mmap_cache_size:
            _, evicted = self._mmap_cache.popitem(last=False)
            mmap = getattr(evicted, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return array

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read one sample and copy it out of the memory map."""

        if index < 0:
            index += self.total_samples
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)

        shard_index = bisect.bisect_right(self.cumulative_samples, index)
        previous = 0 if shard_index == 0 else self.cumulative_samples[shard_index - 1]
        local_sample = index - previous
        start = local_sample * self.sequence_length
        end = start + self.sequence_length
        tokens = np.array(
            self._open_shard(shard_index)[start:end],
            dtype=np.int64,
            copy=True,
        )
        return torch.from_numpy(tokens)


def _affine_parameters(size: int, seed: int, epoch: int) -> tuple[int, int]:
    """Create an O(1)-memory permutation over ``range(size)``."""

    if size <= 1:
        return 1, 0
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1_000_003 * int(epoch))
    multiplier = int(torch.randint(1, size, (), generator=generator).item())
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = int(torch.randint(0, size, (), generator=generator).item())
    return multiplier, offset


class DistributedAffineBatchSampler(Sampler[List[int]]):
    """Shard a topology-independent deterministic stream across ranks."""

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool,
        infinite: bool,
        start_global_sample: int = 0,
        max_batches: Optional[int] = None,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError("rank must be in [0, world_size)")

        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.infinite = bool(infinite)
        self.start_global_sample = int(start_global_sample)
        if self.start_global_sample < 0:
            raise ValueError("start_global_sample cannot be negative")
        self.max_batches = max_batches

        global_batch_size = self.batch_size * self.world_size
        if self.dataset_size < global_batch_size:
            raise ValueError(
                "Dataset is smaller than one complete distributed micro-batch"
            )
        self.global_batch_size = global_batch_size

    def _stream_index(self, global_position: int) -> int:
        """Map an absolute stream position to one deterministic sample."""

        epoch, position = divmod(global_position, self.dataset_size)
        if not self.shuffle:
            return position
        multiplier, offset = _affine_parameters(
            self.dataset_size,
            self.seed,
            epoch,
        )
        return (multiplier * position + offset) % self.dataset_size

    def __iter__(self) -> Iterator[List[int]]:
        """Yield rank-local batches from a world-size-independent cursor."""

        emitted = 0
        global_cursor = self.start_global_sample
        finite_batches = max(
            0,
            (self.dataset_size - global_cursor) // self.global_batch_size,
        )
        while self.infinite or emitted < finite_batches:
            if self.max_batches is not None and emitted >= self.max_batches:
                break
            rank_start = global_cursor + self.rank * self.batch_size
            yield [
                self._stream_index(rank_start + offset)
                for offset in range(self.batch_size)
            ]
            emitted += 1
            global_cursor += self.global_batch_size

    def __len__(self) -> int:
        """Return a finite reporting length for DataLoader."""

        if self.max_batches is not None:
            return self.max_batches
        return max(
            0,
            (self.dataset_size - self.start_global_sample)
            // self.global_batch_size,
        )


def build_dataloader(
    dataset: MemoryMappedTokenDataset,
    data_config: DataConfig,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    training: bool,
    start_global_sample: int = 0,
    max_batches: Optional[int] = None,
) -> DataLoader[torch.Tensor]:
    """Build a worker-safe loader with deterministic distributed sampling."""

    sampler = DistributedAffineBatchSampler(
        len(dataset),
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=data_config.seed,
        shuffle=data_config.shuffle if training else False,
        infinite=training,
        start_global_sample=start_global_sample,
        max_batches=max_batches,
    )
    kwargs: Dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": data_config.num_workers,
        "pin_memory": data_config.pin_memory,
    }
    if data_config.num_workers > 0:
        kwargs["persistent_workers"] = data_config.persistent_workers
        kwargs["prefetch_factor"] = data_config.prefetch_factor
    return DataLoader(dataset, **kwargs)


class DevicePrefetcher:
    """Overlap host-to-device copies with compute on CUDA or ROCm."""

    def __init__(
        self,
        loader: DataLoader[torch.Tensor],
        device: torch.device,
    ) -> None:
        self.iterator = iter(loader)
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.next_batch: Optional[torch.Tensor] = None
        self._preload()

    def _preload(self) -> None:
        """Stage the next batch on a dedicated accelerator stream."""

        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return

        if self.stream is None:
            self.next_batch = batch.to(self.device, dtype=torch.long)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            )

    def next(self) -> torch.Tensor:
        """Return the staged batch and immediately begin the following copy."""

        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        if self.stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self.stream)
            batch.record_stream(current)
        self._preload()
        return batch
