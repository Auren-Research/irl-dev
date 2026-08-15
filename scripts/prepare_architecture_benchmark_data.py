#!/usr/bin/env python3
"""Prepare deterministic train, dev, and confirmation token ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a shard into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_metadata(path: Path, expected_vocab_size: int) -> dict[str, Any]:
    """Validate the benchmark tokenizer and return reproducibility metadata."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError(f"Tokenizer does not contain model.vocab: {path}")
    vocabulary = model["vocab"]
    if len(vocabulary) != expected_vocab_size:
        raise ValueError(
            f"Expected {expected_vocab_size:,} tokenizer entries, "
            f"found {len(vocabulary):,}"
        )
    expected_special_tokens = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    actual_special_tokens = {
        token: vocabulary.get(token) for token in expected_special_tokens
    }
    if actual_special_tokens != expected_special_tokens:
        raise ValueError(
            "Tokenizer special-token IDs do not match the model profiles: "
            f"{actual_special_tokens}"
        )
    return {
        "source_path": str(path.resolve()),
        "sha256": file_sha256(path),
        "vocab_size": len(vocabulary),
        "special_tokens": actual_special_tokens,
    }


def token_range(tokens: np.ndarray, chunk_tokens: int) -> tuple[int, int]:
    """Scan integer IDs in bounded chunks."""

    minimum: int | None = None
    maximum: int | None = None
    for start in range(0, int(tokens.shape[0]), chunk_tokens):
        chunk = tokens[start : start + chunk_tokens]
        chunk_minimum = int(chunk.min())
        chunk_maximum = int(chunk.max())
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
    if minimum is None or maximum is None:
        raise ValueError("The source token shard is empty")
    return minimum, maximum


def validate_existing(
    path: Path,
    *,
    expected_tokens: int,
    expected_dtype: np.dtype[Any],
) -> bool:
    """Return whether an existing split is complete and compatible."""

    if not path.exists():
        return False
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    valid = (
        array.ndim == 1
        and int(array.shape[0]) == expected_tokens
        and array.dtype == expected_dtype
    )
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()
    if not valid:
        raise ValueError(f"Existing split is incomplete or incompatible: {path}")
    return True


def copy_slice(
    source: np.ndarray,
    destination: Path,
    *,
    start: int,
    count: int,
    chunk_tokens: int,
) -> None:
    """Copy a disjoint contiguous range into an atomic NPY shard."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(
        destination,
        expected_tokens=count,
        expected_dtype=source.dtype,
    ):
        print(f"[data] reusing {destination} ({count:,} tokens)", flush=True)
        return

    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=source.dtype,
        shape=(count,),
    )
    for offset in range(0, count, chunk_tokens):
        length = min(chunk_tokens, count - offset)
        output[offset : offset + length] = source[
            start + offset : start + offset + length
        ]
    output.flush()
    del output
    temporary.replace(destination)
    print(f"[data] wrote {destination} ({count:,} tokens)", flush=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Create the frozen non-overlapping architecture-benchmark split."""

    source_path = args.source.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output_root = args.output_root.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    for name in ("train_tokens", "dev_tokens", "holdout_tokens", "chunk_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 1:
        raise ValueError("The source token shard must be one-dimensional")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError("The source token shard must have an integer dtype")

    ranges = {
        "train": (0, args.train_tokens),
        "dev": (args.train_tokens, args.dev_tokens),
        "holdout": (
            args.train_tokens + args.dev_tokens,
            args.holdout_tokens,
        ),
    }
    required_tokens = sum(count for _, count in ranges.values())
    if required_tokens > int(source.shape[0]):
        raise ValueError(
            f"Source has {int(source.shape[0]):,} tokens but the frozen splits "
            f"require {required_tokens:,}"
        )

    minimum_token_id, maximum_token_id = token_range(source, args.chunk_tokens)
    if minimum_token_id < 0 or maximum_token_id >= args.vocab_size:
        raise ValueError(
            "Token IDs fall outside the benchmark vocabulary: "
            f"[{minimum_token_id}, {maximum_token_id}]"
        )

    split_metadata: dict[str, Any] = {}
    for split, (start, count) in ranges.items():
        path = output_root / split / "shard_000.npy"
        copy_slice(
            source,
            path,
            start=start,
            count=count,
            chunk_tokens=args.chunk_tokens,
        )
        split_metadata[split] = {
            "path": str(path),
            "start": start,
            "end_exclusive": start + count,
            "tokens": count,
            "complete_sequences": count // args.sequence_length,
            "unused_tail_tokens": count % args.sequence_length,
            "sha256": file_sha256(path),
        }

    tokenizer = tokenizer_metadata(tokenizer_path, args.vocab_size)
    frozen_tokenizer = output_root / "lunaris-tokenizer.json"
    shutil.copyfile(tokenizer_path, frozen_tokenizer)
    tokenizer["frozen_path"] = str(frozen_tokenizer)

    manifest = {
        "schema_version": 1,
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "dtype": str(source.dtype),
            "tokens": int(source.shape[0]),
            "minimum_token_id": minimum_token_id,
            "maximum_token_id": maximum_token_id,
            "sha256": (
                None if args.skip_source_sha256 else file_sha256(source_path)
            ),
        },
        "tokenizer": tokenizer,
        "sequence_length": args.sequence_length,
        "splits": split_metadata,
        "ranges_overlap": False,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mmap = getattr(source, "_mmap", None)
    if mmap is not None:
        mmap.close()
    print(f"[data] manifest: {manifest_path}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare the controlled 135M architecture benchmark data."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/architecture_benchmark"),
    )
    parser.add_argument("--train-tokens", type=int, default=500_000_000)
    parser.add_argument("--dev-tokens", type=int, default=4_194_304)
    parser.add_argument("--holdout-tokens", type=int, default=4_194_304)
    parser.add_argument("--sequence-length", type=int, default=4_096)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--chunk-tokens", type=int, default=8_000_000)
    parser.add_argument(
        "--skip-source-sha256",
        action="store_true",
        help="Skip only the expensive full-source digest; split digests remain",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
