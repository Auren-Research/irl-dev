#!/usr/bin/env python3
"""Validate and split one tokenized Ultra-FineWeb shard for qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a small metadata file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_metadata(path: Path, expected_vocab_size: int) -> dict[str, Any]:
    """Validate the tokenizer vocabulary and return reproducibility metadata."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError(f"Tokenizer does not contain model.vocab: {path}")

    vocabulary = model["vocab"]
    if len(vocabulary) != expected_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary mismatch: "
            f"expected {expected_vocab_size:,}, found {len(vocabulary):,}"
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
            "Tokenizer special-token IDs do not match the model profile: "
            f"{actual_special_tokens}"
        )

    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "vocab_size": len(vocabulary),
        "special_tokens": actual_special_tokens,
    }


def token_range(
    tokens: np.ndarray,
    *,
    chunk_tokens: int,
) -> tuple[int, int]:
    """Scan the shard in bounded chunks and return its minimum and maximum IDs."""

    minimum: int | None = None
    maximum: int | None = None
    for start in range(0, int(tokens.shape[0]), chunk_tokens):
        chunk = tokens[start : start + chunk_tokens]
        chunk_minimum = int(chunk.min())
        chunk_maximum = int(chunk.max())
        minimum = (
            chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        )
        maximum = (
            chunk_maximum if maximum is None else max(maximum, chunk_maximum)
        )
    if minimum is None or maximum is None:
        raise ValueError("The source shard is empty")
    return minimum, maximum


def validate_existing(
    path: Path,
    *,
    expected_tokens: int,
    expected_dtype: np.dtype[Any],
) -> bool:
    """Return whether an existing output is a complete compatible NumPy shard."""

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
        raise ValueError(f"Existing output is incomplete or incompatible: {path}")
    return True


def copy_slice(
    source: np.ndarray,
    destination: Path,
    *,
    start: int,
    count: int,
    chunk_tokens: int,
) -> None:
    """Copy one contiguous source range into an atomic standalone NPY shard."""

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
    """Prepare deterministic non-overlapping train and validation shards."""

    source_path = args.source.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output_root = args.output_root.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if args.train_tokens <= 0 or args.validation_tokens <= 0:
        raise ValueError("Train and validation token counts must be positive")
    if args.chunk_tokens <= 0:
        raise ValueError("chunk-tokens must be positive")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 1:
        raise ValueError("The source token shard must be one-dimensional")
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError("The source token shard must have an integer dtype")

    total_tokens = int(source.shape[0])
    validation_start = total_tokens - args.validation_tokens
    if args.train_tokens > validation_start:
        raise ValueError(
            "Train and validation ranges overlap: "
            f"train={args.train_tokens:,}, validation_start={validation_start:,}"
        )

    minimum_token_id, maximum_token_id = token_range(
        source,
        chunk_tokens=args.chunk_tokens,
    )
    if minimum_token_id < 0 or maximum_token_id >= args.vocab_size:
        raise ValueError(
            "Token IDs are outside the model vocabulary: "
            f"range=[{minimum_token_id}, {maximum_token_id}], "
            f"vocab_size={args.vocab_size}"
        )

    tokenizer = tokenizer_metadata(tokenizer_path, args.vocab_size)
    train_path = output_root / "train" / "shard_000.npy"
    validation_path = output_root / "validation" / "shard_000.npy"
    copy_slice(
        source,
        train_path,
        start=0,
        count=args.train_tokens,
        chunk_tokens=args.chunk_tokens,
    )
    copy_slice(
        source,
        validation_path,
        start=validation_start,
        count=args.validation_tokens,
        chunk_tokens=args.chunk_tokens,
    )

    manifest = {
        "format_version": 1,
        "source": {
            "path": str(source_path),
            "file_size_bytes": source_path.stat().st_size,
            "dtype": str(source.dtype),
            "total_tokens": total_tokens,
            "minimum_token_id": minimum_token_id,
            "maximum_token_id": maximum_token_id,
        },
        "tokenizer": tokenizer,
        "train": {
            "path": str(train_path),
            "start": 0,
            "tokens": args.train_tokens,
        },
        "validation": {
            "path": str(validation_path),
            "start": validation_start,
            "tokens": args.validation_tokens,
        },
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
        description="Prepare one Ultra-FineWeb NPY shard for MI300X qualification."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/qualification"),
    )
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--validation-tokens", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--chunk-tokens", type=int, default=4_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())

