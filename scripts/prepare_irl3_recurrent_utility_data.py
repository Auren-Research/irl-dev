#!/usr/bin/env python3
"""Prepare non-overlapping train/dev/holdout data for recurrent-utility studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.prepare_ultrafineweb import (
    copy_slice,
    file_sha256,
    token_range,
    tokenizer_metadata,
)


def _range(name: str, start: int, count: int) -> dict[str, int | str]:
    return {
        "name": name,
        "start": int(start),
        "count": int(count),
        "end": int(start + count),
    }


def _validate_ranges(total_tokens: int, ranges: list[dict[str, int | str]]) -> None:
    ordered = sorted(ranges, key=lambda item: int(item["start"]))
    for item in ordered:
        start = int(item["start"])
        end = int(item["end"])
        if start < 0 or end <= start or end > total_tokens:
            raise ValueError(f"Invalid {item['name']} range [{start}, {end})")
    for left, right in zip(ordered, ordered[1:]):
        if int(left["end"]) > int(right["start"]):
            raise ValueError(
                f"Ranges overlap: {left['name']} ends at {left['end']}, "
                f"but {right['name']} starts at {right['start']}"
            )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source.resolve()
    tokenizer_path = args.tokenizer.resolve()
    output_root = args.output_root.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if args.chunk_tokens <= 0:
        raise ValueError("chunk-tokens must be positive")

    source = np.load(source_path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
        raise TypeError("Source must be a one-dimensional integer NumPy array")
    total_tokens = int(source.shape[0])
    ranges = [
        _range("train", args.train_start, args.train_tokens),
        _range("mechanism_dev", args.dev_start, args.dev_tokens),
        _range("confirmation_holdout", args.holdout_start, args.holdout_tokens),
    ]
    _validate_ranges(total_tokens, ranges)

    minimum_token_id, maximum_token_id = token_range(
        source,
        chunk_tokens=args.chunk_tokens,
    )
    if minimum_token_id < 0 or maximum_token_id >= args.vocab_size:
        raise ValueError(
            f"Token IDs [{minimum_token_id}, {maximum_token_id}] are outside "
            f"vocab_size={args.vocab_size}"
        )
    tokenizer = tokenizer_metadata(tokenizer_path, args.vocab_size)

    paths = {
        "train": output_root / "train" / "shard_000.npy",
        "mechanism_dev": output_root / "mechanism_dev" / "shard_000.npy",
        "confirmation_holdout": (
            output_root / "confirmation_holdout" / "shard_000.npy"
        ),
    }
    for item in ranges:
        name = str(item["name"])
        copy_slice(
            source,
            paths[name],
            start=int(item["start"]),
            count=int(item["count"]),
            chunk_tokens=args.chunk_tokens,
        )

    manifest = {
        "format_version": 1,
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "file_size_bytes": source_path.stat().st_size,
            "dtype": str(source.dtype),
            "total_tokens": total_tokens,
            "minimum_token_id": minimum_token_id,
            "maximum_token_id": maximum_token_id,
        },
        "tokenizer": tokenizer,
        "splits": {
            str(item["name"]): {
                "path": str(paths[str(item["name"])]),
                "start": int(item["start"]),
                "tokens": int(item["count"]),
                "end": int(item["end"]),
            }
            for item in ranges
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/irl3_recurrent_utility"),
    )
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-tokens", type=int, default=900_000_000)
    parser.add_argument("--dev-start", type=int, default=970_000_000)
    parser.add_argument("--dev-tokens", type=int, default=10_000_000)
    parser.add_argument("--holdout-start", type=int, default=990_000_000)
    parser.add_argument("--holdout-tokens", type=int, default=10_000_000)
    parser.add_argument("--vocab-size", type=int, default=65_536)
    parser.add_argument("--chunk-tokens", type=int, default=4_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
