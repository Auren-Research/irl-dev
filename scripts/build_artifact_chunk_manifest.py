#!/usr/bin/env python3
"""Split one artifact archive into independently verifiable transfer chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def build_manifest(
    archive: Path,
    output_directory: Path,
    *,
    chunk_bytes: int,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_hash = hashlib.sha256()
    chunks: list[dict[str, object]] = []
    with archive.open("rb") as source:
        index = 0
        while data := source.read(chunk_bytes):
            archive_hash.update(data)
            name = f"chunk-{index:05d}.part"
            path = output_directory / name
            path.write_bytes(data)
            chunks.append(
                {
                    "name": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            index += 1
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash.hexdigest(),
        "chunk_bytes": chunk_bytes,
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=1_048_576)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(
        args.archive,
        args.output_directory,
        chunk_bytes=args.chunk_bytes,
    )
    manifest_path = args.manifest or args.output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
