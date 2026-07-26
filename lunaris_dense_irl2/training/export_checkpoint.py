"""Consolidate a distributed training checkpoint for inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.distributed.checkpoint as dcp

from ..config import LunarisDenseConfig
from ..model import LunarisDenseForCausalLM


def export_checkpoint(
    checkpoint: str | Path,
    output_directory: str | Path,
) -> Path:
    """Load a DCP checkpoint on CPU and write a portable model package."""

    checkpoint_path = Path(checkpoint)
    if not (checkpoint_path / "_SUCCESS").exists():
        raise FileNotFoundError(
            f"Checkpoint is incomplete or missing: {checkpoint_path}"
        )
    metadata = json.loads(
        (checkpoint_path / "metadata.json").read_text(encoding="utf-8")
    )
    model_config = LunarisDenseConfig.from_dict(metadata["train_config"]["model"])
    model = LunarisDenseForCausalLM(model_config)
    state = {"model": model.state_dict()}
    dcp.load(
        state,
        checkpoint_id=checkpoint_path / "dcp",
        no_dist=True,
    )
    model.load_state_dict(state["model"], strict=True)

    destination = Path(output_directory)
    model.save_pretrained(destination)
    (destination / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Parse command-line arguments and export one checkpoint."""

    parser = argparse.ArgumentParser(
        description="Export a Lunaris DCP checkpoint for inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = export_checkpoint(args.checkpoint, args.output)
    print(f"Exported portable checkpoint to {destination}")


if __name__ == "__main__":
    main()
