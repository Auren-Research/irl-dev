"""Architecture dispatch shared by training, preflight, and checkpoint export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..config import LunarisDenseConfig
from ..model import LunarisDenseForCausalLM


def model_config_from_dict(values: Dict[str, Any]) -> Any:
    """Construct the configuration selected by ``architecture_name``."""

    architecture_name = values.get("architecture_name", "lunaris_dense_irl2")
    if architecture_name == "lunaris_dense_irl2":
        return LunarisDenseConfig.from_dict(values)
    if architecture_name == "lunaris_dense_irl_core":
        from lunaris_dense_irl_core.config import LunarisIRLCoreConfig

        return LunarisIRLCoreConfig.from_dict(values)
    if architecture_name == "lunaris_dense_transformer":
        from lunaris_dense_transformer.config import LunarisTransformerConfig

        return LunarisTransformerConfig.from_dict(values)
    raise ValueError(f"Unsupported architecture_name: {architecture_name!r}")


def build_model(config: Any) -> Any:
    """Instantiate the model implementation matching a validated config."""

    if config.architecture_name == "lunaris_dense_irl2":
        return LunarisDenseForCausalLM(config)
    if config.architecture_name == "lunaris_dense_irl_core":
        from lunaris_dense_irl_core.model import LunarisIRLCoreForCausalLM

        return LunarisIRLCoreForCausalLM(config)
    if config.architecture_name == "lunaris_dense_transformer":
        from lunaris_dense_transformer.model import LunarisTransformerForCausalLM

        return LunarisTransformerForCausalLM(config)
    raise ValueError(f"Unsupported architecture_name: {config.architecture_name!r}")


def load_portable_model(directory: str | Path) -> Any:
    """Load a portable checkpoint after inspecting its architecture field."""

    source = Path(directory)
    import json

    values = json.loads((source / "config.json").read_text(encoding="utf-8"))
    architecture_name = values.get("architecture_name", "lunaris_dense_irl2")
    if architecture_name == "lunaris_dense_irl2":
        return LunarisDenseForCausalLM.from_pretrained(source)
    if architecture_name == "lunaris_dense_irl_core":
        from lunaris_dense_irl_core.model import LunarisIRLCoreForCausalLM

        return LunarisIRLCoreForCausalLM.from_pretrained(source)
    if architecture_name == "lunaris_dense_transformer":
        from lunaris_dense_transformer.model import LunarisTransformerForCausalLM

        return LunarisTransformerForCausalLM.from_pretrained(source)
    raise ValueError(f"Unsupported architecture_name: {architecture_name!r}")
