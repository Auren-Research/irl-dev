"""Public API for the parameter-matched conventional Transformer."""

from lunaris_dense_irl2.cache import KVCache
from lunaris_dense_irl2.diagnostics import LunarisCausalLMOutput

from .config import LunarisTransformerConfig
from .model import LunarisTransformerForCausalLM

__all__ = [
    "KVCache",
    "LunarisCausalLMOutput",
    "LunarisTransformerConfig",
    "LunarisTransformerForCausalLM",
]

__version__ = "0.1.0"
