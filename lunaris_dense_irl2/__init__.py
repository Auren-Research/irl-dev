"""Public package API for Lunaris Dense IRL-2."""

from .cache import KVCache
from .config import LunarisDenseConfig
from .diagnostics import IRLDiagnostics, LunarisCausalLMOutput
from .model import LunarisDenseForCausalLM

__all__ = [
    "IRLDiagnostics",
    "KVCache",
    "LunarisCausalLMOutput",
    "LunarisDenseConfig",
    "LunarisDenseForCausalLM",
]

__version__ = "0.4.0"
