"""Public API for the simplified Lunaris Dense IRL Core."""

from lunaris_dense_irl2.cache import KVCache

from .config import LunarisIRLCoreConfig
from .core import IRLCore
from .diagnostics import IRLCoreDiagnostics, LunarisIRLCoreCausalLMOutput
from .model import LunarisIRLCoreForCausalLM

__all__ = [
    "IRLCore",
    "IRLCoreDiagnostics",
    "KVCache",
    "LunarisIRLCoreCausalLMOutput",
    "LunarisIRLCoreConfig",
    "LunarisIRLCoreForCausalLM",
]

__version__ = "0.1.0"
