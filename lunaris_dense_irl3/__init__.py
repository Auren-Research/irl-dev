"""Public API for Lunaris Dense IRL-3."""

from .cache import IRL3Cache
from .config import LunarisIRL3Config
from .diagnostics import (
    IRL3Diagnostics,
    IRL3State,
    LunarisIRL3CausalLMOutput,
)
from .model import LunarisIRL3ForCausalLM

__all__ = [
    "IRL3Cache",
    "IRL3Diagnostics",
    "IRL3State",
    "LunarisIRL3CausalLMOutput",
    "LunarisIRL3Config",
    "LunarisIRL3ForCausalLM",
]

__version__ = "0.1.0"
