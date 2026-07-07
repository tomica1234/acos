"""LLM abstractions for ACOS."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "LLMClient": "packages.llm.client",
    "ModelRegistry": "packages.llm.registry",
    "ModelRouter": "packages.llm.routing",
    "RoutingContext": "packages.llm.routing",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
