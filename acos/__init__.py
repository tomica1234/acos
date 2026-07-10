"""Top-level ACOS package marker.

The runnable ACOS MVP is organized across `apps/`, `packages/`, and
`mcp_servers/`. This package exists so `python -m compileall acos` succeeds and
external tooling can import `acos` as the repository root package.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
