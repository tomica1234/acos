"""Local router for MCP-style tool invocations."""

from __future__ import annotations

from typing import Any

from packages.mcp_client.types import ToolCallResult, ToolHandler


class MCPRouter:
    """Dispatch tool invocations to registered handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, tool_name: str, handler: ToolHandler) -> None:
        self._handlers[tool_name] = handler

    def call(self, tool_name: str, **kwargs: Any) -> ToolCallResult:
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolCallResult(ok=False, error=f"Unknown tool: {tool_name}")
        try:
            return ToolCallResult(ok=True, data=handler(**kwargs))
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ToolCallResult(ok=False, error=str(exc))

    def available_tools(self) -> list[str]:
        return sorted(self._handlers)

