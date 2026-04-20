"""Common MCP request and response types."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class ToolCallRequest(BaseModel):
    """A tool invocation request."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """A normalized tool invocation result."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


ToolHandler = Callable[..., dict[str, Any]]

