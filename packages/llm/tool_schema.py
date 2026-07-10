"""Utility helpers for response and tool schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def build_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for a Pydantic response model."""
    return model.model_json_schema()


def build_tool_manifest(tool_names: list[str]) -> list[dict[str, Any]]:
    """Create a minimal OpenAI-compatible tool manifest."""
    manifest: list[dict[str, Any]] = []
    for tool_name in tool_names:
        manifest.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"MCP tool {tool_name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    return manifest

