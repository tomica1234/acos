"""Utility helpers for response and tool schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

TOOL_PARAMETER_SCHEMAS: dict[str, dict[str, Any]] = {
    "repo_server.read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path to read."},
            "max_chars": {"type": "integer", "description": "Optional maximum characters to return."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "repo_server.search_text": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Literal text to search for in workspace files."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "repo_server.apply_patch": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative file path to create or update."},
            "content": {"type": "string", "description": "Full file contents to write."},
            "operation": {
                "type": "string",
                "enum": ["create", "update"],
                "description": "Whether to create a new file or replace an existing file.",
            },
        },
        "required": ["path", "content", "operation"],
        "additionalProperties": False,
    },
    "test_server.install_package": {
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "Python package spec to install into the active virtualenv, e.g. django or django>=5,<6.",
            },
            "timeout_seconds": {"type": "integer", "description": "Optional installation timeout in seconds."},
        },
        "required": ["package"],
        "additionalProperties": False,
    },
    "test_server.run_command": {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Argument vector executed directly inside the workspace without a shell.",
            },
            "timeout_seconds": {"type": "integer", "description": "Optional timeout in seconds."},
            "mode": {
                "type": "string",
                "enum": ["oneshot", "server"],
                "description": "Use server mode to wait for a local HTTP service to become ready.",
            },
            "port": {"type": "integer", "description": "Optional explicit server port."},
            "http_path": {
                "type": "string",
                "description": "HTTP path to probe once the server is listening, defaults to /.",
            },
            "http_checks": {
                "type": "array",
                "description": "Optional browser-like HTTP checks to run after the local server is ready.",
                "items": {"type": "object"},
            },
        },
        "required": ["argv"],
        "additionalProperties": False,
    },
}


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
                    "parameters": TOOL_PARAMETER_SCHEMAS.get(
                        tool_name,
                        {"type": "object", "properties": {}, "additionalProperties": True},
                    ),
                },
            }
        )
    return manifest
