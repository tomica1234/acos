"""Test MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import TestServer


def build_server(workspace_root: str | Path) -> TestServer:
    return TestServer(workspace_root)

