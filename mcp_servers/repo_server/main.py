"""Repo MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import RepoServer


def build_server(workspace_root: str | Path) -> RepoServer:
    return RepoServer(workspace_root)

