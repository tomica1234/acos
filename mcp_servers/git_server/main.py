"""Git MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import GitServer, RepoServer


def build_server(workspace_root: str | Path) -> GitServer:
    return GitServer(RepoServer(workspace_root))

