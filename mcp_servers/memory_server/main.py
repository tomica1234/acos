"""Memory MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import MemoryServer
from packages.memory.store import SQLiteMemoryStore


def build_server(db_path: str | Path) -> MemoryServer:
    return MemoryServer(SQLiteMemoryStore(db_path))

