"""Notification MCP server skeleton."""

from packages.mcp_client.fake import NotifyServer


def build_server() -> NotifyServer:
    return NotifyServer()

