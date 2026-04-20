"""Test MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import TestServer
from packages.orchestrator.policy import PolicyEngine


def build_server(
    workspace_root: str | Path,
    policy_path: str | Path = "configs/policies.yaml",
) -> TestServer:
    _policy = PolicyEngine.from_path(policy_path)
    return TestServer(workspace_root)
