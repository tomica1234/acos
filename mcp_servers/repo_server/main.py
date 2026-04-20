"""Repo MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import RepoServer
from packages.orchestrator.policy import PolicyEngine


def build_server(
    workspace_root: str | Path,
    policy_path: str | Path = "configs/policies.yaml",
) -> RepoServer:
    policy = PolicyEngine.from_path(policy_path)
    return RepoServer(
        workspace_root,
        workspace_policy=policy.build_workspace_policy(workspace_root),
    )
