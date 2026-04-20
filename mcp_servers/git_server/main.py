"""Git MCP server skeleton."""

from __future__ import annotations

from pathlib import Path

from packages.mcp_client.fake import GitServer, RepoServer
from packages.orchestrator.policy import PolicyEngine


def build_server(
    workspace_root: str | Path,
    policy_path: str | Path = "configs/policies.yaml",
) -> GitServer:
    policy = PolicyEngine.from_path(policy_path)
    repo_server = RepoServer(
        workspace_root,
        workspace_policy=policy.build_workspace_policy(workspace_root),
    )
    return GitServer(repo_server)
