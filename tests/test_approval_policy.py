from __future__ import annotations

from pathlib import Path

from packages.orchestrator.policy import PolicyEngine
from packages.schemas.approvals import PolicyAction

from tests.conftest import config_dir


def test_workspace_read_is_auto_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_tool_call(
        role="architect",
        tool_name="repo_server.read_file",
        arguments={"path": "feature.py"},
        workspace_root=workspace,
    )

    assert decision.policy_action == PolicyAction.ALLOW
    assert decision.operation == "workspace_file_read"


def test_workspace_escape_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_tool_call(
        role="architect",
        tool_name="repo_server.read_file",
        arguments={"path": "../outside.txt"},
        workspace_root=workspace,
    )

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "workspace_escape"


def test_secret_file_read_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_tool_call(
        role="architect",
        tool_name="repo_server.read_file",
        arguments={"path": ".env"},
        workspace_root=workspace,
    )

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "secret_file_read"


def test_mass_delete_requires_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = workspace_policy.classify_path_access("feature.py", "delete", delete_count=25)

    assert decision.policy_action == PolicyAction.REQUIRE_APPROVAL
    assert decision.operation == "mass_delete"


def test_production_deploy_requires_approval() -> None:
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_named_operation("production_deploy")

    assert decision.policy_action == PolicyAction.REQUIRE_APPROVAL


def test_arbitrary_shell_is_denied() -> None:
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_named_operation("arbitrary_shell")

    assert decision.policy_action == PolicyAction.DENY

