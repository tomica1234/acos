from __future__ import annotations

from pathlib import Path

import pytest

from packages.orchestrator.policy import PolicyEngine
from packages.schemas.approvals import PolicyAction

from tests.conftest import config_dir


def test_workspace_file_read_inside_workspace_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src.py").write_text("print('ok')\n", encoding="utf-8")
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access("src.py", "read")

    assert decision.policy_action == PolicyAction.ALLOW
    assert decision.operation == "workspace_file_read"


def test_workspace_parent_traversal_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access("../secret.txt", "read")

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "workspace_escape"


def test_workspace_absolute_escape_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access(str(outside.resolve()), "read")

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "workspace_escape"


def test_workspace_symlink_escape_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope\n", encoding="utf-8")
    try:
        (workspace / "link.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable in this environment: {exc}")
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access("link.txt", "read")

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "workspace_escape"


def test_workspace_forbidden_path_pattern_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access(".env", "read")

    assert decision.policy_action == PolicyAction.DENY
    assert decision.operation == "secret_file_read"


def test_workspace_env_example_template_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml").build_workspace_policy(workspace)

    decision = policy.classify_path_access(".env.example", "read")

    assert decision.policy_action == PolicyAction.ALLOW
    assert decision.operation == "workspace_file_read"
