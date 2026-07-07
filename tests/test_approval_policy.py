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


def test_allowlisted_test_server_commands_are_not_classified_as_shell(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    for command_name in (
        "",
        "auto",
        "django-test",
        "prepare-runtime-auto",
        "runtime-smoke-auto",
        "django-wsgi-check",
        "pytest",
    ):
        decision = policy.classify_tool_call(
            role="runner",
            tool_name="test_server.run_test",
            arguments={"command_name": command_name},
            workspace_root=workspace,
        )

        assert decision.policy_action == PolicyAction.ALLOW
        assert decision.operation == "test_run_allowlisted"


def test_dependency_install_requires_job_opt_in(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    blocked = policy.classify_tool_call(
        role="fixer",
        tool_name="test_server.install_package",
        arguments={"package": "django"},
        workspace_root=workspace,
    )
    allowed = policy.classify_tool_call(
        role="fixer",
        tool_name="test_server.install_package",
        arguments={"package": "django"},
        workspace_root=workspace,
        job_metadata={"constraints": {"allow_dependency_addition": True}},
    )

    assert blocked.policy_action == PolicyAction.REQUIRE_APPROVAL
    assert blocked.operation == "package_install_non_allowlisted"
    assert allowed.policy_action == PolicyAction.ALLOW_AND_AUDIT
    assert allowed.operation == "package_install_allowlisted_virtualenv"


def test_runtime_command_is_allowed_for_workspace_execution(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_tool_call(
        role="runner",
        tool_name="test_server.run_command",
        arguments={"argv": ["python", "manage.py", "migrate"], "mode": "oneshot"},
        workspace_root=workspace,
    )

    assert decision.policy_action == PolicyAction.ALLOW_AND_AUDIT
    assert decision.operation == "workspace_runtime_exec"


def test_runtime_command_flags_destructive_db_migration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    decision = policy.classify_tool_call(
        role="runner",
        tool_name="test_server.run_command",
        arguments={"argv": ["python", "manage.py", "flush"], "mode": "oneshot"},
        workspace_root=workspace,
    )

    assert decision.policy_action == PolicyAction.REQUIRE_APPROVAL
    assert decision.operation == "destructive_db_migration"
