from __future__ import annotations

from pathlib import Path

import pytest

from packages.agents.runner import AgentRunner
from packages.llm.routing import RoutingContext
from packages.mcp_client.fake import FakeMCPEnvironment, RepoServer, TestServer
from packages.mcp_client.router import MCPRouter
from packages.memory.redaction import redact_text
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import FilePatch, ImplementationResult
from packages.schemas.context import ContextPacket
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import ImplementationStatus

from tests.conftest import attach_mock_adapter, config_dir, load_registry


def _packet(role: str, repo_path: str = ".") -> ContextPacket:
    return ContextPacket(
        job_id=f"job-{role}",
        role=role,
        objective=f"Run {role}",
        repo_path=repo_path,
        request_text="Use secret=sk-this-should-be-redacted",
        token_budget=4096,
        model_context_budget=4096,
        selected_model_hint="auto",
    )


def test_redaction_covers_common_secret_formats() -> None:
    text = "\n".join(
        [
            "api_key=sk-abcdefghijklmnopqrstuvwxyz",
            "Authorization: Bearer token-value-1234567890",
            "aws_access_key_id=AKIAABCDEFGHIJKLMNOP",
            "password='super-secret'",
            "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----",
        ]
    )

    redacted = redact_text(text)

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "super-secret" not in redacted
    assert "PRIVATE KEY" not in redacted
    assert "[REDACTED]" in redacted


def test_repo_server_blocks_secret_paths_and_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "inside.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside_file)
    repo = RepoServer(workspace)

    assert repo.repo_tree()["files"] == ["inside.py"]

    with pytest.raises(ValueError):
        repo.read_file(".env.local")

    with pytest.raises(ValueError):
        repo.read_file(".git/config")

    with pytest.raises(ValueError):
        repo.apply_patch("config/secret_key", "value = 1\n", operation="create")

    with pytest.raises(ValueError):
        repo.read_file("link.txt")


def test_test_server_rejects_invalid_timeout_and_unknown_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = TestServer(workspace)

    with pytest.raises(ValueError):
        server.run_test(command_name="pytest", timeout_seconds=0)

    with pytest.raises(ValueError):
        server.run_test(command_name="python -c bad", timeout_seconds=30)


def test_policy_blocks_test_and_dependency_patches() -> None:
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("implementer", "tests/test_feature.py")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("fixer", "tests/test_feature.py")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("implementer", "pyproject.toml")

    with pytest.raises(PermissionError):
        policy.assert_patch_target_allowed("test_writer", ".env.local")

    policy.assert_patch_target_allowed("test_writer", "tests/test_feature.py")


def test_agent_runner_rejects_tool_override_outside_agent_config(tmp_path: Path) -> None:
    registry = load_registry()
    registry.agents["implementer"].allowed_tools = ["repo_server.read_file"]
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="unused",
                patches=[],
            ).model_dump()
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = AgentRunner(
        registry=registry,
        mcp_router=environment.build_router(),
        policy_engine=policy,
        audit_recorder=AuditRecorder(),
    )

    with pytest.raises(PermissionError):
        runner.run(
            role="implementer",
            response_model=ImplementationResult,
            context_packet=_packet("implementer", repo_path=str(tmp_path)),
            routing_context=RoutingContext(role="implementer"),
            allowed_tools=["repo_server.search_text"],
        )


def test_agent_runner_blocks_forbidden_patch_tool_call(tmp_path: Path) -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "implementer": {
                "content": "",
                "tool_calls": [
                    {
                        "name": "repo_server.apply_patch",
                        "arguments": {
                            "path": "tests/test_feature.py",
                            "content": "def test_bad() -> None:\n    assert True\n",
                            "operation": "create",
                        },
                    }
                ],
            }
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = AgentRunner(
        registry=registry,
        mcp_router=environment.build_router(),
        policy_engine=policy,
        audit_recorder=AuditRecorder(),
    )

    with pytest.raises(PermissionError):
        runner.run(
            role="implementer",
            response_model=ImplementationResult,
            context_packet=_packet("implementer", repo_path=str(tmp_path)),
            routing_context=RoutingContext(role="implementer"),
        )


def test_job_runner_blocks_dependency_manifest_patch(tmp_path: Path) -> None:
    registry = load_registry()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = JobRecord(
        job_id="job-1",
        spec=JobSpec(
            request_text="test",
            repo_path=str(tmp_path),
            target_branch="acos/security-check",
        ),
    )

    with pytest.raises(PermissionError):
        runner._apply_patches(
            record,
            "implementer",
            [FilePatch(path="pyproject.toml", content="[project]\nname='bad'\n")],
        )


def test_memory_and_notify_servers_redact_secret_payloads(tmp_path: Path) -> None:
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / "memory.sqlite3",
    )

    environment.memory_server.write_memory(
        uri="memory://job-1/secret",
        content="token=sk-abcdefghijklmnopqrstuvwxyz",
    )
    entries = environment.memory_server.read_memory(uri="memory://job-1")
    serialized = str(entries)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in serialized
    assert "[REDACTED]" in serialized

    payload = environment.notify_server.send_notification(
        body="Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in str(payload)
    assert "[REDACTED]" in str(payload)


def test_mcp_router_redacts_tool_errors() -> None:
    router = MCPRouter()

    def bad_handler() -> dict[str, str]:
        raise RuntimeError("api_key=sk-abcdefghijklmnopqrstuvwxyz")

    router.register("demo.bad", bad_handler)
    result = router.call("demo.bad")

    assert not result.ok
    assert result.error is not None
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in result.error
    assert "[REDACTED]" in result.error
