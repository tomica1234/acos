from datetime import datetime, timezone
from pathlib import Path

import pytest

from packages.llm.errors import AdapterError, StructuredOutputError
from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.completion_verifier import DefinitionOfDoneVerifier
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FixResult,
    ImplementationResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    RuntimePlan,
    SecurityReviewResult,
    SummaryResult,
    TestRunResult,
    TestWriterResult as TestWriterOutput,
)
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    JobStatus,
    ModelCallRecord,
    ModelCallStatus,
    ReviewDecision,
    Severity,
    TestWriterStatus as WriterStatus,
)
from packages.schemas.runtime import RuntimeHttpCheck
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir


def assert_recovery_plan(record, *, status: JobStatus, strategy: str) -> None:
    assert record.status == status
    plan = record.runtime_state["recovery_plan"]
    assert plan["strategy"] == strategy
    assert plan["status"] == "completed"


def assert_recoverable_error(
    record,
    expected: str | None = None,
    *,
    startswith: str | None = None,
    contains: str | None = None,
) -> str:
    assert record.last_error is None
    error = record.runtime_state.get("last_recoverable_error")
    assert isinstance(error, str)
    if expected is not None:
        assert error == expected
    if startswith is not None:
        assert error.startswith(startswith)
    if contains is not None:
        assert contains in error
    event = record.runtime_state.get("current_recovery_event")
    assert isinstance(event, dict)
    assert event.get("error") == error
    return error


def test_clear_active_recovery_state_removes_stale_done_markers(tmp_path: Path) -> None:
    file_recovery_constraints = {
        key: f"stale-{key}" for key in JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS
    }
    stage_recovery_constraints = {
        "failed_stage_ids": ["1"],
        "failed_stages": [{"stage": 1, "task_id": "core"}],
        "failed_task_id": "core-tests",
        "missing_stage_test_patch_stage_ids": ["1"],
        "missing_task_ids": ["core-tests"],
        "stages_missing_test_patches": [{"stage": 1, "task_id": "core"}],
        "unmet_dependencies": ["core"],
    }
    spec = JobSpec(
        job_id="clear-active-recovery",
        request_text="Build it",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "recovery_mode": "task_graph_replanning",
                "recovery_strategy": "task_graph_replanning",
                "recovery_next_actor": "planner",
                "recovery_next_status": "planning",
                "recovery_reason": "stale failure",
                "recovery_failed_task_id": "old-task",
                "recovery_failed_stage": 2,
                "recovery_attempt": 3,
                **file_recovery_constraints,
                **stage_recovery_constraints,
                "max_autonomous_stages": 12,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.FINALIZING)
    record.last_error = "invalid_task_graph"
    record.runtime_state.update(
        {
            "current_recovery_event": {"error": "invalid_task_graph"},
            "last_recoverable_error": "invalid_task_graph",
            "recovery_plan": {"strategy": "task_graph_replanning"},
            **file_recovery_constraints,
            **stage_recovery_constraints,
        }
    )
    record.outputs["last_recoverable_error"] = "invalid_task_graph"
    record.outputs["recovery_history"] = [{"strategy": "task_graph_replanning"}]

    JobRunner._clear_active_recovery_state(record)

    assert record.last_error is None
    assert "current_recovery_event" not in record.runtime_state
    assert "last_recoverable_error" not in record.runtime_state
    assert "recovery_plan" not in record.runtime_state
    for stale_key in (
        *JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS,
        "failed_stage_ids",
        "failed_stages",
        "failed_task_id",
        "missing_stage_test_patch_stage_ids",
        "missing_task_ids",
        "stages_missing_test_patches",
        "unmet_dependencies",
    ):
        assert stale_key not in record.runtime_state
    assert "last_recoverable_error" not in record.outputs
    assert record.outputs["recovery_history"] == [{"strategy": "task_graph_replanning"}]
    assert record.spec.metadata["constraints"] == {"max_autonomous_stages": 12}


def test_consume_completed_recovery_plan_clears_resolved_file_recovery_constraints(
    tmp_path: Path,
) -> None:
    target = "frontend/test/project_scaffold.test.tsx"
    (tmp_path / target).parent.mkdir(parents=True)
    (tmp_path / target).write_text("import { expect, it } from 'vitest'\n", encoding="utf-8")
    spec = JobSpec(
        job_id="consume-resolved-file-recovery",
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        metadata={
            "constraints": {
                "recovery_mode": "patch_operation_mismatch",
                "recovery_strategy": "RETURN_TO_TEST_WRITER",
                "recovery_next_actor": "test_writer",
                "recovery_next_status": JobStatus.WRITING_TESTS.value,
                "recovery_reason": "missing file recovery",
                "recovery_failed_task_id": "project-scaffold-tests",
                "recovery_failed_stage": 1,
                "recovery_attempt": 2,
                "patch_operation_hint": "create",
                "missing_target_file": target,
                "missing_artifacts": [],
                "max_autonomous_stages": 12,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.WRITING_TESTS)
    record.last_error = "target_files_missing:update target does not exist"
    record.runtime_state.update(
        {
            "current_recovery_event": {
                "error": "target_files_missing:update target does not exist",
            },
            "last_recoverable_error": "target_files_missing:update target does not exist",
            "missing_target_file": target,
            "failed_patch_path": target,
            "failed_patch_operation": "update",
            "recovery_plan": {
                "status": "completed",
                "next_status": JobStatus.WRITING_TESTS.value,
                "constraints": {
                    "missing_target_file": target,
                    "patch_operation_hint": "create",
                    "missing_artifacts": [],
                },
            },
        }
    )
    record.outputs["last_recoverable_error"] = (
        "target_files_missing:update target does not exist"
    )

    JobRunner._consume_completed_recovery_plan(record)

    assert record.last_error is None
    assert record.runtime_state["recovery_plan"]["consumed_by_runner"] is True
    assert "current_recovery_event" not in record.runtime_state
    assert "last_recoverable_error" not in record.runtime_state
    assert "last_recoverable_error" not in record.outputs
    assert "missing_target_file" not in record.runtime_state
    assert "failed_patch_path" not in record.runtime_state
    assert record.spec.metadata["constraints"] == {"max_autonomous_stages": 12}


def test_consume_completed_recovery_plan_keeps_unresolved_missing_file_context(
    tmp_path: Path,
) -> None:
    target = "src/app.py"
    spec = JobSpec(
        job_id="consume-unresolved-file-recovery",
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        metadata={
            "constraints": {
                "recovery_strategy": "RETURN_TO_IMPLEMENTER",
                "recovery_next_actor": "implementer",
                "missing_target_file": target,
                "missing_artifacts": [target],
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.IMPLEMENTING)
    record.runtime_state["recovery_plan"] = {
        "status": "completed",
        "next_status": JobStatus.IMPLEMENTING.value,
        "constraints": {
            "missing_target_file": target,
            "missing_artifacts": [target],
        },
    }

    JobRunner._consume_completed_recovery_plan(record)

    assert record.runtime_state["recovery_plan"]["consumed_by_runner"] is True
    assert record.spec.metadata["constraints"]["missing_target_file"] == target
    assert record.spec.metadata["constraints"]["missing_artifacts"] == [target]


def test_consume_completed_recovery_plan_keeps_non_file_recovery_context(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        job_id="consume-invalid-artifact-recovery",
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        metadata={
            "constraints": {
                "recovery_mode": "invalid_artifacts_replan",
                "recovery_strategy": "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
                "recovery_next_actor": "planner",
                "recovery_next_status": JobStatus.REPLANNING.value,
                "invalid_artifacts": ["../outside.py"],
                "missing_artifacts": [],
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.REPLANNING)
    record.runtime_state["recovery_plan"] = {
        "status": "completed",
        "next_status": JobStatus.REPLANNING.value,
        "constraints": {
            "invalid_artifacts": ["../outside.py"],
            "missing_artifacts": [],
        },
    }

    JobRunner._consume_completed_recovery_plan(record)

    assert record.runtime_state["recovery_plan"]["consumed_by_runner"] is True
    assert record.spec.metadata["constraints"]["recovery_next_actor"] == "planner"
    assert record.spec.metadata["constraints"]["invalid_artifacts"] == ["../outside.py"]
    assert record.spec.metadata["constraints"]["missing_artifacts"] == []


def test_run_structured_role_persists_active_status_before_model_call(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    store = InMemoryJobStore()
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    record = store.create(
        JobSpec(
            request_text="Build a feature.",
            repo_path=str(workspace),
            target_branch="acos/active-role-status",
            metadata={"constraints": {"model_timeout_seconds": 7.5}},
        )
    )

    def fake_run(**kwargs):
        persisted = store.get(record.job_id)
        assert persisted.status == JobStatus.ANALYZING
        assert persisted.runtime_state["active_role"] == "pm"
        assert persisted.runtime_state["active_objective"] == "Produce requirements"
        assert isinstance(persisted.runtime_state["active_started_at"], str)
        assert isinstance(persisted.runtime_state["active_model"], str)
        assert persisted.runtime_state["active_model_timeout_seconds"] == 7.5
        assert kwargs["request_timeout_seconds"] == 7.5
        selection = runner.model_router.select_model("pm")
        return (
            PRD(title="Feature", problem_statement="Need feature"),
            selection,
            ModelCallRecord(
                role="pm",
                model_key=selection.model_key,
                provider_key=selection.provider_key,
                status=ModelCallStatus.SUCCESS,
                input_hash="in",
                output_hash="out",
                prompt_tokens_estimate=1,
                completion_tokens_estimate=1,
                total_tokens_estimate=2,
            ),
        )

    runner.agent_runner.run = fake_run

    result = runner._run_structured_role(
        record,
        "pm",
        PRD,
        "Produce requirements",
    )

    assert result.title == "Feature"
    persisted = store.get(record.job_id)
    assert persisted.status == JobStatus.ANALYZING
    assert "active_role" not in persisted.runtime_state
    assert "active_model" not in persisted.runtime_state
    assert "active_model_timeout_seconds" not in persisted.runtime_state
    assert "active_started_at" not in persisted.runtime_state


def test_run_structured_role_clamps_timeout_to_runtime_deadline(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    store = InMemoryJobStore()
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    record = store.create(
        JobSpec(
            request_text="Build a feature.",
            repo_path=str(workspace),
            target_branch="acos/runtime-deadline",
            metadata={
                "constraints": {
                    "model_timeout_seconds": 300.0,
                    "model_timeout_deadline_epoch": datetime.now(timezone.utc).timestamp()
                    + 12.5,
                }
            },
        )
    )

    def fake_run(**kwargs):
        assert 0 < kwargs["request_timeout_seconds"] <= 12.5
        selection = runner.model_router.select_model("pm")
        return (
            PRD(title="Feature", problem_statement="Need feature"),
            selection,
            ModelCallRecord(
                role="pm",
                model_key=selection.model_key,
                provider_key=selection.provider_key,
                status=ModelCallStatus.SUCCESS,
                input_hash="in",
                output_hash="out",
                prompt_tokens_estimate=1,
                completion_tokens_estimate=1,
                total_tokens_estimate=2,
            ),
        )

    runner.agent_runner.run = fake_run

    result = runner._run_structured_role(
        record,
        "pm",
        PRD,
        "Produce requirements",
    )

    assert result.title == "Feature"


def test_run_structured_role_does_not_call_model_after_runtime_deadline(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    store = InMemoryJobStore()
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    record = store.create(
        JobSpec(
            request_text="Build a feature.",
            repo_path=str(workspace),
            target_branch="acos/runtime-deadline-expired",
            metadata={
                "constraints": {
                    "model_timeout_seconds": 300.0,
                    "model_timeout_deadline_epoch": 1.0,
                }
            },
        )
    )

    def fake_run(**kwargs):
        raise AssertionError("expired runtime deadline should skip the model call")

    runner.agent_runner.run = fake_run

    with pytest.raises(AdapterError, match="runtime deadline"):
        runner._run_structured_role(
            record,
            "pm",
            PRD,
            "Produce requirements",
        )
    persisted = store.get(record.job_id)
    assert "active_role" not in persisted.runtime_state
    assert "active_model" not in persisted.runtime_state
    assert "active_model_timeout_seconds" not in persisted.runtime_state
    assert "active_started_at" not in persisted.runtime_state


def test_run_structured_role_clears_active_status_after_adapter_error(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    store = InMemoryJobStore()
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    record = store.create(
        JobSpec(
            request_text="Build a feature.",
            repo_path=str(workspace),
            target_branch="acos/adapter-error-active-status",
            metadata={"constraints": {"model_timeout_seconds": 7.5}},
        )
    )

    def fake_run(**kwargs):
        persisted = store.get(record.job_id)
        assert persisted.runtime_state["active_role"] == "pm"
        assert persisted.runtime_state["active_model_timeout_seconds"] == 7.5
        assert kwargs["request_timeout_seconds"] == 7.5
        raise AdapterError("temporary timeout", code="timeout")

    runner.agent_runner.run = fake_run

    with pytest.raises(AdapterError, match="temporary timeout"):
        runner._run_structured_role(
            record,
            "pm",
            PRD,
            "Produce requirements",
        )
    persisted = store.get(record.job_id)
    assert "active_role" not in persisted.runtime_state
    assert "active_model" not in persisted.runtime_state
    assert "active_model_timeout_seconds" not in persisted.runtime_state
    assert "active_started_at" not in persisted.runtime_state


def test_run_tests_executes_runtime_acceptance_contracts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build runtime app",
            repo_path=str(workspace),
            target_branch="acos/runtime-acceptance-contract",
            metadata={
                "runtime": {
                    "startup_timeout_seconds": 37,
                    "http_checks": [
                        {
                            "name": "health",
                            "method": "GET",
                            "path": "/health",
                            "expect_status": 200,
                        }
                    ],
                },
                "acceptance_checks": [
                    {
                        "name": "home",
                        "method": "GET",
                        "path": "/",
                        "expect_status": 200,
                    }
                ],
                "constraints": {"test_command_name": "pytest"},
            },
        )
    )
    record.status = JobStatus.WRITING_TESTS
    calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        calls.append({"role": role, "tool_name": tool_name, **kwargs})
        if kwargs["command_name"] == "pytest":
            return TestRunResult(
                success=True,
                command=["pytest"],
                output_excerpt="1 passed",
                exit_code=0,
                executed_test_count=1,
            ).model_dump()
        assert kwargs["command_name"] == "runtime-smoke-auto"
        assert kwargs["timeout_seconds"] == 37
        assert kwargs["http_checks"] == [
            {
                "name": "health",
                "method": "GET",
                "path": "/health",
                "expect_status": 200,
            },
            {
                "name": "home",
                "method": "GET",
                "path": "/",
                "expect_status": 200,
            },
        ]
        return TestRunResult(
            success=True,
            command=["runtime-smoke-auto"],
            output_excerpt="runtime checks passed",
            exit_code=0,
        ).model_dump()

    runner._call_tool = fake_call_tool

    result = runner._run_tests(record)

    assert result.success is True
    assert [call["command_name"] for call in calls] == [
        "pytest",
        "runtime-smoke-auto",
    ]
    assert record.outputs["test_run"]["success"] is True
    assert record.outputs["runtime_smoke"]["success"] is True
    assert record.outputs["acceptance_checks"]["success"] is True


def test_run_tests_uses_explicit_runtime_start_command(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build runtime app",
            repo_path=str(workspace),
            target_branch="acos/runtime-start-command-contract",
            metadata={
                "runtime": {
                    "start_command": [
                        "python",
                        "-m",
                        "uvicorn",
                        "app.main:app",
                        "--host",
                        "{host}",
                        "--port",
                        "{port}",
                    ],
                    "http_probe_path": "/healthz",
                    "startup_timeout_seconds": 41,
                },
                "acceptance_checks": [
                    {
                        "name": "home",
                        "method": "GET",
                        "path": "/",
                        "expect_status": 200,
                    }
                ],
                "constraints": {"test_command_name": "pytest"},
            },
        )
    )
    record.status = JobStatus.WRITING_TESTS
    calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        calls.append({"role": role, "tool_name": tool_name, **kwargs})
        if tool_name == "test_server.run_test":
            return TestRunResult(
                success=True,
                command=["pytest"],
                output_excerpt="1 passed",
                exit_code=0,
                executed_test_count=1,
            ).model_dump()
        assert tool_name == "test_server.run_command"
        assert kwargs["argv"] == [
            "python",
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "{host}",
            "--port",
            "{port}",
        ]
        assert kwargs["mode"] == "server"
        assert kwargs["http_path"] == "/healthz"
        assert kwargs["timeout_seconds"] == 41
        assert kwargs["http_checks"] == [
            {
                "name": "home",
                "method": "GET",
                "path": "/",
                "expect_status": 200,
            }
        ]
        return TestRunResult(
            success=True,
            command=["runtime-start-command"],
            output_excerpt="runtime command passed",
            exit_code=0,
        ).model_dump()

    runner._call_tool = fake_call_tool

    result = runner._run_tests(record)

    assert result.success is True
    assert [call["tool_name"] for call in calls] == [
        "test_server.run_test",
        "test_server.run_command",
    ]
    assert record.outputs["runtime_smoke"]["success"] is True
    assert record.outputs["acceptance_checks"]["success"] is True


def test_prd_runtime_contracts_are_synthesized_into_job_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    prd = PRD(
        title="Runtime App",
        problem_statement="Need a web runtime check.",
        runtime=RuntimePlan(
            start_command=[
                "python",
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "{host}",
                "--port",
                "{port}",
            ],
            http_probe_path="/healthz",
        ),
        acceptance_checks=[
            RuntimeHttpCheck(
                name="home",
                method="GET",
                path="/",
                expect_status=200,
            )
        ],
        required_artifacts=["app/main.py", "tests/test_app.py"],
    )
    attach_mock_adapter(registry, {"pm": prd.model_dump()})
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build runtime app",
            repo_path=str(workspace),
            target_branch="acos/prd-runtime-contract-metadata",
        )
    )

    loaded = runner._load_or_refine_prd_for_autonomy(record)

    assert loaded == prd
    assert record.spec.metadata["runtime"]["start_command"] == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    assert record.spec.metadata["runtime"]["http_probe_path"] == "/healthz"
    acceptance_check = record.spec.metadata["acceptance_checks"][0]
    assert acceptance_check["name"] == "home"
    assert acceptance_check["method"] == "GET"
    assert acceptance_check["path"] == "/"
    assert acceptance_check["expect_status"] == 200
    assert record.spec.metadata["required_artifacts"] == [
        "app/main.py",
        "tests/test_app.py",
    ]
    assert record.outputs["execution_contracts"] == {
        "runtime": True,
        "acceptance_checks": True,
        "required_artifacts": ["app/main.py", "tests/test_app.py"],
        "framework_profile": None,
    }


def test_job_runner_review_request_changes_then_fix(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write code",
                        description="Write code",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": [
                ReviewResult(
                    decision=ReviewDecision.REQUEST_CHANGES,
                    summary="Needs work",
                    findings=[
                        {
                            "severity": Severity.MEDIUM,
                            "title": "Refactor",
                            "description": "Please refactor",
                        }
                    ],
                ).model_dump(),
                ReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Looks good now",
                ).model_dump(),
            ],
            "security_reviewer": [
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Looks safe",
                ).model_dump(),
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Still safe",
                ).model_dump(),
            ],
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Addressed review concerns",
                patches=[],
            ).model_dump(),
            "summarizer": SummaryResult(
                summary="Done",
                memory_entries=["feature completed"],
            ).model_dump(),
            "release_manager": ReleaseResult(
                summary="Ready",
                commit_message="acos: finalize feature",
                notify_message="done",
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/review-fixed",
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert len(environment.git_server.commits) == 1


def test_job_runner_max_attempts_triggers_recovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write code",
                        description="Write code",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create buggy module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 0\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="OK",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
            ).model_dump(),
            "fixer": [
                FixResult(status=FixStatus.FIXED, summary="No effective change", patches=[]).model_dump(),
                FixResult(status=FixStatus.FIXED, summary="Still no effective change", patches=[]).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stuck-case",
    )

    record = runner.run_job(spec)

    assert record.status in {JobStatus.REPLANNING, JobStatus.DIAGNOSING}
    assert record.runtime_state["recovery_plan"]["trigger"] in {
        "max_attempts_exceeded",
        "same_failure_threshold_reached",
    }
    assert record.outputs["recovery_history"]


def test_job_runner_fails_without_applying_fixer_patch_when_fixer_reports_failed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write code",
                        description="Write code",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create buggy module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 0\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="OK",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
            ).model_dump(),
            "fixer": FixResult(
                status=FixStatus.FAILED,
                summary="Could not fix the failure safely.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/fixer-failed",
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "fixer_failed:task-1")
    assert (workspace / "feature.py").read_text(encoding="utf-8") == "VALUE = 0\n"


def test_job_runner_pm_uses_context_without_live_tool_calls(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("hello\n", encoding="utf-8")
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(goal="Build feature").model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="noop",
                patches=[],
            ).model_dump(),
            "test_writer": TestWriterOutput(summary="noop", patches=[]).model_dump(),
            "reviewer": ReviewResult(decision=ReviewDecision.APPROVE, summary="ok").model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="ok",
            ).model_dump(),
            "summarizer": SummaryResult(summary="done", memory_entries=[]).model_dump(),
            "release_manager": ReleaseResult(
                summary="ready",
                commit_message="acos: ready",
                notify_message="done",
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())

    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "pm":
            captured["allowed_tools"] = kwargs.get("allowed_tools")
            captured["relevant_files"] = dict(kwargs["context_packet"].relevant_files)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/context-only-pm",
    )
    record = runner.submit(spec)
    runner._prepare_branch(record)
    runner._run_structured_role(record, "pm", PRD, "Produce the product requirements")

    assert captured["allowed_tools"] == []
    assert "__repo_map__.txt" in captured["relevant_files"]


def test_job_runner_can_skip_review_and_release_for_generation_test(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write code",
                        description="Write code",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/generation-test",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert record.outputs["test_run"]["success"] is True
    assert "summary" not in record.outputs
    assert environment.git_server.commits == []


def test_job_runner_runs_planned_tasks_in_dependency_order(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-2",
                        title="Add multiplication",
                        description="Add a multiplication helper after the base module exists.",
                        role="implementer",
                        depends_on=["task-1"],
                    ),
                    PlannedTask(
                        id="task-1",
                        title="Create base module",
                        description="Create the first feature module.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="task-3",
                        title="Cover helpers",
                        description="Add tests for the helpers.",
                        role="test_writer",
                        depends_on=["task-2"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created base module",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Added multiplication helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add base tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add helper tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/split-generation-test",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert record.outputs["test_run"]["success"] is True
    assert record.outputs["implementation_task_count"] == 2
    assert record.outputs["test_writer_task_count"] == 2
    assert [item["task"]["id"] for item in record.outputs["implementation_tasks"]] == [
        "task-1",
        "task-2",
    ]
    assert record.outputs["implementation"]["changed_files"] == ["feature.py"]
    assert "def double" in (workspace / "feature.py").read_text(encoding="utf-8")


def test_job_runner_tests_each_autonomous_stage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="core-tests",
                        title="Test core helper",
                        description="Test the smallest working helper.",
                        role="test_writer",
                        depends_on=["core"],
                    ),
                    PlannedTask(
                        id="extra",
                        title="Add extra helper",
                        description="Add one more helper after the core passes.",
                        role="implementer",
                        depends_on=["core-tests"],
                    ),
                    PlannedTask(
                        id="extra-tests",
                        title="Test extra helper",
                        description="Test the added helper.",
                        role="test_writer",
                        depends_on=["extra"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created core helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Added extra helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add core tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add extra tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/autonomous-stage-test",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert record.outputs["implementation_task_count"] == 2
    assert record.outputs["test_writer_task_count"] == 2
    assert record.outputs["autonomous_stages"][0]["change_summary"]["changed_files"] == [
        "feature.py",
        "tests/test_feature.py",
    ]
    assert record.outputs["autonomous_stages"][0]["change_summary"]["patch_count"] == 2
    stage_test_runs = [
        stage["test_run"]
        for stage in record.outputs["autonomous_stages"]
        if stage["test_run"] is not None
    ]
    assert len(stage_test_runs) == 2
    assert all(stage["success"] for stage in stage_test_runs)


def test_job_runner_synthesizes_stage_tests_when_planner_omits_test_tasks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="extra",
                        title="Add extra helper",
                        description="Add one more helper after the core passes.",
                        role="implementer",
                        depends_on=["core"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created core helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Added extra helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add synthesized core tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add synthesized extra tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/synthesized-stage-tests",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert record.outputs["implementation_task_count"] == 2
    assert record.outputs["test_writer_task_count"] == 2
    assert [stage["test_run"]["success"] for stage in record.outputs["autonomous_stages"]] == [
        True,
        True,
    ]
    assert [
        item["task"]["id"] for item in record.outputs["test_writer_tasks"]
    ] == ["core", "extra"]


def test_job_runner_refines_coarse_task_graph_from_pm_small_parts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                small_parts=["Create add_one helper", "Create double helper"],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                ],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build all helpers",
                tasks=[
                    PlannedTask(
                        id="build-everything",
                        title="Build all helpers",
                        description="Implement all helpers in one pass.",
                        role="implementer",
                        complexity="high",
                    )
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created add_one helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created double helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add add_one tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add double tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/refined-coarse-task-graph",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert record.outputs["task_graph_refinement"]["applied"] is True
    assert record.outputs["task_graph_refinement"]["original_task_count"] == 1
    assert record.outputs["implementation_task_count"] == 2
    assert record.outputs["task_graph_validation"]["small_part_coverage"] == [
        {
            "small_part_index": 1,
            "small_part": "Create add_one helper",
            "task_id": "part-01",
            "covered": True,
        },
        {
            "small_part_index": 2,
            "small_part": "Create double helper",
            "task_id": "part-02",
            "covered": True,
        },
    ]
    assert record.outputs["task_graph_validation"]["acceptance_test_coverage"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "add_one(2) returns 3",
            "task_id": "part-01",
            "covered": True,
        },
        {
            "acceptance_test_index": 2,
            "acceptance_test": "double(4) returns 8",
            "task_id": "part-02",
            "covered": True,
        },
    ]
    assert [item["task"]["id"] for item in record.outputs["implementation_tasks"]] == [
        "part-01",
        "part-02",
    ]
    assert [stage["test_run"]["success"] for stage in record.outputs["autonomous_stages"]] == [
        True,
        True,
    ]


def test_job_runner_refinement_preserves_task_artifact_contracts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                small_parts=["Create add_one helper", "Create double helper"],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                ],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build all helpers",
                tasks=[
                    PlannedTask(
                        id="build-everything",
                        title="Build all helpers",
                        description="Implement all helpers in one pass.",
                        role="implementer",
                        complexity="high",
                        target_files=["feature.py", "docs/"],
                        required_artifacts=[
                            "feature.py",
                            "tests/test_feature.py",
                            "../outside.py",
                            "C:\\outside.py",
                        ],
                    )
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created add_one helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created double helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add add_one tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add double tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/refined-artifact-contract",
        metadata={
            "constraints": {
                "require_task_acceptance_criteria": True,
                "require_task_artifacts": True,
                "task_graph_validation_refinement_attempts": 0,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["task_graph_validation"]["valid"] is True
    assert record.outputs["task_graph_validation"]["executable_task_artifact_count"] == 4
    assert record.outputs["task_graph_refinement"]["inherited_target_files"] == [
        "feature.py"
    ]
    assert record.outputs["task_graph_refinement"][
        "inherited_required_artifacts"
    ] == ["feature.py", "tests/test_feature.py"]
    assert record.outputs["task_graph_refinement"][
        "invalid_inherited_artifacts"
    ] == ["docs/", "../outside.py", "C:\\outside.py"]
    assert record.outputs["task_graph_refinement"]["paired_test_task_count"] == 2
    refined_tasks = record.outputs["task_graph"]["tasks"]
    assert [task["id"] for task in refined_tasks] == [
        "part-01",
        "part-01-tests",
        "part-02",
        "part-02-tests",
    ]
    for task in refined_tasks:
        if task["role"] == "test_writer":
            assert task["target_files"] == ["tests/test_feature.py"]
            assert task["required_artifacts"] == ["tests/test_feature.py"]
        else:
            assert task["target_files"] == ["feature.py"]
            assert task["required_artifacts"] == ["feature.py"]


def test_test_work_item_classifier_uses_word_tokens() -> None:
    assert JobRunner._looks_like_test_work_item("Add focused tests")
    assert JobRunner._looks_like_test_work_item("pytest covers the helper")
    assert not JobRunner._looks_like_test_work_item("Create contests page")
    assert not JobRunner._looks_like_test_work_item("Create protest workflow")


def test_job_runner_plan_job_stops_after_validated_task_graph(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need two helpers",
                smallest_working_core=["Create add_one helper"],
                small_parts=["Create add_one helper", "Create double helper"],
                incremental_milestones=[
                    "add_one works",
                    "double works",
                ],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                ],
                definition_of_done=["All generated tests pass"],
                required_artifacts=[
                    "add_one.py",
                    "double.py",
                    "tests/test_add_one_double.py",
                ],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build helpers",
                tasks=[
                    PlannedTask(
                        id="add-one",
                        title="Create add_one helper",
                        description="Implement add_one.",
                        role="implementer",
                        acceptance_criteria=["add_one(2) returns 3"],
                        target_files=["add_one.py"],
                        required_artifacts=["add_one.py"],
                    ),
                    PlannedTask(
                        id="double",
                        title="Create double helper",
                        description="Implement double.",
                        role="implementer",
                        depends_on=["add-one"],
                        acceptance_criteria=["double(4) returns 8"],
                        target_files=["double.py"],
                        required_artifacts=["double.py"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run during planning only.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/plan-only",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
                "recovery_mode": "task_graph_replanning",
                "recovery_strategy": "REPLAN_TASK",
                "recovery_next_actor": "planner",
                "recovery_next_status": "planning",
                "patch_operation_hint": "create",
                "missing_target_file": "frontend/test/project_scaffold.test.tsx",
            }
        },
    )

    record = runner.plan_job(spec)

    assert record.status == JobStatus.PLANNING
    assert record.last_error is None
    assert record.outputs["planning_only"] == {
        "complete": True,
        "ready_for_implementation": True,
    }
    assert record.outputs["prd"]["small_parts"] == [
        "Create add_one helper",
        "Create double helper",
    ]
    assert record.outputs["task_graph_validation"]["valid"] is True
    assert record.spec.metadata["constraints"] == {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
    }
    assert "implementation" not in record.outputs
    assert "implementation_tasks" not in record.outputs
    assert not (workspace / "feature.py").exists()


def test_job_runner_resume_after_plan_job_uses_existing_planning_outputs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    role_calls = {"pm": 0, "architect": 0, "planner": 0}

    def pm_response(metadata):
        role_calls["pm"] += 1
        return PRD(
            title="Feature",
            problem_statement="Need feature",
            smallest_working_core=["Expose a VALUE constant and test it"],
            small_parts=["Create feature module"],
            incremental_milestones=["Module exists"],
            acceptance_tests=["VALUE equals 1"],
            definition_of_done=["All tests pass"],
            required_artifacts=["feature.py", "tests/test_feature.py"],
        ).model_dump()

    def architect_response(metadata):
        role_calls["architect"] += 1
        return ArchitecturePlan(summary="Simple architecture").model_dump()

    def planner_response(metadata):
        role_calls["planner"] += 1
        return TaskGraph(
            goal="Build feature",
            tasks=[
                PlannedTask(
                    id="core",
                    title="Build core",
                    description="Build the smallest feature.",
                    role="implementer",
                    acceptance_criteria=["VALUE equals 1"],
                ),
                PlannedTask(
                    id="core-tests",
                    title="Test core",
                    description="Add focused tests for VALUE.",
                    role="test_writer",
                    depends_on=["core"],
                    acceptance_criteria=["VALUE equals 1"],
                    target_files=["tests/test_feature.py"],
                    required_artifacts=["tests/test_feature.py"],
                )
            ],
        ).model_dump()

    attach_mock_adapter(
        registry,
        {
            "pm": pm_response,
            "architect": architect_response,
            "planner": planner_response,
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/plan-then-resume",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    planned = runner.plan_job(spec)
    assert planned.status == JobStatus.PLANNING
    assert planned.outputs["planning_only"]["complete"] is True
    planned_task = planned.outputs["task_graph"]["tasks"][0]
    assert planned_task["target_files"] == ["feature.py"]
    assert planned_task["required_artifacts"] == ["feature.py"]
    assert planned.outputs["task_graph_acceptance_enrichment"][
        "artifact_updated_task_ids"
    ] == ["core"]
    planned_job_id = planned.job_id

    resumed = runner.resume_job(planned_job_id)

    assert role_calls == {"pm": 1, "architect": 1, "planner": 1}
    assert resumed.status == JobStatus.DONE
    assert resumed.last_error is None
    assert resumed.outputs["implementation_task_count"] == 1
    assert resumed.outputs["task_graph_validation"]["valid"] is True
    assert resumed.outputs["planning_only"]["ready_for_implementation"] is True
    assert (workspace / "feature.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_job_runner_enriches_multi_task_graph_with_prd_acceptance_criteria(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Create add_one helper"],
                small_parts=["Create add_one helper", "Create double helper"],
                incremental_milestones=[
                    "add_one is implemented and tested",
                    "double is implemented and tested",
                ],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                ],
                definition_of_done=["All generated tests pass"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build helpers",
                tasks=[
                    PlannedTask(
                        id="add-one",
                        title="Create add_one helper",
                        description="Implement add_one.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="double",
                        title="Create double helper",
                        description="Implement double.",
                        role="implementer",
                        depends_on=["add-one"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created add_one helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created double helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add add_one tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add double tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/enriched-task-criteria",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["task_graph_acceptance_enrichment"] == {
        "applied": True,
        "reason": "filled_missing_task_acceptance_criteria_from_prd",
        "updated_task_ids": ["add-one", "double"],
    }
    tasks = record.outputs["task_graph"]["tasks"]
    assert tasks[0]["acceptance_criteria"] == ["add_one(2) returns 3"]
    assert tasks[1]["acceptance_criteria"] == ["double(4) returns 8"]
    assert record.outputs["implementation_task_count"] == 2


def test_job_runner_replaces_placeholder_acceptance_criteria_when_enriching(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = JobRecord(
        job_id="placeholder-task-criteria-enrichment",
        spec=JobSpec(
            job_id="placeholder-task-criteria-enrichment",
            request_text="Create feature",
            repo_path=str(workspace),
        ),
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All tests pass"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["TBD"],
            )
        ],
    )

    refined = runner._enrich_task_graph_acceptance_criteria(record, prd, task_graph)

    assert refined.tasks[0].acceptance_criteria == ["VALUE equals 1"]
    assert record.outputs["task_graph_acceptance_enrichment"] == {
        "applied": True,
        "reason": "filled_missing_task_acceptance_criteria_from_prd",
        "updated_task_ids": ["core"],
    }


def test_job_runner_preserves_existing_acceptance_criteria_when_enriching_later_tasks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Create add_one helper"],
                small_parts=["Create add_one helper", "Create double helper"],
                incremental_milestones=[
                    "add_one is implemented and tested",
                    "double is implemented and tested",
                ],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                ],
                definition_of_done=["All generated tests pass"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build helpers",
                tasks=[
                    PlannedTask(
                        id="add-one",
                        title="Create add_one helper",
                        description="Implement add_one.",
                        role="implementer",
                        acceptance_criteria=["existing add_one criterion"],
                    ),
                    PlannedTask(
                        id="double",
                        title="Create double helper",
                        description="Implement double.",
                        role="implementer",
                        depends_on=["add-one"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created add_one helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created double helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add add_one tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add double tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/enriched-mixed-task-criteria",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["task_graph_acceptance_enrichment"]["updated_task_ids"] == ["double"]
    tasks = record.outputs["task_graph"]["tasks"]
    assert tasks[0]["acceptance_criteria"] == ["existing add_one criterion"]
    assert tasks[1]["acceptance_criteria"] == ["double(4) returns 8"]


def test_task_graph_enrichment_matches_prd_artifacts_to_multi_tasks(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build an English vocabulary app",
            repo_path=str(workspace),
            target_branch="acos/artifact-match",
        )
    )
    prd = PRD(
        title="English Vocab App",
        problem_statement="Need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can authenticate",
            "Teachers can manage word sets",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/words.py",
            "backend/main.py",
            "tests/test_vocab_app.py",
        ],
    )
    task_graph = TaskGraph(
        goal="Build English vocabulary app",
        tasks=[
            PlannedTask(
                id="auth",
                title="User authentication",
                description="Implement authentication and roles.",
                role="implementer",
            ),
            PlannedTask(
                id="word-sets",
                title="Word set CRUD",
                description="Implement word set CRUD operations.",
                role="implementer",
                depends_on=["auth"],
            ),
        ],
    )

    refined = runner._enrich_task_graph_acceptance_criteria(record, prd, task_graph)

    tasks = {task.id: task for task in refined.tasks}
    assert tasks["auth"].target_files == ["backend/auth.py"]
    assert tasks["auth"].required_artifacts == ["backend/auth.py"]
    assert tasks["word-sets"].target_files == ["backend/words.py"]
    assert tasks["word-sets"].required_artifacts == ["backend/words.py"]
    assert tasks["prd-tests"].target_files == ["tests/test_vocab_app.py"]
    assert tasks["prd-tests"].required_artifacts == ["tests/test_vocab_app.py"]
    assert tasks["prd-tests"].depends_on == ["auth", "word-sets"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "synthesized_test_writer_task_ids"
    ] == ["prd-tests"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "artifact_updated_task_ids"
    ] == ["auth", "word-sets", "prd-tests"]

    validation = JobRunner._build_task_graph_validation(
        refined,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert "backend/main.py" in validation["unassigned_required_artifacts"]
    assert "tests/test_vocab_app.py" not in validation["unassigned_required_artifacts"]


def test_task_graph_enrichment_synthesizes_missing_test_writer_task(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/synthesize-test-writer",
        )
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All generated tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module exposing VALUE.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    refined = runner._enrich_task_graph_acceptance_criteria(record, prd, task_graph)

    tasks = {task.id: task for task in refined.tasks}
    assert list(tasks) == ["core", "core-tests"]
    assert tasks["core-tests"].role == "test_writer"
    assert tasks["core-tests"].depends_on == ["core"]
    assert tasks["core-tests"].acceptance_criteria == ["VALUE equals 1"]
    assert tasks["core-tests"].target_files == ["tests/test_feature.py"]
    assert tasks["core-tests"].required_artifacts == ["tests/test_feature.py"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "synthesized_test_writer_task_ids"
    ] == ["core-tests"]

    validation = JobRunner._build_task_graph_validation(
        refined,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["missing_test_writer_tasks"] is False
    assert validation["unassigned_required_artifacts"] == []


def test_task_graph_enrichment_fills_test_writer_prd_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/test-writer-artifact-enrichment",
        )
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All generated tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused regression tests for VALUE.",
                role="test_writer",
                depends_on=["core"],
            ),
        ],
    )

    refined = runner._enrich_task_graph_acceptance_criteria(record, prd, task_graph)

    tasks = {task.id: task for task in refined.tasks}
    assert tasks["tests"].acceptance_criteria == ["VALUE equals 1"]
    assert tasks["tests"].target_files == ["tests/test_feature.py"]
    assert tasks["tests"].required_artifacts == ["tests/test_feature.py"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "updated_task_ids"
    ] == ["tests"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "artifact_updated_task_ids"
    ] == ["tests"]
    assert record.outputs["task_graph_acceptance_enrichment"][
        "inherited_test_artifacts"
    ] == ["tests/test_feature.py"]

    validation = JobRunner._build_task_graph_validation(
        refined,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["test_writer_tasks_missing_target_files"] == []
    assert validation["unassigned_required_artifacts"] == []


def test_job_runner_strict_task_acceptance_criteria_uses_prd_sources(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                small_parts=["Create feature module"],
                acceptance_tests=["VALUE equals 1"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/strict-task-criteria-from-prd",
        metadata={
            "constraints": {
                "require_task_acceptance_criteria": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["task_graph_validation"]["valid"] is True
    assert record.outputs["task_graph_validation"]["require_acceptance_criteria"] is True
    assert record.outputs["task_graph_validation"][
        "implementation_task_acceptance_criteria_count"
    ] == 1
    assert record.outputs["task_graph"]["tasks"][0]["acceptance_criteria"] == ["VALUE equals 1"]
    assert (workspace / "feature.py").exists()


def test_job_runner_blocks_strict_task_without_acceptance_criteria_before_implementation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                small_parts=["Create feature module"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/strict-task-criteria-blocked",
        metadata={
            "constraints": {
                "require_task_acceptance_criteria": True,
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["require_acceptance_criteria"] is True
    assert validation["implementation_task_acceptance_criteria_count"] == 0
    assert validation["errors"] == [
        {"type": "missing_acceptance_criteria", "task_ids": ["core"]}
    ]
    assert not (workspace / "feature.py").exists()


def test_task_graph_validation_requires_task_artifacts_when_requested() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["require_task_artifacts"] is True
    assert validation["implementation_task_artifact_count"] == 0
    assert validation["executable_task_artifact_count"] == 0
    assert validation["errors"] == [
        {"type": "missing_task_artifacts", "task_ids": ["core"]},
        {"type": "missing_required_artifacts", "task_ids": ["core"]},
        {"type": "missing_implementation_target_files", "task_ids": ["core"]},
    ]


def test_task_graph_validation_rejects_implementation_artifacts_without_targets() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["implementation_tasks_missing_target_files"] == ["core"]
    assert "missing_implementation_target_files" in {
        item["type"] for item in validation["errors"]
    }


def test_task_graph_validation_rejects_target_files_without_required_artifacts() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["executable_tasks_missing_required_artifacts"] == ["core"]
    assert {
        "type": "missing_required_artifacts",
        "task_ids": ["core"],
    } in validation["errors"]


def test_task_graph_validation_rejects_required_artifacts_not_targeted() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["other.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["required_artifacts_missing_target_files"] == [
        {"task_id": "core", "role": "implementer", "paths": ["other.py"]}
    ]
    assert {
        "type": "required_artifacts_missing_target_files",
        "items": validation["required_artifacts_missing_target_files"],
    } in validation["errors"]


def test_task_graph_validation_rejects_target_files_not_required() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py", "unverified.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["target_files_missing_required_artifacts"] == [
        {"task_id": "core", "role": "implementer", "paths": ["unverified.py"]}
    ]
    assert {
        "type": "target_files_missing_required_artifacts",
        "items": validation["target_files_missing_required_artifacts"],
    } in validation["errors"]


def test_task_graph_validation_requires_test_writer_artifacts_when_requested() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused regression tests.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["VALUE is covered by a regression test"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["implementation_task_artifact_count"] == 1
    assert validation["executable_task_artifact_count"] == 1
    assert validation["test_writer_tasks_missing_target_files"] == ["tests"]
    assert {"type": "missing_task_artifacts", "task_ids": ["tests"]} in validation["errors"]
    assert {"type": "missing_required_artifacts", "task_ids": ["tests"]} in validation["errors"]
    assert {"type": "missing_test_writer_target_files", "task_ids": ["tests"]} in validation["errors"]
    assert validation["test_writer_dependency_semantic_mismatches"] == [
        {
            "task_id": "tests",
            "depends_on": ["core"],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]


def test_task_graph_validation_requires_test_writer_implementation_dependency() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused regression tests.",
                role="test_writer",
                acceptance_criteria=["VALUE is covered by a regression test"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_missing_implementation_dependencies"] == [
        {
            "task_id": "tests",
            "depends_on": [],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert {
        "type": "test_writer_missing_implementation_dependency",
        "items": validation["test_writer_missing_implementation_dependencies"],
    } in validation["errors"]


def test_task_graph_validation_rejects_unrelated_test_writer_dependency() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="backend-api",
                title="Build backend API",
                description="Expose backend JSON endpoints.",
                role="implementer",
                acceptance_criteria=["Backend API returns JSON"],
                target_files=["backend/main.py"],
                required_artifacts=["backend/main.py"],
            ),
            PlannedTask(
                id="frontend-ui",
                title="Build frontend UI",
                description="Render VALUE in the browser UI.",
                role="implementer",
                acceptance_criteria=["Frontend UI renders VALUE"],
                target_files=["frontend/src/App.tsx"],
                required_artifacts=["frontend/src/App.tsx"],
            ),
            PlannedTask(
                id="frontend-tests",
                title="Test frontend UI",
                description="Test the browser UI rendering VALUE.",
                role="test_writer",
                depends_on=["backend-api"],
                acceptance_criteria=["Frontend UI renders VALUE"],
                target_files=["frontend/test/app.test.tsx"],
                required_artifacts=["frontend/test/app.test.tsx"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_dependency_semantic_mismatches"] == [
        {
            "task_id": "frontend-tests",
            "depends_on": ["backend-api"],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert {
        "type": "test_writer_dependency_semantic_mismatch",
        "items": validation["test_writer_dependency_semantic_mismatches"],
    } in validation["errors"]


def test_task_graph_validation_rejects_single_token_test_writer_dependency_match() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="backend-api",
                title="Build backend API",
                description="Expose backend VALUE endpoint.",
                role="implementer",
                acceptance_criteria=["Backend API returns VALUE"],
                target_files=["backend/main.py"],
                required_artifacts=["backend/main.py"],
            ),
            PlannedTask(
                id="frontend-tests",
                title="Test frontend UI",
                description="Test the browser UI rendering VALUE.",
                role="test_writer",
                depends_on=["backend-api"],
                acceptance_criteria=["VALUE is covered by a regression test"],
                target_files=["frontend/test/app.test.tsx"],
                required_artifacts=["frontend/test/app.test.tsx"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_dependency_semantic_mismatches"] == [
        {
            "task_id": "frontend-tests",
            "depends_on": ["backend-api"],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert {
        "type": "test_writer_dependency_semantic_mismatch",
        "items": validation["test_writer_dependency_semantic_mismatches"],
    } in validation["errors"]


def test_task_graph_validation_allows_aggregate_test_writer_dependencies() -> None:
    task_graph = TaskGraph(
        goal="Build vocabulary app",
        tasks=[
            PlannedTask(
                id="auth",
                title="User authentication",
                description="Implement authentication and roles.",
                role="implementer",
                acceptance_criteria=["Student can register and login"],
                target_files=["backend/auth.py"],
                required_artifacts=["backend/auth.py"],
            ),
            PlannedTask(
                id="word-sets",
                title="Word set CRUD",
                description="Implement word set CRUD operations.",
                role="implementer",
                acceptance_criteria=["Teacher can perform CRUD for word sets"],
                target_files=["backend/words.py"],
                required_artifacts=["backend/words.py"],
            ),
            PlannedTask(
                id="prd-tests",
                title="PRD acceptance tests",
                description="Test authentication and word set CRUD together.",
                role="test_writer",
                depends_on=["auth", "word-sets"],
                acceptance_criteria=[
                    "Student can register and login",
                    "Teacher can perform CRUD for word sets",
                ],
                target_files=["tests/test_vocab_app.py"],
                required_artifacts=["tests/test_vocab_app.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["test_writer_dependency_semantic_mismatches"] == []
    assert validation["test_writer_acceptance_dependency_mismatches"] == []


def test_task_graph_validation_requires_test_writer_acceptance_dependency_match() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="backend-api",
                title="Build backend API",
                description="Expose backend JSON endpoints.",
                role="implementer",
                acceptance_criteria=["Backend API returns VALUE"],
                target_files=["backend/main.py"],
                required_artifacts=["backend/main.py"],
            ),
            PlannedTask(
                id="frontend-ui",
                title="Build frontend UI",
                description="Render VALUE in the browser UI.",
                role="implementer",
                acceptance_criteria=["Frontend UI renders VALUE"],
                target_files=["frontend/src/App.tsx"],
                required_artifacts=["frontend/src/App.tsx"],
            ),
            PlannedTask(
                id="frontend-tests",
                title="Test frontend UI",
                description="Test the browser UI rendering VALUE.",
                role="test_writer",
                depends_on=["backend-api"],
                acceptance_criteria=["Frontend UI renders VALUE"],
                target_files=["frontend/test/app.test.tsx"],
                required_artifacts=["frontend/test/app.test.tsx"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_dependency_semantic_mismatches"] == [
        {
            "task_id": "frontend-tests",
            "depends_on": ["backend-api"],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert validation["test_writer_acceptance_dependency_mismatches"] == [
        {
            "task_id": "frontend-tests",
            "depends_on": ["backend-api"],
            "required_dependency_roles": ["implementer", "scaffold"],
            "uncovered_acceptance_criteria": [
                {
                    "acceptance_criteria_index": 1,
                    "acceptance_criteria": "Frontend UI renders VALUE",
                    "covered": False,
                }
            ],
        }
    ]
    assert {
        "type": "test_writer_acceptance_dependency_mismatch",
        "items": validation["test_writer_acceptance_dependency_mismatches"],
    } in validation["errors"]


def test_task_graph_validation_requires_test_writer_acceptance_criteria() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused regression tests.",
                role="test_writer",
                depends_on=["core"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_tasks_missing_acceptance_criteria"] == ["tests"]
    assert validation["test_writer_task_acceptance_criteria_count"] == 0
    assert validation["executable_task_acceptance_criteria_count"] == 1
    assert {
        "type": "missing_test_writer_acceptance_criteria",
        "task_ids": ["tests"],
    } in validation["errors"]


def test_task_graph_validation_treats_placeholder_acceptance_criteria_as_missing() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["TBD"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused regression tests.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["placeholder"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
    )

    assert validation["valid"] is False
    assert validation["implementation_task_acceptance_criteria_count"] == 0
    assert validation["test_writer_task_acceptance_criteria_count"] == 0
    assert validation["executable_task_acceptance_criteria_count"] == 0
    assert validation["implementation_task_count"] == 1
    assert validation["test_writer_task_count"] == 1
    assert validation["errors"] == [
        {"type": "missing_acceptance_criteria", "task_ids": ["core"]},
        {"type": "missing_test_writer_acceptance_criteria", "task_ids": ["tests"]},
    ]


def test_task_graph_validation_requires_test_writer_to_cover_prd_acceptance() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All generated tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add generic regression tests.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Generated tests pass"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["uncovered_acceptance_tests"] == []
    assert validation["uncovered_test_writer_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "VALUE equals 1",
            "task_id": None,
            "covered": False,
        }
    ]
    assert {
        "type": "semantic_test_writer_acceptance_mismatch",
        "acceptance_test_count": 1,
        "test_writer_task_count": 1,
        "uncovered_test_writer_acceptance_tests": (
            validation["uncovered_test_writer_acceptance_tests"]
        ),
    } in validation["errors"]


def test_task_graph_validation_requires_implementation_to_cover_prd_acceptance() -> None:
    prd = PRD(
        title="WordSet API",
        problem_statement="Teachers need to create word sets.",
        smallest_working_core=["Serve a backend API"],
        small_parts=["Create backend service shell"],
        incremental_milestones=["Backend service starts"],
        acceptance_tests=["POST /api/word-sets creates a WordSet"],
        definition_of_done=["All generated tests pass"],
        required_artifacts=["src/server/index.ts", "tests/backend.test.ts"],
    )
    task_graph = TaskGraph(
        goal="Build WordSet API",
        tasks=[
            PlannedTask(
                id="backend-shell",
                title="Build backend service shell",
                description="Create the backend service shell.",
                role="implementer",
                acceptance_criteria=["Backend service starts"],
                target_files=["src/server/index.ts"],
                required_artifacts=["src/server/index.ts"],
            ),
            PlannedTask(
                id="wordset-create-tests",
                title="Test WordSet creation",
                description="Test the WordSet creation endpoint.",
                role="test_writer",
                depends_on=["backend-shell"],
                acceptance_criteria=["POST /api/word-sets creates a WordSet"],
                target_files=["tests/backend.test.ts"],
                required_artifacts=["tests/backend.test.ts"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["uncovered_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "POST /api/word-sets creates a WordSet",
            "task_id": None,
            "covered": False,
        }
    ]
    assert validation["test_writer_acceptance_test_coverage"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "POST /api/word-sets creates a WordSet",
            "task_id": "wordset-create-tests",
            "covered": True,
        }
    ]
    assert {
        "type": "semantic_acceptance_test_mismatch",
        "acceptance_test_count": 1,
        "implementation_task_count": 1,
        "uncovered_acceptance_tests": validation["uncovered_acceptance_tests"],
    } in validation["errors"]


def test_task_graph_validation_allows_test_writer_dependency_on_scaffold() -> None:
    task_graph = TaskGraph(
        goal="Build scaffold",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create the deterministic app scaffold.",
                role="scaffold",
                acceptance_criteria=["Scaffold exists"],
                target_files=["backend/main.py"],
                required_artifacts=["backend/main.py"],
            ),
            PlannedTask(
                id="project-scaffold-tests",
                title="Test scaffold",
                description="Add focused scaffold tests.",
                role="test_writer",
                depends_on=["project-scaffold"],
                acceptance_criteria=["Scaffold is covered by a smoke test"],
                target_files=["tests/test_project_scaffold.py"],
                required_artifacts=["tests/test_project_scaffold.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["test_writer_missing_implementation_dependencies"] == []


def test_task_graph_validation_rejects_executor_unsatisfiable_dependency_order() -> None:
    task_graph = TaskGraph(
        goal="Build feature incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create the core feature.",
                role="implementer",
                acceptance_criteria=["Core exists"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="extra",
                title="Build extra",
                description="Create extra behavior after later tests pass.",
                role="implementer",
                depends_on=["later-tests"],
                acceptance_criteria=["Extra exists"],
                target_files=["extra.py"],
                required_artifacts=["extra.py"],
            ),
            PlannedTask(
                id="later",
                title="Build later",
                description="Create later behavior.",
                role="implementer",
                acceptance_criteria=["Later exists"],
                target_files=["later.py"],
                required_artifacts=["later.py"],
            ),
            PlannedTask(
                id="later-tests",
                title="Test later",
                description="Test later behavior.",
                role="test_writer",
                depends_on=["later"],
                acceptance_criteria=["Later has tests"],
                target_files=["tests/test_later.py"],
                required_artifacts=["tests/test_later.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["executor_order_dependency_violations"] == [
        {
            "task_id": "extra",
            "role": "implementer",
            "executor_phase": "implementation",
            "unmet_dependencies": ["later-tests"],
            "dependency_roles": [
                {"task_id": "later-tests", "role": "test_writer"}
            ],
        }
    ]
    assert {
        "type": "executor_order_dependency_violations",
        "items": validation["executor_order_dependency_violations"],
    } in validation["errors"]


def test_task_graph_validation_requires_test_writer_for_acceptance_tests() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All generated tests pass"],
        required_artifacts=["feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["test_writer_task_count"] == 0
    assert validation["missing_test_writer_tasks"] is True
    assert validation["missing_test_writer_task_requirements"] == [
        {
            "acceptance_tests": True,
            "test_focused_small_parts": False,
            "prd_test_required_artifacts": [],
        }
    ]
    assert {
        "type": "missing_test_writer_tasks",
        "required_by": {
            "acceptance_tests": True,
            "test_focused_small_parts": False,
            "prd_test_required_artifacts": [],
        },
    } in validation["errors"]
    record = JobRecord(
        job_id="missing-test-writer-context",
        spec=JobSpec(request_text="Build it", repo_path="."),
    )
    runtime_state = JobRunner._task_graph_validation_recovery_state(record, validation)
    assert runtime_state["missing_test_writer_task_requirements"] == [
        {
            "acceptance_tests": True,
            "test_focused_small_parts": False,
            "prd_test_required_artifacts": [],
        }
    ]


def test_task_graph_validation_allows_multiple_acceptance_tests_per_task() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need word set management.",
        smallest_working_core=["Manage word sets"],
        small_parts=["Implement backend API endpoints for WordSet CRUD"],
        incremental_milestones=["WordSet CRUD works"],
        acceptance_tests=[
            "GET /api/word-sets returns the WordSet list",
            "POST /api/word-sets creates a WordSet",
            "DELETE /api/word-sets/:id removes a WordSet",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=["src/server/index.ts", "tests/backend.test.ts"],
    )
    task_graph = TaskGraph(
        goal="Build WordSet CRUD",
        tasks=[
            PlannedTask(
                id="wordset-crud-api",
                title="Implement WordSet CRUD API",
                description="Implement GET, POST, and DELETE endpoints for WordSet CRUD.",
                role="implementer",
                acceptance_criteria=[
                    "GET /api/word-sets returns the WordSet list",
                    "POST /api/word-sets creates a WordSet",
                    "DELETE /api/word-sets/:id removes a WordSet",
                ],
                target_files=["src/server/index.ts"],
                required_artifacts=["src/server/index.ts"],
            ),
            PlannedTask(
                id="backend-tests",
                title="Test WordSet CRUD API",
                description="Test GET, POST, and DELETE WordSet CRUD endpoints.",
                role="test_writer",
                depends_on=["wordset-crud-api"],
                acceptance_criteria=["WordSet CRUD backend tests pass"],
                target_files=["tests/backend.test.ts"],
                required_artifacts=["tests/backend.test.ts"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["uncovered_acceptance_tests"] == []
    assert [
        item["task_id"] for item in validation["acceptance_test_coverage"]
    ] == [
        "wordset-crud-api",
        "wordset-crud-api",
        "wordset-crud-api",
    ]


def test_task_graph_validation_rejects_crud_operation_mismatch() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need word set management.",
        smallest_working_core=["Manage word sets"],
        small_parts=["Implement backend API endpoints for WordSet CRUD"],
        incremental_milestones=["WordSet creation works"],
        acceptance_tests=["POST /api/word-sets creates a WordSet"],
        definition_of_done=["All tests pass"],
        required_artifacts=["src/server/index.ts", "tests/backend.test.ts"],
    )
    task_graph = TaskGraph(
        goal="Build WordSet create API",
        tasks=[
            PlannedTask(
                id="wordset-read-api",
                title="Implement GET WordSet API",
                description="Implement only the GET endpoint for WordSet listing.",
                role="implementer",
                acceptance_criteria=["GET /api/word-sets returns the WordSet list"],
                target_files=["src/server/index.ts"],
                required_artifacts=["src/server/index.ts"],
            ),
            PlannedTask(
                id="backend-read-tests",
                title="Test GET WordSet API",
                description="Test only the GET WordSet endpoint.",
                role="test_writer",
                depends_on=["wordset-read-api"],
                acceptance_criteria=["GET /api/word-sets returns the WordSet list"],
                target_files=["tests/backend.test.ts"],
                required_artifacts=["tests/backend.test.ts"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["uncovered_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "POST /api/word-sets creates a WordSet",
            "task_id": None,
            "covered": False,
        }
    ]
    assert validation["uncovered_test_writer_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "POST /api/word-sets creates a WordSet",
            "task_id": None,
            "covered": False,
        }
    ]
    error_types = {item["type"] for item in validation["errors"]}
    assert "semantic_acceptance_test_mismatch" in error_types
    assert "semantic_test_writer_acceptance_mismatch" in error_types


def test_semantic_task_coverage_chooses_anchor_satisfying_candidate() -> None:
    tasks = [
        PlannedTask(
            id="generic-wordset-get",
            title="Implement backend GET WordSet endpoint",
            description="Return WordSet data from the backend API endpoint.",
            role="implementer",
            acceptance_criteria=["GET /api/word-sets returns WordSet data."],
            target_files=["src/server/index.ts"],
            required_artifacts=["src/server/index.ts"],
        ),
        PlannedTask(
            id="wordset-crud-api",
            title="Implement WordSet CRUD API",
            description="Support the WordSet CRUD behavior.",
            role="implementer",
            acceptance_criteria=["WordSet CRUD works for GET requests."],
            target_files=["src/server/wordsets.ts"],
            required_artifacts=["src/server/wordsets.ts"],
        ),
    ]

    coverage = JobRunner._semantic_task_coverage(
        [
            "Backend API endpoints for WordSet CRUD are implemented; "
            "GET /api/word-sets returns the list of word sets"
        ],
        tasks,
        item_key="acceptance_test",
        index_key="acceptance_test_index",
        allow_reuse=True,
    )

    assert coverage == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": (
                "Backend API endpoints for WordSet CRUD are implemented; "
                "GET /api/word-sets returns the list of word sets"
            ),
            "task_id": "wordset-crud-api",
            "covered": True,
        }
    ]


def test_semantic_item_coverage_chooses_anchor_satisfying_candidate() -> None:
    coverage = JobRunner._semantic_item_coverage(
        ["Implement backend API endpoints for WordSet CRUD"],
        [
            "Backend API endpoint returns WordSet data",
            "WordSet CRUD API works",
        ],
        item_key="small_part",
        index_key="small_part_index",
        candidate_key="acceptance_test",
        candidate_index_key="acceptance_test_index",
    )

    assert coverage == [
        {
            "small_part_index": 1,
            "small_part": "Implement backend API endpoints for WordSet CRUD",
            "acceptance_test_index": 2,
            "acceptance_test": "WordSet CRUD API works",
            "covered": True,
        }
    ]


def test_task_graph_validation_rejects_invalid_artifact_paths() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Build the smallest feature.",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["../outside.py", "C:\\outside.py"],
                required_artifacts=["feature.py/"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["invalid_task_artifact_count"] == 1
    assert validation["invalid_task_artifacts"] == [
        {
            "task_id": "core",
            "paths": ["../outside.py", "C:\\outside.py", "feature.py/"],
        }
    ]
    error_types = {item["type"] for item in validation["errors"]}
    assert "missing_task_artifacts" in error_types
    assert "invalid_task_artifacts" in error_types


def test_task_graph_validation_requires_prd_artifacts_assigned_to_tasks() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["prd_required_artifact_count"] == 2
    assert validation["assigned_required_artifact_count"] == 1
    assert validation["unassigned_required_artifacts"] == ["tests/test_feature.py"]
    assert "unassigned_required_artifacts" in {
        item["type"] for item in validation["errors"]
    }


def test_missing_prd_implementation_artifact_is_appended_to_matching_task() -> None:
    tasks = [
        PlannedTask(
            id="backend-api",
            title="Implement backend API",
            description="Build backend server endpoints.",
            role="implementer",
            acceptance_criteria=["Backend API responds."],
            target_files=["src/server/index.ts"],
            required_artifacts=["src/server/index.ts"],
        ),
        PlannedTask(
            id="frontend-ui",
            title="Build frontend React UI",
            description="Render the client UI.",
            role="implementer",
            acceptance_criteria=["Frontend renders."],
            target_files=["src/client/App.tsx"],
            required_artifacts=["src/client/App.tsx"],
        ),
    ]

    updated_tasks, assignments = JobRunner._assign_missing_prd_implementation_artifacts(
        tasks,
        ["src/client/App.tsx", "src/client/main.tsx"],
    )

    assert assignments == [{"task_id": "frontend-ui", "path": "src/client/main.tsx"}]
    by_id = {task.id: task for task in updated_tasks}
    assert by_id["backend-api"].target_files == ["src/server/index.ts"]
    assert by_id["frontend-ui"].target_files == [
        "src/client/App.tsx",
        "src/client/main.tsx",
    ]
    assert by_id["frontend-ui"].required_artifacts == [
        "src/client/App.tsx",
        "src/client/main.tsx",
    ]


def test_task_graph_validation_rejects_invalid_prd_required_artifacts() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "../outside.py", "C:\\outside.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["invalid_prd_required_artifacts"] == [
        "../outside.py",
        "C:\\outside.py",
    ]
    assert "invalid_prd_required_artifacts" in {
        item["type"] for item in validation["errors"]
    }


def test_task_graph_validation_requires_owner_target_files_for_prd_artifacts() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused tests.",
                role="test_writer",
                acceptance_criteria=["VALUE is covered"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["assigned_required_artifact_count"] == 2
    assert validation["unassigned_required_artifacts"] == []
    assert validation["unowned_required_artifacts"] == [
        {"path": "feature.py", "expected_roles": ["implementer"]},
        {"path": "tests/test_feature.py", "expected_roles": ["test_writer"]},
    ]
    error_types = {item["type"] for item in validation["errors"]}
    assert "unowned_required_artifacts" in error_types
    assert "missing_test_writer_target_files" in error_types


def test_task_graph_validation_rejects_role_mismatched_target_files() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused tests.",
                role="test_writer",
                acceptance_criteria=["VALUE is covered"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["role_mismatched_target_files"] == [
        {
            "task_id": "core",
            "role": "implementer",
            "path": "tests/test_feature.py",
            "expected_roles": ["test_writer"],
        },
        {
            "task_id": "tests",
            "role": "test_writer",
            "path": "feature.py",
            "expected_roles": ["implementer"],
        },
    ]
    error_types = {item["type"] for item in validation["errors"]}
    assert "role_mismatched_target_files" in error_types
    assert "unowned_required_artifacts" in error_types


def test_task_graph_validation_rejects_scaffold_for_app_source_artifacts() -> None:
    prd = PRD(
        title="Auth API",
        problem_statement="Need authentication endpoints",
        smallest_working_core=["Expose login endpoint"],
        small_parts=["Create auth backend module"],
        incremental_milestones=["Auth module exists"],
        acceptance_tests=["Auth login endpoint validates credentials"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/auth.py", "tests/test_auth.py"],
    )
    task_graph = TaskGraph(
        goal="Build auth API",
        tasks=[
            PlannedTask(
                id="auth-scaffold",
                title="Auth backend scaffold",
                description="Create the auth backend module.",
                role="scaffold",
                acceptance_criteria=["Auth login endpoint validates credentials"],
                target_files=["backend/auth.py"],
                required_artifacts=["backend/auth.py"],
            ),
            PlannedTask(
                id="auth-tests",
                title="Test auth backend",
                description="Add focused auth backend tests.",
                role="test_writer",
                depends_on=["auth-scaffold"],
                acceptance_criteria=["Auth login endpoint is covered"],
                target_files=["tests/test_auth.py"],
                required_artifacts=["tests/test_auth.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["role_mismatched_target_files"] == [
        {
            "task_id": "auth-scaffold",
            "role": "scaffold",
            "path": "backend/auth.py",
            "expected_roles": ["implementer"],
        }
    ]
    assert validation["role_mismatched_required_artifacts"] == [
        {
            "task_id": "auth-scaffold",
            "role": "scaffold",
            "path": "backend/auth.py",
            "expected_roles": ["implementer"],
        }
    ]
    assert {
        "role_mismatched_target_files",
        "role_mismatched_required_artifacts",
    }.issubset({item["type"] for item in validation["errors"]})


def test_task_graph_validation_attempt_preserves_artifact_role_details(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
    )
    record = runner.store.create(
        JobSpec(request_text="Build feature", repo_path=str(workspace))
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        small_parts=["Create feature module"],
        acceptance_tests=["VALUE equals 1"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused tests",
                role="test_writer",
                acceptance_criteria=["VALUE is covered"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )
    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    runner._record_task_graph_validation_attempt(
        record,
        attempt=0,
        action="initial",
        validation=validation,
    )

    attempt = record.outputs["task_graph_validation_attempts"][0]
    assert attempt["role_mismatched_target_files"] == [
        {
            "task_id": "core",
            "role": "implementer",
            "path": "tests/test_feature.py",
            "expected_roles": ["test_writer"],
        },
        {
            "task_id": "tests",
            "role": "test_writer",
            "path": "feature.py",
            "expected_roles": ["implementer"],
        },
    ]
    assert attempt["unowned_required_artifacts"] == [
        {"path": "feature.py", "expected_roles": ["implementer"]},
        {"path": "tests/test_feature.py", "expected_roles": ["test_writer"]},
    ]
    assert (
        f"task_graph_validation_detail: role_mismatched_target_files="
        f"{attempt['role_mismatched_target_files']}"
    ) in JobRunner._task_graph_validation_repair_logs(prd, validation)


def test_task_graph_validation_rejects_role_mismatched_required_artifacts() -> None:
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create feature module",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Test core",
                description="Add focused tests.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["VALUE is covered"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert validation["role_mismatched_required_artifacts"] == [
        {
            "task_id": "core",
            "role": "implementer",
            "path": "tests/test_feature.py",
            "expected_roles": ["test_writer"],
        },
        {
            "task_id": "tests",
            "role": "test_writer",
            "path": "feature.py",
            "expected_roles": ["implementer"],
        },
    ]
    assert "role_mismatched_required_artifacts" in {
        item["type"] for item in validation["errors"]
    }


def test_task_graph_validation_allows_project_setup_test_artifact_on_scaffold() -> None:
    prd = PRD(
        title="Project setup",
        problem_statement="Need runnable starter app",
        smallest_working_core=["Create deterministic scaffold"],
        small_parts=["Create project scaffold"],
        incremental_milestones=["Scaffold files exist"],
        acceptance_tests=["Project setup smoke test exists"],
        definition_of_done=["All scaffold artifacts exist"],
        required_artifacts=[
            "backend/main.py",
            "backend/tests/test_project_setup.py",
        ],
    )
    task_graph = TaskGraph(
        goal="Create scaffold",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create deterministic starter files.",
                role="scaffold",
                acceptance_criteria=["Scaffold files exist"],
                target_files=[
                    "backend/main.py",
                    "backend/tests/test_project_setup.py",
                ],
                required_artifacts=[
                    "backend/main.py",
                    "backend/tests/test_project_setup.py",
                ],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is True
    assert validation["unowned_required_artifacts"] == []
    assert validation["role_mismatched_target_files"] == []
    assert validation["missing_test_writer_tasks"] is False
    assert validation["project_setup_scaffold_covers_test_artifacts"] is True


def test_job_runner_stops_when_implementation_reports_blocked(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.BLOCKED,
                summary="Need missing credentials before coding.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Should not run",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "def test_placeholder() -> None:\n    assert 'feature' in 'feature test'\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/implementation-blocked",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "implementation_blocked:core")
    assert record.outputs["implementation_tasks"][0]["result"]["status"] == "blocked"
    assert "test_writer_tasks" not in record.outputs
    assert "test_run" not in record.outputs
    assert not (workspace / "feature.py").exists()
    assert not (workspace / "tests" / "test_feature.py").exists()


def test_job_runner_fails_when_implementation_reports_failed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.FAILED,
                summary="Could not produce a coherent patch.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(summary="Should not run", patches=[]).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/implementation-failed",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.IMPLEMENTING, strategy="RETURN_TO_IMPLEMENTER")
    assert_recoverable_error(record, "implementation_failed:core")
    assert record.outputs["implementation_tasks"][0]["result"]["status"] == "failed"
    assert "test_writer_tasks" not in record.outputs
    assert "test_run" not in record.outputs
    assert not (workspace / "feature.py").exists()


def test_job_runner_stops_when_test_writer_reports_blocked(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                status="blocked",
                summary="Need acceptance criteria before writing useful tests.",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/test-writer-blocked",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.WRITING_TESTS,
        strategy="RETURN_TO_TEST_WRITER",
    )
    assert_recoverable_error(record, "test_writer_blocked:core")
    assert record.outputs["test_writer_tasks"][0]["result"]["status"] == "blocked"
    assert (workspace / "feature.py").exists()
    assert not (workspace / "tests" / "test_feature.py").exists()
    assert "test_run" not in record.outputs


def test_job_runner_fails_when_test_writer_reports_failed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                status="failed",
                summary="Could not produce a coherent test patch.",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/test-writer-failed",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.WRITING_TESTS,
        strategy="RETURN_TO_TEST_WRITER",
    )
    assert_recoverable_error(record, "test_writer_failed:core")
    assert record.outputs["test_writer_tasks"][0]["result"]["status"] == "failed"
    assert (workspace / "feature.py").exists()
    assert not (workspace / "tests" / "test_feature.py").exists()
    assert "test_run" not in record.outputs


def test_job_runner_records_completion_integrity_when_all_planned_tasks_finish(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/completion-integrity-pass",
        metadata={
            "constraints": {
                "require_completion_integrity": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["completion_integrity"] == {
        "passed": True,
        "failure_reasons": [],
        "require_completion_integrity": True,
        "require_test_evidence": False,
        "require_stage_test_patches": False,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 1,
        "stages_missing_test_patches": [],
        "failed_stages": [],
    }


def test_completion_integrity_fails_when_autonomous_stage_failed(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-failed-stage",
    )
    record = store.create(spec)
    record.completed_task_ids = ["core"]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "status": "failed_for_recovery",
            "failure_reason": "implementation_produced_no_changes",
            "task": {"id": "core", "role": "implementer"},
            "test_run": {"success": True},
        }
    ]
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    )

    report = JobRunner._build_completion_integrity_report(
        record,
        task_graph,
        TestRunResult(success=True, executed_test_count=1),
        require_completion_integrity=True,
        require_test_evidence=False,
        require_stage_test_patches=False,
    )

    assert report["passed"] is False
    assert report["failure_reasons"] == ["failed_stages:1"]
    assert report["failed_stages"] == [
        {
            "stage": 1,
            "task_id": "core",
            "failure_reason": "implementation_produced_no_changes",
        }
    ]


def test_completion_integrity_uses_passed_task_graph_for_artifact_evidence(
    tmp_path: Path,
) -> None:
    class SpyDefinitionOfDoneVerifier(DefinitionOfDoneVerifier):
        seen_task_graph: dict | None = None
        seen_test_run: dict | None = None

        def verify(self, record: JobRecord):
            self.seen_task_graph = record.outputs.get("task_graph")
            self.seen_test_run = record.outputs.get("test_run")
            return super().verify(record)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    verifier = SpyDefinitionOfDoneVerifier()
    runner.completion_verifier = verifier
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with required artifact",
            repo_path=str(workspace),
            target_branch="acos/completion-integrity-sync-task-graph",
            metadata={
                "constraints": {
                    "require_completion_integrity": True,
                    "require_test_evidence": True,
                }
            },
        )
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                target_files=["missing.py"],
                required_artifacts=["missing.py"],
            )
        ],
    )
    test_result = TestRunResult(success=True, executed_test_count=1)
    record.completed_task_ids = ["core"]
    record.audit_events.append({"event": "verified"})
    record.checkpoints.append({"kind": "stage"})

    passed = runner._validate_completion_integrity(record, task_graph, test_result)

    assert passed is False
    assert verifier.seen_task_graph == task_graph.model_dump()
    assert verifier.seen_test_run == test_result.model_dump()
    report = record.outputs["completion_integrity"]
    assert "required_artifact_missing:missing.py" in report["failure_reasons"]
    assert "target_file_missing:missing.py" in report["failure_reasons"]
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["required_artifacts"] == ["missing.py"]
    assert constraints["target_files"] == ["missing.py"]
    assert constraints["missing_artifacts"] == ["missing.py"]


def test_completion_integrity_fails_when_stage_test_run_failed_without_status(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-legacy-failed-stage",
    )
    record = store.create(spec)
    record.completed_task_ids = ["core"]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": {"id": "core", "role": "implementer"},
            "test_run": {"success": False},
        }
    ]
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    )

    report = JobRunner._build_completion_integrity_report(
        record,
        task_graph,
        TestRunResult(success=True, executed_test_count=1),
        require_completion_integrity=True,
        require_test_evidence=False,
        require_stage_test_patches=False,
    )

    assert report["passed"] is False
    assert report["failure_reasons"] == ["failed_stages:1"]
    assert report["failed_stages"] == [
        {
            "stage": 1,
            "task_id": "core",
            "failure_reason": "tests_failed",
        }
    ]


def test_completion_integrity_allows_superseded_failed_stage_after_later_pass(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-superseded-stage",
    )
    record = store.create(spec)
    record.completed_task_ids = ["core"]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": {"id": "core", "role": "implementer"},
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": {"id": "core", "role": "implementer"},
            "test_run": {"success": True},
        },
    ]
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    )

    report = JobRunner._build_completion_integrity_report(
        record,
        task_graph,
        TestRunResult(success=True, executed_test_count=1),
        require_completion_integrity=True,
        require_test_evidence=False,
        require_stage_test_patches=False,
    )

    assert report["passed"] is True
    assert report["failure_reasons"] == []
    assert report["failed_stages"] == []


def test_completion_integrity_report_records_failed_test_reason(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-test-failed",
    )
    record = store.create(spec)
    record.completed_task_ids = ["core"]
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    )

    report = JobRunner._build_completion_integrity_report(
        record,
        task_graph,
        TestRunResult(success=False, executed_test_count=1),
        require_completion_integrity=True,
        require_test_evidence=True,
        require_stage_test_patches=False,
    )
    runtime_state = JobRunner._completion_integrity_recovery_state(
        record,
        report["failure_reasons"],
    )

    assert report["passed"] is False
    assert report["failure_reasons"] == ["test_failed"]
    assert report["test_success"] is False
    assert runtime_state["completion_integrity_failure_reasons"] == ["test_failed"]


def test_completion_integrity_recovery_state_preserves_missing_task_ids(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-missing-task-context",
    )
    record = store.create(spec)

    runtime_state = JobRunner._completion_integrity_recovery_state(
        record,
        ["missing_tasks:core-tests|docs"],
    )

    assert runtime_state["missing_task_ids"] == ["core-tests", "docs"]
    assert runtime_state["completion_integrity_failure_reasons"] == [
        "missing_tasks:core-tests|docs"
    ]


def test_completion_integrity_recovery_state_preserves_stage_context(
    tmp_path: Path,
) -> None:
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(tmp_path),
        target_branch="acos/completion-integrity-stage-context",
    )
    record = store.create(spec)
    record.outputs["completion_integrity"] = {
        "failed_stages": [
            {
                "stage": 2,
                "task_id": "core",
                "failure_reason": "tests_failed",
            }
        ],
        "stages_missing_test_patches": [
            {
                "stage": 1,
                "task_id": "core",
                "implementation_patch_count": 1,
                "test_patch_count": 0,
            }
        ],
    }

    runtime_state = JobRunner._completion_integrity_recovery_state(
        record,
        ["missing_stage_test_patches:1", "failed_stages:2"],
    )

    assert runtime_state["completion_integrity_failure_reasons"] == [
        "missing_stage_test_patches:1",
        "failed_stages:2",
    ]
    assert runtime_state["missing_stage_test_patch_stage_ids"] == ["1"]
    assert runtime_state["failed_stage_ids"] == ["2"]
    assert runtime_state["stages_missing_test_patches"] == [
        {
            "stage": 1,
            "task_id": "core",
            "implementation_patch_count": 1,
            "test_patch_count": 0,
        }
    ]
    assert runtime_state["failed_stages"] == [
        {
            "stage": 2,
            "task_id": "core",
            "failure_reason": "tests_failed",
        }
    ]


def test_job_runner_blocks_completion_without_test_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(summary="No tests added", patches=[]).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
        scripted_test_results=[
            TestRunResult(
                success=True,
                command=["pytest"],
                output_excerpt="no tests ran in 0.01s",
                exit_code=0,
                executed_test_count=0,
            )
        ],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/test-evidence-block",
        metadata={
            "constraints": {
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.REPLANNING,
        strategy="REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
    )
    assert_recoverable_error(
        record,
        startswith="completion_integrity_failed:",
        contains="missing_test_evidence",
    )
    assert record.runtime_state["recovery_plan"]["strategy"] == (
        "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    )
    assert record.outputs["completion_integrity"]["passed"] is False
    assert "missing_test_evidence" in record.outputs["completion_integrity"]["failure_reasons"]
    assert record.outputs["completion_integrity"]["executed_test_count"] == 0


def test_job_runner_blocks_completion_when_implementation_stage_has_no_test_patch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Existing tests are enough",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
        scripted_test_results=[
            TestRunResult(
                success=True,
                command=["pytest"],
                output_excerpt="1 passed in 0.01s",
                exit_code=0,
                executed_test_count=1,
            )
        ],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stage-test-patch-evidence-block",
        metadata={
            "constraints": {
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "require_stage_test_patches": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.REPLANNING,
        strategy="REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
    )
    assert_recoverable_error(
        record,
        startswith="completion_integrity_failed:",
        contains="missing_stage_test_patches:1",
    )
    assert record.runtime_state["recovery_plan"]["strategy"] == (
        "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    )
    report = record.outputs["completion_integrity"]
    assert report["passed"] is False
    assert "missing_stage_test_patches:1" in report["failure_reasons"]
    assert report["executed_test_count"] == 1
    assert report["stages_missing_test_patches"] == [
        {
            "stage": 1,
            "task_id": "core",
            "implementation_patch_count": 1,
            "test_patch_count": 0,
        }
    ]


def test_completion_integrity_requires_test_path_patch_not_any_test_writer_patch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="Create module",
        patches=[
            {
                "path": "feature.py",
                "content": "VALUE = 1\n",
                "operation": "create",
            }
        ],
    )
    test_writer = TestWriterOutput(
        summary="Document existing test strategy",
        patches=[
            {
                "path": "README.md",
                "content": "# Feature\n\nTests still need to be written.\n",
                "operation": "create",
            }
        ],
    )
    summary = runner._build_stage_change_summary(implementation, [test_writer])
    assert summary["test_writer_patch_count"] == 1
    assert summary["test_patch_count"] == 0
    assert summary["test_files"] == []
    assert summary["test_writer_files"] == ["README.md"]

    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/non-test-writer-patch",
    )
    record = runner.store.create(spec)
    record.completed_task_ids = ["core"]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": {
                "id": "core",
                "role": "implementer",
            },
            "change_summary": summary,
            "test_run": {"success": True},
        }
    ]
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    )

    report = JobRunner._build_completion_integrity_report(
        record,
        task_graph,
        TestRunResult(success=True, executed_test_count=1),
        require_completion_integrity=True,
        require_test_evidence=True,
        require_stage_test_patches=True,
    )

    assert report["passed"] is False
    assert report["failure_reasons"] == ["missing_stage_test_patches:1"]
    assert report["stages_missing_test_patches"] == [
        {
            "stage": 1,
            "task_id": "core",
            "implementation_patch_count": 1,
            "test_patch_count": 0,
        }
    ]


def test_stage_change_summary_counts_case_insensitive_test_paths(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="Create module",
        patches=[
            {
                "path": "src/App.tsx",
                "content": "export default function App() { return null }\n",
                "operation": "create",
            }
        ],
    )
    test_writer = TestWriterOutput(
        summary="Add frontend spec",
        patches=[
            {
                "path": "Frontend/Test/App.Spec.tsx",
                "content": (
                    "import { expect, test } from 'vitest'\n\n"
                    "test('app spec path is tracked', () => {\n"
                    "  expect('Frontend/Test/App.Spec.tsx').toContain('Spec')\n"
                    "})\n"
                ),
                "operation": "create",
            }
        ],
    )

    summary = runner._build_stage_change_summary(implementation, [test_writer])

    assert summary["test_patch_count"] == 1
    assert summary["test_files"] == ["Frontend/Test/App.Spec.tsx"]


def test_job_runner_blocks_unsupported_autonomous_task_role_before_implementation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                    ),
                    PlannedTask(
                        id="release-notes",
                        title="Prepare release notes",
                        description="Document what changed after implementation.",
                        role="release_manager",
                        depends_on=["core"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/completion-integrity-block",
        metadata={
            "constraints": {
                "require_completion_integrity": True,
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["require_executable_task_roles"] is True
    assert validation["unsupported_task_role_count"] == 1
    assert validation["errors"] == [
        {
                "type": "unsupported_autonomous_task_roles",
                "items": [{"task_id": "release-notes", "role": "release_manager"}],
                "allowed_roles": ["implementer", "scaffold", "test_writer"],
            }
        ]
    assert "completion_integrity" not in record.outputs
    assert "test_run" not in record.outputs
    assert not (workspace / "feature.py").exists()


def test_job_runner_blocks_invalid_task_graph_before_implementation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build invalid graph",
                tasks=[
                    PlannedTask(
                        id="views",
                        title="Implement views",
                        description="Build views after models.",
                        role="implementer",
                        depends_on=["models"],
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/invalid-task-graph",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    assert record.outputs["task_graph_validation"]["valid"] is False
    assert record.outputs["task_graph_validation"]["errors"][0]["type"] == "unknown_dependencies"
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["task_graph_validation_errors"] == ["unknown_dependencies"]
    assert constraints["unknown_dependencies"] == [
        {"task_id": "views", "dependency": "models"}
    ]
    assert record.spec.metadata["constraints"]["task_graph_validation_errors"] == [
        "unknown_dependencies"
    ]
    assert not (workspace / "feature.py").exists()


def test_job_runner_blocks_test_writer_without_implementation_dependency(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                small_parts=["Create feature module"],
                acceptance_tests=["VALUE equals 1"],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature with ambiguous test ordering",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Create feature module.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                        target_files=["feature.py"],
                    ),
                    PlannedTask(
                        id="tests",
                        title="Test core",
                        description="Add tests for VALUE.",
                        role="test_writer",
                        acceptance_criteria=["VALUE is tested"],
                        target_files=["tests/test_feature.py"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Should not run",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "def test_placeholder() -> None:\n    assert 'feature' in 'feature test'\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/test-writer-dependency",
        metadata={
            "constraints": {
                "require_task_artifacts": True,
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["test_writer_missing_implementation_dependencies"] == [
        {
            "task_id": "tests",
            "depends_on": [],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert not (workspace / "feature.py").exists()
    assert not (workspace / "tests" / "test_feature.py").exists()


def test_job_runner_blocks_unverified_target_file_before_implementation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Create feature module.",
                        role="implementer",
                        acceptance_criteria=["VALUE equals 1"],
                        target_files=["feature.py", "unverified.py"],
                        required_artifacts=["feature.py"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature",
        repo_path=str(workspace),
        target_branch="acos/unverified-target",
        metadata={
            "constraints": {
                "require_task_artifacts": True,
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["target_files_missing_required_artifacts"] == [
        {"task_id": "core", "role": "implementer", "paths": ["unverified.py"]}
    ]
    assert not (workspace / "feature.py").exists()
    assert not (workspace / "unverified.py").exists()


def test_job_runner_blocks_executor_unsatisfiable_dependency_order_before_implementation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Create the core feature.",
                        role="implementer",
                        acceptance_criteria=["Core exists"],
                        target_files=["feature.py"],
                        required_artifacts=["feature.py"],
                    ),
                    PlannedTask(
                        id="extra",
                        title="Build extra",
                        description="Create extra behavior after later tests pass.",
                        role="implementer",
                        depends_on=["later-tests"],
                        acceptance_criteria=["Extra exists"],
                        target_files=["extra.py"],
                        required_artifacts=["extra.py"],
                    ),
                    PlannedTask(
                        id="later",
                        title="Build later",
                        description="Create later behavior.",
                        role="implementer",
                        acceptance_criteria=["Later exists"],
                        target_files=["later.py"],
                        required_artifacts=["later.py"],
                    ),
                    PlannedTask(
                        id="later-tests",
                        title="Test later",
                        description="Test later behavior.",
                        role="test_writer",
                        depends_on=["later"],
                        acceptance_criteria=["Later has tests"],
                        target_files=["tests/test_later.py"],
                        required_artifacts=["tests/test_later.py"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Should not run",
                patches=[
                    {
                        "path": "tests/test_later.py",
                        "content": "def test_placeholder() -> None:\n    assert 'feature' in 'feature test'\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/executor-order-dependency",
        metadata={
            "constraints": {
                "require_task_artifacts": True,
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "invalid_task_graph")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["executor_order_dependency_violations"] == [
        {
            "task_id": "extra",
            "role": "implementer",
            "executor_phase": "implementation",
            "unmet_dependencies": ["later-tests"],
            "dependency_roles": [
                {"task_id": "later-tests", "role": "test_writer"}
            ],
        }
    ]
    assert not (workspace / "feature.py").exists()
    assert not (workspace / "tests" / "test_later.py").exists()


def test_autonomous_loop_blocks_tail_test_writer_with_unmet_dependency(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "test_writer": TestWriterOutput(
                summary="Should not run before core exists.",
                patches=[
                    {
                        "path": "tests/test_core.py",
                        "content": (
                            "def test_core_exists() -> None:\n"
                            "    assert 'core' in 'core exists'\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
    )
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/tail-test-dependency-guard",
        )
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core-tests",
                title="Test core",
                description="Add tests for the core implementation.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior is tested"],
                target_files=["tests/test_core.py"],
                required_artifacts=["tests/test_core.py"],
            )
        ],
    )

    implementation_results, test_writer_results, test_result, stages = (
        runner._run_autonomous_task_loop(record, task_graph)
    )

    assert implementation_results == []
    assert test_writer_results == []
    assert stages == []
    assert test_result.success is True
    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "unmet_task_dependencies:core")
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["failed_task_id"] == "core-tests"
    assert constraints["unmet_dependencies"] == ["core"]
    assert "test_writer_tasks" not in record.outputs
    assert not (workspace / "tests" / "test_core.py").exists()


def test_autonomous_loop_blocks_recorded_implementation_with_unmet_dependency(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "test_writer": TestWriterOutput(
                summary="Should not validate a task before its dependency completes.",
                patches=[
                    {
                        "path": "tests/test_extra.py",
                        "content": (
                            "def test_extra_exists() -> None:\n"
                            "    assert 'extra' in 'extra exists'\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
    )
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/recorded-implementation-dependency-guard",
        )
    )
    task = PlannedTask(
        id="extra",
        title="Build extra",
        description="Build extra behavior after the core is complete.",
        role="implementer",
        depends_on=["core"],
        acceptance_criteria=["Extra behavior works"],
        target_files=["extra.py"],
        required_artifacts=["extra.py"],
    )
    record.outputs["implementation_tasks"] = [
        {
            "task": task.model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Previously recorded extra implementation.",
                changed_files=["extra.py"],
            ).model_dump(),
        }
    ]
    runner.store.update(record)
    task_graph = TaskGraph(goal="Build feature", tasks=[task])

    implementation_results, test_writer_results, test_result, stages = (
        runner._run_autonomous_task_loop(record, task_graph)
    )

    assert [item.summary for item in implementation_results] == [
        "Previously recorded extra implementation."
    ]
    assert test_writer_results == []
    assert stages == []
    assert test_result.success is True
    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "unmet_task_dependencies:core")
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["failed_task_id"] == "extra"
    assert constraints["unmet_dependencies"] == ["core"]
    assert "test_writer_tasks" not in record.outputs
    assert not (workspace / "tests" / "test_extra.py").exists()
    assert record.completed_task_ids == []


def test_autonomous_loop_blocks_inconsistent_completed_task_dependency(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "test_writer": TestWriterOutput(
                summary="Should not run for an inconsistent completed task.",
                patches=[
                    {
                        "path": "tests/test_extra.py",
                        "content": (
                            "def test_extra_exists() -> None:\n"
                            "    assert 'extra' in 'extra exists'\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
    )
    record = runner.store.create(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/completed-task-dependency-guard",
        )
    )
    task = PlannedTask(
        id="extra",
        title="Build extra",
        description="Build extra behavior after the core is complete.",
        role="implementer",
        depends_on=["core"],
        acceptance_criteria=["Extra behavior works"],
        target_files=["extra.py"],
        required_artifacts=["extra.py"],
    )
    record.completed_task_ids = ["extra"]
    runner.store.update(record)
    task_graph = TaskGraph(goal="Build feature", tasks=[task])

    implementation_results, test_writer_results, test_result, stages = (
        runner._run_autonomous_task_loop(record, task_graph)
    )

    assert implementation_results == []
    assert test_writer_results == []
    assert stages == []
    assert test_result.success is True
    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "unmet_task_dependencies:core")
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["failed_task_id"] == "extra"
    assert constraints["unmet_dependencies"] == ["core"]
    assert "test_writer_tasks" not in record.outputs
    assert not (workspace / "tests" / "test_extra.py").exists()


def test_job_runner_repairs_invalid_task_graph_before_implementation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": [
                TaskGraph(
                    goal="Build invalid graph",
                    tasks=[
                        PlannedTask(
                            id="views",
                            title="Implement views",
                            description="Build views after models.",
                            role="implementer",
                            depends_on=["models"],
                        )
                    ],
                ).model_dump(),
                TaskGraph(
                    goal="Build valid graph",
                    tasks=[
                        PlannedTask(
                            id="core",
                            title="Implement core",
                            description="Build the smallest working core.",
                            role="implementer",
                        )
                    ],
                ).model_dump(),
            ],
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/repair-task-graph",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["task_graph_validation"]["valid"] is True
    assert [
        attempt["valid"] for attempt in record.outputs["task_graph_validation_attempts"]
    ] == [False, True]
    assert record.outputs["task_graph_validation_attempts"][0]["errors"][0]["type"] == (
        "unknown_dependencies"
    )
    assert record.outputs["implementation_tasks"][0]["task"]["id"] == "core"
    assert (workspace / "feature.py").exists()


def test_job_runner_resumes_blocked_task_graph_repair(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    invalid_graph = TaskGraph(
        goal="Build invalid graph",
        tasks=[
            PlannedTask(
                id="views",
                title="Implement views",
                description="Build views after models.",
                role="implementer",
                depends_on=["models"],
            )
        ],
    ).model_dump()
    valid_graph = TaskGraph(
        goal="Build valid graph",
        tasks=[
            PlannedTask(
                id="core",
                title="Implement core",
                description="Build the smallest working core.",
                role="implementer",
            )
        ],
    ).model_dump()
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": [invalid_graph, invalid_graph, valid_graph],
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/resume-task-graph-repair",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    blocked = runner.run_job(spec)

    assert_recovery_plan(blocked, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(blocked, "invalid_task_graph")
    assert blocked.outputs["task_graph_validation"]["valid"] is False

    resumed = runner.resume_job(blocked.job_id)

    assert resumed.status == JobStatus.DONE
    assert resumed.last_error is None
    assert resumed.outputs["task_graph_validation"]["valid"] is True
    assert resumed.outputs["implementation_tasks"][0]["task"]["id"] == "core"
    assert JobStatus.PLANNING in resumed.history
    assert (workspace / "feature.py").exists()


def test_job_runner_repairs_task_graph_that_under_covers_prd_small_parts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need three helpers",
                smallest_working_core=["Create add_one helper"],
                small_parts=[
                    "Create add_one helper",
                    "Create double helper",
                    "Create triple helper",
                ],
                incremental_milestones=[
                    "add_one works",
                    "double works",
                    "triple works",
                ],
                acceptance_tests=[
                    "add_one(2) returns 3",
                    "double(4) returns 8",
                    "triple(3) returns 9",
                ],
                definition_of_done=["All generated tests pass"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": [
                TaskGraph(
                    goal="Build only two helpers",
                    tasks=[
                        PlannedTask(
                            id="add-one",
                            title="Create add_one helper",
                            description="Implement add_one.",
                            role="implementer",
                        ),
                        PlannedTask(
                            id="double",
                            title="Create double helper",
                            description="Implement double.",
                            role="implementer",
                            depends_on=["add-one"],
                        ),
                    ],
                ).model_dump(),
                TaskGraph(
                    goal="Build all helpers",
                    tasks=[
                        PlannedTask(
                            id="add-one",
                            title="Create add_one helper",
                            description="Implement add_one.",
                            role="implementer",
                        ),
                        PlannedTask(
                            id="double",
                            title="Create double helper",
                            description="Implement double.",
                            role="implementer",
                            depends_on=["add-one"],
                        ),
                        PlannedTask(
                            id="triple",
                            title="Create triple helper",
                            description="Implement triple.",
                            role="implementer",
                            depends_on=["double"],
                        ),
                    ],
                ).model_dump(),
            ],
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created add_one helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created double helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created triple helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n\n\n"
                                "def triple(value: int) -> int:\n"
                                "    return value * 3\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add add_one tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add double tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add triple tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double, triple\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n\n\n"
                                "def test_triple() -> None:\n"
                                "    assert triple(3) == 9\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/repair-undercovered-small-parts",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert [
        attempt["valid"] for attempt in record.outputs["task_graph_validation_attempts"]
    ] == [False, True]
    assert record.outputs["task_graph_validation_attempts"][0]["errors"][0]["type"] == (
        "undercovered_small_parts"
    )
    assert record.outputs["task_graph_validation_attempts"][0][
        "uncovered_small_parts"
    ] == [
        {
            "small_part_index": 3,
            "small_part": "Create triple helper",
            "task_id": None,
            "covered": False,
        }
    ]
    assert record.outputs["task_graph_validation"]["small_part_count"] == 3
    assert record.outputs["task_graph_validation"]["implementation_task_count"] == 3
    assert record.outputs["task_graph_validation"]["uncovered_small_parts"] == []
    assert record.outputs["task_graph_validation"]["small_part_coverage"] == [
        {
            "small_part_index": 1,
            "small_part": "Create add_one helper",
            "task_id": "add-one",
            "covered": True,
        },
        {
            "small_part_index": 2,
            "small_part": "Create double helper",
            "task_id": "double",
            "covered": True,
        },
        {
            "small_part_index": 3,
            "small_part": "Create triple helper",
            "task_id": "triple",
            "covered": True,
        },
    ]
    assert record.outputs["task_graph_validation"]["uncovered_acceptance_tests"] == []
    assert record.outputs["task_graph_validation"]["acceptance_test_coverage"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "add_one(2) returns 3",
            "task_id": "add-one",
            "covered": True,
        },
        {
            "acceptance_test_index": 2,
            "acceptance_test": "double(4) returns 8",
            "task_id": "double",
            "covered": True,
        },
        {
            "acceptance_test_index": 3,
            "acceptance_test": "triple(3) returns 9",
            "task_id": "triple",
            "covered": True,
        },
    ]
    assert [item["task"]["id"] for item in record.outputs["implementation_tasks"]] == [
        "add-one",
        "double",
        "triple",
    ]


def test_task_graph_validation_rejects_semantic_small_part_mismatch() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need account-based vocabulary practice.",
        smallest_working_core=["Health check"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=["Auth works", "Word sets work"],
        acceptance_tests=[
            "User can register and login",
            "Teacher can create a word set",
        ],
        definition_of_done=["All tests pass"],
    )
    task_graph = TaskGraph(
        goal="Build app",
        tasks=[
            PlannedTask(
                id="backend-health",
                title="Backend health endpoint",
                description="Create a FastAPI health endpoint.",
                role="implementer",
                acceptance_criteria=["Health endpoint returns 200"],
            ),
            PlannedTask(
                id="database-setup",
                title="Database setup",
                description="Create database configuration and migrations.",
                role="implementer",
                acceptance_criteria=["Database initializes"],
            ),
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_executable_task_roles=True,
    )

    assert validation["valid"] is False
    error_types = {item["type"] for item in validation["errors"]}
    assert "semantic_small_part_mismatch" in error_types
    assert "semantic_acceptance_test_mismatch" in error_types
    assert validation["uncovered_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "task_id": None,
            "covered": False,
        },
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "task_id": None,
            "covered": False,
        },
    ]


def test_task_graph_validation_requires_anchor_token_overlap() -> None:
    prd = PRD(
        title="Auth App",
        problem_statement="Users need secure account access.",
        smallest_working_core=["Serve an authenticated app shell"],
        small_parts=["User authentication and roles"],
        incremental_milestones=["Users can sign in"],
        acceptance_tests=["Student can register and login"],
        definition_of_done=["All tests pass"],
    )
    task_graph = TaskGraph(
        goal="Build auth app",
        tasks=[
            PlannedTask(
                id="profile-roles",
                title="User profile roles page",
                description="Render a user profile page with role labels.",
                role="implementer",
                acceptance_criteria=["User profile roles page renders"],
            )
        ],
    )

    validation = JobRunner._build_task_graph_validation(
        task_graph,
        prd=prd,
        require_acceptance_criteria=True,
        require_executable_task_roles=True,
    )

    assert validation["valid"] is False
    error_types = {item["type"] for item in validation["errors"]}
    assert "semantic_small_part_mismatch" in error_types
    assert "semantic_acceptance_test_mismatch" in error_types
    assert validation["uncovered_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "task_id": None,
            "covered": False,
        }
    ]


def test_job_runner_blocks_empty_task_graph_before_implementation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(goal="Build nothing", tasks=[]).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "SHOULD_NOT_EXIST = True\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/empty-task-graph",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["implementation_task_count"] == 0
    assert validation["errors"][0]["type"] == "empty_task_graph"
    assert not (workspace / "feature.py").exists()


def test_job_runner_blocks_task_graph_without_implementation_tasks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Only test",
                tasks=[
                    PlannedTask(
                        id="tests",
                        title="Write tests",
                        description="Write tests without an implementation task.",
                        role="test_writer",
                    )
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Should not run",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "def test_placeholder() -> None:\n    assert 'feature' in 'feature test'\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/no-implementation-task",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["implementation_task_count"] == 0
    assert validation["errors"][0]["type"] == "missing_implementation_tasks"
    assert not (workspace / "tests" / "test_feature.py").exists()


def test_job_runner_blocks_dependency_cycle_before_implementation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build cyclic graph",
                tasks=[
                    PlannedTask(
                        id="a",
                        title="A",
                        description="Build A",
                        role="implementer",
                        depends_on=["b"],
                    ),
                    PlannedTask(
                        id="b",
                        title="B",
                        description="Build B",
                        role="implementer",
                        depends_on=["a"],
                    ),
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Should not run",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/cyclic-task-graph",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "task_graph_validation_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    validation = record.outputs["task_graph_validation"]
    assert validation["valid"] is False
    assert validation["errors"][0]["type"] == "dependency_cycle"
    assert validation["errors"][0]["task_ids"] == ["a", "b", "a"]


def test_job_runner_blocks_before_exceeding_autonomous_stage_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="core-tests",
                        title="Test core helper",
                        description="Test the smallest working helper.",
                        role="test_writer",
                        depends_on=["core"],
                    ),
                    PlannedTask(
                        id="extra",
                        title="Add extra helper",
                        description="Add one more helper after the core passes.",
                        role="implementer",
                        depends_on=["core-tests"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created core helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Should not run",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value * 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": TestWriterOutput(
                summary="Add core tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stage-limit",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "max_autonomous_stages": 1,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["autonomous_stage_limit"]["max_autonomous_stages"] == 1
    assert record.outputs["autonomous_stage_limit"]["recovery_action"] == "auto_bump_stage_limit"
    assert len(record.outputs["autonomous_stages"]) >= 1
    assert "double" in (workspace / "feature.py").read_text(encoding="utf-8")


def test_job_runner_blocks_oversized_agent_patch_output_before_applying(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create helper files.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create too many files",
                patches=[
                    {
                        "path": "one.py",
                        "content": "ONE = 1\n",
                        "operation": "create",
                    },
                    {
                        "path": "two.py",
                        "content": "TWO = 2\n",
                        "operation": "create",
                    },
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/patch-limit",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "max_patches_per_agent_output": 1,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DIAGNOSING
    assert_recoverable_error(
        record,
        "quality_gate_recoverable:patch_limit_exceeded:implementer:2>1",
    )
    assert record.runtime_state["recovery_plan"]["strategy"] == "DIAGNOSE_FAILURE"
    assert not (workspace / "one.py").exists()
    assert not (workspace / "two.py").exists()


def test_job_runner_can_review_and_fix_each_stage_before_completion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add core tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": [
                ReviewResult(
                    decision=ReviewDecision.REQUEST_CHANGES,
                    summary="Add a docstring before moving on",
                    findings=[
                        {
                            "severity": Severity.LOW,
                            "title": "Missing docstring",
                            "description": "Document the helper before the next stage.",
                        }
                    ],
                ).model_dump(),
                ReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Stage is review-clean",
                ).model_dump(),
            ],
            "security_reviewer": [
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Safe",
                ).model_dump(),
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Still safe",
                ).model_dump(),
            ],
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Added docstring",
                patches=[
                    {
                        "path": "feature.py",
                        "content": (
                            "def add_one(value: int) -> int:\n"
                            "    \"\"\"Return value plus one.\"\"\"\n"
                            "    return value + 1\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stage-review-gate",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "stage_review": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    stage = record.outputs["autonomous_stages"][0]
    assert stage["stage_review"]["review"]["decision"] == "approve"
    assert stage["post_review_test_run"]["success"] is True
    assert record.completed_task_ids == ["core"]
    assert "Return value plus one." in (workspace / "feature.py").read_text(encoding="utf-8")


def test_job_runner_stops_stage_review_when_fixer_reports_failed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add core tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.REQUEST_CHANGES,
                summary="Add a docstring before moving on",
                findings=[
                    {
                        "severity": Severity.LOW,
                        "title": "Missing docstring",
                        "description": "Document the helper before the next stage.",
                    }
                ],
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
            ).model_dump(),
            "fixer": FixResult(
                status=FixStatus.FAILED,
                summary="Could not address stage review safely.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": (
                            "def add_one(value: int) -> int:\n"
                            "    \"\"\"Return value plus one.\"\"\"\n"
                            "    return value + 1\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stage-review-fixer-failed",
        metadata={
            "constraints": {
                "skip_review": True,
                "skip_release": True,
                "stage_review": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "fixer_failed:core")
    stage = record.outputs["autonomous_stages"][0]
    assert stage["stage_review"]["review"]["decision"] == "request_changes"
    assert "post_review_test_run" not in stage
    assert record.completed_task_ids == []
    assert '"""Return value plus one."""' not in (
        workspace / "feature.py"
    ).read_text(encoding="utf-8")


def test_job_runner_fails_review_cycle_without_applying_failed_fixer_patch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create helper.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add core tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.REQUEST_CHANGES,
                summary="Add a docstring before release",
                findings=[
                    {
                        "severity": Severity.LOW,
                        "title": "Missing docstring",
                        "description": "Document the helper before release.",
                    }
                ],
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
            ).model_dump(),
            "fixer": FixResult(
                status=FixStatus.FAILED,
                summary="Could not address review safely.",
                patches=[
                    {
                        "path": "feature.py",
                        "content": (
                            "def add_one(value: int) -> int:\n"
                            "    \"\"\"Return value plus one.\"\"\"\n"
                            "    return value + 1\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/review-fixer-failed",
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert_recoverable_error(record, "fixer_failed:core")
    assert '"""Return value plus one."""' not in (
        workspace / "feature.py"
    ).read_text(encoding="utf-8")


def test_job_runner_persists_autonomous_checkpoints_when_later_stage_gets_stuck(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature incrementally",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Create core helper",
                        description="Create the smallest working helper.",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="core-tests",
                        title="Test core helper",
                        description="Test the smallest working helper.",
                        role="test_writer",
                        depends_on=["core"],
                    ),
                    PlannedTask(
                        id="extra",
                        title="Add extra helper",
                        description="Add one more helper after the core passes.",
                        role="implementer",
                        depends_on=["core-tests"],
                    ),
                    PlannedTask(
                        id="extra-tests",
                        title="Test extra helper",
                        description="Test the added helper.",
                        role="test_writer",
                        depends_on=["extra"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Created core helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": "def add_one(value: int) -> int:\n    return value + 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Added buggy extra helper",
                    patches=[
                        {
                            "path": "feature.py",
                            "content": (
                                "def add_one(value: int) -> int:\n"
                                "    return value + 1\n\n\n"
                                "def double(value: int) -> int:\n"
                                "    return value + 2\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add core tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n"
                            ),
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add extra tests",
                    patches=[
                        {
                            "path": "tests/test_feature.py",
                            "content": (
                                "from feature import add_one, double\n\n\n"
                                "def test_add_one() -> None:\n"
                                "    assert add_one(2) == 3\n\n\n"
                                "def test_double() -> None:\n"
                                "    assert double(4) == 8\n"
                            ),
                            "operation": "update",
                        }
                    ],
                ).model_dump(),
            ],
            "fixer": FixResult(
                status=FixStatus.STUCK,
                summary="Cannot fix this stage",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/autonomous-stage-checkpoints",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(record, status=JobStatus.REPLANNING, strategy="REPLAN_TASK")
    assert record.completed_task_ids == ["core", "core-tests"]
    assert len(record.outputs["autonomous_stages"]) == 2
    assert record.outputs["autonomous_stages"][0]["test_run"]["success"] is True
    assert record.outputs["autonomous_stages"][1]["test_run"]["success"] is False
    stage_checkpoints = [
        checkpoint for checkpoint in record.checkpoints if "task_id" in checkpoint
    ]
    assert [checkpoint["task_id"] for checkpoint in stage_checkpoints] == ["core", "extra"]
    assert [checkpoint["test_success"] for checkpoint in stage_checkpoints] == [True, False]


def test_job_runner_resume_revalidates_unfinished_failed_stage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "feature.py").write_text(
        "def add_one(value: int) -> int:\n"
        "    return value + 1\n\n\n"
        "def double(value: int) -> int:\n"
        "    return value + 2\n",
        encoding="utf-8",
    )
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_feature.py").write_text(
        "from feature import add_one, double\n\n\n"
        "def test_add_one() -> None:\n"
        "    assert add_one(2) == 3\n\n\n"
        "def test_double() -> None:\n"
        "    assert double(4) == 8\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    task_graph = TaskGraph(
        goal="Build feature incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Create core helper",
                description="Create the smallest working helper.",
                role="implementer",
            ),
            PlannedTask(
                id="core-tests",
                title="Test core helper",
                description="Test the smallest working helper.",
                role="test_writer",
                depends_on=["core"],
            ),
            PlannedTask(
                id="extra",
                title="Add extra helper",
                description="Add one more helper after the core passes.",
                role="implementer",
                depends_on=["core-tests"],
            ),
            PlannedTask(
                id="extra-tests",
                title="Test extra helper",
                description="Test the added helper.",
                role="test_writer",
                depends_on=["extra"],
            ),
        ],
    )
    attach_mock_adapter(
        registry,
        {
            "test_writer": TestWriterOutput(
                summary="Refresh extra tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one, double\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n\n\n"
                            "def test_double() -> None:\n"
                            "    assert double(4) == 8\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Fix double helper",
                patches=[
                    {
                        "path": "feature.py",
                        "content": (
                            "def add_one(value: int) -> int:\n"
                            "    return value + 1\n\n\n"
                            "def double(value: int) -> int:\n"
                            "    return value * 2\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    store = InMemoryJobStore()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    spec = JobSpec(
        job_id="resume-failed-stage",
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/resume-failed-stage",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.outputs["pm"] = PRD(title="Feature", problem_statement="Need feature").model_dump()
    record.outputs["architecture"] = ArchitecturePlan(summary="Simple architecture").model_dump()
    record.outputs["planner"] = task_graph.model_dump()
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["implementation_tasks"] = [
        {
            "task": task_graph.tasks[0].model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper",
            ).model_dump(),
        },
        {
            "task": task_graph.tasks[2].model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Added buggy extra helper",
            ).model_dump(),
        },
    ]
    record.outputs["test_writer_tasks"] = [
        {
            "task": task_graph.tasks[1].model_dump(),
            "result": TestWriterOutput(summary="Add core tests").model_dump(),
        },
        {
            "task": task_graph.tasks[3].model_dump(),
            "result": TestWriterOutput(summary="Add extra tests").model_dump(),
        },
    ]
    record.completed_task_ids = ["core", "core-tests"]
    store.update(record)

    resumed = runner.resume_job("resume-failed-stage")

    assert resumed.status.value == "done"
    assert resumed.completed_task_ids == ["core", "core-tests", "extra", "extra-tests"]
    assert resumed.outputs["test_run"]["success"] is True
    assert "return value * 2" in (workspace / "feature.py").read_text(encoding="utf-8")


def test_job_runner_resume_retries_failed_implementation_record(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    task_graph = TaskGraph(
        goal="Build feature incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Create core helper",
                description="Create the smallest working helper.",
                role="implementer",
            )
        ],
    )
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper on retry",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add core tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    store = InMemoryJobStore()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    spec = JobSpec(
        job_id="resume-failed-implementation",
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/resume-failed-implementation",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )
    record = store.create(spec)
    record.status = JobStatus.FAILED
    record.last_error = "implementation_failed:core"
    record.outputs["pm"] = PRD(title="Feature", problem_statement="Need feature").model_dump()
    record.outputs["architecture"] = ArchitecturePlan(summary="Simple architecture").model_dump()
    record.outputs["planner"] = task_graph.model_dump()
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["implementation_tasks"] = [
        {
            "task": task_graph.tasks[0].model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.FAILED,
                summary="Previous implementation failed",
            ).model_dump(),
        }
    ]
    store.update(record)

    resumed = runner.resume_job("resume-failed-implementation")

    assert resumed.status == JobStatus.DONE
    assert resumed.last_error is None
    assert len(resumed.outputs["implementation_tasks"]) == 2
    assert resumed.outputs["implementation_tasks"][1]["result"]["status"] == "implemented"
    assert resumed.outputs["implementation"]["status"] == "implemented"
    assert resumed.completed_task_ids == ["core"]
    assert (workspace / "feature.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_job_runner_resume_retries_failed_test_writer_record(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    task_graph = TaskGraph(
        goal="Build feature incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Create core helper",
                description="Create the smallest working helper.",
                role="implementer",
            ),
            PlannedTask(
                id="core-tests",
                title="Test core helper",
                description="Test the smallest working helper.",
                role="test_writer",
                depends_on=["core"],
            ),
        ],
    )
    attach_mock_adapter(
        registry,
        {
            "test_writer": TestWriterOutput(
                status=WriterStatus.TESTS_WRITTEN,
                summary="Added core tests on retry",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    store = InMemoryJobStore()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    spec = JobSpec(
        job_id="resume-failed-test-writer",
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/resume-failed-test-writer",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )
    record = store.create(spec)
    record.status = JobStatus.FAILED
    record.last_error = "test_writer_failed:core-tests"
    record.outputs["pm"] = PRD(title="Feature", problem_statement="Need feature").model_dump()
    record.outputs["architecture"] = ArchitecturePlan(summary="Simple architecture").model_dump()
    record.outputs["planner"] = task_graph.model_dump()
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["implementation_tasks"] = [
        {
            "task": task_graph.tasks[0].model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Previous implementation succeeded",
                changed_files=["feature.py"],
            ).model_dump(),
        }
    ]
    record.outputs["test_writer_tasks"] = [
        {
            "task": task_graph.tasks[1].model_dump(),
            "result": TestWriterOutput(
                status=WriterStatus.FAILED,
                summary="Previous test writer failed",
            ).model_dump(),
        }
    ]
    store.update(record)

    resumed = runner.resume_job("resume-failed-test-writer")

    assert resumed.status == JobStatus.DONE
    assert resumed.last_error is None
    assert len(resumed.outputs["test_writer_tasks"]) == 2
    assert resumed.outputs["test_writer_tasks"][0]["result"]["status"] == "failed"
    assert resumed.outputs["test_writer_tasks"][1]["result"]["status"] == "tests_written"
    assert resumed.outputs["test_writer"]["status"] == "tests_written"
    assert resumed.completed_task_ids == ["core", "core-tests"]
    assert (workspace / "tests" / "test_feature.py").exists()


def test_job_runner_resumes_from_completed_autonomous_tasks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "feature.py").write_text(
        "def add_one(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_feature.py").write_text(
        "from feature import add_one\n\n\n"
        "def test_add_one() -> None:\n"
        "    assert add_one(2) == 3\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    task_graph = TaskGraph(
        goal="Build feature incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Create core helper",
                description="Create the smallest working helper.",
                role="implementer",
            ),
            PlannedTask(
                id="core-tests",
                title="Test core helper",
                description="Test the smallest working helper.",
                role="test_writer",
                depends_on=["core"],
            ),
            PlannedTask(
                id="extra",
                title="Add extra helper",
                description="Add one more helper after the core passes.",
                role="implementer",
                depends_on=["core-tests"],
            ),
            PlannedTask(
                id="extra-tests",
                title="Test extra helper",
                description="Test the added helper.",
                role="test_writer",
                depends_on=["extra"],
            ),
        ],
    )
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Added extra helper",
                patches=[
                    {
                        "path": "feature.py",
                        "content": (
                            "def add_one(value: int) -> int:\n"
                            "    return value + 1\n\n\n"
                            "def double(value: int) -> int:\n"
                            "    return value * 2\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add extra tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import add_one, double\n\n\n"
                            "def test_add_one() -> None:\n"
                            "    assert add_one(2) == 3\n\n\n"
                            "def test_double() -> None:\n"
                            "    assert double(4) == 8\n"
                        ),
                        "operation": "update",
                    }
                ],
            ).model_dump(),
        },
    )
    store = InMemoryJobStore()
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    spec = JobSpec(
        job_id="resume-me",
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/resume-stage-test",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )
    record = store.create(spec)
    record.status = JobStatus.TESTING
    record.outputs["pm"] = PRD(title="Feature", problem_statement="Need feature").model_dump()
    record.outputs["architecture"] = ArchitecturePlan(
        summary="Simple architecture",
    ).model_dump()
    record.outputs["planner"] = task_graph.model_dump()
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["implementation_tasks"] = [
        {
            "task": task_graph.tasks[0].model_dump(),
            "result": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created core helper",
            ).model_dump(),
        }
    ]
    record.outputs["test_writer_tasks"] = [
        {
            "task": task_graph.tasks[1].model_dump(),
            "result": TestWriterOutput(summary="Add core tests").model_dump(),
        }
    ]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "implementation": record.outputs["implementation_tasks"][0]["result"],
            "test_writer_results": [record.outputs["test_writer_tasks"][0]["result"]],
            "test_run": {
                "success": True,
                "command": ["pytest"],
                "failed_tests": [],
                "output_excerpt": "passed",
                "exit_code": 0,
            },
        }
    ]
    record.completed_task_ids = ["core", "core-tests"]
    store.update(record)

    resumed = runner.resume_job("resume-me")

    assert resumed.status.value == "done"
    assert resumed.completed_task_ids == ["core", "core-tests", "extra", "extra-tests"]
    assert resumed.outputs["implementation_task_count"] == 2
    assert resumed.outputs["test_writer_task_count"] == 2
    assert len(resumed.outputs["autonomous_stages"]) == 2
    assert "def double" in (workspace / "feature.py").read_text(encoding="utf-8")


def test_job_runner_records_prd_quality_report_for_sparse_prd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/prd-quality-report",
        metadata={"constraints": {"skip_review": True, "skip_release": True}},
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["prd_quality"]["passed"] is False
    assert record.outputs["prd_quality"]["missing"] == [
        "smallest_working_core",
        "small_parts",
        "incremental_milestones",
        "acceptance_tests",
        "definition_of_done",
        "required_artifacts",
    ]


def test_job_runner_blocks_when_strict_prd_quality_required(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a complex feature",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-quality",
        metadata={"constraints": {"require_prd_quality": True}},
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:"
        "smallest_working_core,small_parts,incremental_milestones,"
        "acceptance_tests,definition_of_done,required_artifacts",
    )
    assert record.outputs["prd_quality"]["passed"] is False
    assert "architect" not in record.outputs


def test_job_runner_blocks_prd_quality_when_open_questions_remain(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a feature module"],
                small_parts=["Create feature module", "Add focused tests"],
                incremental_milestones=["Module exists", "Tests exist"],
                acceptance_tests=["Feature module exists", "Focused tests exist"],
                definition_of_done=["All tests pass"],
                required_artifacts=["feature.py", "tests/test_feature.py"],
                open_questions=["Which persistence backend should be used?"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a feature with unresolved PM questions",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-open-questions",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:open_questions_resolved",
    )
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == ["open_questions_resolved"]
    assert constraints["prd_quality_warnings"] == ["open_questions_present"]
    assert constraints["prd_open_questions"] == [
        "Which persistence backend should be used?"
    ]
    assert record.spec.metadata["constraints"]["prd_quality_missing"] == [
        "open_questions_resolved"
    ]
    assert record.outputs["prd_quality"]["passed"] is False
    assert record.outputs["prd_quality"]["missing"] == ["open_questions_resolved"]
    assert record.outputs["prd_quality"]["warnings"] == ["open_questions_present"]
    assert "architect" not in record.outputs


def test_job_runner_blocks_prd_quality_when_source_required_artifact_missing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a feature module"],
                small_parts=["Create feature module"],
                incremental_milestones=["Module exists"],
                acceptance_tests=["Feature module exists"],
                definition_of_done=["All tests pass"],
                required_artifacts=["tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a feature with tests",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-source-artifact",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:required_source_artifacts",
    )
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == ["required_source_artifacts"]
    assert constraints["prd_required_artifacts"] == ["tests/test_feature.py"]
    assert constraints["test_required_artifacts"] == ["tests/test_feature.py"]
    assert "source_required_artifacts" not in constraints
    assert record.spec.metadata["constraints"]["prd_quality_missing"] == [
        "required_source_artifacts"
    ]
    assert record.outputs["prd_quality"]["passed"] is False
    assert record.outputs["prd_quality"]["missing"] == ["required_source_artifacts"]
    assert record.outputs["prd_quality"]["source_required_artifacts"] == []
    assert record.outputs["prd_quality"]["test_required_artifacts"] == [
        "tests/test_feature.py"
    ]
    assert "architect" not in record.outputs


def test_large_autonomous_prd_quality_requires_split_small_parts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a feature module"],
                small_parts=["Create feature module"],
                incremental_milestones=["Module exists"],
                acceptance_tests=["Feature module exists"],
                definition_of_done=["All tests pass"],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a feature as a large autonomous run",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-split-small-parts",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "max_autonomous_stages": 1,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:small_parts_split_for_autonomy",
    )
    report = record.outputs["prd_quality"]
    assert report["passed"] is False
    assert report["missing"] == ["small_parts_split_for_autonomy"]
    assert report["warnings"] == ["small_parts_has_single_item"]
    assert report["required_small_part_count"] == 2
    assert report["small_parts_split_for_autonomy"] is False
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == ["small_parts_split_for_autonomy"]
    assert constraints["required_small_part_count"] == 2
    assert constraints["implementation_required_artifacts"] == ["feature.py"]
    assert "architect" not in record.outputs


def test_job_runner_blocks_prd_quality_when_acceptance_tests_are_not_observable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a feature module"],
                small_parts=["Create backend feature module"],
                incremental_milestones=["Module exists"],
                acceptance_tests=["Create backend feature module"],
                definition_of_done=["All tests pass"],
                required_artifacts=["backend/main.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a feature with executable acceptance tests",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-observable-acceptance",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:acceptance_tests_observable",
    )
    report = record.outputs["prd_quality"]
    assert report["missing"] == ["acceptance_tests_observable"]
    assert report["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "Create backend feature module",
        }
    ]
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == ["acceptance_tests_observable"]
    assert constraints["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "Create backend feature module",
        }
    ]
    assert "architect" not in record.outputs


def test_job_runner_blocks_prd_quality_when_incremental_milestones_do_not_cover_parts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a feature module"],
                small_parts=["Create feature module", "Add focused tests"],
                incremental_milestones=["Feature module exists"],
                acceptance_tests=["Feature module exists", "Focused tests exist"],
                definition_of_done=["All tests pass"],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a feature with incremental milestones",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-milestone-coverage",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:incremental_milestones_cover_small_parts",
    )
    report = record.outputs["prd_quality"]
    assert report["missing"] == ["incremental_milestones_cover_small_parts"]
    assert report["incremental_milestone_count"] == 1
    assert report["missing_incremental_milestone_count"] == 1
    assert report["required_incremental_milestone_count"] == 2
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == [
        "incremental_milestones_cover_small_parts"
    ]
    assert constraints["required_incremental_milestone_count"] == 2
    assert "architect" not in record.outputs


def test_job_runner_refines_sparse_prd_before_implementation_when_required(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PRD(
                    title="Feature",
                    problem_statement="Need feature",
                    smallest_working_core=["Expose a VALUE constant and test it"],
                    small_parts=["Create feature module", "Add focused tests"],
                    incremental_milestones=[
                        "Smallest module exists",
                        "Tests prove observable behavior",
                    ],
                    acceptance_tests=[
                        "VALUE equals 1",
                        "test_value asserts VALUE equals 1",
                    ],
                    definition_of_done=["All tests pass"],
                    required_artifacts=["feature.py", "tests/test_feature.py"],
                ).model_dump(),
            ],
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a complex feature",
        repo_path=str(workspace),
        target_branch="acos/refine-prd-quality",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.DONE
    assert record.outputs["prd_quality"]["passed"] is True
    assert [
        attempt["passed"] for attempt in record.outputs["prd_quality_attempts"]
    ] == [False, True]
    assert record.outputs["prd"]["small_parts"] == [
        "Create feature module",
        "Add focused tests",
    ]
    assert (workspace / "feature.py").exists()


def test_job_runner_blocks_prd_quality_when_acceptance_tests_do_not_cover_small_parts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Expose a VALUE constant and test it"],
                small_parts=["Create feature module", "Add focused tests"],
                incremental_milestones=[
                    "Smallest module exists",
                    "Tests prove observable behavior",
                ],
                acceptance_tests=["VALUE equals 1"],
                definition_of_done=["All tests pass"],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a complex feature",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-acceptance-coverage",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:acceptance_tests_cover_small_parts",
    )
    assert record.outputs["prd_quality"] == {
        "passed": False,
        "missing": ["acceptance_tests_cover_small_parts"],
        "warnings": [],
        "small_part_count": 2,
        "smallest_working_core_covered_by_small_parts": True,
        "smallest_working_core_coverage": [
            {
                "core_index": 1,
                "smallest_working_core": "Expose a VALUE constant and test it",
                "required_anchor_tokens": [],
                "covered_anchor_tokens": [],
                "missing_anchor_tokens": [],
                "covered": True,
            }
        ],
        "uncovered_smallest_working_core": [],
        "incremental_milestone_count": 2,
        "incremental_milestones_cover_small_parts": True,
        "incremental_milestones_semantically_cover_small_parts": True,
        "incremental_milestone_small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Create feature module",
                "required_anchor_tokens": [],
                "covered_anchor_tokens": [],
                "missing_anchor_tokens": [],
                "covered": True,
            },
            {
                "small_part_index": 2,
                "small_part": "Add focused tests",
                "required_anchor_tokens": [],
                "covered_anchor_tokens": [],
                "missing_anchor_tokens": [],
                "covered": True,
            },
        ],
        "uncovered_incremental_milestone_small_parts": [],
        "missing_incremental_milestone_count": 0,
        "required_incremental_milestone_count": 2,
        "acceptance_test_count": 1,
        "acceptance_tests_cover_small_parts": False,
        "missing_acceptance_test_count": 1,
        "acceptance_tests_semantically_cover_small_parts": False,
        "acceptance_tests_observable": True,
        "acceptance_test_small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Create feature module",
                "acceptance_test_index": 1,
                "acceptance_test": "VALUE equals 1",
                "covered": True,
            },
            {
                "small_part_index": 2,
                "small_part": "Add focused tests",
                "acceptance_test_index": None,
                "acceptance_test": None,
                "covered": False,
            },
        ],
        "uncovered_acceptance_small_parts": [
            {
                "small_part_index": 2,
                "small_part": "Add focused tests",
                "acceptance_test_index": None,
                "acceptance_test": None,
                "covered": False,
            }
        ],
        "non_observable_acceptance_tests": [],
        "definition_of_done_count": 1,
        "required_artifact_count": 2,
        "required_artifacts": ["feature.py", "tests/test_feature.py"],
        "source_required_artifact_count": 1,
        "source_required_artifacts": ["feature.py"],
        "implementation_required_artifact_count": 1,
        "implementation_required_artifacts": ["feature.py"],
        "implementation_artifacts_cover_small_parts": True,
        "implementation_artifact_small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Create feature module",
                "required_surfaces": ["implementation"],
                "covered_surfaces": ["implementation"],
                "missing_surfaces": [],
                "implementation_artifacts": ["feature.py"],
                "covered": True,
            }
        ],
        "uncovered_implementation_artifact_small_parts": [],
        "implementation_artifacts_semantically_cover_small_parts": True,
        "implementation_artifact_domain_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Create feature module",
                "required_anchor_tokens": [],
                "required_domain_tokens": [],
                "covered_anchor_tokens": [],
                "covered_domain_tokens": [],
                "missing_anchor_tokens": [],
                "implementation_artifacts": [],
                "covered": True,
            }
        ],
        "uncovered_implementation_artifact_domain_small_parts": [],
        "test_required_artifact_count": 1,
        "test_required_artifacts": ["tests/test_feature.py"],
        "test_artifacts_semantically_cover_small_parts": True,
        "test_artifact_domain_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Create feature module",
                "required_anchor_tokens": [],
                "required_domain_tokens": [],
                "covered_anchor_tokens": [],
                "covered_domain_tokens": [],
                "missing_anchor_tokens": [],
                "test_artifacts": [],
                "covered": True,
            }
        ],
        "uncovered_test_artifact_domain_small_parts": [],
        "invalid_required_artifacts": [],
    }
    assert "architect" not in record.outputs


def test_prd_quality_requires_test_artifact_when_acceptance_tests_exist() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Feature module exists"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["required_test_artifacts"]
    assert report["required_artifact_count"] == 1
    assert report["required_artifacts"] == ["feature.py"]
    assert report["source_required_artifact_count"] == 1
    assert report["source_required_artifacts"] == ["feature.py"]
    assert report["implementation_required_artifact_count"] == 1
    assert report["implementation_required_artifacts"] == ["feature.py"]
    assert report["test_required_artifact_count"] == 0
    assert report["test_required_artifacts"] == []


def test_prd_quality_requires_source_artifact_when_acceptance_tests_exist() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Feature module exists"],
        definition_of_done=["All tests pass"],
        required_artifacts=["tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["required_source_artifacts"]
    assert report["required_artifact_count"] == 1
    assert report["required_artifacts"] == ["tests/test_feature.py"]
    assert report["source_required_artifact_count"] == 0
    assert report["source_required_artifacts"] == []
    assert report["implementation_required_artifact_count"] == 0
    assert report["implementation_required_artifacts"] == []
    assert report["test_required_artifact_count"] == 1
    assert report["test_required_artifacts"] == ["tests/test_feature.py"]


def test_prd_quality_requires_implementation_source_for_app_work() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create backend feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Backend feature module exists"],
        definition_of_done=["All tests pass"],
        required_artifacts=["README.md", "package.json", "tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["required_implementation_artifacts"]
    assert report["source_required_artifacts"] == ["README.md", "package.json"]
    assert report["implementation_required_artifacts"] == []
    assert report["test_required_artifacts"] == ["tests/test_feature.py"]


def test_prd_quality_requires_implementation_artifacts_for_each_surface() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=[
            "Create backend API endpoints",
            "Create frontend UI component",
        ],
        incremental_milestones=[
            "Backend API endpoints exist",
            "Frontend UI component exists",
        ],
        acceptance_tests=[
            "Backend API endpoints return HTTP responses",
            "Frontend UI component renders vocabulary form",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/main.py",
            "tests/test_project_setup.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["implementation_artifacts_cover_small_parts"]
    assert report["implementation_required_artifacts"] == ["backend/main.py"]
    assert report["implementation_artifacts_cover_small_parts"] is False
    assert report["uncovered_implementation_artifact_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Create frontend UI component",
            "required_surfaces": ["frontend"],
            "covered_surfaces": [],
            "missing_surfaces": ["frontend"],
            "implementation_artifacts": [],
            "covered": False,
        }
    ]


def test_prd_quality_accepts_required_artifacts_covering_backend_and_frontend() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=[
            "Create backend API endpoints",
            "Create frontend UI component",
        ],
        incremental_milestones=[
            "Backend API endpoints exist",
            "Frontend UI component exists",
        ],
        acceptance_tests=[
            "Backend API endpoints return HTTP responses",
            "Frontend UI component renders vocabulary form",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/word_sets.py",
            "frontend/src/App.tsx",
            "tests/test_project_setup.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["implementation_artifacts_cover_small_parts"] is True
    assert report["uncovered_implementation_artifact_small_parts"] == []


def test_prd_quality_rejects_generic_implementation_artifacts_missing_domain_tokens() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Word set CRUD operations work",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/main.py",
            "backend/tests/test_auth.py",
            "backend/tests/test_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == [
        "implementation_artifacts_semantically_cover_small_parts"
    ]
    assert report["implementation_artifacts_semantically_cover_small_parts"] is False
    assert report["uncovered_implementation_artifact_domain_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "required_anchor_tokens": ["auth"],
            "required_domain_tokens": ["auth", "role", "user"],
            "covered_anchor_tokens": [],
            "covered_domain_tokens": [],
            "missing_anchor_tokens": ["auth"],
            "implementation_artifacts": [],
            "covered": False,
        },
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "required_anchor_tokens": ["crud"],
            "required_domain_tokens": ["crud", "set", "word"],
            "covered_anchor_tokens": [],
            "covered_domain_tokens": [],
            "missing_anchor_tokens": ["crud"],
            "implementation_artifacts": [],
            "covered": False,
        },
    ]


def test_prd_quality_accepts_domain_specific_implementation_artifacts() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Word set CRUD operations work",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "backend/tests/test_auth.py",
            "backend/tests/test_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["implementation_artifacts_semantically_cover_small_parts"] is True
    assert report["uncovered_implementation_artifact_domain_small_parts"] == []


def test_prd_quality_rejects_generic_test_artifacts_missing_domain_tokens() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Word set CRUD operations work",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "backend/tests/test_project_setup.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["test_artifacts_semantically_cover_small_parts"]
    assert report["test_artifacts_semantically_cover_small_parts"] is False
    assert report["uncovered_test_artifact_domain_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "required_anchor_tokens": ["auth"],
            "required_domain_tokens": ["auth", "role", "user"],
            "covered_anchor_tokens": [],
            "covered_domain_tokens": [],
            "missing_anchor_tokens": ["auth"],
            "test_artifacts": [],
            "covered": False,
        },
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "required_anchor_tokens": ["crud"],
            "required_domain_tokens": ["crud", "set", "word"],
            "covered_anchor_tokens": [],
            "covered_domain_tokens": [],
            "missing_anchor_tokens": ["crud"],
            "test_artifacts": [],
            "covered": False,
        },
    ]


def test_prd_quality_accepts_domain_specific_test_artifacts() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Word set CRUD operations work",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "backend/tests/test_auth.py",
            "backend/tests/test_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["test_artifacts_semantically_cover_small_parts"] is True
    assert report["uncovered_test_artifact_domain_small_parts"] == []


def test_job_runner_blocks_prd_quality_when_implementation_artifacts_miss_surface(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="English Vocab App",
                problem_statement="Students need vocabulary practice.",
                smallest_working_core=["Serve a vocabulary app shell"],
                small_parts=[
                    "Create backend API endpoints",
                    "Create frontend UI component",
                ],
                incremental_milestones=[
                    "Backend API endpoints exist",
                    "Frontend UI component exists",
                ],
                acceptance_tests=[
                    "Backend API endpoints return HTTP responses",
                    "Frontend UI component renders vocabulary form",
                ],
                definition_of_done=["All tests pass"],
                required_artifacts=[
                    "backend/main.py",
                    "tests/test_project_setup.py",
                ],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create an English vocabulary app",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-artifact-surfaces",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:implementation_artifacts_cover_small_parts",
    )
    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["prd_quality_missing"] == [
        "implementation_artifacts_cover_small_parts"
    ]
    assert constraints["uncovered_implementation_artifact_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Create frontend UI component",
            "required_surfaces": ["frontend"],
            "covered_surfaces": [],
            "missing_surfaces": ["frontend"],
            "implementation_artifacts": [],
            "covered": False,
        }
    ]
    assert "architect" not in record.outputs


def test_prd_quality_rejects_acceptance_tests_that_restate_work_items() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create backend feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Create backend feature module"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/main.py", "tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["acceptance_tests_observable"]
    assert report["acceptance_tests_semantically_cover_small_parts"] is True
    assert report["acceptance_tests_observable"] is False
    assert report["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "Create backend feature module",
        }
    ]


def test_prd_quality_rejects_small_parts_that_do_not_cover_core_anchors() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need quizzes and tracked progress.",
        smallest_working_core=["Generate quizzes and track progress"],
        small_parts=["Create quiz generator"],
        incremental_milestones=["Quiz generator exists"],
        acceptance_tests=["Quiz generator returns questions"],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/src/routes/quiz.js",
            "tests/quiz.test.js",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["smallest_working_core_covered_by_small_parts"]
    assert report["smallest_working_core_covered_by_small_parts"] is False
    assert report["uncovered_smallest_working_core"] == [
        {
            "core_index": 1,
            "smallest_working_core": "Generate quizzes and track progress",
            "required_anchor_tokens": ["progress", "quiz"],
            "covered_anchor_tokens": ["quiz"],
            "missing_anchor_tokens": ["progress"],
            "covered": False,
        }
    ]


def test_prd_quality_accepts_small_parts_covering_core_anchors() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need quizzes and tracked progress.",
        smallest_working_core=["Generate quizzes and track progress"],
        small_parts=[
            "Create quiz generator",
            "Track user progress",
        ],
        incremental_milestones=[
            "Quiz generator exists",
            "Progress tracking exists",
        ],
        acceptance_tests=[
            "Quiz generator returns questions",
            "Progress tracker stores user progress",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/src/routes/quiz.js",
            "backend/src/routes/progress.js",
            "tests/quiz_progress.test.js",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["smallest_working_core_covered_by_small_parts"] is True
    assert report["uncovered_smallest_working_core"] == []


def test_prd_quality_rejects_generic_tests_pass_acceptance_tests() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create backend feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["All tests pass"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/main.py", "tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == [
        "acceptance_tests_semantically_cover_small_parts",
        "acceptance_tests_observable",
    ]
    assert report["acceptance_tests_observable"] is False
    assert report["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "All tests pass",
        }
    ]


def test_prd_quality_rejects_generic_feature_works_acceptance_tests() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Feature works"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/main.py", "tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["acceptance_tests_observable"]
    assert report["acceptance_tests_semantically_cover_small_parts"] is True
    assert report["acceptance_tests_observable"] is False
    assert report["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "Feature works",
        }
    ]


def test_prd_quality_rejects_domain_works_acceptance_tests() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=["User authentication and roles"],
        incremental_milestones=["Users can sign in"],
        acceptance_tests=["User authentication works"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/auth.py", "tests/test_auth.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["acceptance_tests_observable"]
    assert report["acceptance_tests_semantically_cover_small_parts"] is True
    assert report["acceptance_tests_observable"] is False
    assert report["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "User authentication works",
        }
    ]


def test_prd_quality_requires_incremental_milestones_for_each_small_part() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module", "Add focused tests"],
        incremental_milestones=["Feature module exists"],
        acceptance_tests=["Feature module exists", "Focused tests exist"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["incremental_milestones_cover_small_parts"]
    assert report["incremental_milestone_count"] == 1
    assert report["incremental_milestones_cover_small_parts"] is False
    assert report["missing_incremental_milestone_count"] == 1
    assert report["required_incremental_milestone_count"] == 2


def test_prd_quality_rejects_incremental_milestones_missing_domain_anchors() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in",
            "Backend module exists",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "tests/test_auth_and_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == [
        "incremental_milestones_semantically_cover_small_parts"
    ]
    assert report["incremental_milestones_cover_small_parts"] is True
    assert report["incremental_milestones_semantically_cover_small_parts"] is False
    assert report["uncovered_incremental_milestone_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "required_anchor_tokens": ["crud"],
            "covered_anchor_tokens": [],
            "missing_anchor_tokens": ["crud"],
            "covered": False,
        }
    ]


def test_prd_quality_accepts_incremental_milestones_with_domain_anchors() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need account-based vocabulary practice.",
        smallest_working_core=["Serve an authenticated vocabulary shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Word set CRUD operations work",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "tests/test_auth_and_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["incremental_milestones_semantically_cover_small_parts"] is True
    assert report["uncovered_incremental_milestone_small_parts"] == []


def test_prd_quality_allows_docs_artifact_for_docs_only_work() -> None:
    prd = PRD(
        title="Docs",
        problem_statement="Users need setup instructions.",
        smallest_working_core=["Document setup steps"],
        small_parts=["Write README setup guide"],
        incremental_milestones=["README guide exists"],
        acceptance_tests=["README setup guide contains install and run steps"],
        definition_of_done=["Docs tests pass"],
        required_artifacts=["README.md", "tests/test_readme.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["source_required_artifacts"] == ["README.md"]
    assert report["implementation_required_artifacts"] == []
    assert report["test_required_artifacts"] == ["tests/test_readme.py"]


def test_prd_quality_rejects_invalid_required_artifact_paths() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Feature module exists"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "../outside.py", "C:\\outside.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == [
        "required_artifacts_valid_paths",
        "required_test_artifacts",
    ]
    assert report["required_artifact_count"] == 1
    assert report["required_artifacts"] == ["feature.py"]
    assert report["source_required_artifact_count"] == 1
    assert report["source_required_artifacts"] == ["feature.py"]
    assert report["implementation_required_artifact_count"] == 1
    assert report["implementation_required_artifacts"] == ["feature.py"]
    assert report["test_required_artifact_count"] == 0
    assert report["test_required_artifacts"] == []
    assert report["invalid_required_artifacts"] == ["../outside.py", "C:\\outside.py"]


def test_prd_quality_requires_required_artifacts() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Feature module exists"],
        definition_of_done=["All tests pass"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["required_artifacts"]
    assert report["required_artifact_count"] == 0
    assert report["required_artifacts"] == []
    assert report["source_required_artifact_count"] == 0
    assert report["source_required_artifacts"] == []
    assert report["test_required_artifact_count"] == 0
    assert report["test_required_artifacts"] == []


def test_prd_quality_rejects_placeholder_prd_sections() -> None:
    prd = PRD(
        title="TBD",
        problem_statement="TODO",
        smallest_working_core=["..."],
        small_parts=["placeholder"],
        incremental_milestones=["to be determined"],
        acceptance_tests=["n/a"],
        definition_of_done=["none"],
        required_artifacts=["TBD"],
        open_questions=["No open questions"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == [
        "title",
        "problem_statement",
        "smallest_working_core",
        "small_parts",
        "incremental_milestones",
        "acceptance_tests",
        "definition_of_done",
        "required_artifacts",
    ]
    assert report["warnings"] == []
    assert report["small_part_count"] == 0
    assert report["acceptance_test_count"] == 0
    assert report["definition_of_done_count"] == 0
    assert report["required_artifact_count"] == 0


def test_prd_quality_ignores_placeholder_open_questions() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need quizzes and tracked progress.",
        smallest_working_core=["Generate quizzes and track progress"],
        small_parts=[
            "Create quiz generator",
            "Track user progress",
        ],
        incremental_milestones=[
            "Quiz generator exists",
            "Progress tracking exists",
        ],
        acceptance_tests=[
            "Quiz generator returns questions",
            "Progress tracker stores user progress",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/src/routes/quiz.js",
            "backend/src/routes/progress.js",
            "tests/quiz_progress.test.js",
        ],
        open_questions=["None"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert "open_questions_resolved" not in report["missing"]
    assert report["warnings"] == []


def test_prd_quality_blocks_unresolved_open_questions() -> None:
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module", "Add focused tests"],
        incremental_milestones=["Module exists", "Tests exist"],
        acceptance_tests=["Feature module exists", "Focused tests exist"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
        open_questions=["Which persistence backend should be used?"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["open_questions_resolved"]
    assert report["warnings"] == ["open_questions_present"]


def test_prd_quality_repair_logs_name_uncovered_small_parts() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        acceptance_tests=["GET /api/health returns 200"],
        open_questions=["Which database should store word progress?"],
    )
    report = {
        "missing": [
            "acceptance_tests_cover_small_parts",
            "open_questions_resolved",
        ],
        "warnings": ["open_questions_present"],
        "uncovered_acceptance_small_parts": [
            {
                "small_part_index": 2,
                "small_part": "Define shared TypeScript types for WordSet and UserProgress",
            },
            {
                "small_part_index": 4,
                "small_part": "Implement backend API endpoints for WordSet CRUD",
            },
        ],
    }

    logs = JobRunner._prd_quality_repair_logs(prd, report)

    assert logs[:3] == [
        "The previous PRD was not specific enough for autonomous execution.",
        "Missing fields: acceptance_tests_cover_small_parts, open_questions_resolved",
        "Warnings: open_questions_present",
    ]
    assert any(
        "2: Define shared TypeScript types for WordSet and UserProgress" in item
        for item in logs
    )
    assert any(
        "4: Implement backend API endpoints for WordSet CRUD" in item
        for item in logs
    )
    assert any("Current acceptance_tests: GET /api/health returns 200" == item for item in logs)
    assert any(
        "Open questions blocking autonomy: Which database should store word progress?"
        == item
        for item in logs
    )
    assert any(
        "Resolve open_questions before implementation" in item for item in logs
    )


def test_prd_quality_deterministically_repairs_uncovered_acceptance_tests(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build vocabulary app",
            repo_path=str(workspace),
            metadata={"constraints": {"require_prd_quality": True}},
        )
    )
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        smallest_working_core=["Generate quizzes and track progress"],
        small_parts=[
            "Quiz generation endpoint",
            "Progress tracking endpoint",
        ],
        incremental_milestones=["Quiz API works", "Progress API works"],
        acceptance_tests=["React app renders the word list"],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/src/routes/quiz.js",
            "backend/src/routes/progress.js",
            "tests/quiz_progress.test.js",
        ],
    )

    repaired = runner._refine_prd_quality_for_autonomy(record, prd)

    assert repaired is not None
    assert record.outputs["prd_quality"]["passed"] is True
    assert [
        attempt["action"] for attempt in record.outputs["prd_quality_attempts"]
    ] == ["initial", "deterministic_repair"]
    assert record.outputs["prd_quality_deterministic_repair"] == {
        "applied": True,
        "added_acceptance_tests": [
            (
                "Quiz generation endpoint works and can be verified by an "
                "observable app or API check."
            ),
            (
                "Progress tracking endpoint works and can be verified by an "
                "observable app or API check."
            ),
        ],
    }
    assert repaired.acceptance_tests[-2:] == record.outputs[
        "prd_quality_deterministic_repair"
    ]["added_acceptance_tests"]


def test_prd_quality_deterministically_repairs_non_observable_acceptance_tests(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build backend feature",
            repo_path=str(workspace),
            metadata={"constraints": {"require_prd_quality": True}},
        )
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create backend feature module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["Create backend feature module"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/main.py", "tests/test_feature.py"],
    )

    repaired = runner._refine_prd_quality_for_autonomy(record, prd)

    assert repaired is not None
    assert record.outputs["prd_quality"]["passed"] is True
    assert [
        attempt["action"] for attempt in record.outputs["prd_quality_attempts"]
    ] == ["initial", "deterministic_repair"]
    assert record.outputs["prd_quality_deterministic_repair"] == {
        "applied": True,
        "added_acceptance_tests": [
            (
                "Create backend feature module works and can be verified by an "
                "observable app or API check."
            )
        ],
    }
    assert repaired.acceptance_tests == record.outputs[
        "prd_quality_deterministic_repair"
    ]["added_acceptance_tests"]


def test_prd_quality_deterministically_repairs_missing_incremental_milestones(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build feature incrementally",
            repo_path=str(workspace),
            metadata={"constraints": {"require_prd_quality": True}},
        )
    )
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a feature module"],
        small_parts=["Create feature module", "Add focused tests"],
        incremental_milestones=["Feature module exists"],
        acceptance_tests=["Feature module exists", "Focused tests exist"],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )

    repaired = runner._refine_prd_quality_for_autonomy(record, prd)

    assert repaired is not None
    assert record.outputs["prd_quality"]["passed"] is True
    assert [
        attempt["action"] for attempt in record.outputs["prd_quality_attempts"]
    ] == ["initial", "deterministic_repair"]
    assert record.outputs["prd_quality_deterministic_repair"] == {
        "applied": True,
        "added_incremental_milestones": [
            "Automated checks for Add focused tests exist and pass"
        ],
    }
    assert repaired.incremental_milestones == [
        "Feature module exists",
        "Automated checks for Add focused tests exist and pass",
    ]


def test_prd_quality_deterministically_repairs_semantic_incremental_milestones(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.store.create(
        JobSpec(
            request_text="Build vocabulary app incrementally",
            repo_path=str(workspace),
            metadata={"constraints": {"require_prd_quality": True}},
        )
    )
    prd = PRD(
        title="English Vocab App",
        problem_statement="Teachers need vocabulary management.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Users can sign in with roles",
            "Backend module exists",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "tests/test_auth_and_word_sets.py",
        ],
    )

    repaired = runner._refine_prd_quality_for_autonomy(record, prd)

    assert repaired is not None
    assert record.outputs["prd_quality"]["passed"] is True
    assert [
        attempt["action"] for attempt in record.outputs["prd_quality_attempts"]
    ] == ["initial", "deterministic_repair"]
    assert record.outputs["prd_quality_deterministic_repair"] == {
        "applied": True,
        "added_incremental_milestones": [
            "Word set CRUD operations works and is ready for focused tests"
        ],
    }
    assert repaired.incremental_milestones[-1] == (
        "Word set CRUD operations works and is ready for focused tests"
    )


def test_semantic_tokens_split_camel_case_domain_terms() -> None:
    tokens = JobRunner._semantic_tokens("WordSet QuizQuestion UserProgress quizzes")
    file_tokens = JobRunner._semantic_tokens("tests/test_add_one_double.py")

    assert {"word", "set", "quiz", "question", "user", "progress"}.issubset(tokens)
    assert {"add_one", "double"}.issubset(file_tokens)


def test_prd_quality_accepts_semantically_covered_acceptance_tests() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need account-based vocabulary practice.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=[
            "User authentication and roles",
            "Word set CRUD operations",
        ],
        incremental_milestones=[
            "Students can sign in",
            "Teachers can perform CRUD for word sets",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/auth.py",
            "backend/word_sets.py",
            "frontend/src/App.tsx",
            "tests/test_auth_and_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["acceptance_tests_semantically_cover_small_parts"] is True
    assert report["uncovered_acceptance_small_parts"] == []
    assert report["acceptance_test_small_part_coverage"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "acceptance_test_index": 1,
            "acceptance_test": "Student can register and login",
            "covered": True,
        },
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "acceptance_test_index": 2,
            "acceptance_test": "Teacher can perform CRUD for word sets",
            "covered": True,
        },
    ]


def test_prd_quality_does_not_treat_word_list_as_crud_acceptance() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=["Backend word set CRUD API"],
        incremental_milestones=["Backend word set CRUD API works"],
        acceptance_tests=["React word list component renders vocabulary words"],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/word_sets.py",
            "frontend/src/App.tsx",
            "tests/test_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["acceptance_tests_semantically_cover_small_parts"]
    assert report["acceptance_tests_semantically_cover_small_parts"] is False
    assert report["uncovered_acceptance_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "Backend word set CRUD API",
            "acceptance_test_index": None,
            "acceptance_test": None,
            "covered": False,
        }
    ]


def test_prd_quality_accepts_crud_operations_without_crud_acronym() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        smallest_working_core=["Serve a vocabulary app shell"],
        small_parts=["Backend word set CRUD API"],
        incremental_milestones=[
            "Backend word set API supports create, read, update, and delete operations"
        ],
        acceptance_tests=[
            "Backend word set API supports create, read, update, and delete operations"
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/word_sets.py",
            "frontend/src/App.tsx",
            "tests/test_word_sets.py",
        ],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is True
    assert report["missing"] == []
    assert report["acceptance_tests_semantically_cover_small_parts"] is True
    assert report["uncovered_acceptance_small_parts"] == []


def test_prd_quality_requires_anchor_token_overlap() -> None:
    prd = PRD(
        title="Auth App",
        problem_statement="Users need secure account access.",
        smallest_working_core=["Serve an authenticated app shell"],
        small_parts=["User authentication and roles"],
        incremental_milestones=["Users can sign in"],
        acceptance_tests=["User profile roles page renders"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/auth.py", "tests/test_auth.py"],
    )

    report = JobRunner._build_prd_quality_report(prd)

    assert report["passed"] is False
    assert report["missing"] == ["acceptance_tests_semantically_cover_small_parts"]
    assert report["acceptance_tests_semantically_cover_small_parts"] is False
    assert report["uncovered_acceptance_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "acceptance_test_index": None,
            "acceptance_test": None,
            "covered": False,
        }
    ]


def test_job_runner_blocks_prd_quality_when_acceptance_tests_semantically_mismatch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="English Vocab App",
                problem_statement="Students need account-based vocabulary practice.",
                smallest_working_core=["Serve a vocabulary app shell"],
                small_parts=[
                    "User authentication and roles",
                    "Word set CRUD operations",
                ],
                incremental_milestones=[
                    "Students can sign in",
                    "Teachers can perform CRUD for word sets",
                ],
                acceptance_tests=[
                    "Health endpoint returns 200",
                    "Database initializes successfully",
                ],
                definition_of_done=["All tests pass"],
                required_artifacts=[
                    "backend/auth.py",
                    "backend/word_sets.py",
                    "frontend/src/App.tsx",
                    "tests/test_auth_and_word_sets.py",
                ],
            ).model_dump(),
            "architect": ArchitecturePlan(summary="Should not run").model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create an English vocabulary app",
        repo_path=str(workspace),
        target_branch="acos/strict-prd-semantic-acceptance",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "prd_quality_refinement_attempts": 0,
            }
        },
    )

    record = runner.run_job(spec)

    assert_recovery_plan(
        record,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(
        record,
        "prd_quality_gate_failed:acceptance_tests_semantically_cover_small_parts",
    )
    assert record.outputs["prd_quality"]["passed"] is False
    assert record.outputs["prd_quality"]["missing"] == [
        "acceptance_tests_semantically_cover_small_parts"
    ]
    assert record.outputs["prd_quality"]["uncovered_acceptance_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "User authentication and roles",
            "acceptance_test_index": None,
            "acceptance_test": None,
            "covered": False,
        },
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "acceptance_test_index": None,
            "acceptance_test": None,
            "covered": False,
        },
    ]
    assert "architect" not in record.outputs


def test_job_runner_resumes_blocked_prd_quality_repair(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    sparse_prd = PRD(title="Feature", problem_statement="Need feature").model_dump()
    good_prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose a VALUE constant and test it"],
        small_parts=["Create feature module", "Add focused tests"],
        incremental_milestones=[
            "Smallest module exists",
            "Tests prove observable behavior",
        ],
        acceptance_tests=[
            "VALUE equals 1",
            "test_value asserts VALUE equals 1",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    ).model_dump()
    attach_mock_adapter(
        registry,
        {
            "pm": [sparse_prd, sparse_prd, sparse_prd, good_prd],
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Build core",
                        description="Build the smallest feature.",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": (
                            "from feature import VALUE\n\n\n"
                            "def test_value() -> None:\n"
                            "    assert VALUE == 1\n"
                        ),
                        "operation": "create",
                    }
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create a complex feature",
        repo_path=str(workspace),
        target_branch="acos/resume-prd-quality",
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "skip_review": True,
                "skip_release": True,
            }
        },
    )

    blocked = runner.run_job(spec)

    assert_recovery_plan(
        blocked,
        status=JobStatus.ANALYZING,
        strategy="REVISE_PRD_AND_ARCHITECTURE",
    )
    assert_recoverable_error(blocked, startswith="prd_quality_gate_failed:")

    resumed = runner.resume_job(blocked.job_id)

    assert resumed.status == JobStatus.DONE
    assert resumed.last_error is None
    assert resumed.outputs["prd_quality"]["passed"] is True
    assert resumed.outputs["prd"] == good_prd
    assert JobStatus.ANALYZING in resumed.history
    assert (workspace / "feature.py").exists()


def test_job_runner_includes_job_constraints_in_agent_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Recover a repeated failure",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "recovery_mode": "repeated_failure",
                "recovery_strategy": "escalated_retry",
                "recovery_attempt": 1,
                "stage_review": True,
            }
        },
    )
    record = InMemoryJobStore().create(spec)

    constraints = runner._context_constraints(record)

    assert "job_constraint recovery_mode=repeated_failure" in constraints
    assert "job_constraint recovery_strategy=escalated_retry" in constraints
    assert "job_constraint recovery_attempt=1" in constraints
    assert "job_constraint stage_review=True" in constraints


def test_job_runner_clears_planning_repair_constraints_after_prd_passes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Build with repaired planning",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "planning_repair_strategy_change": True,
                "planning_repair_repeated_prd_missing": "acceptance_tests",
                "pm_strategy_change": True,
                "pm_strategy": "planning_repair_strategy_change",
                "pm_intervention_count": 1,
                "prd_quality_missing": ["acceptance_tests"],
                "prd_quality_warnings": ["open_questions_present"],
                "prd_open_questions": ["Which backend?"],
                "uncovered_acceptance_small_parts": [
                    {"small_part_index": 1, "small_part": "Create module"}
                ],
                "uncovered_implementation_artifact_small_parts": [
                    {
                        "small_part_index": 1,
                        "small_part": "Create frontend UI",
                        "missing_surfaces": ["frontend"],
                    }
                ],
                "non_observable_acceptance_tests": [
                    {
                        "acceptance_test_index": 1,
                        "acceptance_test": "Create module",
                    }
                ],
                "invalid_required_artifacts": ["../outside.py"],
                "failed_task_id": "core-tests",
                "missing_task_ids": ["core-tests"],
                "unmet_dependencies": ["core"],
                "prd_required_artifacts": ["tests/test_feature.py"],
                "required_incremental_milestone_count": 2,
                "required_small_part_count": 2,
                "source_required_artifacts": ["feature.py"],
                "implementation_required_artifacts": ["feature.py"],
                "test_required_artifacts": ["tests/test_feature.py"],
                "recovery_mode": "prd_quality_revision",
                "recovery_strategy": "REVISE_PRD_AND_ARCHITECTURE",
                "recovery_next_actor": "pm",
                "recovery_next_status": "analyzing",
                "recovery_reason": "PRD quality gate failed",
                "recovery_failed_task_id": "pm",
                "recovery_failed_stage": 1,
                "recovery_attempt": 2,
                "recovery_step_count": 1,
            }
        },
    )
    record = store.create(spec)
    runner.store = store
    prd = PRD(
        title="Feature",
        problem_statement="Need feature",
        smallest_working_core=["Expose VALUE"],
        small_parts=["Create module"],
        incremental_milestones=["Module exists"],
        acceptance_tests=["VALUE equals 1"],
        definition_of_done=["Tests pass"],
        required_artifacts=["feature.py", "tests/test_feature.py"],
    )

    result = runner._refine_prd_quality_for_autonomy(record, prd)

    assert result == prd
    constraints = record.spec.metadata["constraints"]
    for key in (
        "planning_repair_strategy_change",
        "planning_repair_repeated_prd_missing",
        "pm_strategy_change",
        "pm_strategy",
        "pm_intervention_count",
        "prd_quality_missing",
        "prd_quality_warnings",
        "prd_open_questions",
        "uncovered_acceptance_small_parts",
        "uncovered_implementation_artifact_small_parts",
        "non_observable_acceptance_tests",
        "invalid_required_artifacts",
        "failed_task_id",
        "missing_task_ids",
        "unmet_dependencies",
        "prd_required_artifacts",
        "required_incremental_milestone_count",
        "required_small_part_count",
        "source_required_artifacts",
        "implementation_required_artifacts",
        "test_required_artifacts",
        "recovery_mode",
        "recovery_strategy",
        "recovery_next_actor",
        "recovery_next_status",
        "recovery_reason",
        "recovery_failed_task_id",
        "recovery_failed_stage",
        "recovery_attempt",
        "recovery_step_count",
    ):
        assert key not in constraints


def test_job_runner_clears_planning_repair_constraints_after_task_graph_validates(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    store = InMemoryJobStore()
    spec = JobSpec(
        request_text="Build with repaired graph",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "planning_repair_strategy_change": True,
                "planning_repair_repeated_task_graph_error_types": "unknown_dependencies",
                "pm_strategy_change": True,
                "pm_strategy": "planning_repair_strategy_change",
                "pm_intervention_count": 1,
                "task_graph_validation_errors": ["unknown_dependencies"],
                "failed_task_id": "core-tests",
                "unmet_dependencies": ["core"],
                "unknown_dependencies": [
                    {"task_id": "views", "dependency": "models"}
                ],
                "duplicate_task_ids": ["core"],
                "dependency_cycle_task_ids": ["core"],
                "uncovered_small_parts": [
                    {"small_part_index": 1, "small_part": "Create module"}
                ],
                "uncovered_acceptance_tests": [
                    {"acceptance_test_index": 1, "acceptance_test": "VALUE equals 1"}
                ],
                "role_mismatched_target_files": [
                    {"task_id": "core", "path": "tests/test_feature.py"}
                ],
                "test_writer_dependency_semantic_mismatches": [
                    {"task_id": "frontend-tests", "depends_on": ["backend-api"]}
                ],
                "test_writer_acceptance_dependency_mismatches": [
                    {
                        "task_id": "frontend-tests",
                        "depends_on": ["backend-api"],
                        "uncovered_acceptance_criteria": [
                            {
                                "acceptance_criteria_index": 1,
                                "acceptance_criteria": "Frontend UI renders VALUE",
                                "covered": False,
                            }
                        ],
                    }
                ],
                "recovery_mode": "task_graph_repair",
                "recovery_strategy": "REPLAN_TASK",
                "recovery_next_actor": "planner",
                "recovery_next_status": "replanning",
                "recovery_reason": "invalid task graph",
                "recovery_failed_task_id": "planner",
                "recovery_failed_stage": 1,
                "recovery_attempt": 2,
                "recovery_step_count": 1,
            }
        },
    )
    record = store.create(spec)
    for key in JobRunner.TASK_GRAPH_VALIDATION_DETAIL_KEYS:
        record.spec.metadata["constraints"].setdefault(
            key,
            [{"detail_key": key}],
        )
    record.outputs["task_graph"] = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            )
        ],
    ).model_dump()
    runner.store = store
    prd = PRD(title="Feature", problem_statement="Need feature")

    result = runner._load_or_repair_task_graph_for_autonomy(record, prd)

    assert result is not None
    constraints = record.spec.metadata["constraints"]
    for key in (
        "planning_repair_strategy_change",
        "planning_repair_repeated_task_graph_error_types",
        "pm_strategy_change",
        "pm_strategy",
        "pm_intervention_count",
        "task_graph_validation_errors",
        "failed_task_id",
        "unmet_dependencies",
        "unknown_dependencies",
        "duplicate_task_ids",
        "dependency_cycle_task_ids",
        "uncovered_small_parts",
        "uncovered_acceptance_tests",
        "role_mismatched_target_files",
        "test_writer_dependency_semantic_mismatches",
        "test_writer_acceptance_dependency_mismatches",
        "recovery_mode",
        "recovery_strategy",
        "recovery_next_actor",
        "recovery_next_status",
        "recovery_reason",
        "recovery_failed_task_id",
        "recovery_failed_stage",
        "recovery_attempt",
        "recovery_step_count",
    ):
        assert key not in constraints
    for key in JobRunner.TASK_GRAPH_VALIDATION_DETAIL_KEYS:
        assert key not in constraints


def test_clear_planning_repair_constraints_keeps_active_implementation_recovery(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        request_text="Recover implementation",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "planning_repair_repeated_prd_missing": "acceptance_tests",
                "prd_quality_missing": ["acceptance_tests"],
                "recovery_mode": "implementation_failure",
                "recovery_strategy": "replan_current_task",
                "recovery_reason": "the implementer failed before producing a safe completed change",
                "failed_task_id": "core",
                "recovery_failed_task_id": "core",
                "recovery_failed_stage": 2,
                "recovery_attempt": 3,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.PLANNING)

    JobRunner._clear_planning_repair_constraints(record)

    constraints = record.spec.metadata["constraints"]
    assert "planning_repair_repeated_prd_missing" not in constraints
    assert "prd_quality_missing" not in constraints
    assert constraints.items() >= {
        "recovery_mode": "implementation_failure",
        "recovery_strategy": "replan_current_task",
        "recovery_reason": "the implementer failed before producing a safe completed change",
        "failed_task_id": "core",
        "recovery_failed_task_id": "core",
        "recovery_failed_stage": 2,
        "recovery_attempt": 3,
    }.items()


def test_job_runner_adds_recovery_guidance_to_agent_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Recovered implementation",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Recover a failed implementation",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "recovery_mode": "implementation_failure",
                "recovery_strategy": "replan_current_task",
                "recovery_reason": "the implementer failed before producing a safe completed change",
                "recovery_failed_task_id": "core",
                "recovery_attempt": 2,
            }
        },
    )
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.PLANNING
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "implementer":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(
        record,
        "implementer",
        ImplementationResult,
        "Recover the failed task",
        logs=["existing log"],
    )

    assert captured["logs"][0].startswith(
        "recovery_context: mode=implementation_failure; strategy=replan_current_task;"
    )
    assert "failed_task_id=core" in captured["logs"][0]
    assert captured["logs"][1] == (
        "recovery_instruction: re-scope the failed task into a smaller, testable step "
        "before implementing more code."
    )
    assert captured["logs"][2] == "existing log"


def test_job_runner_adds_pm_recovery_playbook_to_agent_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Recovered dependency setup",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Recover dependencies",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "pm_stall_recovery": True,
                "pm_strategy_change": True,
                "pm_strategy": "dependency_alignment_first",
                "pm_next_actor": "implementer",
                "pm_reason": "diagnosed_repeated_failure",
                "pm_recovery_playbook": (
                    "inspect dependency manifests before touching application logic"
                ),
                "pm_success_criteria": "dependency import smoke test passes",
            }
        },
    )
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.PLANNING
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "implementer":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(
        record,
        "implementer",
        ImplementationResult,
        "Recover the failed task",
    )

    assert captured["logs"][0].startswith(
        "pm_stall_recovery: strategy=dependency_alignment_first;"
    )
    assert "next_actor=implementer" in captured["logs"][0]
    assert captured["logs"][1] == (
        "pm_recovery_playbook: inspect dependency manifests before touching "
        "application logic"
    )
    assert captured["logs"][2] == (
        "pm_recovery_success_criteria: dependency import smoke test passes"
    )
    assert captured["logs"][3] == (
        "pm_stall_instruction: execute the PM recovery playbook before normal "
        "feature work; success means the diagnosed signature is removed, not merely "
        "that a new patch was attempted."
    )


def test_job_runner_adds_planning_repair_guidance_to_pm_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="Feature",
                problem_statement="Need feature",
                smallest_working_core=["Core"],
                small_parts=["Part"],
                incremental_milestones=["Milestone"],
                acceptance_tests=["VALUE equals 1"],
                definition_of_done=["Tests pass"],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(request_text="Repair PRD", repo_path=str(workspace))
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.SUBMITTED
    record.outputs["prd_quality_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 1,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 2,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
    ]
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "pm":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(record, "pm", PRD, "Repair product requirements")

    assert captured["logs"] == [
        (
            "planning_repair_context: consecutive_prd_failures=3; "
            "consecutive_task_graph_failures=0; "
            "repeated_prd_missing=acceptance_tests; "
            "repeated_task_graph_error_types=none"
        ),
        (
            "planning_repair_instruction: change the requirements strategy; explicitly fill "
            "the repeated missing PRD fields with concrete, testable details before moving on."
        ),
    ]


def test_job_runner_adds_planning_repair_guidance_to_planner_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "planner": TaskGraph(
                goal="Repair graph",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                    )
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(request_text="Repair task graph", repo_path=str(workspace))
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.DESIGNING
    record.outputs["task_graph_validation_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
        {
            "attempt": 1,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
        {
            "attempt": 2,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
    ]
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "planner":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(record, "planner", TaskGraph, "Repair task graph")

    assert captured["logs"] == [
        (
            "planning_repair_context: consecutive_prd_failures=0; "
            "consecutive_task_graph_failures=3; "
            "repeated_prd_missing=none; "
            "repeated_task_graph_error_types=unknown_dependencies"
        ),
        (
            "planning_repair_instruction: change the task graph strategy; simplify or split "
            "the plan so repeated validation errors are removed instead of reusing the same graph."
        ),
    ]


def test_job_runner_adds_task_graph_details_to_planner_repair_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "planner": TaskGraph(
                goal="Repair graph",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                    )
                ],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(request_text="Repair task graph", repo_path=str(workspace))
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.DESIGNING
    role_mismatch_detail = [
        {
            "task_id": "core",
            "role": "implementer",
            "path": "frontend/test/project_scaffold.test.tsx",
            "expected_roles": ["test_writer"],
        }
    ]
    record.outputs["task_graph_validation_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "valid": False,
            "errors": [{"type": "role_mismatched_target_files"}],
            "role_mismatched_target_files": role_mismatch_detail,
        },
        {
            "attempt": 1,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "role_mismatched_target_files"}],
            "role_mismatched_target_files": role_mismatch_detail,
        },
        {
            "attempt": 2,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "role_mismatched_target_files"}],
            "role_mismatched_target_files": role_mismatch_detail,
        },
    ]
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "planner":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(record, "planner", TaskGraph, "Repair task graph")

    assert captured["logs"] == [
        (
            "planning_repair_context: consecutive_prd_failures=0; "
            "consecutive_task_graph_failures=3; "
            "repeated_prd_missing=none; "
            "repeated_task_graph_error_types=role_mismatched_target_files"
        ),
        (
            "planning_repair_instruction: change the task graph strategy; simplify or split "
            "the plan so repeated validation errors are removed instead of reusing the same graph."
        ),
        f"planning_repair_task_graph_detail: role_mismatched_target_files={role_mismatch_detail}",
    ]


def test_job_runner_adds_recovery_history_to_agent_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Next implementation",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    task_graph = TaskGraph(
        goal="Build after recovery",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
            PlannedTask(
                id="extra",
                title="Extra",
                description="Build extra",
                role="implementer",
                depends_on=["core"],
            ),
        ],
    )
    spec = JobSpec(
        request_text="Continue after recovery",
        repo_path=str(workspace),
    )
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.PLANNING
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py", "tests/test_feature.py"],
                "patch_count": 2,
            },
            "test_run": {"success": True},
        },
    ]
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "implementer":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(
        record,
        "implementer",
        ImplementationResult,
        "Continue with the next task",
        task=task_graph.tasks[1],
    )

    assert captured["logs"] == [
        (
            "recovered_failure: task_id=core; failed_stage=1; resolved_by_stage=2; "
            "failed_files=feature.py; resolved_files=feature.py,tests/test_feature.py; "
            "failed_patch_count=1; resolved_patch_count=2"
        )
    ]


def test_job_runner_recovers_structured_output_max_steps_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    store = InMemoryJobStore()
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
    )
    spec = JobSpec(request_text="Recover structured output", repo_path=str(workspace))
    record = store.create(spec)

    def fail_with_max_steps(_record):
        raise StructuredOutputError(
            "Agent fixer exceeded max_steps=24 without a valid structured response; "
            "last_model=ornith_35b_q4; last_status=success"
        )

    monkeypatch.setattr(runner, "_load_or_refine_prd_for_autonomy", fail_with_max_steps)

    recovered = runner._run_record(record, resume=True)

    assert recovered.status == JobStatus.STRATEGY_CHANGE
    assert_recoverable_error(recovered, startswith="Agent fixer exceeded max_steps=24")
    assert recovered.runtime_state["recovery_plan"]["strategy"] == (
        "RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD"
    )
    assert recovered.spec.metadata["constraints"]["max_steps_exceeded_role"] == "fixer"


def test_job_runner_adds_structured_output_recovery_guidance_to_fixer_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Small structured retry",
                patches=[],
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Recover fixer output",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "recovery_mode": "agent_max_steps_structured_output",
                "recovery_strategy": "RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD",
                "max_steps_exceeded_role": "fixer",
                "avoid_tool_loop": True,
                "force_structured_output": True,
                "retry_small_scope": True,
            }
        },
    )
    store = InMemoryJobStore()
    record = store.create(spec)
    record.status = JobStatus.TESTING
    runner.store = store
    captured: dict[str, object] = {}
    original_run = runner.agent_runner.run

    def capture_run(*args, **kwargs):
        if kwargs.get("role") == "fixer":
            captured["logs"] = list(kwargs["context_packet"].logs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(runner.agent_runner, "run", capture_run)

    runner._run_structured_role(record, "fixer", FixResult, "Retry with structured output")

    assert captured["logs"][:3] == [
        (
            "recovery_context: mode=agent_max_steps_structured_output; "
            "strategy=RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD; attempt=1; "
            "failed_task_id=unknown; failed_stage=unknown; reason=unspecified"
        ),
        (
            "recovery_instruction: the previous agent exhausted tool steps without returning "
            "valid JSON. Inspect only files already named in the diagnosis or retrieval trace, "
            "make the smallest necessary patch, then return the required structured JSON. "
            "Do not continue broad repository exploration."
        ),
        (
            "structured_output_recovery: previous_role=fixer; avoid_tool_loop=true; "
            "return_schema_first=true; retry_small_scope=true"
        ),
    ]
