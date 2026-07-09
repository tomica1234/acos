from __future__ import annotations

from pathlib import Path

from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.recovery_executor import RecoveryExecutor
from packages.schemas.agent_outputs import (
    FilePatch,
    ImplementationResult,
    PRD,
    TestRunResult,
    TestWriterResult,
)
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import ImplementationStatus, JobStatus
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir


def _runner(
    tmp_path: Path,
    *,
    scenario: dict | None = None,
    scripted_test_results: list[TestRunResult] | None = None,
) -> tuple[JobRunner, FakeMCPEnvironment, JobRecord]:
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    if scenario is not None:
        attach_mock_adapter(registry, scenario)
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=tmp_path,
        memory_db_path=tmp_path / ".memory.sqlite3",
        scripted_test_results=scripted_test_results,
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=InMemoryJobStore(),
    )
    spec = JobSpec(
        request_text="Build an English vocabulary test app",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/project-setup-regression",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RUNNING)
    runner.store.update(record)
    return runner, environment, record


def _bad_project_setup_graph() -> TaskGraph:
    return TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-setup",
                title="Project setup",
                description="Create monorepo backend/frontend/shared project setup",
                role="architect",
                target_files=[],
                required_artifacts=[],
            )
        ],
    )


def test_project_setup_architect_empty_task_is_normalized_to_scaffold(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)

    normalized = runner._normalize_project_setup_task_graph(
        record,
        _bad_project_setup_graph(),
    )
    task = normalized.tasks[0]

    assert task.id == "project-setup"
    assert task.role in {"scaffold", "implementer"}
    assert "architect" not in JobRunner.IMPLEMENTATION_TASK_ROLES
    assert task.target_files
    assert task.required_artifacts
    assert set(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS).issubset(
        set(task.required_artifacts)
    )


def test_project_setup_normalization_uses_canonical_artifacts_only(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    task_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create backend frontend shared monorepo setup",
                role="architect",
                target_files=["../outside.py", "docs/extra.md"],
                required_artifacts=["C:\\outside.py", "backend/main.py"],
            )
        ],
    )

    normalized = runner._normalize_project_setup_task_graph(record, task_graph)
    task = normalized.tasks[0]

    assert task.role == "scaffold"
    assert task.target_files == JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS
    assert task.required_artifacts == JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS
    assert record.outputs["task_graph_normalization"][
        "ignored_project_setup_artifacts"
    ] == [
        {
            "task_id": "project-scaffold",
            "paths": ["../outside.py", "docs/extra.md", "C:\\outside.py"],
        }
    ]
    validation = runner._build_task_graph_validation(
        normalized,
        require_task_artifacts=True,
        ignored_project_setup_artifacts=runner._ignored_project_setup_artifacts(record),
    )
    assert validation["valid"] is False
    assert validation["ignored_project_setup_artifacts"] == [
        {
            "task_id": "project-scaffold",
            "paths": ["../outside.py", "docs/extra.md", "C:\\outside.py"],
        }
    ]
    assert {
        error["type"] for error in validation["errors"]
    } == {"ignored_project_setup_artifacts"}


def test_architect_role_normalization_does_not_advertise_project_setup_artifacts(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="feature-design",
                title="Feature design handoff",
                description="Implement the feature from the architecture handoff.",
                role="architect",
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            )
        ],
    )

    normalized = runner._normalize_project_setup_task_graph(record, task_graph)
    task = normalized.tasks[0]
    normalization = record.outputs["task_graph_normalization"]
    validation = runner._build_task_graph_validation(
        normalized,
        require_task_artifacts=True,
    )

    assert task.role == "implementer"
    assert task.target_files == ["feature.py"]
    assert normalization["normalized_task_ids"] == ["feature-design"]
    assert normalization["project_setup_task_ids"] == []
    assert normalization["role_normalized_task_ids"] == ["feature-design"]
    assert normalization["required_artifacts"] == []
    assert normalization["ignored_project_setup_artifacts"] == []
    assert validation["valid"] is True
    assert validation["ignored_project_setup_artifacts"] == []


def test_project_setup_ignored_artifacts_are_recovered_as_invalid_task_graph(
    tmp_path: Path,
) -> None:
    task_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create backend frontend shared monorepo setup",
                role="architect",
                target_files=["backend/auth.py"],
                required_artifacts=["backend/main.py", "../outside.py"],
            )
        ],
    )
    runner, _environment, record = _runner(
        tmp_path,
        scenario={"planner": task_graph.model_dump()},
    )
    record.spec.metadata["constraints"] = {
        "require_task_artifacts": True,
        "task_graph_validation_refinement_attempts": 0,
    }
    prd = PRD(
        title="English vocabulary app",
        problem_statement="Build the app.",
    )

    result = runner._load_or_repair_task_graph_for_autonomy(record, prd)

    assert result is None
    validation = record.outputs["task_graph_validation"]
    ignored = [
        {
            "task_id": "project-scaffold",
            "paths": ["backend/auth.py", "../outside.py"],
        }
    ]
    assert validation["valid"] is False
    assert validation["ignored_project_setup_artifacts"] == ignored
    assert validation["errors"] == [
        {
            "type": "ignored_project_setup_artifacts",
            "items": ignored,
        }
    ]
    plan = record.runtime_state["recovery_plan"]
    assert record.status == JobStatus.REPLANNING
    assert plan["strategy"] == "REPLAN_TASK"
    assert plan["constraints"]["task_graph_validation_errors"] == [
        "ignored_project_setup_artifacts"
    ]
    assert plan["constraints"]["ignored_project_setup_artifacts"] == ignored
    assert record.spec.metadata["constraints"]["ignored_project_setup_artifacts"] == ignored


def test_project_setup_normalization_clears_stale_ignored_artifacts(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    bad_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create backend frontend shared monorepo setup",
                role="architect",
                target_files=["backend/auth.py"],
                required_artifacts=["backend/main.py"],
            )
        ],
    )
    clean_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="auth-backend",
                title="Build auth backend",
                description="Implement backend auth endpoints.",
                role="implementer",
                target_files=["backend/auth.py"],
                required_artifacts=["backend/auth.py"],
            )
        ],
    )

    runner._normalize_project_setup_task_graph(record, bad_graph)
    assert runner._ignored_project_setup_artifacts(record) == [
        {"task_id": "project-scaffold", "paths": ["backend/auth.py"]}
    ]

    normalized = runner._normalize_project_setup_task_graph(record, clean_graph)

    assert normalized == clean_graph
    assert "task_graph_normalization" not in record.outputs
    assert runner._ignored_project_setup_artifacts(record) == []


def test_project_setup_repair_can_clear_stale_ignored_artifacts_and_pass(
    tmp_path: Path,
) -> None:
    bad_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create backend frontend shared monorepo setup",
                role="architect",
                target_files=["backend/auth.py"],
                required_artifacts=["backend/main.py"],
            )
        ],
    )
    repaired_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="auth-backend",
                title="Build auth backend",
                description="Implement backend auth endpoints.",
                role="implementer",
                target_files=["backend/auth.py"],
                required_artifacts=["backend/auth.py"],
            )
        ],
    )
    runner, _environment, record = _runner(
        tmp_path,
        scenario={"planner": [bad_graph.model_dump(), repaired_graph.model_dump()]},
    )
    record.spec.metadata["constraints"] = {
        "require_task_artifacts": True,
        "task_graph_validation_refinement_attempts": 1,
    }
    prd = PRD(
        title="English vocabulary app",
        problem_statement="Build the app.",
    )

    result = runner._load_or_repair_task_graph_for_autonomy(record, prd)

    assert result == repaired_graph
    assert record.outputs["task_graph_validation"]["valid"] is True
    assert record.outputs["task_graph_validation"]["ignored_project_setup_artifacts"] == []
    assert "task_graph_normalization" not in record.outputs
    assert "recovery_plan" not in record.runtime_state
    assert "ignored_project_setup_artifacts" not in record.spec.metadata["constraints"]


def test_project_setup_phrase_in_app_task_does_not_trigger_scaffold(
    tmp_path: Path,
) -> None:
    source_path = "backend/auth.py"
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Implemented auth backend.",
                changed_files=[source_path],
                patches=[
                    FilePatch(
                        path=source_path,
                        operation="create",
                        content="def authenticate() -> bool:\n    return True\n",
                    )
                ],
            ).model_dump(),
        },
    )
    task_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="auth-backend",
                title="Build auth backend",
                description=(
                    "Implement auth endpoints after project setup is available."
                ),
                role="implementer",
                target_files=[source_path],
                required_artifacts=[source_path],
            )
        ],
    )

    normalized = runner._normalize_project_setup_task_graph(record, task_graph)
    results = runner._run_implementation_tasks(record, normalized)

    assert normalized == task_graph
    assert results[0].summary == "Implemented auth backend."
    assert (tmp_path / source_path).exists()
    assert not (tmp_path / "frontend/package.json").exists()
    assert "project_setup_scaffold" not in record.outputs
    assert "task_graph_normalization" not in record.outputs


def test_project_setup_cannot_enter_test_writer_before_required_artifacts_exist(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    task = runner._normalize_project_setup_task_graph(
        record,
        _bad_project_setup_graph(),
    ).tasks[0]

    assert not runner._project_setup_artifacts_ready(record, task)
    assert not runner._ensure_project_setup_ready_before_test_writer(record, task)
    assert record.runtime_state["recovery_plan"]["trigger"] == "required_artifacts_missing"
    assert record.runtime_state["recovery_plan"]["status"] != "completed"


def test_project_scaffold_role_runs_deterministic_scaffold_before_test_writer(
    tmp_path: Path,
) -> None:
    test_path = "frontend/test/project_scaffold.test.tsx"
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "test_writer": TestWriterResult(
                summary="Add project scaffold frontend smoke test.",
                changed_files=[test_path],
                patches=[
                    FilePatch(
                        path=test_path,
                        operation="create",
                        content=(
                            "import { describe, expect, it } from 'vitest'\n\n"
                            "describe('project scaffold', () => {\n"
                            "  it('loads the scaffold smoke test', () => {\n"
                            "    const label = 'ACOS project scaffold is ready'\n"
                            "    expect(label).toContain('project scaffold')\n"
                            "  })\n"
                            "})\n"
                        ),
                    )
                ],
            ).model_dump(),
        },
        scripted_test_results=[TestRunResult(success=True)],
    )
    task_graph = TaskGraph(
        goal="Build English vocabulary test app",
        tasks=[
            PlannedTask(
                id="project-scaffold-test",
                title="Frontend scaffold smoke test",
                description="Add the frontend scaffold smoke test.",
                role="test_writer",
                target_files=[test_path],
            ),
            PlannedTask(
                id="project-scaffold",
                title="Project scaffold",
                description="Create monorepo backend/frontend/shared project scaffold",
                role="scaffold",
                target_files=list(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS),
                required_artifacts=list(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS),
            ),
        ],
    )

    runner._active_record = record
    try:
        implementation_results, test_writer_results, _test_result, _stages = (
            runner._run_autonomous_task_loop(record, task_graph)
        )
    finally:
        runner._active_record = None

    assert implementation_results[0].summary == "Created deterministic project setup scaffold."
    assert test_writer_results[0].patches[0].path == test_path
    for artifact in JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS:
        assert (tmp_path / artifact).exists(), artifact
    assert (tmp_path / test_path).exists()
    assert record.outputs["project_setup_scaffold"]["missing_artifacts"] == []
    apply_patch_roles = [
        event.role
        for event in record.audit_events
        if event.action == "repo_server.apply_patch"
    ]
    assert apply_patch_roles[: len(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS)] == [
        "orchestrator"
    ] * len(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS)
    assert "test_writer" in apply_patch_roles[
        len(JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS) :
    ]


def test_generic_scaffold_role_can_create_dependency_manifest(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "scaffold": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Created root package manifest.",
                changed_files=["package.json"],
                patches=[
                    FilePatch(
                        path="package.json",
                        operation="create",
                        content='{"private": true, "scripts": {"dev": "vite"}}\n',
                    )
                ],
            ).model_dump(),
        },
    )
    task_graph = TaskGraph(
        goal="Create node project scaffold",
        tasks=[
            PlannedTask(
                id="node-manifest",
                title="Create package manifest",
                description="Create the root dependency manifest for the app.",
                role="scaffold",
                target_files=["package.json"],
                required_artifacts=["package.json"],
            ),
        ],
    )

    results = runner._run_implementation_tasks(record, task_graph)

    assert results[0].summary == "Created root package manifest."
    assert (tmp_path / "package.json").exists()
    assert "last_recoverable_error" not in record.runtime_state
    assert "scaffold" in record.outputs
    assert "scaffold_model_selection" in record.outputs
    assert "implementer" not in record.outputs


def test_update_missing_test_file_recovery_returns_to_test_writer_with_create_hint(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    patch = FilePatch(
        path="backend/tests/test_project_setup.py",
        operation="update",
        content="def test_project_setup() -> None:\n    assert 'project' in 'project setup'\n",
    )

    runner._apply_patches(record, "test_writer", [patch])

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["strategy"] == "RETURN_TO_TEST_WRITER"
    assert plan["next_actor"] == "test_writer"
    assert record.status == JobStatus.WRITING_TESTS
    assert constraints["patch_operation_hint"] == "create"
    assert constraints["missing_target_file"] == "backend/tests/test_project_setup.py"


def test_update_directory_target_recovery_returns_create_hint(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    target = tmp_path / "backend/tests/test_project_setup.py"
    target.mkdir(parents=True)
    patch = FilePatch(
        path="backend/tests/test_project_setup.py",
        operation="update",
        content="def test_project_setup() -> None:\n    assert 'project' in 'project setup'\n",
    )

    runner._apply_patches(record, "test_writer", [patch])

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert target.is_dir()
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert plan["next_actor"] == "planner"
    assert record.status == JobStatus.REPLANNING
    assert constraints["recovery_mode"] == "non_file_artifacts_replan"
    assert "patch_operation_hint" not in constraints
    assert constraints["missing_target_file"] == "backend/tests/test_project_setup.py"
    assert constraints["non_file_artifacts"] == ["backend/tests/test_project_setup.py"]
    metadata_constraints = record.spec.metadata["constraints"]
    assert metadata_constraints["recovery_next_actor"] == "planner"
    assert metadata_constraints["recovery_strategy"] == (
        "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    )
    assert "patch_operation_hint" not in metadata_constraints


def test_update_invalid_target_replans_without_create_hint(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    patch = FilePatch(
        path="../outside.py",
        operation="update",
        content="VALUE = 1\n",
    )

    runner._apply_patches(record, "implementer", [patch])

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert plan["next_actor"] == "planner"
    assert record.status == JobStatus.REPLANNING
    assert constraints["recovery_mode"] == "invalid_artifacts_replan"
    assert constraints["invalid_artifacts"] == ["../outside.py"]
    assert "patch_operation_hint" not in constraints
    assert not (tmp_path.parent / "outside.py").exists()


def test_missing_frontend_test_file_create_hint_rewrites_update_to_create(
    tmp_path: Path,
) -> None:
    test_path = "frontend/test/project_scaffold.test.tsx"
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "test_writer": TestWriterResult(
                summary="Create missing project scaffold frontend test.",
                changed_files=[test_path],
                patches=[
                    FilePatch(
                        path=test_path,
                        operation="update",
                        content=(
                            "import { describe, expect, it } from 'vitest'\n\n"
                            "describe('project scaffold', () => {\n"
                            "  it('exists', () => {\n"
                            "    const label = 'ACOS project scaffold is ready'\n"
                            "    expect(label).toContain('project scaffold')\n"
                            "  })\n"
                            "})\n"
                        ),
                    )
                ],
            ).model_dump(),
        },
    )
    record.spec.metadata["constraints"] = {
        "patch_operation_hint": "create",
        "missing_target_file": test_path,
    }
    task = PlannedTask(
        id="project-scaffold-test",
        title="Frontend scaffold smoke test",
        description="Create the missing frontend test file.",
        role="test_writer",
    )

    result = runner._run_test_writer_task(record, task, [], [])

    assert result.patches[0].operation == "create"
    assert (tmp_path / test_path).exists()
    saved_task = record.outputs["test_writer_tasks"][0]["task"]
    assert test_path in saved_task["target_files"]
    assert test_path in saved_task["required_artifacts"]
    retrieval_trace = "\n".join(
        item["action"] for item in record.outputs.get("retrieval_trace", [])
    )
    assert "missing_file_context" in retrieval_trace


def test_stage_test_gate_rewrites_declared_new_test_file_update_to_create(
    tmp_path: Path,
) -> None:
    test_path = "backend/test/scaffold.test.js"
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "test_writer": TestWriterResult(
                summary="Add scaffold backend tests.",
                changed_files=[test_path],
                patches=[
                    FilePatch(
                        path=test_path,
                        operation="update",
                        content=(
                            "describe('scaffold backend', () => {\n"
                            "  it('names the scaffold target', () => {\n"
                            "    const target = 'backend/test/scaffold.test.js'\n"
                            "    if (!target.includes('scaffold')) {\n"
                            "      throw new Error('missing scaffold target')\n"
                            "    }\n"
                            "  })\n"
                            "})\n"
                        ),
                    )
                ],
            ).model_dump(),
        },
    )
    task = PlannedTask(
        id="scaffold-backend",
        title="Scaffold backend project structure",
        description="Create backend package and source scaffold.",
        role="scaffold",
        target_files=["backend/package.json"],
        required_artifacts=["backend/package.json"],
    )
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="Created backend scaffold.",
        changed_files=["backend/package.json"],
        patches=[],
    )

    result = runner._run_stage_test_gate(record, task, [implementation], [])

    assert result.patches[0].operation == "create"
    assert (tmp_path / test_path).exists()
    assert "last_recoverable_error" not in record.runtime_state
    assert record.outputs["patch_operation_rewrites"] == [
        {
            "role": "test_writer",
            "path": test_path,
            "from": "update",
            "to": "create",
            "reason": "test_writer_declared_new_test_file",
            "stage": "structured_output",
        }
    ]


def test_recreate_target_files_recovery_waits_until_artifacts_exist(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-executor",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-1",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_TEST_WRITER",
        "next_status": JobStatus.WRITING_TESTS.value,
        "next_actor": "test_writer",
        "steps": ["RETURN_TO_TEST_WRITER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": ["backend/tests/test_project_setup.py"],
            "missing_target_file": "backend/tests/test_project_setup.py",
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert plan["status"] != "completed"
    assert plan["constraints"]["missing_artifacts"] == [
        "backend/tests/test_project_setup.py"
    ]
    assert record.status == JobStatus.WRITING_TESTS


def test_recreate_target_files_ignores_stale_runtime_missing_when_plan_has_target(
    tmp_path: Path,
) -> None:
    test_path = "frontend/test/project_scaffold.test.tsx"
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-ignore-stale-missing",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["missing_artifacts"] = ["stale.py"]
    record.runtime_state["recovery_plan"] = {
        "id": "plan-ignore-stale-missing",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_TEST_WRITER",
        "next_status": JobStatus.WRITING_TESTS.value,
        "next_actor": "test_writer",
        "steps": ["RETURN_TO_TEST_WRITER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "missing_target_file": test_path,
            "patch_operation_hint": "create",
            "required_artifacts": [test_path],
            "target_files": [test_path],
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert plan["status"] == "running"
    assert plan["next_actor"] == "test_writer"
    assert plan["next_status"] == JobStatus.WRITING_TESTS.value
    assert plan["constraints"]["return_to_role"] == "test_writer"
    assert plan["constraints"]["missing_artifacts"] == [test_path]
    assert "stale.py" not in plan["constraints"]["missing_artifacts"]
    assert record.status == JobStatus.WRITING_TESTS
    assert not (tmp_path / "stale.py").exists()


def test_recreate_target_files_routes_project_setup_bundle_to_scaffold(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-project-setup-owner",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-project-setup-owner",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [
                "backend/main.py",
                "backend/tests/test_project_setup.py",
            ],
            "target_files": [
                "backend/main.py",
                "backend/tests/test_project_setup.py",
            ],
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert plan["status"] == "running"
    assert plan["next_actor"] == "scaffold"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["return_to_role"] == "scaffold"
    assert plan["constraints"]["missing_artifacts"] == [
        "backend/main.py",
        "backend/tests/test_project_setup.py",
    ]
    assert record.status == JobStatus.IMPLEMENTING


def test_recreate_target_files_routes_app_backend_file_to_implementer(
    tmp_path: Path,
) -> None:
    source_path = "backend/auth.py"
    spec = JobSpec(
        request_text="Build auth backend",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-app-backend-owner",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-app-backend-owner",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [source_path],
            "target_files": [source_path],
            "missing_target_file": source_path,
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)
    assert not (tmp_path / source_path).exists()
    assert record.runtime_state["recovery_plan"]["status"] == "running"

    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert not (tmp_path / source_path).exists()
    assert plan["status"] == "completed"
    assert plan["next_actor"] == "implementer"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["return_to_role"] == "implementer"
    assert plan["constraints"]["deterministic_creation_attempted"] is True
    assert plan["constraints"]["deterministically_created_files"] == []
    assert plan["constraints"]["missing_artifacts"] == [source_path]
    assert "force_project_setup_scaffold" not in plan["constraints"]
    assert record.spec.metadata["constraints"]["recovery_next_actor"] == "implementer"
    assert "force_project_setup_scaffold" not in record.spec.metadata["constraints"]
    assert record.status == JobStatus.IMPLEMENTING


def test_recreate_target_files_routes_mixed_setup_and_app_source_to_implementer(
    tmp_path: Path,
) -> None:
    setup_path = "backend/main.py"
    source_path = "backend/auth.py"
    spec = JobSpec(
        request_text="Build auth backend",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-mixed-source-owner",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-mixed-source-owner",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [setup_path, source_path],
            "target_files": [setup_path, source_path],
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)
    assert not (tmp_path / setup_path).exists()
    assert not (tmp_path / source_path).exists()
    first_plan = record.runtime_state["recovery_plan"]
    assert first_plan["status"] == "running"
    assert first_plan["next_actor"] == "implementer"
    assert first_plan["constraints"]["return_to_role"] == "implementer"

    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert (tmp_path / setup_path).exists()
    assert not (tmp_path / source_path).exists()
    assert plan["status"] == "completed"
    assert plan["next_actor"] == "implementer"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["deterministically_created_files"] == [setup_path]
    assert plan["constraints"]["missing_artifacts"] == [source_path]
    assert "force_project_setup_scaffold" not in plan["constraints"]
    assert "force_project_setup_scaffold" not in record.spec.metadata["constraints"]
    assert record.status == JobStatus.IMPLEMENTING


def test_recreate_target_files_recovery_replans_invalid_artifact_paths(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-invalid-artifacts",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-invalid",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": ["../outside.py", "C:\\outside.py"],
            "target_files": ["../outside.py", "C:\\outside.py"],
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert plan["status"] == "completed"
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert plan["next_actor"] == "planner"
    assert plan["next_status"] == JobStatus.REPLANNING.value
    assert plan["constraints"]["invalid_artifacts"] == [
        "../outside.py",
        "C:/outside.py",
    ]
    assert plan["constraints"]["missing_artifacts"] == []
    assert record.runtime_state["planner_repair_requested"] is True
    assert record.status == JobStatus.REPLANNING
    assert record.spec.metadata["constraints"]["recovery_next_actor"] == "planner"
    assert record.spec.metadata["constraints"]["recovery_strategy"] == (
        "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    )
    assert "deterministic_creation_attempted" not in plan["constraints"]
    assert not (tmp_path.parent / "outside.py").exists()


def test_recovery_executor_sync_drops_stale_file_recovery_metadata(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-stale-file-context",
        metadata={
            "constraints": {
                "recovery_mode": "target_files_missing",
                "recovery_strategy": "RETURN_TO_TEST_WRITER",
                "recovery_next_actor": "test_writer",
                "recovery_next_status": JobStatus.WRITING_TESTS.value,
                "return_to_role": "test_writer",
                "patch_operation_hint": "create",
                "missing_target_file": "frontend/test/project_scaffold.test.tsx",
                "missing_artifacts": ["frontend/test/project_scaffold.test.tsx"],
                "failed_patch_role": "test_writer",
                "failed_patch_path": "frontend/test/project_scaffold.test.tsx",
                "failed_patch_operation": "update",
                "max_autonomous_stages": 12,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-invalid-now",
        "trigger": "target_files_invalid",
        "strategy": "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
        "next_status": JobStatus.REPLANNING.value,
        "next_actor": "planner",
        "steps": ["REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "recovery_mode": "invalid_artifacts_replan",
            "invalid_artifacts": ["../outside.py"],
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    constraints = record.spec.metadata["constraints"]
    assert record.runtime_state["recovery_plan"]["status"] == "completed"
    assert record.status == JobStatus.REPLANNING
    assert constraints["max_autonomous_stages"] == 12
    assert constraints["recovery_mode"] == "invalid_artifacts_replan"
    assert constraints["recovery_strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert constraints["recovery_next_actor"] == "planner"
    assert constraints["invalid_artifacts"] == ["../outside.py"]
    for stale_key in (
        "return_to_role",
        "patch_operation_hint",
        "missing_target_file",
        "missing_artifacts",
        "failed_patch_role",
        "failed_patch_path",
        "failed_patch_operation",
    ):
        assert stale_key not in constraints


def test_recreate_target_files_treats_directory_target_as_missing(
    tmp_path: Path,
) -> None:
    test_path = "frontend/test/project_scaffold.test.tsx"
    (tmp_path / test_path).mkdir(parents=True)
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-directory-target",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-directory-target",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_TEST_WRITER",
        "next_status": JobStatus.WRITING_TESTS.value,
        "next_actor": "test_writer",
        "steps": ["RETURN_TO_TEST_WRITER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [test_path],
            "target_files": [test_path],
            "missing_target_file": test_path,
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert (tmp_path / test_path).is_dir()
    assert plan["status"] == "completed"
    assert plan["next_actor"] == "planner"
    assert plan["next_status"] == JobStatus.REPLANNING.value
    assert plan["constraints"]["recovery_mode"] == "non_file_artifacts_replan"
    assert plan["constraints"]["non_file_artifacts"] == [test_path]
    assert plan["constraints"]["missing_artifacts"] == [test_path]
    assert "deterministically_created_files" not in plan["constraints"]
    assert record.runtime_state["planner_repair_requested"] is True
    assert record.status == JobStatus.REPLANNING


def test_recreate_target_files_returns_uncreated_implementation_file_to_owner(
    tmp_path: Path,
) -> None:
    source_path = "src/app.py"
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-source-owner",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-source-owner",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [source_path],
            "target_files": [source_path],
            "missing_target_file": source_path,
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)
    assert not (tmp_path / source_path).exists()
    assert record.runtime_state["recovery_plan"]["status"] == "running"

    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert not (tmp_path / source_path).exists()
    assert plan["status"] == "completed"
    assert plan["next_actor"] == "implementer"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["deterministic_creation_attempted"] is True
    assert plan["constraints"]["deterministically_created_files"] == []
    assert plan["constraints"]["missing_artifacts"] == [source_path]
    assert record.spec.metadata["constraints"]["recovery_next_actor"] == "implementer"
    assert record.spec.metadata["constraints"]["missing_artifacts"] == [source_path]
    assert record.status == JobStatus.IMPLEMENTING


def test_recreate_target_files_keeps_remaining_missing_after_partial_creation(
    tmp_path: Path,
) -> None:
    source_path = "src/app.py"
    test_path = "tests/test_app.py"
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/recovery-partial-create",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-partial-create",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [source_path, test_path],
            "target_files": [source_path, test_path],
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)
    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert (tmp_path / test_path).exists()
    assert not (tmp_path / source_path).exists()
    assert plan["status"] == "completed"
    assert plan["next_actor"] == "implementer"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["deterministic_creation_attempted"] is True
    assert plan["constraints"]["deterministically_created_files"] == [test_path]
    assert plan["constraints"]["missing_artifacts"] == [source_path]
    assert record.spec.metadata["constraints"]["recovery_next_actor"] == "implementer"
    assert record.spec.metadata["constraints"]["deterministically_created_files"] == [
        test_path
    ]
    assert record.spec.metadata["constraints"]["missing_artifacts"] == [source_path]
    assert record.status == JobStatus.IMPLEMENTING


def test_repeated_missing_test_file_is_created_deterministically_after_two_failures(
    tmp_path: Path,
) -> None:
    test_path = "frontend/test/project_scaffold.test.tsx"
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/repeated-missing-test",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RECOVERING)
    record.runtime_state["recovery_plan"] = {
        "id": "plan-frontend-test",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_TEST_WRITER",
        "next_status": JobStatus.WRITING_TESTS.value,
        "next_actor": "test_writer",
        "steps": ["RETURN_TO_TEST_WRITER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": [test_path],
            "target_files": [test_path],
            "missing_target_file": test_path,
            "patch_operation_hint": "create",
        },
    }
    executor = RecoveryExecutor()

    executor.execute_until_ready(record)
    assert not (tmp_path / test_path).exists()
    assert record.runtime_state["recovery_plan"]["status"] == "running"

    executor.execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert (tmp_path / test_path).exists()
    assert plan["status"] == "completed"
    assert plan["constraints"]["deterministically_created_files"] == [test_path]
    assert plan["constraints"]["missing_artifacts"] == []
    assert record.status == JobStatus.WRITING_TESTS
    assert record.history.count(JobStatus.WRITING_TESTS) <= 2


def test_project_setup_scaffold_creates_required_files(tmp_path: Path) -> None:
    runner, _environment, record = _runner(tmp_path)
    task = runner._normalize_project_setup_task_graph(
        record,
        _bad_project_setup_graph(),
    ).tasks[0]

    result = runner._run_project_setup_scaffold(record, task)

    assert result.status == ImplementationStatus.IMPLEMENTED
    for artifact in JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS:
        assert (tmp_path / artifact).exists(), artifact
    evidence = record.outputs["project_setup_scaffold"]["artifact_evidence"]
    assert all(item["exists"] for item in evidence)


def test_project_setup_scaffold_blocks_on_non_file_artifact(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    (tmp_path / "README.md").mkdir()
    task = runner._normalize_project_setup_task_graph(
        record,
        _bad_project_setup_graph(),
    ).tasks[0]

    result = runner._run_project_setup_scaffold(record, task)

    assert result.status == ImplementationStatus.BLOCKED
    scaffold = record.outputs["project_setup_scaffold"]
    readme_evidence = next(
        item for item in scaffold["artifact_evidence"] if item["path"] == "README.md"
    )
    assert readme_evidence["exists"] is False
    assert readme_evidence["path_exists"] is True
    assert readme_evidence["is_file"] is False
    assert "README.md" in scaffold["missing_artifacts"]
    assert scaffold["non_file_artifacts"] == ["README.md"]
    plan = record.runtime_state["recovery_plan"]
    assert plan["trigger"] == "required_artifacts_missing"
    assert "README.md" in plan["constraints"]["non_file_artifacts"]
    assert not (tmp_path / "backend/main.py").exists()


def test_completion_integrity_passes_non_file_artifacts_to_recovery(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    (tmp_path / "docs").mkdir()
    record.spec.metadata["constraints"] = {
        "require_completion_integrity": True,
        "require_test_evidence": True,
    }
    task_graph = TaskGraph(
        goal="Build docs-backed feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Build core",
                description="Create the implementation artifact.",
                role="implementer",
                target_files=["docs"],
                required_artifacts=["docs"],
            )
        ],
    )
    test_result = TestRunResult(
        success=True,
        command=["pytest"],
        output_excerpt="1 passed",
        exit_code=0,
        executed_test_count=1,
    )
    record.completed_task_ids.append("core")
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["test_run"] = test_result.model_dump()
    record.runtime_state["missing_artifacts"] = ["stale.py"]
    record.audit_events.append({"event": "verified"})
    record.checkpoints.append({"kind": "stage"})

    passed = runner._validate_completion_integrity(record, task_graph, test_result)

    assert passed is False
    report = record.outputs["completion_integrity"]
    assert "required_artifact_non_file:docs" in report["failure_reasons"]
    assert "target_file_non_file:docs" in report["failure_reasons"]
    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["trigger"] == "completion_integrity_failed"
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert record.status == JobStatus.REPLANNING
    assert constraints["required_artifacts"] == ["docs"]
    assert constraints["target_files"] == ["docs"]
    assert constraints["non_file_artifacts"] == ["docs"]
    assert "missing_artifacts" not in constraints


def test_failed_stage_enters_recovery_without_marking_task_complete(
    tmp_path: Path,
) -> None:
    test_path = "tests/test_feature.py"
    runner, _environment, record = _runner(
        tmp_path,
        scenario={
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Claimed implementation without changing files.",
                changed_files=[],
                patches=[],
            ).model_dump(),
            "test_writer": TestWriterResult(
                summary="Add smoke test.",
                changed_files=[test_path],
                patches=[
                    FilePatch(
                        path=test_path,
                        operation="create",
                        content="def test_smoke() -> None:\n    assert 1 == 1\n",
                    )
                ],
            ).model_dump(),
        },
        scripted_test_results=[TestRunResult(success=True)],
    )
    task_graph = TaskGraph(
        goal="Build feature",
        tasks=[
            PlannedTask(
                id="core",
                title="Core feature",
                description="Create the feature implementation.",
                role="implementer",
                acceptance_criteria=["Feature implementation exists."],
            )
        ],
    )

    runner._active_record = record
    try:
        _implementation_results, _test_writer_results, _test_result, stages = (
            runner._run_autonomous_task_loop(record, task_graph)
        )
    finally:
        runner._active_record = None

    assert stages[0]["status"] == "failed_for_recovery"
    assert stages[0]["failure_reason"] == "implementation_produced_no_changes"
    assert record.completed_task_ids == []
    assert record.status == JobStatus.DIAGNOSING
    assert record.runtime_state["recovery_plan"]["trigger"] == (
        "implementation_produced_no_changes"
    )


def test_zero_patch_implementation_stage_is_failed_for_recovery_even_if_tests_pass(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="No files changed",
        changed_files=[],
        patches=[],
    )
    stage_result = {
        "stage": 1,
        "task": PlannedTask(
            id="core",
            title="Core",
            description="Build core",
            role="implementer",
            required_artifacts=["backend/main.py"],
        ).model_dump(),
        "implementation": implementation.model_dump(),
        "test_writer_results": [],
        "change_summary": runner._build_stage_change_summary(implementation, []),
        "test_run": TestRunResult(success=True).model_dump(),
    }

    runner._record_stage_checkpoint(record, stage_result)

    assert stage_result["status"] == "failed_for_recovery"
    assert stage_result["failure_reason"] in {
        "implementation_produced_no_changes",
        "required_artifacts_missing",
    }
    assert record.status in {JobStatus.DIAGNOSING, JobStatus.REPLANNING}
    assert record.runtime_state["recovery_plan"]["trigger"] in {
        "implementation_produced_no_changes",
        "required_artifacts_missing",
    }


def test_stage_checkpoint_rejects_missing_target_file_without_required_artifact(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="Created a different feature module.",
        changed_files=["feature.py"],
        patches=[
            FilePatch(
                path="feature.py",
                operation="create",
                content="VALUE = 1\n",
            )
        ],
    )
    stage_result = {
        "stage": 1,
        "task": PlannedTask(
            id="core",
            title="Core",
            description="Build core",
            role="implementer",
            target_files=["src/app.py"],
            required_artifacts=[],
        ).model_dump(),
        "implementation": implementation.model_dump(),
        "test_writer_results": [],
        "change_summary": runner._build_stage_change_summary(implementation, []),
        "test_run": TestRunResult(success=True).model_dump(),
    }

    runner._record_stage_checkpoint(record, stage_result)

    assert stage_result["status"] == "failed_for_recovery"
    assert stage_result["failure_reason"] == "required_artifacts_missing"
    assert stage_result["missing_artifacts"] == ["src/app.py"]
    plan = record.runtime_state["recovery_plan"]
    assert plan["trigger"] == "required_artifacts_missing"
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert plan["constraints"]["target_files"] == ["src/app.py"]
    assert plan["constraints"]["missing_artifacts"] == ["src/app.py"]
    assert "required_artifacts" not in plan["constraints"]
    assert record.status == JobStatus.REPLANNING


def test_stage_checkpoint_rejects_invalid_required_artifact_paths(
    tmp_path: Path,
) -> None:
    runner, _environment, record = _runner(tmp_path)
    (tmp_path / "docs").mkdir()
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="Create feature module",
        changed_files=["feature.py"],
        patches=[
            FilePatch(
                path="feature.py",
                operation="create",
                content="VALUE = 1\n",
            )
        ],
    )
    stage_result = {
        "stage": 1,
        "task": PlannedTask(
            id="core",
            title="Core",
            description="Build core",
            role="implementer",
            required_artifacts=["../outside.py", "C:\\outside.py", "docs"],
        ).model_dump(),
        "implementation": implementation.model_dump(),
        "test_writer_results": [],
        "change_summary": runner._build_stage_change_summary(implementation, []),
        "test_run": TestRunResult(success=True).model_dump(),
    }

    runner._record_stage_checkpoint(record, stage_result)

    assert stage_result["status"] == "failed_for_recovery"
    assert stage_result["failure_reason"] == "required_artifacts_missing"
    assert stage_result["missing_artifacts"] == [
        "../outside.py",
        "C:\\outside.py",
        "docs",
    ]
    assert stage_result["invalid_artifacts"] == ["../outside.py", "C:\\outside.py"]


def test_frontend_unlimited_mode_sets_autonomous_until_done() -> None:
    source = Path("frontend/src/App.tsx").read_text(encoding="utf-8")

    assert "autonomous_until_done: unlimitedCycles" in source
    assert "'--autonomous-until-done'" in source
    assert "autonomous_until_done: true" in source
