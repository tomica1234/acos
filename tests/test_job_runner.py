from pathlib import Path

from packages.llm.errors import StructuredOutputError
from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
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
    SecurityReviewResult,
    SummaryResult,
    TestRunResult,
    TestWriterResult as TestWriterOutput,
)
from packages.schemas.jobs import JobSpec
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
                required_artifacts=["feature.py", "tests/test_feature.py"],
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
                    ),
                    PlannedTask(
                        id="double",
                        title="Create double helper",
                        description="Implement double.",
                        role="implementer",
                        depends_on=["add-one"],
                        acceptance_criteria=["double(4) returns 8"],
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
    assert record.outputs["task_graph_acceptance_enrichment"][
        "artifact_updated_task_ids"
    ] == ["auth", "word-sets"]

    validation = JobRunner._build_task_graph_validation(
        refined,
        prd=prd,
        require_acceptance_criteria=True,
        require_task_artifacts=True,
    )

    assert validation["valid"] is False
    assert "backend/main.py" in validation["unassigned_required_artifacts"]
    assert "tests/test_vocab_app.py" in validation["unassigned_required_artifacts"]


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
    assert validation["errors"] == [
        {"type": "missing_task_artifacts", "task_ids": ["tests"]},
        {"type": "missing_required_artifacts", "task_ids": ["tests"]},
        {"type": "missing_test_writer_target_files", "task_ids": ["tests"]},
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
    assert {
        "type": "missing_test_writer_tasks",
        "required_by": {
            "acceptance_tests": True,
            "test_focused_small_parts": False,
            "prd_test_required_artifacts": [],
        },
    } in validation["errors"]


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
        {"path": "feature.py", "expected_roles": ["implementer", "scaffold"]},
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
            "expected_roles": ["implementer", "scaffold"],
        },
    ]
    error_types = {item["type"] for item in validation["errors"]}
    assert "role_mismatched_target_files" in error_types
    assert "unowned_required_artifacts" in error_types


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
            "expected_roles": ["implementer", "scaffold"],
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
                        "content": "def test_placeholder() -> None:\n    assert True\n",
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
                        "content": "def test_placeholder() -> None:\n    assert True\n",
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
                        "content": "def test_placeholder() -> None:\n    assert True\n",
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
                        "content": "def test_placeholder() -> None:\n    assert True\n",
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
        "acceptance_test_count": 1,
        "acceptance_tests_cover_small_parts": False,
        "missing_acceptance_test_count": 1,
        "acceptance_tests_semantically_cover_small_parts": False,
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
        "definition_of_done_count": 1,
        "required_artifact_count": 2,
        "test_required_artifact_count": 1,
        "test_required_artifacts": ["tests/test_feature.py"],
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
    assert report["test_required_artifact_count"] == 0
    assert report["test_required_artifacts"] == []


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
    assert report["test_required_artifact_count"] == 0
    assert report["test_required_artifacts"] == []


def test_prd_quality_repair_logs_name_uncovered_small_parts() -> None:
    prd = PRD(
        title="English Vocab App",
        problem_statement="Students need vocabulary practice.",
        acceptance_tests=["GET /api/health returns 200"],
    )
    report = {
        "missing": ["acceptance_tests_cover_small_parts"],
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
        "Missing fields: acceptance_tests_cover_small_parts",
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


def test_semantic_tokens_split_camel_case_domain_terms() -> None:
    tokens = JobRunner._semantic_tokens("WordSet QuizQuestion UserProgress")

    assert {"word", "set", "quiz", "question", "user", "progress"}.issubset(tokens)


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
            "Teachers can manage word sets",
        ],
        acceptance_tests=[
            "Student can register and login",
            "Teacher can perform CRUD for word sets",
        ],
        definition_of_done=["All tests pass"],
        required_artifacts=[
            "backend/main.py",
            "frontend/src/App.tsx",
            "tests/test_auth_and_words.py",
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


def test_prd_quality_requires_anchor_token_overlap() -> None:
    prd = PRD(
        title="Auth App",
        problem_statement="Users need secure account access.",
        smallest_working_core=["Serve an authenticated app shell"],
        small_parts=["User authentication and roles"],
        incremental_milestones=["Users can sign in"],
        acceptance_tests=["User profile roles page renders"],
        definition_of_done=["All tests pass"],
        required_artifacts=["backend/main.py", "tests/test_auth.py"],
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
                    "Teachers can manage word sets",
                ],
                acceptance_tests=[
                    "Health endpoint returns 200",
                    "Database initializes successfully",
                ],
                definition_of_done=["All tests pass"],
                required_artifacts=[
                    "backend/main.py",
                    "frontend/src/App.tsx",
                    "tests/test_english_vocab.py",
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
    assert "planning_repair_strategy_change" not in record.spec.metadata["constraints"]
    assert "planning_repair_repeated_prd_missing" not in record.spec.metadata["constraints"]


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
            }
        },
    )
    record = store.create(spec)
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
    assert "planning_repair_strategy_change" not in record.spec.metadata["constraints"]
    assert (
        "planning_repair_repeated_task_graph_error_types"
        not in record.spec.metadata["constraints"]
    )


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
