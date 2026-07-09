from pathlib import Path

from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment, RepoServer
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.quality_gates import QualityGateError
from packages.orchestrator.recovery_governor import (
    RecoveryGovernor,
    is_hard_terminal_status,
)
from packages.orchestrator.task_graph_validation import (
    TASK_GRAPH_VALIDATION_CONTEXT_KEYS,
    TASK_GRAPH_VALIDATION_DETAIL_KEYS,
)
from packages.orchestrator.worker_daemon import WorkerDaemon
from packages.schemas.agent_outputs import TestRunResult
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus
from packages.schemas.tasks import PlannedTask

from tests.conftest import config_dir


def _record(last_error: str | None = None, status: JobStatus = JobStatus.STUCK) -> JobRecord:
    spec = JobSpec(
        request_text="Build it",
        repo_path=".",
        target_branch="acos/recovery-test",
    )
    return JobRecord(
        job_id=spec.job_id,
        spec=spec,
        status=status,
        last_error=last_error,
    )


def _runner(tmp_path: Path) -> tuple[JobRunner, FakeMCPEnvironment]:
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
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=InMemoryJobStore(),
    )
    return runner, environment


def test_recovery_governor_turns_same_failure_into_diagnosis_plan() -> None:
    record = _record("same_failure_threshold_reached")
    plan = RecoveryGovernor().recover(record)

    assert record.status == JobStatus.DIAGNOSING
    assert plan.strategy == "RETRY_WITH_DIFFERENT_STRATEGY"
    assert record.last_error is None
    assert record.runtime_state["last_recoverable_error"] == "same_failure_threshold_reached"
    assert record.runtime_state["current_recovery_event"]["error"] == (
        "same_failure_threshold_reached"
    )
    assert record.runtime_state["recovery_plan"]["trigger"] == "same_failure_threshold_reached"
    assert record.outputs["recovery_history"][-1]["steps"] == [
        "DIAGNOSE_FAILURE",
        "EXPAND_CONTEXT",
        "RETRY_WITH_DIFFERENT_STRATEGY",
    ]


def test_recovery_governor_maps_max_attempts_to_replanning() -> None:
    record = _record("max_attempts_exceeded")
    RecoveryGovernor().recover(record)

    assert record.status == JobStatus.REPLANNING
    assert record.runtime_state["recovery_plan"]["steps"] == [
        "DIAGNOSE_FAILURE",
        "REPLAN_TASK",
    ]


def test_recovery_governor_preserves_test_writer_dependency_context() -> None:
    record = _record("invalid_task_graph")
    plan = RecoveryGovernor().build_plan(
        record,
        error="invalid_task_graph",
        runtime_state={
            "task_graph_validation_errors": [
                "test_writer_dependency_semantic_mismatch",
                "test_writer_acceptance_dependency_mismatch",
            ],
            "test_writer_dependency_semantic_mismatches": [
                {
                    "task_id": "frontend-tests",
                    "depends_on": ["backend-api"],
                    "required_dependency_roles": ["implementer", "scaffold"],
                }
            ],
            "test_writer_acceptance_dependency_mismatches": [
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
            ],
        },
    )

    assert plan.constraints["test_writer_dependency_semantic_mismatches"] == [
        {
            "task_id": "frontend-tests",
            "depends_on": ["backend-api"],
            "required_dependency_roles": ["implementer", "scaffold"],
        }
    ]
    assert plan.constraints["test_writer_acceptance_dependency_mismatches"] == [
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


def test_recovery_governor_preserves_all_task_graph_validation_details() -> None:
    assert JobRunner.TASK_GRAPH_VALIDATION_DETAIL_KEYS == TASK_GRAPH_VALIDATION_DETAIL_KEYS
    assert JobRunner.TASK_GRAPH_VALIDATION_CONTEXT_KEYS == TASK_GRAPH_VALIDATION_CONTEXT_KEYS
    runtime_state = {
        key: [{"detail_key": key}]
        for key in TASK_GRAPH_VALIDATION_CONTEXT_KEYS
    }

    constraints = RecoveryGovernor._task_graph_context_constraints(runtime_state)

    for key in TASK_GRAPH_VALIDATION_CONTEXT_KEYS:
        assert constraints[key] == [{"detail_key": key}]


def test_recovery_governor_preserves_prd_quality_artifact_context() -> None:
    constraints = RecoveryGovernor._prd_quality_context_constraints(
        {
            "prd_quality_missing": ["required_implementation_artifacts"],
            "prd_quality_warnings": ["open_questions_present"],
            "prd_open_questions": ["Which UI?"],
            "non_observable_acceptance_tests": [
                {
                    "acceptance_test_index": 1,
                    "acceptance_test": "Create frontend UI",
                }
            ],
            "uncovered_smallest_working_core": [
                {
                    "core_index": 1,
                    "smallest_working_core": "Generate quizzes and track progress",
                    "missing_anchor_tokens": ["progress"],
                }
            ],
            "uncovered_incremental_milestone_small_parts": [
                {
                    "small_part_index": 2,
                    "small_part": "Word set CRUD operations",
                    "missing_anchor_tokens": ["crud"],
                }
            ],
            "uncovered_implementation_artifact_small_parts": [
                {
                    "small_part_index": 1,
                    "small_part": "Create frontend UI",
                    "missing_surfaces": ["frontend"],
                }
            ],
            "uncovered_implementation_artifact_domain_small_parts": [
                {
                    "small_part_index": 2,
                    "small_part": "Word set CRUD operations",
                    "required_domain_tokens": ["crud", "set", "word"],
                    "covered_domain_tokens": [],
                }
            ],
            "uncovered_test_artifact_domain_small_parts": [
                {
                    "small_part_index": 2,
                    "small_part": "Word set CRUD operations",
                    "required_domain_tokens": ["crud", "set", "word"],
                    "covered_domain_tokens": [],
                }
            ],
            "invalid_required_artifacts": ["../outside.py"],
            "required_incremental_milestone_count": 2,
            "required_small_part_count": 2,
            "prd_required_artifacts": [
                "README.md",
                "frontend/src/App.tsx",
                "tests/test_app.py",
            ],
            "source_required_artifacts": ["README.md", "frontend/src/App.tsx"],
            "implementation_required_artifacts": ["frontend/src/App.tsx"],
            "test_required_artifacts": ["tests/test_app.py"],
        }
    )

    assert constraints["prd_quality_missing"] == [
        "required_implementation_artifacts"
    ]
    assert constraints["implementation_required_artifacts"] == [
        "frontend/src/App.tsx"
    ]
    assert constraints["required_small_part_count"] == 2
    assert constraints["required_incremental_milestone_count"] == 2
    assert constraints["non_observable_acceptance_tests"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "Create frontend UI",
        }
    ]
    assert constraints["uncovered_smallest_working_core"] == [
        {
            "core_index": 1,
            "smallest_working_core": "Generate quizzes and track progress",
            "missing_anchor_tokens": ["progress"],
        }
    ]
    assert constraints["uncovered_incremental_milestone_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "missing_anchor_tokens": ["crud"],
        }
    ]
    assert constraints["uncovered_implementation_artifact_small_parts"] == [
        {
            "small_part_index": 1,
            "small_part": "Create frontend UI",
            "missing_surfaces": ["frontend"],
        }
    ]
    assert constraints["uncovered_implementation_artifact_domain_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "required_domain_tokens": ["crud", "set", "word"],
            "covered_domain_tokens": [],
        }
    ]
    assert constraints["uncovered_test_artifact_domain_small_parts"] == [
        {
            "small_part_index": 2,
            "small_part": "Word set CRUD operations",
            "required_domain_tokens": ["crud", "set", "word"],
            "covered_domain_tokens": [],
        }
    ]
    assert constraints["source_required_artifacts"] == [
        "README.md",
        "frontend/src/App.tsx",
    ]
    assert constraints["test_required_artifacts"] == ["tests/test_app.py"]


def test_recovery_governor_maps_agent_max_steps_to_strategy_change() -> None:
    record = _record(
        "Agent fixer exceeded max_steps=24 without a valid structured response; "
        "last_model=ornith_35b_q4; last_status=success"
    )
    RecoveryGovernor().recover(record)

    plan = record.runtime_state["recovery_plan"]
    assert record.status == JobStatus.STRATEGY_CHANGE
    assert record.last_error is None
    assert record.runtime_state["last_recoverable_error"].startswith(
        "Agent fixer exceeded max_steps=24"
    )
    assert plan["trigger"] == "agent_max_steps_exceeded"
    assert plan["strategy"] == "RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD"
    assert plan["next_actor"] == "fixer"
    assert plan["constraints"] == {
        "recovery_mode": "agent_max_steps_structured_output",
        "max_steps_exceeded_role": "fixer",
        "avoid_tool_loop": True,
        "force_structured_output": True,
        "retry_small_scope": True,
        "expand_context": True,
    }


def test_recovery_governor_maps_review_attempts_to_revision_paths() -> None:
    governor = RecoveryGovernor()
    design = _record("design_review_max_attempts_exceeded")
    acceptance = _record("acceptance_review_max_attempts_exceeded")

    governor.recover(design)
    governor.recover(acceptance)

    assert design.runtime_state["recovery_plan"]["strategy"] == "REVISE_PRD_AND_ARCHITECTURE"
    assert design.runtime_state["recovery_plan"]["next_actor"] == "pm"
    assert acceptance.runtime_state["recovery_plan"]["strategy"] == (
        "SPLIT_TASK_OR_REDEFINE_ACCEPTANCE"
    )
    assert acceptance.runtime_state["recovery_plan"]["next_actor"] == "planner"


def test_recovery_governor_preserves_artifact_stage_context() -> None:
    record = _record("required_artifacts_missing:stage:core")

    RecoveryGovernor().recover(
        record,
        error="required_artifacts_missing:stage:core",
        runtime_state={
            "failed_stage": 2,
            "failed_task_id": "core",
            "stage_failure_reason": "required_artifacts_missing",
            "required_artifacts": ["backend/main.py"],
            "target_files": ["backend/main.py"],
            "missing_artifacts": ["backend/main.py"],
            "invalid_artifacts": ["../outside.py"],
        },
    )

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert record.status == JobStatus.REPLANNING
    assert constraints["recovery_mode"] == "required_artifacts_replan"
    assert constraints["failed_stage"] == 2
    assert constraints["failed_task_id"] == "core"
    assert constraints["stage_failure_reason"] == "required_artifacts_missing"
    assert constraints["required_artifacts"] == ["backend/main.py"]
    assert constraints["target_files"] == ["backend/main.py"]
    assert constraints["missing_artifacts"] == ["backend/main.py"]
    assert constraints["invalid_artifacts"] == ["../outside.py"]


def test_recovery_governor_preserves_completion_missing_task_context() -> None:
    record = _record("completion_integrity_failed:missing_tasks:core-tests")

    RecoveryGovernor().recover(
        record,
        error="completion_integrity_failed:missing_tasks:core-tests",
        runtime_state={
            "completion_integrity_failure_reasons": [
                "missing_tasks:core-tests|prd-tests"
            ],
            "missing_task_ids": ["core-tests", "prd-tests"],
        },
    )

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["strategy"] == "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"
    assert record.status == JobStatus.REPLANNING
    assert constraints["recovery_mode"] == "required_artifacts_replan"
    assert constraints["missing_task_ids"] == ["core-tests", "prd-tests"]


def test_recovery_governor_preserves_unmet_dependency_context() -> None:
    record = _record("unmet_task_dependencies:core")

    RecoveryGovernor().recover(
        record,
        error="unmet_task_dependencies:core",
        runtime_state={
            "failed_task_id": "core-tests",
            "unmet_dependencies": ["core"],
        },
    )

    plan = record.runtime_state["recovery_plan"]
    constraints = plan["constraints"]
    assert plan["strategy"] == "REPLAN_TASK"
    assert record.status == JobStatus.REPLANNING
    assert constraints["recovery_mode"] == "task_graph_repair"
    assert constraints["failed_task_id"] == "core-tests"
    assert constraints["unmet_dependencies"] == ["core"]


def test_recovery_governor_drops_stale_patch_context_for_artifact_replan() -> None:
    record = _record("required_artifacts_missing:stage:core")

    RecoveryGovernor().recover(
        record,
        error="required_artifacts_missing:stage:core",
        runtime_state={
            "failed_stage": 2,
            "failed_task_id": "core",
            "stage_failure_reason": "required_artifacts_missing",
            "required_artifacts": ["backend/main.py"],
            "target_files": ["backend/main.py"],
            "missing_artifacts": ["backend/main.py"],
            "failed_patch_role": "test_writer",
            "failed_patch_path": "frontend/test/project_scaffold.test.tsx",
            "failed_patch_operation": "update",
            "missing_target_file": "frontend/test/project_scaffold.test.tsx",
        },
    )

    constraints = record.runtime_state["recovery_plan"]["constraints"]
    assert constraints["failed_stage"] == 2
    assert constraints["failed_task_id"] == "core"
    assert constraints["missing_artifacts"] == ["backend/main.py"]
    assert "failed_patch_role" not in constraints
    assert "failed_patch_path" not in constraints
    assert "failed_patch_operation" not in constraints
    assert "missing_target_file" not in constraints


def test_recovery_governor_routes_case_insensitive_missing_test_target() -> None:
    missing = "Frontend/Test/App.Spec.tsx"
    record = _record(f"target_files_missing:update target does not exist: {missing}")

    RecoveryGovernor().recover(record)

    plan = record.runtime_state["recovery_plan"]
    assert record.status == JobStatus.WRITING_TESTS
    assert plan["strategy"] == "RETURN_TO_TEST_WRITER"
    assert plan["next_actor"] == "test_writer"
    assert plan["constraints"]["missing_target_file"] == missing


def test_quality_gate_error_is_recoverable_unless_policy_denied(tmp_path: Path) -> None:
    runner, _environment = _runner(tmp_path)
    record = _record(status=JobStatus.TESTING)

    runner._recover_record(
        record,
        error=runner._quality_gate_recovery_error(
            QualityGateError("Fixer attempted to weaken tests")
        ),
    )

    assert record.status == JobStatus.WRITING_TESTS
    assert record.last_error is None
    assert record.runtime_state["last_recoverable_error"] == (
        "test_patch_quality_failed:Fixer attempted to weaken tests"
    )
    assert record.runtime_state["recovery_plan"]["strategy"] == "RETURN_TO_TEST_WRITER"

    policy_record = _record(status=JobStatus.BLOCKED)
    runner._recover_record(
        policy_record,
        error=runner._quality_gate_recovery_error(
            QualityGateError("policy_denied:direct_main_write")
        ),
    )

    assert policy_record.status == JobStatus.POLICY_HARD_STOP
    assert policy_record.last_error == "policy_denied:direct_main_write"
    assert is_hard_terminal_status(policy_record.status)


def test_gather_relevant_files_uses_targets_artifacts_and_failure_logs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/app.py").write_text("VALUE = 0\n", encoding="utf-8")
    (tmp_path / "tests/test_app.py").write_text("from src.app import VALUE\n", encoding="utf-8")
    (tmp_path / "docs/spec.md").write_text("# Spec\n", encoding="utf-8")
    runner, environment = _runner(tmp_path)
    environment.repo_server.modified_files.add("docs/spec.md")
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        target_branch="acos/context-test",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec)
    record.outputs["test_run"] = TestRunResult(
        success=False,
        output_excerpt="FAILED tests/test_app.py::test_value\nsrc/app.py:1: AssertionError",
        exit_code=1,
    ).model_dump()
    task = PlannedTask(
        id="core",
        title="Core",
        description="Build core",
        role="implementer",
        target_files=["src/app.py"],
        required_artifacts=["docs/spec.md"],
    )

    files = runner._gather_relevant_files("implementer", record=record, task=task)

    assert "src/app.py" in files
    assert "tests/test_app.py" in files
    assert "docs/spec.md" in files
    assert "task.target_files" in files["__retrieval_trace__.txt"]
    assert "failure_log" in files["__retrieval_trace__.txt"]
    assert "git.modified_files" in files["__retrieval_trace__.txt"]
    assert record.runtime_state["retrieval_trace"]


def test_repo_server_search_text_returns_line_number_and_context(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text(
        "before\nneedle = 1\nafter\n",
        encoding="utf-8",
    )
    server = RepoServer(tmp_path)

    result = server.search_text("needle", context_lines=1)

    match = result["matches"][0]
    assert match["path"] == "src/app.py"
    assert match["line_number"] == 2
    assert match["before"] == ["before"]
    assert match["match"] == "needle = 1"
    assert match["after"] == ["after"]


def test_worker_daemon_recovers_blocked_stuck_failed_before_processing(tmp_path: Path) -> None:
    runner, _environment = _runner(tmp_path)
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        target_branch="acos/worker-test",
    )
    record = runner.store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    runner.store.update(record)
    daemon = WorkerDaemon(runner=runner, store=runner.store)

    recovered = daemon.normalize_before_processing(record)

    assert recovered.status == JobStatus.DIAGNOSING
    assert JobStatus.RECOVERING in recovered.history
    assert recovered.history[-1] == JobStatus.DIAGNOSING
    assert recovered.runtime_state["recovery_plan"]["strategy"] == (
        "RETRY_WITH_DIFFERENT_STRATEGY"
    )


def test_only_done_cancelled_and_policy_hard_stop_are_hard_terminal() -> None:
    assert is_hard_terminal_status(JobStatus.DONE)
    assert is_hard_terminal_status(JobStatus.CANCELLED)
    assert is_hard_terminal_status(JobStatus.POLICY_HARD_STOP)
    assert not is_hard_terminal_status(JobStatus.BLOCKED)
    assert not is_hard_terminal_status(JobStatus.STUCK)
    assert not is_hard_terminal_status(JobStatus.FAILED)
