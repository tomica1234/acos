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


def test_recovery_governor_maps_agent_max_steps_to_strategy_change() -> None:
    record = _record(
        "Agent fixer exceeded max_steps=24 without a valid structured response; "
        "last_model=ornith_35b_q4; last_status=success"
    )
    RecoveryGovernor().recover(record)

    plan = record.runtime_state["recovery_plan"]
    assert record.status == JobStatus.STRATEGY_CHANGE
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
    assert record.runtime_state["recovery_plan"]["strategy"] == "RETURN_TO_TEST_WRITER"

    policy_record = _record(status=JobStatus.BLOCKED)
    runner._recover_record(
        policy_record,
        error=runner._quality_gate_recovery_error(
            QualityGateError("policy_denied:direct_main_write")
        ),
    )

    assert policy_record.status == JobStatus.POLICY_HARD_STOP
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
    assert recovered.history[-2:] == [JobStatus.RECOVERING, JobStatus.DIAGNOSING]
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
