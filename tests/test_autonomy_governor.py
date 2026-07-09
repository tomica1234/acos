from pathlib import Path

from packages.orchestrator.autonomy_governor import AutonomyGovernor, apply_recovery_plan
from packages.orchestrator.progress import summarize_job_progress
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus
from packages.schemas.tasks import PlannedTask, TaskGraph


def _record(tmp_path: Path, *, status: JobStatus, last_error: str) -> JobRecord:
    spec = JobSpec(
        job_id="autonomy-governor-job",
        request_text="Build the app autonomously.",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=status)
    record.last_error = last_error
    record.outputs["task_graph"] = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(
                id="project-init",
                title="Project init",
                description="Create the initial project",
                role="implementer",
            )
        ],
    ).model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": record.outputs["task_graph"]["tasks"][0],
            "test_run": {"success": False},
        }
    ]
    return record


def test_governor_continues_repeated_failure_with_strategy_change(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        status=JobStatus.STUCK,
        last_error="same_failure_threshold_reached",
    )
    record.failure_count = 2
    record.same_test_failure_count = 2
    summary = summarize_job_progress(record)

    decision = AutonomyGovernor().decide(record, summary)
    plan = apply_recovery_plan(record, decision)

    assert summary["resume"]["action"] == "recover_repeated_failure"
    assert summary["resume"]["can_auto_continue"] is True
    assert decision.action == "continue"
    assert decision.can_apply_automatically is True
    assert plan["strategy"] == "escalated_retry"
    assert record.spec.metadata["constraints"]["recovery_strategy"] == "escalated_retry"
    assert record.outputs["pm_interventions"][0]["applied"] is True


def test_governor_continues_completion_integrity_failure(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        status=JobStatus.BLOCKED,
        last_error="completion_integrity_failed:missing_test_evidence",
    )

    summary = summarize_job_progress(record)
    decision = AutonomyGovernor().decide(record, summary)

    assert summary["resume"]["action"] == "completion_audit_recovery"
    assert summary["resume"]["can_auto_continue"] is True
    assert decision.action == "continue"
    assert decision.strategy == "completion_audit"


def test_governor_enables_strict_gates_for_prd_quality_recovery(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path,
        status=JobStatus.BLOCKED,
        last_error="prd_quality_gate_failed:acceptance_tests",
    )
    record.outputs["prd_quality"] = {
        "passed": False,
        "missing": ["acceptance_tests"],
        "warnings": [],
    }

    summary = summarize_job_progress(record)
    decision = AutonomyGovernor().decide(record, summary)
    plan = apply_recovery_plan(record, decision)

    assert decision.strategy == "planning_repair_strategy_change"
    assert decision.next_actor == "pm"
    assert decision.constraints.items() >= {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
    }.items()
    assert plan["next_actor"] == "pm"
    assert record.spec.metadata["constraints"].items() >= {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
    }.items()


def test_governor_enables_strict_gates_for_task_graph_recovery(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path,
        status=JobStatus.BLOCKED,
        last_error="invalid_task_graph",
    )
    record.outputs["task_graph_validation"] = {
        "valid": False,
        "errors": [{"type": "missing_task_artifacts"}],
    }

    summary = summarize_job_progress(record)
    decision = AutonomyGovernor().decide(record, summary)

    assert decision.strategy == "task_graph_replanning"
    assert decision.next_actor == "planner"
    assert decision.constraints.items() >= {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
    }.items()


def test_policy_hard_stop_is_the_only_human_inspection_path(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        status=JobStatus.BLOCKED,
        last_error="policy_hard_stop:direct_main_write",
    )
    summary = summarize_job_progress(record)

    decision = AutonomyGovernor().decide(record, summary)

    assert summary["resume"]["action"] == "inspect_policy_hard_stop"
    assert summary["resume"]["can_auto_continue"] is False
    assert decision.action == "inspect"
    assert decision.can_apply_automatically is False
    assert decision.strategy == "policy_hard_stop"
