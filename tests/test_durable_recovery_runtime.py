from datetime import timedelta
from pathlib import Path

import pytest

from packages.mcp_client.fake import RepoServer
from packages.orchestrator.completion_verifier import DefinitionOfDoneVerifier
from packages.orchestrator.job_store import SQLiteJobStore
from packages.orchestrator.recovery_executor import RecoveryExecutor
from packages.orchestrator.recovery_governor import RecoveryGovernor
from packages.orchestrator.statuses import (
    is_hard_terminal_status,
    is_recoverable_status,
    is_settled_status,
)
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobRecord, JobSpec, utc_now
from packages.schemas.models import JobStatus
from packages.schemas.runtime import JobLease, RuntimeIssue, RuntimeIssueStatus, RuntimeIssueType, WorkerHeartbeat


def _record(tmp_path: Path, *, status: JobStatus = JobStatus.STUCK, error: str = "same_failure_threshold_reached") -> JobRecord:
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/durable-test",
    )
    return JobRecord(job_id=spec.job_id, spec=spec, status=status, last_error=error)


def test_status_split_treats_only_done_cancelled_policy_as_settled() -> None:
    assert is_hard_terminal_status(JobStatus.DONE)
    assert is_hard_terminal_status(JobStatus.CANCELLED)
    assert is_hard_terminal_status(JobStatus.POLICY_HARD_STOP)
    assert is_recoverable_status(JobStatus.BLOCKED)
    assert is_recoverable_status(JobStatus.STUCK)
    assert is_recoverable_status(JobStatus.FAILED)
    assert not is_settled_status(JobStatus.BLOCKED)
    assert not is_settled_status(JobStatus.STUCK)
    assert not is_settled_status(JobStatus.FAILED)


def test_recovery_executor_executes_and_persists_plan_steps(tmp_path: Path) -> None:
    store = SQLiteJobStore(tmp_path / "acos.sqlite3")
    record = store.update(_record(tmp_path))
    RecoveryGovernor().recover(record)

    executed = RecoveryExecutor(store).execute_until_ready(record)
    plan = executed.runtime_state["recovery_plan"]

    assert executed.status == JobStatus.DIAGNOSING
    assert plan["status"] == "completed"
    assert plan["current_step_index"] == len(plan["steps"])
    assert plan["executed_steps"] == plan["steps"]
    assert store.list_checkpoints(job_id=record.job_id)


def test_sqlite_job_store_persists_runtime_state_and_leases(tmp_path: Path) -> None:
    db_path = tmp_path / "acos.sqlite3"
    store = SQLiteJobStore(db_path)
    spec = JobSpec(request_text="Build it", repo_path=str(tmp_path))
    record = store.create(spec)
    record.runtime_state["recovery_plan"] = {"status": "pending"}
    store.update(record)
    store.save_checkpoint(
        CheckpointRecord(
            job_id=record.job_id,
            checkpoint_key="recovery:test",
            step_name="DIAGNOSE_FAILURE",
            idempotency_key="one",
            status="completed",
        )
    )
    store.save_runtime_issue(
        RuntimeIssue(
            id="issue-1",
            job_id=record.job_id,
            provider_key="local",
            issue_type=RuntimeIssueType.TIMEOUT,
            message="timeout",
            status=RuntimeIssueStatus.WAITING,
        )
    )
    store.save_worker_heartbeat(WorkerHeartbeat(worker_id="worker-1"))
    store.save_job_lease(
        JobLease(
            job_id=record.job_id,
            worker_id="worker-1",
            expires_at=utc_now() + timedelta(seconds=30),
        )
    )

    reopened = SQLiteJobStore(db_path)

    assert reopened.get(record.job_id).runtime_state["recovery_plan"]["status"] == "pending"
    assert reopened.list_checkpoints(job_id=record.job_id)
    assert reopened.get_runtime_issue("issue-1").status == RuntimeIssueStatus.WAITING
    assert reopened.list_worker_heartbeats()[0].worker_id == "worker-1"
    assert reopened.get_job_lease(record.job_id).worker_id == "worker-1"


def test_repo_server_patch_conflicts_are_recoverable_errors(tmp_path: Path) -> None:
    server = RepoServer(tmp_path)
    with pytest.raises(ValueError, match="target_files_missing"):
        server.apply_patch("missing.py", operation="update", content="x = 1\n")

    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = server.apply_patch(
        "app.py",
        operation="update",
        unified_diff="--- app.py\n+++ app.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
    )

    assert result["rollback"]["old_sha256"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_recovery_executor_prioritizes_implementation_for_mixed_missing_artifacts(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, status=JobStatus.RECOVERING, error="")
    record.runtime_state["recovery_plan"] = {
        "id": "plan-mixed-missing",
        "trigger": "target_files_missing",
        "strategy": "RETURN_TO_IMPLEMENTER",
        "next_status": JobStatus.IMPLEMENTING.value,
        "next_actor": "implementer",
        "steps": ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
        "current_step_index": 0,
        "status": "pending",
        "constraints": {
            "required_artifacts": ["src/app.py", "tests/test_app.py"],
            "target_files": ["src/app.py", "tests/test_app.py"],
        },
    }

    RecoveryExecutor().execute_until_ready(record)

    plan = record.runtime_state["recovery_plan"]
    assert plan["status"] == "running"
    assert plan["next_actor"] == "implementer"
    assert plan["next_status"] == JobStatus.IMPLEMENTING.value
    assert plan["constraints"]["return_to_role"] == "implementer"
    assert plan["constraints"]["missing_artifacts"] == [
        "src/app.py",
        "tests/test_app.py",
    ]
    assert record.status == JobStatus.IMPLEMENTING


def test_completion_verifier_reports_missing_evidence(tmp_path: Path) -> None:
    record = _record(tmp_path, status=JobStatus.FINALIZING, error="")
    record.outputs["task_graph"] = {
        "goal": "Build it",
        "tasks": [
            {
                "id": "core",
                "title": "Core",
                "description": "Core",
                "role": "implementer",
                "target_files": ["src/app.py"],
                "required_artifacts": ["README.md"],
            }
        ],
    }
    record.outputs["test_run"] = {"success": False}

    result = DefinitionOfDoneVerifier().verify(record)

    assert not result.passed
    assert "planned_task_not_done:core" in result.missing_evidence
    assert "target_file_missing:src/app.py" in result.missing_evidence
    assert "unit_tests_success" in result.missing_evidence


def test_completion_verifier_rejects_invalid_and_non_file_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    record = _record(tmp_path, status=JobStatus.FINALIZING, error="")
    record.completed_task_ids.append("core")
    record.outputs["task_graph"] = {
        "goal": "Build it",
        "tasks": [
            {
                "id": "core",
                "title": "Core",
                "description": "Core",
                "role": "implementer",
                "target_files": ["C:\\outside.py", "docs"],
                "required_artifacts": ["../outside.py"],
            }
        ],
    }
    record.outputs["test_run"] = {"success": True}
    record.audit_events.append({"event": "verified"})
    record.checkpoints.append({"kind": "stage"})

    result = DefinitionOfDoneVerifier().verify(record)

    assert not result.passed
    assert "required_artifact_invalid:../outside.py" in result.missing_evidence
    assert "target_file_invalid:C:\\outside.py" in result.missing_evidence
    assert "target_file_missing:docs" in result.missing_evidence
