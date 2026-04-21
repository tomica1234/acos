from __future__ import annotations

from packages.schemas.audit import AuditEvent
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobSpec
from packages.schemas.models import JobStatus, TaskComplexity, TaskStatus
from packages.schemas.runtime import RuntimeIssue, RuntimeIssueType
from packages.schemas.tasks import TaskRecord
from packages.orchestrator.job_store import SQLiteJobStore


def test_sqlite_job_store_persists_jobs_tasks_checkpoints_and_audit(tmp_path) -> None:
    db_path = tmp_path / ".acos" / "acos.sqlite3"
    store = SQLiteJobStore(db_path)
    spec = JobSpec(
        request_text="Implement durable runtime",
        repo_path=str(tmp_path),
        workspace_root=str(tmp_path),
        target_branch="acos/durable-runtime",
    )
    record = store.create(spec, status=JobStatus.QUEUED)
    record.status = JobStatus.RUNNING
    record.audit_events.append(
        AuditEvent(
            event_type="tool_call",
            role="implementer",
            action="repo_server.apply_patch",
            status="success",
            tool_name="repo_server.apply_patch",
        )
    )
    store.update(record)
    store.save_tasks(
        record.job_id,
        [
            TaskRecord(
                task_id="task-1",
                job_id=record.job_id,
                status=TaskStatus.RUNNING,
                title="Implement runtime",
                description="Add durable runtime support",
                role="implementer",
                complexity=TaskComplexity.HIGH,
            )
        ],
    )
    store.save_checkpoint(
        CheckpointRecord(
            job_id=record.job_id,
            task_id="task-1",
            checkpoint_key="task:task-1:implementer_completed",
            step_name="implementer",
            idempotency_key="task:task-1:implementer_completed",
            status="completed",
        )
    )
    store.save_runtime_issue(
        RuntimeIssue(
            id="issue-1",
            job_id=record.job_id,
            provider_key="local_qwen",
            model_key="qwen_35b",
            issue_type=RuntimeIssueType.TIMEOUT,
            message="provider timed out",
        )
    )
    store.record_notification(
        {
            "job_id": record.job_id,
            "kind": "runtime_wait",
            "channel": "console",
            "message": "provider wait",
        }
    )

    reopened = SQLiteJobStore(db_path)
    loaded = reopened.get(record.job_id)

    assert loaded.status == JobStatus.RUNNING
    assert reopened.list_tasks(record.job_id)[0].task_id == "task-1"
    assert reopened.get_checkpoint(
        job_id=record.job_id,
        task_id="task-1",
        checkpoint_key="task:task-1:implementer_completed",
    ) is not None
    assert reopened.get_runtime_issue("issue-1").provider_key == "local_qwen"
    assert reopened.list_notifications(job_id=record.job_id)[0]["kind"] == "runtime_wait"
    assert loaded.audit_events[0].tool_name == "repo_server.apply_patch"
