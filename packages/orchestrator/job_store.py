"""Durable job persistence backends."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from packages.schemas.audit import AuditEvent
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus, TaskComplexity, TaskStatus
from packages.schemas.runtime import JobLease, RuntimeIssue, WorkerHeartbeat
from packages.schemas.tasks import PlannedTask, TaskGraph, TaskRecord


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _json_loads(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    return json.loads(payload)


def _derive_title(spec: JobSpec) -> str:
    if spec.title:
        return spec.title
    metadata_title = spec.metadata.get("title")
    if isinstance(metadata_title, str) and metadata_title.strip():
        return metadata_title.strip()
    first_line = spec.request_text.strip().splitlines()[0] if spec.request_text.strip() else "ACOS job"
    return first_line[:120]


class JobStore(Protocol):
    """Common interface for durable job persistence."""

    def create(self, spec: JobSpec, *, status: JobStatus | None = None) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord: ...
    def update(self, record: JobRecord) -> JobRecord: ...
    def list_jobs(self, *, statuses: Sequence[JobStatus] | None = None) -> list[JobRecord]: ...
    def save_tasks(self, job_id: str, tasks: Sequence[TaskRecord]) -> None: ...
    def upsert_task(self, task: TaskRecord) -> TaskRecord: ...
    def list_tasks(self, job_id: str) -> list[TaskRecord]: ...
    def get_task(self, job_id: str, task_id: str) -> TaskRecord: ...
    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord: ...
    def get_checkpoint(
        self, *, job_id: str, checkpoint_key: str, task_id: str | None = None
    ) -> CheckpointRecord | None: ...
    def list_checkpoints(
        self, *, job_id: str, task_id: str | None = None
    ) -> list[CheckpointRecord]: ...
    def save_runtime_issue(self, issue: RuntimeIssue) -> RuntimeIssue: ...
    def get_runtime_issue(self, issue_id: str) -> RuntimeIssue: ...
    def list_runtime_issues(
        self, *, job_id: str | None = None, status: str | None = None
    ) -> list[RuntimeIssue]: ...
    def save_worker_heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeat: ...
    def list_worker_heartbeats(self) -> list[WorkerHeartbeat]: ...
    def save_job_lease(self, lease: JobLease) -> JobLease: ...
    def get_job_lease(self, job_id: str) -> JobLease | None: ...
    def release_job_lease(self, job_id: str) -> None: ...
    def list_job_leases(self) -> list[JobLease]: ...
    def record_notification(self, payload: dict[str, Any]) -> None: ...
    def list_notifications(self, *, job_id: str | None = None) -> list[dict[str, Any]]: ...


class InMemoryJobStore:
    """In-memory store with optional JSON persistence."""

    def __init__(self, backing_path: str | Path | None = None) -> None:
        self._records: dict[str, JobRecord] = {}
        self._tasks: dict[str, dict[str, TaskRecord]] = {}
        self._checkpoints: dict[str, list[CheckpointRecord]] = {}
        self._runtime_issues: dict[str, RuntimeIssue] = {}
        self._heartbeats: dict[str, WorkerHeartbeat] = {}
        self._leases: dict[str, JobLease] = {}
        self._notifications: list[dict[str, Any]] = []
        self.backing_path = Path(backing_path) if backing_path is not None else None
        self._load()

    def _load(self) -> None:
        if self.backing_path is None or not self.backing_path.exists():
            return
        payload = json.loads(self.backing_path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            record = JobRecord.model_validate(item)
            self._records[record.job_id] = record
        for item in payload.get("tasks", []):
            task = TaskRecord.model_validate(item)
            self._tasks.setdefault(task.job_id, {})[task.task_id] = task
        for item in payload.get("checkpoints", []):
            checkpoint = CheckpointRecord.model_validate(item)
            self._checkpoints.setdefault(checkpoint.job_id, []).append(checkpoint)
        for item in payload.get("runtime_issues", []):
            issue = RuntimeIssue.model_validate(item)
            self._runtime_issues[issue.id] = issue
        for item in payload.get("heartbeats", []):
            heartbeat = WorkerHeartbeat.model_validate(item)
            self._heartbeats[heartbeat.worker_id] = heartbeat
        for item in payload.get("leases", []):
            lease = JobLease.model_validate(item)
            self._leases[lease.job_id] = lease
        self._notifications = list(payload.get("notifications", []))

    def _flush(self) -> None:
        if self.backing_path is None:
            return
        self.backing_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [record.model_dump(mode="json") for record in self._records.values()],
            "tasks": [
                task.model_dump(mode="json")
                for tasks in self._tasks.values()
                for task in tasks.values()
            ],
            "checkpoints": [
                checkpoint.model_dump(mode="json")
                for checkpoints in self._checkpoints.values()
                for checkpoint in checkpoints
            ],
            "runtime_issues": [
                issue.model_dump(mode="json") for issue in self._runtime_issues.values()
            ],
            "heartbeats": [
                heartbeat.model_dump(mode="json") for heartbeat in self._heartbeats.values()
            ],
            "leases": [lease.model_dump(mode="json") for lease in self._leases.values()],
            "notifications": list(self._notifications),
        }
        self.backing_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def create(self, spec: JobSpec, *, status: JobStatus | None = None) -> JobRecord:
        initial_status = status or JobStatus.SUBMITTED
        record = JobRecord(
            job_id=spec.job_id,
            title=_derive_title(spec),
            spec=spec,
            status=initial_status,
            history=[initial_status],
        )
        self._records[record.job_id] = record
        self._flush()
        return record

    def get(self, job_id: str) -> JobRecord:
        return self._records[job_id]

    def update(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        self._records[record.job_id] = record
        self._sync_tasks_from_record(record)
        self._flush()
        return record

    def list_jobs(self, *, statuses: Sequence[JobStatus] | None = None) -> list[JobRecord]:
        if statuses is None:
            return list(self._records.values())
        allowed = set(statuses)
        return [record for record in self._records.values() if record.status in allowed]

    def save_tasks(self, job_id: str, tasks: Sequence[TaskRecord]) -> None:
        self._tasks[job_id] = {task.task_id: task for task in tasks}
        self._flush()

    def upsert_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        self._tasks.setdefault(task.job_id, {})[task.task_id] = task
        self._flush()
        return task

    def list_tasks(self, job_id: str) -> list[TaskRecord]:
        return sorted(
            self._tasks.get(job_id, {}).values(),
            key=lambda item: item.created_at,
        )

    def get_task(self, job_id: str, task_id: str) -> TaskRecord:
        return self._tasks[job_id][task_id]

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        checkpoint.updated_at = utc_now()
        self._checkpoints.setdefault(checkpoint.job_id, [])
        items = [
            item
            for item in self._checkpoints[checkpoint.job_id]
            if item.id != checkpoint.id
        ]
        items.append(checkpoint)
        self._checkpoints[checkpoint.job_id] = items
        self._flush()
        return checkpoint

    def get_checkpoint(
        self, *, job_id: str, checkpoint_key: str, task_id: str | None = None
    ) -> CheckpointRecord | None:
        matches = [
            checkpoint
            for checkpoint in self._checkpoints.get(job_id, [])
            if checkpoint.checkpoint_key == checkpoint_key
            and (task_id is None or checkpoint.task_id == task_id)
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: item.updated_at)
        return matches[-1]

    def list_checkpoints(
        self, *, job_id: str, task_id: str | None = None
    ) -> list[CheckpointRecord]:
        items = self._checkpoints.get(job_id, [])
        if task_id is None:
            return list(items)
        return [item for item in items if item.task_id == task_id]

    def save_runtime_issue(self, issue: RuntimeIssue) -> RuntimeIssue:
        issue.updated_at = utc_now()
        self._runtime_issues[issue.id] = issue
        self._flush()
        return issue

    def get_runtime_issue(self, issue_id: str) -> RuntimeIssue:
        return self._runtime_issues[issue_id]

    def list_runtime_issues(
        self, *, job_id: str | None = None, status: str | None = None
    ) -> list[RuntimeIssue]:
        items = list(self._runtime_issues.values())
        if job_id is not None:
            items = [item for item in items if item.job_id == job_id]
        if status is not None:
            items = [item for item in items if item.status.value == status or item.status == status]
        return items

    def save_worker_heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeat:
        heartbeat.heartbeat_at = utc_now()
        self._heartbeats[heartbeat.worker_id] = heartbeat
        self._flush()
        return heartbeat

    def list_worker_heartbeats(self) -> list[WorkerHeartbeat]:
        return list(self._heartbeats.values())

    def save_job_lease(self, lease: JobLease) -> JobLease:
        self._leases[lease.job_id] = lease
        self._flush()
        return lease

    def get_job_lease(self, job_id: str) -> JobLease | None:
        return self._leases.get(job_id)

    def release_job_lease(self, job_id: str) -> None:
        self._leases.pop(job_id, None)
        self._flush()

    def list_job_leases(self) -> list[JobLease]:
        return list(self._leases.values())

    def record_notification(self, payload: dict[str, Any]) -> None:
        self._notifications.append(dict(payload))
        self._flush()

    def list_notifications(self, *, job_id: str | None = None) -> list[dict[str, Any]]:
        if job_id is None:
            return list(self._notifications)
        return [item for item in self._notifications if item.get("job_id") == job_id]

    def _sync_tasks_from_record(self, record: JobRecord) -> None:
        task_graph = record.outputs.get("task_graph") or record.outputs.get("planner")
        if not isinstance(task_graph, dict):
            return
        tasks = task_graph.get("tasks")
        if not isinstance(tasks, list):
            return
        self.save_tasks(
            record.job_id,
            [
                TaskRecord.from_planned_task(
                    job_id=record.job_id,
                    task=PlannedTask.model_validate(task),
                )
                for task in tasks
            ],
        )


class SQLiteJobStore:
    """SQLite-backed store for durable ACOS execution state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_branch TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    workspace_root TEXT,
                    current_phase TEXT,
                    current_task_id TEXT,
                    last_error TEXT,
                    runtime_error TEXT,
                    provider_status TEXT,
                    failure_count INTEGER NOT NULL,
                    same_test_failure_count INTEGER NOT NULL,
                    pending_approval_id TEXT,
                    pending_runtime_issue_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    heartbeat_at TEXT,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    metadata_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    runtime_state_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    spec_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    role TEXT NOT NULL,
                    complexity TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    target_files_json TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    pending_approval_id TEXT,
                    pending_runtime_issue_id TEXT,
                    last_error TEXT,
                    checkpoint_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    role TEXT,
                    requested_by TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    proposed_action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    approval_token_hash TEXT,
                    approver TEXT,
                    resolution_reason TEXT,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    step TEXT,
                    provider_key TEXT,
                    model_key TEXT,
                    tool_name TEXT,
                    input_hash TEXT,
                    output_hash TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    checkpoint_key TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_issues (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    provider_key TEXT NOT NULL,
                    model_key TEXT,
                    issue_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    next_retry_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS worker_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_leases (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_call_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    role TEXT NOT NULL,
                    model_key TEXT,
                    provider_key TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_call_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    role TEXT NOT NULL,
                    tool_name TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    channel TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def create(self, spec: JobSpec, *, status: JobStatus | None = None) -> JobRecord:
        initial_status = status or JobStatus.SUBMITTED
        record = JobRecord(
            job_id=spec.job_id,
            title=_derive_title(spec),
            spec=spec,
            status=initial_status,
            history=[initial_status],
        )
        return self.update(record)

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def update(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, title, status, target_branch, repo_path, workspace_root, current_phase,
                    current_task_id, last_error, runtime_error, provider_status,
                    failure_count, same_test_failure_count, pending_approval_id,
                    pending_runtime_issue_id, created_at, updated_at, started_at,
                    completed_at, heartbeat_at, lease_owner, lease_expires_at,
                    metadata_json, outputs_json, runtime_state_json, history_json, spec_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    target_branch=excluded.target_branch,
                    repo_path=excluded.repo_path,
                    workspace_root=excluded.workspace_root,
                    current_phase=excluded.current_phase,
                    current_task_id=excluded.current_task_id,
                    last_error=excluded.last_error,
                    runtime_error=excluded.runtime_error,
                    provider_status=excluded.provider_status,
                    failure_count=excluded.failure_count,
                    same_test_failure_count=excluded.same_test_failure_count,
                    pending_approval_id=excluded.pending_approval_id,
                    pending_runtime_issue_id=excluded.pending_runtime_issue_id,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at,
                    heartbeat_at=excluded.heartbeat_at,
                    lease_owner=excluded.lease_owner,
                    lease_expires_at=excluded.lease_expires_at,
                    metadata_json=excluded.metadata_json,
                    outputs_json=excluded.outputs_json,
                    runtime_state_json=excluded.runtime_state_json,
                    history_json=excluded.history_json,
                    spec_json=excluded.spec_json
                """,
                (
                    record.job_id,
                    record.title or _derive_title(record.spec),
                    record.status.value,
                    record.spec.target_branch,
                    record.spec.repo_path,
                    record.spec.workspace_root,
                    record.current_phase,
                    record.current_task_id,
                    record.last_error,
                    record.runtime_error,
                    record.provider_status,
                    record.failure_count,
                    record.same_test_failure_count,
                    record.pending_approval_id,
                    record.pending_runtime_issue_id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.started_at.isoformat() if record.started_at else None,
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.heartbeat_at.isoformat() if record.heartbeat_at else None,
                    record.lease_owner,
                    record.lease_expires_at.isoformat() if record.lease_expires_at else None,
                    _json_dumps(record.spec.metadata),
                    _json_dumps(record.outputs),
                    _json_dumps(record.runtime_state),
                    _json_dumps([status.value for status in record.history]),
                    _json_dumps(record.spec.model_dump(mode="json")),
                ),
            )
            self._sync_audit_events(conn, record)
            self._sync_tasks_from_record(conn, record)
            conn.commit()
        return record

    def list_jobs(self, *, statuses: Sequence[JobStatus] | None = None) -> list[JobRecord]:
        query = "SELECT * FROM jobs"
        params: list[str] = []
        if statuses:
            query += " WHERE status IN (" + ",".join("?" for _ in statuses) + ")"
            params = [status.value for status in statuses]
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def save_tasks(self, job_id: str, tasks: Sequence[TaskRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE job_id = ?", (job_id,))
            for task in tasks:
                self._insert_task(conn, task)
            conn.commit()

    def upsert_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        with self._connect() as conn:
            self._insert_task(conn, task)
            conn.commit()
        return task

    def list_tasks(self, job_id: str) -> list[TaskRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE job_id = ? ORDER BY created_at ASC",
                (job_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_task(self, job_id: str, task_id: str) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE job_id = ? AND id = ?",
                (job_id, task_id),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row_to_task(row)

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        checkpoint.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(
                    id, job_id, task_id, checkpoint_key, step_name, idempotency_key,
                    status, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    task_id=excluded.task_id,
                    checkpoint_key=excluded.checkpoint_key,
                    step_name=excluded.step_name,
                    idempotency_key=excluded.idempotency_key,
                    status=excluded.status,
                    result_json=excluded.result_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    checkpoint.id,
                    checkpoint.job_id,
                    checkpoint.task_id,
                    checkpoint.checkpoint_key,
                    checkpoint.step_name,
                    checkpoint.idempotency_key,
                    checkpoint.status,
                    _json_dumps(checkpoint.result_json),
                    checkpoint.created_at.isoformat(),
                    checkpoint.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return checkpoint

    def get_checkpoint(
        self, *, job_id: str, checkpoint_key: str, task_id: str | None = None
    ) -> CheckpointRecord | None:
        query = "SELECT * FROM checkpoints WHERE job_id = ? AND checkpoint_key = ?"
        params: list[str | None] = [job_id, checkpoint_key]
        if task_id is None:
            query += " AND task_id IS NULL"
        else:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return self._row_to_checkpoint(row) if row is not None else None

    def list_checkpoints(
        self, *, job_id: str, task_id: str | None = None
    ) -> list[CheckpointRecord]:
        query = "SELECT * FROM checkpoints WHERE job_id = ?"
        params: list[str | None] = [job_id]
        if task_id is not None:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def save_runtime_issue(self, issue: RuntimeIssue) -> RuntimeIssue:
        issue.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_issues(
                    id, job_id, provider_key, model_key, issue_type, message, status,
                    retry_count, next_retry_at, created_at, updated_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    provider_key=excluded.provider_key,
                    model_key=excluded.model_key,
                    issue_type=excluded.issue_type,
                    message=excluded.message,
                    status=excluded.status,
                    retry_count=excluded.retry_count,
                    next_retry_at=excluded.next_retry_at,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    resolved_at=excluded.resolved_at
                """,
                (
                    issue.id,
                    issue.job_id,
                    issue.provider_key,
                    issue.model_key,
                    issue.issue_type.value,
                    issue.message,
                    issue.status.value,
                    issue.retry_count,
                    issue.next_retry_at.isoformat() if issue.next_retry_at else None,
                    issue.created_at.isoformat(),
                    issue.updated_at.isoformat(),
                    issue.resolved_at.isoformat() if issue.resolved_at else None,
                ),
            )
            conn.commit()
        return issue

    def get_runtime_issue(self, issue_id: str) -> RuntimeIssue:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runtime_issues WHERE id = ?",
                (issue_id,),
            ).fetchone()
        if row is None:
            raise KeyError(issue_id)
        return self._row_to_runtime_issue(row)

    def list_runtime_issues(
        self, *, job_id: str | None = None, status: str | None = None
    ) -> list[RuntimeIssue]:
        query = "SELECT * FROM runtime_issues"
        clauses: list[str] = []
        params: list[str] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_runtime_issue(row) for row in rows]

    def save_worker_heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeat:
        heartbeat.heartbeat_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO worker_heartbeats(worker_id, status, heartbeat_at, details_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    status=excluded.status,
                    heartbeat_at=excluded.heartbeat_at,
                    details_json=excluded.details_json
                """,
                (
                    heartbeat.worker_id,
                    heartbeat.status,
                    heartbeat.heartbeat_at.isoformat(),
                    _json_dumps(heartbeat.details),
                ),
            )
            conn.commit()
        return heartbeat

    def list_worker_heartbeats(self) -> list[WorkerHeartbeat]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT worker_id, status, heartbeat_at, details_json FROM worker_heartbeats ORDER BY worker_id"
            ).fetchall()
        return [
            WorkerHeartbeat(
                worker_id=row["worker_id"],
                status=row["status"],
                heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]),
                details=_json_loads(row["details_json"], {}),
            )
            for row in rows
        ]

    def save_job_lease(self, lease: JobLease) -> JobLease:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_leases(job_id, worker_id, acquired_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    acquired_at=excluded.acquired_at,
                    expires_at=excluded.expires_at
                """,
                (
                    lease.job_id,
                    lease.worker_id,
                    lease.acquired_at.isoformat(),
                    lease.expires_at.isoformat(),
                ),
            )
            conn.commit()
        return lease

    def get_job_lease(self, job_id: str) -> JobLease | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_leases WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return JobLease(
            job_id=row["job_id"],
            worker_id=row["worker_id"],
            acquired_at=datetime.fromisoformat(row["acquired_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
        )

    def release_job_lease(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM job_leases WHERE job_id = ?", (job_id,))
            conn.commit()

    def list_job_leases(self) -> list[JobLease]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM job_leases ORDER BY job_id").fetchall()
        return [
            JobLease(
                job_id=row["job_id"],
                worker_id=row["worker_id"],
                acquired_at=datetime.fromisoformat(row["acquired_at"]),
                expires_at=datetime.fromisoformat(row["expires_at"]),
            )
            for row in rows
        ]

    def record_notification(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notifications(job_id, channel, kind, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    payload.get("job_id"),
                    str(payload.get("channel", "console")),
                    str(payload.get("kind", "status")),
                    _json_dumps(payload),
                    utc_now().isoformat(),
                ),
            )
            conn.commit()

    def list_notifications(self, *, job_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload_json FROM notifications"
        params: list[str] = []
        if job_id is not None:
            query += " WHERE job_id = ?"
            params.append(job_id)
        query += " ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_json_loads(row["payload_json"], {}) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        spec = JobSpec.model_validate(_json_loads(row["spec_json"], {}))
        audit_events = self._list_audit_events(row["id"])
        return JobRecord(
            job_id=row["id"],
            title=row["title"],
            spec=spec,
            status=JobStatus(row["status"]),
            history=[JobStatus(item) for item in _json_loads(row["history_json"], [])],
            outputs=_json_loads(row["outputs_json"], {}),
            audit_events=audit_events,
            failure_count=row["failure_count"],
            same_test_failure_count=row["same_test_failure_count"],
            last_error=row["last_error"],
            runtime_error=row["runtime_error"],
            provider_status=row["provider_status"],
            current_phase=row["current_phase"],
            current_task_id=row["current_task_id"],
            pending_approval_id=row["pending_approval_id"],
            pending_runtime_issue_id=row["pending_runtime_issue_id"],
            runtime_state=_json_loads(row["runtime_state_json"], {}),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]) if row["heartbeat_at"] else None,
            lease_owner=row["lease_owner"],
            lease_expires_at=datetime.fromisoformat(row["lease_expires_at"]) if row["lease_expires_at"] else None,
        )

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["id"],
            job_id=row["job_id"],
            status=TaskStatus(row["status"]),
            title=row["title"],
            description=row["description"],
            role=row["role"],
            complexity=TaskComplexity(row["complexity"]),
            dependencies=_json_loads(row["dependencies_json"], []),
            target_files=_json_loads(row["target_files_json"], []),
            attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            pending_approval_id=row["pending_approval_id"],
            pending_runtime_issue_id=row["pending_runtime_issue_id"],
            last_error=row["last_error"],
            checkpoint_key=row["checkpoint_key"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_checkpoint(self, row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row["id"],
            job_id=row["job_id"],
            task_id=row["task_id"],
            checkpoint_key=row["checkpoint_key"],
            step_name=row["step_name"],
            idempotency_key=row["idempotency_key"],
            status=row["status"],
            result_json=_json_loads(row["result_json"], {}),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _row_to_runtime_issue(self, row: sqlite3.Row) -> RuntimeIssue:
        return RuntimeIssue.model_validate(
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "provider_key": row["provider_key"],
                "model_key": row["model_key"],
                "issue_type": row["issue_type"],
                "message": row["message"],
                "status": row["status"],
                "retry_count": row["retry_count"],
                "next_retry_at": row["next_retry_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "resolved_at": row["resolved_at"],
            }
        )

    def _insert_task(self, conn: sqlite3.Connection, task: TaskRecord) -> None:
        conn.execute(
            """
            INSERT INTO tasks(
                id, job_id, status, title, description, role, complexity,
                dependencies_json, target_files_json, attempt_count, max_attempts,
                pending_approval_id, pending_runtime_issue_id, last_error,
                checkpoint_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                job_id=excluded.job_id,
                status=excluded.status,
                title=excluded.title,
                description=excluded.description,
                role=excluded.role,
                complexity=excluded.complexity,
                dependencies_json=excluded.dependencies_json,
                target_files_json=excluded.target_files_json,
                attempt_count=excluded.attempt_count,
                max_attempts=excluded.max_attempts,
                pending_approval_id=excluded.pending_approval_id,
                pending_runtime_issue_id=excluded.pending_runtime_issue_id,
                last_error=excluded.last_error,
                checkpoint_key=excluded.checkpoint_key,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at
            """,
            (
                task.task_id,
                task.job_id,
                task.status.value,
                task.title,
                task.description,
                task.role,
                task.complexity.value,
                _json_dumps(task.dependencies),
                _json_dumps(task.target_files),
                task.attempt_count,
                task.max_attempts,
                task.pending_approval_id,
                task.pending_runtime_issue_id,
                task.last_error,
                task.checkpoint_key,
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
            ),
        )

    def _sync_tasks_from_record(self, conn: sqlite3.Connection, record: JobRecord) -> None:
        task_graph = record.outputs.get("task_graph") or record.outputs.get("planner")
        if not isinstance(task_graph, dict):
            return
        tasks = task_graph.get("tasks")
        if not isinstance(tasks, list):
            return
        conn.execute("DELETE FROM tasks WHERE job_id = ?", (record.job_id,))
        for item in tasks:
            self._insert_task(
                conn,
                TaskRecord.from_planned_task(
                    job_id=record.job_id,
                    task=PlannedTask.model_validate(item),
                ),
            )

    def _sync_audit_events(self, conn: sqlite3.Connection, record: JobRecord) -> None:
        conn.execute("DELETE FROM audit_events WHERE job_id = ?", (record.job_id,))
        conn.execute("DELETE FROM model_call_records WHERE job_id = ?", (record.job_id,))
        conn.execute("DELETE FROM tool_call_records WHERE job_id = ?", (record.job_id,))
        for event in record.audit_events:
            conn.execute(
                """
                INSERT INTO audit_events(
                    job_id, task_id, timestamp, event_type, role, action, status,
                    step, provider_key, model_key, tool_name, input_hash, output_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    event.task_id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.role,
                    event.action,
                    event.status,
                    event.step,
                    event.provider_key,
                    event.model_key,
                    event.tool_name,
                    event.input_hash,
                    event.output_hash,
                    _json_dumps(event.metadata),
                ),
            )
            if event.event_type == "model_call":
                conn.execute(
                    """
                    INSERT INTO model_call_records(job_id, task_id, role, model_key, provider_key, status, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.job_id,
                        event.task_id,
                        event.role,
                        event.model_key or event.metadata.get("model_key"),
                        event.provider_key or event.metadata.get("provider_key"),
                        event.status,
                        _json_dumps(event.metadata),
                    ),
                )
            if event.event_type == "tool_call":
                conn.execute(
                    """
                    INSERT INTO tool_call_records(job_id, task_id, role, tool_name, status, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.job_id,
                        event.task_id,
                        event.role,
                        event.tool_name or event.metadata.get("tool_name"),
                        event.status,
                        _json_dumps(event.metadata),
                    ),
                )

    def _list_audit_events(self, job_id: str) -> list[AuditEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE job_id = ? ORDER BY id ASC",
                (job_id,),
            ).fetchall()
        return [
            AuditEvent(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type=row["event_type"],
                role=row["role"],
                action=row["action"],
                status=row["status"],
                job_id=job_id,
                task_id=row["task_id"],
                step=row["step"],
                provider_key=row["provider_key"],
                model_key=row["model_key"],
                tool_name=row["tool_name"],
                input_hash=row["input_hash"],
                output_hash=row["output_hash"],
                metadata=_json_loads(row["metadata_json"], {}),
            )
            for row in rows
        ]
