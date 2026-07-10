"""Job record stores."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobRecord, JobSpec, utc_now, validate_job_id_string
from packages.schemas.models import JobStatus
from packages.schemas.runtime import JobLease, RuntimeIssue, WorkerHeartbeat
from packages.schemas.tasks import PlannedTask, TaskRecord


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _json_loads(payload: str | None, default: Any) -> Any:
    if not payload:
        return default
    return json.loads(payload)


def _derive_title(spec: JobSpec) -> str:
    if spec.title:
        return spec.title
    title = spec.metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    first_line = spec.request_text.strip().splitlines()[0] if spec.request_text.strip() else "ACOS job"
    return first_line[:120]


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

    def _persist(self) -> None:
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
        temp_path = self.backing_path.with_name(
            f".{self.backing_path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        payload_text = json.dumps(payload, sort_keys=True, indent=2)
        FileJobStore._write_atomic(self, temp_path, self.backing_path, payload_text)

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
        self._persist()
        return record

    def get(self, job_id: str) -> JobRecord:
        return self._records[job_id]

    def update(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        self._records[record.job_id] = record
        self._sync_tasks_from_record(record)
        self._persist()
        return record

    def list_jobs(self, *, statuses: Sequence[JobStatus] | None = None) -> list[JobRecord]:
        records = list(self._records.values())
        if statuses is None:
            return records
        allowed = set(statuses)
        return [record for record in records if record.status in allowed]

    def save_tasks(self, job_id: str, tasks: Sequence[TaskRecord]) -> None:
        self._tasks[job_id] = {task.task_id: task for task in tasks}
        self._persist()

    def upsert_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        self._tasks.setdefault(task.job_id, {})[task.task_id] = task
        self._persist()
        return task

    def list_tasks(self, job_id: str) -> list[TaskRecord]:
        return list(self._tasks.get(job_id, {}).values())

    def get_task(self, job_id: str, task_id: str) -> TaskRecord:
        return self._tasks[job_id][task_id]

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        checkpoint.updated_at = utc_now()
        items = [
            item
            for item in self._checkpoints.get(checkpoint.job_id, [])
            if item.id != checkpoint.id
        ]
        items.append(checkpoint)
        self._checkpoints[checkpoint.job_id] = items
        self._persist()
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
        return matches[-1] if matches else None

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
        self._persist()
        return issue

    def get_runtime_issue(self, issue_id: str) -> RuntimeIssue:
        return self._runtime_issues[issue_id]

    def list_runtime_issues(
        self, *, job_id: str | None = None, status: str | None = None
    ) -> list[RuntimeIssue]:
        issues = list(self._runtime_issues.values())
        if job_id is not None:
            issues = [issue for issue in issues if issue.job_id == job_id]
        if status is not None:
            issues = [issue for issue in issues if issue.status.value == status]
        return issues

    def save_worker_heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeat:
        heartbeat.heartbeat_at = utc_now()
        self._heartbeats[heartbeat.worker_id] = heartbeat
        self._persist()
        return heartbeat

    def list_worker_heartbeats(self) -> list[WorkerHeartbeat]:
        return list(self._heartbeats.values())

    def save_job_lease(self, lease: JobLease) -> JobLease:
        self._leases[lease.job_id] = lease
        self._persist()
        return lease

    def get_job_lease(self, job_id: str) -> JobLease | None:
        return self._leases.get(job_id)

    def release_job_lease(self, job_id: str) -> None:
        self._leases.pop(job_id, None)
        self._persist()

    def list_job_leases(self) -> list[JobLease]:
        return list(self._leases.values())

    def record_notification(self, payload: dict[str, Any]) -> None:
        self._notifications.append(dict(payload))
        self._persist()

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


class FileJobStore(InMemoryJobStore):
    """Persist job records as one JSON file per job."""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._load_existing_records()

    def create(self, spec: JobSpec, *, status: JobStatus | None = None) -> JobRecord:
        existing = self._records.get(spec.job_id)
        if existing is not None:
            return existing
        initial_status = status or JobStatus.SUBMITTED
        return self.update(
            JobRecord(
                job_id=spec.job_id,
                title=_derive_title(spec),
                spec=spec,
                status=initial_status,
                history=[initial_status],
            )
        )

    def update(self, record: JobRecord) -> JobRecord:
        super().update(record)
        path = self._path_for(record.job_id)
        temp_path = self._temp_path_for(record.job_id)
        self._write_atomic(temp_path, path, record.model_dump_json(indent=2))
        return record

    def _load_existing_records(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                self._quarantine_invalid_record(path)
                continue
            self._records[record.job_id] = record

    def _path_for(self, job_id: str) -> Path:
        validate_job_id_string(job_id)
        return self.root / f"{job_id}.json"

    def _temp_path_for(self, job_id: str) -> Path:
        validate_job_id_string(job_id)
        return self.root / f".{job_id}.{os.getpid()}.{uuid4().hex}.json.tmp"

    def _write_atomic(self, temp_path: Path, path: Path, payload: str) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: PermissionError | None = None
        for attempt in range(8):
            try:
                temp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))

        # Windows can deny replace() while another process briefly has the
        # destination open. Preserve progress by overwriting in place instead
        # of failing the whole autonomous run.
        try:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return
        except PermissionError:
            if last_error is not None:
                raise last_error
            raise

    def _quarantine_invalid_record(self, path: Path) -> None:
        quarantine_path = path.with_suffix(path.suffix + ".invalid")
        counter = 1
        while quarantine_path.exists():
            quarantine_path = path.with_suffix(path.suffix + f".invalid.{counter}")
            counter += 1
        path.replace(quarantine_path)


class SQLiteJobStore(InMemoryJobStore):
    """SQLite-backed durable store for jobs, leases, heartbeats, and recovery state."""

    def __init__(self, db_path: str | Path) -> None:
        InMemoryJobStore.__init__(self)
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    job_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, task_id)
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    checkpoint_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runtime_issues (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS worker_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_leases (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create(self, spec: JobSpec, *, status: JobStatus | None = None) -> JobRecord:
        try:
            return self.get(spec.job_id)
        except KeyError:
            pass
        initial_status = status or JobStatus.SUBMITTED
        return self.update(
            JobRecord(
                job_id=spec.job_id,
                title=_derive_title(spec),
                spec=spec,
                status=initial_status,
                history=[initial_status],
            )
        )

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobRecord.model_validate_json(row["payload_json"])

    def update(self, record: JobRecord) -> JobRecord:
        record.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(job_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.job_id,
                    record.status.value,
                    record.model_dump_json(),
                    record.updated_at.isoformat(),
                ),
            )
            conn.commit()
        self._sync_tasks_from_record(record)
        return record

    def list_jobs(self, *, statuses: Sequence[JobStatus] | None = None) -> list[JobRecord]:
        query = "SELECT payload_json FROM jobs"
        params: list[str] = []
        if statuses:
            query += " WHERE status IN (" + ",".join("?" for _ in statuses) + ")"
            params = [status.value for status in statuses]
        query += " ORDER BY updated_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [JobRecord.model_validate_json(row["payload_json"]) for row in rows]

    def save_tasks(self, job_id: str, tasks: Sequence[TaskRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tasks WHERE job_id = ?", (job_id,))
            for task in tasks:
                self._save_task(conn, task)
            conn.commit()

    def upsert_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        with self._connect() as conn:
            self._save_task(conn, task)
            conn.commit()
        return task

    def list_tasks(self, job_id: str) -> list[TaskRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM tasks WHERE job_id = ? ORDER BY updated_at ASC",
                (job_id,),
            ).fetchall()
        return [TaskRecord.model_validate_json(row["payload_json"]) for row in rows]

    def get_task(self, job_id: str, task_id: str) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM tasks WHERE job_id = ? AND task_id = ?",
                (job_id, task_id),
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return TaskRecord.model_validate_json(row["payload_json"])

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        checkpoint.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(id, job_id, task_id, checkpoint_key, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    task_id=excluded.task_id,
                    checkpoint_key=excluded.checkpoint_key,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    checkpoint.id,
                    checkpoint.job_id,
                    checkpoint.task_id,
                    checkpoint.checkpoint_key,
                    checkpoint.model_dump_json(),
                    checkpoint.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return checkpoint

    def get_checkpoint(
        self, *, job_id: str, checkpoint_key: str, task_id: str | None = None
    ) -> CheckpointRecord | None:
        query = "SELECT payload_json FROM checkpoints WHERE job_id = ? AND checkpoint_key = ?"
        params: list[str | None] = [job_id, checkpoint_key]
        if task_id is None:
            query += " AND task_id IS NULL"
        else:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return CheckpointRecord.model_validate_json(row["payload_json"]) if row else None

    def list_checkpoints(
        self, *, job_id: str, task_id: str | None = None
    ) -> list[CheckpointRecord]:
        query = "SELECT payload_json FROM checkpoints WHERE job_id = ?"
        params: list[str | None] = [job_id]
        if task_id is not None:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " ORDER BY updated_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [CheckpointRecord.model_validate_json(row["payload_json"]) for row in rows]

    def save_runtime_issue(self, issue: RuntimeIssue) -> RuntimeIssue:
        issue.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_issues(id, job_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    issue.id,
                    issue.job_id,
                    issue.status.value,
                    issue.model_dump_json(),
                    issue.updated_at.isoformat(),
                ),
            )
            conn.commit()
        return issue

    def get_runtime_issue(self, issue_id: str) -> RuntimeIssue:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM runtime_issues WHERE id = ?",
                (issue_id,),
            ).fetchone()
        if row is None:
            raise KeyError(issue_id)
        return RuntimeIssue.model_validate_json(row["payload_json"])

    def list_runtime_issues(
        self, *, job_id: str | None = None, status: str | None = None
    ) -> list[RuntimeIssue]:
        query = "SELECT payload_json FROM runtime_issues"
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
        query += " ORDER BY updated_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RuntimeIssue.model_validate_json(row["payload_json"]) for row in rows]

    def save_worker_heartbeat(self, heartbeat: WorkerHeartbeat) -> WorkerHeartbeat:
        heartbeat.heartbeat_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO worker_heartbeats(worker_id, payload_json, heartbeat_at)
                VALUES (?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    heartbeat_at=excluded.heartbeat_at
                """,
                (
                    heartbeat.worker_id,
                    heartbeat.model_dump_json(),
                    heartbeat.heartbeat_at.isoformat(),
                ),
            )
            conn.commit()
        return heartbeat

    def list_worker_heartbeats(self) -> list[WorkerHeartbeat]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM worker_heartbeats ORDER BY worker_id ASC"
            ).fetchall()
        return [WorkerHeartbeat.model_validate_json(row["payload_json"]) for row in rows]

    def save_job_lease(self, lease: JobLease) -> JobLease:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_leases(job_id, worker_id, expires_at, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    worker_id=excluded.worker_id,
                    expires_at=excluded.expires_at,
                    payload_json=excluded.payload_json
                """,
                (
                    lease.job_id,
                    lease.worker_id,
                    lease.expires_at.isoformat(),
                    lease.model_dump_json(),
                ),
            )
            conn.commit()
        return lease

    def get_job_lease(self, job_id: str) -> JobLease | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM job_leases WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return JobLease.model_validate_json(row["payload_json"]) if row else None

    def release_job_lease(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM job_leases WHERE job_id = ?", (job_id,))
            conn.commit()

    def list_job_leases(self) -> list[JobLease]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM job_leases ORDER BY job_id ASC").fetchall()
        return [JobLease.model_validate_json(row["payload_json"]) for row in rows]

    def record_notification(self, payload: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notifications(job_id, payload_json, created_at) VALUES (?, ?, ?)",
                (
                    payload.get("job_id"),
                    _json_dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
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

    @staticmethod
    def _save_task(conn: sqlite3.Connection, task: TaskRecord) -> None:
        conn.execute(
            """
            INSERT INTO tasks(job_id, task_id, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id, task_id) DO UPDATE SET
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """,
            (
                task.job_id,
                task.task_id,
                task.model_dump_json(),
                task.updated_at.isoformat(),
            ),
        )
