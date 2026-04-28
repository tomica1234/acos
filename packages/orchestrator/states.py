"""Explicit job status transitions."""

from __future__ import annotations

from datetime import datetime, timezone

from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


TERMINAL_STATUSES = {
    JobStatus.DONE,
    JobStatus.BLOCKED,
    JobStatus.STUCK,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {
        JobStatus.SUBMITTED,
        JobStatus.RUNNING,
        JobStatus.RECOVERING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
    },
    JobStatus.SUBMITTED: {
        JobStatus.RUNNING,
        JobStatus.ANALYZING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.RECOVERING,
        JobStatus.FAILED,
    },
    JobStatus.RUNNING: {
        JobStatus.ANALYZING,
        JobStatus.DESIGNING,
        JobStatus.PLANNING,
        JobStatus.IMPLEMENTING,
        JobStatus.WRITING_TESTS,
        JobStatus.REVIEWING,
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.FINALIZING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.PAUSED,
        JobStatus.CRASHED,
        JobStatus.RECOVERING,
        JobStatus.BLOCKED,
        JobStatus.STUCK,
        JobStatus.FAILED,
        JobStatus.DONE,
    },
    JobStatus.RESUMING: {
        JobStatus.RUNNING,
        JobStatus.ANALYZING,
        JobStatus.DESIGNING,
        JobStatus.PLANNING,
        JobStatus.IMPLEMENTING,
        JobStatus.WRITING_TESTS,
        JobStatus.REVIEWING,
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.FINALIZING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.DONE,
    },
    JobStatus.RECOVERING: {
        JobStatus.RESUMING,
        JobStatus.RUNNING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.RETRYING_PROVIDER: {
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.RESUMING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.WAITING_RUNTIME: {
        JobStatus.RESUMING,
        JobStatus.RETRYING_PROVIDER,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.PROVIDER_UNAVAILABLE: {
        JobStatus.WAITING_RUNTIME,
        JobStatus.RETRYING_PROVIDER,
        JobStatus.RESUMING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.WAITING_APPROVAL: {
        JobStatus.RESUMING,
        JobStatus.ANALYZING,
        JobStatus.DESIGNING,
        JobStatus.PLANNING,
        JobStatus.IMPLEMENTING,
        JobStatus.WRITING_TESTS,
        JobStatus.REVIEWING,
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.FINALIZING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.PAUSED: {
        JobStatus.RESUMING,
        JobStatus.CANCELLED,
    },
    JobStatus.ANALYZING: {
        JobStatus.DESIGNING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.DESIGNING: {
        JobStatus.PLANNING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.PLANNING: {
        JobStatus.REVIEWING,
        JobStatus.IMPLEMENTING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.IMPLEMENTING: {
        JobStatus.WRITING_TESTS,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.WRITING_TESTS: {
        JobStatus.REVIEWING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.REVIEWING: {
        JobStatus.DESIGNING,
        JobStatus.PLANNING,
        JobStatus.IMPLEMENTING,
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.FINALIZING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
    },
    JobStatus.TESTING: {
        JobStatus.RUNNING,
        JobStatus.REVIEWING,
        JobStatus.FINALIZING,
        JobStatus.FIXING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.FAILED,
        JobStatus.STUCK,
    },
    JobStatus.FIXING: {
        JobStatus.REVIEWING,
        JobStatus.TESTING,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.STUCK,
        JobStatus.FAILED,
    },
    JobStatus.FINALIZING: {
        JobStatus.DONE,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.FAILED,
    },
}


def apply_transition(record: JobRecord, new_status: JobStatus) -> None:
    """Update job status while enforcing allowed transitions."""
    if record.status == new_status:
        return
    if record.status in TERMINAL_STATUSES and new_status in TERMINAL_STATUSES:
        record.status = new_status
        record.history.append(new_status)
        record.updated_at = datetime.now(timezone.utc)
        return
    allowed = ALLOWED_TRANSITIONS.get(record.status, set())
    if new_status not in allowed:
        raise ValueError(f"Invalid state transition: {record.status} -> {new_status}")
    record.status = new_status
    record.history.append(new_status)
    record.updated_at = datetime.now(timezone.utc)
