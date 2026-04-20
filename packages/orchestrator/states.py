"""Explicit job status transitions."""

from __future__ import annotations

from datetime import datetime, timezone

from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus

ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.SUBMITTED: {JobStatus.ANALYZING, JobStatus.WAITING_APPROVAL, JobStatus.FAILED},
    JobStatus.WAITING_APPROVAL: {
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
    JobStatus.ANALYZING: {JobStatus.DESIGNING, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.DESIGNING: {JobStatus.PLANNING, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.PLANNING: {JobStatus.IMPLEMENTING, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.IMPLEMENTING: {JobStatus.WRITING_TESTS, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.WRITING_TESTS: {JobStatus.REVIEWING, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.REVIEWING: {JobStatus.TESTING, JobStatus.FIXING, JobStatus.WAITING_APPROVAL, JobStatus.BLOCKED, JobStatus.FAILED},
    JobStatus.TESTING: {JobStatus.FINALIZING, JobStatus.FIXING, JobStatus.WAITING_APPROVAL, JobStatus.FAILED},
    JobStatus.FIXING: {JobStatus.REVIEWING, JobStatus.TESTING, JobStatus.WAITING_APPROVAL, JobStatus.STUCK, JobStatus.FAILED},
    JobStatus.FINALIZING: {JobStatus.DONE, JobStatus.WAITING_APPROVAL, JobStatus.FAILED},
}


def apply_transition(record: JobRecord, new_status: JobStatus) -> None:
    """Update job status while enforcing allowed transitions."""
    if record.status == new_status:
        return
    allowed = ALLOWED_TRANSITIONS.get(record.status, set())
    if new_status not in allowed:
        raise ValueError(f"Invalid state transition: {record.status} -> {new_status}")
    record.status = new_status
    record.history.append(new_status)
    record.updated_at = datetime.now(timezone.utc)
