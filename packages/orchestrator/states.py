"""Explicit job status transitions."""

from __future__ import annotations

from datetime import datetime, timezone

from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus

PLANNING_ENTRY_STATUSES = {
    JobStatus.ANALYZING,
    JobStatus.DESIGNING,
    JobStatus.PLANNING,
}

RECOVERY_ENTRY_STATUSES = {
    *PLANNING_ENTRY_STATUSES,
    JobStatus.DIAGNOSING,
    JobStatus.REPLANNING,
    JobStatus.IMPLEMENTING,
    JobStatus.WRITING_TESTS,
    JobStatus.FIXING,
    JobStatus.TESTING,
}

ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.SUBMITTED: {
        JobStatus.ANALYZING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.ANALYZING: {
        JobStatus.DESIGNING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.DESIGNING: {
        JobStatus.PLANNING,
        JobStatus.REPLANNING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.PLANNING: {
        JobStatus.IMPLEMENTING,
        JobStatus.REPLANNING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.IMPLEMENTING: {
        JobStatus.WRITING_TESTS,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.WRITING_TESTS: {
        JobStatus.REVIEWING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.REVIEWING: {
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.BLOCKED,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.TESTING: {
        JobStatus.IMPLEMENTING,
        JobStatus.WRITING_TESTS,
        JobStatus.REVIEWING,
        JobStatus.FINALIZING,
        JobStatus.FIXING,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.DIAGNOSING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.FIXING: {
        JobStatus.REVIEWING,
        JobStatus.TESTING,
        JobStatus.STUCK,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.DIAGNOSING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.FINALIZING: {
        JobStatus.DONE,
        JobStatus.FAILED,
        JobStatus.RECOVERING,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.RECOVERING: {
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.WAITING_RUNTIME,
        JobStatus.WAITING_APPROVAL,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.REPLANNING: {
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.DIAGNOSING: {
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.STRATEGY_CHANGE: {
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.WAITING_APPROVAL: {
        JobStatus.RECOVERING,
        JobStatus.REPLANNING,
        JobStatus.POLICY_HARD_STOP,
        JobStatus.CANCELLED,
    },
    JobStatus.WAITING_RUNTIME: {
        JobStatus.RECOVERING,
        JobStatus.REPLANNING,
        JobStatus.POLICY_HARD_STOP,
        JobStatus.CANCELLED,
    },
    JobStatus.BLOCKED: {
        JobStatus.RECOVERING,
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.STUCK: {
        JobStatus.RECOVERING,
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
    JobStatus.FAILED: {
        JobStatus.RECOVERING,
        *RECOVERY_ENTRY_STATUSES,
        JobStatus.POLICY_HARD_STOP,
    },
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
