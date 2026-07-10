"""Shared ACOS job status classification helpers."""

from __future__ import annotations

from packages.schemas.models import JobStatus


HARD_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.DONE,
        JobStatus.CANCELLED,
        JobStatus.POLICY_HARD_STOP,
    }
)

WAITING_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.PAUSED,
    }
)

RECOVERABLE_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.BLOCKED,
        JobStatus.STUCK,
        JobStatus.FAILED,
    }
)

RUNNABLE_STATUSES: frozenset[JobStatus] = frozenset(
    {
        JobStatus.SUBMITTED,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.ANALYZING,
        JobStatus.DESIGNING,
        JobStatus.PLANNING,
        JobStatus.REPLANNING,
        JobStatus.IMPLEMENTING,
        JobStatus.WRITING_TESTS,
        JobStatus.REVIEWING,
        JobStatus.DIAGNOSING,
        JobStatus.STRATEGY_CHANGE,
        JobStatus.RECOVERING,
        JobStatus.TESTING,
        JobStatus.FIXING,
        JobStatus.FINALIZING,
        JobStatus.RESUMING,
        JobStatus.RETRYING_PROVIDER,
    }
)


def is_hard_terminal_status(status: JobStatus) -> bool:
    return status in HARD_TERMINAL_STATUSES


def is_waiting_status(status: JobStatus) -> bool:
    return status in WAITING_STATUSES


def is_recoverable_status(status: JobStatus) -> bool:
    return status in RECOVERABLE_STATUSES


def is_runnable_status(status: JobStatus) -> bool:
    return status in RUNNABLE_STATUSES or is_recoverable_status(status)


def is_settled_status(status: JobStatus) -> bool:
    return is_hard_terminal_status(status) or is_waiting_status(status)
