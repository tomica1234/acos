"""Worker helpers for durable autonomous job recovery."""

from __future__ import annotations

from dataclasses import dataclass

from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.recovery_governor import (
    RecoveryGovernor,
    is_hard_terminal_status,
    is_recoverable_status,
    is_waiting_status,
)
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


@dataclass
class WorkerDaemon:
    """Resume jobs while treating BLOCKED/STUCK/FAILED as recoverable states."""

    runner: JobRunner
    store: InMemoryJobStore
    recovery_governor: RecoveryGovernor | None = None

    def __post_init__(self) -> None:
        if self.recovery_governor is None:
            self.recovery_governor = self.runner.recovery_governor

    @staticmethod
    def is_hard_terminal_status(status: JobStatus) -> bool:
        return is_hard_terminal_status(status)

    @staticmethod
    def is_settled_status(status: JobStatus) -> bool:
        return is_hard_terminal_status(status) or is_waiting_status(status)

    def should_process(self, record: JobRecord) -> bool:
        return not self.is_settled_status(record.status)

    def normalize_before_processing(self, record: JobRecord) -> JobRecord:
        if is_recoverable_status(record.status):
            record.status = JobStatus.RECOVERING
            record.history.append(JobStatus.RECOVERING)
            self.store.update(record)
            assert self.recovery_governor is not None
            self.recovery_governor.recover(record)
            self.store.update(record)
        return record

    def run_once(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if not self.should_process(record):
            return record
        self.normalize_before_processing(record)
        if self.is_settled_status(record.status):
            return record
        return self.runner.resume_job(job_id)
