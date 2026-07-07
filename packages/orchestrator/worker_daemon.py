"""Durable worker helpers for autonomous job recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.leases import LeaseManager
from packages.orchestrator.recovery_governor import RecoveryGovernor
from packages.orchestrator.runtime import RuntimeManager
from packages.orchestrator.statuses import (
    RUNNABLE_STATUSES,
    is_hard_terminal_status,
    is_recoverable_status,
    is_settled_status,
    is_waiting_status,
)
from packages.schemas.jobs import JobRecord, utc_now
from packages.schemas.models import JobStatus


@dataclass
class WorkerConfig:
    id: str = "local-worker"
    poll_interval_seconds: float = 5.0
    heartbeat_interval_seconds: float = 10.0
    lease_ttl_seconds: int = 120
    max_concurrent_jobs: int = 1
    recover_stale_jobs_after_seconds: int = 180


@dataclass
class WorkerDaemon:
    """Poll jobs while treating BLOCKED/STUCK/FAILED as recoverable states."""

    runner: JobRunner
    store: InMemoryJobStore
    recovery_governor: RecoveryGovernor | None = None
    runtime_manager: RuntimeManager | None = None
    config: WorkerConfig = field(default_factory=WorkerConfig)

    def __post_init__(self) -> None:
        if self.recovery_governor is None:
            self.recovery_governor = self.runner.recovery_governor
        self.leases = LeaseManager(self.store)
        self._shutdown_requested = False
        self._last_heartbeat_at = 0.0

    @staticmethod
    def is_hard_terminal_status(status: JobStatus) -> bool:
        return is_hard_terminal_status(status)

    @staticmethod
    def is_settled_status(status: JobStatus) -> bool:
        return is_settled_status(status)

    def should_process(self, record: JobRecord) -> bool:
        return not self.is_settled_status(record.status)

    def normalize_before_processing(self, record: JobRecord) -> JobRecord:
        if is_recoverable_status(record.status):
            record.status = JobStatus.RECOVERING
            record.history.append(JobStatus.RECOVERING)
            self.store.update(record)
            self.runner._recover_record(record, error=record.last_error)
            self.store.update(record)
        return record

    def run_once(self, job_id: str | None = None) -> JobRecord | list[JobRecord]:
        self._heartbeat_if_due(force=True)
        self.recover_stale_jobs()
        if self.runtime_manager is not None:
            self.runtime_manager.maybe_resume_waiting_jobs()
        if job_id is not None:
            return self._run_one_job(job_id)
        processed: list[JobRecord] = []
        runnable_statuses = [*RUNNABLE_STATUSES, JobStatus.BLOCKED, JobStatus.STUCK, JobStatus.FAILED]
        for record in self.store.list_jobs(statuses=runnable_statuses)[: self.config.max_concurrent_jobs]:
            processed.append(self._run_one_job(record.job_id))
            self._heartbeat_if_due()
        return processed

    def run_forever(self) -> None:
        while not self._shutdown_requested:
            self.run_once()
            time.sleep(self.config.poll_interval_seconds)

    def run_until_job_settled(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float | None = None,
        max_iterations: int | None = None,
    ) -> JobRecord:
        iterations = 0
        sleep_seconds = (
            float(poll_interval_seconds)
            if poll_interval_seconds is not None
            else float(self.config.poll_interval_seconds)
        )
        while True:
            record = self.store.get(job_id)
            if is_settled_status(record.status):
                return record
            result = self.run_once(job_id)
            iterations += 1
            latest = result if isinstance(result, JobRecord) else self.store.get(job_id)
            if is_settled_status(latest.status):
                return latest
            if max_iterations is not None and iterations >= max_iterations:
                return latest
            time.sleep(max(0.0, sleep_seconds))

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def recover_stale_jobs(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        self.leases.recover_stale_leases()
        now = utc_now()
        statuses = [status for status in RUNNABLE_STATUSES if not is_waiting_status(status)]
        for record in self.store.list_jobs(statuses=statuses):
            if record.heartbeat_at is None:
                continue
            if (now - record.heartbeat_at).total_seconds() <= self.config.recover_stale_jobs_after_seconds:
                continue
            record.status = JobStatus.RECOVERING
            record.last_error = "stale_heartbeat_detected"
            self.store.update(record)
            recovered.append(record)
        return recovered

    def _run_one_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if not self.should_process(record):
            return record
        if not self.leases.acquire_job_lease(job_id, self.config.id, self.config.lease_ttl_seconds):
            return record
        try:
            record = self.normalize_before_processing(self.store.get(job_id))
            if self.is_settled_status(record.status):
                return record
            record.lease_owner = self.config.id
            record.heartbeat_at = utc_now()
            if record.status in {JobStatus.QUEUED, JobStatus.SUBMITTED, JobStatus.RESUMING}:
                record.status = JobStatus.RUNNING
                record.history.append(JobStatus.RUNNING)
            self.store.update(record)
            return self.runner.resume_job(job_id)
        finally:
            latest = self.store.get(job_id)
            if is_settled_status(latest.status) or is_recoverable_status(latest.status):
                self.leases.release_job_lease(job_id, self.config.id)
            else:
                self.leases.renew_job_lease(job_id, self.config.id, self.config.lease_ttl_seconds)

    def _heartbeat_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat_at < self.config.heartbeat_interval_seconds:
            return
        self.leases.record_heartbeat(self.config.id, details={"mode": "daemon"})
        self._last_heartbeat_at = now
