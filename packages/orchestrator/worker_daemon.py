"""Durable worker daemon."""

from __future__ import annotations

import signal
import time
from datetime import timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import JobStore, utc_now
from packages.orchestrator.leases import LeaseManager
from packages.orchestrator.runtime import RuntimeManager
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "local-worker"
    poll_interval_seconds: int = 5
    heartbeat_interval_seconds: int = 10
    lease_ttl_seconds: int = 120
    max_concurrent_jobs: int = 1
    max_concurrent_tasks_per_job: int = 1
    recover_stale_jobs_after_seconds: int = 180
    graceful_shutdown_timeout_seconds: int = 30
    default_job_timeout_minutes: int = 720
    log_dir: str = ".acos/logs"
    sqlite_path: str = ".acos/acos.sqlite3"
    continue_independent_tasks_while_waiting_approval: bool = True


class WorkerDaemon:
    """Poll queued jobs, renew heartbeats, and resume durable execution."""

    def __init__(
        self,
        *,
        runner: JobRunner,
        store: JobStore,
        runtime_manager: RuntimeManager,
        config: WorkerConfig,
    ) -> None:
        self.runner = runner
        self.store = store
        self.runtime_manager = runtime_manager
        self.config = config
        self.leases = LeaseManager(store)
        self._shutdown_requested = False
        self._last_heartbeat_at = 0.0

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        runner: JobRunner,
        store: JobStore,
        runtime_manager: RuntimeManager,
    ) -> "WorkerDaemon":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(
            runner=runner,
            store=store,
            runtime_manager=runtime_manager,
            config=WorkerConfig(**payload["worker"]),
        )

    def run_once(self) -> list[JobRecord]:
        self._heartbeat_if_due(force=True)
        self.recover_stale_jobs()
        self.runtime_manager.maybe_resume_waiting_jobs()
        processed: list[JobRecord] = []
        runnable = self.store.list_jobs(
            statuses=[
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.RECOVERING,
                JobStatus.RESUMING,
                JobStatus.SUBMITTED,
                JobStatus.ANALYZING,
                JobStatus.DESIGNING,
                JobStatus.PLANNING,
                JobStatus.IMPLEMENTING,
                JobStatus.WRITING_TESTS,
                JobStatus.REVIEWING,
                JobStatus.TESTING,
                JobStatus.FIXING,
                JobStatus.FINALIZING,
            ]
        )
        for record in runnable[: self.config.max_concurrent_jobs]:
            if not self.leases.acquire_job_lease(record.job_id, self.config.id, self.config.lease_ttl_seconds):
                continue
            latest = self.store.get(record.job_id)
            latest.lease_owner = self.config.id
            latest.lease_expires_at = utc_now() + timedelta(seconds=self.config.lease_ttl_seconds)
            latest.heartbeat_at = utc_now()
            if latest.status in {JobStatus.QUEUED, JobStatus.SUBMITTED, JobStatus.RESUMING, JobStatus.RECOVERING}:
                latest.status = JobStatus.RUNNING
            self.store.update(latest)
            processed_record = self.runner.run_next_step(latest.job_id)
            processed.append(processed_record)
            if processed_record.status in {
                JobStatus.DONE,
                JobStatus.BLOCKED,
                JobStatus.STUCK,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.WAITING_APPROVAL,
                JobStatus.WAITING_RUNTIME,
                JobStatus.PROVIDER_UNAVAILABLE,
                JobStatus.PAUSED,
            }:
                self.leases.release_job_lease(latest.job_id, self.config.id)
            else:
                self.leases.renew_job_lease(latest.job_id, self.config.id, self.config.lease_ttl_seconds)
            self._heartbeat_if_due()
        return processed

    def run_forever(self) -> None:
        self._install_signal_handlers()
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
            if _is_settled_status(record.status):
                return record
            processed = self.run_once()
            iterations += 1
            latest = self.store.get(job_id)
            if _is_settled_status(latest.status):
                return latest
            if max_iterations is not None and iterations >= max_iterations:
                return latest
            if not any(item.job_id == job_id for item in processed):
                time.sleep(max(0.0, sleep_seconds))

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def recover_stale_jobs(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        self.leases.recover_stale_leases()
        now = utc_now()
        for record in self.store.list_jobs(
            statuses=[JobStatus.RUNNING, JobStatus.ANALYZING, JobStatus.DESIGNING, JobStatus.PLANNING, JobStatus.IMPLEMENTING, JobStatus.WRITING_TESTS, JobStatus.REVIEWING, JobStatus.TESTING, JobStatus.FIXING, JobStatus.FINALIZING]
        ):
            if record.heartbeat_at is None:
                continue
            age = (now - record.heartbeat_at).total_seconds()
            if age <= self.config.recover_stale_jobs_after_seconds:
                continue
            record.status = JobStatus.RECOVERING
            record.last_error = "stale_heartbeat_detected"
            self.store.update(record)
            recovered.append(record)
        return recovered

    def _heartbeat_if_due(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat_at < self.config.heartbeat_interval_seconds:
            return
        self.leases.record_heartbeat(self.config.id, details={"mode": "daemon"})
        self._last_heartbeat_at = now

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, lambda signum, frame: self.request_shutdown())
        signal.signal(signal.SIGINT, lambda signum, frame: self.request_shutdown())


def _is_settled_status(status: JobStatus) -> bool:
    return status in {
        JobStatus.DONE,
        JobStatus.BLOCKED,
        JobStatus.STUCK,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.WAITING_APPROVAL,
        JobStatus.WAITING_RUNTIME,
        JobStatus.PROVIDER_UNAVAILABLE,
        JobStatus.PAUSED,
    }
