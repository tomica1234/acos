"""Lease and heartbeat helpers for durable workers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from packages.schemas.jobs import utc_now
from packages.schemas.runtime import JobLease, WorkerHeartbeat


class LeaseManager:
    """Coordinate single-worker ownership of a job."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def acquire_job_lease(self, job_id: str, worker_id: str, ttl_seconds: int) -> bool:
        now = utc_now()
        current = self.store.get_job_lease(job_id)
        if current is not None and current.expires_at > now and current.worker_id != worker_id:
            return False
        self.store.save_job_lease(
            JobLease(
                job_id=job_id,
                worker_id=worker_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
        )
        return True

    def renew_job_lease(self, job_id: str, worker_id: str, ttl_seconds: int) -> bool:
        current = self.store.get_job_lease(job_id)
        if current is None or current.worker_id != worker_id:
            return False
        current.expires_at = utc_now() + timedelta(seconds=ttl_seconds)
        self.store.save_job_lease(current)
        return True

    def release_job_lease(self, job_id: str, worker_id: str | None = None) -> None:
        current = self.store.get_job_lease(job_id)
        if current is None:
            return
        if worker_id is not None and current.worker_id != worker_id:
            return
        self.store.release_job_lease(job_id)

    def recover_stale_leases(self) -> list[JobLease]:
        now = utc_now()
        stale: list[JobLease] = []
        for lease in self.store.list_job_leases():
            if lease.expires_at <= now:
                stale.append(lease)
                self.store.release_job_lease(lease.job_id)
        return stale

    def record_heartbeat(self, worker_id: str, *, details: dict[str, str] | None = None) -> WorkerHeartbeat:
        heartbeat = WorkerHeartbeat(worker_id=worker_id, details=details or {})
        return self.store.save_worker_heartbeat(heartbeat)
