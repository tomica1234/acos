"""Lease and heartbeat helpers for worker coordination."""

from __future__ import annotations

from datetime import timedelta

from packages.orchestrator.job_store import JobStore, utc_now
from packages.schemas.runtime import JobLease, WorkerHeartbeat


class LeaseManager:
    """Manage job leases and worker heartbeats."""

    def __init__(self, store: JobStore) -> None:
        self.store = store

    def acquire_job_lease(self, job_id: str, worker_id: str, ttl_seconds: int) -> bool:
        current = self.store.get_job_lease(job_id)
        now = utc_now()
        if current is not None and current.worker_id != worker_id and current.expires_at > now:
            return False
        lease = JobLease(
            job_id=job_id,
            worker_id=worker_id,
            acquired_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        self.store.save_job_lease(lease)
        return True

    def renew_job_lease(self, job_id: str, worker_id: str, ttl_seconds: int) -> bool:
        current = self.store.get_job_lease(job_id)
        if current is None or current.worker_id != worker_id:
            return False
        current.expires_at = utc_now() + timedelta(seconds=ttl_seconds)
        self.store.save_job_lease(current)
        return True

    def release_job_lease(self, job_id: str, worker_id: str) -> bool:
        current = self.store.get_job_lease(job_id)
        if current is None or current.worker_id != worker_id:
            return False
        self.store.release_job_lease(job_id)
        return True

    def find_stale_leases(self) -> list[JobLease]:
        now = utc_now()
        return [lease for lease in self.store.list_job_leases() if lease.expires_at <= now]

    def recover_stale_leases(self) -> list[JobLease]:
        stale = self.find_stale_leases()
        for lease in stale:
            self.store.release_job_lease(lease.job_id)
        return stale

    def record_heartbeat(self, worker_id: str, *, status: str = "alive", details: dict[str, str] | None = None) -> WorkerHeartbeat:
        heartbeat = WorkerHeartbeat(worker_id=worker_id, status=status, details=details or {})
        return self.store.save_worker_heartbeat(heartbeat)
