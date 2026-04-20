"""In-memory job store."""

from __future__ import annotations

from packages.schemas.jobs import JobRecord, JobSpec


class InMemoryJobStore:
    """A minimal job record store for the MVP."""

    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}

    def create(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(job_id=spec.job_id, spec=spec)
        self._records[record.job_id] = record
        return record

    def get(self, job_id: str) -> JobRecord:
        return self._records[job_id]

    def update(self, record: JobRecord) -> JobRecord:
        self._records[record.job_id] = record
        return record

