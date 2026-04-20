"""Job store with optional JSON-backed persistence."""

from __future__ import annotations

import json
from pathlib import Path

from packages.schemas.jobs import JobRecord, JobSpec


class InMemoryJobStore:
    """A tiny job record store that can optionally persist to JSON."""

    def __init__(self, backing_path: str | Path | None = None) -> None:
        self._records: dict[str, JobRecord] = {}
        self.backing_path = Path(backing_path) if backing_path is not None else None
        self._load()

    def _load(self) -> None:
        if self.backing_path is None or not self.backing_path.exists():
            return
        payload = json.loads(self.backing_path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            record = JobRecord.model_validate(item)
            self._records[record.job_id] = record

    def _flush(self) -> None:
        if self.backing_path is None:
            return
        self.backing_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [record.model_dump(mode="json") for record in self._records.values()]
        }
        self.backing_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def create(self, spec: JobSpec) -> JobRecord:
        record = JobRecord(job_id=spec.job_id, spec=spec)
        self._records[record.job_id] = record
        self._flush()
        return record

    def get(self, job_id: str) -> JobRecord:
        return self._records[job_id]

    def update(self, record: JobRecord) -> JobRecord:
        self._records[record.job_id] = record
        self._flush()
        return record
