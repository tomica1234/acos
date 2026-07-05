"""Job record stores."""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

from packages.schemas.jobs import JobRecord, JobSpec, utc_now


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
        record.updated_at = utc_now()
        self._records[record.job_id] = record
        return record


class FileJobStore(InMemoryJobStore):
    """Persist job records as one JSON file per job."""

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._load_existing_records()

    def create(self, spec: JobSpec) -> JobRecord:
        existing = self._records.get(spec.job_id)
        if existing is not None:
            return existing
        return self.update(JobRecord(job_id=spec.job_id, spec=spec))

    def update(self, record: JobRecord) -> JobRecord:
        super().update(record)
        path = self._path_for(record.job_id)
        temp_path = self._temp_path_for(record.job_id)
        self._write_atomic(temp_path, path, record.model_dump_json(indent=2))
        return record

    def _load_existing_records(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                self._quarantine_invalid_record(path)
                continue
            self._records[record.job_id] = record

    def _path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _temp_path_for(self, job_id: str) -> Path:
        return self.root / f".{job_id}.{os.getpid()}.{uuid4().hex}.json.tmp"

    def _write_atomic(self, temp_path: Path, path: Path, payload: str) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: PermissionError | None = None
        for attempt in range(8):
            try:
                temp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))

        # Windows can deny replace() while another process briefly has the
        # destination open. Preserve progress by overwriting in place instead
        # of failing the whole autonomous run.
        try:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            return
        except PermissionError:
            if last_error is not None:
                raise last_error
            raise

    def _quarantine_invalid_record(self, path: Path) -> None:
        quarantine_path = path.with_suffix(path.suffix + ".invalid")
        counter = 1
        while quarantine_path.exists():
            quarantine_path = path.with_suffix(path.suffix + f".invalid.{counter}")
            counter += 1
        path.replace(quarantine_path)
