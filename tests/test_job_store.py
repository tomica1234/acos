from pathlib import Path

from packages.orchestrator.job_store import FileJobStore
from packages.schemas.jobs import JobSpec
from packages.schemas.models import JobStatus


def test_file_job_store_persists_and_reloads_records(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-1",
        request_text="Build a durable feature.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)

    record = store.create(spec)
    record.status = JobStatus.TESTING
    record.completed_task_ids.append("core")
    record.checkpoints.append({"kind": "autonomous_stage", "stage": 1})
    store.update(record)

    reloaded = FileJobStore(jobs_dir).get("job-1")

    assert reloaded.status == JobStatus.TESTING
    assert reloaded.completed_task_ids == ["core"]
    assert reloaded.checkpoints == [{"kind": "autonomous_stage", "stage": 1}]


def test_file_job_store_writes_records_atomically(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-atomic",
        request_text="Build without losing progress.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)

    record = store.create(spec)
    record.status = JobStatus.TESTING
    store.update(record)

    assert (jobs_dir / "job-atomic.json").exists()
    assert not (jobs_dir / ".job-atomic.json.tmp").exists()
    assert FileJobStore(jobs_dir).get("job-atomic").status == JobStatus.TESTING


def test_file_job_store_falls_back_when_windows_replace_is_denied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-win-replace",
        request_text="Keep saving on Windows.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)

    original_replace = Path.replace
    monkeypatch.setattr("packages.orchestrator.job_store.time.sleep", lambda _: None)

    def deny_temp_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(".job-win-replace.") and self.suffix == ".tmp":
            raise PermissionError("[WinError 5] Access is denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", deny_temp_replace)

    record.status = JobStatus.TESTING
    store.update(record)

    assert FileJobStore(jobs_dir).get("job-win-replace").status == JobStatus.TESTING


def test_file_job_store_quarantines_invalid_records_on_load(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    (jobs_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    spec = JobSpec(
        job_id="job-valid",
        request_text="Keep loading healthy jobs.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    store.create(spec)

    reloaded = FileJobStore(jobs_dir)

    assert reloaded.get("job-valid").spec.request_text == "Keep loading healthy jobs."
    assert not (jobs_dir / "broken.json").exists()
    assert (jobs_dir / "broken.json.invalid").exists()
