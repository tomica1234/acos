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
