from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.worker import main as worker_main
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus


def test_worker_request_applies_strict_quality_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeStore:
        def __init__(self, sqlite_path: str) -> None:
            captured["sqlite_path"] = sqlite_path

        def create(
            self,
            spec: JobSpec,
            *,
            status: JobStatus = JobStatus.SUBMITTED,
        ) -> JobRecord:
            captured["spec"] = spec
            record = JobRecord(job_id=spec.job_id, spec=spec, status=status)
            captured["record"] = record
            return record

    class FakeDaemon:
        def __init__(self, **kwargs: object) -> None:
            captured["daemon_kwargs"] = kwargs

        def run_once(self, job_id: str) -> JobRecord:
            captured["run_once_job_id"] = job_id
            record = captured["record"]
            assert isinstance(record, JobRecord)
            record.status = JobStatus.DONE
            return record

    def fake_build_default_runner(
        *,
        config_dir: str,
        workspace_root: Path,
        store: FakeStore,
    ) -> tuple[object, object]:
        captured["config_dir"] = config_dir
        captured["workspace_root"] = workspace_root
        captured["runner_store"] = store
        return object(), object()

    monkeypatch.setattr(worker_main, "SQLiteJobStore", FakeStore)
    monkeypatch.setattr(worker_main, "WorkerDaemon", FakeDaemon)
    monkeypatch.setattr(worker_main, "build_default_runner", fake_build_default_runner)

    exit_code = worker_main.main(
        [
            "--repo",
            str(tmp_path),
            "--request",
            "Build a production app.",
            "--branch",
            "acos/worker-request",
            "--sqlite-path",
            str(tmp_path / ".acos" / "acos.sqlite3"),
        ]
    )

    assert exit_code == 0
    spec = captured["spec"]
    assert isinstance(spec, JobSpec)
    assert spec.request_text == "Build a production app."
    assert spec.repo_path == str(tmp_path.resolve())
    assert spec.target_branch == "acos/worker-request"
    assert captured["run_once_job_id"] == spec.job_id
    assert spec.metadata["constraints"] == {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
