from __future__ import annotations

import plistlib
from pathlib import Path
from types import SimpleNamespace

import yaml

from apps import cli
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus


class _StubStore:
    def list_worker_heartbeats(self):
        return [SimpleNamespace(model_dump=lambda mode="json": {"worker_id": "worker-1", "status": "alive"})]

    def list_runtime_issues(self):
        return []


class _StubRunner:
    def __init__(self, spec: JobSpec | None = None) -> None:
        self.store = _StubStore()
        self.runtime_manager = SimpleNamespace(maybe_resume_waiting_jobs=lambda: [])
        self._record = JobRecord(
            job_id=(spec.job_id if spec else "job-1"),
            spec=spec or JobSpec(request_text="x", repo_path="."),
            status=JobStatus.QUEUED,
        )

    def submit(self, spec: JobSpec) -> JobRecord:
        self._record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.QUEUED)
        return self._record

    def get(self, job_id: str) -> JobRecord:
        return self._record

    def list_jobs(self, statuses=None):
        return [self._record]

    def pause_job(self, job_id: str) -> JobRecord:
        self._record.status = JobStatus.PAUSED
        return self._record

    def resume_job(self, job_id: str) -> JobRecord:
        self._record.status = JobStatus.DONE
        return self._record

    def cancel_job(self, job_id: str) -> JobRecord:
        self._record.status = JobStatus.CANCELLED
        return self._record

    def get_notifications(self, job_id: str):
        return [{"kind": "runtime_wait"}]

    def get_events(self, job_id: str):
        return []


def test_daemon_status_command(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "load_runner_for_workspace", lambda **kwargs: _StubRunner())

    exit_code = cli.main(["daemon", "status", "--workspace", str(tmp_path)])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["heartbeats"][0]["worker_id"] == "worker-1"


def test_jobs_submit_and_show_commands(monkeypatch, tmp_path, capsys) -> None:
    runner = _StubRunner()
    monkeypatch.setattr(cli, "load_runner_for_workspace", lambda **kwargs: runner)
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "request_text": "Implement durable runtime",
                "repo_path": str(tmp_path),
                "target_branch": "acos/durable-runtime",
            }
        ),
        encoding="utf-8",
    )

    submit_exit = cli.main(["jobs", "submit", "--file", str(job_file), "--workspace", str(tmp_path)])
    show_exit = cli.main(["jobs", "show", runner._record.job_id, "--workspace", str(tmp_path)])

    assert submit_exit == 0
    assert show_exit == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["job"]["status"] == "queued"


def test_runtime_status_and_check_provider_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_runner_for_workspace", lambda **kwargs: _StubRunner())
    checker = SimpleNamespace(
        check_provider=lambda provider: SimpleNamespace(
            model_dump=lambda mode="json": {"provider_key": provider, "status": "ok"},
            status=SimpleNamespace(value="ok"),
        )
    )
    monkeypatch.setattr(cli, "build_health_checker", lambda config_dir: (object(), checker))

    runtime_exit = cli.main(["runtime", "status"])
    provider_exit = cli.main(["check-provider", "--provider", "local_qwen"])

    assert runtime_exit == 0
    assert provider_exit == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["provider_key"] == "local_qwen"


def test_launchd_plist_generation(tmp_path) -> None:
    payload = cli.build_launchd_plist(workspace_root=tmp_path, config_dir="configs")
    data = plistlib.loads(plistlib.dumps(payload))

    assert data["Label"] == "com.acos.worker"
    assert data["ProgramArguments"][:4] == ["acos", "worker", "run", "--forever"]
    assert data["WorkingDirectory"] == str(tmp_path.resolve())
