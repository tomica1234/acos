from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from apps import cli
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus

from tests.conftest import config_dir


def _copy_configs(tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    target.mkdir()
    for name in [
        "model_providers.yaml",
        "agents.yaml",
        "model_routing.yaml",
        "policies.yaml",
        "runtime.yaml",
        "worker.yaml",
    ]:
        (target / name).write_text((config_dir() / name).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_validate_config_succeeds(capsys) -> None:
    exit_code = cli.main(["validate-config", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_validate_config_fails_for_broken_config(tmp_path, capsys) -> None:
    configs = _copy_configs(tmp_path)
    providers = yaml.safe_load((configs / "model_providers.yaml").read_text(encoding="utf-8"))
    providers["models"]["qwen_35b"]["provider"] = "missing_provider"
    (configs / "model_providers.yaml").write_text(yaml.safe_dump(providers), encoding="utf-8")

    exit_code = cli.main(["validate-config", "--config-dir", str(configs)])

    assert exit_code == 1
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("missing_provider" in error for error in payload["errors"])


def test_list_models_returns_expected_fields(capsys) -> None:
    exit_code = cli.main(["list-models", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    model_keys = {item["model_key"] for item in payload["models"]}
    assert "qwen_35b" in model_keys
    assert "qwen_small" not in model_keys
    qwen = next(item for item in payload["models"] if item["model_key"] == "qwen_35b")
    assert qwen["provider"] == "local_qwen"
    assert qwen["display_name"] == "Qwen 3.6 35B A3B"
    assert qwen["supports_tool_calling"] is True
    assert qwen["supports_structured_output"] is False
    assert "agentic" in qwen["tags"]


def test_list_agents_returns_role_model_mapping(capsys) -> None:
    exit_code = cli.main(["list-agents", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    implementer = next(item for item in payload["agents"] if item["role"] == "implementer")
    assert implementer["primary_model"] == "qwen_35b"
    assert implementer["fallback_models"] == []
    assert implementer["allowed_tools_count"] == 4
    assert implementer["output_schema"] == "ImplementationResult"


def test_resolve_model_for_implementer_returns_qwen_35b(capsys) -> None:
    exit_code = cli.main(["resolve-model", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["selected_model"] == "qwen_35b"
    assert payload["provider"] == "local_qwen"
    assert payload["routing_reason"] == "role_default"
    assert payload["fallback_candidates"] == []


def test_resolve_model_with_repeated_failures_returns_escalation(capsys) -> None:
    exit_code = cli.main(
        [
            "resolve-model",
            "--config-dir",
            str(config_dir()),
            "--role",
            "implementer",
            "--repeated-failures",
            "2",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["selected_model"] == "qwen_35b"
    assert payload["routing_reason"] == "escalation"
    assert payload["details"]["repeated_failures"] == 2
    assert payload["escalation_condition_summary"]["escalated_model"] == "qwen_35b"


def test_explain_routing_includes_human_readable_sections(capsys) -> None:
    exit_code = cli.main(["explain-routing", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["role"] == "implementer"
    assert payload["normal_model"]["model_key"] == "qwen_35b"
    assert payload["current_selection"]["selected_model"] == "qwen_35b"
    assert payload["capability_requirements"]["requires_tools"] is True
    assert payload["context_budget"]["agent_context_budget_tokens"] == 262144
    assert any("normally uses qwen_35b" in line for line in payload["human_summary"])


def test_debug_token_budget_reports_resolved_values(tmp_path: Path, capsys) -> None:
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "request_text": "Inspect token budget",
                "repo_path": str(tmp_path),
                "workspace_root": str(tmp_path),
                "target_branch": "acos/debug-budget",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "debug",
            "token-budget",
            "--config-dir",
            str(config_dir()),
            "--role",
            "pm",
            "--file",
            str(job_file),
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["role"] == "pm"
    assert payload["selected_model"] == "qwen_35b"
    assert payload["configured_max_output_tokens"] == "auto"
    assert isinstance(payload["resolved_max_output_tokens"], int)
    assert payload["context_budget_tokens"] == 262144


def test_run_job_accepts_friendly_yaml_shape(tmp_path: Path, monkeypatch, capsys) -> None:
    captured: dict[str, JobSpec] = {}

    class StubRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["spec"] = spec
            return JobRecord(
                job_id=spec.job_id,
                spec=spec,
                status=JobStatus.DONE,
                outputs={"summary": {"status": "ok"}},
            )

    def fake_build_default_runner(config_dir: str, workspace_root: str):
        return StubRunner(), object()

    monkeypatch.setattr(cli, "build_default_runner", fake_build_default_runner)
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "title": "Sample feature",
                "requester_input": "READMEにセットアップ手順を追加してください",
                "repo_path": str(tmp_path),
                "target_branch": "acos/sample-job",
                "repo_url": None,
                "base_branch": "main",
                "autonomy_level": 4,
                "notification_channel": "console",
                "constraints": {
                    "allow_dependency_addition": False,
                    "allow_db_migration": False,
                    "allow_external_network_during_tests": False,
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    assert captured["spec"].request_text == "READMEにセットアップ手順を追加してください"
    assert captured["spec"].target_branch == "acos/sample-job"
    assert captured["spec"].metadata["title"] == "Sample feature"
    assert captured["spec"].metadata["base_branch"] == "main"
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"
    assert payload["metadata"]["notification_channel"] == "console"


def test_run_job_bootstraps_missing_workspace(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "greenfield-workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "request_text": "Create a new FastAPI app",
                "repo_path": str(workspace),
                "workspace_root": str(workspace),
                "target_branch": "acos/greenfield-app",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class StubRunner:
        def __init__(self) -> None:
            self._record: JobRecord | None = None

        def submit(self, spec: JobSpec) -> JobRecord:
            assert Path(spec.workspace_root or spec.repo_path) == workspace.resolve()
            self._record = JobRecord(
                job_id=spec.job_id,
                spec=spec,
                status=JobStatus.SUBMITTED,
                outputs={},
            )
            return self._record

        def run_next_step(self, job_id: str) -> JobRecord:
            assert self._record is not None
            self._record.status = JobStatus.DONE
            self._record.current_phase = "release"
            self._record.outputs = {"summary": {"status": "ok"}}
            return self._record

        def get_notifications(self, job_id: str) -> list[dict[str, str]]:
            return []

    def fake_build_default_runner(**_: object) -> tuple[StubRunner, object]:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / ".acos_memory.sqlite3").touch(exist_ok=True)
        (workspace / ".acos_approvals.sqlite3").touch(exist_ok=True)
        assert workspace.is_dir()
        assert (workspace / ".acos_memory.sqlite3").exists()
        assert (workspace / ".acos_approvals.sqlite3").exists()
        return StubRunner(), object()

    monkeypatch.setattr(
        cli,
        "build_default_runner",
        fake_build_default_runner,
    )

    exit_code = cli.main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_run_job_prints_live_progress_to_stderr(tmp_path: Path, monkeypatch, capsys) -> None:
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "request_text": "Create a new FastAPI app",
                "repo_path": str(tmp_path),
                "target_branch": "acos/live-progress",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class StubRunner:
        def __init__(self) -> None:
            self._record: JobRecord | None = None

        def submit(self, spec: JobSpec) -> JobRecord:
            self._record = JobRecord(
                job_id=spec.job_id,
                spec=spec,
                status=JobStatus.SUBMITTED,
                outputs={},
            )
            return self._record

        def run_next_step(self, job_id: str) -> JobRecord:
            assert self._record is not None
            if self._record.status == JobStatus.SUBMITTED:
                self._record.status = JobStatus.TESTING
                self._record.current_phase = "tests"
                self._record.outputs["test_run"] = {
                    "success": True,
                    "command": ["pytest", "-q"],
                }
                return self._record
            self._record.status = JobStatus.DONE
            self._record.current_phase = "runtime_smoke"
            self._record.outputs["runtime_smoke"] = {
                "success": True,
                "command": ["runtime-smoke-auto"],
            }
            return self._record

        def get_notifications(self, job_id: str) -> list[dict[str, str]]:
            return [{"job_id": job_id, "kind": "job_completed", "message": "finished"}]

    monkeypatch.setattr(
        cli,
        "load_runner_for_workspace",
        lambda **_: StubRunner(),
    )

    exit_code = cli.main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "submitted job=" in captured.err
    assert "phase=tests" in captured.err
    assert "notification kind=job_completed detail=finished" in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["status"] == "done"


def test_run_job_uses_durable_worker_when_available(tmp_path: Path, monkeypatch, capsys) -> None:
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "request_text": "Create a new FastAPI app",
                "repo_path": str(tmp_path),
                "target_branch": "acos/durable-run-job",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class StubRunner:
        def __init__(self) -> None:
            self._record: JobRecord | None = None

        def submit(self, spec: JobSpec) -> JobRecord:
            self._record = JobRecord(
                job_id=spec.job_id,
                spec=spec,
                status=JobStatus.SUBMITTED,
                outputs={},
            )
            return self._record

        def get(self, job_id: str) -> JobRecord:
            assert self._record is not None
            return self._record

        def get_notifications(self, job_id: str) -> list[dict[str, str]]:
            return []

    class StubDaemon:
        def __init__(self, runner: StubRunner) -> None:
            self.runner = runner
            self.calls = 0
            self.config = SimpleNamespace(poll_interval_seconds=0)

        def run_once(self) -> list[JobRecord]:
            self.calls += 1
            record = self.runner.get("ignored")
            if record.status == JobStatus.SUBMITTED:
                record.status = JobStatus.TESTING
                record.current_phase = "tests"
                record.outputs["test_run"] = {
                    "success": True,
                    "command": ["pytest", "-q"],
                }
            else:
                record.status = JobStatus.DONE
                record.current_phase = "release"
                record.outputs["summary"] = {"status": "ok"}
            return [record]

    runner = StubRunner()
    daemon = StubDaemon(runner)
    monkeypatch.setattr(
        cli,
        "load_runner_for_workspace",
        lambda **_: runner,
    )
    monkeypatch.setattr(
        cli,
        "build_worker_daemon",
        lambda **_: daemon,
    )

    exit_code = cli.main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    assert daemon.calls >= 2
    captured = capsys.readouterr()
    assert "phase=tests" in captured.err
    payload = yaml.safe_load(captured.out)
    assert payload["status"] == "done"
