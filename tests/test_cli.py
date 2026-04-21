from __future__ import annotations

from pathlib import Path

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
    assert {"qwen_35b", "qwen_small"} <= model_keys
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
    assert "qwen_small" in implementer["fallback_models"]
    assert implementer["allowed_tools_count"] == 4
    assert implementer["output_schema"] == "ImplementationResult"


def test_resolve_model_for_implementer_returns_qwen_35b(capsys) -> None:
    exit_code = cli.main(["resolve-model", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["selected_model"] == "qwen_35b"
    assert payload["provider"] == "local_qwen"
    assert payload["routing_reason"] == "role_default"
    assert "qwen_small" in payload["fallback_candidates"]


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
    assert payload["context_budget"]["agent_context_budget_tokens"] == 128000
    assert any("normally uses qwen_35b" in line for line in payload["human_summary"])


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

    def fake_run_job(self, spec: JobSpec) -> JobRecord:
        assert Path(spec.workspace_root or spec.repo_path) == workspace.resolve()
        assert workspace.is_dir()
        assert (workspace / ".acos_memory.sqlite3").exists()
        assert (workspace / ".acos_approvals.sqlite3").exists()
        return JobRecord(
            job_id=spec.job_id,
            spec=spec,
            status=JobStatus.DONE,
            outputs={"summary": {"status": "ok"}},
        )

    monkeypatch.setattr(cli.JobRunner, "run_job", fake_run_job)

    exit_code = cli.main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"
