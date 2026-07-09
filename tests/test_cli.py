from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apps.cli import (
    SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS,
    apply_constraint_overrides,
    apply_recovery_overrides,
    autonomous_result_payload,
    build_job_spec_from_request,
    load_job_spec_from_file,
    main,
    probe_provider,
    supervised_model_timeout_seconds,
    supervised_model_timeout_deadline_epoch,
)
from packages.orchestrator.job_constraints import STRICT_JOB_CONSTRAINTS
from packages.orchestrator.job_runner import build_default_runner
from packages.orchestrator.job_store import FileJobStore
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import config_dir


def _strict_ready_task_graph() -> TaskGraph:
    return TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["Core behavior works"],
                target_files=["src/core.py"],
                required_artifacts=["src/core.py"],
            ),
        ],
    )


def _mark_strict_planning_ready(record: JobRecord, task_graph: TaskGraph) -> None:
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["prd_quality"] = {
        "passed": True,
        "missing": [],
        "warnings": [],
    }
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "errors": [],
        "task_count": len(task_graph.tasks),
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "implementation_task_artifact_count": 1,
        "require_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_executable_task_roles": True,
        "unsupported_task_role_count": 0,
        "small_part_count": 1,
        "small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Build core",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_small_parts": [],
        "acceptance_test_count": 1,
        "acceptance_test_coverage": [
            {
                "acceptance_test_index": 1,
                "acceptance_test": "Core behavior works",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_acceptance_tests": [],
    }


def _copy_configs(tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    target.mkdir()
    for name in [
        "model_providers.yaml",
        "agents.yaml",
        "model_routing.yaml",
        "policies.yaml",
    ]:
        (target / name).write_text((config_dir() / name).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_validate_config_succeeds(capsys) -> None:
    exit_code = main(["validate-config", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_validate_config_fails_for_broken_config(tmp_path, capsys) -> None:
    configs = _copy_configs(tmp_path)
    providers = yaml.safe_load((configs / "model_providers.yaml").read_text(encoding="utf-8"))
    providers["models"]["ornith_35b_q4"]["provider"] = "missing_provider"
    (configs / "model_providers.yaml").write_text(yaml.safe_dump(providers), encoding="utf-8")

    exit_code = main(["validate-config", "--config-dir", str(configs)])

    assert exit_code == 1
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("missing_provider" in error for error in payload["errors"])


def test_list_models_returns_expected_fields(capsys) -> None:
    exit_code = main(["list-models", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    model_keys = {item["model_key"] for item in payload["models"]}
    assert "ornith_35b_q4" in model_keys
    ornith = next(item for item in payload["models"] if item["model_key"] == "ornith_35b_q4")
    assert ornith["provider"] == "local_ornith"
    assert ornith["supports_tool_calling"] is True
    assert "agentic" in ornith["tags"]


def test_list_agents_returns_role_model_mapping(capsys) -> None:
    exit_code = main(["list-agents", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    implementer = next(item for item in payload["agents"] if item["role"] == "implementer")
    assert implementer["primary_model"] == "ornith_35b_q4"
    assert "mock_structured" in implementer["fallback_models"]


def test_resolve_model_for_implementer_returns_ornith_35b_q4(capsys) -> None:
    exit_code = main(["resolve-model", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["model_key"] == "ornith_35b_q4"
    assert payload["reason"] == "role_default"


def test_resolve_model_with_repeated_failures_returns_escalation(capsys) -> None:
    exit_code = main(
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
    assert payload["model_key"] == "ncmoe40_q4"
    assert payload["reason"] == "escalation"
    assert payload["details"]["repeated_failures"] == 2


def test_explain_routing_includes_selection_and_conditions(capsys) -> None:
    exit_code = main(["explain-routing", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["role"] == "implementer"
    assert payload["selection"]["model_key"] == "ornith_35b_q4"
    assert payload["primary_model"] == "ornith_35b_q4"
    assert "mock_structured" in payload["fallback_models"]
    assert "timeout" in payload["fallback_errors"]


def test_load_job_spec_from_file_supports_job_yaml_shape(tmp_path: Path) -> None:
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "title": "demo-job",
                "requester_input": "Build something useful.",
                "repo_path": str(tmp_path / "repo"),
                "workspace_root": str(tmp_path / "workspace"),
                "target_branch": "acos/demo",
                "constraints": {"allow_dependency_addition": True},
                "autonomy_level": 4,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    spec = load_job_spec_from_file(job_file)

    assert spec.title == "demo-job"
    assert spec.request_text == "Build something useful."
    assert spec.target_branch == "acos/demo"
    assert spec.workspace_root == str((tmp_path / "workspace").resolve())
    assert spec.repo_path == str((tmp_path / "repo").resolve())
    assert spec.metadata["constraints"]["allow_dependency_addition"] is True
    assert spec.metadata["autonomy_level"] == 4


def test_build_job_spec_from_request_resolves_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    spec = build_job_spec_from_request(
        request_text="  Build something substantial.  ",
        repo_path=workspace,
        workspace_root=None,
        target_branch="acos/direct-request",
        job_id="direct-request-job",
        title="Direct Request",
    )

    assert spec.job_id == "direct-request-job"
    assert spec.title == "Direct Request"
    assert spec.request_text == "Build something substantial."
    assert spec.repo_path == str(workspace.resolve())
    assert spec.workspace_root == str(workspace.resolve())
    assert spec.target_branch == "acos/direct-request"


def test_build_default_runner_creates_missing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "missing-workspace"

    runner, environment = build_default_runner(
        config_dir=config_dir(),
        workspace_root=workspace,
    )

    assert runner is not None
    assert environment is not None
    assert workspace.is_dir()
    assert not (workspace / ".acos_memory.sqlite3").exists()
    memory_files = list((config_dir().parent / ".acos" / "memory").glob("*.sqlite3"))
    assert memory_files
    assert "mock_structured" not in runner.registry.get_agent("implementer").fallback_models


def test_probe_provider_reports_unreachable_local_server() -> None:
    runner, _ = build_default_runner(
        config_dir=config_dir(),
        workspace_root=Path.cwd(),
    )
    runner.registry.providers["local_ornith"].base_url = "http://127.0.0.1:9/v1"

    payload = probe_provider(runner.registry, "local_ornith", timeout_seconds=0.1)

    assert payload["provider"] == "local_ornith"
    assert payload["healthy"] is False
    assert payload["status"] in {"down", "error"}


def test_check_provider_command_uses_probe_result(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": False,
            "status": "down",
            "error": "connection refused",
        },
    )

    exit_code = main(
        [
            "check-provider",
            "--config-dir",
            str(config_dir()),
            "--provider",
            "local_ornith",
        ]
    )

    assert exit_code == 1
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["provider"] == "local_ornith"
    assert payload["status"] == "down"


def test_run_job_uses_workspace_root_from_job_file(tmp_path: Path, monkeypatch, capsys) -> None:
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build something useful.",
                "repo_path": str(repo),
                "workspace_root": str(workspace),
                "target_branch": "acos/demo",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["constraints"] = dict(spec.metadata["constraints"])
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["config_dir"] = str(config_dir)
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(["run-job", "--config-dir", str(config_dir()), "--file", str(job_file)])

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["constraints"] == {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    assert captured["store"] is not None
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_run_job_can_set_initial_autonomous_stage_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build something large.",
                "repo_path": str(workspace),
                "target_branch": "acos/chunked-demo",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["constraints"] = dict(spec.metadata["constraints"])
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "run-job",
            "--config-dir",
            str(config_dir()),
            "--file",
            str(job_file),
            "--max-autonomous-stages",
            "2",
            "--require-prd-quality",
            "--stage-review",
            "--test-timeout-seconds",
            "600",
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["constraints"]["max_autonomous_stages"] == 2
    assert captured["constraints"]["require_prd_quality"] is True
    assert captured["constraints"]["stage_review"] is True
    assert captured["constraints"]["test_timeout_seconds"] == 600
    assert captured["store"] is not None
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_plan_job_from_direct_request_stops_before_implementation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    jobs_dir = tmp_path / "jobs"
    summary_file = tmp_path / "plan-summary.json"
    captured: dict[str, object] = {}

    class DummyRunner:
        def plan_job(self, spec: JobSpec) -> JobRecord:
            captured["spec"] = spec
            captured["constraints"] = dict(spec.metadata["constraints"])
            record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.PLANNING)
            record.outputs["task_graph"] = TaskGraph(
                goal="Build it",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                        acceptance_criteria=["core works"],
                        target_files=["feature.py"],
                    )
                ],
            ).model_dump()
            record.outputs["prd_quality"] = {
                "passed": True,
                "missing": [],
                "warnings": [],
            }
            record.outputs["task_graph_validation"] = {
                "valid": True,
                "task_count": 1,
                "implementation_task_count": 1,
                "implementation_task_acceptance_criteria_count": 1,
                "implementation_task_artifact_count": 1,
                "require_acceptance_criteria": True,
                "require_task_artifacts": True,
                "require_executable_task_roles": True,
                "unsupported_task_role_count": 0,
                "small_part_count": 1,
                "small_part_coverage": [
                    {
                        "small_part_index": 1,
                        "small_part": "Build core",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_small_parts": [],
                "acceptance_test_count": 1,
                "acceptance_test_coverage": [
                    {
                        "acceptance_test_index": 1,
                        "acceptance_test": "core works",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_acceptance_tests": [],
                "errors": [],
            }
            record.outputs["planning_only"] = {
                "complete": True,
                "ready_for_implementation": True,
            }
            return record

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "plan-job",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large project tracker.",
            "--repo-path",
            str(workspace),
            "--job-id",
            "plan-only-job",
            "--jobs-dir",
            str(jobs_dir),
            "--summary-file",
            str(summary_file),
        ]
    )

    assert exit_code == 0
    spec = captured["spec"]
    assert isinstance(spec, JobSpec)
    assert spec.request_text == "Build a large project tracker."
    assert spec.repo_path == str(workspace.resolve())
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["store"] is not None
    assert captured["constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload
    assert payload["job_id"] == "plan-only-job"
    assert payload["status"] == "planning"
    assert payload["planning_complete"] is True
    assert payload["terminal_reason"] == "planned"
    expected_continue_args = [
        "continue-job",
        "--job-id",
        "plan-only-job",
        "--max-steps",
        "1",
        "--json-summary",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["operator_decision"]["action"] == "continue"
    assert payload["operator_decision"]["command"] == "acos " + " ".join(
        expected_continue_args
    )
    assert payload["stop_summary"]["terminal_reason"] == "planned"
    assert payload["stop_summary"]["operator_action"] == "continue"
    assert payload["stop_summary"]["planning_summary"] == payload["summary"][
        "planning_summary"
    ]
    assert payload["stop_summary"]["planning_summary"]["ready_for_implementation"] is True


def test_plan_job_preflight_stops_before_runner_when_provider_unhealthy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    summary_file = tmp_path / "plan-provider-down.json"

    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": False,
            "status": "down",
            "probe_timeout_seconds": timeout_seconds,
        },
    )

    def fail_build_default_runner(*args, **kwargs):
        raise AssertionError("plan-job should not start a runner when preflight fails")

    monkeypatch.setattr("apps.cli.build_default_runner", fail_build_default_runner)

    exit_code = main(
        [
            "plan-job",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large project tracker.",
            "--repo-path",
            str(tmp_path / "workspace"),
            "--job-id",
            "plan-provider-down-job",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--summary-file",
            str(summary_file),
            "--preflight-provider",
            "local_ornith",
            "--preflight-timeout",
            "0.25",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload
    assert payload["job_id"] == "plan-provider-down-job"
    assert payload["terminal_reason"] == "provider_unhealthy"
    assert payload["provider_preflight"] == {
        "provider": "local_ornith",
        "healthy": False,
        "status": "down",
        "probe_timeout_seconds": 0.25,
    }
    assert payload["operator_decision"]["action"] == "inspect"
    assert payload["stop_summary"] == {
        "terminal_reason": "provider_unhealthy",
        "operator_action": "inspect",
        "operator_command": None,
        "resume_action": "check_provider",
        "can_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
        "provider_event_count": 1,
        "last_provider_event": payload["provider_events"][-1],
    }


def test_plan_job_can_suggest_supervised_execution_after_planning(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    jobs_dir = tmp_path / "jobs"
    summary_file = tmp_path / "plan-summary.json"
    final_summary_file = tmp_path / "final-summary.json"
    cycle_summary_dir = tmp_path / "cycles"

    class DummyRunner:
        def plan_job(self, spec: JobSpec) -> JobRecord:
            record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.PLANNING)
            record.outputs["task_graph"] = TaskGraph(
                goal="Build it",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                        acceptance_criteria=["core works"],
                        target_files=["feature.py"],
                    )
                ],
            ).model_dump()
            record.outputs["prd_quality"] = {
                "passed": True,
                "missing": [],
                "warnings": [],
            }
            record.outputs["task_graph_validation"] = {
                "valid": True,
                "task_count": 1,
                    "implementation_task_count": 1,
                    "implementation_task_acceptance_criteria_count": 1,
                    "implementation_task_artifact_count": 1,
                    "require_acceptance_criteria": True,
                    "require_task_artifacts": True,
                    "require_executable_task_roles": True,
                "unsupported_task_role_count": 0,
                "small_part_count": 1,
                "small_part_coverage": [
                    {
                        "small_part_index": 1,
                        "small_part": "Build core",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_small_parts": [],
                "acceptance_test_count": 1,
                "acceptance_test_coverage": [
                    {
                        "acceptance_test_index": 1,
                        "acceptance_test": "core works",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_acceptance_tests": [],
                "errors": [],
            }
            record.outputs["planning_only"] = {
                "complete": True,
                "ready_for_implementation": True,
            }
            return record

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "plan-job",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large project tracker.",
            "--repo-path",
            str(workspace),
            "--job-id",
            "plan-supervise-job",
            "--jobs-dir",
            str(jobs_dir),
            "--summary-file",
            str(summary_file),
            "--supervise-after-planning",
            "--supervise-max-cycles",
            "7",
            "--supervise-steps-per-cycle",
            "2",
            "--supervise-max-stalled-cycles",
            "4",
            "--supervise-max-runtime-seconds",
            "3600",
            "--supervise-summary-file",
            str(final_summary_file),
            "--supervise-summary-dir",
            str(cycle_summary_dir),
            "--supervise-preflight-provider",
            "local_ornith",
            "--supervise-preflight-timeout",
            "0.5",
            "--supervise-autonomous-until-done",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "plan-supervise-job",
        "--max-cycles",
        "7",
        "--steps-per-cycle",
        "2",
        "--max-stalled-cycles",
        "4",
        "--max-runtime-seconds",
        "3600.0",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
        "--workspace",
        str(workspace.resolve()),
        "--summary-file",
        str(final_summary_file),
        "--summary-dir",
        str(cycle_summary_dir),
        "--large-autonomous",
        "--preflight-provider",
        "local_ornith",
        "--preflight-timeout",
        "0.5",
        "--pm-stall-recovery",
        "--autonomous-until-done",
    ]
    assert payload["planning_complete"] is True
    assert payload["terminal_reason"] == "planned"
    assert payload["autonomous_until_done"] is True
    assert payload["can_supervise_continue"] is True
    assert payload["next_supervise_cli_args"] == expected_supervise_args
    assert payload["next_supervise_command"] == "acos " + " ".join(
        expected_supervise_args
    )
    assert payload["operator_decision"]["action"] == "supervise"
    assert payload["operator_decision"]["command"] == payload["next_supervise_command"]
    assert payload["stop_summary"]["operator_action"] == "supervise"
    assert payload["stop_summary"]["operator_command"] == payload[
        "next_supervise_command"
    ]
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload


def test_run_job_large_autonomous_sets_safe_defaults(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build something large.",
                "repo_path": str(workspace),
                "target_branch": "acos/large-autonomous",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["constraints"] = dict(spec.metadata["constraints"])
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "run-job",
            "--config-dir",
            str(config_dir()),
            "--file",
            str(job_file),
            "--large-autonomous",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_large_autonomous_overrides_disabled_quality_gates() -> None:
    spec = JobSpec(
        request_text="Build something large.",
        repo_path=".",
        metadata={
            "constraints": {
                key: False for key in STRICT_JOB_CONSTRAINTS
            }
        },
    )

    apply_constraint_overrides(spec, large_autonomous=True)

    constraints = spec.metadata["constraints"]
    for key, value in STRICT_JOB_CONSTRAINTS.items():
        assert constraints[key] is value
    assert constraints["test_timeout_seconds"] == 1200


def test_jobs_submit_applies_strict_quality_gates(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Queue a normal job.",
                "repo_path": str(workspace),
                "target_branch": "acos/queued-job",
                "metadata": {
                    "constraints": {
                        key: False for key in STRICT_JOB_CONSTRAINTS
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class DummyRunner:
        def submit(self, spec: JobSpec) -> JobRecord:
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.QUEUED)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path):
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "jobs",
            "submit",
            "--config-dir",
            str(config_dir()),
            "--workspace",
            str(workspace),
            "--file",
            str(job_file),
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    constraints = payload["job"]["spec"]["metadata"]["constraints"]
    for key, value in STRICT_JOB_CONSTRAINTS.items():
        assert constraints[key] is value
    assert constraints["test_timeout_seconds"] == 1200


def test_run_demo_uses_strict_ready_planning_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    workspace = tmp_path / "demo-workspace"

    exit_code = main(
        [
            "run-demo",
            "--config-dir",
            str(config_dir()),
            "--workspace",
            str(workspace),
        ]
    )

    assert exit_code == 0
    payload, _end_index = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert payload["status"] == JobStatus.DONE.value
    constraints = payload["spec"]["metadata"]["constraints"]
    for key, value in STRICT_JOB_CONSTRAINTS.items():
        assert constraints[key] is value
    assert constraints["test_timeout_seconds"] == 1200
    assert payload["outputs"]["prd_quality"]["passed"] is True
    assert payload["outputs"]["task_graph_validation"]["valid"] is True
    assert payload["outputs"]["completion_integrity"]["passed"] is True


def test_run_autonomous_starts_large_job_and_continues(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build something large.",
                "repo_path": str(workspace),
                "target_branch": "acos/large-autonomous",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary_file = tmp_path / "summaries" / "run-autonomous.json"
    captured: dict[str, object] = {"resume_limits": []}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["run_constraints"] = dict(spec.metadata["constraints"])
            record = captured["store"].create(spec)
            record.status = JobStatus.BLOCKED
            record.last_error = "autonomous_stage_limit_reached"
            record.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": 1,
                "completed_stage_count": 1,
            }
            captured["store"].update(record)
            return record

        def resume_job(self, job_id: str) -> JobRecord:
            record = captured["store"].get(job_id)
            captured["resume_limits"].append(
                record.spec.metadata["constraints"]["max_autonomous_stages"]
            )
            record.status = JobStatus.DONE
            record.last_error = None
            captured["store"].update(record)
            return record

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "run-autonomous",
            "--config-dir",
            str(config_dir()),
            "--file",
            str(job_file),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--max-steps",
            "1",
            "--json-summary",
            "--summary-file",
            str(summary_file),
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["run_constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    assert captured["resume_limits"] == [2]
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload
    assert payload["started"] is True
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["max_steps"] == 1
    assert payload["step_events"] == [
        {
            "step": 1,
            "action": "raise_stage_limit_or_resume",
            "task_id": None,
            "status_before": "blocked",
            "last_error_before": "autonomous_stage_limit_reached",
            "recovery_strategy": "raise_stage_limit",
            "recovery_mode": "stage_limit",
            "max_autonomous_stages": 2,
            "status_after": "done",
            "last_error_after": None,
        }
    ]
    assert payload["terminal_reason"] == "done"
    assert payload["next_action"] == "none"
    assert payload["can_continue"] is False
    assert payload["next_continue_cli_args"] == []
    assert payload["next_continue_command"] is None
    assert payload["status"] == "done"
    assert payload["summary"]["status"] == "done"


def test_run_supervised_starts_job_and_runs_supervision_cycles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build a larger app autonomously.",
                "repo_path": str(workspace),
                "target_branch": "acos/run-supervised",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    final_summary_file = tmp_path / "summaries" / "run-supervised-final.json"
    cycle_summary_dir = tmp_path / "summaries" / "run-supervised-cycles"
    captured: dict[str, object] = {"resume_limits": []}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["run_constraints"] = dict(spec.metadata["constraints"])
            record = captured["store"].create(spec)
            record.status = JobStatus.BLOCKED
            record.last_error = "autonomous_stage_limit_reached"
            record.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": 1,
                "completed_stage_count": 1,
            }
            captured["store"].update(record)
            return record

        def resume_job(self, job_id: str) -> JobRecord:
            record = captured["store"].get(job_id)
            limit = record.spec.metadata["constraints"]["max_autonomous_stages"]
            captured["resume_limits"].append(limit)
            if len(captured["resume_limits"]) < 2:
                record.status = JobStatus.BLOCKED
                record.last_error = "autonomous_stage_limit_reached"
                record.outputs["autonomous_stage_limit"] = {
                    "max_autonomous_stages": limit,
                    "completed_stage_count": limit,
                }
            else:
                record.status = JobStatus.DONE
                record.last_error = None
            captured["store"].update(record)
            return record

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--file",
            str(job_file),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--max-cycles",
            "2",
            "--steps-per-cycle",
            "1",
            "--summary-file",
            str(final_summary_file),
            "--summary-dir",
            str(cycle_summary_dir),
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["run_constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    assert captured["resume_limits"] == [2, 3]
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(final_summary_file.read_text(encoding="utf-8")) == payload
    assert payload["started"] is True
    assert payload["continued"] is True
    assert payload["initial_status"] == "blocked"
    assert payload["status"] == "done"
    assert payload["terminal_reason"] == "done"
    assert payload["cycles_run"] == 2
    assert payload["max_cycles"] == 2
    assert payload["steps_per_cycle"] == 1
    assert payload["steps_run"] == 2
    assert [event["step"] for event in payload["step_events"]] == [1, 2]
    assert [event["cycle"] for event in payload["step_events"]] == [1, 2]
    assert len(payload["cycle_summaries"]) == 2
    for cycle in range(1, 3):
        cycle_payload = json.loads(
            (cycle_summary_dir / f"cycle-{cycle:03d}.json").read_text(encoding="utf-8")
        )
        assert cycle_payload == payload["cycle_summaries"][cycle - 1]


def test_run_supervised_can_plan_first_then_supervise(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    final_summary_file = tmp_path / "summaries" / "plan-first-final.json"
    captured: dict[str, object] = {"plan_count": 0, "resume_count": 0}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            raise AssertionError("run-supervised --plan-first should not call run_job")

        def plan_job(self, spec: JobSpec) -> JobRecord:
            captured["plan_count"] += 1
            captured["spec"] = spec
            captured["constraints"] = dict(spec.metadata["constraints"])
            record = captured["store"].create(spec)
            record.status = JobStatus.PLANNING
            record.outputs["task_graph"] = TaskGraph(
                goal="Build it",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                        acceptance_criteria=["core works"],
                        target_files=["feature.py"],
                    )
                ],
            ).model_dump()
            record.outputs["prd_quality"] = {
                "passed": True,
                "missing": [],
                "warnings": [],
            }
            record.outputs["task_graph_validation"] = {
                "valid": True,
                "task_count": 1,
                "implementation_task_count": 1,
                "implementation_task_acceptance_criteria_count": 1,
                "implementation_task_artifact_count": 1,
                "require_acceptance_criteria": True,
                "require_task_artifacts": True,
                "require_executable_task_roles": True,
                "unsupported_task_role_count": 0,
                "small_part_count": 1,
                "small_part_coverage": [
                    {
                        "small_part_index": 1,
                        "small_part": "Build core",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_small_parts": [],
                "acceptance_test_count": 1,
                "acceptance_test_coverage": [
                    {
                        "acceptance_test_index": 1,
                        "acceptance_test": "core works",
                        "task_id": "core",
                        "covered": True,
                    }
                ],
                "uncovered_acceptance_tests": [],
                "errors": [],
            }
            record.outputs["planning_only"] = {
                "complete": True,
                "ready_for_implementation": True,
            }
            captured["store"].update(record)
            return record

        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            record = captured["store"].get(job_id)
            record.status = JobStatus.DONE
            record.last_error = None
            captured["store"].update(record)
            return record

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large project tracker.",
            "--repo-path",
            str(workspace),
            "--job-id",
            "plan-first-supervised-job",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--summary-file",
            str(final_summary_file),
            "--plan-first",
        ]
    )

    assert exit_code == 0
    assert captured["plan_count"] == 1
    assert captured["resume_count"] == 1
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(final_summary_file.read_text(encoding="utf-8")) == payload
    assert payload["job_id"] == "plan-first-supervised-job"
    assert payload["planned_first"] is True
    assert payload["planning_complete"] is True
    assert payload["planning_result"]["terminal_reason"] == "planned"
    assert payload["planning_result"]["planning_complete"] is True
    assert payload["initial_status"] == "planning"
    assert payload["status"] == "done"
    assert payload["terminal_reason"] == "done"
    assert payload["cycles_run"] == 1
    assert payload["steps_run"] == 1
    assert payload["step_events"] == [
        {
            "step": 1,
            "action": "continue_next_task",
            "task_id": "core",
            "status_before": "planning",
            "last_error_before": None,
            "max_autonomous_stages": 1,
            "status_after": "done",
            "last_error_after": None,
            "cycle": 1,
            "cycle_step": 1,
        }
    ]


def test_run_supervised_can_start_from_direct_request(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "direct-workspace"
    captured: dict[str, object] = {}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["spec"] = spec
            captured["constraints"] = dict(spec.metadata["constraints"])
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.time", lambda: 1000.0)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a polished project tracker with tests.",
            "--repo-path",
            str(workspace),
            "--target-branch",
            "acos/direct-supervised",
            "--job-id",
            "direct-supervised-job",
            "--title",
            "Direct Supervised",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--max-runtime-seconds",
            "45",
        ]
    )

    assert exit_code == 0
    spec = captured["spec"]
    assert isinstance(spec, JobSpec)
    assert spec.job_id == "direct-supervised-job"
    assert spec.title == "Direct Supervised"
    assert spec.request_text == "Build a polished project tracker with tests."
    assert spec.repo_path == str(workspace.resolve())
    assert spec.workspace_root == str(workspace.resolve())
    assert spec.target_branch == "acos/direct-supervised"
    assert captured["workspace_root"] == str(workspace.resolve())
    assert captured["constraints"] == {
        "max_autonomous_stages": 1,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
        "model_timeout_seconds": 45.0,
        "model_timeout_deadline_epoch": 1045.0,
    }
    assert captured["store"] is not None
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "direct-supervised-job"
    assert payload["started"] is True
    assert payload["continued"] is False
    assert payload["done"] is True
    assert payload["initial_status"] == "done"
    assert payload["cycles_run"] == 0
    assert payload["provider_events"] == []
    assert payload["operator_decision"]["action"] == "done"
    assert payload["stop_summary"] == {
        "terminal_reason": "done",
        "operator_action": "done",
        "operator_command": None,
        "resume_action": "none",
        "can_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "provider_event_count": 0,
        "last_provider_event": None,
    }


def test_supervised_model_timeout_caps_long_runtime_budget() -> None:
    assert supervised_model_timeout_seconds(None) is None
    assert supervised_model_timeout_seconds(45.0) == 45.0
    assert (
        supervised_model_timeout_seconds(900.0)
        == SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS
    )
    assert supervised_model_timeout_seconds(900.0, elapsed_seconds=870.0) == 30.0
    assert supervised_model_timeout_seconds(900.0, elapsed_seconds=900.0) == 0.0


def test_supervised_model_timeout_deadline_epoch_uses_runtime_budget() -> None:
    assert supervised_model_timeout_deadline_epoch(None, started_epoch=1000.0) is None
    assert supervised_model_timeout_deadline_epoch(45.0, started_epoch=1000.0) == 1045.0


def test_run_supervised_caps_initial_model_timeout_for_long_runtime_budget(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "direct-workspace"
    captured: dict[str, object] = {}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["constraints"] = dict(spec.metadata["constraints"])
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.time", lambda: 2000.0)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a vocabulary app.",
            "--repo-path",
            str(workspace),
            "--job-id",
            "long-runtime-direct-supervised-job",
            "--max-runtime-seconds",
            "900",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"]["model_timeout_seconds"] == (
        SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS
    )
    assert captured["constraints"]["model_timeout_deadline_epoch"] == 2900.0
    json.loads(capsys.readouterr().out)


def test_run_supervised_plan_first_sets_runtime_deadline(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "plan-first-workspace"
    captured: dict[str, object] = {}

    class DummyRunner:
        def plan_job(self, spec: JobSpec) -> JobRecord:
            captured["constraints"] = dict(spec.metadata["constraints"])
            record = captured["store"].create(spec)
            record.status = JobStatus.DONE
            record.outputs["planning_only"] = {
                "complete": True,
                "ready_for_implementation": True,
            }
            captured["store"].update(record)
            return record

        def run_job(self, spec: JobSpec) -> JobRecord:
            raise AssertionError("run-supervised --plan-first should not call run_job")

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.time", lambda: 4000.0)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a vocabulary app.",
            "--repo-path",
            str(workspace),
            "--job-id",
            "plan-first-runtime-deadline-job",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--plan-first",
            "--max-runtime-seconds",
            "900",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"]["model_timeout_seconds"] == (
        SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS
    )
    assert captured["constraints"]["model_timeout_deadline_epoch"] == 4900.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planning_result"]["planning_complete"] is True


def test_run_supervised_preflight_stops_before_runner_when_provider_unhealthy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    summary_file = tmp_path / "summaries" / "provider-down.json"

    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": False,
            "status": "down",
            "probe_timeout_seconds": timeout_seconds,
        },
    )

    def fail_build_default_runner(*args, **kwargs):
        raise AssertionError("run-supervised should not start a runner when preflight fails")

    monkeypatch.setattr("apps.cli.build_default_runner", fail_build_default_runner)

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large app.",
            "--repo-path",
            str(tmp_path / "workspace"),
            "--job-id",
            "provider-down-job",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--preflight-provider",
            "local_ornith",
            "--preflight-timeout",
            "0.25",
            "--summary-file",
            str(summary_file),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload
    assert payload["job_id"] == "provider-down-job"
    assert payload["status"] == "blocked"
    assert payload["terminal_reason"] == "provider_unhealthy"
    assert payload["next_action"] == "check_provider"
    assert payload["can_continue"] is False
    assert payload["can_supervise_continue"] is False
    assert payload["next_supervise_cli_args"] == []
    assert payload["next_supervise_command"] is None
    assert payload["provider_preflight"] == {
        "provider": "local_ornith",
        "healthy": False,
        "status": "down",
        "probe_timeout_seconds": 0.25,
    }
    assert payload["provider_events"] == [
        {
            "cycle": None,
            "phase": "pre_start",
            "healthy": False,
            "terminal": True,
            "provider_preflight": payload["provider_preflight"],
        }
    ]
    assert payload["operator_decision"] == {
        "action": "inspect",
        "command": None,
        "resume_action": "check_provider",
        "reason": "provider_unhealthy",
        "requires_explicit_override": False,
        "autonomy_ready": None,
        "blocking_items": [
            {"type": "provider_unhealthy", "provider": "local_ornith"}
        ],
        "planning_strategy_change_recommended": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
    }
    assert payload["stop_summary"] == {
        "terminal_reason": "provider_unhealthy",
        "operator_action": "inspect",
        "operator_command": None,
        "resume_action": "check_provider",
        "can_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
        "provider_event_count": 1,
        "last_provider_event": payload["provider_events"][-1],
    }


def test_run_supervised_includes_successful_provider_preflight(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": True,
            "status": "ok",
            "probe_timeout_seconds": timeout_seconds,
        },
    )

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            return JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)

    monkeypatch.setattr(
        "apps.cli.build_default_runner",
        lambda config_dir, workspace_root, store=None: (DummyRunner(), None),
    )

    exit_code = main(
        [
            "run-supervised",
            "--config-dir",
            str(config_dir()),
            "--request",
            "Build a large app.",
            "--repo-path",
            str(tmp_path / "workspace"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--preflight-provider",
            "local_ornith",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["done"] is True
    assert payload["provider_preflight"] == {
        "provider": "local_ornith",
        "healthy": True,
        "status": "ok",
        "probe_timeout_seconds": 5.0,
    }
    assert payload["provider_events"] == [
        {
            "cycle": None,
            "phase": "pre_start",
            "healthy": True,
            "terminal": False,
            "provider_preflight": payload["provider_preflight"],
        }
    ]
    assert payload["stop_summary"] == {
        "terminal_reason": "done",
        "operator_action": "done",
        "operator_command": None,
        "resume_action": "none",
        "can_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "provider_preflight": payload["provider_preflight"],
        "provider_event_count": 1,
        "last_provider_event": payload["provider_events"][-1],
    }


def test_job_status_reads_persisted_record(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-test",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    record.completed_task_ids = ["core"]
    record.checkpoints = [{"kind": "autonomous_stage", "stage": 1}]
    record.outputs["task_graph"] = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
            PlannedTask(
                id="extra",
                title="Extra",
                description="Build extra",
                role="implementer",
                depends_on=["core"],
            ),
        ],
    ).model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": {"id": "core"},
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": True},
        }
    ]
    record.outputs["prd_quality_attempts"] = [
        {"attempt": 0, "action": "initial", "passed": False, "missing": ["small_parts"]}
    ]
    record.outputs["task_graph_validation_attempts"] = [
        {"attempt": 0, "action": "initial", "valid": True, "errors": []}
    ]
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-test",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "testing"
    assert payload["completed_task_ids"] == ["core"]
    assert payload["checkpoint_count"] == 1
    assert payload["total_tasks"] == 2
    assert payload["pending_task_ids"] == ["extra"]
    assert payload["next_task"]["id"] == "extra"
    assert payload["change_summary"]["changed_files"] == ["feature.py"]
    assert payload["change_summary"]["patch_count"] == 1
    assert payload["planning_quality"]["prd_quality_attempt_count"] == 1
    assert payload["planning_quality"]["task_graph_validation_attempt_count"] == 1
    assert payload["resume"]["suggested_cli_args"] == [
        "resume-job",
        "--job-id",
        "job-status-test",
    ]


def test_job_status_can_print_next_resume_command(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-next-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-command",
            "--next-command",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos resume-job --job-id job-status-next-command "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_can_print_next_continue_command(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-next-continue-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-continue-command",
            "--next-continue-command",
            "--continue-max-steps",
            "3",
            "--continue-json-summary",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-continue-command "
        "--max-steps 3 --json-summary "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_next_continue_command_uses_blocked_recovery_for_repeated_failure(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = _strict_ready_task_graph()
    spec = JobSpec(
        job_id="job-status-blocked-recovery-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    _mark_strict_planning_ready(record, task_graph)
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-blocked-recovery-command",
            "--next-continue-command",
            "--continue-max-steps",
            "2",
            "--continue-json-summary",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-blocked-recovery-command "
        "--max-steps 2 --json-summary "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_next_command_falls_back_to_blocked_recovery(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = _strict_ready_task_graph()
    spec = JobSpec(
        job_id="job-status-next-command-blocked-recovery",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    _mark_strict_planning_ready(record, task_graph)
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-command-blocked-recovery",
            "--next-command",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-command-blocked-recovery "
        "--max-steps 1 "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_can_print_next_operator_command_for_normal_continue(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-next-operator-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-operator-command",
            "--next-operator-command",
            "--continue-max-steps",
            "3",
            "--continue-json-summary",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-operator-command "
        "--max-steps 3 --json-summary "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_next_operator_command_uses_blocked_recovery(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="job-status-next-operator-blocked-recovery",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-operator-blocked-recovery",
            "--next-operator-command",
            "--continue-max-steps",
            "2",
            "--continue-json-summary",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-operator-blocked-recovery "
        "--max-steps 2 --json-summary "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_next_operator_command_uses_completion_integrity_recovery(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="job-status-next-operator-completion-integrity",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-operator-completion-integrity",
            "--next-operator-command",
            "--continue-max-steps",
            "2",
            "--continue-json-summary",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-operator-completion-integrity "
        "--max-steps 2 --json-summary "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_next_command_falls_back_to_completion_integrity_recovery(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="job-status-next-command-completion-integrity",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-command-completion-integrity",
            "--next-command",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos continue-job --job-id job-status-next-command-completion-integrity "
        "--max-steps 1 "
        f"--jobs-dir {jobs_dir}"
    )


def test_job_status_can_print_next_supervise_command(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    summary_file = tmp_path / "summaries" / "final.json"
    summary_dir = tmp_path / "summaries" / "cycles"
    spec = JobSpec(
        job_id="job-status-next-supervise-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-next-supervise-command",
            "--next-supervise-command",
            "--supervise-max-cycles",
            "4",
            "--supervise-steps-per-cycle",
            "2",
            "--supervise-max-stalled-cycles",
            "5",
            "--supervise-max-runtime-seconds",
            "60",
            "--supervise-summary-file",
            str(summary_file),
            "--supervise-summary-dir",
            str(summary_dir),
            "--supervise-workspace",
            str(workspace),
            "--supervise-large-autonomous",
            "--supervise-require-prd-quality",
            "--supervise-stage-review",
            "--supervise-test-timeout-seconds",
            "900",
            "--supervise-max-autonomous-stages",
            "6",
            "--supervise-preflight-provider",
            "local_ornith",
            "--supervise-preflight-timeout",
            "7",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == (
        "acos supervise-job --job-id job-status-next-supervise-command "
        "--max-cycles 4 --steps-per-cycle 2 --max-stalled-cycles 5 "
        "--max-runtime-seconds 60.0 "
        f"--jobs-dir {jobs_dir} --workspace {workspace} --summary-file {summary_file} "
        f"--summary-dir {summary_dir} "
        "--max-autonomous-stages 6 --large-autonomous --require-prd-quality "
        "--stage-review --test-timeout-seconds 900 --preflight-provider local_ornith "
        "--preflight-timeout 7.0"
    )


def test_job_status_preserves_autonomous_until_done_in_supervise_command_and_json(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-autonomous-until-done",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-autonomous-until-done",
            "--next-supervise-command",
            "--supervise-max-runtime-seconds",
            "60",
            "--supervise-autonomous-until-done",
        ]
    )

    assert exit_code == 0
    command = capsys.readouterr().out.strip()
    assert "--autonomous-until-done" in command
    assert "--pm-stall-recovery" in command

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-autonomous-until-done",
            "--json",
            "--supervise-max-runtime-seconds",
            "60",
            "--supervise-autonomous-until-done",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    supervision = payload["supervision"]
    assert supervision["autonomous_until_done"] is True
    assert "--autonomous-until-done" in supervision["next_supervise_cli_args"]
    assert "--autonomous-until-done" in supervision["next_supervise_command"]
    assert "--pm-stall-recovery" in supervision["next_supervise_cli_args"]


def test_job_status_next_supervise_command_is_empty_for_done_job(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-done-supervise-command",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.DONE
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-done-supervise-command",
            "--next-supervise-command",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == ""


def test_job_status_can_print_json_summary(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-json",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-json",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job-status-json"
    assert payload["status"] == "testing"
    assert payload["pending_task_ids"] == ["core"]
    expected_continue_args = [
        "continue-job",
        "--job-id",
        "job-status-json",
        "--max-steps",
        "1",
        "--jobs-dir",
        str(jobs_dir),
    ]
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "job-status-json",
        "--max-cycles",
        "10",
        "--steps-per-cycle",
        "1",
        "--max-stalled-cycles",
        "3",
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["continuation"] == {
        "can_continue": True,
        "next_continue_cli_args": expected_continue_args,
        "next_continue_command": "acos " + " ".join(expected_continue_args),
        "can_blocked_recovery_continue": False,
        "blocked_recovery_continue_cli_args": [],
        "blocked_recovery_continue_command": None,
    }
    assert payload["supervision"] == {
        "can_supervise_continue": True,
        "next_supervise_cli_args": expected_supervise_args,
        "next_supervise_command": "acos " + " ".join(expected_supervise_args),
        "autonomous_until_done": False,
    }
    assert payload["operator_decision"] == {
        "action": "continue",
        "command": "acos " + " ".join(expected_continue_args),
        "resume_action": "continue_next_task",
        "reason": None,
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "blocking_items": [],
        "planning_strategy_change_recommended": False,
    }
    assert payload["operator_summary"] == {
        "operator_action": "continue",
        "operator_command": "acos " + " ".join(expected_continue_args),
        "resume_action": "continue_next_task",
        "can_continue": True,
        "can_blocked_recovery_continue": False,
        "can_supervise_continue": True,
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "planning_strategy_change_recommended": False,
        "command_source": "continuation",
    }


def test_job_status_json_includes_planning_summary_for_plan_only_job(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-plan-only",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
            }
        },
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.PLANNING
    record.outputs["planning_only"] = {
        "complete": True,
        "ready_for_implementation": True,
    }
    record.outputs["task_graph"] = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["core works"],
            ),
        ],
    ).model_dump()
    record.outputs["prd_quality"] = {
        "passed": True,
        "missing": [],
        "warnings": [],
    }
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "small_part_count": 1,
        "small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Build core",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_small_parts": [],
        "acceptance_test_count": 1,
        "acceptance_test_coverage": [
            {
                "acceptance_test_index": 1,
                "acceptance_test": "core works",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_acceptance_tests": [],
        "errors": [],
    }
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-plan-only",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["planning_summary"]["complete"] is True
    assert payload["planning_summary"]["ready_for_implementation"] is True
    assert payload["planning_summary"]["small_part_coverage"] == [
        {
            "small_part_index": 1,
            "small_part": "Build core",
            "task_id": "core",
            "covered": True,
        }
    ]
    assert payload["planning_summary"]["acceptance_test_coverage"] == [
        {
            "acceptance_test_index": 1,
            "acceptance_test": "core works",
            "task_id": "core",
            "covered": True,
        }
    ]
    assert payload["operator_summary"]["operator_action"] == "continue"
    assert payload["operator_summary"]["resume_action"] == "continue_next_task"


def test_job_status_json_includes_blocked_recovery_continuation(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="job-status-json-blocked-recovery",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-json-blocked-recovery",
            "--continue-max-steps",
            "2",
            "--continue-json-summary",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    expected_continue_args = [
        "continue-job",
        "--job-id",
        "job-status-json-blocked-recovery",
        "--max-steps",
        "2",
        "--json-summary",
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["resume"]["can_auto_continue"] is True
    assert payload["continuation"] == {
        "can_continue": True,
        "next_continue_cli_args": expected_continue_args,
        "next_continue_command": "acos " + " ".join(expected_continue_args),
        "can_blocked_recovery_continue": False,
        "blocked_recovery_continue_cli_args": [],
        "blocked_recovery_continue_command": None,
    }
    assert payload["operator_decision"] == {
        "action": "continue",
        "command": "acos " + " ".join(expected_continue_args),
        "resume_action": "recover_repeated_failure",
        "reason": "same_failure_threshold_reached",
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "blocking_items": [],
        "planning_strategy_change_recommended": False,
        "failure_classification": "repeated_test_failure",
        "recommended_recovery": {
            "strategy": "escalated_retry",
            "reason": (
                "same test failure repeated until the autonomous fixer threshold was reached"
            ),
            "failed_task_id": "core",
            "failed_stage": 1,
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "repeated_failure",
                "recovery_strategy": "escalated_retry",
            },
        },
    }
    assert payload["operator_summary"] == {
        "operator_action": "continue",
        "operator_command": "acos " + " ".join(expected_continue_args),
        "resume_action": "recover_repeated_failure",
        "can_continue": True,
        "can_blocked_recovery_continue": False,
        "can_supervise_continue": True,
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "planning_strategy_change_recommended": False,
        "command_source": "continuation",
        "failure_classification": "repeated_test_failure",
        "recommended_recovery": payload["operator_decision"]["recommended_recovery"],
    }


def test_job_status_json_includes_completion_integrity_recovery(
    tmp_path: Path,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="job-status-completion-integrity",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-completion-integrity",
            "--continue-max-steps",
            "2",
            "--continue-json-summary",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    expected_continue_args = [
        "continue-job",
        "--job-id",
        "job-status-completion-integrity",
        "--max-steps",
        "2",
        "--json-summary",
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["completion_integrity"] == record.outputs["completion_integrity"]
    assert payload["resume"]["action"] == "completion_audit_recovery"
    assert payload["continuation"] == {
        "can_continue": True,
        "next_continue_cli_args": expected_continue_args,
        "next_continue_command": "acos " + " ".join(expected_continue_args),
        "can_blocked_recovery_continue": False,
        "blocked_recovery_continue_cli_args": [],
        "blocked_recovery_continue_command": None,
    }
    assert payload["operator_decision"] == {
        "action": "continue",
        "command": "acos " + " ".join(expected_continue_args),
        "resume_action": "completion_audit_recovery",
        "reason": "completion_integrity_failed:missing_test_evidence",
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "blocking_items": [],
        "planning_strategy_change_recommended": False,
        "failure_classification": "completion_integrity_failed",
        "recommended_recovery": {
            "strategy": "completion_audit",
            "reason": (
                "the completion integrity gate found missing work or missing evidence"
            ),
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                "recovery_mode": "completion_integrity",
                "recovery_strategy": "completion_audit",
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "require_stage_test_patches": True,
            },
            "failed_task_id": None,
            "failed_stage": None,
        },
    }


def test_job_status_json_marks_done_job_as_not_supervisable(tmp_path: Path, capsys) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="job-status-json-done",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.DONE
    store.update(record)

    exit_code = main(
        [
            "job-status",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "job-status-json-done",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "done"
    assert payload["supervision"] == {
        "can_supervise_continue": False,
        "next_supervise_cli_args": [],
        "next_supervise_command": None,
        "autonomous_until_done": False,
    }
    assert payload["operator_decision"] == {
        "action": "done",
        "command": None,
        "resume_action": "none",
        "reason": None,
        "requires_explicit_override": False,
        "autonomy_ready": False,
        "blocking_items": [{"type": "task_graph_missing"}],
        "planning_strategy_change_recommended": False,
    }
    assert payload["operator_summary"] == {
        "operator_action": "done",
        "operator_command": None,
        "resume_action": "none",
        "can_continue": False,
        "can_blocked_recovery_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "autonomy_ready": False,
        "planning_strategy_change_recommended": False,
        "command_source": None,
            "blocking_items": [{"type": "task_graph_missing"}],
    }


def test_run_job_rejects_non_positive_autonomous_stage_limit(tmp_path: Path) -> None:
    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        yaml.safe_dump(
            {
                "requester_input": "Build something large.",
                "repo_path": str(tmp_path),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "run-job",
                "--file",
                str(job_file),
                "--max-autonomous-stages",
                "0",
            ]
        )

    assert exc_info.value.code == 2


def test_resume_job_rejects_non_positive_test_timeout() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "resume-job",
                "--job-id",
                "example",
                "--test-timeout-seconds",
                "-1",
            ]
        )

    assert exc_info.value.code == 2


def test_resume_job_can_raise_autonomous_stage_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="stage-limit-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            resumed.status = JobStatus.DONE
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "resume-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "stage-limit-job",
            "--max-autonomous-stages",
            "3",
            "--require-prd-quality",
            "--stage-review",
            "--test-timeout-seconds",
            "900",
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace)
    assert captured["constraints"]["max_autonomous_stages"] == 3
    assert captured["constraints"]["require_prd_quality"] is True
    assert captured["constraints"]["require_task_acceptance_criteria"] is True
    assert captured["constraints"]["require_task_artifacts"] is True
    assert captured["constraints"]["require_completion_integrity"] is True
    assert captured["constraints"]["require_test_evidence"] is True
    assert captured["constraints"]["require_stage_test_patches"] is True
    assert captured["constraints"]["stage_review"] is True
    assert captured["constraints"]["test_timeout_seconds"] == 900
    assert captured["status_before_resume"] == JobStatus.TESTING
    assert captured["last_error_before_resume"] is None
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_resume_job_can_bump_autonomous_stage_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="bump-stage-limit-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            resumed.status = JobStatus.DONE
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "resume-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "bump-stage-limit-job",
            "--bump-stage-limit",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"]["max_autonomous_stages"] == 2
    assert captured["status_before_resume"] == JobStatus.TESTING
    assert captured["last_error_before_resume"] is None
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_resume_job_records_planning_strategy_change_constraints(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="resume-planning-strategy-change-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "prd_quality_gate_failed:acceptance_tests"
    record.outputs["prd_quality_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 1,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 2,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
    ]
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints_before_resume"] = dict(
                resumed.spec.metadata["constraints"]
            )
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "resume-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "resume-planning-strategy-change-job",
        ]
    )

    assert exit_code == 0
    assert captured["constraints_before_resume"].items() >= {
        "planning_repair_strategy_change": True,
        "planning_repair_consecutive_prd_failures": 3,
        "planning_repair_consecutive_task_graph_failures": 0,
        "planning_repair_repeated_prd_missing": "acceptance_tests",
        "planning_repair_repeated_task_graph_error_types": "none",
    }.items()
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_continue_job_auto_bumps_stage_limit_and_resumes(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-stage-limit-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            resumed.status = JobStatus.DONE
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-stage-limit-job",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace)
    assert captured["constraints"]["max_autonomous_stages"] == 2
    assert captured["constraints"]["require_prd_quality"] is True
    assert captured["constraints"]["require_task_acceptance_criteria"] is True
    assert captured["constraints"]["require_task_artifacts"] is True
    assert captured["constraints"]["require_completion_integrity"] is True
    assert captured["constraints"]["require_test_evidence"] is True
    assert captured["constraints"]["require_stage_test_patches"] is True
    assert captured["constraints"]["stage_review"] is True
    assert captured["constraints"]["test_timeout_seconds"] == 1200
    assert captured["status_before_resume"] == JobStatus.TESTING
    assert captured["last_error_before_resume"] is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["started"] is False
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["terminal_reason"] == "done"
    assert payload["can_continue"] is False
    assert payload["status"] == "done"
    assert payload["summary"]["status"] == "done"


def test_continue_job_json_summary_marks_max_steps_reached(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-max-steps-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    summary_file = tmp_path / "summaries" / "continue.json"
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "autonomous_stage_limit_reached"
            resumed.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": limit,
                "completed_stage_count": limit,
            }
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-max-steps-job",
            "--max-steps",
            "1",
            "--json-summary",
            "--summary-file",
            str(summary_file),
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload
    assert payload["status"] == "blocked"
    assert payload["steps_run"] == 1
    assert payload["max_steps"] == 1
    assert payload["step_events"] == [
        {
            "step": 1,
            "action": "raise_stage_limit_or_resume",
            "task_id": None,
            "status_before": "blocked",
            "last_error_before": "autonomous_stage_limit_reached",
            "recovery_strategy": "raise_stage_limit",
            "recovery_mode": "stage_limit",
            "max_autonomous_stages": 2,
            "status_after": "blocked",
            "last_error_after": "autonomous_stage_limit_reached",
        }
    ]
    assert payload["terminal_reason"] == "max_steps_reached"
    assert payload["next_action"] == "raise_stage_limit_or_resume"
    assert payload["can_continue"] is True
    expected_args = [
        "continue-job",
        "--job-id",
        "continue-max-steps-job",
        "--max-steps",
        "1",
        "--json-summary",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["next_continue_cli_args"] == expected_args
    assert payload["next_continue_command"] == "acos " + " ".join(expected_args)


def test_continue_job_retries_planning_quality_repairs_without_forced_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-planning-quality-repair-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"require_prd_quality": True}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "prd_quality_gate_failed:acceptance_tests"
    record.outputs["prd_quality"] = {
        "passed": False,
        "missing": ["acceptance_tests"],
        "warnings": [],
    }
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-planning-quality-repair-job",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace)
    assert captured["status_before_resume"] == JobStatus.BLOCKED
    assert captured["last_error_before_resume"] == "prd_quality_gate_failed:acceptance_tests"
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["step_events"][0]["action"] == "improve_planning_quality"
    assert payload["step_events"][0]["planning_blocking_items"] == [
        {"type": "task_graph_missing"},
        {"type": "prd_quality_not_passed", "missing": ["acceptance_tests"]},
    ]
    assert "forced_recovery" not in payload["step_events"][0]
    assert payload["status"] == "done"


def test_continue_job_records_planning_strategy_change_constraints(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-planning-strategy-change-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "prd_quality_gate_failed:acceptance_tests"
    record.outputs["prd_quality_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 1,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 2,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
    ]
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints_before_resume"] = dict(
                resumed.spec.metadata["constraints"]
            )
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-planning-strategy-change-job",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["constraints_before_resume"].items() >= {
        "planning_repair_strategy_change": True,
        "planning_repair_consecutive_prd_failures": 3,
        "planning_repair_consecutive_task_graph_failures": 0,
        "planning_repair_repeated_prd_missing": "acceptance_tests",
        "planning_repair_repeated_task_graph_error_types": "none",
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["step_events"][0]["action"] == "improve_planning_quality"
    assert payload["step_events"][0]["planning_strategy_change_recommended"] is True


def test_continue_job_can_run_multiple_guarded_steps(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-multi-step-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {"limits": []}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            captured["limits"].append(limit)
            if len(captured["limits"]) == 1:
                resumed.status = JobStatus.BLOCKED
                resumed.last_error = "autonomous_stage_limit_reached"
                resumed.outputs["autonomous_stage_limit"] = {
                    "max_autonomous_stages": limit,
                    "completed_stage_count": limit,
                }
            else:
                resumed.status = JobStatus.DONE
                resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-multi-step-job",
            "--max-steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert captured["limits"] == [2, 3]
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"


def test_autonomous_result_payload_blocks_auto_continue_after_repeated_failure(
    tmp_path: Path,
) -> None:
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="repeated-failure-payload",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.STUCK)
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]

    payload = autonomous_result_payload(
        record,
        steps_run=0,
        max_steps=1,
        started=False,
    )

    assert payload["next_action"] == "recover_repeated_failure"
    assert payload["can_continue"] is True
    expected_continue_args = [
        "continue-job",
        "--job-id",
        "repeated-failure-payload",
        "--max-steps",
        "1",
        "--json-summary",
    ]
    assert payload["next_continue_cli_args"] == expected_continue_args
    assert payload["next_continue_command"] == "acos " + " ".join(expected_continue_args)
    assert payload["can_blocked_recovery_continue"] is False
    assert payload["blocked_recovery_continue_cli_args"] == []
    assert payload["blocked_recovery_continue_command"] is None
    assert payload["summary"]["failure_analysis"]["classification"] == "repeated_test_failure"
    assert payload["operator_decision"] == {
        "action": "continue",
        "command": "acos " + " ".join(expected_continue_args),
        "resume_action": "recover_repeated_failure",
        "reason": "same_failure_threshold_reached",
        "requires_explicit_override": False,
        "autonomy_ready": True,
        "blocking_items": [],
        "planning_strategy_change_recommended": False,
        "failure_classification": "repeated_test_failure",
        "recommended_recovery": {
            "strategy": "escalated_retry",
            "reason": (
                "same test failure repeated until the autonomous fixer threshold was reached"
            ),
            "failed_task_id": "core",
            "failed_stage": 1,
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "repeated_failure",
                "recovery_strategy": "escalated_retry",
            },
        },
    }


def test_apply_recovery_overrides_records_pm_strategy_change_for_diagnosis(
    tmp_path: Path,
) -> None:
    spec = JobSpec(
        job_id="diagnosis-recovery-job",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.STUCK)
    summary = {
        "failure_diagnosis": {
            "classification": "missing_dependency",
            "root_cause": "pydantic-settings and pydantic versions are incompatible",
            "recommended_fix_strategy": "align dependency versions",
            "retry_mode": "targeted_fix",
            "should_retry": False,
            "failure_signature": "ModuleNotFoundError: pydantic._internal._signature",
        },
        "failure_analysis": {
            "recommended_recovery": {
                "strategy": "diagnosis_guided_retry",
                "reason": "the same deterministic failure repeated",
                "failed_task_id": "project-init",
                "failed_stage": 1,
                "constraints": {
                    "recovery_mode": "diagnosed_repeated_failure",
                    "recovery_strategy": "diagnosis_guided_retry",
                    "stage_review": True,
                },
            },
        },
    }

    recovery = apply_recovery_overrides(record, summary)

    constraints = record.spec.metadata["constraints"]
    assert recovery is summary["failure_analysis"]["recommended_recovery"]
    assert constraints["recovery_strategy"] == "diagnosis_guided_retry"
    assert constraints["pm_strategy_change"] is True
    assert constraints["pm_strategy"] == "dependency_alignment_first"
    assert constraints["pm_next_actor"] == "implementer"
    assert "dependency manifests" in constraints["pm_recovery_playbook"]
    assert "dependency import smoke test" in constraints["pm_success_criteria"]
    assert constraints["diagnosis_root_cause"] == (
        "pydantic-settings and pydantic versions are incompatible"
    )
    assert constraints["diagnosis_recommended_fix_strategy"] == (
        "align dependency versions"
    )
    assert constraints["diagnosis_should_retry"] is False
    assert record.outputs["pm_interventions"][0]["strategy"] == (
        "dependency_alignment_first"
    )
    assert record.outputs["pm_interventions"][0]["next_actor"] == "implementer"
    assert "dependency manifests" in record.outputs["pm_interventions"][0]["playbook"]
    assert record.outputs["pm_interventions"][0]["diagnosis"]["classification"] == (
        "missing_dependency"
    )


def test_continue_job_stops_before_repeated_failure_recovery_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="continue-repeated-failure-default",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)

    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-repeated-failure-default",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["status_before_resume"] == JobStatus.STUCK
    assert captured["last_error_before_resume"] == "same_failure_threshold_reached"
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "repeated_failure",
        "recovery_strategy": "escalated_retry",
        "recovery_reason": (
            "same test failure repeated until the autonomous fixer threshold was reached"
        ),
        "recovery_failed_task_id": "core",
        "recovery_failed_stage": 1,
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["status"] == "done"
    assert payload["step_events"][0]["action"] == "recover_repeated_failure"
    assert payload["step_events"][0]["recovery_strategy"] == "escalated_retry"
    assert payload["can_continue"] is False


def test_continue_job_stops_before_completion_integrity_recovery_by_default(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="continue-completion-integrity-default",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }
    store.update(record)

    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-completion-integrity-default",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["status_before_resume"] == JobStatus.TESTING
    assert captured["last_error_before_resume"] is None
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "completion_integrity",
        "recovery_strategy": "completion_audit",
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "recovery_reason": (
            "the completion integrity gate found missing work or missing evidence"
        ),
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["status"] == "done"
    assert payload["step_events"][0]["action"] == "completion_audit_recovery"
    assert payload["step_events"][0]["recovery_strategy"] == "completion_audit"
    assert payload["can_continue"] is False


def test_continue_job_can_force_repeated_failure_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="continue-repeated-failure-forced",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            captured["failure_count_before_resume"] = resumed.failure_count
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-repeated-failure-forced",
            "--allow-repeated-failure-recovery",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["status_before_resume"] == JobStatus.STUCK
    assert captured["last_error_before_resume"] == "same_failure_threshold_reached"
    assert captured["failure_count_before_resume"] == 2
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "repeated_failure",
        "recovery_strategy": "escalated_retry",
        "recovery_reason": (
            "same test failure repeated until the autonomous fixer threshold was reached"
        ),
        "recovery_failed_task_id": "core",
        "recovery_failed_stage": 1,
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["step_events"][0]["action"] == "recover_repeated_failure"
    assert "forced_recovery" not in payload["step_events"][0]
    assert payload["status"] == "done"


def test_continue_job_can_force_completion_integrity_recovery(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="continue-completion-integrity-forced",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["status_before_resume"] = resumed.status
            captured["last_error_before_resume"] = resumed.last_error
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-completion-integrity-forced",
            "--allow-blocked-recovery",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["status_before_resume"] == JobStatus.TESTING
    assert captured["last_error_before_resume"] is None
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "completion_integrity",
        "recovery_strategy": "completion_audit",
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "recovery_reason": (
            "the completion integrity gate found missing work or missing evidence"
        ),
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["step_events"][0]["action"] == "completion_audit_recovery"
    assert "forced_recovery" not in payload["step_events"][0]
    assert payload["step_events"][0]["recovery_strategy"] == "completion_audit"
    assert payload["status"] == "done"


def test_continue_job_can_force_recurring_failure_recovery_with_blocked_alias(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="continue-recurring-failure-forced",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "same_failure_threshold_reached"
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": True},
        },
        {
            "stage": 3,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        },
    ]
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-recurring-failure-forced",
            "--allow-blocked-recovery",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "recurring_failure",
        "recovery_strategy": "split_or_clarify_task",
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "stage_review": True,
        "recovery_reason": "the same task failed again after a previous autonomous recovery",
        "recovery_failed_task_id": "core",
        "recovery_failed_stage": 3,
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["step_events"][0]["action"] == "split_or_clarify_task"
    assert "forced_recovery" not in payload["step_events"][0]
    assert payload["step_events"][0]["recovery_strategy"] == "split_or_clarify_task"
    assert payload["step_events"][0]["recovery_mode"] == "recurring_failure"


def test_continue_job_applies_recovery_recommendation_for_auto_continue_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="continue-implementation-failure",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.FAILED
    record.last_error = "implementation_failed:core"
    record.failure_count = 1
    task_graph = _strict_ready_task_graph()
    _mark_strict_planning_ready(record, task_graph)
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints_before_resume"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-implementation-failure",
            "--json-summary",
        ]
    )

    assert exit_code == 0
    assert captured["constraints_before_resume"].items() >= {
        "recovery_mode": "implementation_failure",
        "recovery_strategy": "replan_current_task",
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "stage_review": True,
        "recovery_reason": "the implementer failed before producing a safe completed change",
        "recovery_failed_task_id": "core",
        "recovery_attempt": 1,
    }.items()
    payload = json.loads(capsys.readouterr().out)
    assert payload["continued"] is True
    assert payload["steps_run"] == 1
    assert payload["step_events"][0]["action"] == "continue_next_task"
    assert payload["step_events"][0]["recovery_strategy"] == "replan_current_task"
    assert payload["step_events"][0]["recovery_mode"] == "implementation_failure"
    assert "forced_recovery" not in payload["step_events"][0]


def test_supervise_job_runs_cycles_until_done_and_writes_summaries(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-stage-limit-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    final_summary_file = tmp_path / "summaries" / "final.json"
    cycle_summary_dir = tmp_path / "summaries" / "cycles"
    captured: dict[str, object] = {"limits": []}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            captured["limits"].append(limit)
            if len(captured["limits"]) < 3:
                resumed.status = JobStatus.BLOCKED
                resumed.last_error = "autonomous_stage_limit_reached"
                resumed.outputs["autonomous_stage_limit"] = {
                    "max_autonomous_stages": limit,
                    "completed_stage_count": limit,
                }
            else:
                resumed.status = JobStatus.DONE
                resumed.last_error = None
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["workspace_root"] = str(workspace_root)
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-stage-limit-job",
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--summary-file",
            str(final_summary_file),
            "--summary-dir",
            str(cycle_summary_dir),
        ]
    )

    assert exit_code == 0
    assert captured["workspace_root"] == str(workspace)
    assert captured["limits"] == [2, 3, 4]
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(final_summary_file.read_text(encoding="utf-8")) == payload
    assert payload["status"] == "done"
    assert payload["terminal_reason"] == "done"
    assert payload["cycles_run"] == 3
    assert payload["max_cycles"] == 3
    assert payload["steps_per_cycle"] == 1
    assert payload["steps_run"] == 3
    assert payload["operator_decision"] == {
        "action": "done",
        "command": None,
        "resume_action": "none",
        "reason": None,
        "requires_explicit_override": False,
        "autonomy_ready": False,
        "blocking_items": [
            {"type": "task_graph_missing"},
            {"type": "prd_quality_not_passed", "missing": []},
        ],
        "planning_strategy_change_recommended": False,
    }
    assert [event["step"] for event in payload["step_events"]] == [1, 2, 3]
    assert [event["cycle"] for event in payload["step_events"]] == [1, 2, 3]
    assert [event["cycle_step"] for event in payload["step_events"]] == [1, 1, 1]
    assert len(payload["cycle_summaries"]) == 3
    for cycle in range(1, 4):
        cycle_payload = json.loads(
            (cycle_summary_dir / f"cycle-{cycle:03d}.json").read_text(encoding="utf-8")
        )
        assert cycle_payload == payload["cycle_summaries"][cycle - 1]
        assert cycle_payload["cycle"] == cycle


def test_supervise_job_stops_after_repeated_stalled_cycles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-stalled-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            resumed.status = JobStatus.TESTING
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-stalled-job",
            "--max-cycles",
            "5",
            "--steps-per-cycle",
            "1",
            "--max-stalled-cycles",
            "1",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "testing"
    assert payload["done"] is False
    assert payload["terminal_reason"] == "stalled"
    assert payload["can_continue"] is False
    assert payload["next_continue_cli_args"] == []
    assert payload["next_continue_command"] is None
    assert payload["cycles_run"] == 2
    assert payload["steps_run"] == 2
    assert payload["stalled"] is True
    assert payload["stalled_cycle_count"] == 1
    assert payload["max_stalled_cycles"] == 1
    assert [cycle["stalled_cycle_count"] for cycle in payload["cycle_summaries"]] == [0, 1]
    first_marker = payload["cycle_summaries"][0]["progress_marker"]
    second_marker = payload["cycle_summaries"][1]["progress_marker"]
    assert first_marker == second_marker
    assert second_marker["status"] == "testing"
    assert second_marker["resume_action"] == "continue_next_task"
    assert second_marker["resume_task_id"] == "core"
    stopped_cycle = payload["cycle_summaries"][1]
    assert stopped_cycle["terminal_reason"] == "stalled"
    assert stopped_cycle["can_continue"] is False
    assert stopped_cycle["stall_analysis"] == {
        "stalled": True,
        "stalled_cycle_count": 1,
        "max_stalled_cycles": 1,
        "repeated_cycle_count": 2,
        "repeated_progress_marker": second_marker,
        "reason": "same_progress_marker_repeated",
    }
    assert stopped_cycle["pm_decision"]["action"] == "change_strategy"
    assert stopped_cycle["pm_decision"]["strategy"] == "split_or_simplify_next_task"
    assert stopped_cycle["pm_decision"]["can_apply_automatically"] is False
    assert stopped_cycle["pm_decision"]["applied"] is False
    assert stopped_cycle["operator_decision"]["action"] == "supervise"
    assert stopped_cycle["operator_decision"]["reason"] is None
    assert payload["stall_analysis"] == {
        "stalled": True,
        "stalled_cycle_count": 1,
        "max_stalled_cycles": 1,
        "repeated_cycle_count": 2,
        "repeated_progress_marker": second_marker,
        "reason": "same_progress_marker_repeated",
    }
    assert payload["operator_decision"]["action"] == "supervise"
    assert payload["operator_decision"]["requires_explicit_override"] is False
    assert payload["operator_decision"]["reason"] is None
    assert "stall_analysis" not in payload["operator_decision"]
    assert payload["pm_decision"] == stopped_cycle["pm_decision"]
    assert payload["pm_interventions"] == []
    assert payload["stop_summary"] == {
        "terminal_reason": "stalled",
        "operator_action": "supervise",
        "operator_command": payload["operator_decision"]["command"],
        "resume_action": "continue_next_task",
        "can_continue": False,
        "can_supervise_continue": False,
        "requires_explicit_override": False,
        "stall_analysis": payload["stall_analysis"],
        "pm_decision": payload["pm_decision"],
        "pm_intervention_count": 0,
        "provider_event_count": 0,
        "last_provider_event": None,
    }


def test_supervise_job_pm_recovery_changes_strategy_after_stall(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-pm-recovery-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    _mark_strict_planning_ready(record, _strict_ready_task_graph())
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            resumed.status = JobStatus.TESTING
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-pm-recovery-job",
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--max-stalled-cycles",
            "1",
            "--pm-stall-recovery",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_reason"] == "max_steps_reached"
    assert payload["stalled"] is False
    assert payload["pm_stall_recovery"] is True
    assert payload["pm_stall_recoveries"] == 1
    assert len(payload["cycle_summaries"]) == 3
    recovered_cycle = payload["cycle_summaries"][1]
    assert recovered_cycle["terminal_reason"] == "pm_strategy_change"
    assert recovered_cycle["pm_recovery_applied"] is True
    assert recovered_cycle["pm_decision"]["applied"] is True
    assert recovered_cycle["pm_decision"]["strategy"] == "split_or_simplify_next_task"
    assert payload["pm_decision"] == recovered_cycle["pm_decision"]
    assert payload["pm_interventions"] == [recovered_cycle["pm_decision"]]

    updated = FileJobStore(jobs_dir).get("supervise-pm-recovery-job")
    constraints = updated.spec.metadata["constraints"]
    assert constraints["pm_stall_recovery"] is True
    assert constraints["pm_strategy_change"] is True
    assert constraints["pm_strategy"] == "split_or_simplify_next_task"
    assert constraints["recovery_strategy"] == "split_or_clarify_task"
    assert constraints["pm_focus_task_id"] == "core"


def test_supervise_job_auto_resumes_diagnosed_failure_with_pm_strategy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_graph = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(id="project-init", title="Init", description="Build init", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="supervise-diagnosis-auto-recovery-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.STUCK
    record.last_error = "diagnosed_repeated_failure:missing_dependency"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
        }
    ]
    record.outputs["failure_diagnosis"] = {
        "classification": "missing_dependency",
        "root_cause": "pydantic-settings and pydantic versions are incompatible",
        "failed_files": ["backend/app/config.py"],
        "failed_tests": ["backend/tests"],
        "recommended_fix_strategy": "align dependency versions",
        "confidence": 0.95,
        "should_retry": False,
        "retry_mode": "targeted_fix",
        "failure_signature": "ModuleNotFoundError: pydantic._internal._signature",
    }
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            constraints = resumed.spec.metadata["constraints"]
            assert constraints["pm_strategy"] == "dependency_alignment_first"
            assert constraints["pm_next_actor"] == "implementer"
            assert "dependency manifests" in constraints["pm_recovery_playbook"]
            resumed.status = JobStatus.DONE
            resumed.last_error = None
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-diagnosis-auto-recovery-job",
            "--max-cycles",
            "2",
            "--steps-per-cycle",
            "1",
            "--pm-stall-recovery",
        ]
    )

    assert exit_code == 0
    assert captured["resume_count"] == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["done"] is True
    assert payload["steps_run"] == 1
    assert "forced_recovery" not in payload["step_events"][0]
    assert payload["step_events"][0]["recovery_strategy"] == "diagnosis_guided_retry"
    assert payload["pm_interventions"][0]["strategy"] == "dependency_alignment_first"


def test_supervise_job_treats_planning_quality_attempts_as_progress(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-planning-progress-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"require_prd_quality": True}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "prd_quality_gate_failed:acceptance_tests"
    record.outputs["prd_quality"] = {
        "passed": False,
        "missing": ["acceptance_tests"],
        "warnings": [],
    }
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            attempts = resumed.outputs.setdefault("prd_quality_attempts", [])
            attempts.append(
                {
                    "attempt": captured["resume_count"],
                    "action": "refine",
                    "passed": False,
                    "missing": ["acceptance_tests"],
                    "warnings": [],
                }
            )
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "prd_quality_gate_failed:acceptance_tests"
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-planning-progress-job",
            "--max-cycles",
            "2",
            "--steps-per-cycle",
            "1",
            "--max-stalled-cycles",
            "1",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_reason"] == "max_steps_reached"
    assert payload["stalled"] is False
    assert payload["stalled_cycle_count"] == 0
    assert [cycle["stalled_cycle_count"] for cycle in payload["cycle_summaries"]] == [0, 0]
    assert payload["stall_analysis"] == {
        "stalled": False,
        "stalled_cycle_count": 0,
        "max_stalled_cycles": 1,
        "repeated_cycle_count": 0,
        "repeated_progress_marker": None,
        "reason": None,
    }
    assert [
        cycle["progress_marker"]["prd_quality_attempt_count"]
        for cycle in payload["cycle_summaries"]
    ] == [1, 2]
    assert payload["cycle_summaries"][1]["summary"]["planning_quality"][
        "prd_quality_attempt_count"
    ] == 2


def test_supervise_job_stops_after_runtime_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-runtime-limit-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}
    summary_file = tmp_path / "summaries" / "runtime-final.json"
    times = iter([100.0, 101.5])

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            assert resumed.spec.metadata["constraints"]["model_timeout_seconds"] == 1.0
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "autonomous_stage_limit_reached"
            resumed.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": limit,
                "completed_stage_count": limit,
            }
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.monotonic", lambda: next(times))
    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": True,
            "status": "ok",
            "probe_timeout_seconds": timeout_seconds,
        },
    )

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-runtime-limit-job",
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--max-runtime-seconds",
            "1",
            "--summary-file",
            str(summary_file),
            "--preflight-provider",
            "local_ornith",
            "--preflight-timeout",
            "9",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["terminal_reason"] == "runtime_limit"
    assert payload["can_continue"] is False
    assert payload["next_continue_cli_args"] == []
    assert payload["next_continue_command"] is None
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "supervise-runtime-limit-job",
        "--max-cycles",
        "3",
        "--steps-per-cycle",
        "1",
        "--max-stalled-cycles",
        "3",
        "--max-runtime-seconds",
        "1.0",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
        "--summary-file",
        str(summary_file),
        "--preflight-provider",
        "local_ornith",
        "--preflight-timeout",
        "9.0",
    ]
    assert payload["can_supervise_continue"] is True
    assert payload["next_supervise_cli_args"] == expected_supervise_args
    assert payload["next_supervise_command"] == "acos " + " ".join(expected_supervise_args)
    assert payload["operator_decision"].items() >= {
        "action": "supervise",
        "command": "acos " + " ".join(expected_supervise_args),
        "resume_action": "raise_stage_limit_or_resume",
        "reason": "autonomous_stage_limit_reached",
        "requires_explicit_override": False,
        "autonomy_ready": False,
        "planning_strategy_change_recommended": False,
    }.items()
    assert payload["operator_decision"]["blocking_items"] == [
        {"type": "task_graph_missing"},
        {"type": "prd_quality_not_passed", "missing": []},
    ]
    assert payload["operator_decision"]["runtime_analysis"] == {
        "runtime_limited": True,
        "elapsed_seconds": 1.5,
        "max_runtime_seconds": 1.0,
        "reason": "runtime_limit_reached",
    }
    assert payload["operator_decision"]["failure_classification"] == (
        "autonomous_stage_limit_reached"
    )
    assert payload["operator_decision"]["recommended_recovery"] == {
        "strategy": "raise_stage_limit",
        "reason": "autonomous stage limit was reached and can be bumped",
        "failed_task_id": None,
        "failed_stage": None,
        "preserve_failure_counts_for_model_escalation": False,
        "constraints": {
            "recovery_mode": "stage_limit",
            "recovery_strategy": "raise_stage_limit",
        },
    }
    assert payload["cycles_run"] == 1
    assert payload["steps_run"] == 1
    assert payload["runtime_limited"] is True
    assert payload["elapsed_seconds"] == 1.5
    assert payload["max_runtime_seconds"] == 1.0
    assert payload["runtime_analysis"] == {
        "runtime_limited": True,
        "elapsed_seconds": 1.5,
        "max_runtime_seconds": 1.0,
        "reason": "runtime_limit_reached",
    }
    assert payload["cycle_summaries"][0]["elapsed_seconds"] == 1.5
    runtime_cycle = payload["cycle_summaries"][0]
    assert runtime_cycle["terminal_reason"] == "runtime_limit"
    assert runtime_cycle["can_continue"] is False
    assert runtime_cycle["runtime_limited"] is True
    assert runtime_cycle["runtime_analysis"] == {
        "runtime_limited": True,
        "elapsed_seconds": 1.5,
        "max_runtime_seconds": 1.0,
        "reason": "runtime_limit_reached",
    }
    assert runtime_cycle["operator_decision"]["action"] == "supervise"
    assert runtime_cycle["operator_decision"]["reason"] == "autonomous_stage_limit_reached"
    assert runtime_cycle["operator_decision"]["runtime_analysis"] == runtime_cycle[
        "runtime_analysis"
    ]
    assert payload["stop_summary"] == {
        "terminal_reason": "runtime_limit",
        "operator_action": "supervise",
        "operator_command": "acos " + " ".join(expected_supervise_args),
        "resume_action": "raise_stage_limit_or_resume",
        "can_continue": False,
        "can_supervise_continue": True,
        "requires_explicit_override": False,
        "runtime_analysis": payload["runtime_analysis"],
        "provider_preflight": payload["provider_preflight"],
        "provider_event_count": len(payload["provider_events"]),
        "last_provider_event": payload["provider_events"][-1],
    }


def test_supervise_job_caps_model_timeout_for_long_runtime_budget(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-long-runtime-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.time", lambda: 3000.0)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-long-runtime-job",
            "--max-cycles",
            "1",
            "--steps-per-cycle",
            "1",
            "--max-runtime-seconds",
            "900",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"]["model_timeout_seconds"] == (
        SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS
    )
    assert captured["constraints"]["model_timeout_deadline_epoch"] == 3900.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_reason"] == "done"


def test_supervise_job_without_runtime_clears_stale_model_timeout_deadline(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-stale-deadline-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={
            "constraints": {
                "max_autonomous_stages": 1,
                "model_timeout_deadline_epoch": 1.0,
            }
        },
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-stale-deadline-job",
            "--max-cycles",
            "1",
            "--steps-per-cycle",
            "1",
        ]
    )

    assert exit_code == 0
    assert "model_timeout_deadline_epoch" not in captured["constraints"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_reason"] == "done"


def test_supervise_job_preserves_autonomous_until_done_in_raw_json_and_next_args(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="supervise-autonomous-until-done-job",
        request_text="Build continuously.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}
    summary_file = tmp_path / "summaries" / "autonomous-until-done.json"
    times = iter([100.0, 101.5])

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "autonomous_stage_limit_reached"
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr("apps.cli.monotonic", lambda: next(times))

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-autonomous-until-done-job",
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--max-runtime-seconds",
            "1",
            "--summary-file",
            str(summary_file),
            "--autonomous-until-done",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["autonomous_until_done"] is True
    assert "--autonomous-until-done" in payload["next_supervise_cli_args"]
    assert "--autonomous-until-done" in payload["next_supervise_command"]
    assert json.loads(summary_file.read_text(encoding="utf-8")) == payload


def test_supervise_job_suggests_supervise_command_after_max_cycles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-max-cycles-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "autonomous_stage_limit_reached"
            resumed.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": limit,
                "completed_stage_count": limit,
            }
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-max-cycles-job",
            "--max-cycles",
            "1",
            "--steps-per-cycle",
            "1",
            "--max-stalled-cycles",
            "2",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_reason"] == "max_steps_reached"
    assert payload["can_supervise_continue"] is True
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "supervise-max-cycles-job",
        "--max-cycles",
        "1",
        "--steps-per-cycle",
        "1",
        "--max-stalled-cycles",
        "2",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
    ]
    assert payload["next_supervise_cli_args"] == expected_supervise_args
    assert payload["next_supervise_command"] == "acos " + " ".join(expected_supervise_args)


def test_supervise_job_preflight_stops_before_resume_when_provider_unhealthy(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="supervise-provider-down-job",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    FileJobStore(jobs_dir).create(spec)
    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: {
            "provider": provider_name,
            "healthy": False,
            "status": "down",
        },
    )

    def fail_build_default_runner(*args, **kwargs):
        raise AssertionError("supervise-job should not resume when preflight fails")

    monkeypatch.setattr("apps.cli.build_default_runner", fail_build_default_runner)

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-provider-down-job",
            "--preflight-provider",
            "local_ornith",
        ]
    )

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "supervise-provider-down-job"
    assert payload["terminal_reason"] == "provider_unhealthy"
    assert payload["provider_preflight"]["provider"] == "local_ornith"
    assert payload["provider_events"] == [
        {
            "cycle": None,
            "phase": "pre_start",
            "healthy": False,
            "terminal": True,
            "provider_preflight": payload["provider_preflight"],
        }
    ]
    assert payload["can_supervise_continue"] is True
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "supervise-provider-down-job",
        "--max-cycles",
        "10",
        "--steps-per-cycle",
        "1",
        "--max-stalled-cycles",
        "3",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
        "--preflight-provider",
        "local_ornith",
        "--preflight-timeout",
        "5.0",
    ]
    assert payload["next_supervise_cli_args"] == expected_supervise_args
    assert payload["next_supervise_command"] == "acos " + " ".join(expected_supervise_args)
    assert payload["operator_decision"] == {
        "action": "supervise",
        "command": "acos " + " ".join(expected_supervise_args),
        "resume_action": "check_provider",
        "reason": "provider_unhealthy",
        "requires_explicit_override": False,
        "autonomy_ready": None,
        "blocking_items": [
            {"type": "provider_unhealthy", "provider": "local_ornith"}
        ],
        "planning_strategy_change_recommended": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
    }
    assert payload["stop_summary"] == {
        "terminal_reason": "provider_unhealthy",
        "operator_action": "supervise",
        "operator_command": "acos " + " ".join(expected_supervise_args),
        "resume_action": "check_provider",
        "can_continue": False,
        "can_supervise_continue": True,
        "requires_explicit_override": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
        "provider_event_count": 1,
        "last_provider_event": payload["provider_events"][-1],
    }


def test_supervise_job_checks_provider_before_each_cycle(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="supervise-provider-drops-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 1}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.BLOCKED
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}
    preflight_results = iter(
        [
            {"provider": "local_ornith", "healthy": True, "status": "ok", "attempt": 1},
            {"provider": "local_ornith", "healthy": True, "status": "ok", "attempt": 2},
            {"provider": "local_ornith", "healthy": False, "status": "down", "attempt": 3},
        ]
    )

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            limit = resumed.spec.metadata["constraints"]["max_autonomous_stages"]
            resumed.status = JobStatus.BLOCKED
            resumed.last_error = "autonomous_stage_limit_reached"
            resumed.outputs["autonomous_stage_limit"] = {
                "max_autonomous_stages": limit,
                "completed_stage_count": limit,
            }
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)
    monkeypatch.setattr(
        "apps.cli.probe_provider",
        lambda registry, provider_name, timeout_seconds=5.0: next(preflight_results),
    )

    exit_code = main(
        [
            "supervise-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "supervise-provider-drops-job",
            "--max-cycles",
            "3",
            "--steps-per-cycle",
            "1",
            "--preflight-provider",
            "local_ornith",
        ]
    )

    assert exit_code == 1
    assert captured["resume_count"] == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert payload["terminal_reason"] == "provider_unhealthy"
    assert payload["provider_unhealthy"] is True
    assert payload["provider_preflight"] == {
        "provider": "local_ornith",
        "healthy": False,
        "status": "down",
        "attempt": 3,
    }
    assert payload["cycles_run"] == 1
    assert len(payload["cycle_summaries"]) == 1
    assert payload["cycle_summaries"][0]["provider_preflight"]["attempt"] == 2
    assert payload["provider_events"] == [
        {
            "cycle": None,
            "phase": "pre_start",
            "healthy": True,
            "terminal": False,
            "provider_preflight": {
                "provider": "local_ornith",
                "healthy": True,
                "status": "ok",
                "attempt": 1,
            },
        },
        {
            "cycle": 1,
            "phase": "pre_cycle",
            "healthy": True,
            "terminal": False,
            "provider_preflight": {
                "provider": "local_ornith",
                "healthy": True,
                "status": "ok",
                "attempt": 2,
            },
        },
        {
            "cycle": 2,
            "phase": "pre_cycle",
            "healthy": False,
            "terminal": True,
            "provider_preflight": payload["provider_preflight"],
        },
    ]
    assert payload["can_supervise_continue"] is True
    expected_supervise_args = [
        "supervise-job",
        "--job-id",
        "supervise-provider-drops-job",
        "--max-cycles",
        "3",
        "--steps-per-cycle",
        "1",
        "--max-stalled-cycles",
        "3",
        "--config-dir",
        str(config_dir()),
        "--jobs-dir",
        str(jobs_dir),
        "--preflight-provider",
        "local_ornith",
        "--preflight-timeout",
        "5.0",
    ]
    assert payload["next_supervise_cli_args"] == expected_supervise_args
    assert payload["next_supervise_command"] == "acos " + " ".join(expected_supervise_args)
    assert payload["operator_decision"] == {
        "action": "supervise",
        "command": "acos " + " ".join(expected_supervise_args),
        "resume_action": "check_provider",
        "reason": "provider_unhealthy",
        "requires_explicit_override": False,
        "autonomy_ready": None,
        "blocking_items": [
            {"type": "provider_unhealthy", "provider": "local_ornith"}
        ],
        "planning_strategy_change_recommended": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": payload["provider_preflight"],
    }


def test_continue_job_does_not_resume_completed_job(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="continue-done-job",
        request_text="Build something useful.",
        repo_path=str(tmp_path),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.DONE
    store.update(record)

    def fail_build_default_runner(*args, **kwargs):
        raise AssertionError("continue-job should not build a runner for completed jobs")

    monkeypatch.setattr("apps.cli.build_default_runner", fail_build_default_runner)

    exit_code = main(
        [
            "continue-job",
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "continue-done-job",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["continued"] is False
    assert payload["summary"]["status"] == "done"


def test_resume_job_large_autonomous_preserves_existing_stage_limit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from packages.orchestrator.job_store import FileJobStore

    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="large-autonomous-resume-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
        metadata={"constraints": {"max_autonomous_stages": 4}},
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    store.update(record)
    captured: dict[str, object] = {}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            resumed = captured["store"].get(job_id)
            captured["constraints"] = dict(resumed.spec.metadata["constraints"])
            resumed.status = JobStatus.DONE
            return resumed

    def fake_build_default_runner(config_dir: str | Path, workspace_root: str | Path, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr("apps.cli.build_default_runner", fake_build_default_runner)

    exit_code = main(
        [
            "resume-job",
            "--config-dir",
            str(config_dir()),
            "--jobs-dir",
            str(jobs_dir),
            "--job-id",
            "large-autonomous-resume-job",
            "--large-autonomous",
        ]
    )

    assert exit_code == 0
    assert captured["constraints"] == {
        "max_autonomous_stages": 4,
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "done"
