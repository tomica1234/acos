from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from apps import cli as cli_main
from apps.api import main as api_main
from packages.orchestrator.job_store import FileJobStore
from packages.schemas.audit import AuditEvent
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus
from packages.schemas.tasks import PlannedTask, TaskGraph


def test_supervised_api_stops_before_runner_when_provider_unhealthy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "maybe_probe_provider",
        lambda config_dir, provider_name, timeout_seconds: {
            "provider": provider_name,
            "healthy": False,
            "status": "down",
            "probe_timeout_seconds": timeout_seconds,
        },
    )

    def fail_build_default_runner(*args, **kwargs):
        raise AssertionError("provider preflight should stop before runner creation")

    monkeypatch.setattr(api_main, "build_default_runner", fail_build_default_runner)

    client = TestClient(api_main.create_app())
    response = client.post(
        "/jobs/supervised",
        json={
            "request_text": "Build something useful.",
            "repo_path": str(tmp_path / "workspace"),
            "jobs_dir": str(tmp_path / "jobs"),
            "preflight_provider": "local_ornith",
            "preflight_timeout": 0.25,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["terminal_reason"] == "provider_unhealthy"
    assert payload["operator_decision"]["action"] == "inspect"
    assert payload["provider_preflight"] == {
        "provider": "local_ornith",
        "healthy": False,
        "status": "down",
        "probe_timeout_seconds": 0.25,
    }


def test_supervised_api_can_plan_first_then_supervise(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"plan_count": 0, "resume_count": 0}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            raise AssertionError("plan_first should not call run_job")

        def plan_job(self, spec: JobSpec) -> JobRecord:
            captured["plan_count"] += 1
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
                "require_acceptance_criteria": True,
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

    def fake_build_default_runner(config_dir, workspace_root, store=None):
        captured["store"] = store
        captured["workspace_root"] = str(workspace_root)
        return DummyRunner(), None

    monkeypatch.setattr(api_main, "maybe_probe_provider", lambda **kwargs: None)
    monkeypatch.setattr(api_main, "build_default_runner", fake_build_default_runner)
    monkeypatch.setattr(cli_main, "build_default_runner", fake_build_default_runner)

    client = TestClient(api_main.create_app())
    response = client.post(
        "/jobs/supervised",
        json={
            "request_text": "Build something useful.",
            "repo_path": str(tmp_path / "workspace"),
            "job_id": "api-plan-first-job",
            "jobs_dir": str(tmp_path / "jobs"),
            "plan_first": True,
            "max_cycles": 2,
            "steps_per_cycle": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["plan_count"] == 1
    assert captured["resume_count"] == 1
    assert captured["workspace_root"] == str((tmp_path / "workspace").resolve())
    assert payload["job_id"] == "api-plan-first-job"
    assert payload["planned_first"] is True
    assert payload["planning_complete"] is True
    assert payload["status"] == "done"
    assert payload["terminal_reason"] == "done"


def test_job_progress_api_reads_requested_jobs_dir(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    spec = JobSpec(
        job_id="progress-api-job",
        request_text="Build something useful.",
        repo_path=str(tmp_path / "workspace"),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
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
            )
        ],
    ).model_dump()
    record.outputs["planning_only"] = {
        "complete": True,
        "ready_for_implementation": True,
    }
    record.outputs["pm_interventions"] = [
        {
            "action": "change_strategy",
            "strategy": "split_or_simplify_next_task",
            "applied": True,
        }
    ]
    record.audit_events.append(
        AuditEvent(
            timestamp="2026-07-04T00:00:00Z",
            event_type="model_call",
            role="pm",
            action="ornith_35b_q4",
            status="success",
        )
    )
    store.update(record)

    client = TestClient(api_main.create_app())
    response = client.get(
        "/jobs/progress-api-job/progress",
        params={"jobs_dir": str(jobs_dir)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "progress-api-job"
    assert payload["status"] == "planning"
    assert payload["history"] == ["submitted"]
    assert payload["outputs_keys"] == ["planning_only", "pm_interventions", "task_graph"]
    assert payload["pm_interventions"] == [
        {
            "action": "change_strategy",
            "strategy": "split_or_simplify_next_task",
            "applied": True,
        }
    ]
    assert payload["summary"]["pending_task_count"] == 1
    assert payload["summary"]["next_task"]["id"] == "core"
    assert payload["recent_audit_events"] == [
        {
            "timestamp": "2026-07-04T00:00:00Z",
            "event_type": "model_call",
            "role": "pm",
            "action": "ornith_35b_q4",
            "status": "success",
            "input_hash": None,
            "output_hash": None,
            "metadata": {},
        }
    ]


def test_supervise_existing_job_api_continues_saved_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs_dir = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = JobSpec(
        job_id="api-existing-supervise-job",
        request_text="Build something useful.",
        repo_path=str(workspace),
    )
    store = FileJobStore(jobs_dir)
    record = store.create(spec)
    record.status = JobStatus.TESTING
    record.outputs["task_graph"] = TaskGraph(
        goal="Build it",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["core works"],
            )
        ],
    ).model_dump()
    store.update(record)
    captured: dict[str, object] = {"resume_count": 0}

    class DummyRunner:
        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            resumed = captured["store"].get(job_id)
            resumed.status = JobStatus.DONE
            captured["store"].update(resumed)
            return resumed

    def fake_build_default_runner(config_dir, workspace_root, store=None):
        captured["store"] = store
        captured["workspace_root"] = str(workspace_root)
        return DummyRunner(), None

    monkeypatch.setattr(cli_main, "build_default_runner", fake_build_default_runner)

    client = TestClient(api_main.create_app())
    response = client.post(
        "/jobs/api-existing-supervise-job/supervise",
        json={
            "jobs_dir": str(jobs_dir),
            "max_cycles": 2,
            "steps_per_cycle": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert captured["resume_count"] == 1
    assert captured["workspace_root"] == str(workspace)
    assert payload["job_id"] == "api-existing-supervise-job"
    assert payload["status"] == "done"
    assert payload["terminal_reason"] == "done"
    assert payload["resumed_existing_job"] is True


def test_background_supervised_job_runs_until_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"run_count": 0, "resume_count": 0}

    class DummyRunner:
        def run_job(self, spec: JobSpec) -> JobRecord:
            captured["run_count"] += 1
            record = captured["store"].create(spec)
            record.status = JobStatus.TESTING
            record.outputs["task_graph"] = TaskGraph(
                goal="Build it",
                tasks=[
                    PlannedTask(
                        id="core",
                        title="Core",
                        description="Build core",
                        role="implementer",
                        acceptance_criteria=["core works"],
                    )
                ],
            ).model_dump()
            captured["store"].update(record)
            return record

        def resume_job(self, job_id: str) -> JobRecord:
            captured["resume_count"] += 1
            record = captured["store"].get(job_id)
            record.status = JobStatus.DONE
            captured["store"].update(record)
            return record

    def fake_build_default_runner(config_dir, workspace_root, store=None):
        captured["store"] = store
        return DummyRunner(), None

    monkeypatch.setattr(api_main, "build_default_runner", fake_build_default_runner)
    monkeypatch.setattr(cli_main, "build_default_runner", fake_build_default_runner)

    client = TestClient(api_main.create_app())
    response = client.post(
        "/jobs/supervised/background",
        json={
            "request_text": "Build something useful.",
            "repo_path": str(tmp_path / "workspace"),
            "job_id": "api-background-job",
            "jobs_dir": str(tmp_path / "jobs"),
            "plan_first": False,
            "max_cycles": 3,
            "steps_per_cycle": 1,
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    payload: dict[str, object] = {}
    for _ in range(20):
        status_response = client.get(f"/background-runs/{run_id}")
        assert status_response.status_code == 200
        payload = status_response.json()
        if payload["status"] == "done":
            break
        time.sleep(0.05)

    assert payload["status"] == "done"
    assert payload["job_id"] == "api-background-job"
    assert payload["batches_run"] == 1
    assert payload["last_result"]["terminal_reason"] == "done"
    assert captured["run_count"] == 1
    assert captured["resume_count"] == 1


def test_background_run_stop_marks_stop_requested(tmp_path: Path) -> None:
    client = TestClient(api_main.create_app())
    response = client.post(
        "/jobs/supervised/background",
        json={
            "request_text": "Build something useful.",
            "repo_path": str(tmp_path / "workspace"),
            "job_id": "api-background-stop-job",
            "jobs_dir": str(tmp_path / "jobs"),
            "preflight_provider": "missing-provider",
        },
    )

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    stop_response = client.post(f"/background-runs/{run_id}/stop")

    assert stop_response.status_code == 200
    payload = stop_response.json()
    assert payload["stop_requested"] is True
    assert payload["status"] in {"queued", "running", "stopping", "paused", "error"}
