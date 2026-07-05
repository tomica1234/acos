"""FastAPI app for ACOS."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apps.cli import (
    apply_constraint_overrides,
    autonomous_result_payload,
    build_job_spec_from_request,
    maybe_probe_provider,
    planning_result_payload,
    provider_unhealthy_payload,
    stop_summary_payload,
    supervise_persisted_job,
    _provider_preflight_event,
)
from packages.llm.registry import ModelRegistry
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.job_store import FileJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.progress import summarize_job_progress
from packages.schemas.models import JobStatus
from packages.schemas.jobs import JobRecord, JobSpec


class SubmitJobRequest(BaseModel):
    request_text: str
    repo_path: str
    target_branch: str = "acos/default"
    metadata: dict[str, object] = Field(default_factory=dict)


class SupervisedJobRequest(BaseModel):
    request_text: str
    repo_path: str
    workspace_root: str | None = None
    target_branch: str = "acos/default"
    job_id: str | None = None
    title: str | None = None
    jobs_dir: str = ".acos/jobs"
    max_cycles: int = 10
    steps_per_cycle: int = 1
    max_stalled_cycles: int = 3
    max_runtime_seconds: float | None = None
    summary_file: str | None = None
    summary_dir: str | None = None
    plan_first: bool = True
    preflight_provider: str | None = None
    preflight_timeout: float = 5.0
    max_autonomous_stages: int | None = None
    require_prd_quality: bool = True
    stage_review: bool = True
    test_timeout_seconds: int | None = None
    allow_blocked_recovery: bool = False
    pm_stall_recovery: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SuperviseExistingJobRequest(BaseModel):
    jobs_dir: str = ".acos/jobs"
    workspace: str | None = None
    max_cycles: int = 10
    steps_per_cycle: int = 1
    max_stalled_cycles: int = 3
    max_runtime_seconds: float | None = None
    summary_file: str | None = None
    summary_dir: str | None = None
    preflight_provider: str | None = None
    preflight_timeout: float = 5.0
    max_autonomous_stages: int | None = None
    require_prd_quality: bool = True
    stage_review: bool = True
    test_timeout_seconds: int | None = None
    allow_blocked_recovery: bool = False
    pm_stall_recovery: bool = True


class BackgroundSupervisedJobRequest(SupervisedJobRequest):
    max_batches: int | None = None


class ProviderProbeRequest(BaseModel):
    provider: str = "local_ornith"
    timeout: float = 5.0


def create_app(job_runner: JobRunner | None = None) -> FastAPI:
    app = FastAPI(title="ACOS API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.job_runner = job_runner
    app.state.background_runs = {}
    app.state.background_run_lock = threading.Lock()
    app.state.job_runner_factory = lambda: build_default_runner(
        config_dir=Path(__file__).resolve().parents[2] / "configs",
        workspace_root=Path("."),
    )[0]
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    def get_runner() -> JobRunner:
        runner = app.state.job_runner
        if runner is None:
            runner = app.state.job_runner_factory()
            app.state.job_runner = runner
        return runner

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _update_background_run(run_id: str, **updates: Any) -> dict[str, Any]:
        with app.state.background_run_lock:
            run = app.state.background_runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "status": "queued",
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "batches_run": 0,
                    "stop_requested": False,
                },
            )
            run.update(updates)
            run["updated_at"] = _now_iso()
            return dict(run)

    def _get_background_run(run_id: str) -> dict[str, Any]:
        with app.state.background_run_lock:
            run = app.state.background_runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            return dict(run)

    def _background_stop_requested(run_id: str) -> bool:
        with app.state.background_run_lock:
            run = app.state.background_runs.get(run_id, {})
            return bool(run.get("stop_requested"))

    def _should_continue_background_batch(result: dict[str, Any]) -> bool:
        if result.get("done") is True or result.get("status") == "done":
            return False
        return result.get("terminal_reason") in {"max_steps_reached", "runtime_limit"}

    def _background_terminal_status(
        result: dict[str, Any],
        *,
        stop_requested: bool,
        max_batches_reached: bool,
    ) -> str:
        if stop_requested:
            return "stopped"
        if result.get("done") is True or result.get("status") == "done":
            return "done"
        if max_batches_reached:
            return "paused"
        terminal_reason = result.get("terminal_reason")
        if terminal_reason in {"provider_unhealthy", "stalled", "no_resume_action"}:
            return "paused"
        return "paused"

    def _supervise_existing_payload_from(
        payload: SupervisedJobRequest,
    ) -> SuperviseExistingJobRequest:
        return SuperviseExistingJobRequest(
            jobs_dir=payload.jobs_dir,
            workspace=payload.workspace_root or payload.repo_path,
            max_cycles=payload.max_cycles,
            steps_per_cycle=payload.steps_per_cycle,
            max_stalled_cycles=payload.max_stalled_cycles,
            max_runtime_seconds=payload.max_runtime_seconds,
            summary_file=payload.summary_file,
            summary_dir=payload.summary_dir,
            preflight_provider=payload.preflight_provider,
            preflight_timeout=payload.preflight_timeout,
            max_autonomous_stages=payload.max_autonomous_stages,
            require_prd_quality=payload.require_prd_quality,
            stage_review=payload.stage_review,
            test_timeout_seconds=payload.test_timeout_seconds,
            allow_blocked_recovery=payload.allow_blocked_recovery,
            pm_stall_recovery=payload.pm_stall_recovery,
        )

    def _run_background_supervised_job(
        run_id: str,
        payload: BackgroundSupervisedJobRequest,
    ) -> None:
        latest: dict[str, Any] = {}
        batches_run = 0
        try:
            _update_background_run(
                run_id,
                status="running",
                jobs_dir=payload.jobs_dir,
                max_batches=payload.max_batches,
                batch_cycles=payload.max_cycles,
            )
            latest = submit_supervised_job(payload)
            batches_run = 1
            job_id = str(latest.get("job_id") or payload.job_id or "")
            _update_background_run(
                run_id,
                status="running",
                job_id=job_id,
                batches_run=batches_run,
                last_result=latest,
            )
            while _should_continue_background_batch(latest):
                if _background_stop_requested(run_id):
                    break
                if payload.max_batches is not None and batches_run >= payload.max_batches:
                    break
                latest = supervise_existing_job(job_id, _supervise_existing_payload_from(payload))
                batches_run += 1
                _update_background_run(
                    run_id,
                    status="running",
                    job_id=job_id,
                    batches_run=batches_run,
                    last_result=latest,
                )
            stop_requested = _background_stop_requested(run_id)
            max_batches_reached = (
                payload.max_batches is not None and batches_run >= payload.max_batches
            )
            _update_background_run(
                run_id,
                status=_background_terminal_status(
                    latest,
                    stop_requested=stop_requested,
                    max_batches_reached=max_batches_reached,
                ),
                stop_requested=stop_requested,
                batches_run=batches_run,
                last_result=latest,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary for worker threads
            _update_background_run(
                run_id,
                status="error",
                error=str(exc),
                batches_run=batches_run,
                last_result=latest or None,
            )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", response_model=JobRecord)
    def submit_job(payload: SubmitJobRequest) -> JobRecord:
        spec = JobSpec(
            request_text=payload.request_text,
            repo_path=payload.repo_path,
            target_branch=payload.target_branch,
            metadata=payload.metadata,
        )
        return get_runner().run_job(spec)

    @app.post("/providers/check")
    def check_provider(payload: ProviderProbeRequest) -> dict[str, object] | None:
        return maybe_probe_provider(
            config_dir=config_dir,
            provider_name=payload.provider,
            timeout_seconds=payload.timeout,
        )

    @app.post("/jobs/supervised")
    def submit_supervised_job(payload: SupervisedJobRequest) -> dict[str, Any]:
        try:
            spec = build_job_spec_from_request(
                request_text=payload.request_text,
                repo_path=payload.repo_path,
                workspace_root=payload.workspace_root,
                target_branch=payload.target_branch,
                job_id=payload.job_id,
                title=payload.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        spec.metadata.update(payload.metadata)
        provider_preflight = maybe_probe_provider(
            config_dir=config_dir,
            provider_name=payload.preflight_provider,
            timeout_seconds=payload.preflight_timeout,
        )
        if provider_preflight is not None and not provider_preflight.get("healthy"):
            return provider_unhealthy_payload(
                provider_preflight=provider_preflight,
                job_id=spec.job_id,
                started=False,
            )
        apply_constraint_overrides(
            spec,
            max_autonomous_stages=payload.max_autonomous_stages,
            large_autonomous=True,
            require_prd_quality=payload.require_prd_quality,
            stage_review=payload.stage_review,
            test_timeout_seconds=payload.test_timeout_seconds,
        )
        store = FileJobStore(payload.jobs_dir)
        runner, _ = build_default_runner(
            config_dir=config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            store=store,
        )
        planning_payload: dict[str, Any] | None = None
        if payload.plan_first:
            record = runner.plan_job(spec)
            planning_payload = planning_result_payload(
                record,
                started=True,
                config_dir=config_dir,
                jobs_dir=payload.jobs_dir,
            )
            if not planning_payload["planning_complete"]:
                if provider_preflight is not None:
                    planning_payload.setdefault("provider_preflight", provider_preflight)
                    planning_payload["provider_events"] = [
                        _provider_preflight_event(
                            provider_preflight=provider_preflight,
                            cycle=None,
                            phase="pre_start",
                        )
                    ]
                    planning_payload["stop_summary"] = stop_summary_payload(
                        planning_payload
                    )
                return planning_payload
        else:
            record = runner.run_job(spec)
        if record.status == JobStatus.DONE:
            result = autonomous_result_payload(
                record,
                steps_run=0,
                max_steps=payload.max_cycles * payload.steps_per_cycle,
                started=True,
                config_dir=config_dir,
                jobs_dir=payload.jobs_dir,
                continued=False,
            )
            result.update(
                {
                    "cycles_run": 0,
                    "max_cycles": payload.max_cycles,
                    "steps_per_cycle": payload.steps_per_cycle,
                    "stalled": False,
                    "stalled_cycle_count": 0,
                    "max_stalled_cycles": payload.max_stalled_cycles,
                    "runtime_limited": False,
                    "elapsed_seconds": 0.0,
                    "max_runtime_seconds": payload.max_runtime_seconds,
                    "can_supervise_continue": False,
                    "next_supervise_cli_args": [],
                    "next_supervise_command": None,
                    "provider_events": [],
                    "cycle_summaries": [],
                    "initial_status": record.status.value,
                }
            )
        else:
            initial_status = record.status.value
            result = supervise_persisted_job(
                store=store,
                job_id=record.job_id,
                config_dir=config_dir,
                workspace=spec.workspace_root or spec.repo_path,
                max_cycles=payload.max_cycles,
                steps_per_cycle=payload.steps_per_cycle,
                max_stalled_cycles=payload.max_stalled_cycles,
                max_runtime_seconds=payload.max_runtime_seconds,
                jobs_dir=payload.jobs_dir,
                summary_file=payload.summary_file,
                summary_dir=payload.summary_dir,
                max_autonomous_stages=payload.max_autonomous_stages,
                large_autonomous=True,
                require_prd_quality=payload.require_prd_quality,
                stage_review=payload.stage_review,
                test_timeout_seconds=payload.test_timeout_seconds,
                preflight_provider=payload.preflight_provider,
                preflight_timeout=payload.preflight_timeout,
                allow_repeated_failure_recovery=payload.allow_blocked_recovery,
                pm_stall_recovery=payload.pm_stall_recovery,
            )
            result["started"] = True
            result["initial_status"] = initial_status
        if payload.plan_first:
            result["planned_first"] = True
            result["planning_complete"] = bool(
                planning_payload and planning_payload.get("planning_complete")
            )
            result["planning_result"] = planning_payload
        if provider_preflight is not None:
            result.setdefault("provider_preflight", provider_preflight)
            result["provider_events"] = [
                _provider_preflight_event(
                    provider_preflight=provider_preflight,
                    cycle=None,
                    phase="pre_start",
                ),
                *result.get("provider_events", []),
            ]
        result["stop_summary"] = stop_summary_payload(result)
        return result

    @app.post("/jobs/{job_id}/supervise")
    def supervise_existing_job(
        job_id: str,
        payload: SuperviseExistingJobRequest,
    ) -> dict[str, Any]:
        store = FileJobStore(payload.jobs_dir)
        try:
            record = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        result = supervise_persisted_job(
            store=store,
            job_id=job_id,
            config_dir=config_dir,
            workspace=payload.workspace or record.spec.workspace_root or record.spec.repo_path,
            max_cycles=payload.max_cycles,
            steps_per_cycle=payload.steps_per_cycle,
            max_stalled_cycles=payload.max_stalled_cycles,
            max_runtime_seconds=payload.max_runtime_seconds,
            jobs_dir=payload.jobs_dir,
            summary_file=payload.summary_file,
            summary_dir=payload.summary_dir,
            max_autonomous_stages=payload.max_autonomous_stages,
            large_autonomous=True,
            require_prd_quality=payload.require_prd_quality,
            stage_review=payload.stage_review,
            test_timeout_seconds=payload.test_timeout_seconds,
            preflight_provider=payload.preflight_provider,
            preflight_timeout=payload.preflight_timeout,
            allow_repeated_failure_recovery=payload.allow_blocked_recovery,
            pm_stall_recovery=payload.pm_stall_recovery,
        )
        result["started"] = False
        result["resumed_existing_job"] = True
        result["stop_summary"] = stop_summary_payload(result)
        return result

    @app.post("/jobs/supervised/background")
    def start_background_supervised_job(
        payload: BackgroundSupervisedJobRequest,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        _update_background_run(
            run_id,
            status="queued",
            jobs_dir=payload.jobs_dir,
            job_id=payload.job_id,
            max_batches=payload.max_batches,
            batch_cycles=payload.max_cycles,
        )
        thread = threading.Thread(
            target=_run_background_supervised_job,
            args=(run_id, payload),
            daemon=True,
            name=f"acos-background-{run_id[:8]}",
        )
        thread.start()
        return _get_background_run(run_id)

    @app.get("/background-runs/{run_id}")
    def get_background_run(run_id: str) -> dict[str, Any]:
        try:
            return _get_background_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="background run not found") from exc

    @app.get("/background-runs")
    def list_background_runs(job_id: str | None = None) -> list[dict[str, Any]]:
        with app.state.background_run_lock:
            runs = [dict(run) for run in app.state.background_runs.values()]
        if job_id is not None:
            runs = [run for run in runs if run.get("job_id") == job_id]
        return sorted(runs, key=lambda run: str(run.get("updated_at", "")), reverse=True)

    @app.post("/background-runs/{run_id}/stop")
    def stop_background_run(run_id: str) -> dict[str, Any]:
        try:
            run = _get_background_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="background run not found") from exc
        status = str(run.get("status"))
        next_status = status if status in {"done", "paused", "stopped", "error"} else "stopping"
        return _update_background_run(
            run_id,
            status=next_status,
            stop_requested=True,
        )

    @app.get("/jobs/{job_id}/progress")
    def get_job_progress(
        job_id: str,
        jobs_dir: str = Query(default=".acos/jobs"),
    ) -> dict[str, Any]:
        try:
            record = FileJobStore(jobs_dir).get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        summary = summarize_job_progress(record)
        recent_events: list[dict[str, Any]] = []
        for event in record.audit_events[-12:]:
            if hasattr(event, "model_dump"):
                recent_events.append(event.model_dump(mode="json"))
            elif isinstance(event, dict):
                recent_events.append(event)
        return {
            "job_id": record.job_id,
            "status": record.status.value,
            "history": list(record.history),
            "outputs_keys": sorted(record.outputs.keys()),
            "completed_task_ids": list(record.completed_task_ids),
            "checkpoint_count": len(record.checkpoints),
            "last_error": record.last_error,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "summary": summary,
            "pm_interventions": record.outputs.get("pm_interventions", []),
            "recent_audit_events": recent_events,
        }

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return get_runner().get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/models")
    def list_models() -> list[dict[str, object]]:
        registry = ModelRegistry.from_paths(
            provider_path=Path(__file__).resolve().parents[2] / "configs" / "model_providers.yaml",
            agents_path=Path(__file__).resolve().parents[2] / "configs" / "agents.yaml",
            routing_path=Path(__file__).resolve().parents[2] / "configs" / "model_routing.yaml",
        )
        return [model.model_dump() for model in registry.list_models()]

    @app.get("/agents")
    def list_agents() -> dict[str, dict[str, object]]:
        registry = ModelRegistry.from_paths(
            provider_path=Path(__file__).resolve().parents[2] / "configs" / "model_providers.yaml",
            agents_path=Path(__file__).resolve().parents[2] / "configs" / "agents.yaml",
            routing_path=Path(__file__).resolve().parents[2] / "configs" / "model_routing.yaml",
        )
        return {
            role: agent.model_dump()
            for role, agent in sorted(registry.agents.items())
        }

    @app.get("/config/validate")
    def validate_config() -> dict[str, object]:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        registry = ModelRegistry.from_paths(
            provider_path=config_dir / "model_providers.yaml",
            agents_path=config_dir / "agents.yaml",
            routing_path=config_dir / "model_routing.yaml",
        )
        policy = PolicyEngine.from_path(config_dir / "policies.yaml")
        errors = registry.validate(policy=policy)
        return {"ok": not errors, "errors": errors}

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=8080)
