"""FastAPI app for ACOS."""

from __future__ import annotations

import threading
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

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
from packages.orchestrator.approval import ApprovalError
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.job_store import FileJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.progress import summarize_job_progress
from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.schemas.approvals import ApprovalActionPayload, ApprovalRequest
from packages.schemas.models import JobStatus
from packages.schemas.jobs import JobRecord, JobSpec, validate_job_id_string


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

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return validate_job_id_string(value)


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


def create_app(
    job_runner: JobRunner | None = None,
    *,
    config_dir: str | Path | None = None,
    workspace_root: str | Path = ".",
) -> FastAPI:
    app = FastAPI(title="ACOS API", version="0.1.0")
    cors_origins = [
        origin.strip()
        for origin in os.environ.get("ACOS_CORS_ALLOW_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    resolved_config_dir = Path(config_dir or (Path(__file__).resolve().parents[2] / "configs"))
    app.state.job_runner = job_runner
    app.state.workspace_root = str(Path(workspace_root).resolve())
    app.state.background_runs = {}
    app.state.background_run_lock = threading.Lock()
    app.state.job_runner_factory = lambda: build_default_runner(
        config_dir=resolved_config_dir,
        workspace_root=Path(app.state.workspace_root),
    )[0]
    config_dir = resolved_config_dir
    rate_limit: dict[str, list[float]] = {}

    @app.middleware("http")
    async def _security_middleware(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            error = _authorize_mutation(request)
            if error is not None:
                return error
        return await call_next(request)

    def get_runner() -> JobRunner:
        runner = app.state.job_runner
        if runner is None:
            runner = app.state.job_runner_factory()
            app.state.job_runner = runner
        return runner

    def find_runner_for_approval(approval_id: str) -> tuple[JobRunner, ApprovalRequest]:
        runner = get_runner()
        if runner.approval_gateway is None:
            raise HTTPException(status_code=404, detail="approval gateway not configured")
        try:
            return runner, runner.approval_gateway.get(approval_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="approval not found") from exc

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _authorize_mutation(request: Request) -> JSONResponse | None:
        token = os.environ.get("ACOS_API_TOKEN")
        dev_disabled = os.environ.get("ACOS_LOCAL_DEV_AUTH_DISABLED", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        if not dev_disabled and not token:
            return JSONResponse(
                {"detail": "ACOS_API_TOKEN must be configured or local dev auth disabled explicitly"},
                status_code=503,
            )
        if token:
            provided = request.headers.get("x-acos-api-token")
            auth = request.headers.get("authorization", "")
            bearer = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else None
            if provided != token and bearer != token:
                return JSONResponse({"detail": "unauthorized"}, status_code=401)
        key = request.client.host if request.client else "unknown"
        now = datetime.now(timezone.utc).timestamp()
        bucket = [item for item in rate_limit.get(key, []) if now - item < 60]
        bucket.append(now)
        rate_limit[key] = bucket
        max_per_minute = int(os.environ.get("ACOS_RATE_LIMIT_PER_MINUTE", "120"))
        if len(bucket) > max_per_minute:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        return None

    def _assert_repo_allowed(repo_path: str) -> None:
        allowlist = [
            item.strip()
            for item in os.environ.get("ACOS_REPO_ALLOWLIST", "").split(os.pathsep)
            if item.strip()
        ]
        if not allowlist:
            return
        resolved = Path(repo_path).resolve()
        for root in allowlist:
            allowed = Path(root).resolve()
            if allowed in [resolved, *resolved.parents]:
                return
        raise HTTPException(status_code=403, detail="repo path is outside ACOS_REPO_ALLOWLIST")

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
        if result.get("status") in {
            "blocked",
            "stuck",
            "failed",
            "recovering",
            "diagnosing",
            "replanning",
            "strategy_change",
        }:
            return True
        return result.get("terminal_reason") in {
            "max_steps_reached",
            "runtime_limit",
            "provider_unhealthy",
        }

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
        _assert_repo_allowed(payload.repo_path)
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
            _assert_repo_allowed(payload.repo_path)
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
        _assert_repo_allowed(payload.workspace or record.spec.workspace_root or record.spec.repo_path)
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
                payload = event.model_dump(mode="json")
                if payload.get("tool_name") is None:
                    payload.pop("tool_name", None)
                recent_events.append(payload)
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
            "last_recoverable_error": summary.get("last_recoverable_error"),
            "current_recovery_event": summary.get("current_recovery_event"),
            "recovery_plan": summary.get("recovery_plan"),
            "model_metrics": summary.get("model_metrics"),
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

    @app.get("/approvals")
    def list_approvals(job_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        approvals = get_runner().list_approvals(job_id=job_id)
        return {"approvals": [item.model_dump(mode="json") for item in approvals]}

    @app.get("/approvals/{approval_id}")
    def get_approval(approval_id: str) -> dict[str, Any]:
        _runner, approval = find_runner_for_approval(approval_id)
        return approval.model_dump(mode="json")

    @app.post("/approvals/{approval_id}/approve")
    def approve_approval(
        approval_id: str,
        payload: ApprovalActionPayload,
    ) -> dict[str, Any]:
        runner, _approval = find_runner_for_approval(approval_id)
        if runner.approval_gateway is None:
            raise HTTPException(status_code=404, detail="approval gateway not configured")
        try:
            approval = runner.approval_gateway.approve(
                approval_id,
                token=payload.token,
                approver=payload.approver,
            )
            record = runner.resume_job(approval.job_id)
        except ApprovalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "approval": approval.model_dump(mode="json"),
            "job": record.model_dump(mode="json"),
        }

    @app.post("/approvals/{approval_id}/reject")
    def reject_approval(
        approval_id: str,
        payload: ApprovalActionPayload,
    ) -> dict[str, Any]:
        runner, _approval = find_runner_for_approval(approval_id)
        if runner.approval_gateway is None:
            raise HTTPException(status_code=404, detail="approval gateway not configured")
        try:
            approval = runner.approval_gateway.reject(
                approval_id,
                token=payload.token,
                approver=payload.approver,
                reason=payload.reason,
            )
            record = runner.resume_job(approval.job_id)
        except ApprovalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "approval": approval.model_dump(mode="json"),
            "job": record.model_dump(mode="json"),
        }

    @app.get("/approvals/{approval_id}/approve")
    def approve_via_link(
        approval_id: str,
        token: str = Query(...),
    ) -> dict[str, Any]:
        return approve_approval(approval_id, ApprovalActionPayload(token=token))

    @app.get("/approvals/{approval_id}/reject")
    def reject_via_link(
        approval_id: str,
        token: str = Query(...),
        reason: str | None = None,
    ) -> dict[str, Any]:
        return reject_approval(approval_id, ApprovalActionPayload(token=token, reason=reason))

    @app.get("/worker/status")
    def worker_status() -> dict[str, Any]:
        heartbeats = get_runner().store.list_worker_heartbeats()
        return {
            "status": "alive" if heartbeats else "idle",
            "heartbeats": [item.model_dump(mode="json") for item in heartbeats],
        }

    @app.get("/worker/heartbeats")
    def worker_heartbeats() -> dict[str, Any]:
        return {
            "heartbeats": [
                item.model_dump(mode="json") for item in get_runner().store.list_worker_heartbeats()
            ]
        }

    @app.get("/runtime/status")
    def runtime_status() -> dict[str, Any]:
        runner = get_runner()
        return {
            "runtime_issues": [
                issue.model_dump(mode="json") for issue in runner.store.list_runtime_issues()
            ],
            "waiting_jobs": [
                item.model_dump(mode="json")
                for item in runner.list_jobs(
                    statuses=[
                        JobStatus.WAITING_RUNTIME,
                        JobStatus.PROVIDER_UNAVAILABLE,
                        JobStatus.RETRYING_PROVIDER,
                    ]
                )
            ],
        }

    @app.post("/runtime/check")
    def runtime_check() -> dict[str, Any]:
        runner = get_runner()
        resumed = runner.runtime_manager.maybe_resume_waiting_jobs() if runner.runtime_manager else []
        return {"resumed_jobs": [item.model_dump(mode="json") for item in resumed]}

    @app.get("/providers")
    def providers() -> dict[str, list[dict[str, Any]]]:
        registry = ModelRegistry.from_paths(
            provider_path=config_dir / "model_providers.yaml",
            agents_path=config_dir / "agents.yaml",
            routing_path=config_dir / "model_routing.yaml",
        )
        return {
            "providers": [provider.model_dump(mode="json") for provider in registry.providers.values()]
        }

    @app.get("/providers/{provider_key}/health")
    def provider_health(provider_key: str) -> dict[str, Any]:
        registry = ModelRegistry.from_paths(
            provider_path=config_dir / "model_providers.yaml",
            agents_path=config_dir / "agents.yaml",
            routing_path=config_dir / "model_routing.yaml",
        )
        checker = ProviderHealthChecker(registry)
        return checker.check_provider(provider_key).model_dump(mode="json")

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
