"""FastAPI app for ACOS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from packages.llm.registry import ModelRegistry
from packages.orchestrator.approval import ApprovalError
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.approvals import ApprovalActionPayload, ApprovalRequest
from packages.schemas.jobs import JobRecord, JobSpec


class SubmitJobRequest(BaseModel):
    request_text: str
    repo_path: str
    target_branch: str = "acos/default"
    metadata: dict[str, object] = Field(default_factory=dict)
    workspace_root: str | None = None


def create_app(
    job_runner: JobRunner | None = None,
    *,
    config_dir: str | Path | None = None,
    workspace_root: str | Path = ".",
) -> FastAPI:
    app = FastAPI(title="ACOS API", version="0.1.0")
    resolved_config_dir = Path(config_dir or (Path(__file__).resolve().parents[2] / "configs"))
    app.state.config_dir = resolved_config_dir
    app.state.workspace_root = str(Path(workspace_root).resolve())
    app.state.runners: dict[str, JobRunner] = {}
    if job_runner is not None:
        app.state.runners[app.state.workspace_root] = job_runner

    def get_runner_for_workspace(root: str | Path | None = None) -> JobRunner:
        key = str(Path(root or app.state.workspace_root).resolve())
        runner = app.state.runners.get(key)
        if runner is None:
            runner, _ = build_default_runner(
                config_dir=resolved_config_dir,
                workspace_root=key,
            )
            app.state.runners[key] = runner
        return runner

    def iter_runners() -> list[JobRunner]:
        if not app.state.runners:
            return [get_runner_for_workspace(app.state.workspace_root)]
        return list(app.state.runners.values())

    def find_runner_for_job(job_id: str) -> JobRunner:
        for runner in iter_runners():
            try:
                runner.get(job_id)
                return runner
            except KeyError:
                continue
        raise HTTPException(status_code=404, detail="job not found")

    def find_runner_for_approval(
        approval_id: str,
        workspace: str | None = None,
    ) -> tuple[JobRunner, ApprovalRequest]:
        candidate_runners = (
            [get_runner_for_workspace(workspace)]
            if workspace is not None
            else iter_runners()
        )
        for runner in candidate_runners:
            if runner.approval_gateway is None:
                continue
            try:
                return runner, runner.approval_gateway.get(approval_id)
            except KeyError:
                continue
        raise HTTPException(status_code=404, detail="approval not found")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", response_model=JobRecord)
    def submit_job(payload: SubmitJobRequest) -> JobRecord:
        runner = get_runner_for_workspace(payload.workspace_root or payload.repo_path)
        spec = JobSpec(
            request_text=payload.request_text,
            repo_path=payload.repo_path,
            target_branch=payload.target_branch,
            metadata=payload.metadata,
            workspace_root=payload.workspace_root or payload.repo_path,
        )
        return runner.run_job(spec)

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        return find_runner_for_job(job_id).get(job_id)

    @app.post("/jobs/{job_id}/resume", response_model=JobRecord)
    def resume_job(job_id: str) -> JobRecord:
        return find_runner_for_job(job_id).resume_job(job_id)

    @app.get("/approvals")
    def list_approvals(
        job_id: str | None = None,
        workspace: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        runners = [get_runner_for_workspace(workspace)] if workspace is not None else iter_runners()
        approvals: list[ApprovalRequest] = []
        seen: set[str] = set()
        for runner in runners:
            for approval in runner.list_approvals(job_id=job_id):
                if approval.id in seen:
                    continue
                approvals.append(approval)
                seen.add(approval.id)
        return {"approvals": [item.model_dump(mode="json") for item in approvals]}

    @app.get("/approvals/{approval_id}")
    def get_approval(
        approval_id: str,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        _runner, approval = find_runner_for_approval(approval_id, workspace=workspace)
        return approval.model_dump(mode="json")

    @app.post("/approvals/{approval_id}/approve")
    def approve_approval(
        approval_id: str,
        payload: ApprovalActionPayload,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        runner, _approval = find_runner_for_approval(approval_id, workspace=workspace)
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
        workspace: str | None = None,
    ) -> dict[str, Any]:
        runner, _approval = find_runner_for_approval(approval_id, workspace=workspace)
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
        workspace: str | None = None,
    ) -> dict[str, Any]:
        payload = ApprovalActionPayload(token=token)
        return approve_approval(approval_id, payload, workspace=workspace)

    @app.get("/approvals/{approval_id}/reject")
    def reject_via_link(
        approval_id: str,
        token: str = Query(...),
        reason: str | None = None,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        payload = ApprovalActionPayload(token=token, reason=reason)
        return reject_approval(approval_id, payload, workspace=workspace)

    @app.get("/models")
    def list_models() -> list[dict[str, object]]:
        registry = ModelRegistry.from_paths(
            provider_path=resolved_config_dir / "model_providers.yaml",
            agents_path=resolved_config_dir / "agents.yaml",
            routing_path=resolved_config_dir / "model_routing.yaml",
        )
        return [model.model_dump() for model in registry.list_models()]

    @app.get("/agents")
    def list_agents() -> dict[str, dict[str, object]]:
        registry = ModelRegistry.from_paths(
            provider_path=resolved_config_dir / "model_providers.yaml",
            agents_path=resolved_config_dir / "agents.yaml",
            routing_path=resolved_config_dir / "model_routing.yaml",
        )
        return {
            role: agent.model_dump()
            for role, agent in sorted(registry.agents.items())
        }

    @app.get("/config/validate")
    def validate_config() -> dict[str, object]:
        registry = ModelRegistry.from_paths(
            provider_path=resolved_config_dir / "model_providers.yaml",
            agents_path=resolved_config_dir / "agents.yaml",
            routing_path=resolved_config_dir / "model_routing.yaml",
        )
        policy = PolicyEngine.from_path(resolved_config_dir / "policies.yaml")
        errors = registry.validate(policy=policy)
        return {"ok": not errors, "errors": errors}

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="127.0.0.1", port=8080)
