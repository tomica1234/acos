"""FastAPI app for ACOS."""

from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from packages.llm.registry import ModelRegistry
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.jobs import JobRecord, JobSpec


class SubmitJobRequest(BaseModel):
    request_text: str
    repo_path: str
    target_branch: str = "acos/default"
    metadata: dict[str, object] = Field(default_factory=dict)


def create_app(job_runner: JobRunner | None = None) -> FastAPI:
    app = FastAPI(title="ACOS API", version="0.1.0")
    app.state.job_runner = job_runner
    app.state.job_runner_factory = lambda: build_default_runner(
        config_dir=Path(__file__).resolve().parents[2] / "configs",
        workspace_root=Path("."),
    )[0]

    def get_runner() -> JobRunner:
        runner = app.state.job_runner
        if runner is None:
            runner = app.state.job_runner_factory()
            app.state.job_runner = runner
        return runner

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
