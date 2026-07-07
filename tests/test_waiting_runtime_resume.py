from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from packages.llm.errors import AdapterError
from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore, utc_now
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.runtime import RuntimeManager
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    ImplementationResult,
    PMReviewResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    SecurityReviewResult,
    SummaryResult,
    TestWriterResult,
)
from packages.schemas.jobs import JobSpec
from packages.schemas.models import ImplementationStatus, JobStatus, ProviderType, ReviewDecision, Severity
from packages.schemas.runtime import ProviderHealth, ProviderHealthStatus, RuntimeConfig, ResumeConfig
from packages.schemas.tasks import PlannedTask, TaskGraph
from tests.conftest import attach_mock_adapter, config_dir


class _DownAdapter:
    def generate(self, **_: object) -> object:
        raise AdapterError("provider down", code="timeout")


class _HealthChecker:
    def __init__(self, status: ProviderHealthStatus) -> None:
        self.status = status

    def check_model(self, model_key: str) -> ProviderHealth:
        return ProviderHealth(
            provider_key="local_qwen",
            model_key=model_key,
            status=self.status,
            message=self.status.value,
        )

    def check_provider(self, provider_key: str) -> ProviderHealth:
        return ProviderHealth(
            provider_key=provider_key,
            status=self.status,
            message=self.status.value,
        )


def _build_runner(tmp_path: Path, *, auto_resume: bool = True) -> tuple[JobRunner, FakeMCPEnvironment, ModelRegistry]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    registry.register_adapter_factory(ProviderType.OPENAI_COMPATIBLE, lambda provider, model: _DownAdapter())
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    store = InMemoryJobStore(workspace / ".jobs.json")
    runtime_manager = RuntimeManager(
        store=store,
        health_checker=_HealthChecker(ProviderHealthStatus.CONNECTION_ERROR),
        config=RuntimeConfig(
            resume=ResumeConfig(auto_resume_after_provider_recovery=auto_resume),
        ),
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        store=store,
        runtime_manager=runtime_manager,
    )
    return runner, environment, registry


def _attach_success_scenario(registry: ModelRegistry) -> None:
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Delivery matches the request",
                ).model_dump(),
            ],
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write code",
                        description="Write code",
                        role="implementer",
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterResult(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Looks good",
                findings=[],
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Looks safe",
                findings=[],
            ).model_dump(),
            "summarizer": SummaryResult(
                summary="Done",
                memory_entries=["feature completed"],
            ).model_dump(),
            "release_manager": ReleaseResult(
                summary="Ready",
                commit_message="acos: finalize feature",
                notify_message="done",
            ).model_dump(),
        },
    )


def test_waiting_runtime_resume_flow(tmp_path) -> None:
    runner, environment, registry = _build_runner(tmp_path)
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str((tmp_path / "workspace").resolve()),
        target_branch="acos/runtime-wait",
    )

    record = runner.run_job(spec)

    assert record.status == JobStatus.WAITING_RUNTIME
    assert environment.notify_server.runtime_notifications

    _attach_success_scenario(registry)
    runner.llm_client._adapters.clear()
    runner.agent_runner._adapter_cache.clear()
    runner.runtime_manager.health_checker = _HealthChecker(ProviderHealthStatus.OK)
    issue = runner.store.get_runtime_issue(record.pending_runtime_issue_id)
    issue.next_retry_at = utc_now() - timedelta(seconds=1)
    runner.store.save_runtime_issue(issue)

    resumed = runner.runtime_manager.maybe_resume_waiting_jobs()
    final = runner.resume_job(record.job_id)

    assert resumed
    assert runner.get(record.job_id).status == JobStatus.DONE
    assert final.status == JobStatus.DONE


def test_waiting_runtime_manual_resume_pauses_after_recovery(tmp_path) -> None:
    runner, _environment, _registry = _build_runner(tmp_path, auto_resume=False)
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str((tmp_path / "workspace").resolve()),
        target_branch="acos/runtime-manual",
    )
    record = runner.run_job(spec)
    issue = runner.store.get_runtime_issue(record.pending_runtime_issue_id)
    issue.next_retry_at = utc_now() - timedelta(seconds=1)
    runner.store.save_runtime_issue(issue)
    runner.runtime_manager.health_checker = _HealthChecker(ProviderHealthStatus.OK)

    runner.runtime_manager.maybe_resume_waiting_jobs()

    assert runner.get(record.job_id).status == JobStatus.PAUSED
