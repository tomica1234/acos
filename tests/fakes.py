from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from packages.agents.runner import AgentRunner
from packages.llm.routing import ModelRouter
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.approval import ApprovalGateway, SQLiteApprovalStore
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FixResult,
    ImplementationResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    SecurityReviewResult,
    SummaryResult,
    TestRunResult,
    TestWriterResult,
)
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    ReviewDecision,
)
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir, load_registry


@dataclass(slots=True)
class FakeAgentRunner:
    """A recording wrapper around the real AgentRunner used by vertical slice tests."""

    delegate: AgentRunner
    invocations: list[dict[str, Any]] = field(default_factory=list)

    def run(self, *args: Any, **kwargs: Any):
        role = kwargs.get("role")
        result = self.delegate.run(*args, **kwargs)
        output, selection, record = result
        self.invocations.append(
            {
                "role": role,
                "model_key": selection.model_key,
                "provider_key": selection.provider_key,
                "reason": selection.reason.value,
                "status": record.status.value,
            }
        )
        return output, selection, record


@dataclass(slots=True)
class VerticalSliceHarness:
    """Build a deterministic ACOS stack for scenario-driven integration tests."""

    workspace: Path
    environment: FakeMCPEnvironment
    runner: JobRunner
    fake_agent_runner: FakeAgentRunner

    def run_job(
        self,
        *,
        request_text: str = "Build the requested feature",
        target_branch: str = "acos/vertical-slice",
    ) -> JobRecord:
        spec = JobSpec(
            request_text=request_text,
            repo_path=str(self.workspace),
            target_branch=target_branch,
        )
        return self.runner.run_job(spec)

    def memory_entries(self, *, limit: int = 20) -> list[dict[str, str]]:
        return self.environment.memory_server.read_memory(limit=limit)["entries"]

    def role_invocations(self, role: str) -> list[dict[str, Any]]:
        return [item for item in self.fake_agent_runner.invocations if item["role"] == role]


@dataclass(slots=True)
class ApprovalHarness:
    workspace: Path
    environment: FakeMCPEnvironment
    runner: JobRunner
    fake_agent_runner: FakeAgentRunner

    def run_job(
        self,
        *,
        request_text: str = "Update the large feature file safely",
        target_branch: str = "acos/approval-slice",
    ) -> JobRecord:
        spec = JobSpec(
            request_text=request_text,
            repo_path=str(self.workspace),
            workspace_root=str(self.workspace),
            target_branch=target_branch,
        )
        return self.runner.run_job(spec)


def build_vertical_slice_harness(
    tmp_path: Path,
    *,
    scenario: dict[str, Any],
    scripted_test_results: Iterable[TestRunResult] | None = None,
    max_attempts_per_task: int = 3,
    max_same_failure_repeats: int = 2,
) -> VerticalSliceHarness:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = load_registry()
    attach_mock_adapter(registry, scenario)
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
        scripted_test_results=scripted_test_results,
    )
    model_router = ModelRouter(registry)
    real_agent_runner = AgentRunner(
        registry=registry,
        model_router=model_router,
        mcp_router=environment.build_router(),
        policy_engine=policy,
    )
    fake_agent_runner = FakeAgentRunner(delegate=real_agent_runner)
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        model_router=model_router,
        agent_runner=fake_agent_runner,
    )
    runner.max_attempts_per_task = max_attempts_per_task
    runner.max_same_failure_repeats = max_same_failure_repeats
    return VerticalSliceHarness(
        workspace=workspace,
        environment=environment,
        runner=runner,
        fake_agent_runner=fake_agent_runner,
    )


def build_approval_harness(tmp_path: Path) -> ApprovalHarness:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "feature.py").write_text("VALUE = 0\n" * 2505, encoding="utf-8")
    registry = load_registry()
    scenario = base_vertical_slice_scenario(
        implementer=ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Replace the large file with the final implementation.",
            changed_files=["feature.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "VALUE = 1\n",
                    "operation": "update",
                }
            ],
        ).model_dump(),
        test_writer=TestWriterResult(
            summary="Add a unit test for VALUE.",
            changed_files=["tests/test_feature.py"],
            patches=[
                {
                    "path": "tests/test_feature.py",
                    "content": (
                        "from feature import VALUE\n\n\n"
                        "def test_value() -> None:\n"
                        "    assert VALUE == 1\n"
                    ),
                    "operation": "create",
                }
            ],
            test_strategy=["assert exported constant equals one"],
        ).model_dump(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    attach_mock_adapter(registry, scenario)
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    workspace_policy = policy.build_workspace_policy(workspace)
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
        workspace_policy=workspace_policy,
    )
    model_router = ModelRouter(registry)
    real_agent_runner = AgentRunner(
        registry=registry,
        model_router=model_router,
        mcp_router=environment.build_router(),
        policy_engine=policy,
    )
    fake_agent_runner = FakeAgentRunner(delegate=real_agent_runner)
    approval_gateway = ApprovalGateway(
        SQLiteApprovalStore(workspace / ".approvals.sqlite3"),
        request_ttl_minutes=policy.config.approval.request_ttl_minutes,
        allow_cli_approval=policy.config.approval.allow_cli_approval,
        allow_http_approval=policy.config.approval.allow_http_approval,
        allow_notification_links=policy.config.approval.allow_notification_links,
        require_signed_tokens=policy.config.approval.require_signed_tokens,
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=environment.build_router(),
        model_router=model_router,
        agent_runner=fake_agent_runner,
        approval_gateway=approval_gateway,
        store=InMemoryJobStore(workspace / ".jobs.json"),
    )
    return ApprovalHarness(
        workspace=workspace,
        environment=environment,
        runner=runner,
        fake_agent_runner=fake_agent_runner,
    )


def implementation_patch(content: str, *, operation: str = "create") -> dict[str, str]:
    return {"path": "feature.py", "content": content, "operation": operation}


def test_patch(assertion: str = "assert add(2, 3) == 5") -> dict[str, str]:
    return {
        "path": "tests/test_feature.py",
        "content": (
            "from feature import add\n\n\n"
            "def test_add() -> None:\n"
            f"    {assertion}\n"
        ),
        "operation": "create",
    }


def approval_review(summary: str = "Looks good") -> dict[str, Any]:
    return ReviewResult(
        decision=ReviewDecision.APPROVE,
        summary=summary,
    ).model_dump()


def approval_security_review(summary: str = "Looks safe") -> dict[str, Any]:
    return SecurityReviewResult(
        decision=ReviewDecision.APPROVE,
        summary=summary,
    ).model_dump()


def base_vertical_slice_scenario(
    *,
    implementer: Any,
    test_writer: Any,
    reviewer: Any,
    security_reviewer: Any,
    fixer: Any | None = None,
    summarizer: Any | None = None,
    release_manager: Any | None = None,
) -> dict[str, Any]:
    return {
        "pm": PRD(
            title="Add Helper",
            problem_statement="Need a correct add helper",
            goals=["Pass tests"],
        ).model_dump(),
        "architect": ArchitecturePlan(
            summary="Single module and pytest test.",
            components=["feature.py", "tests/test_feature.py"],
        ).model_dump(),
        "planner": TaskGraph(
            goal="Implement and validate add helper",
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="Implement helper",
                    description="Create add function and tests",
                    role="implementer",
                )
            ],
        ).model_dump(),
        "implementer": implementer,
        "test_writer": test_writer,
        "reviewer": reviewer,
        "security_reviewer": security_reviewer,
        "fixer": fixer
        if fixer is not None
        else FixResult(
            status=FixStatus.FIXED,
            summary="No fixer work required",
            patches=[],
        ).model_dump(),
        "summarizer": summarizer
        if summarizer is not None
        else SummaryResult(
            summary="Feature implemented and validated.",
            memory_entries=["vertical slice completed"],
        ).model_dump(),
        "release_manager": release_manager
        if release_manager is not None
        else ReleaseResult(
            summary="Ready for release",
            commit_message="acos: finalize vertical slice",
            notify_message="ACOS job completed",
        ).model_dump(),
    }


def implemented_result(summary: str, content: str, *, operation: str = "create") -> dict[str, Any]:
    return ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary=summary,
        changed_files=["feature.py"],
        patches=[implementation_patch(content, operation=operation)],
    ).model_dump()


def make_test_writer_result(summary: str = "Add unit test") -> dict[str, Any]:
    return TestWriterResult(
        summary=summary,
        changed_files=["tests/test_feature.py"],
        patches=[test_patch()],
        test_strategy=["happy path"],
    ).model_dump()
