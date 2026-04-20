from pathlib import Path

from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.job_runner import JobRunner
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
    TestWriterResult as TestWriterOutput,
)
from packages.schemas.jobs import JobSpec
from packages.schemas.models import FixStatus, ImplementationStatus, ReviewDecision, Severity
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir
from tests.fakes import build_approval_harness


def test_job_runner_review_request_changes_then_fix(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
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
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import VALUE\n\n\ndef test_value() -> None:\n    assert VALUE == 1\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "reviewer": [
                ReviewResult(
                    decision=ReviewDecision.REQUEST_CHANGES,
                    summary="Needs work",
                    findings=[
                        {
                            "severity": Severity.MEDIUM,
                            "title": "Refactor",
                            "description": "Please refactor",
                        }
                    ],
                ).model_dump(),
                ReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Looks good now",
                ).model_dump(),
            ],
            "security_reviewer": [
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Looks safe",
                ).model_dump(),
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Still safe",
                ).model_dump(),
            ],
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Addressed review concerns",
                patches=[],
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
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/review-fixed",
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert len(environment.git_server.commits) == 1


def test_job_runner_max_attempts_stuck(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(title="Feature", problem_statement="Need feature").model_dump(),
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
                summary="Create buggy module",
                patches=[
                    {
                        "path": "feature.py",
                        "content": "VALUE = 0\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
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
                summary="OK",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
            ).model_dump(),
            "fixer": [
                FixResult(status=FixStatus.FIXED, summary="No effective change", patches=[]).model_dump(),
                FixResult(status=FixStatus.FIXED, summary="Still no effective change", patches=[]).model_dump(),
            ],
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create feature with tests",
        repo_path=str(workspace),
        target_branch="acos/stuck-case",
    )

    record = runner.run_job(spec)

    assert record.status.value == "stuck"


def test_job_runner_enters_waiting_approval_and_can_resume(tmp_path: Path) -> None:
    harness = build_approval_harness(tmp_path)

    record = harness.run_job()

    assert record.status.value == "waiting_approval"
    assert record.pending_approval_id is not None
    assert harness.environment.notify_server.approval_notifications

    harness.runner.approval_gateway.approve(
        record.pending_approval_id,
        token=None,
        approver="cli",
    )
    resumed = harness.runner.resume_job(record.job_id)

    assert resumed.status.value == "done"
    assert resumed.pending_approval_id is None


def test_job_runner_reject_blocks_job(tmp_path: Path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()

    harness.runner.approval_gateway.reject(
        record.pending_approval_id,
        token=None,
        approver="cli",
        reason="do not modify this file",
    )
    blocked = harness.runner.resume_job(record.job_id)

    assert blocked.status.value == "blocked"
    assert blocked.last_error == "do not modify this file"
