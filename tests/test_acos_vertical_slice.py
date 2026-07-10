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
from packages.schemas.models import FixStatus, ImplementationStatus, ReviewDecision
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir


def test_acos_vertical_slice_runs_end_to_end(tmp_path: Path) -> None:
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
            "pm": PRD(
                title="Add Helper",
                problem_statement="Need a correct add helper",
                goals=["Pass tests"],
            ).model_dump(),
            "architect": ArchitecturePlan(
                summary="Single module and pytest test."
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
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create buggy implementation",
                changed_files=["feature.py"],
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                        "operation": "create",
                    }
                ],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add unit test",
                changed_files=["tests/test_feature.py"],
                patches=[
                    {
                        "path": "tests/test_feature.py",
                        "content": "from feature import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
                        "operation": "create",
                    }
                ],
                test_strategy=["happy path"],
            ).model_dump(),
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Proceed to tests",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="No security risks in this slice",
            ).model_dump(),
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Correct implementation",
                changed_files=["feature.py"],
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                        "operation": "update",
                    }
                ],
                addressed_failures=["test_add"],
            ).model_dump(),
            "summarizer": SummaryResult(
                summary="Feature implemented and validated.",
                memory_entries=["add helper implemented", "tests passing"],
            ).model_dump(),
            "release_manager": ReleaseResult(
                summary="Ready for release",
                commit_message="feat: implement add helper",
                notify_message="ACOS job completed",
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
        request_text="Implement add helper and tests",
        repo_path=str(workspace),
        target_branch="acos/vertical-slice",
    )

    record = runner.run_job(spec)
    feature_file = (workspace / "feature.py").read_text(encoding="utf-8")
    memory_entries = environment.memory_server.read_memory(limit=10)["entries"]

    assert record.status.value == "done"
    assert "return a + b" in feature_file
    assert record.outputs["test_run"]["success"] is True
    assert len(environment.git_server.commits) == 1
    assert environment.notify_server.notifications == ["ACOS job completed"]
    assert len(memory_entries) >= 2
    assert any(event.event_type == "tool_call" for event in record.audit_events)
    assert any(event.event_type == "model_call" for event in record.audit_events)
