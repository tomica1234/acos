from pathlib import Path

from packages.llm.errors import AdapterError
from packages.llm.registry import ModelRegistry
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.job_runner import JobRunner
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FilePatch,
    FixResult,
    ImplementationResult,
    PMReviewResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    RuntimePlan,
    SecurityReviewResult,
    SummaryResult,
    TestRunResult,
    TestWriterResult as TestWriterOutput,
)
from packages.schemas.jobs import JobSpec
from packages.schemas.models import FixStatus, ImplementationStatus, ProviderType, ReviewDecision, Severity
from packages.schemas.runtime import RuntimeHttpCheck
from packages.schemas.tasks import PlannedTask, TaskGraph

from tests.conftest import attach_mock_adapter, config_dir
from tests.fakes import build_approval_harness


def _django_scaffold_artifacts(project_name: str) -> list[str]:
    return [
        "manage.py",
        f"{project_name}/__init__.py",
        f"{project_name}/settings.py",
        f"{project_name}/urls.py",
        f"{project_name}/wsgi.py",
    ]


def _fastapi_scaffold_artifacts() -> list[str]:
    return ["app/__init__.py", "app/main.py"]


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
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="First delivery matches the request",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Second delivery matches the request",
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
                        required_artifacts=["feature.py", "tests/test_feature.py"],
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
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="First delivery matches the request",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Second delivery matches the request",
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
                        required_artifacts=["feature.py", "tests/test_feature.py"],
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


def test_job_runner_executes_multiple_planned_tasks_in_order(tmp_path: Path) -> None:
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
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="First delivery matches the request",
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Second delivery matches the request",
                ).model_dump(),
            ],
            "architect": ArchitecturePlan(summary="Simple architecture").model_dump(),
            "planner": TaskGraph(
                goal="Build feature",
                tasks=[
                    PlannedTask(
                        id="task-1",
                        title="Write first file",
                        description="Create feature_a.py",
                        role="implementer",
                    ),
                    PlannedTask(
                        id="task-2",
                        title="Write second file",
                        description="Create feature_b.py after task-1",
                        role="implementer",
                        dependencies=["task-1"],
                    ),
                ],
            ).model_dump(),
            "implementer": [
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Create first module",
                    changed_files=["feature_a.py"],
                    patches=[
                        {
                            "path": "feature_a.py",
                            "content": "VALUE_A = 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Create second module",
                    changed_files=["feature_b.py"],
                    patches=[
                        {
                            "path": "feature_b.py",
                            "content": "VALUE_B = 2\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
            ],
            "test_writer": [
                TestWriterOutput(
                    summary="Add first tests",
                    changed_files=["tests/test_feature_a.py"],
                    patches=[
                        {
                            "path": "tests/test_feature_a.py",
                            "content": "from feature_a import VALUE_A\n\n\ndef test_value_a() -> None:\n    assert VALUE_A == 1\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
                TestWriterOutput(
                    summary="Add second tests",
                    changed_files=["tests/test_feature_b.py"],
                    patches=[
                        {
                            "path": "tests/test_feature_b.py",
                            "content": "from feature_b import VALUE_B\n\n\ndef test_value_b() -> None:\n    assert VALUE_B == 2\n",
                            "operation": "create",
                        }
                    ],
                ).model_dump(),
            ],
            "reviewer": [
                ReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="First task looks good",
                ).model_dump(),
                ReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Second task looks good",
                ).model_dump(),
            ],
            "security_reviewer": [
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="First task is safe",
                ).model_dump(),
                SecurityReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Second task is safe",
                ).model_dump(),
            ],
            "summarizer": SummaryResult(
                summary="Done",
                memory_entries=["feature completed"],
            ).model_dump(),
            "release_manager": ReleaseResult(
                summary="Ready",
                commit_message="acos: finalize multi-task feature",
                notify_message="done",
            ).model_dump(),
        },
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
        scripted_test_results=[
            TestRunResult(success=True, command=["pytest"], exit_code=0),
            TestRunResult(success=True, command=["prepare-runtime-auto"], exit_code=0),
            TestRunResult(success=True, command=["runtime-smoke-auto"], exit_code=0),
            TestRunResult(success=True, command=["pytest"], exit_code=0),
            TestRunResult(success=True, command=["prepare-runtime-auto"], exit_code=0),
            TestRunResult(success=True, command=["runtime-smoke-auto"], exit_code=0),
        ],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    spec = JobSpec(
        request_text="Create two sequential features with tests",
        repo_path=str(workspace),
        target_branch="acos/multi-task",
    )

    record = runner.run_job(spec)

    assert record.status.value == "done"
    assert (workspace / "feature_a.py").read_text(encoding="utf-8") == "VALUE_A = 1\n"
    assert (workspace / "feature_b.py").read_text(encoding="utf-8") == "VALUE_B = 2\n"
    assert [task["status"] for task in record.outputs["task_graph"]["tasks"]] == ["done", "done"]


def test_job_runner_normalizes_absolute_patch_paths_within_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace,
        memory_db_path=workspace / ".memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    record = runner.submit(
        JobSpec(
            request_text="Create a feature file",
            repo_path=str(workspace),
            workspace_root=str(workspace),
            target_branch="acos/absolute-path-patch",
        )
    )

    runner._apply_patches(
        record,
        "implementer",
        [
            FilePatch(
                path=str(workspace / "feature.py"),
                content="VALUE = 1\n",
                operation="create",
            )
        ],
    )

    assert (workspace / "feature.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_job_runner_uses_runtime_commands_from_job_metadata(tmp_path: Path, monkeypatch) -> None:
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
                        required_artifacts=["manage.py", "feature.py", "tests/test_feature.py"],
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create module",
                patches=[
                    {
                        "path": "manage.py",
                        "content": "print('stub manage')\n",
                        "operation": "create",
                    },
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
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="OK",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
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
        scripted_test_results=[TestRunResult(success=True, command=["pytest"], exit_code=0)],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    original_call_tool = runner._call_tool
    runtime_calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            runtime_calls.append({"role": role, "tool_name": tool_name, **kwargs})
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(runner, "_call_tool", fake_call_tool)

    record = runner.run_job(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/runtime-commands",
            metadata={
                "runtime": {
                    "prepare_commands": [["python", "manage.py", "migrate"]],
                    "start_command": ["python", "manage.py", "runserver", "{host}:{port}", "--noreload"],
                    "http_probe_path": "/healthz",
                    "http_checks": [
                        {
                            "name": "home",
                            "path": "/",
                            "expect_status": 200,
                            "body_contains": ["Todo"],
                        }
                    ],
                },
                "acceptance_checks": [
                    {
                        "name": "create",
                        "method": "POST",
                        "path": "/create/",
                        "form": {"title": "milk"},
                        "expect_status": 200,
                        "body_contains": ["milk"],
                    }
                ],
            },
        )
    )

    assert record.status.value == "done"
    assert runtime_calls[0]["mode"] == "oneshot"
    assert runtime_calls[0]["argv"] == ["python", "manage.py", "migrate"]
    assert runtime_calls[1]["mode"] == "server"
    assert runtime_calls[1]["argv"] == ["python", "manage.py", "runserver", "{host}:{port}", "--noreload"]
    assert runtime_calls[1]["http_path"] == "/healthz"
    assert runtime_calls[1]["http_checks"] == [
        {
            "name": "home",
            "method": "GET",
            "path": "/",
            "headers": {},
            "expect_status": 200,
            "body_contains": ["Todo"],
            "body_not_contains": [],
            "follow_redirects": True,
            "use_csrf_from_last_response": True,
        }
    ]
    assert runtime_calls[2]["mode"] == "server"
    assert runtime_calls[2]["argv"] == ["python", "manage.py", "runserver", "{host}:{port}", "--noreload"]
    assert runtime_calls[2]["http_path"] == "/healthz"
    assert runtime_calls[2]["http_checks"] == [
        {
            "name": "create",
            "method": "POST",
            "path": "/create/",
            "headers": {},
            "form": {"title": "milk"},
            "expect_status": 200,
            "body_contains": ["milk"],
            "body_not_contains": [],
            "follow_redirects": True,
            "use_csrf_from_last_response": True,
        }
    ]


def test_job_runner_uses_django_framework_profile_defaults(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_name = workspace.name
    scaffold_artifacts = _django_scaffold_artifacts(project_name)
    required_artifacts = [*scaffold_artifacts, "feature.py", "tests/test_feature.py"]
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                    required_artifacts=required_artifacts,
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Delivery matches the request",
                    required_artifacts=required_artifacts,
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
                        required_artifacts=required_artifacts,
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
                    },
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
        scripted_test_results=[TestRunResult(success=True, command=["pytest"], exit_code=0)],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    original_call_tool = runner._call_tool
    runtime_calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            runtime_calls.append({"role": role, "tool_name": tool_name, **kwargs})
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(runner, "_call_tool", fake_call_tool)

    record = runner.run_job(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/django-profile-defaults",
            metadata={"framework_profile": "django-web"},
        )
    )

    assert record.status.value == "done"
    assert runtime_calls[0]["argv"] == ["python", "manage.py", "makemigrations"]
    assert runtime_calls[1]["argv"] == ["python", "manage.py", "migrate", "--noinput"]
    assert runtime_calls[2]["argv"] == ["python", "manage.py", "runserver", "{host}:{port}", "--noreload"]
    assert runtime_calls[2]["mode"] == "server"
    assert (workspace / "manage.py").exists()
    assert (workspace / project_name / "settings.py").exists()
    assert 'DJANGO_SETTINGS_MODULE' in (workspace / "manage.py").read_text(encoding="utf-8")


def test_job_runner_uses_fastapi_framework_profile_defaults(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    required_artifacts = [*_fastapi_scaffold_artifacts(), "tests/test_app.py"]
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                    required_artifacts=required_artifacts,
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Delivery matches the request",
                    required_artifacts=required_artifacts,
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
                        required_artifacts=required_artifacts,
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Rely on deterministic scaffold for app bootstrap",
                patches=[],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_app.py",
                        "content": "def test_placeholder() -> None:\n    assert 1 == 1\n",
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
        scripted_test_results=[TestRunResult(success=True, command=["pytest"], exit_code=0)],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    original_call_tool = runner._call_tool
    runtime_calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            runtime_calls.append({"role": role, "tool_name": tool_name, **kwargs})
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(runner, "_call_tool", fake_call_tool)

    record = runner.run_job(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/fastapi-profile-defaults",
            metadata={
                "framework_profile": "fastapi-api",
                "framework_entrypoint": "app.main:app",
            },
        )
    )

    assert record.status.value == "done"
    assert runtime_calls[0]["argv"] == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    assert runtime_calls[0]["mode"] == "server"
    assert (workspace / "app" / "main.py").exists()


def test_job_runner_scaffold_does_not_overwrite_existing_framework_files(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_name = workspace.name
    scaffold_artifacts = _django_scaffold_artifacts(project_name)
    required_artifacts = [*scaffold_artifacts, "feature.py", "tests/test_feature.py"]
    existing_manage = "print('custom manage')\n"
    existing_settings = "CUSTOM = True\n"
    (workspace / "manage.py").write_text(existing_manage, encoding="utf-8")
    (workspace / project_name).mkdir()
    (workspace / project_name / "settings.py").write_text(existing_settings, encoding="utf-8")
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(title="Feature", problem_statement="Need feature").model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                    required_artifacts=required_artifacts,
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Delivery matches the request",
                    required_artifacts=required_artifacts,
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
                        required_artifacts=required_artifacts,
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Create feature only",
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
            "reviewer": ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="OK",
            ).model_dump(),
            "security_reviewer": SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Safe",
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
        scripted_test_results=[TestRunResult(success=True, command=["pytest"], exit_code=0)],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    original_call_tool = runner._call_tool

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(runner, "_call_tool", fake_call_tool)

    record = runner.run_job(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/django-scaffold-preserve-existing",
            metadata={"framework_profile": "django-web"},
        )
    )

    assert record.status.value == "done"
    assert (workspace / "manage.py").read_text(encoding="utf-8") == existing_manage
    assert (workspace / project_name / "settings.py").read_text(encoding="utf-8") == existing_settings
    assert (workspace / project_name / "wsgi.py").exists()


def test_job_runner_uses_prd_generated_execution_contract(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    required_artifacts = [*_fastapi_scaffold_artifacts(), "tests/test_app.py"]
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )
    attach_mock_adapter(
        registry,
        {
            "pm": [
                PRD(
                    title="Status API",
                    problem_statement="Need a FastAPI status API",
                    framework_profile="fastapi-api",
                    framework_entrypoint="app.main:app",
                    required_artifacts=["tests/test_app.py"],
                    acceptance_checks=[
                        RuntimeHttpCheck(
                            name="home",
                            method="GET",
                            path="/",
                            expect_status=200,
                        )
                    ],
                    runtime=RuntimePlan(
                        http_checks=[
                            RuntimeHttpCheck(
                                name="health",
                                method="GET",
                                path="/healthz",
                                expect_status=200,
                            )
                        ]
                    ),
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Design is complete",
                    required_artifacts=required_artifacts,
                ).model_dump(),
                PMReviewResult(
                    decision=ReviewDecision.APPROVE,
                    summary="Delivery matches the request",
                    required_artifacts=required_artifacts,
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
                        required_artifacts=required_artifacts,
                    )
                ],
            ).model_dump(),
            "implementer": ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="Rely on synthesized scaffold for app bootstrap",
                patches=[],
            ).model_dump(),
            "test_writer": TestWriterOutput(
                summary="Add tests",
                patches=[
                    {
                        "path": "tests/test_app.py",
                        "content": "def test_placeholder() -> None:\n    assert 1 == 1\n",
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
        scripted_test_results=[TestRunResult(success=True, command=["pytest"], exit_code=0)],
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    original_call_tool = runner._call_tool
    runtime_calls: list[dict[str, object]] = []

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            runtime_calls.append({"role": role, "tool_name": tool_name, **kwargs})
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(runner, "_call_tool", fake_call_tool)

    record = runner.run_job(
        JobSpec(
            request_text="Create feature with tests",
            repo_path=str(workspace),
            target_branch="acos/prd-generated-contract",
        )
    )

    assert record.status.value == "done"
    assert record.spec.metadata["framework_profile"] == "fastapi-api"
    assert record.spec.metadata["framework_entrypoint"] == "app.main:app"
    assert record.spec.metadata["required_artifacts"] == required_artifacts
    assert runtime_calls[0]["argv"] == [
        "python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    assert runtime_calls[0]["http_checks"] == [
        {
            "name": "health",
            "method": "GET",
            "path": "/healthz",
            "headers": {},
            "expect_status": 200,
            "body_contains": [],
            "body_not_contains": [],
            "follow_redirects": True,
            "use_csrf_from_last_response": True,
        }
    ]
    assert runtime_calls[1]["http_checks"] == [
        {
            "name": "home",
            "method": "GET",
            "path": "/",
            "headers": {},
            "expect_status": 200,
            "body_contains": [],
            "body_not_contains": [],
            "follow_redirects": True,
            "use_csrf_from_last_response": True,
        }
    ]


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


def test_job_runner_surfaces_timeout_when_fallbacks_are_exhausted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_paths(
        provider_path=config_dir() / "model_providers.yaml",
        agents_path=config_dir() / "agents.yaml",
        routing_path=config_dir() / "model_routing.yaml",
    )

    class TimeoutAdapter:
        def generate(self, **_: object) -> object:
            raise AdapterError("provider timed out", code="timeout")

    registry.register_adapter_factory(
        ProviderType.OPENAI_COMPATIBLE,
        lambda provider, model: TimeoutAdapter(),
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
        target_branch="acos/provider-timeout",
    )

    record = runner.run_job(spec)

    assert record.status.value == "failed"
    assert record.last_error == (
        "Fallbacks exhausted for role pm after timeout; "
        "attempted models: qwen_35b"
    )
