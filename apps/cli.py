"""CLI for running ACOS locally."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import uvicorn
import yaml

from packages.llm.adapters.mock import MockAdapter
from packages.llm.errors import ConfigValidationError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.policy import PolicyEngine
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FixResult,
    ImplementationResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    SecurityReviewResult,
    SummaryResult,
    TestWriterResult,
)
from packages.schemas.jobs import JobSpec
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    ReviewDecision,
    Severity,
    TaskComplexity,
)
from packages.schemas.tasks import PlannedTask, TaskGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config-dir", default="configs")

    list_models = subparsers.add_parser("list-models")
    list_models.add_argument("--config-dir", default="configs")

    list_agents = subparsers.add_parser("list-agents")
    list_agents.add_argument("--config-dir", default="configs")

    resolve_model = subparsers.add_parser("resolve-model")
    resolve_model.add_argument("--config-dir", default="configs")
    resolve_model.add_argument("--role", required=True)
    resolve_model.add_argument("--repeated-failures", "--failure-count", dest="repeated_failures", type=int, default=0)
    resolve_model.add_argument(
        "--same-test-failures",
        "--same-test-failure-count",
        dest="same_test_failures",
        type=int,
        default=0,
    )
    resolve_model.add_argument("--changed-files", type=int, default=0)
    resolve_model.add_argument(
        "--task-complexity",
        choices=["low", "medium", "high", "critical"],
        default="medium",
    )
    resolve_model.add_argument("--security-sensitive", action="store_true")
    resolve_model.add_argument("--last-error")
    resolve_model.add_argument("--context-tokens", type=int, default=0)

    explain_routing = subparsers.add_parser("explain-routing")
    explain_routing.add_argument("--config-dir", default="configs")
    explain_routing.add_argument("--role", required=True)
    explain_routing.add_argument("--repeated-failures", type=int, default=0)
    explain_routing.add_argument("--same-test-failures", type=int, default=0)
    explain_routing.add_argument("--changed-files", type=int, default=0)
    explain_routing.add_argument(
        "--task-complexity",
        choices=["low", "medium", "high", "critical"],
        default="medium",
    )
    explain_routing.add_argument("--security-sensitive", action="store_true")
    explain_routing.add_argument("--last-error")
    explain_routing.add_argument("--context-tokens", type=int, default=0)

    list_tools = subparsers.add_parser("list-tools")
    list_tools.add_argument("--config-dir", default="configs")
    list_tools.add_argument("--role")

    api = subparsers.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--config-dir", default="configs")
    worker.add_argument("--repo", default=".")
    worker.add_argument("--request", required=True)
    worker.add_argument("--branch", default="acos/default")

    run_demo = subparsers.add_parser("run-demo")
    run_demo.add_argument("--workspace", required=True)
    run_demo.add_argument("--config-dir", default="configs")

    run_job = subparsers.add_parser("run-job")
    run_job.add_argument("--config-dir", default="configs")
    run_job.add_argument("--file", required=True)
    return parser


def load_registry_and_policy(
    config_dir: str | Path,
) -> tuple[ModelRegistry, PolicyEngine]:
    config_path = Path(config_dir)
    registry = ModelRegistry.load_from_paths(
        provider_path=config_path / "model_providers.yaml",
        agents_path=config_path / "agents.yaml",
        routing_path=config_path / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_path / "policies.yaml")
    registry.validate_or_raise(policy=policy)
    return registry, policy


def validate_config_bundle(config_dir: str | Path) -> list[str]:
    config_path = Path(config_dir)
    try:
        registry = ModelRegistry.load_from_paths(
            provider_path=config_path / "model_providers.yaml",
            agents_path=config_path / "agents.yaml",
            routing_path=config_path / "model_routing.yaml",
        )
        policy = PolicyEngine.from_path(config_path / "policies.yaml")
    except ConfigValidationError as exc:
        return list(exc.errors)
    except Exception as exc:
        return [str(exc)]
    return registry.validate(policy=policy)


def build_cli_routing_context(args: argparse.Namespace) -> RoutingContext:
    return RoutingContext(
        role=args.role,
        task_complexity=TaskComplexity(args.task_complexity),
        failure_count=args.repeated_failures,
        same_test_failure_count=args.same_test_failures,
        changed_files_count=args.changed_files,
        security_sensitive=args.security_sensitive,
        context_tokens=args.context_tokens,
        last_error=args.last_error,
    )


def build_demo_runner(config_dir: str | Path, workspace: str | Path) -> tuple[JobRunner, FakeMCPEnvironment]:
    config_path = Path(config_dir)
    workspace_root = Path(workspace)
    workspace_root.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry.from_paths(
        provider_path=config_path / "model_providers.yaml",
        agents_path=config_path / "agents.yaml",
        routing_path=config_path / "model_routing.yaml",
    )
    scenario = {
        "pm": PRD(
            title="Demo Feature",
            problem_statement="Implement a simple add function.",
            users=["developer"],
            goals=["Provide a correct add helper"],
            constraints=["Use deterministic tests"],
            success_criteria=["pytest passes"],
        ).model_dump(),
        "architect": ArchitecturePlan(
            summary="Use a single module plus pytest coverage.",
            components=["feature.py", "tests/test_feature.py"],
            data_flows=["test imports feature.add"],
            risks=["Incorrect arithmetic implementation"],
            decisions=["Keep the example intentionally small"],
        ).model_dump(),
        "planner": TaskGraph(
            goal="Build and validate add helper",
            tasks=[
                PlannedTask(
                    id="task-1",
                    title="Implement add helper",
                    description="Create a tiny function and test.",
                    role="implementer",
                )
            ],
            notes=["This is a local demo path"],
        ).model_dump(),
        "implementer": ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Create an initial implementation with a bug.",
            changed_files=["feature.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                    "operation": "create",
                }
            ],
        ).model_dump(),
        "test_writer": TestWriterResult(
            summary="Add a simple unit test.",
            changed_files=["tests/test_feature.py"],
            patches=[
                {
                    "path": "tests/test_feature.py",
                    "content": "from feature import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
                    "operation": "create",
                }
            ],
            test_strategy=["Validate positive integer addition"],
        ).model_dump(),
        "reviewer": ReviewResult(
            decision=ReviewDecision.APPROVE,
            summary="Implementation is acceptable for demo purposes.",
            findings=[],
        ).model_dump(),
        "security_reviewer": SecurityReviewResult(
            decision=ReviewDecision.APPROVE,
            summary="No security-sensitive behavior in demo scope.",
            findings=[],
        ).model_dump(),
        "fixer": FixResult(
            status=FixStatus.FIXED,
            summary="Correct the arithmetic bug.",
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
            summary="Implemented and validated add helper.",
            memory_entries=["demo feature completed", "add helper now passes tests"],
        ).model_dump(),
        "release_manager": ReleaseResult(
            summary="Ready for release.",
            commit_message="feat: complete demo add helper",
            notify_message="ACOS demo job finished successfully.",
        ).model_dump(),
    }
    shared_mock = MockAdapter(scenario=scenario)
    registry.register_adapter_factory(
        provider_type=registry.get_provider("local_qwen").type,
        factory=lambda provider, model: shared_mock,
    )
    registry.register_adapter_factory(
        provider_type=registry.get_provider("local_small").type,
        factory=lambda provider, model: shared_mock,
    )
    registry.register_adapter_factory(
        provider_type=registry.get_provider("mock_provider").type,
        factory=lambda provider, model: shared_mock,
    )
    policy = PolicyEngine.from_path(config_path / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace_root,
        memory_db_path=workspace_root / ".acos_memory.sqlite3",
    )
    runner = JobRunner(registry=registry, policy=policy, router=environment.build_router())
    return runner, environment


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        errors = validate_config_bundle(args.config_dir)
        if errors:
            print(yaml.safe_dump({"ok": False, "errors": errors}, sort_keys=False))
            return 1
        print(yaml.safe_dump({"ok": True, "errors": []}, sort_keys=False))
        return 0
    if args.command == "list-models":
        registry, _ = load_registry_and_policy(args.config_dir)
        payload = {
            "models": [
                {
                    "model_key": model.model_id,
                    "provider": model.provider,
                    "model_id": model.model,
                    "max_context_tokens": model.max_context_tokens,
                    "supports_tool_calling": model.supports_tool_calling,
                    "tags": list(model.tags),
                }
                for model in registry.list_models()
            ]
        }
        print(yaml.safe_dump(payload, sort_keys=False))
        return 0
    if args.command == "list-agents":
        registry, _ = load_registry_and_policy(args.config_dir)
        payload = {
            "agents": [
                {
                    "role": agent.role,
                    "primary_model": agent.primary_model,
                    "fallback_models": list(agent.fallback_models),
                    "context_budget_tokens": agent.context_budget_tokens,
                    "max_output_tokens": agent.max_output_tokens,
                    "allow_tools": agent.allow_tools,
                    "allowed_tools": list(agent.allowed_tools),
                    "require_json_schema": agent.require_json_schema,
                    "output_schema": agent.output_schema,
                }
                for agent in registry.list_agents()
            ]
        }
        print(yaml.safe_dump(payload, sort_keys=False))
        return 0
    if args.command == "resolve-model":
        registry, _ = load_registry_and_policy(args.config_dir)
        router = ModelRouter(registry)
        selection = router.select_model(build_cli_routing_context(args))
        explanation = router.explain_routing(build_cli_routing_context(args))
        payload = {
            "role": selection.role,
            "model_key": selection.model_key,
            "provider_key": selection.provider_key,
            "reason": selection.reason.value,
            "details": selection.details,
            "primary_model": explanation["primary_model"],
            "fallback_models": explanation["fallback_models"],
            "fallback_errors": explanation["fallback_errors"],
            "escalation": explanation["escalation"],
        }
        print(yaml.safe_dump(payload, sort_keys=False))
        return 0
    if args.command == "explain-routing":
        registry, _ = load_registry_and_policy(args.config_dir)
        explanation = ModelRouter(registry).explain_routing(build_cli_routing_context(args))
        print(yaml.safe_dump(explanation, sort_keys=False))
        return 0
    if args.command == "list-tools":
        _, policy = load_registry_and_policy(args.config_dir)
        print(yaml.safe_dump(policy.list_allowed_tools(role=args.role), sort_keys=False))
        return 0
    if args.command == "api":
        uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.command == "worker":
        from apps.worker.main import main as worker_main

        return worker_main(
            [
                "--config-dir",
                args.config_dir,
                "--repo",
                args.repo,
                "--request",
                args.request,
                "--branch",
                args.branch,
            ]
        )
    if args.command == "run-demo":
        runner, environment = build_demo_runner(args.config_dir, args.workspace)
        spec = JobSpec(
            request_text="Create a correct add helper with tests.",
            repo_path=str(Path(args.workspace).resolve()),
            target_branch="acos/demo",
        )
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        print(environment.notify_server.notifications)
        return 0 if record.status.value == "done" else 1
    if args.command == "run-job":
        payload = yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}
        repo_path = payload.get("repo_path", ".")
        runner, _ = build_default_runner(config_dir=args.config_dir, workspace_root=repo_path)
        spec = JobSpec.model_validate(payload)
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1
    return 1
