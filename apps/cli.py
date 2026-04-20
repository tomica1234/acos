"""CLI for running ACOS locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import uvicorn
import yaml

from packages.llm.adapters.mock import MockAdapter
from packages.llm.errors import ConfigValidationError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.orchestrator.approval import ApprovalError
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
    worker.add_argument("--request")
    worker.add_argument("--branch", default="acos/default")
    worker.add_argument("--file")

    run_demo = subparsers.add_parser("run-demo")
    run_demo.add_argument("--workspace", required=True)
    run_demo.add_argument("--config-dir", default="configs")

    run_job = subparsers.add_parser("run-job")
    run_job.add_argument("--config-dir", default="configs")
    run_job.add_argument("--file", required=True)

    approvals = subparsers.add_parser("approvals")
    approvals_subparsers = approvals.add_subparsers(dest="approvals_command", required=True)

    approvals_list = approvals_subparsers.add_parser("list")
    approvals_list.add_argument("--config-dir", default="configs")
    approvals_list.add_argument("--workspace", default=".")
    approvals_list.add_argument("--job-id")

    approvals_show = approvals_subparsers.add_parser("show")
    approvals_show.add_argument("approval_id")
    approvals_show.add_argument("--config-dir", default="configs")
    approvals_show.add_argument("--workspace", default=".")

    approvals_approve = approvals_subparsers.add_parser("approve")
    approvals_approve.add_argument("approval_id")
    approvals_approve.add_argument("--config-dir", default="configs")
    approvals_approve.add_argument("--workspace", default=".")
    approvals_approve.add_argument("--token")
    approvals_approve.add_argument("--approver", default="cli")

    approvals_reject = approvals_subparsers.add_parser("reject")
    approvals_reject.add_argument("approval_id")
    approvals_reject.add_argument("--config-dir", default="configs")
    approvals_reject.add_argument("--workspace", default=".")
    approvals_reject.add_argument("--token")
    approvals_reject.add_argument("--approver", default="cli")
    approvals_reject.add_argument("--reason", default="rejected via CLI")

    jobs = subparsers.add_parser("jobs")
    jobs_subparsers = jobs.add_subparsers(dest="jobs_command", required=True)

    jobs_resume = jobs_subparsers.add_parser("resume")
    jobs_resume.add_argument("job_id")
    jobs_resume.add_argument("--config-dir", default="configs")
    jobs_resume.add_argument("--workspace", default=".")
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


def dump_yaml(payload: dict[str, Any]) -> None:
    safe_payload = json.loads(json.dumps(payload, default=str))
    print(yaml.safe_dump(safe_payload, sort_keys=False, allow_unicode=True))


def serialize_model(model_key: str, model: Any) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "provider": model.provider,
        "model_id": model.model,
        "display_name": model.display_name,
        "max_context_tokens": model.max_context_tokens,
        "supports_tool_calling": model.supports_tool_calling,
        "supports_structured_output": model.supports_structured_output,
        "tags": list(model.tags),
    }


def serialize_agent(agent: Any) -> dict[str, Any]:
    return {
        "role": agent.role,
        "primary_model": agent.primary_model,
        "fallback_models": list(agent.fallback_models),
        "context_budget_tokens": agent.context_budget_tokens,
        "allowed_tools_count": len(agent.allowed_tools),
        "allowed_tools": list(agent.allowed_tools),
        "output_schema": agent.output_schema,
    }


def summarize_escalation_conditions(registry: ModelRegistry, role: str) -> dict[str, Any] | None:
    config = registry.routing.escalation.get(role)
    if config is None:
        return None
    conditions = {
        key: value
        for key, value in config.escalate_when.model_dump(mode="json").items()
        if value not in (None, [], {}, False)
    }
    return {
        "escalated_model": config.escalated_model,
        "conditions": conditions,
    }


def build_context_budget_note(
    registry: ModelRegistry,
    role: str,
    selected_model_key: str,
    current_context_tokens: int,
) -> dict[str, Any]:
    agent = registry.get_agent(role)
    selected_model = registry.get_model(selected_model_key)
    return {
        "agent_context_budget_tokens": agent.context_budget_tokens,
        "selected_model_max_context_tokens": selected_model.max_context_tokens,
        "selected_model_max_output_tokens": selected_model.max_output_tokens,
        "current_context_tokens": current_context_tokens,
        "note": (
            "The router enforces both the role budget and the selected model limit. "
            "If the packet grows too large, ContextBuilder must truncate or summarize."
        ),
    }


def explain_routing_for_humans(
    registry: ModelRegistry,
    routing_context: RoutingContext,
) -> dict[str, Any]:
    router = ModelRouter(registry)
    explanation = router.explain_routing(routing_context)
    agent = registry.get_agent(routing_context.role)
    primary_model = registry.get_model(agent.primary_model)
    selected_model = registry.get_model(explanation["selection"]["model_key"])
    fallback_models = [
        serialize_model(model_key, registry.get_model(model_key))
        for model_key in agent.fallback_models
    ]
    capability_requirements = explanation["capability_requirements"]
    escalation_summary = summarize_escalation_conditions(registry, routing_context.role)
    summary_lines = [
        (
            f"{routing_context.role} normally uses {primary_model.model_id} "
            f"({primary_model.display_name})."
        ),
        (
            "Fallback models: "
            + (
                ", ".join(model["model_key"] for model in fallback_models)
                if fallback_models
                else "none"
            )
            + "."
        ),
        (
            "Capability requirements: "
            f"tools={capability_requirements['requires_tools']}, "
            f"strict_json={capability_requirements['requires_strict_json']}."
        ),
        (
            f"Current selection is {selected_model.model_id} because "
            f"{explanation['selection']['reason']} matched."
        ),
    ]
    if escalation_summary is not None:
        summary_lines.append(
            "Escalation conditions: "
            + yaml.safe_dump(escalation_summary["conditions"], sort_keys=True).strip()
        )
    return {
        "role": routing_context.role,
        "human_summary": summary_lines,
        "normal_model": serialize_model(primary_model.model_id, primary_model),
        "fallback_models": fallback_models,
        "fallback_errors": list(registry.routing.fallback.on_errors),
        "escalation_conditions": escalation_summary,
        "capability_requirements": capability_requirements,
        "context_budget": build_context_budget_note(
            registry=registry,
            role=routing_context.role,
            selected_model_key=selected_model.model_id,
            current_context_tokens=routing_context.context_tokens,
        ),
        "current_selection": {
            "selected_model": explanation["selection"]["model_key"],
            "provider": explanation["selection"]["provider_key"],
            "routing_reason": explanation["selection"]["reason"],
            "details": explanation["selection"]["details"],
        },
    }


def load_job_spec_from_file(path: str | Path) -> JobSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("job file must contain a YAML mapping")
    request_text = payload.get("request_text") or payload.get("requester_input")
    if not request_text:
        raise ValueError("job file requires request_text or requester_input")
    repo_path = str(Path(payload.get("repo_path", ".")).resolve())
    target_branch = (
        payload.get("target_branch")
        or payload.get("working_branch")
        or payload.get("branch")
        or "acos/default"
    )
    reserved = {
        "job_id",
        "request_text",
        "requester_input",
        "repo_path",
        "workspace_root",
        "target_branch",
        "working_branch",
        "branch",
        "metadata",
    }
    metadata = dict(payload.get("metadata", {}))
    for key, value in payload.items():
        if key not in reserved:
            metadata[key] = value
    spec_payload: dict[str, Any] = {
        "request_text": request_text,
        "repo_path": repo_path,
        "target_branch": target_branch,
        "metadata": metadata,
        "workspace_root": str(Path(payload.get("workspace_root", repo_path)).resolve()),
    }
    if "job_id" in payload:
        spec_payload["job_id"] = payload["job_id"]
    return JobSpec.model_validate(spec_payload)


def load_runner_for_workspace(
    *,
    config_dir: str | Path,
    workspace_root: str | Path,
) -> JobRunner:
    runner, _ = build_default_runner(
        config_dir=config_dir,
        workspace_root=workspace_root,
    )
    return runner


def serialize_approval(approval: Any) -> dict[str, Any]:
    return approval.model_dump(mode="json")


def build_job_result_payload(record: Any) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "status": record.status.value,
        "target_branch": record.spec.target_branch,
        "repo_path": record.spec.repo_path,
        "workspace_root": record.spec.workspace_root,
        "metadata": record.spec.metadata,
        "failure_count": record.failure_count,
        "same_test_failure_count": record.same_test_failure_count,
        "last_error": record.last_error,
        "pending_approval_id": record.pending_approval_id,
        "audit_event_count": len(record.audit_events),
        "outputs": record.outputs,
    }


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
    for provider_type in {provider.type for provider in registry.providers.values()}:
        registry.register_adapter_factory(
            provider_type=provider_type,
            factory=lambda provider, model, adapter=shared_mock: adapter,
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
            dump_yaml({"ok": False, "errors": errors})
            return 1
        dump_yaml({"ok": True, "errors": []})
        return 0
    if args.command == "list-models":
        registry, _ = load_registry_and_policy(args.config_dir)
        payload = {
            "models": [
                serialize_model(model.model_id, model)
                for model in registry.list_models()
            ]
        }
        dump_yaml(payload)
        return 0
    if args.command == "list-agents":
        registry, _ = load_registry_and_policy(args.config_dir)
        payload = {
            "agents": [serialize_agent(agent) for agent in registry.list_agents()]
        }
        dump_yaml(payload)
        return 0
    if args.command == "resolve-model":
        registry, _ = load_registry_and_policy(args.config_dir)
        router = ModelRouter(registry)
        routing_context = build_cli_routing_context(args)
        selection = router.select_model(routing_context)
        explanation = router.explain_routing(routing_context)
        model = registry.get_model(selection.model_key)
        payload = {
            "role": selection.role,
            "selected_model": selection.model_key,
            "provider": selection.provider_key,
            "model_id": model.model,
            "display_name": model.display_name,
            "routing_reason": selection.reason.value,
            "fallback_candidates": explanation["fallback_models"],
            "escalation_condition_summary": summarize_escalation_conditions(
                registry, selection.role
            ),
            "capability_requirements": explanation["capability_requirements"],
            "context_budget": build_context_budget_note(
                registry=registry,
                role=selection.role,
                selected_model_key=selection.model_key,
                current_context_tokens=routing_context.context_tokens,
            ),
            "model_key": selection.model_key,
            "provider_key": selection.provider_key,
            "reason": selection.reason.value,
            "details": selection.details,
            "fallback_models": explanation["fallback_models"],
            "fallback_errors": explanation["fallback_errors"],
        }
        dump_yaml(payload)
        return 0
    if args.command == "explain-routing":
        registry, _ = load_registry_and_policy(args.config_dir)
        routing_context = build_cli_routing_context(args)
        dump_yaml(explain_routing_for_humans(registry, routing_context))
        return 0
    if args.command == "list-tools":
        _, policy = load_registry_and_policy(args.config_dir)
        print(yaml.safe_dump(policy.list_allowed_tools(role=args.role), sort_keys=False))
        return 0
    if args.command == "api":
        uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.command == "worker":
        if args.file:
            spec = load_job_spec_from_file(args.file)
            runner = load_runner_for_workspace(
                config_dir=args.config_dir,
                workspace_root=spec.workspace_root or spec.repo_path,
            )
            record = runner.run_job(spec)
            print(yaml.safe_dump(build_job_result_payload(record), sort_keys=False, allow_unicode=True))
            return 0 if record.status.value == "done" else 1
        if args.request is None:
            dump_yaml(
                {
                    "status": "idle",
                    "message": (
                        "ACOS worker MVP is available. Pass --request or --file to execute a single job."
                    ),
                }
            )
            return 0
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
        dump_yaml(
            {
                **build_job_result_payload(record),
                "notifications": list(environment.notify_server.notifications),
            }
        )
        return 0 if record.status.value == "done" else 1
    if args.command == "run-job":
        spec = load_job_spec_from_file(args.file)
        runner = load_runner_for_workspace(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
        )
        record = runner.run_job(spec)
        dump_yaml(build_job_result_payload(record))
        return 0 if record.status.value == "done" else 1
    if args.command == "approvals":
        runner = load_runner_for_workspace(
            config_dir=args.config_dir,
            workspace_root=args.workspace,
        )
        if args.approvals_command == "list":
            payload = {
                "approvals": [
                    serialize_approval(item)
                    for item in runner.list_approvals(job_id=args.job_id)
                ]
            }
            dump_yaml(payload)
            return 0
        if args.approvals_command == "show":
            if runner.approval_gateway is None:
                dump_yaml({"ok": False, "error": "approval gateway is not configured"})
                return 1
            try:
                approval = runner.approval_gateway.get(args.approval_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "approval not found"})
                return 1
            dump_yaml({"approval": serialize_approval(approval)})
            return 0
        if args.approvals_command == "approve":
            if runner.approval_gateway is None:
                dump_yaml({"ok": False, "error": "approval gateway is not configured"})
                return 1
            try:
                approval = runner.approval_gateway.approve(
                    args.approval_id,
                    token=args.token,
                    approver=args.approver,
                )
                record = runner.resume_job(approval.job_id)
            except (ApprovalError, KeyError) as exc:
                dump_yaml({"ok": False, "error": str(exc)})
                return 1
            dump_yaml(
                {
                    "ok": True,
                    "approval": serialize_approval(approval),
                    "job": build_job_result_payload(record),
                }
            )
            return 0
        if args.approvals_command == "reject":
            if runner.approval_gateway is None:
                dump_yaml({"ok": False, "error": "approval gateway is not configured"})
                return 1
            try:
                approval = runner.approval_gateway.reject(
                    args.approval_id,
                    token=args.token,
                    approver=args.approver,
                    reason=args.reason,
                )
                record = runner.resume_job(approval.job_id)
            except (ApprovalError, KeyError) as exc:
                dump_yaml({"ok": False, "error": str(exc)})
                return 1
            dump_yaml(
                {
                    "ok": True,
                    "approval": serialize_approval(approval),
                    "job": build_job_result_payload(record),
                }
            )
            return 0
    if args.command == "jobs":
        runner = load_runner_for_workspace(
            config_dir=args.config_dir,
            workspace_root=args.workspace,
        )
        if args.jobs_command == "resume":
            try:
                record = runner.resume_job(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            dump_yaml({"ok": True, "job": build_job_result_payload(record)})
            return 0
    return 1
