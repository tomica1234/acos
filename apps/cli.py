"""CLI for running ACOS locally."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence, TextIO

import uvicorn
import yaml

from packages.agents.config import get_role_prompt
from packages.llm.budget import (
    compute_max_output_tokens,
    estimate_tokens_from_messages,
    resolve_configured_max_output_tokens,
)
from packages.llm.adapters.mock import MockAdapter
from packages.llm.errors import ConfigValidationError
from packages.llm.messages import build_messages
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.orchestrator.approval import ApprovalError
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.orchestrator.runtime import RuntimeManager
from packages.orchestrator.worker_daemon import WorkerConfig, WorkerDaemon
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
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    JobStatus,
    ReviewDecision,
    TaskComplexity,
)
from packages.schemas.tasks import PlannedTask, TaskGraph
from packages.schemas.runtime import RuntimeConfig


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

    debug = subparsers.add_parser("debug")
    debug_subparsers = debug.add_subparsers(dest="debug_command", required=True)
    debug_token_budget = debug_subparsers.add_parser("token-budget")
    debug_token_budget.add_argument("--config-dir", default="configs")
    debug_token_budget.add_argument("--workspace", default=".")
    debug_token_budget.add_argument("--role", required=True)
    source_group = debug_token_budget.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file")
    source_group.add_argument("--job-id")

    api = subparsers.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8080)

    worker = subparsers.add_parser("worker")
    worker.add_argument("worker_action", nargs="?", choices=["run", "recover"], default="run")
    worker.add_argument("--config-dir", default="configs")
    worker.add_argument("--repo", default=".")
    worker.add_argument("--workspace")
    worker.add_argument("--request")
    worker.add_argument("--branch", default="acos/default")
    worker.add_argument("--file")
    worker.add_argument("--forever", action="store_true")

    run_demo = subparsers.add_parser("run-demo")
    run_demo.add_argument("--workspace", required=True)
    run_demo.add_argument("--config-dir", default="configs")

    run_job = subparsers.add_parser("run-job")
    run_job.add_argument("--config-dir", default="configs")
    run_job.add_argument("--file", required=True)
    run_job.add_argument("--quiet", action="store_true")

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

    jobs_submit = jobs_subparsers.add_parser("submit")
    jobs_submit.add_argument("--config-dir", default="configs")
    jobs_submit.add_argument("--workspace", default=".")
    jobs_submit.add_argument("--file", required=True)

    jobs_list = jobs_subparsers.add_parser("list")
    jobs_list.add_argument("--config-dir", default="configs")
    jobs_list.add_argument("--workspace", default=".")

    jobs_show = jobs_subparsers.add_parser("show")
    jobs_show.add_argument("job_id")
    jobs_show.add_argument("--config-dir", default="configs")
    jobs_show.add_argument("--workspace", default=".")

    jobs_status = jobs_subparsers.add_parser("status")
    jobs_status.add_argument("job_id")
    jobs_status.add_argument("--config-dir", default="configs")
    jobs_status.add_argument("--workspace", default=".")

    jobs_watch = jobs_subparsers.add_parser("watch")
    jobs_watch.add_argument("job_id")
    jobs_watch.add_argument("--config-dir", default="configs")
    jobs_watch.add_argument("--workspace", default=".")
    jobs_watch.add_argument("--poll-interval", type=float, default=1.0)
    jobs_watch.add_argument("--max-iterations", type=int)

    jobs_pause = jobs_subparsers.add_parser("pause")
    jobs_pause.add_argument("job_id")
    jobs_pause.add_argument("--config-dir", default="configs")
    jobs_pause.add_argument("--workspace", default=".")

    jobs_resume = jobs_subparsers.add_parser("resume")
    jobs_resume.add_argument("job_id")
    jobs_resume.add_argument("--config-dir", default="configs")
    jobs_resume.add_argument("--workspace", default=".")

    jobs_cancel = jobs_subparsers.add_parser("cancel")
    jobs_cancel.add_argument("job_id")
    jobs_cancel.add_argument("--config-dir", default="configs")
    jobs_cancel.add_argument("--workspace", default=".")

    jobs_logs = jobs_subparsers.add_parser("logs")
    jobs_logs.add_argument("job_id")
    jobs_logs.add_argument("--config-dir", default="configs")
    jobs_logs.add_argument("--workspace", default=".")

    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("runtime_action", nargs="?", choices=["status", "check", "watch"], default="status")
    runtime.add_argument("--config-dir", default="configs")
    runtime.add_argument("--workspace", default=".")
    runtime.add_argument("--poll-interval", type=float, default=2.0)
    runtime.add_argument("--max-iterations", type=int)

    check_provider = subparsers.add_parser("check-provider")
    check_provider.add_argument("--config-dir", default="configs")
    check_provider.add_argument("--provider", required=True)

    check_model = subparsers.add_parser("check-model")
    check_model.add_argument("--config-dir", default="configs")
    check_model.add_argument("--model", required=True)

    daemon = subparsers.add_parser("daemon")
    daemon.add_argument(
        "daemon_action",
        nargs="?",
        choices=["start", "stop", "status", "logs", "install-launchd", "uninstall-launchd"],
        default="status",
    )
    daemon.add_argument("--config-dir", default="configs")
    daemon.add_argument("--workspace", default=".")
    daemon.add_argument("--foreground", action="store_true")
    daemon.add_argument("--detach", action="store_true")
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
        if (config_path / "runtime.yaml").exists():
            runtime_payload = yaml.safe_load((config_path / "runtime.yaml").read_text(encoding="utf-8")) or {}
            RuntimeConfig(**(runtime_payload.get("runtime") or {}))
        if (config_path / "worker.yaml").exists():
            worker_payload = yaml.safe_load((config_path / "worker.yaml").read_text(encoding="utf-8")) or {}
            WorkerConfig(**(worker_payload.get("worker") or {}))
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


def build_token_budget_debug_payload(
    *,
    runner: JobRunner,
    role: str,
    spec: JobSpec,
    record: JobRecord | None = None,
) -> dict[str, Any]:
    registry = runner.registry
    agent = registry.get_agent(role)
    debug_record = (
        record.model_copy(deep=True)
        if record is not None
        else JobRecord(
            job_id=spec.job_id,
            spec=spec,
            status=JobStatus.QUEUED,
            current_role=role,
        )
    )
    previous_record = getattr(runner, "_active_record", None)
    runner._active_record = debug_record
    try:
        relevant_files = runner._gather_relevant_files(role)
        diff = (
            runner._call_tool(role, "git_server.diff").get("diff", "")
            if runner.policy.is_tool_allowed(role, "git_server.diff")
            else ""
        )
        memory_summaries = runner._read_memory(role)
    finally:
        runner._active_record = previous_record
    routing_context = RoutingContext(
        role=role,
        failure_count=record.failure_count if record is not None else 0,
        same_test_failure_count=record.same_test_failure_count if record is not None else 0,
        changed_files_count=len(relevant_files),
        security_sensitive=False,
        last_error=record.last_error if record is not None else None,
    )
    selection = runner.model_router.select_model(routing_context)
    selected_model = registry.get_model(selection.model_key)
    packet = runner.context_builder.build(
        job_id=debug_record.job_id,
        role=role,
        objective=f"Debug token budget for {role}",
        repo_path=spec.workspace_root or spec.repo_path,
        request_text=spec.request_text,
        constraints=list(runner.policy.config.risk_rules.deny),
        relevant_files=relevant_files,
        diff=diff,
        memory_summaries=memory_summaries,
        logs=[],
        token_budget=agent.context_budget_tokens,
        agent_config=agent,
        selected_model=selected_model,
        metadata={"debug": True},
    )
    messages = build_messages(get_role_prompt(role), packet)
    estimated_input_tokens = estimate_tokens_from_messages(messages)
    configured_max_output_tokens = resolve_configured_max_output_tokens(
        selection.max_output_tokens,
        selected_model.max_output_tokens,
        runner.token_budget_policy.default_output_tokens,
    )
    resolved_max_output_tokens = compute_max_output_tokens(
        model_max_context_tokens=selected_model.max_context_tokens,
        estimated_input_tokens=estimated_input_tokens,
        configured_max_output_tokens=configured_max_output_tokens,
        safety_margin_tokens=runner.token_budget_policy.safety_margin_tokens,
        minimum_output_tokens=runner.token_budget_policy.minimum_output_tokens,
        hard_max_output_tokens=runner.token_budget_policy.hard_max_output_tokens,
    )
    return {
        "role": role,
        "selected_model": selection.model_key,
        "model_max_context_tokens": selected_model.max_context_tokens,
        "context_budget_tokens": agent.context_budget_tokens,
        "estimated_input_tokens": estimated_input_tokens,
        "configured_max_output_tokens": configured_max_output_tokens,
        "resolved_max_output_tokens": resolved_max_output_tokens,
        "safety_margin_tokens": runner.token_budget_policy.safety_margin_tokens,
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


def build_worker_daemon(
    *,
    config_dir: str | Path,
    workspace_root: str | Path,
    runner: JobRunner,
) -> WorkerDaemon | None:
    runtime_manager = getattr(runner, "runtime_manager", None)
    store = getattr(runner, "store", None)
    if runtime_manager is None or store is None:
        return None
    return WorkerDaemon.from_path(
        Path(config_dir) / "worker.yaml",
        runner=runner,
        store=store,
        runtime_manager=runtime_manager,
    )


def serialize_approval(approval: Any) -> dict[str, Any]:
    return approval.model_dump(mode="json")


def build_job_result_payload(record: Any) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "title": record.title,
        "status": record.status.value,
        "target_branch": record.spec.target_branch,
        "repo_path": record.spec.repo_path,
        "workspace_root": record.spec.workspace_root,
        "metadata": record.spec.metadata,
        "failure_count": record.failure_count,
        "same_test_failure_count": record.same_test_failure_count,
        "last_error": record.last_error,
        "runtime_error": getattr(record, "runtime_error", None),
        "provider_status": getattr(record, "provider_status", None),
        "current_phase": getattr(record, "current_phase", None),
        "current_task_id": getattr(record, "current_task_id", None),
        "pending_approval_id": record.pending_approval_id,
        "pending_runtime_issue_id": getattr(record, "pending_runtime_issue_id", None),
        "audit_event_count": len(record.audit_events),
        "outputs": record.outputs,
    }


TERMINAL_JOB_STATUSES = {
    JobStatus.DONE,
    JobStatus.BLOCKED,
    JobStatus.STUCK,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}

WAITING_JOB_STATUSES = {
    JobStatus.WAITING_APPROVAL,
    JobStatus.WAITING_RUNTIME,
    JobStatus.PROVIDER_UNAVAILABLE,
    JobStatus.PAUSED,
}


def _print_job_progress(message: str, *, stream: TextIO) -> None:
    print(message, file=stream, flush=True)


def _format_job_progress(record: JobRecord) -> str:
    parts = [
        f"job={record.job_id}",
        f"status={record.status.value}",
    ]
    if record.current_phase:
        parts.append(f"phase={record.current_phase}")
    if record.current_task_id:
        parts.append(f"task={record.current_task_id}")
    if record.failure_count:
        parts.append(f"failures={record.failure_count}")
    result_key = {
        "tests": "test_run",
        "runtime_prepare": "runtime_prepare",
        "runtime_smoke": "runtime_smoke",
        "acceptance_checks": "acceptance_checks",
    }.get(record.current_phase or "")
    if result_key and isinstance(record.outputs.get(result_key), dict):
        result = record.outputs[result_key]
        if "success" in result:
            parts.append(f"success={bool(result['success'])}")
        command = result.get("command")
        if isinstance(command, list) and command:
            parts.append(f"command={' '.join(str(item) for item in command)}")
    if record.pending_approval_id:
        parts.append(f"approval_id={record.pending_approval_id}")
    if record.pending_runtime_issue_id:
        parts.append(f"runtime_issue_id={record.pending_runtime_issue_id}")
    if record.last_error and record.status in TERMINAL_JOB_STATUSES | WAITING_JOB_STATUSES:
        parts.append(f"detail={record.last_error}")
    return " ".join(parts)


def _emit_new_job_notifications(
    runner: Any,
    *,
    job_id: str,
    seen_count: int,
    stream: TextIO,
) -> int:
    if not hasattr(runner, "get_notifications"):
        return seen_count
    notifications = list(runner.get_notifications(job_id))
    for payload in notifications[seen_count:]:
        kind = str(payload.get("kind", "status"))
        message = (
            payload.get("message")
            or payload.get("reason")
            or payload.get("operation")
            or payload.get("cli_command")
            or kind
        )
        _print_job_progress(f"notification kind={kind} detail={message}", stream=stream)
    return len(notifications)


def run_job_with_live_progress(
    runner: Any,
    spec: JobSpec,
    *,
    quiet: bool = False,
    stream: TextIO | None = None,
    daemon: WorkerDaemon | None = None,
) -> JobRecord:
    supports_durable_progress = daemon is not None and hasattr(runner, "submit") and hasattr(runner, "get")
    supports_oneshot_progress = hasattr(runner, "submit") and hasattr(runner, "run_next_step")
    if quiet:
        if supports_durable_progress:
            record = runner.submit(spec)
            return daemon.run_until_job_settled(record.job_id)
        return runner.run_job(spec)
    if not (supports_durable_progress or supports_oneshot_progress):
        return runner.run_job(spec)
    progress_stream = stream or sys.stderr
    record = runner.submit(spec)
    _print_job_progress(
        " ".join(
            [
                "submitted",
                f"job={record.job_id}",
                f"title={record.title}",
                f"target_branch={record.spec.target_branch}",
            ]
        ),
        stream=progress_stream,
    )
    seen_notifications = _emit_new_job_notifications(
        runner,
        job_id=record.job_id,
        seen_count=0,
        stream=progress_stream,
    )
    while record.status not in TERMINAL_JOB_STATUSES and record.status not in WAITING_JOB_STATUSES:
        if daemon is not None:
            processed = daemon.run_once()
            record = runner.get(record.job_id)
            if (
                record.status not in TERMINAL_JOB_STATUSES
                and record.status not in WAITING_JOB_STATUSES
                and not any(item.job_id == record.job_id for item in processed)
            ):
                time.sleep(max(0.0, float(daemon.config.poll_interval_seconds)))
                record = runner.get(record.job_id)
        else:
            record = runner.run_next_step(record.job_id)
        _print_job_progress(_format_job_progress(record), stream=progress_stream)
        seen_notifications = _emit_new_job_notifications(
            runner,
            job_id=record.job_id,
            seen_count=seen_notifications,
            stream=progress_stream,
        )
    return record


def workspace_runtime_paths(workspace_root: str | Path) -> dict[str, Path]:
    workspace = Path(workspace_root).resolve()
    acos_dir = workspace / ".acos"
    logs_dir = acos_dir / "logs"
    jobs_log_dir = logs_dir / "jobs"
    return {
        "workspace": workspace,
        "acos_dir": acos_dir,
        "logs_dir": logs_dir,
        "jobs_log_dir": jobs_log_dir,
        "runtime_db": acos_dir / "acos.sqlite3",
        "worker_log": logs_dir / "worker.log",
        "worker_out_log": logs_dir / "worker.out.log",
        "worker_err_log": logs_dir / "worker.err.log",
        "audit_log": logs_dir / "audit.log",
        "runtime_log": logs_dir / "runtime.log",
        "pid_file": acos_dir / "worker.pid",
        "launchd_plist": Path.home() / "Library" / "LaunchAgents" / "com.acos.worker.plist",
    }


def ensure_runtime_directories(workspace_root: str | Path) -> dict[str, Path]:
    paths = workspace_runtime_paths(workspace_root)
    paths["acos_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["jobs_log_dir"].mkdir(parents=True, exist_ok=True)
    for key in ("worker_log", "worker_out_log", "worker_err_log", "audit_log", "runtime_log"):
        paths[key].touch(exist_ok=True)
    return paths


def build_health_checker(config_dir: str | Path) -> tuple[ModelRegistry, ProviderHealthChecker]:
    registry, _ = load_registry_and_policy(config_dir)
    return registry, ProviderHealthChecker(registry)


def append_log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (path.read_text(encoding="utf-8") if path.exists() else "") + message.rstrip() + "\n",
        encoding="utf-8",
    )


def build_launchd_plist(
    *,
    workspace_root: str | Path,
    config_dir: str | Path,
    keep_alive: bool = True,
    run_at_load: bool = True,
) -> dict[str, Any]:
    paths = ensure_runtime_directories(workspace_root)
    return {
        "Label": "com.acos.worker",
        "ProgramArguments": [
            "acos",
            "worker",
            "run",
            "--forever",
            "--config-dir",
            str(Path(config_dir).resolve()),
            "--repo",
            str(paths["workspace"]),
        ],
        "WorkingDirectory": str(paths["workspace"]),
        "StandardOutPath": str(paths["worker_out_log"]),
        "StandardErrorPath": str(paths["worker_err_log"]),
        "KeepAlive": keep_alive,
        "RunAtLoad": run_at_load,
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
    if args.command == "debug":
        if args.debug_command == "token-budget":
            if args.job_id:
                runner = load_runner_for_workspace(
                    config_dir=args.config_dir,
                    workspace_root=args.workspace,
                )
                record = runner.get(args.job_id)
                payload = build_token_budget_debug_payload(
                    runner=runner,
                    role=args.role,
                    spec=record.spec,
                    record=record,
                )
                dump_yaml(payload)
                return 0
            spec = load_job_spec_from_file(args.file)
            runner = load_runner_for_workspace(
                config_dir=args.config_dir,
                workspace_root=spec.workspace_root or spec.repo_path,
            )
            payload = build_token_budget_debug_payload(
                runner=runner,
                role=args.role,
                spec=spec,
            )
            dump_yaml(payload)
            return 0
    if args.command == "api":
        uvicorn.run("apps.api.main:app", host=args.host, port=args.port, reload=False)
        return 0
    if args.command == "worker":
        from apps.worker.main import main as worker_main

        forwarded_args = [
            args.worker_action,
            "--config-dir",
            args.config_dir,
            "--repo",
            args.workspace or args.repo,
        ]
        if args.file:
            forwarded_args.extend(["--file", args.file])
        if args.request:
            forwarded_args.extend(["--request", args.request])
        if args.branch:
            forwarded_args.extend(["--branch", args.branch])
        if args.forever:
            forwarded_args.append("--forever")
        return worker_main(forwarded_args)
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
        daemon = build_worker_daemon(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            runner=runner,
        )
        record = run_job_with_live_progress(runner, spec, quiet=args.quiet, daemon=daemon)
        dump_yaml(build_job_result_payload(record))
        return 0 if record.status.value == "done" else 1
    if args.command == "check-provider":
        _registry, checker = build_health_checker(args.config_dir)
        health = checker.check_provider(args.provider)
        dump_yaml(health.model_dump(mode="json"))
        return 0 if health.status.value == "ok" else 1
    if args.command == "check-model":
        _registry, checker = build_health_checker(args.config_dir)
        health = checker.check_model(args.model)
        dump_yaml(health.model_dump(mode="json"))
        return 0 if health.status.value == "ok" else 1
    if args.command == "runtime":
        runner = load_runner_for_workspace(
            config_dir=args.config_dir,
            workspace_root=args.workspace,
        )
        if args.runtime_action == "check":
            resumed = runner.runtime_manager.maybe_resume_waiting_jobs() if runner.runtime_manager else []
            dump_yaml(
                {
                    "ok": True,
                    "resumed_jobs": [build_job_result_payload(item) for item in resumed],
                    "runtime_issues": [
                        issue.model_dump(mode="json")
                        for issue in runner.store.list_runtime_issues()
                    ],
                }
            )
            return 0
        if args.runtime_action == "watch":
            iterations = 0
            while True:
                resumed = runner.runtime_manager.maybe_resume_waiting_jobs() if runner.runtime_manager else []
                dump_yaml(
                    {
                        "resumed_jobs": [build_job_result_payload(item) for item in resumed],
                        "runtime_issues": [
                            issue.model_dump(mode="json")
                            for issue in runner.store.list_runtime_issues()
                        ],
                    }
                )
                iterations += 1
                if args.max_iterations is not None and iterations >= args.max_iterations:
                    return 0
                time.sleep(args.poll_interval)
        dump_yaml(
            {
                "runtime_issues": [
                    issue.model_dump(mode="json")
                    for issue in runner.store.list_runtime_issues()
                ],
                "waiting_jobs": [
                    build_job_result_payload(item)
                    for item in runner.list_jobs(
                        statuses=[
                            item
                            for item in (
                                JobStatus.WAITING_RUNTIME,
                                JobStatus.PROVIDER_UNAVAILABLE,
                                JobStatus.RETRYING_PROVIDER,
                            )
                        ]
                    )
                ],
            }
        )
        return 0
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
        if args.jobs_command == "submit":
            spec = load_job_spec_from_file(args.file)
            runner = load_runner_for_workspace(
                config_dir=args.config_dir,
                workspace_root=spec.workspace_root or spec.repo_path,
            )
            record = runner.submit(spec)
            dump_yaml({"ok": True, "job": build_job_result_payload(record)})
            return 0
        if args.jobs_command == "list":
            dump_yaml({"jobs": [build_job_result_payload(item) for item in runner.list_jobs()]})
            return 0
        if args.jobs_command in {"show", "status"}:
            try:
                record = runner.get(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            payload = build_job_result_payload(record)
            if args.jobs_command == "status":
                payload = {
                    "job_id": record.job_id,
                    "status": record.status.value,
                    "current_phase": record.current_phase,
                    "last_error": record.last_error,
                    "runtime_error": record.runtime_error,
                    "pending_approval_id": record.pending_approval_id,
                    "pending_runtime_issue_id": record.pending_runtime_issue_id,
                }
            dump_yaml({"ok": True, "job": payload})
            return 0
        if args.jobs_command == "watch":
            iterations = 0
            while True:
                try:
                    record = runner.get(args.job_id)
                except KeyError:
                    dump_yaml({"ok": False, "error": "job not found"})
                    return 1
                dump_yaml({"job": build_job_result_payload(record)})
                iterations += 1
                if record.status in {
                    JobStatus.DONE,
                    JobStatus.BLOCKED,
                    JobStatus.STUCK,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                    JobStatus.WAITING_APPROVAL,
                    JobStatus.WAITING_RUNTIME,
                    JobStatus.PROVIDER_UNAVAILABLE,
                    JobStatus.PAUSED,
                }:
                    return 0
                if args.max_iterations is not None and iterations >= args.max_iterations:
                    return 0
                time.sleep(args.poll_interval)
        if args.jobs_command == "pause":
            try:
                record = runner.pause_job(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            dump_yaml({"ok": True, "job": build_job_result_payload(record)})
            return 0
        if args.jobs_command == "resume":
            try:
                record = runner.resume_job(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            dump_yaml({"ok": True, "job": build_job_result_payload(record)})
            return 0
        if args.jobs_command == "cancel":
            try:
                record = runner.cancel_job(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            dump_yaml({"ok": True, "job": build_job_result_payload(record)})
            return 0
        if args.jobs_command == "logs":
            try:
                runner.get(args.job_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "job not found"})
                return 1
            dump_yaml(
                {
                    "ok": True,
                    "notifications": runner.get_notifications(args.job_id),
                    "events": [event.model_dump(mode="json") for event in runner.get_events(args.job_id)],
                }
            )
            return 0
    if args.command == "daemon":
        workspace_root = args.workspace
        runner = load_runner_for_workspace(
            config_dir=args.config_dir,
            workspace_root=workspace_root,
        )
        if runner.runtime_manager is None:
            dump_yaml({"ok": False, "error": "runtime manager is not configured"})
            return 1
        daemon_paths = ensure_runtime_directories(workspace_root)
        worker = WorkerDaemon.from_path(
            Path(args.config_dir) / "worker.yaml",
            runner=runner,
            store=runner.store,
            runtime_manager=runner.runtime_manager,
        )
        if args.daemon_action == "start":
            if args.detach:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "apps.worker.main",
                        "run",
                        "--forever",
                        "--config-dir",
                        str(Path(args.config_dir).resolve()),
                        "--repo",
                        str(Path(workspace_root).resolve()),
                    ],
                    cwd=str(Path(workspace_root).resolve()),
                    stdout=daemon_paths["worker_out_log"].open("a", encoding="utf-8"),
                    stderr=daemon_paths["worker_err_log"].open("a", encoding="utf-8"),
                    start_new_session=True,
                )
                daemon_paths["pid_file"].write_text(str(process.pid), encoding="utf-8")
                dump_yaml({"ok": True, "pid": process.pid, "mode": "detached"})
                return 0
            dump_yaml({"ok": True, "mode": "foreground"})
            worker.run_forever()
            return 0
        if args.daemon_action == "stop":
            if daemon_paths["pid_file"].exists():
                pid = int(daemon_paths["pid_file"].read_text(encoding="utf-8").strip())
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                daemon_paths["pid_file"].unlink(missing_ok=True)
                dump_yaml({"ok": True, "stopped_pid": pid})
                return 0
            dump_yaml({"ok": True, "message": "no running daemon pid file"})
            return 0
        if args.daemon_action == "logs":
            content = ""
            for candidate in (daemon_paths["worker_out_log"], daemon_paths["worker_log"], daemon_paths["worker_err_log"]):
                if candidate.exists():
                    content += candidate.read_text(encoding="utf-8")
            dump_yaml({"ok": True, "logs": content[-20000:]})
            return 0
        if args.daemon_action == "install-launchd":
            plist_payload = build_launchd_plist(
                workspace_root=workspace_root,
                config_dir=args.config_dir,
            )
            daemon_paths["launchd_plist"].parent.mkdir(parents=True, exist_ok=True)
            daemon_paths["launchd_plist"].write_bytes(plistlib.dumps(plist_payload))
            dump_yaml(
                {
                    "ok": True,
                    "plist_path": str(daemon_paths["launchd_plist"]),
                    "launchctl_bootstrap": f"launchctl bootstrap gui/$(id -u) {daemon_paths['launchd_plist']}",
                    "launchctl_bootout": f"launchctl bootout gui/$(id -u) {daemon_paths['launchd_plist']}",
                    "launchctl_kickstart": "launchctl kickstart -k gui/$(id -u)/com.acos.worker",
                }
            )
            return 0
        if args.daemon_action == "uninstall-launchd":
            daemon_paths["launchd_plist"].unlink(missing_ok=True)
            dump_yaml({"ok": True, "plist_removed": str(daemon_paths["launchd_plist"])})
            return 0
        dump_yaml(
            {
                "ok": True,
                "pid_file_exists": daemon_paths["pid_file"].exists(),
                "launchd_plist_exists": daemon_paths["launchd_plist"].exists(),
                "heartbeats": [
                    heartbeat.model_dump(mode="json")
                    for heartbeat in runner.store.list_worker_heartbeats()
                ],
            }
        )
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
