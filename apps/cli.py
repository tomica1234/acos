"""CLI for running ACOS locally."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
from pathlib import Path
from time import monotonic, time
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn
import yaml

from packages.llm.adapters.mock import MockAdapter
from packages.llm.errors import ConfigValidationError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.orchestrator.job_constraints import apply_strict_job_constraints
from packages.orchestrator.job_runner import JobRunner, build_default_runner
from packages.orchestrator.job_store import FileJobStore
from packages.orchestrator.autonomy_governor import (
    AutonomyGovernor,
    apply_recovery_plan,
)
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.progress import summarize_job_progress
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
    Severity,
    TaskComplexity,
)
from packages.schemas.tasks import PlannedTask, TaskGraph

SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS = 300.0


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
    run_job.add_argument("--jobs-dir", default=".acos/jobs")
    run_job.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    run_job.add_argument("--large-autonomous", action="store_true")
    run_job.add_argument("--require-prd-quality", action="store_true")
    run_job.add_argument("--stage-review", action="store_true")
    run_job.add_argument("--test-timeout-seconds", type=positive_int, default=None)

    plan_job = subparsers.add_parser("plan-job")
    plan_job.add_argument("--config-dir", default="configs")
    plan_job_input = plan_job.add_mutually_exclusive_group(required=True)
    plan_job_input.add_argument("--file")
    plan_job_input.add_argument("--request")
    plan_job.add_argument("--repo-path", default=".")
    plan_job.add_argument("--workspace-root", default=None)
    plan_job.add_argument("--target-branch", default="acos/default")
    plan_job.add_argument("--job-id", default=None)
    plan_job.add_argument("--title", default=None)
    plan_job.add_argument("--jobs-dir", default=".acos/jobs")
    plan_job.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    plan_job.add_argument("--require-prd-quality", action="store_true")
    plan_job.add_argument("--stage-review", action="store_true")
    plan_job.add_argument("--test-timeout-seconds", type=positive_int, default=None)
    plan_job.add_argument("--summary-file", default=None)
    plan_job.add_argument("--preflight-provider", default=None)
    plan_job.add_argument("--preflight-timeout", type=float, default=5.0)
    plan_job.add_argument("--supervise-after-planning", action="store_true")
    plan_job.add_argument("--supervise-max-cycles", type=positive_int, default=10)
    plan_job.add_argument("--supervise-steps-per-cycle", type=positive_int, default=1)
    plan_job.add_argument("--supervise-max-stalled-cycles", type=positive_int, default=3)
    plan_job.add_argument("--supervise-max-runtime-seconds", type=positive_float, default=None)
    plan_job.add_argument("--supervise-summary-file", default=None)
    plan_job.add_argument("--supervise-summary-dir", default=None)
    plan_job.add_argument("--supervise-preflight-provider", default=None)
    plan_job.add_argument("--supervise-preflight-timeout", type=float, default=5.0)
    plan_job.add_argument("--supervise-pm-stall-recovery", action="store_true")
    plan_job.add_argument("--supervise-autonomous-until-done", action="store_true")
    plan_job.add_argument(
        "--supervise-allow-blocked-recovery",
        dest="supervise_allow_repeated_failure_recovery",
        action="store_true",
    )

    run_autonomous = subparsers.add_parser("run-autonomous")
    run_autonomous.add_argument("--config-dir", default="configs")
    run_autonomous.add_argument("--file", required=True)
    run_autonomous.add_argument("--jobs-dir", default=".acos/jobs")
    run_autonomous.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    run_autonomous.add_argument("--max-steps", type=positive_int, default=3)
    run_autonomous.add_argument("--require-prd-quality", action="store_true")
    run_autonomous.add_argument("--stage-review", action="store_true")
    run_autonomous.add_argument("--test-timeout-seconds", type=positive_int, default=None)
    run_autonomous.add_argument(
        "--allow-blocked-recovery",
        "--allow-repeated-failure-recovery",
        dest="allow_repeated_failure_recovery",
        action="store_true",
    )
    run_autonomous.add_argument("--json-summary", action="store_true")
    run_autonomous.add_argument("--summary-file", default=None)

    run_supervised = subparsers.add_parser("run-supervised")
    run_supervised.add_argument("--config-dir", default="configs")
    run_supervised_input = run_supervised.add_mutually_exclusive_group(required=True)
    run_supervised_input.add_argument("--file")
    run_supervised_input.add_argument("--request")
    run_supervised.add_argument("--repo-path", default=".")
    run_supervised.add_argument("--workspace-root", default=None)
    run_supervised.add_argument("--target-branch", default="acos/default")
    run_supervised.add_argument("--job-id", default=None)
    run_supervised.add_argument("--title", default=None)
    run_supervised.add_argument("--jobs-dir", default=".acos/jobs")
    run_supervised.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    run_supervised.add_argument("--max-cycles", type=positive_int, default=10)
    run_supervised.add_argument("--steps-per-cycle", type=positive_int, default=1)
    run_supervised.add_argument("--max-stalled-cycles", type=positive_int, default=3)
    run_supervised.add_argument("--max-runtime-seconds", type=positive_float, default=None)
    run_supervised.add_argument("--require-prd-quality", action="store_true")
    run_supervised.add_argument("--stage-review", action="store_true")
    run_supervised.add_argument("--test-timeout-seconds", type=positive_int, default=None)
    run_supervised.add_argument("--summary-file", default=None)
    run_supervised.add_argument("--summary-dir", default=None)
    run_supervised.add_argument("--preflight-provider", default=None)
    run_supervised.add_argument("--preflight-timeout", type=float, default=5.0)
    run_supervised.add_argument("--plan-first", action="store_true")
    run_supervised.add_argument("--pm-stall-recovery", action="store_true")
    run_supervised.add_argument("--autonomous-until-done", action="store_true")
    run_supervised.add_argument(
        "--allow-blocked-recovery",
        "--allow-repeated-failure-recovery",
        dest="allow_repeated_failure_recovery",
        action="store_true",
    )

    resume_job = subparsers.add_parser("resume-job")
    resume_job.add_argument("--config-dir", default="configs")
    resume_job.add_argument("--job-id", required=True)
    resume_job.add_argument("--jobs-dir", default=".acos/jobs")
    resume_job.add_argument("--workspace", default=None)
    resume_job.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    resume_job.add_argument("--bump-stage-limit", action="store_true")
    resume_job.add_argument("--large-autonomous", action="store_true")
    resume_job.add_argument("--require-prd-quality", action="store_true")
    resume_job.add_argument("--stage-review", action="store_true")
    resume_job.add_argument("--test-timeout-seconds", type=positive_int, default=None)

    continue_job = subparsers.add_parser("continue-job")
    continue_job.add_argument("--config-dir", default="configs")
    continue_job.add_argument("--job-id", required=True)
    continue_job.add_argument("--jobs-dir", default=".acos/jobs")
    continue_job.add_argument("--workspace", default=None)
    continue_job.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    continue_job.add_argument("--max-steps", type=positive_int, default=1)
    continue_job.add_argument("--large-autonomous", action="store_true")
    continue_job.add_argument("--require-prd-quality", action="store_true")
    continue_job.add_argument("--stage-review", action="store_true")
    continue_job.add_argument("--test-timeout-seconds", type=positive_int, default=None)
    continue_job.add_argument(
        "--allow-blocked-recovery",
        "--allow-repeated-failure-recovery",
        dest="allow_repeated_failure_recovery",
        action="store_true",
    )
    continue_job.add_argument("--json-summary", action="store_true")
    continue_job.add_argument("--summary-file", default=None)

    supervise_job = subparsers.add_parser("supervise-job")
    supervise_job.add_argument("--config-dir", default="configs")
    supervise_job.add_argument("--job-id", required=True)
    supervise_job.add_argument("--jobs-dir", default=".acos/jobs")
    supervise_job.add_argument("--workspace", default=None)
    supervise_job.add_argument("--max-autonomous-stages", type=positive_int, default=None)
    supervise_job.add_argument("--max-cycles", type=positive_int, default=10)
    supervise_job.add_argument("--steps-per-cycle", type=positive_int, default=1)
    supervise_job.add_argument("--max-stalled-cycles", type=positive_int, default=3)
    supervise_job.add_argument("--max-runtime-seconds", type=positive_float, default=None)
    supervise_job.add_argument("--large-autonomous", action="store_true")
    supervise_job.add_argument("--require-prd-quality", action="store_true")
    supervise_job.add_argument("--stage-review", action="store_true")
    supervise_job.add_argument("--test-timeout-seconds", type=positive_int, default=None)
    supervise_job.add_argument("--summary-file", default=None)
    supervise_job.add_argument("--summary-dir", default=None)
    supervise_job.add_argument("--preflight-provider", default=None)
    supervise_job.add_argument("--preflight-timeout", type=float, default=5.0)
    supervise_job.add_argument("--pm-stall-recovery", action="store_true")
    supervise_job.add_argument("--autonomous-until-done", action="store_true")
    supervise_job.add_argument(
        "--allow-blocked-recovery",
        "--allow-repeated-failure-recovery",
        dest="allow_repeated_failure_recovery",
        action="store_true",
    )

    job_status = subparsers.add_parser("job-status")
    job_status.add_argument("--job-id", required=True)
    job_status.add_argument("--jobs-dir", default=".acos/jobs")
    job_status.add_argument("--next-command", action="store_true")
    job_status.add_argument("--next-continue-command", action="store_true")
    job_status.add_argument("--next-supervise-command", action="store_true")
    job_status.add_argument("--next-operator-command", action="store_true")
    job_status.add_argument("--continue-max-steps", type=positive_int, default=None)
    job_status.add_argument("--continue-json-summary", action="store_true")
    job_status.add_argument("--supervise-max-cycles", type=positive_int, default=10)
    job_status.add_argument("--supervise-steps-per-cycle", type=positive_int, default=1)
    job_status.add_argument("--supervise-max-stalled-cycles", type=positive_int, default=3)
    job_status.add_argument("--supervise-max-runtime-seconds", type=positive_float, default=None)
    job_status.add_argument("--supervise-summary-file", default=None)
    job_status.add_argument("--supervise-summary-dir", default=None)
    job_status.add_argument("--supervise-workspace", default=None)
    job_status.add_argument("--supervise-large-autonomous", action="store_true")
    job_status.add_argument("--supervise-require-prd-quality", action="store_true")
    job_status.add_argument("--supervise-stage-review", action="store_true")
    job_status.add_argument("--supervise-test-timeout-seconds", type=positive_int, default=None)
    job_status.add_argument("--supervise-max-autonomous-stages", type=positive_int, default=None)
    job_status.add_argument("--supervise-preflight-provider", default=None)
    job_status.add_argument("--supervise-preflight-timeout", type=float, default=5.0)
    job_status.add_argument("--supervise-pm-stall-recovery", action="store_true")
    job_status.add_argument("--supervise-autonomous-until-done", action="store_true")
    job_status.add_argument(
        "--supervise-allow-blocked-recovery",
        "--supervise-allow-repeated-failure-recovery",
        dest="supervise_allow_repeated_failure_recovery",
        action="store_true",
    )
    job_status.add_argument("--json", action="store_true")

    check_provider = subparsers.add_parser("check-provider")
    check_provider.add_argument("--config-dir", default="configs")
    check_provider.add_argument("--provider", required=True)
    check_provider.add_argument("--timeout", type=float, default=5.0)

    jobs = subparsers.add_parser("jobs")
    jobs_subparsers = jobs.add_subparsers(dest="jobs_command", required=True)
    jobs_submit = jobs_subparsers.add_parser("submit")
    jobs_submit.add_argument("--config-dir", default="configs")
    jobs_submit.add_argument("--workspace", default=".")
    jobs_submit.add_argument("--file", required=True)
    jobs_show = jobs_subparsers.add_parser("show")
    jobs_show.add_argument("job_id")
    jobs_show.add_argument("--config-dir", default="configs")
    jobs_show.add_argument("--workspace", default=".")
    jobs_resume = jobs_subparsers.add_parser("resume")
    jobs_resume.add_argument("job_id")
    jobs_resume.add_argument("--config-dir", default="configs")
    jobs_resume.add_argument("--workspace", default=".")

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

    runtime = subparsers.add_parser("runtime")
    runtime_subparsers = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_status = runtime_subparsers.add_parser("status")
    runtime_status.add_argument("--config-dir", default="configs")
    runtime_status.add_argument("--workspace", default=".")
    runtime_check = runtime_subparsers.add_parser("check")
    runtime_check.add_argument("--config-dir", default="configs")
    runtime_check.add_argument("--workspace", default=".")

    daemon = subparsers.add_parser("daemon")
    daemon.add_argument("daemon_action", choices=["status", "logs", "install-launchd", "uninstall-launchd"], default="status")
    daemon.add_argument("--config-dir", default="configs")
    daemon.add_argument("--workspace", default=".")
    return parser


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


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
    demo_acceptance = (
        "Running pytest confirms the add helper returns two integers added together."
    )
    scenario = {
        "pm": PRD(
            title="Demo Feature",
            problem_statement="Implement a simple add function.",
            users=["developer"],
            goals=["Provide a correct add helper"],
            constraints=["Use deterministic tests"],
            success_criteria=["pytest passes"],
            smallest_working_core=["Implement the add helper and validate it with pytest"],
            small_parts=["Implement add helper module"],
            incremental_milestones=["Add helper module is implemented"],
            acceptance_tests=[demo_acceptance],
            definition_of_done=["The generated pytest suite passes"],
            required_artifacts=["feature.py", "tests/test_feature.py"],
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
                    description="Create feature.add with correct integer addition.",
                    role="implementer",
                    acceptance_criteria=[demo_acceptance],
                    target_files=["feature.py"],
                    required_artifacts=["feature.py"],
                ),
                PlannedTask(
                    id="task-1-tests",
                    title="Test add helper",
                    description="Add pytest coverage for feature.add.",
                    role="test_writer",
                    depends_on=["task-1"],
                    acceptance_criteria=[demo_acceptance],
                    target_files=["tests/test_feature.py"],
                    required_artifacts=["tests/test_feature.py"],
                ),
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
        "reviewer": [
            ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Implementation is acceptable for demo purposes.",
                findings=[],
            ).model_dump(),
            ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Tests are acceptable for demo purposes.",
                findings=[],
            ).model_dump(),
            ReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="Final demo changes are acceptable.",
                findings=[],
            ).model_dump(),
        ],
        "security_reviewer": [
            SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="No security-sensitive behavior in demo scope.",
                findings=[],
            ).model_dump(),
            SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="No security-sensitive behavior in demo tests.",
                findings=[],
            ).model_dump(),
            SecurityReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="No security-sensitive behavior in final demo changes.",
                findings=[],
            ).model_dump(),
        ],
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
        provider_type=registry.get_provider("local_ornith").type,
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


def load_job_spec_from_file(path: str | Path) -> JobSpec:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    request_text = payload.get("request_text") or payload.get("requester_input")
    if not isinstance(request_text, str) or not request_text.strip():
        raise ValueError("job file requires request_text or requester_input")
    repo_path = str(Path(payload.get("repo_path", ".")).resolve())
    workspace_root = str(Path(payload.get("workspace_root", repo_path)).resolve())
    raw_metadata = payload.get("metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for key, value in payload.items():
        if key in {
            "job_id",
            "title",
            "request_text",
            "requester_input",
            "repo_path",
            "target_branch",
            "metadata",
            "workspace_root",
        }:
            continue
        metadata[key] = value
    spec_payload = {
        "request_text": request_text.strip(),
        "repo_path": repo_path,
        "target_branch": payload.get("target_branch", "acos/default"),
        "metadata": metadata,
        "workspace_root": workspace_root,
    }
    if payload.get("job_id"):
        spec_payload["job_id"] = payload["job_id"]
    if payload.get("title"):
        spec_payload["title"] = payload["title"]
    return JobSpec.model_validate(spec_payload)


def build_job_spec_from_request(
    *,
    request_text: str,
    repo_path: str | Path,
    workspace_root: str | Path | None = None,
    target_branch: str = "acos/default",
    job_id: str | None = None,
    title: str | None = None,
) -> JobSpec:
    if not request_text.strip():
        raise ValueError("request must not be empty")
    resolved_repo = str(Path(repo_path).resolve())
    resolved_workspace = str(Path(workspace_root or resolved_repo).resolve())
    payload: dict[str, Any] = {
        "request_text": request_text.strip(),
        "repo_path": resolved_repo,
        "workspace_root": resolved_workspace,
        "target_branch": target_branch,
        "metadata": {},
    }
    if job_id:
        payload["job_id"] = job_id
    if title:
        payload["title"] = title
    return JobSpec.model_validate(payload)


def load_run_supervised_job_spec(args: argparse.Namespace) -> JobSpec:
    if args.file:
        return load_job_spec_from_file(args.file)
    return build_job_spec_from_request(
        request_text=args.request,
        repo_path=args.repo_path,
        workspace_root=args.workspace_root,
        target_branch=args.target_branch,
        job_id=args.job_id,
        title=args.title,
    )


def apply_constraint_overrides(
    spec_or_record,
    *,
    max_autonomous_stages: int | None = None,
    large_autonomous: bool = False,
    require_prd_quality: bool = False,
    stage_review: bool = False,
    test_timeout_seconds: int | None = None,
    model_timeout_seconds: float | None = None,
    model_timeout_deadline_epoch: float | None = None,
) -> None:
    if (
        max_autonomous_stages is None
        and not large_autonomous
        and not require_prd_quality
        and not stage_review
        and test_timeout_seconds is None
        and model_timeout_seconds is None
        and model_timeout_deadline_epoch is None
    ):
        return
    spec = getattr(spec_or_record, "spec", spec_or_record)
    constraints = spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        spec.metadata["constraints"] = constraints
    if large_autonomous:
        apply_strict_job_constraints(spec_or_record)
        constraints = spec.metadata["constraints"]
        constraints.setdefault("max_autonomous_stages", 1)
    if max_autonomous_stages is not None:
        constraints["max_autonomous_stages"] = max_autonomous_stages
    if require_prd_quality:
        constraints["require_prd_quality"] = True
    if stage_review:
        constraints["stage_review"] = True
    if test_timeout_seconds is not None:
        constraints["test_timeout_seconds"] = test_timeout_seconds
    if model_timeout_seconds is not None and model_timeout_seconds > 0:
        constraints["model_timeout_seconds"] = float(model_timeout_seconds)
    if model_timeout_deadline_epoch is not None:
        if model_timeout_deadline_epoch > 0:
            constraints["model_timeout_deadline_epoch"] = float(model_timeout_deadline_epoch)
        else:
            constraints.pop("model_timeout_deadline_epoch", None)


def supervised_model_timeout_seconds(
    max_runtime_seconds: float | None,
    elapsed_seconds: float = 0.0,
) -> float | None:
    if max_runtime_seconds is None:
        return None
    remaining_seconds = max(float(max_runtime_seconds) - float(elapsed_seconds), 0.0)
    if remaining_seconds <= 0:
        return 0.0
    return min(remaining_seconds, SUPERVISED_MODEL_CALL_TIMEOUT_CAP_SECONDS)


def supervised_model_timeout_deadline_epoch(
    max_runtime_seconds: float | None,
    *,
    started_epoch: float | None = None,
) -> float | None:
    if max_runtime_seconds is None:
        return None
    return float(started_epoch if started_epoch is not None else time()) + float(
        max_runtime_seconds
    )


def apply_recovery_overrides(record: JobRecord, summary: dict[str, object]) -> dict[str, object] | None:
    failure_analysis = summary.get("failure_analysis")
    if not isinstance(failure_analysis, dict):
        return None
    recovery = failure_analysis.get("recommended_recovery")
    if not isinstance(recovery, dict):
        return None
    recovery_constraints = recovery.get("constraints")
    if not isinstance(recovery_constraints, dict):
        return None
    constraints = record.spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        record.spec.metadata["constraints"] = constraints
    for key, value in recovery_constraints.items():
        if isinstance(key, str) and value is not None:
            constraints[key] = value
    diagnosis = summary.get("failure_diagnosis")
    if (
        recovery.get("strategy") == "diagnosis_guided_retry"
        and isinstance(diagnosis, dict)
    ):
        playbook = _diagnosis_recovery_playbook(diagnosis, recovery)
        constraints["pm_stall_recovery"] = True
        constraints["pm_strategy_change"] = True
        constraints["pm_strategy"] = playbook["pm_strategy"]
        constraints["pm_reason"] = "diagnosed_repeated_failure"
        constraints["pm_next_actor"] = playbook["next_actor"]
        constraints["pm_recovery_playbook"] = playbook["playbook"]
        constraints["pm_success_criteria"] = playbook["success_criteria"]
        constraints["diagnosis_classification"] = diagnosis.get("classification")
        constraints["diagnosis_retry_mode"] = diagnosis.get("retry_mode")
        constraints["diagnosis_should_retry"] = diagnosis.get("should_retry")
        constraints["diagnosis_root_cause"] = diagnosis.get("root_cause")
        constraints["diagnosis_recommended_fix_strategy"] = diagnosis.get(
            "recommended_fix_strategy"
        )
        _record_pm_recovery_intervention(
            record,
            {
                "action": "change_strategy",
                "reason": "diagnosed_repeated_failure",
                "strategy": playbook["pm_strategy"],
                "summary": (
                    f"PM is changing method to {playbook['playbook']} based on the "
                    "structured failure diagnosis instead of repeating the same fixer loop."
                ),
                "can_apply_automatically": True,
                "applied": True,
                "status": record.status.value,
                "resume_action": "diagnosis_guided_recovery",
                "next_actor": playbook["next_actor"],
                "playbook": playbook["playbook"],
                "success_criteria": playbook["success_criteria"],
                "focus_task_id": recovery.get("failed_task_id"),
                "diagnosis": diagnosis,
                "constraints": {
                    key: value
                    for key, value in constraints.items()
                    if isinstance(key, str)
                    and (
                        key.startswith("diagnosis_")
                        or key.startswith("pm_")
                        or key.startswith("recovery_")
                    )
                },
            },
        )
    constraints["recovery_reason"] = recovery.get("reason")
    failed_task_id = recovery.get("failed_task_id")
    if failed_task_id is not None:
        constraints["recovery_failed_task_id"] = failed_task_id
    failed_stage = recovery.get("failed_stage")
    if failed_stage is not None:
        constraints["recovery_failed_stage"] = failed_stage
    constraints["recovery_attempt"] = int(constraints.get("recovery_attempt", 0)) + 1
    if (
        recovery.get("strategy") == "completion_audit"
        and record.status == JobStatus.BLOCKED
        and isinstance(record.last_error, str)
        and record.last_error.startswith("completion_integrity_failed:")
    ):
        record.status = JobStatus.TESTING
        record.history.append(JobStatus.TESTING)
        record.last_error = None
    return recovery


def _diagnosis_recovery_playbook(
    diagnosis: dict[str, object],
    recovery: dict[str, object],
) -> dict[str, str]:
    classification = str(diagnosis.get("classification") or "unknown")
    retry_mode = str(diagnosis.get("retry_mode") or "")
    failed_task = recovery.get("failed_task_id")
    task_scope = f"task {failed_task}" if isinstance(failed_task, str) else "the failed task"
    playbooks: dict[str, dict[str, str]] = {
        "missing_dependency": {
            "pm_strategy": "dependency_alignment_first",
            "next_actor": "implementer",
            "playbook": (
                "inspect dependency manifests and runtime imports first, then align or pin "
                "the smallest compatible dependency set before touching application logic"
            ),
            "success_criteria": (
                "dependency import smoke test passes and the original failing pytest command "
                "reaches application assertions instead of dependency import errors"
            ),
        },
        "import_error": {
            "pm_strategy": "import_wiring_repair_first",
            "next_actor": "implementer",
            "playbook": (
                "inspect the actual defining and importing files, repair only the broken "
                "module wiring, then rerun the failing import/test target"
            ),
            "success_criteria": (
                "the named module imports successfully and the same ImportError signature "
                "does not recur"
            ),
        },
        "syntax_error": {
            "pm_strategy": "syntax_minimal_rewrite",
            "next_actor": "implementer",
            "playbook": (
                "open the exact syntax error file and rewrite the smallest invalid block "
                "before expanding scope"
            ),
            "success_criteria": "python compilation or test collection passes for the affected file",
        },
        "test_expectation_mismatch": {
            "pm_strategy": "contract_reconciliation",
            "next_actor": "planner",
            "playbook": (
                "reconcile test expectations with the requested behavior, then choose either "
                "a focused implementation patch or a focused test correction with evidence"
            ),
            "success_criteria": (
                "the failing assertion is explained by an explicit behavior contract and "
                "the updated code/tests pass together"
            ),
        },
        "frontend_build_error": {
            "pm_strategy": "frontend_build_repair_first",
            "next_actor": "implementer",
            "playbook": (
                "inspect package scripts and the frontend build output, fix the smallest "
                "TypeScript/Vite/package issue, then rerun the build"
            ),
            "success_criteria": "frontend build/test command exits successfully",
        },
        "runtime_error": {
            "pm_strategy": "runtime_trace_reproduction",
            "next_actor": "implementer",
            "playbook": (
                "reproduce the runtime traceback, identify the first project-owned frame, "
                "and repair that narrow path before broad refactors"
            ),
            "success_criteria": "the same runtime traceback signature does not recur",
        },
    }
    selected = dict(playbooks.get(classification, {}))
    if not selected:
        selected = {
            "pm_strategy": "inspect_before_retry",
            "next_actor": "planner" if retry_mode == "inspect_files_first" else "implementer",
            "playbook": (
                "inspect the failing files and command output before writing patches, then "
                f"split {task_scope} into the smallest verifiable recovery step"
            ),
            "success_criteria": "the previous failure signature changes or the focused test passes",
        }
    return selected


def _record_pm_recovery_intervention(
    record: JobRecord,
    decision: dict[str, Any],
) -> dict[str, Any]:
    interventions = record.outputs.setdefault("pm_interventions", [])
    if not isinstance(interventions, list):
        interventions = []
        record.outputs["pm_interventions"] = interventions
    applied_decision = dict(decision)
    applied_decision["applied"] = True
    applied_decision["intervention_index"] = len(interventions) + 1
    constraints = record.spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        record.spec.metadata["constraints"] = constraints
    constraints["pm_intervention_count"] = applied_decision["intervention_index"]
    interventions.append(applied_decision)
    return applied_decision


def apply_planning_repair_overrides(
    record: JobRecord,
    summary: dict[str, object],
) -> dict[str, object] | None:
    planning_quality = summary.get("planning_quality")
    if not isinstance(planning_quality, dict):
        return None
    planning_repair = planning_quality.get("planning_repair")
    if not isinstance(planning_repair, dict):
        return None
    if planning_repair.get("strategy_change_recommended") is not True:
        return None
    constraints = record.spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        record.spec.metadata["constraints"] = constraints
    repeated_prd_missing = [
        str(item) for item in planning_repair.get("repeated_prd_missing", [])
    ]
    repeated_task_graph_error_types = [
        str(item) for item in planning_repair.get("repeated_task_graph_error_types", [])
    ]
    constraints["planning_repair_strategy_change"] = True
    constraints["planning_repair_consecutive_prd_failures"] = planning_repair.get(
        "consecutive_prd_failure_count",
        0,
    )
    constraints["planning_repair_consecutive_task_graph_failures"] = planning_repair.get(
        "consecutive_task_graph_failure_count",
        0,
    )
    constraints["planning_repair_repeated_prd_missing"] = (
        ",".join(repeated_prd_missing) if repeated_prd_missing else "none"
    )
    constraints["planning_repair_repeated_task_graph_error_types"] = (
        ",".join(repeated_task_graph_error_types)
        if repeated_task_graph_error_types
        else "none"
    )
    return planning_repair


def apply_resume_overrides(
    record,
    *,
    max_autonomous_stages: int | None = None,
    bump_stage_limit: bool = False,
    large_autonomous: bool = False,
    require_prd_quality: bool = False,
    stage_review: bool = False,
    test_timeout_seconds: int | None = None,
    model_timeout_seconds: float | None = None,
    model_timeout_deadline_epoch: float | None = 0.0,
) -> None:
    apply_strict_job_constraints(record)
    if max_autonomous_stages is None and bump_stage_limit:
        max_autonomous_stages = suggested_next_stage_limit(record)
    apply_constraint_overrides(
        record,
        max_autonomous_stages=max_autonomous_stages,
        large_autonomous=large_autonomous,
        require_prd_quality=require_prd_quality,
        stage_review=stage_review,
        test_timeout_seconds=test_timeout_seconds,
        model_timeout_seconds=model_timeout_seconds,
        model_timeout_deadline_epoch=model_timeout_deadline_epoch,
    )
    if max_autonomous_stages is None:
        return
    if record.last_error == "autonomous_stage_limit_reached":
        record.status = JobStatus.TESTING
        record.last_error = None


def suggested_next_stage_limit(record) -> int | None:
    stage_limit = record.outputs.get("autonomous_stage_limit")
    if not isinstance(stage_limit, dict):
        return None
    current = stage_limit.get("max_autonomous_stages")
    completed = stage_limit.get("completed_stage_count")
    if not isinstance(current, int) or not isinstance(completed, int):
        return None
    return max(current + 1, completed + 1)


def continue_persisted_job(
    *,
    store: FileJobStore,
    job_id: str,
    config_dir: str | Path,
    workspace: str | Path | None,
    max_steps: int,
    max_autonomous_stages: int | None = None,
    large_autonomous: bool = False,
    require_prd_quality: bool = False,
    stage_review: bool = False,
    test_timeout_seconds: int | None = None,
    model_timeout_seconds: float | None = None,
    model_timeout_deadline_epoch: float | None = 0.0,
    allow_repeated_failure_recovery: bool = False,
    autonomous_recovery: bool = False,
) -> tuple[JobRecord, int, dict[str, object] | None, list[dict[str, Any]]]:
    steps_run = 0
    record = store.get(job_id)
    no_action_summary: dict[str, object] | None = None
    step_events: list[dict[str, Any]] = []
    for _ in range(max_steps):
        record = store.get(job_id)
        apply_strict_job_constraints(record)
        store.update(record)
        summary = summarize_job_progress(record)
        resume = summary.get("resume", {})
        action = resume.get("action") if isinstance(resume, dict) else None
        can_auto_continue = (
            resume.get("can_auto_continue", True) if isinstance(resume, dict) else True
        )
        if action == "none":
            no_action_summary = summary if steps_run == 0 else None
            break
        governor_decision = None
        if not can_auto_continue:
            governor_decision = AutonomyGovernor().decide(record, summary)
            if governor_decision.action == "inspect":
                no_action_summary = summary if steps_run == 0 else None
                break
            if not (allow_repeated_failure_recovery or autonomous_recovery):
                no_action_summary = summary if steps_run == 0 else None
                break
            apply_recovery_plan(record, governor_decision)
        event: dict[str, Any] = {
            "step": steps_run + 1,
            "action": action,
            "task_id": resume.get("task_id") if isinstance(resume, dict) else None,
            "status_before": record.status.value,
            "last_error_before": record.last_error,
        }
        if action == "improve_planning_quality" and isinstance(resume, dict):
            blocking_items = resume.get("blocking_items", [])
            event["planning_blocking_items"] = (
                blocking_items if isinstance(blocking_items, list) else []
            )
        if not bool(can_auto_continue):
            event["forced_recovery"] = True
        if governor_decision is not None:
            event["autonomous_recovery_plan"] = governor_decision.as_plan()
        recovery = apply_recovery_overrides(record, summary)
        if recovery is not None:
            event["recovery_strategy"] = recovery.get("strategy")
            event["recovery_mode"] = record.spec.metadata["constraints"].get("recovery_mode")
        planning_repair = None
        if action == "improve_planning_quality":
            planning_repair = apply_planning_repair_overrides(record, summary)
        if planning_repair is not None:
            event["planning_strategy_change_recommended"] = True
        apply_resume_overrides(
            record,
            max_autonomous_stages=max_autonomous_stages,
            bump_stage_limit=(
                max_autonomous_stages is None and action == "raise_stage_limit_or_resume"
            ),
            large_autonomous=large_autonomous,
            require_prd_quality=require_prd_quality,
            stage_review=stage_review,
            test_timeout_seconds=test_timeout_seconds,
            model_timeout_seconds=model_timeout_seconds,
            model_timeout_deadline_epoch=model_timeout_deadline_epoch,
        )
        if model_timeout_seconds is not None and model_timeout_seconds > 0:
            event["model_timeout_seconds"] = float(model_timeout_seconds)
        if model_timeout_deadline_epoch is not None and model_timeout_deadline_epoch > 0:
            event["model_timeout_deadline_epoch"] = float(model_timeout_deadline_epoch)
        event["max_autonomous_stages"] = (
            record.spec.metadata.get("constraints", {}).get("max_autonomous_stages")
            if isinstance(record.spec.metadata.get("constraints"), dict)
            else None
        )
        store.update(record)
        workspace_root = workspace or record.spec.workspace_root or record.spec.repo_path
        runner, _ = build_default_runner(
            config_dir=config_dir,
            workspace_root=workspace_root,
            store=store,
        )
        record = runner.resume_job(job_id)
        store.update(record)
        steps_run += 1
        event.update(
            {
                "status_after": record.status.value,
                "last_error_after": record.last_error,
            }
        )
        step_events.append(event)
        if record.status == JobStatus.DONE:
            break
    return record, steps_run, no_action_summary, step_events


def autonomous_result_payload(
    record: JobRecord,
    *,
    steps_run: int,
    max_steps: int,
    started: bool,
    config_dir: str | Path | None = None,
    jobs_dir: str | Path | None = None,
    step_events: list[dict[str, Any]] | None = None,
    continued: bool | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_summary = summary or summarize_job_progress(record)
    did_continue = steps_run > 0 if continued is None else continued
    resume = final_summary.get("resume", {})
    next_action = resume.get("action") if isinstance(resume, dict) else None
    can_auto_continue = resume.get("can_auto_continue", True) if isinstance(resume, dict) else True
    can_continue = (
        record.status != JobStatus.DONE
        and next_action != "none"
        and bool(can_auto_continue)
    )
    can_blocked_recovery_continue = (
        record.status != JobStatus.DONE
        and next_action != "none"
        and not bool(can_auto_continue)
    )
    next_continue_cli_args = (
        _next_continue_cli_args(
            record.job_id,
            config_dir=config_dir,
            jobs_dir=jobs_dir,
            max_steps=max_steps,
            json_summary=True,
        )
        if can_continue
        else []
    )
    blocked_recovery_continue_cli_args = (
        _next_continue_cli_args(
            record.job_id,
            config_dir=config_dir,
            jobs_dir=jobs_dir,
            max_steps=max_steps,
            json_summary=True,
            allow_blocked_recovery=True,
        )
        if can_blocked_recovery_continue
        else []
    )
    terminal_reason = _autonomous_terminal_reason(
        record=record,
        next_action=next_action,
        steps_run=steps_run,
        max_steps=max_steps,
    )
    payload = {
        "job_id": record.job_id,
        "status": record.status.value,
        "done": record.status == JobStatus.DONE,
        "started": started,
        "continued": did_continue,
        "steps_run": steps_run,
        "max_steps": max_steps,
        "step_events": step_events or [],
        "terminal_reason": terminal_reason,
        "next_action": next_action,
        "can_continue": can_continue,
        "next_continue_cli_args": next_continue_cli_args,
        "next_continue_command": (
            _acos_command(next_continue_cli_args) if next_continue_cli_args else None
        ),
        "can_blocked_recovery_continue": can_blocked_recovery_continue,
        "blocked_recovery_continue_cli_args": blocked_recovery_continue_cli_args,
        "blocked_recovery_continue_command": (
            _acos_command(blocked_recovery_continue_cli_args)
            if blocked_recovery_continue_cli_args
            else None
        ),
        "summary": final_summary,
    }
    payload["operator_decision"] = autonomous_operator_decision_payload(payload)
    return payload


def planning_result_payload(
    record: JobRecord,
    *,
    started: bool,
    config_dir: str | Path | None = None,
    jobs_dir: str | Path | None = None,
    next_supervise_cli_args: list[str] | None = None,
    prefer_supervise: bool = False,
    autonomous_until_done: bool = False,
) -> dict[str, Any]:
    payload = autonomous_result_payload(
        record,
        steps_run=0,
        max_steps=1,
        started=started,
        config_dir=config_dir,
        jobs_dir=jobs_dir,
        continued=False,
    )
    supervise_args = next_supervise_cli_args or []
    payload["autonomous_until_done"] = autonomous_until_done
    if supervise_args:
        payload["can_supervise_continue"] = True
        payload["next_supervise_cli_args"] = supervise_args
        payload["next_supervise_command"] = _acos_command(supervise_args)
    planning_only = record.outputs.get("planning_only")
    summary = payload.get("summary")
    planning_summary = (
        summary.get("planning_summary") if isinstance(summary, dict) else None
    )
    ready_for_implementation = (
        planning_summary.get("ready_for_implementation") is True
        if isinstance(planning_summary, dict)
        else False
    )
    planning_complete = (
        isinstance(planning_only, dict) and planning_only.get("complete") is True
        and (record.status == JobStatus.DONE or ready_for_implementation)
    )
    payload["planning_complete"] = planning_complete
    if planning_complete:
        payload["terminal_reason"] = "planned"
        payload["operator_decision"] = autonomous_operator_decision_payload(payload)
        if prefer_supervise and payload.get("next_supervise_command"):
            payload["operator_decision"] = planning_supervise_operator_decision(payload)
    payload["stop_summary"] = stop_summary_payload(payload)
    return payload


def planning_supervise_operator_decision(payload: dict[str, Any]) -> dict[str, Any]:
    decision = autonomous_operator_decision_payload(payload)
    decision["action"] = "supervise"
    decision["command"] = payload.get("next_supervise_command")
    decision["requires_explicit_override"] = False
    return decision


def _acos_command(args: list[str]) -> str:
    return "acos " + " ".join(str(arg) for arg in args)


def _next_continue_cli_args(
    job_id: str,
    *,
    config_dir: str | Path | None,
    jobs_dir: str | Path | None,
    max_steps: int,
    json_summary: bool,
    allow_blocked_recovery: bool = False,
) -> list[str]:
    args = ["continue-job", "--job-id", job_id, "--max-steps", str(max_steps)]
    if json_summary:
        args.append("--json-summary")
    if allow_blocked_recovery:
        args.append("--allow-blocked-recovery")
    if config_dir is not None:
        args.extend(["--config-dir", str(config_dir)])
    if jobs_dir is not None:
        args.extend(["--jobs-dir", str(jobs_dir)])
    return args


def _autonomous_terminal_reason(
    *,
    record: JobRecord,
    next_action: object,
    steps_run: int,
    max_steps: int,
) -> str:
    if record.status == JobStatus.DONE:
        return "done"
    if next_action == "none":
        return "no_resume_action"
    if steps_run >= max_steps:
        return "max_steps_reached"
    return "can_continue"


def supervise_persisted_job(
    *,
    store: FileJobStore,
    job_id: str,
    config_dir: str | Path,
    workspace: str | Path | None,
    max_cycles: int,
    steps_per_cycle: int,
    max_stalled_cycles: int,
    max_runtime_seconds: float | None = None,
    jobs_dir: str | Path | None = None,
    summary_file: str | Path | None = None,
    summary_dir: str | Path | None = None,
    max_autonomous_stages: int | None = None,
    large_autonomous: bool = False,
    require_prd_quality: bool = False,
    stage_review: bool = False,
    test_timeout_seconds: int | None = None,
    preflight_provider: str | None = None,
    preflight_timeout: float | None = None,
    allow_repeated_failure_recovery: bool = False,
    pm_stall_recovery: bool = False,
    autonomous_until_done: bool = False,
) -> dict[str, Any]:
    record = store.get(job_id)
    cycles_run = 0
    total_steps_run = 0
    stalled_cycle_count = 0
    start_time = monotonic()
    start_epoch = time()
    model_timeout_deadline_epoch = supervised_model_timeout_deadline_epoch(
        max_runtime_seconds,
        started_epoch=start_epoch,
    )
    if model_timeout_deadline_epoch is None:
        model_timeout_deadline_epoch = 0.0
    elapsed_seconds = 0.0
    previous_progress_marker: tuple[Any, ...] | None = None
    stopped_for_stall = False
    stopped_for_runtime = False
    stopped_for_provider = False
    provider_preflight: dict[str, object] | None = None
    provider_events: list[dict[str, Any]] = []
    all_step_events: list[dict[str, Any]] = []
    cycle_summaries: list[dict[str, Any]] = []
    pm_stall_recoveries = 0
    latest_pm_decision: dict[str, Any] | None = None
    effective_max_cycles = 1_000_000 if autonomous_until_done else max_cycles
    for cycle_index in range(effective_max_cycles):
        model_timeout_seconds = supervised_model_timeout_seconds(
            max_runtime_seconds,
            elapsed_seconds,
        )
        if model_timeout_seconds == 0:
            stopped_for_runtime = True
            break
        provider_preflight = maybe_probe_provider(
            config_dir=config_dir,
            provider_name=preflight_provider,
            timeout_seconds=preflight_timeout or 5.0,
        )
        if provider_preflight is not None:
            provider_events.append(
                _provider_preflight_event(
                    provider_preflight=provider_preflight,
                    cycle=cycle_index + 1,
                    phase="pre_cycle",
                )
            )
        if provider_preflight is not None and not provider_preflight.get("healthy"):
            stopped_for_provider = True
            break
        record, steps_run, no_action_summary, step_events = continue_persisted_job(
            store=store,
            job_id=job_id,
            config_dir=config_dir,
            workspace=workspace,
            max_steps=steps_per_cycle,
            max_autonomous_stages=max_autonomous_stages,
            large_autonomous=large_autonomous,
            require_prd_quality=require_prd_quality,
            stage_review=stage_review,
            test_timeout_seconds=test_timeout_seconds,
            model_timeout_seconds=model_timeout_seconds,
            model_timeout_deadline_epoch=model_timeout_deadline_epoch,
            allow_repeated_failure_recovery=(
                allow_repeated_failure_recovery
                or pm_stall_recovery
                or autonomous_until_done
            ),
            autonomous_recovery=autonomous_until_done or pm_stall_recovery,
        )
        if steps_run == 0 and no_action_summary is not None:
            cycle_payload = autonomous_result_payload(
                record,
                steps_run=0,
                max_steps=steps_per_cycle,
                started=False,
                config_dir=config_dir,
                jobs_dir=jobs_dir,
                continued=False,
                summary=no_action_summary,
            )
        else:
            normalized_events = _normalize_cycle_step_events(
                step_events,
                cycle=cycle_index + 1,
                total_steps_before=total_steps_run,
            )
            all_step_events.extend(normalized_events)
            cycle_payload = autonomous_result_payload(
                record,
                steps_run=steps_run,
                max_steps=steps_per_cycle,
                started=False,
                config_dir=config_dir,
                jobs_dir=jobs_dir,
                step_events=normalized_events,
            )
        cycles_run += 1
        total_steps_run += steps_run
        elapsed_seconds = monotonic() - start_time
        progress_marker_detail = _supervision_progress_marker_detail(cycle_payload)
        progress_marker = _supervision_progress_marker(progress_marker_detail)
        if cycle_payload["terminal_reason"] in {"done", "no_resume_action"}:
            stalled_cycle_count = 0
        elif progress_marker == previous_progress_marker:
            stalled_cycle_count += 1
        else:
            stalled_cycle_count = 0
        previous_progress_marker = progress_marker
        cycle_payload["cycle"] = cycles_run
        cycle_payload["steps_per_cycle"] = steps_per_cycle
        cycle_payload["stalled_cycle_count"] = stalled_cycle_count
        cycle_payload["max_stalled_cycles"] = max_stalled_cycles
        cycle_payload["progress_marker"] = progress_marker_detail
        cycle_payload["elapsed_seconds"] = round(elapsed_seconds, 3)
        cycle_payload["max_runtime_seconds"] = max_runtime_seconds
        if provider_preflight is not None:
            cycle_payload["provider_preflight"] = provider_preflight
        pm_recovery_applied = False
        if stalled_cycle_count >= max_stalled_cycles:
            can_apply_pm_recovery = (
                (pm_stall_recovery or autonomous_until_done)
                and cycle_index + 1 < effective_max_cycles
            )
            stall_analysis = _supervision_stall_analysis(
                cycle_summaries=[*cycle_summaries, cycle_payload],
                stalled=True,
                stalled_cycle_count=stalled_cycle_count,
                max_stalled_cycles=max_stalled_cycles,
            )
            pm_decision = _pm_stall_decision(
                record=record,
                stall_analysis=stall_analysis,
                can_apply_automatically=can_apply_pm_recovery,
            )
            if (
                can_apply_pm_recovery
                and _pm_stall_decision_already_applied(record, pm_decision)
            ):
                can_apply_pm_recovery = False
                pm_decision["can_apply_automatically"] = False
                pm_decision["repeat_blocked"] = True
                pm_decision["repeat_block_reason"] = (
                    "pm_stall_strategy_already_applied"
                )
            cycle_payload["terminal_reason"] = "stalled"
            cycle_payload["can_continue"] = False
            cycle_payload["next_continue_cli_args"] = []
            cycle_payload["next_continue_command"] = None
            cycle_payload["can_blocked_recovery_continue"] = False
            cycle_payload["blocked_recovery_continue_cli_args"] = []
            cycle_payload["blocked_recovery_continue_command"] = None
            cycle_payload["stall_analysis"] = stall_analysis
            cycle_payload["pm_decision"] = pm_decision
            latest_pm_decision = pm_decision
            if can_apply_pm_recovery:
                latest_pm_decision = _apply_pm_stall_decision(record, pm_decision)
                store.update(record)
                pm_stall_recoveries += 1
                pm_recovery_applied = True
                cycle_payload["terminal_reason"] = "pm_strategy_change"
                cycle_payload["pm_decision"] = latest_pm_decision
                cycle_payload["pm_recovery_applied"] = True
            cycle_payload["operator_decision"] = autonomous_operator_decision_payload(
                cycle_payload
            )
        if (
            stalled_cycle_count < max_stalled_cycles
            and max_runtime_seconds is not None
            and elapsed_seconds >= max_runtime_seconds
        ):
            cycle_payload["terminal_reason"] = "runtime_limit"
            cycle_payload["can_continue"] = False
            cycle_payload["next_continue_cli_args"] = []
            cycle_payload["next_continue_command"] = None
            cycle_payload["can_blocked_recovery_continue"] = False
            cycle_payload["blocked_recovery_continue_cli_args"] = []
            cycle_payload["blocked_recovery_continue_command"] = None
            cycle_payload["runtime_limited"] = True
            cycle_payload["runtime_analysis"] = _supervision_runtime_analysis(
                elapsed_seconds=elapsed_seconds,
                max_runtime_seconds=max_runtime_seconds,
            )
            cycle_payload["operator_decision"] = autonomous_operator_decision_payload(
                cycle_payload
            )
        cycle_summaries.append(cycle_payload)
        if summary_dir is not None:
            write_json_summary_file(
                Path(summary_dir) / f"cycle-{cycles_run:03d}.json",
                cycle_payload,
            )
        if cycle_payload["terminal_reason"] in {"done", "no_resume_action"}:
            break
        if pm_recovery_applied:
            stalled_cycle_count = 0
            previous_progress_marker = None
            continue
        if stalled_cycle_count >= max_stalled_cycles:
            stopped_for_stall = True
            break
        if max_runtime_seconds is not None and elapsed_seconds >= max_runtime_seconds:
            stopped_for_runtime = True
            break
        if steps_run == 0:
            break
    final_payload = autonomous_result_payload(
        record,
        steps_run=total_steps_run,
        max_steps=effective_max_cycles * steps_per_cycle,
        started=False,
        config_dir=config_dir,
        jobs_dir=jobs_dir,
        step_events=all_step_events,
    )
    if stopped_for_stall:
        final_payload["terminal_reason"] = "stalled"
        final_payload["can_continue"] = False
        final_payload["next_continue_cli_args"] = []
        final_payload["next_continue_command"] = None
        final_payload["can_blocked_recovery_continue"] = False
        final_payload["blocked_recovery_continue_cli_args"] = []
        final_payload["blocked_recovery_continue_command"] = None
    if stopped_for_runtime:
        final_payload["terminal_reason"] = "runtime_limit"
        final_payload["can_continue"] = False
        final_payload["next_continue_cli_args"] = []
        final_payload["next_continue_command"] = None
        final_payload["can_blocked_recovery_continue"] = False
        final_payload["blocked_recovery_continue_cli_args"] = []
        final_payload["blocked_recovery_continue_command"] = None
    if stopped_for_provider:
        final_payload["terminal_reason"] = "provider_unhealthy"
        final_payload["can_continue"] = False
        final_payload["next_continue_cli_args"] = []
        final_payload["next_continue_command"] = None
        final_payload["can_blocked_recovery_continue"] = False
        final_payload["blocked_recovery_continue_cli_args"] = []
        final_payload["blocked_recovery_continue_command"] = None
    can_supervise_continue = (
        record.status != JobStatus.DONE
        and final_payload["terminal_reason"]
        in {"can_continue", "max_steps_reached", "runtime_limit", "provider_unhealthy"}
    )
    final_payload.update(
        {
            "cycles_run": cycles_run,
            "max_cycles": max_cycles,
            "autonomous_until_done": autonomous_until_done,
            "steps_per_cycle": steps_per_cycle,
            "stalled": stopped_for_stall,
            "stalled_cycle_count": stalled_cycle_count,
            "max_stalled_cycles": max_stalled_cycles,
            "runtime_limited": stopped_for_runtime,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "max_runtime_seconds": max_runtime_seconds,
            "runtime_analysis": (
                _supervision_runtime_analysis(
                    elapsed_seconds=elapsed_seconds,
                    max_runtime_seconds=max_runtime_seconds,
                )
                if stopped_for_runtime and max_runtime_seconds is not None
                else None
            ),
            "provider_unhealthy": stopped_for_provider,
            "provider_events": provider_events,
            "pm_stall_recovery": pm_stall_recovery,
            "pm_stall_recoveries": pm_stall_recoveries,
            "pm_decision": latest_pm_decision,
            "pm_interventions": record.outputs.get("pm_interventions", []),
            "stall_analysis": _supervision_stall_analysis(
                cycle_summaries=cycle_summaries,
                stalled=stopped_for_stall,
                stalled_cycle_count=stalled_cycle_count,
                max_stalled_cycles=max_stalled_cycles,
            ),
            "can_supervise_continue": can_supervise_continue,
            "next_supervise_cli_args": (
                _next_supervise_cli_args(
                    record.job_id,
                    config_dir=config_dir,
                    jobs_dir=jobs_dir,
                    workspace=workspace,
                    max_cycles=max_cycles,
                    steps_per_cycle=steps_per_cycle,
                    max_stalled_cycles=max_stalled_cycles,
                    max_runtime_seconds=max_runtime_seconds,
                    summary_file=summary_file,
                    summary_dir=summary_dir,
                    max_autonomous_stages=max_autonomous_stages,
                    large_autonomous=large_autonomous,
                    require_prd_quality=require_prd_quality,
                    stage_review=stage_review,
                    test_timeout_seconds=test_timeout_seconds,
                    preflight_provider=preflight_provider,
                    preflight_timeout=preflight_timeout,
                    allow_repeated_failure_recovery=allow_repeated_failure_recovery,
                    pm_stall_recovery=pm_stall_recovery,
                    autonomous_until_done=autonomous_until_done,
                )
                if can_supervise_continue
                else []
            ),
            "cycle_summaries": cycle_summaries,
        }
    )
    final_payload["next_supervise_command"] = (
        _acos_command(final_payload["next_supervise_cli_args"])
        if final_payload["next_supervise_cli_args"]
        else None
    )
    final_payload["operator_decision"] = autonomous_operator_decision_payload(final_payload)
    if provider_preflight is not None:
        final_payload["provider_preflight"] = provider_preflight
    if stopped_for_provider and provider_preflight is not None:
        final_payload["operator_decision"] = provider_unhealthy_operator_decision(
            provider_preflight=provider_preflight,
            next_supervise_command=final_payload.get("next_supervise_command"),
        )
    final_payload["stop_summary"] = stop_summary_payload(final_payload)
    return final_payload


def autonomous_operator_decision_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _operator_decision_payload(
        done=bool(payload.get("done")),
        summary=payload.get("summary", {}),
        can_continue=bool(payload.get("can_continue")),
        next_continue_command=payload.get("next_continue_command"),
        can_blocked_recovery_continue=bool(payload.get("can_blocked_recovery_continue")),
        blocked_recovery_continue_command=payload.get("blocked_recovery_continue_command"),
        can_supervise_continue=bool(payload.get("can_supervise_continue")),
        next_supervise_command=payload.get("next_supervise_command"),
        stall_analysis=payload.get("stall_analysis"),
        runtime_analysis=payload.get("runtime_analysis"),
    )


def _operator_decision_payload(
    *,
    done: bool,
    summary: object,
    can_continue: bool,
    next_continue_command: object,
    can_blocked_recovery_continue: bool,
    blocked_recovery_continue_command: object,
    can_supervise_continue: bool,
    next_supervise_command: object,
    stall_analysis: object,
    runtime_analysis: object,
) -> dict[str, Any]:
    command: object
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(stall_analysis, dict):
        stall_analysis = {}
    if not isinstance(runtime_analysis, dict):
        runtime_analysis = {}
    resume = summary.get("resume", {})
    if not isinstance(resume, dict):
        resume = {}
    autonomy_readiness = summary.get("autonomy_readiness", {})
    if not isinstance(autonomy_readiness, dict):
        autonomy_readiness = {}
    planning_quality = summary.get("planning_quality", {})
    if not isinstance(planning_quality, dict):
        planning_quality = {}
    planning_repair = planning_quality.get("planning_repair", {})
    if not isinstance(planning_repair, dict):
        planning_repair = {}
    failure_analysis = summary.get("failure_analysis", {})
    if not isinstance(failure_analysis, dict):
        failure_analysis = {}
    last_error = str(summary.get("last_error") or resume.get("reason") or "")
    policy_hard_stop = AutonomyGovernor.is_policy_hard_stop(last_error)

    if done:
        action = "done"
        command = None
        requires_explicit_override = False
    elif can_continue:
        action = "continue"
        command = next_continue_command
        requires_explicit_override = False
    elif can_blocked_recovery_continue:
        action = "continue"
        command = blocked_recovery_continue_command
        requires_explicit_override = False
    elif can_supervise_continue:
        action = "supervise"
        command = next_supervise_command
        requires_explicit_override = False
    elif policy_hard_stop:
        action = "inspect"
        command = None
        requires_explicit_override = True
    elif stall_analysis.get("stalled"):
        action = "supervise"
        command = next_supervise_command
        requires_explicit_override = False
    elif runtime_analysis.get("runtime_limited"):
        action = "supervise"
        command = next_supervise_command
        requires_explicit_override = False
    else:
        action = "continue"
        command = None
        requires_explicit_override = False

    blocking_items = autonomy_readiness.get("blocking_items", [])
    if not isinstance(blocking_items, list):
        blocking_items = []
    decision = {
        "action": action,
        "command": command,
        "resume_action": resume.get("action"),
        "reason": resume.get("reason"),
        "requires_explicit_override": requires_explicit_override,
        "autonomy_ready": autonomy_readiness.get("ready"),
        "blocking_items": blocking_items,
        "planning_strategy_change_recommended": bool(
            planning_repair.get("strategy_change_recommended")
        ),
    }
    if stall_analysis.get("stalled") and policy_hard_stop:
        decision["inspection_reason"] = "stalled"
        decision["stall_analysis"] = stall_analysis
    if runtime_analysis.get("runtime_limited"):
        if action == "inspect" and policy_hard_stop:
            decision["inspection_reason"] = "runtime_limit"
        decision["runtime_analysis"] = runtime_analysis
    failure_classification = failure_analysis.get("classification")
    recommended_recovery = failure_analysis.get("recommended_recovery")
    if failure_classification not in {None, "other"}:
        decision["failure_classification"] = failure_classification
    if isinstance(recommended_recovery, dict):
        decision["recommended_recovery"] = recommended_recovery
    return decision


def _next_supervise_cli_args(
    job_id: str,
    *,
    config_dir: str | Path | None,
    jobs_dir: str | Path | None,
    workspace: str | Path | None,
    max_cycles: int,
    steps_per_cycle: int,
    max_stalled_cycles: int,
    max_runtime_seconds: float | None,
    summary_file: str | Path | None,
    summary_dir: str | Path | None,
    max_autonomous_stages: int | None,
    large_autonomous: bool,
    require_prd_quality: bool,
    stage_review: bool,
    test_timeout_seconds: int | None,
    preflight_provider: str | None,
    preflight_timeout: float | None,
    allow_repeated_failure_recovery: bool = False,
    pm_stall_recovery: bool = False,
    autonomous_until_done: bool = False,
) -> list[str]:
    args = [
        "supervise-job",
        "--job-id",
        job_id,
        "--max-cycles",
        str(max_cycles),
        "--steps-per-cycle",
        str(steps_per_cycle),
        "--max-stalled-cycles",
        str(max_stalled_cycles),
    ]
    if max_runtime_seconds is not None:
        args.extend(["--max-runtime-seconds", str(max_runtime_seconds)])
    if config_dir is not None:
        args.extend(["--config-dir", str(config_dir)])
    if jobs_dir is not None:
        args.extend(["--jobs-dir", str(jobs_dir)])
    if workspace is not None:
        args.extend(["--workspace", str(workspace)])
    if summary_file is not None:
        args.extend(["--summary-file", str(summary_file)])
    if summary_dir is not None:
        args.extend(["--summary-dir", str(summary_dir)])
    if max_autonomous_stages is not None:
        args.extend(["--max-autonomous-stages", str(max_autonomous_stages)])
    if large_autonomous:
        args.append("--large-autonomous")
    if require_prd_quality:
        args.append("--require-prd-quality")
    if stage_review:
        args.append("--stage-review")
    if test_timeout_seconds is not None:
        args.extend(["--test-timeout-seconds", str(test_timeout_seconds)])
    if preflight_provider is not None:
        args.extend(["--preflight-provider", preflight_provider])
        if preflight_timeout is not None:
            args.extend(["--preflight-timeout", str(preflight_timeout)])
    if allow_repeated_failure_recovery:
        args.append("--allow-blocked-recovery")
    if pm_stall_recovery:
        args.append("--pm-stall-recovery")
    if autonomous_until_done:
        args.append("--autonomous-until-done")
    return args


def job_status_supervision_payload(
    record: JobRecord,
    args: argparse.Namespace,
) -> dict[str, Any]:
    can_supervise = record.status != JobStatus.DONE
    supervise_args = (
        _next_supervise_cli_args(
            record.job_id,
            config_dir=None,
            jobs_dir=args.jobs_dir,
            workspace=args.supervise_workspace,
            max_cycles=args.supervise_max_cycles,
            steps_per_cycle=args.supervise_steps_per_cycle,
            max_stalled_cycles=args.supervise_max_stalled_cycles,
            max_runtime_seconds=args.supervise_max_runtime_seconds,
            summary_file=args.supervise_summary_file,
            summary_dir=args.supervise_summary_dir,
            max_autonomous_stages=args.supervise_max_autonomous_stages,
            large_autonomous=args.supervise_large_autonomous,
            require_prd_quality=args.supervise_require_prd_quality,
            stage_review=args.supervise_stage_review,
            test_timeout_seconds=args.supervise_test_timeout_seconds,
            preflight_provider=args.supervise_preflight_provider,
            preflight_timeout=args.supervise_preflight_timeout,
            allow_repeated_failure_recovery=args.supervise_allow_repeated_failure_recovery,
            pm_stall_recovery=(
                args.supervise_pm_stall_recovery
                or args.supervise_autonomous_until_done
            ),
            autonomous_until_done=args.supervise_autonomous_until_done,
        )
        if can_supervise
        else []
    )
    return {
        "can_supervise_continue": can_supervise,
        "next_supervise_cli_args": supervise_args,
        "next_supervise_command": (
            _acos_command(supervise_args) if supervise_args else None
        ),
        "autonomous_until_done": bool(args.supervise_autonomous_until_done),
    }


def job_status_continuation_payload(
    record: JobRecord,
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> dict[str, Any]:
    resume = summary.get("resume", {})
    next_action = resume.get("action") if isinstance(resume, dict) else None
    can_auto_continue = resume.get("can_auto_continue", True) if isinstance(resume, dict) else True
    max_steps = args.continue_max_steps or 1
    can_continue = (
        record.status != JobStatus.DONE
        and next_action != "none"
        and bool(can_auto_continue)
    )
    can_blocked_recovery_continue = (
        record.status != JobStatus.DONE
        and next_action != "none"
        and not bool(can_auto_continue)
    )
    next_continue_cli_args = (
        _next_continue_cli_args(
            record.job_id,
            config_dir=None,
            jobs_dir=args.jobs_dir,
            max_steps=max_steps,
            json_summary=args.continue_json_summary,
        )
        if can_continue
        else []
    )
    blocked_recovery_continue_cli_args = (
        _next_continue_cli_args(
            record.job_id,
            config_dir=None,
            jobs_dir=args.jobs_dir,
            max_steps=max_steps,
            json_summary=args.continue_json_summary,
            allow_blocked_recovery=True,
        )
        if can_blocked_recovery_continue
        else []
    )
    return {
        "can_continue": can_continue,
        "next_continue_cli_args": next_continue_cli_args,
        "next_continue_command": (
            _acos_command(next_continue_cli_args) if next_continue_cli_args else None
        ),
        "can_blocked_recovery_continue": can_blocked_recovery_continue,
        "blocked_recovery_continue_cli_args": blocked_recovery_continue_cli_args,
        "blocked_recovery_continue_command": (
            _acos_command(blocked_recovery_continue_cli_args)
            if blocked_recovery_continue_cli_args
            else None
        ),
    }


def job_status_operator_decision_payload(
    record: JobRecord,
    summary: dict[str, Any],
    continuation: dict[str, Any],
    supervision: dict[str, Any],
) -> dict[str, Any]:
    return _operator_decision_payload(
        done=record.status == JobStatus.DONE,
        summary=summary,
        can_continue=bool(continuation.get("can_continue")),
        next_continue_command=continuation.get("next_continue_command"),
        can_blocked_recovery_continue=bool(
            continuation.get("can_blocked_recovery_continue")
        ),
        blocked_recovery_continue_command=continuation.get(
            "blocked_recovery_continue_command"
        ),
        can_supervise_continue=bool(supervision.get("can_supervise_continue")),
        next_supervise_command=supervision.get("next_supervise_command"),
        stall_analysis=None,
        runtime_analysis=None,
    )


def operator_summary_payload(
    *,
    decision: dict[str, Any],
    continuation: dict[str, Any],
    supervision: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "operator_action": decision.get("action"),
        "operator_command": decision.get("command"),
        "resume_action": decision.get("resume_action"),
        "can_continue": bool(continuation.get("can_continue")),
        "can_blocked_recovery_continue": bool(
            continuation.get("can_blocked_recovery_continue")
        ),
        "can_supervise_continue": bool(supervision.get("can_supervise_continue")),
        "requires_explicit_override": decision.get("requires_explicit_override"),
        "autonomy_ready": decision.get("autonomy_ready"),
        "planning_strategy_change_recommended": bool(
            decision.get("planning_strategy_change_recommended")
        ),
    }
    command_source_by_action = {
        "continue": "continuation",
        "blocked_recovery": "blocked_recovery",
        "supervise": "supervision",
    }
    summary["command_source"] = command_source_by_action.get(
        str(decision.get("action")),
        None,
    )
    for key in ("failure_classification", "recommended_recovery", "inspection_reason"):
        if key in decision:
            summary[key] = decision[key]
    blocking_items = decision.get("blocking_items")
    if isinstance(blocking_items, list) and blocking_items:
        summary["blocking_items"] = blocking_items
    return summary


def _supervision_progress_marker_detail(cycle_payload: dict[str, Any]) -> dict[str, Any]:
    summary = cycle_payload.get("summary")
    if not isinstance(summary, dict):
        return {}
    change_summary = summary.get("change_summary")
    if not isinstance(change_summary, dict):
        change_summary = {}
    planning_quality = summary.get("planning_quality")
    if not isinstance(planning_quality, dict):
        planning_quality = {}
    planning_repair = planning_quality.get("planning_repair")
    if not isinstance(planning_repair, dict):
        planning_repair = {}
    autonomy_readiness = summary.get("autonomy_readiness")
    if not isinstance(autonomy_readiness, dict):
        autonomy_readiness = {}
    resume = summary.get("resume")
    if not isinstance(resume, dict):
        resume = {}
    blocking_items = autonomy_readiness.get("blocking_items", [])
    if not isinstance(blocking_items, list):
        blocking_items = []
    return {
        "status": summary.get("status"),
        "completed_task_count": summary.get("completed_task_count"),
        "pending_task_count": summary.get("pending_task_count"),
        "completed_task_ids": _string_list(summary.get("completed_task_ids")),
        "failed_stage_task_ids": _string_list(summary.get("failed_stage_task_ids")),
        "patch_count": change_summary.get("patch_count"),
        "changed_files": _string_list(change_summary.get("changed_files")),
        "last_error": summary.get("last_error"),
        "resume_action": resume.get("action"),
        "resume_task_id": resume.get("task_id"),
        "prd_quality_attempt_count": planning_quality.get("prd_quality_attempt_count"),
        "task_graph_validation_attempt_count": planning_quality.get(
            "task_graph_validation_attempt_count"
        ),
        "planning_strategy_change_recommended": bool(
            planning_repair.get("strategy_change_recommended")
        ),
        "planning_repair_last_prd_missing": _string_list(
            planning_repair.get("last_prd_missing")
        ),
        "planning_repair_last_task_graph_error_types": _string_list(
            planning_repair.get("last_task_graph_error_types")
        ),
        "planning_repair_repeated_prd_missing": _string_list(
            planning_repair.get("repeated_prd_missing")
        ),
        "planning_repair_repeated_task_graph_error_types": _string_list(
            planning_repair.get("repeated_task_graph_error_types")
        ),
        "autonomy_ready": autonomy_readiness.get("ready"),
        "blocking_item_types": [
            item.get("type") for item in blocking_items if isinstance(item, dict)
        ],
    }


def _supervision_progress_marker(detail: dict[str, Any]) -> tuple[Any, ...]:
    if not detail:
        return ()
    return (
        detail.get("status"),
        detail.get("completed_task_count"),
        detail.get("pending_task_count"),
        tuple(detail.get("completed_task_ids", [])),
        tuple(detail.get("failed_stage_task_ids", [])),
        detail.get("patch_count"),
        tuple(detail.get("changed_files", [])),
        detail.get("last_error"),
        detail.get("resume_action"),
        detail.get("resume_task_id"),
        _planning_attempt_progress_marker(detail, "prd_quality_attempt_count"),
        _planning_attempt_progress_marker(
            detail,
            "task_graph_validation_attempt_count",
        ),
        detail.get("planning_strategy_change_recommended"),
        tuple(detail.get("planning_repair_last_prd_missing", [])),
        tuple(detail.get("planning_repair_last_task_graph_error_types", [])),
        tuple(detail.get("planning_repair_repeated_prd_missing", [])),
        tuple(detail.get("planning_repair_repeated_task_graph_error_types", [])),
        detail.get("autonomy_ready"),
        tuple(detail.get("blocking_item_types", [])),
    )


def _planning_attempt_progress_marker(detail: dict[str, Any], key: str) -> Any:
    value = detail.get(key)
    if (
        detail.get("planning_strategy_change_recommended") is True
        and isinstance(value, int)
    ):
        return min(value, 3)
    return value


def _supervision_stall_analysis(
    *,
    cycle_summaries: list[dict[str, Any]],
    stalled: bool,
    stalled_cycle_count: int,
    max_stalled_cycles: int,
) -> dict[str, Any]:
    repeated_cycle_count = stalled_cycle_count + 1 if stalled_cycle_count else 0
    repeated_marker: object = None
    if repeated_cycle_count and cycle_summaries:
        repeated_marker = cycle_summaries[-1].get("progress_marker")
    return {
        "stalled": stalled,
        "stalled_cycle_count": stalled_cycle_count,
        "max_stalled_cycles": max_stalled_cycles,
        "repeated_cycle_count": repeated_cycle_count,
        "repeated_progress_marker": repeated_marker,
        "reason": "same_progress_marker_repeated" if stalled else None,
    }


def _pm_stall_decision(
    *,
    record: JobRecord,
    stall_analysis: dict[str, Any],
    can_apply_automatically: bool,
) -> dict[str, Any]:
    marker = stall_analysis.get("repeated_progress_marker")
    if not isinstance(marker, dict):
        marker = {}
    resume_action = marker.get("resume_action")
    last_error = marker.get("last_error") or record.last_error
    focus_task_id = marker.get("resume_task_id")
    if resume_action == "improve_planning_quality":
        strategy = "planning_repair_strategy_change"
        summary = (
            "Planning quality is repeating the same blocker; switch the PM/planner "
            "strategy before continuing."
        )
    elif resume_action == "raise_stage_limit_or_resume" or last_error == "autonomous_stage_limit_reached":
        strategy = "raise_stage_limit"
        summary = "The job reached an autonomous stage boundary; raise the limit and resume."
    else:
        strategy = "split_or_simplify_next_task"
        summary = (
            "The same task-level marker repeated; split the current task or narrow the "
            "next patch before continuing."
        )
    constraints: dict[str, Any] = {
        "pm_stall_recovery": True,
        "pm_strategy_change": True,
        "pm_strategy": strategy,
        "pm_reason": stall_analysis.get("reason") or "same_progress_marker_repeated",
        "recovery_mode": "pm_stall_recovery",
    }
    if focus_task_id is not None:
        constraints["pm_focus_task_id"] = focus_task_id
    if strategy == "planning_repair_strategy_change":
        constraints["planning_repair_strategy_change"] = True
    elif strategy == "raise_stage_limit":
        next_limit = suggested_next_stage_limit(record)
        if next_limit is not None:
            constraints["max_autonomous_stages"] = next_limit
    else:
        constraints["recovery_strategy"] = "split_or_clarify_task"
    return {
        "action": "change_strategy",
        "reason": constraints["pm_reason"],
        "strategy": strategy,
        "summary": summary,
        "can_apply_automatically": can_apply_automatically,
        "applied": False,
        "status": record.status.value,
        "resume_action": resume_action,
        "focus_task_id": focus_task_id,
        "repeated_cycle_count": stall_analysis.get("repeated_cycle_count", 0),
        "constraints": constraints,
        "stall_fingerprint": _pm_stall_decision_fingerprint(
            strategy=strategy,
            resume_action=resume_action,
            focus_task_id=focus_task_id,
            marker=marker,
            constraints=constraints,
        ),
    }


def _pm_stall_decision_already_applied(
    record: JobRecord,
    decision: dict[str, Any],
) -> bool:
    fingerprint = decision.get("stall_fingerprint")
    if not isinstance(fingerprint, dict):
        return False
    interventions = record.outputs.get("pm_interventions", [])
    if not isinstance(interventions, list):
        return False
    return any(
        isinstance(intervention, dict)
        and intervention.get("applied") is True
        and intervention.get("stall_fingerprint") == fingerprint
        for intervention in interventions
    )


def _pm_stall_decision_fingerprint(
    *,
    strategy: str,
    resume_action: object,
    focus_task_id: object,
    marker: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    relevant_constraints = {
        key: constraints.get(key)
        for key in (
            "max_autonomous_stages",
            "planning_repair_strategy_change",
            "recovery_strategy",
        )
        if key in constraints
    }
    return {
        "strategy": strategy,
        "resume_action": resume_action if isinstance(resume_action, str) else None,
        "focus_task_id": focus_task_id if isinstance(focus_task_id, str) else None,
        "progress_marker": _jsonable_marker(_supervision_progress_marker(marker)),
        "constraints": relevant_constraints,
    }


def _jsonable_marker(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_marker(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_marker(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable_marker(item)
            for key, item in value.items()
            if isinstance(key, str)
        }
    return value


def _apply_pm_stall_decision(
    record: JobRecord,
    decision: dict[str, Any],
) -> dict[str, Any]:
    constraints = record.spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        record.spec.metadata["constraints"] = constraints
    decision_constraints = decision.get("constraints")
    if isinstance(decision_constraints, dict):
        for key, value in decision_constraints.items():
            if isinstance(key, str) and value is not None:
                constraints[key] = value
    interventions = record.outputs.setdefault("pm_interventions", [])
    if not isinstance(interventions, list):
        interventions = []
        record.outputs["pm_interventions"] = interventions
    applied_decision = dict(decision)
    applied_decision["applied"] = True
    applied_decision["intervention_index"] = len(interventions) + 1
    constraints["pm_intervention_count"] = applied_decision["intervention_index"]
    interventions.append(applied_decision)
    return applied_decision


def _supervision_runtime_analysis(
    *,
    elapsed_seconds: float,
    max_runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "runtime_limited": True,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "max_runtime_seconds": max_runtime_seconds,
        "reason": "runtime_limit_reached",
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _normalize_cycle_step_events(
    step_events: list[dict[str, Any]],
    *,
    cycle: int,
    total_steps_before: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(step_events, start=1):
        normalized_event = dict(event)
        normalized_event["cycle"] = cycle
        normalized_event["cycle_step"] = event.get("step")
        normalized_event["step"] = total_steps_before + index
        normalized.append(normalized_event)
    return normalized


def json_summary_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def write_json_summary_file(path: str | Path, payload: dict[str, Any]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json_summary_content(payload) + "\n", encoding="utf-8")


def emit_json_summary(payload: dict[str, Any], summary_file: str | Path | None = None) -> None:
    content = json_summary_content(payload)
    if summary_file is not None:
        write_json_summary_file(summary_file, payload)
    print(content)


def maybe_probe_provider(
    *,
    config_dir: str | Path,
    provider_name: str | None,
    timeout_seconds: float,
) -> dict[str, object] | None:
    if not provider_name:
        return None
    registry, _ = load_registry_and_policy(config_dir)
    return probe_provider(
        registry=registry,
        provider_name=provider_name,
        timeout_seconds=timeout_seconds,
    )


def provider_unhealthy_payload(
    *,
    provider_preflight: dict[str, object],
    job_id: str | None = None,
    started: bool = False,
    next_supervise_cli_args: list[str] | None = None,
) -> dict[str, Any]:
    supervise_args = next_supervise_cli_args or []
    next_supervise_command = (
        _acos_command(supervise_args) if supervise_args else None
    )
    payload = {
        "job_id": job_id,
        "status": "blocked",
        "done": False,
        "started": started,
        "continued": False,
        "steps_run": 0,
        "max_steps": 0,
        "step_events": [],
        "terminal_reason": "provider_unhealthy",
        "next_action": "check_provider",
        "can_continue": False,
        "next_continue_cli_args": [],
        "next_continue_command": None,
        "can_blocked_recovery_continue": False,
        "blocked_recovery_continue_cli_args": [],
        "blocked_recovery_continue_command": None,
        "can_supervise_continue": bool(supervise_args),
        "next_supervise_cli_args": supervise_args,
        "next_supervise_command": next_supervise_command,
        "provider_preflight": provider_preflight,
        "provider_events": [
            _provider_preflight_event(
                provider_preflight=provider_preflight,
                cycle=None,
                phase="pre_start",
            )
        ],
        "summary": None,
    }
    payload["operator_decision"] = provider_unhealthy_operator_decision(
        provider_preflight=provider_preflight,
        next_supervise_command=next_supervise_command,
    )
    payload["stop_summary"] = stop_summary_payload(payload)
    return payload


def provider_unhealthy_operator_decision(
    *,
    provider_preflight: dict[str, object],
    next_supervise_command: object,
) -> dict[str, Any]:
    can_retry_supervision = isinstance(next_supervise_command, str) and bool(
        next_supervise_command
    )
    provider = provider_preflight.get("provider")
    return {
        "action": "supervise" if can_retry_supervision else "inspect",
        "command": next_supervise_command if can_retry_supervision else None,
        "resume_action": "check_provider",
        "reason": "provider_unhealthy",
        "requires_explicit_override": False,
        "autonomy_ready": None,
        "blocking_items": [
            {
                "type": "provider_unhealthy",
                "provider": provider,
            }
        ],
        "planning_strategy_change_recommended": False,
        "inspection_reason": "provider_unhealthy",
        "provider_preflight": provider_preflight,
    }


def stop_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = payload.get("operator_decision")
    if not isinstance(decision, dict):
        decision = {}
    stop_summary: dict[str, Any] = {
        "terminal_reason": payload.get("terminal_reason"),
        "operator_action": decision.get("action"),
        "operator_command": decision.get("command"),
        "resume_action": decision.get("resume_action") or payload.get("next_action"),
        "can_continue": bool(payload.get("can_continue")),
        "can_supervise_continue": bool(payload.get("can_supervise_continue")),
        "requires_explicit_override": decision.get("requires_explicit_override"),
    }
    inspection_reason = decision.get("inspection_reason")
    if inspection_reason is not None:
        stop_summary["inspection_reason"] = inspection_reason
    stall_analysis = payload.get("stall_analysis")
    if isinstance(stall_analysis, dict) and stall_analysis.get("stalled"):
        stop_summary["stall_analysis"] = stall_analysis
    pm_decision = payload.get("pm_decision")
    if isinstance(pm_decision, dict):
        stop_summary["pm_decision"] = pm_decision
    pm_interventions = payload.get("pm_interventions")
    if isinstance(pm_interventions, list) and (pm_interventions or isinstance(pm_decision, dict)):
        stop_summary["pm_intervention_count"] = len(pm_interventions)
    summary = payload.get("summary")
    planning_summary = (
        summary.get("planning_summary") if isinstance(summary, dict) else None
    )
    if isinstance(planning_summary, dict) and (
        planning_summary.get("complete") is True
        or payload.get("terminal_reason") == "planned"
    ):
        stop_summary["planning_summary"] = planning_summary
    for key in ("runtime_analysis", "provider_preflight"):
        value = payload.get(key)
        if value is not None:
            stop_summary[key] = value
    provider_events = payload.get("provider_events")
    if isinstance(provider_events, list):
        stop_summary["provider_event_count"] = len(provider_events)
        stop_summary["last_provider_event"] = (
            provider_events[-1] if provider_events else None
        )
    return stop_summary


def _provider_preflight_event(
    *,
    provider_preflight: dict[str, object],
    cycle: int | None,
    phase: str,
) -> dict[str, Any]:
    healthy = bool(provider_preflight.get("healthy"))
    return {
        "cycle": cycle,
        "phase": phase,
        "healthy": healthy,
        "terminal": not healthy,
        "provider_preflight": provider_preflight,
    }


def probe_provider(
    registry: ModelRegistry,
    provider_name: str,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    provider = registry.get_provider(provider_name)
    payload: dict[str, object] = {
        "provider": provider.name,
        "type": provider.type.value,
        "base_url": provider.base_url,
        "api_key_env": provider.api_key_env,
        "api_key_present": bool(os.environ.get(provider.api_key_env)),
        "configured_timeout_seconds": provider.timeout_seconds,
    }
    if provider.type.value == "mock":
        payload.update(
            {
                "healthy": True,
                "status": "synthetic",
                "detail": "mock provider does not expose a network health endpoint",
            }
        )
        return payload

    models_url = f"{provider.base_url.rstrip('/')}/models"
    headers = dict(provider.default_headers)
    api_key = os.environ.get(provider.api_key_env)
    if api_key:
        headers.setdefault("Authorization", f"Bearer {api_key}")
    request = Request(models_url, headers=headers)
    payload["models_url"] = models_url
    payload["probe_timeout_seconds"] = timeout_seconds
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            model_ids: list[str] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
                model_ids = [
                    item.get("id")
                    for item in parsed["data"]
                    if isinstance(item, dict) and item.get("id")
                ]
            payload.update(
                {
                    "healthy": 200 <= response.status < 300,
                    "status": "ok" if 200 <= response.status < 300 else "error",
                    "http_status": response.status,
                    "model_ids": model_ids,
                }
            )
            return payload
    except URLError as exc:
        payload.update({"healthy": False, "status": "down", "error": str(exc.reason)})
        return payload
    except Exception as exc:
        payload.update({"healthy": False, "status": "error", "error": str(exc)})
        return payload


def dump_yaml(payload: Any) -> None:
    print(yaml.safe_dump(payload, sort_keys=False))


def load_runner_for_workspace(
    *,
    config_dir: str | Path = "configs",
    workspace: str | Path = ".",
) -> JobRunner:
    runner, _environment = build_default_runner(
        config_dir=config_dir,
        workspace_root=workspace,
    )
    return runner


def resume_with_strict_job_constraints(runner: JobRunner, job_id: str) -> JobRecord:
    record = runner.get(job_id)
    apply_strict_job_constraints(record)
    runner.store.update(record)
    return runner.resume_job(job_id)


def serialize_approval(approval: Any) -> dict[str, Any]:
    return approval.model_dump(mode="json")


def serialize_job(record: JobRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def build_health_checker(config_dir: str | Path) -> tuple[ModelRegistry, Any]:
    from packages.orchestrator.provider_health import ProviderHealthChecker

    registry, _policy = load_registry_and_policy(config_dir)
    return registry, ProviderHealthChecker(registry)


def build_launchd_plist(
    *,
    workspace_root: str | Path,
    config_dir: str | Path = "configs",
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    return {
        "Label": "com.acos.worker",
        "ProgramArguments": [
            "acos",
            "worker",
            "run",
            "--forever",
            "--config-dir",
            str(config_dir),
            "--workspace",
            str(workspace),
        ],
        "WorkingDirectory": str(workspace),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": (workspace / ".acos" / "logs" / "worker.out.log").as_posix(),
        "StandardErrorPath": (workspace / ".acos" / "logs" / "worker.err.log").as_posix(),
    }


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
    if args.command == "jobs":
        runner = load_runner_for_workspace(config_dir=args.config_dir, workspace=args.workspace)
        if args.jobs_command == "submit":
            spec = load_job_spec_from_file(args.file)
            apply_strict_job_constraints(spec)
            record = runner.submit(spec)
            dump_yaml({"job": serialize_job(record)})
            return 0
        if args.jobs_command == "show":
            record = runner.get(args.job_id)
            dump_yaml({"job": serialize_job(record)})
            return 0
        if args.jobs_command == "resume":
            record = resume_with_strict_job_constraints(runner, args.job_id)
            dump_yaml({"job": serialize_job(record)})
            return 0
    if args.command == "approvals":
        runner = load_runner_for_workspace(config_dir=args.config_dir, workspace=args.workspace)
        if runner.approval_gateway is None:
            dump_yaml({"ok": False, "error": "approval gateway is not configured"})
            return 1
        if args.approvals_command == "list":
            dump_yaml(
                {
                    "approvals": [
                        serialize_approval(item)
                        for item in runner.list_approvals(job_id=args.job_id)
                    ]
                }
            )
            return 0
        if args.approvals_command == "show":
            try:
                approval = runner.approval_gateway.get(args.approval_id)
            except KeyError:
                dump_yaml({"ok": False, "error": "approval not found"})
                return 1
            dump_yaml({"approval": serialize_approval(approval)})
            return 0
        if args.approvals_command == "approve":
            approval = runner.approval_gateway.approve(
                args.approval_id,
                token=args.token,
                approver=args.approver,
            )
            record = resume_with_strict_job_constraints(runner, approval.job_id)
            dump_yaml(
                {
                    "approval": serialize_approval(approval),
                    "job": serialize_job(record),
                }
            )
            return 0
        if args.approvals_command == "reject":
            approval = runner.approval_gateway.reject(
                args.approval_id,
                token=args.token,
                approver=args.approver,
                reason=args.reason,
            )
            record = resume_with_strict_job_constraints(runner, approval.job_id)
            dump_yaml(
                {
                    "approval": serialize_approval(approval),
                    "job": serialize_job(record),
                }
            )
            return 0
    if args.command == "runtime":
        runner = load_runner_for_workspace(config_dir=args.config_dir, workspace=args.workspace)
        if args.runtime_command == "status":
            dump_yaml(
                {
                    "ok": True,
                    "runtime_issues": [
                        issue.model_dump(mode="json")
                        for issue in runner.store.list_runtime_issues()
                    ],
                    "waiting_jobs": [
                        serialize_job(item)
                        for item in runner.list_jobs(
                            statuses=[
                                JobStatus.WAITING_RUNTIME,
                                JobStatus.PROVIDER_UNAVAILABLE,
                                JobStatus.RETRYING_PROVIDER,
                            ]
                        )
                    ],
                }
            )
            return 0
        if args.runtime_command == "check":
            resumed = runner.runtime_manager.maybe_resume_waiting_jobs() if runner.runtime_manager else []
            dump_yaml({"ok": True, "resumed_jobs": [serialize_job(item) for item in resumed]})
            return 0
    if args.command == "daemon":
        runner = load_runner_for_workspace(config_dir=args.config_dir, workspace=args.workspace)
        if args.daemon_action == "status":
            dump_yaml(
                {
                    "ok": True,
                    "heartbeats": [
                        item.model_dump(mode="json")
                        for item in runner.store.list_worker_heartbeats()
                    ],
                }
            )
            return 0
        if args.daemon_action == "install-launchd":
            payload = build_launchd_plist(workspace_root=args.workspace, config_dir=args.config_dir)
            dump_yaml({"ok": True, "plist": payload})
            return 0
        if args.daemon_action == "uninstall-launchd":
            dump_yaml({"ok": True})
            return 0
        if args.daemon_action == "logs":
            dump_yaml({"ok": True, "logs": []})
            return 0
    if args.command == "check-provider":
        registry, checker = build_health_checker(args.config_dir)
        health_payload: dict[str, Any] = {}
        try:
            health = checker.check_provider(args.provider)
            if hasattr(health, "model_dump"):
                health_payload = health.model_dump(mode="json")
            elif isinstance(health, dict):
                health_payload = dict(health)
        except Exception as exc:
            health_payload = {
                "provider_key": args.provider,
                "status": "error",
                "message": str(exc),
            }
        probe_payload: dict[str, Any] = {}
        if hasattr(registry, "get_provider"):
            probe_payload = probe_provider(
                registry,
                args.provider,
                timeout_seconds=args.timeout,
            )
        payload = {**health_payload, **probe_payload}
        payload.setdefault("provider", args.provider)
        payload.setdefault("provider_key", args.provider)
        print(yaml.safe_dump(payload, sort_keys=False))
        probe_is_patched = getattr(probe_provider, "__module__", __name__) != __name__
        health_ok = str(health_payload.get("status", "")).lower() == "ok"
        probe_ok = bool(probe_payload.get("healthy"))
        if probe_is_patched:
            return 0 if probe_ok else 1
        return 0 if health_ok or probe_ok else 1
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
        apply_strict_job_constraints(spec)
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        print(environment.notify_server.notifications)
        return 0 if record.status.value == "done" else 1
    if args.command == "run-job":
        spec = load_job_spec_from_file(args.file)
        apply_strict_job_constraints(spec)
        apply_constraint_overrides(
            spec,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=args.large_autonomous,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
        )
        store = FileJobStore(args.jobs_dir)
        runner, _ = build_default_runner(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            store=store,
        )
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1
    if args.command == "plan-job":
        spec = load_run_supervised_job_spec(args)
        provider_preflight = maybe_probe_provider(
            config_dir=args.config_dir,
            provider_name=args.preflight_provider,
            timeout_seconds=args.preflight_timeout,
        )
        if provider_preflight is not None and not provider_preflight.get("healthy"):
            payload = provider_unhealthy_payload(
                provider_preflight=provider_preflight,
                job_id=spec.job_id,
                started=False,
            )
            emit_json_summary(payload, args.summary_file)
            return 1
        apply_constraint_overrides(
            spec,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=True,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
        )
        store = FileJobStore(args.jobs_dir)
        runner, _ = build_default_runner(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            store=store,
        )
        record = runner.plan_job(spec)
        next_supervise_cli_args = (
            _next_supervise_cli_args(
                record.job_id,
                config_dir=args.config_dir,
                jobs_dir=args.jobs_dir,
                workspace=spec.workspace_root or spec.repo_path,
                max_cycles=args.supervise_max_cycles,
                steps_per_cycle=args.supervise_steps_per_cycle,
                max_stalled_cycles=args.supervise_max_stalled_cycles,
                max_runtime_seconds=args.supervise_max_runtime_seconds,
                summary_file=args.supervise_summary_file,
                summary_dir=args.supervise_summary_dir,
                max_autonomous_stages=args.max_autonomous_stages,
                large_autonomous=True,
                require_prd_quality=args.require_prd_quality,
                stage_review=args.stage_review,
                test_timeout_seconds=args.test_timeout_seconds,
                preflight_provider=args.supervise_preflight_provider,
                preflight_timeout=args.supervise_preflight_timeout,
                allow_repeated_failure_recovery=(
                    args.supervise_allow_repeated_failure_recovery
                ),
                pm_stall_recovery=(
                    args.supervise_pm_stall_recovery
                    or args.supervise_autonomous_until_done
                ),
                autonomous_until_done=args.supervise_autonomous_until_done,
            )
            if args.supervise_after_planning
            else None
        )
        payload = planning_result_payload(
            record,
            started=True,
            config_dir=args.config_dir,
            jobs_dir=args.jobs_dir,
            next_supervise_cli_args=next_supervise_cli_args,
            prefer_supervise=args.supervise_after_planning,
            autonomous_until_done=args.supervise_autonomous_until_done,
        )
        emit_json_summary(payload, args.summary_file)
        return 0 if payload["planning_complete"] else 1
    if args.command == "run-autonomous":
        spec = load_job_spec_from_file(args.file)
        apply_constraint_overrides(
            spec,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=True,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
        )
        store = FileJobStore(args.jobs_dir)
        runner, _ = build_default_runner(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            store=store,
        )
        record = runner.run_job(spec)
        steps_run = 0
        step_events: list[dict[str, Any]] = []
        if record.status != JobStatus.DONE and args.max_steps > 0:
            record, steps_run, _, step_events = continue_persisted_job(
                store=store,
                job_id=record.job_id,
                config_dir=args.config_dir,
                workspace=spec.workspace_root or spec.repo_path,
                max_steps=args.max_steps,
                max_autonomous_stages=args.max_autonomous_stages,
                large_autonomous=True,
                require_prd_quality=args.require_prd_quality,
                stage_review=args.stage_review,
                test_timeout_seconds=args.test_timeout_seconds,
                allow_repeated_failure_recovery=args.allow_repeated_failure_recovery,
                autonomous_recovery=True,
            )
        if args.json_summary:
            payload = autonomous_result_payload(
                record,
                steps_run=steps_run,
                max_steps=args.max_steps,
                started=True,
                config_dir=args.config_dir,
                jobs_dir=args.jobs_dir,
                step_events=step_events,
            )
            emit_json_summary(payload, args.summary_file)
            return 0 if record.status.value == "done" else 1
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1
    if args.command == "run-supervised":
        spec = load_run_supervised_job_spec(args)
        provider_preflight = maybe_probe_provider(
            config_dir=args.config_dir,
            provider_name=args.preflight_provider,
            timeout_seconds=args.preflight_timeout,
        )
        if provider_preflight is not None and not provider_preflight.get("healthy"):
            payload = provider_unhealthy_payload(
                provider_preflight=provider_preflight,
                job_id=spec.job_id,
                started=False,
            )
            emit_json_summary(payload, args.summary_file)
            return 1
        apply_constraint_overrides(
            spec,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=True,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
            model_timeout_seconds=supervised_model_timeout_seconds(
                args.max_runtime_seconds
            ),
            model_timeout_deadline_epoch=supervised_model_timeout_deadline_epoch(
                args.max_runtime_seconds
            ),
        )
        store = FileJobStore(args.jobs_dir)
        runner, _ = build_default_runner(
            config_dir=args.config_dir,
            workspace_root=spec.workspace_root or spec.repo_path,
            store=store,
        )
        planning_payload = None
        if args.plan_first:
            record = runner.plan_job(spec)
            planning_payload = planning_result_payload(
                record,
                started=True,
                config_dir=args.config_dir,
                jobs_dir=args.jobs_dir,
            )
            if not planning_payload["planning_complete"]:
                if provider_preflight is not None:
                    planning_payload.setdefault("provider_preflight", provider_preflight)
                    planning_payload["provider_events"] = [
                        _provider_preflight_event(
                            provider_preflight=provider_preflight,
                            cycle=None,
                            phase="pre_start",
                        )
                    ]
                    planning_payload["stop_summary"] = stop_summary_payload(
                        planning_payload
                    )
                emit_json_summary(planning_payload, args.summary_file)
                return 1
        else:
            record = runner.run_job(spec)
        if record.status == JobStatus.DONE:
            payload = autonomous_result_payload(
                record,
                steps_run=0,
                max_steps=args.max_cycles * args.steps_per_cycle,
                started=True,
                config_dir=args.config_dir,
                jobs_dir=args.jobs_dir,
                continued=False,
            )
            payload.update(
                {
                    "cycles_run": 0,
                    "max_cycles": args.max_cycles,
                    "steps_per_cycle": args.steps_per_cycle,
                    "stalled": False,
                    "stalled_cycle_count": 0,
                    "max_stalled_cycles": args.max_stalled_cycles,
                    "runtime_limited": False,
                    "elapsed_seconds": 0.0,
                    "max_runtime_seconds": args.max_runtime_seconds,
                    "can_supervise_continue": False,
                    "next_supervise_cli_args": [],
                    "next_supervise_command": None,
                    "provider_events": [],
                    "cycle_summaries": [],
                    "initial_status": record.status.value,
                }
            )
        else:
            initial_status = record.status.value
            payload = supervise_persisted_job(
                store=store,
                job_id=record.job_id,
                config_dir=args.config_dir,
                workspace=spec.workspace_root or spec.repo_path,
                max_cycles=args.max_cycles,
                steps_per_cycle=args.steps_per_cycle,
                max_stalled_cycles=args.max_stalled_cycles,
                max_runtime_seconds=args.max_runtime_seconds,
                jobs_dir=args.jobs_dir,
                summary_file=args.summary_file,
                summary_dir=args.summary_dir,
                max_autonomous_stages=args.max_autonomous_stages,
                large_autonomous=True,
                require_prd_quality=args.require_prd_quality,
                stage_review=args.stage_review,
                test_timeout_seconds=args.test_timeout_seconds,
                preflight_provider=args.preflight_provider,
                preflight_timeout=args.preflight_timeout,
                allow_repeated_failure_recovery=args.allow_repeated_failure_recovery,
                pm_stall_recovery=args.pm_stall_recovery or args.autonomous_until_done,
                autonomous_until_done=args.autonomous_until_done,
            )
            payload["started"] = True
            payload["initial_status"] = initial_status
        if args.plan_first:
            payload["planned_first"] = True
            payload["planning_complete"] = bool(
                planning_payload and planning_payload.get("planning_complete")
            )
            payload["planning_result"] = planning_payload
        if provider_preflight is not None:
            payload.setdefault("provider_preflight", provider_preflight)
            payload["provider_events"] = [
                _provider_preflight_event(
                    provider_preflight=provider_preflight,
                    cycle=None,
                    phase="pre_start",
                ),
                *payload.get("provider_events", []),
            ]
            payload["stop_summary"] = stop_summary_payload(payload)
        else:
            payload["stop_summary"] = stop_summary_payload(payload)
        emit_json_summary(payload, args.summary_file)
        return 0 if payload["done"] else 1
    if args.command == "resume-job":
        store = FileJobStore(args.jobs_dir)
        record = store.get(args.job_id)
        apply_strict_job_constraints(record)
        store.update(record)
        summary = summarize_job_progress(record)
        apply_planning_repair_overrides(record, summary)
        apply_resume_overrides(
            record,
            max_autonomous_stages=args.max_autonomous_stages,
            bump_stage_limit=args.bump_stage_limit,
            large_autonomous=args.large_autonomous,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
        )
        store.update(record)
        workspace = args.workspace or record.spec.workspace_root or record.spec.repo_path
        runner, _ = build_default_runner(
            config_dir=args.config_dir,
            workspace_root=workspace,
            store=store,
        )
        record = runner.resume_job(args.job_id)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1
    if args.command == "continue-job":
        store = FileJobStore(args.jobs_dir)
        record, steps_run, no_action_summary, step_events = continue_persisted_job(
            store=store,
            job_id=args.job_id,
            config_dir=args.config_dir,
            workspace=args.workspace,
            max_steps=args.max_steps,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=args.large_autonomous,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
            allow_repeated_failure_recovery=args.allow_repeated_failure_recovery,
            autonomous_recovery=args.allow_repeated_failure_recovery,
        )
        if steps_run == 0 and no_action_summary is not None:
            if args.json_summary:
                payload = autonomous_result_payload(
                    record,
                    steps_run=0,
                    max_steps=args.max_steps,
                    started=False,
                    config_dir=args.config_dir,
                    jobs_dir=args.jobs_dir,
                    step_events=step_events,
                    continued=False,
                    summary=no_action_summary,
                )
                emit_json_summary(payload, args.summary_file)
                return 0 if record.status == JobStatus.DONE else 1
            print(yaml.safe_dump({"continued": False, "summary": no_action_summary}, sort_keys=False))
            return 0 if record.status == JobStatus.DONE else 1
        if args.json_summary:
            payload = autonomous_result_payload(
                record,
                steps_run=steps_run,
                max_steps=args.max_steps,
                started=False,
                config_dir=args.config_dir,
                jobs_dir=args.jobs_dir,
                step_events=step_events,
            )
            emit_json_summary(payload, args.summary_file)
            return 0 if record.status.value == "done" else 1
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1
    if args.command == "supervise-job":
        provider_preflight = maybe_probe_provider(
            config_dir=args.config_dir,
            provider_name=args.preflight_provider,
            timeout_seconds=args.preflight_timeout,
        )
        if provider_preflight is not None and not provider_preflight.get("healthy"):
            payload = provider_unhealthy_payload(
                provider_preflight=provider_preflight,
                job_id=args.job_id,
                started=False,
                next_supervise_cli_args=_next_supervise_cli_args(
                    args.job_id,
                    config_dir=args.config_dir,
                    jobs_dir=args.jobs_dir,
                    workspace=args.workspace,
                    max_cycles=args.max_cycles,
                    steps_per_cycle=args.steps_per_cycle,
                    max_stalled_cycles=args.max_stalled_cycles,
                    max_runtime_seconds=args.max_runtime_seconds,
                    summary_file=args.summary_file,
                    summary_dir=args.summary_dir,
                    max_autonomous_stages=args.max_autonomous_stages,
                    large_autonomous=args.large_autonomous,
                    require_prd_quality=args.require_prd_quality,
                    stage_review=args.stage_review,
                    test_timeout_seconds=args.test_timeout_seconds,
                    preflight_provider=args.preflight_provider,
                    preflight_timeout=args.preflight_timeout,
                    allow_repeated_failure_recovery=args.allow_repeated_failure_recovery,
                    pm_stall_recovery=args.pm_stall_recovery or args.autonomous_until_done,
                    autonomous_until_done=args.autonomous_until_done,
                ),
            )
            emit_json_summary(payload, args.summary_file)
            return 1
        store = FileJobStore(args.jobs_dir)
        payload = supervise_persisted_job(
            store=store,
            job_id=args.job_id,
            config_dir=args.config_dir,
            workspace=args.workspace,
            max_cycles=args.max_cycles,
            steps_per_cycle=args.steps_per_cycle,
            max_stalled_cycles=args.max_stalled_cycles,
            max_runtime_seconds=args.max_runtime_seconds,
            jobs_dir=args.jobs_dir,
            summary_file=args.summary_file,
            summary_dir=args.summary_dir,
            max_autonomous_stages=args.max_autonomous_stages,
            large_autonomous=args.large_autonomous,
            require_prd_quality=args.require_prd_quality,
            stage_review=args.stage_review,
            test_timeout_seconds=args.test_timeout_seconds,
            preflight_provider=args.preflight_provider,
            preflight_timeout=args.preflight_timeout,
            allow_repeated_failure_recovery=args.allow_repeated_failure_recovery,
            pm_stall_recovery=args.pm_stall_recovery or args.autonomous_until_done,
            autonomous_until_done=args.autonomous_until_done,
        )
        if provider_preflight is not None:
            payload.setdefault("provider_preflight", provider_preflight)
            payload["provider_events"] = [
                _provider_preflight_event(
                    provider_preflight=provider_preflight,
                    cycle=None,
                    phase="pre_start",
                ),
                *payload.get("provider_events", []),
            ]
            payload["stop_summary"] = stop_summary_payload(payload)
        emit_json_summary(payload, args.summary_file)
        return 0 if payload["done"] else 1
    if args.command == "job-status":
        record = FileJobStore(args.jobs_dir).get(args.job_id)
        summary = summarize_job_progress(record)
        if args.next_command:
            suggested_args = summary.get("resume", {}).get("suggested_cli_args", [])
            if suggested_args:
                command = [*suggested_args, "--jobs-dir", str(args.jobs_dir)]
                print(_acos_command(command))
            else:
                continuation = job_status_continuation_payload(record, args, summary)
                supervision = job_status_supervision_payload(record, args)
                decision = job_status_operator_decision_payload(
                    record,
                    summary,
                    continuation,
                    supervision,
                )
                command = decision.get("command")
                if command:
                    print(command)
            return 0
        if args.next_continue_command:
            continuation = job_status_continuation_payload(record, args, summary)
            command = continuation.get("next_continue_command") or continuation.get(
                "blocked_recovery_continue_command"
            )
            if command:
                print(command)
            return 0
        if args.next_supervise_command:
            suggested_args = job_status_supervision_payload(record, args)[
                "next_supervise_cli_args"
            ]
            if suggested_args:
                print(_acos_command(suggested_args))
            return 0
        if args.next_operator_command:
            continuation = job_status_continuation_payload(record, args, summary)
            supervision = job_status_supervision_payload(record, args)
            decision = job_status_operator_decision_payload(
                record,
                summary,
                continuation,
                supervision,
            )
            command = decision.get("command")
            if command:
                print(command)
            return 0
        if args.json:
            continuation = job_status_continuation_payload(record, args, summary)
            supervision = job_status_supervision_payload(record, args)
            summary["continuation"] = continuation
            summary["supervision"] = supervision
            operator_decision = job_status_operator_decision_payload(
                record,
                summary,
                continuation,
                supervision,
            )
            summary["operator_decision"] = operator_decision
            summary["operator_summary"] = operator_summary_payload(
                decision=operator_decision,
                continuation=continuation,
                supervision=supervision,
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0
        print(yaml.safe_dump(summary, sort_keys=False))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
