"""ACOS job orchestration engine."""

from __future__ import annotations

import hashlib
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.agents.runner import AgentRunner
from packages.llm.budget import estimate_tokens
from packages.llm.client import LLMClient
from packages.llm.errors import AdapterError, StructuredOutputError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.approval import ApprovalGateway
from packages.orchestrator.completion_verifier import DefinitionOfDoneVerifier
from packages.orchestrator.context_builder import ContextBuilder
from packages.orchestrator.execution_contracts import synthesize_job_metadata_from_prd
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.progress import summarize_job_progress
from packages.orchestrator.quality_gates import (
    QualityGateError,
    artifact_path_exists,
    ensure_fixer_safe,
    ensure_reviews_pass,
    ensure_test_patch_quality,
    invalid_artifact_paths,
    valid_artifact_paths,
)
from packages.orchestrator.recovery_executor import RecoveryExecutor
from packages.orchestrator.recovery_governor import (
    RecoveryGovernor,
    is_hard_terminal_status,
    is_recoverable_status,
    is_waiting_status,
)
from packages.orchestrator.runtime import RuntimeManager
from packages.orchestrator.scaffolds import build_scaffold
from packages.orchestrator.states import apply_transition
from packages.orchestrator.task_graph_validation import (
    TASK_GRAPH_VALIDATION_CONTEXT_KEYS as TASK_GRAPH_VALIDATION_CONTEXT_KEYS_SOURCE,
    TASK_GRAPH_VALIDATION_DETAIL_KEYS as TASK_GRAPH_VALIDATION_DETAIL_KEYS_SOURCE,
)
from packages.schemas.approvals import ApprovalStatus, PolicyAction
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FailureDiagnosis,
    FilePatch,
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
    FailureClassification,
    FailureRetryMode,
    FixStatus,
    ImplementationStatus,
    JobStatus,
    ReviewDecision,
    TaskComplexity,
    TestWriterStatus,
)
from packages.schemas.runtime import RuntimeIssueType
from packages.schemas.tasks import PlannedTask, TaskGraph


def _disable_mock_fallback_models(registry: ModelRegistry) -> None:
    for agent in registry.agents.values():
        agent.fallback_models = [
            model_key
            for model_key in agent.fallback_models
            if registry.get_provider(registry.get_model(model_key).provider).type.value != "mock"
        ]


class JobWaitingForApproval(RuntimeError):
    """Internal control-flow marker for durable approval waits."""


class JobRunner:
    """Run ACOS jobs across explicit role phases."""

    CONTEXT_ONLY_ROLES = {
        "pm",
        "architect",
        "planner",
    }
    IMPLEMENTATION_TASK_ROLES = {"implementer", "scaffold"}
    TEST_TASK_ROLES = {"test_writer"}
    PROJECT_SETUP_REQUIRED_ARTIFACTS = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/tests/test_project_setup.py",
        "frontend/package.json",
        "frontend/vite.config.js",
        "frontend/src/main.tsx",
        "frontend/src/App.tsx",
        "shared/.gitkeep",
        ".gitignore",
        "package.json",
        "README.md",
        ".env.example",
    ]
    PROJECT_SETUP_KEYWORDS = (
        "project-scaffold",
        "project scaffold",
        "project-setup",
        "project setup",
        "verify-project-setup",
        "monorepo",
        "backend/frontend/shared",
        "backend frontend shared",
    )
    RECOVERY_METADATA_CONSTRAINT_KEYS = {
        "recovery_mode",
        "recovery_strategy",
        "recovery_next_actor",
        "recovery_next_status",
        "recovery_reason",
        "recovery_failed_task_id",
        "recovery_failed_stage",
        "recovery_attempt",
        "recovery_step_count",
    }
    FILE_RECOVERY_CONSTRAINT_KEYS = {
        "deterministic_creation_attempted",
        "deterministically_created_files",
        "failed_patch_operation",
        "failed_patch_path",
        "failed_patch_role",
        "empty_artifacts",
        "missing_artifacts",
        "missing_target_file",
        "patch_operation_hint",
        "recreate_target_files_attempt",
        "return_to_role",
    }
    SEMANTIC_ANCHOR_TOKENS = {
        "auth",
        "billing",
        "crud",
        "download",
        "email",
        "oauth",
        "payment",
        "permission",
        "practice",
        "progress",
        "quiz",
        "search",
        "session",
        "upload",
    }
    CRUD_OPERATION_TOKENS = {"create", "read", "update", "delete"}
    BACKEND_SURFACE_TOKENS = {
        "api",
        "apis",
        "backend",
        "crud",
        "database",
        "db",
        "endpoint",
        "endpoints",
        "fastapi",
        "persistence",
        "route",
        "routes",
        "server",
        "service",
        "services",
    }
    FRONTEND_SURFACE_TOKENS = {
        "browser",
        "client",
        "component",
        "components",
        "css",
        "form",
        "forms",
        "frontend",
        "jsx",
        "mobile",
        "page",
        "pages",
        "react",
        "screen",
        "tsx",
        "ui",
        "view",
        "views",
        "vite",
    }
    SHARED_SURFACE_TOKENS = {
        "contract",
        "contracts",
        "dto",
        "interface",
        "interfaces",
        "model",
        "models",
        "schema",
        "schemas",
        "shared",
        "type",
        "types",
    }
    IMPLEMENTATION_ARTIFACT_GENERIC_TOKENS = {
        "api",
        "app",
        "backend",
        "client",
        "common",
        "component",
        "contract",
        "css",
        "endpoint",
        "fastapi",
        "file",
        "frontend",
        "html",
        "implementation",
        "interface",
        "js",
        "jsx",
        "main",
        "model",
        "page",
        "py",
        "route",
        "schema",
        "server",
        "service",
        "shared",
        "source",
        "src",
        "ts",
        "tsx",
        "type",
        "ui",
        "view",
        "web",
    }
    TASK_GRAPH_VALIDATION_CONTEXT_KEYS = TASK_GRAPH_VALIDATION_CONTEXT_KEYS_SOURCE
    TASK_GRAPH_VALIDATION_DETAIL_KEYS = TASK_GRAPH_VALIDATION_DETAIL_KEYS_SOURCE

    def __init__(
        self,
        registry: ModelRegistry,
        policy: PolicyEngine,
        router: MCPRouter,
        store: InMemoryJobStore | None = None,
        model_router: ModelRouter | None = None,
        agent_runner: AgentRunner | None = None,
        approval_gateway: ApprovalGateway | None = None,
        runtime_manager: RuntimeManager | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.router = router
        self.store = store or InMemoryJobStore()
        if runtime_manager is not None:
            _disable_mock_fallback_models(registry)
        self.audit = AuditRecorder()
        self.context_builder = ContextBuilder()
        self.completion_verifier = DefinitionOfDoneVerifier()
        self.recovery_governor = RecoveryGovernor()
        self.recovery_executor = RecoveryExecutor(self.store)
        self.model_router = model_router or ModelRouter(registry)
        self.llm_client = LLMClient(registry, self.model_router)
        self.agent_runner = agent_runner or AgentRunner(
            llm_client=self.llm_client,
            registry=registry,
            mcp_router=router,
            policy_engine=policy,
            audit_recorder=self.audit,
        )
        self.approval_gateway = approval_gateway
        self.runtime_manager = runtime_manager
        self.max_attempts_per_task = 3
        self.max_same_failure_repeats = 2
        self.max_steps_per_agent = 6
        self._active_record: JobRecord | None = None

    def submit(self, spec: JobSpec) -> JobRecord:
        return self.store.create(spec)

    def get(self, job_id: str) -> JobRecord:
        return self.store.get(job_id)

    def list_jobs(self, statuses: list[JobStatus] | None = None) -> list[JobRecord]:
        return self.store.list_jobs(statuses=statuses)

    def get_events(self, job_id: str) -> list[Any]:
        return list(self.store.get(job_id).audit_events)

    def get_notifications(self, job_id: str) -> list[dict[str, Any]]:
        return self.store.list_notifications(job_id=job_id)

    def list_approvals(self, job_id: str | None = None) -> list[Any]:
        if self.approval_gateway is None:
            return []
        return self.approval_gateway.list_all(job_id=job_id)

    def pause_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        apply_transition(record, JobStatus.PAUSED)
        return self.store.update(record)

    def cancel_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        apply_transition(record, JobStatus.CANCELLED)
        return self.store.update(record)

    def run_job(self, spec: JobSpec) -> JobRecord:
        record = self.store.create(spec)
        return self._run_record(record, resume=False)

    def plan_job(self, spec: JobSpec) -> JobRecord:
        record = self.store.create(spec)
        return self._plan_record(record, resume=False)

    def resume_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        return self._run_record(record, resume=True)

    def run_until_done_or_hard_stop(
        self,
        spec_or_job_id: JobSpec | str,
        *,
        max_cycles: int = 1000,
    ) -> JobRecord:
        """Resume through recoverable failures until DONE, CANCELLED, hard stop, or wait."""

        if isinstance(spec_or_job_id, JobSpec):
            record = self.run_job(spec_or_job_id)
        else:
            record = self.resume_job(spec_or_job_id)
        cycles = 0
        while (
            not self._is_terminal_status(record.status)
            and not self._is_waiting_status(record.status)
            and cycles < max_cycles
        ):
            cycles += 1
            record = self.resume_job(record.job_id)
        record.runtime_state["run_until_done_or_hard_stop_cycles"] = cycles
        return self.store.update(record)

    def _plan_record(self, record: JobRecord, *, resume: bool) -> JobRecord:
        self._active_record = record
        try:
            if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                return record
            if resume and self._recover_record_if_needed(record):
                if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                    return self.store.update(record)
            if resume:
                self.recovery_executor.execute_until_ready(record)
                if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                    return self.store.update(record)
                self._consume_completed_recovery_plan(record)
            if not resume:
                self._prepare_branch(record)
            prd = self._load_or_refine_prd_for_autonomy(record)
            if prd is None:
                return self.store.update(record)
            architecture = self._load_or_run_role(
                record,
                "architect",
                ArchitecturePlan,
                "Design the system architecture",
                memory_key="architecture",
            )
            task_graph = self._load_or_repair_task_graph_for_autonomy(record, prd)
            if task_graph is None:
                return self.store.update(record)
            record.outputs["prd"] = prd.model_dump()
            record.outputs["architecture"] = architecture.model_dump()
            record.outputs["task_graph"] = task_graph.model_dump()
            record.outputs["planning_only"] = {
                "complete": True,
                "ready_for_implementation": True,
            }
            record.last_error = None
            self._clear_active_recovery_state(record)
            return self.store.update(record)
        except QualityGateError as exc:
            self._recover_record(
                record,
                error=self._quality_gate_recovery_error(exc),
            )
            return self.store.update(record)
        except AdapterError as exc:
            return self._handle_provider_adapter_error(record, exc)
        except StructuredOutputError as exc:
            self._recover_record(record, error=str(exc))
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            self._recover_record(record, error=str(exc))
            return self.store.update(record)
        finally:
            self._active_record = None

    def _run_record(self, record: JobRecord, *, resume: bool) -> JobRecord:
        self._active_record = record
        try:
            if resume and self._resume_approval_if_ready(record):
                if record.status == JobStatus.BLOCKED:
                    return self.store.update(record)
            if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                return record
            if resume and self._recover_record_if_needed(record):
                if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                    return self.store.update(record)
            if resume:
                self._consume_completed_recovery_plan(record)
            if not resume:
                self._prepare_branch(record)
            prd = self._load_or_refine_prd_for_autonomy(record)
            if prd is None:
                return self.store.update(record)
            architecture = self._load_or_run_role(
                record,
                "architect",
                ArchitecturePlan,
                "Design the system architecture",
                memory_key="architecture",
            )
            task_graph = self._load_or_repair_task_graph_for_autonomy(record, prd)
            if task_graph is None:
                return self.store.update(record)
            primary_task = self._choose_primary_task(task_graph)
            scaffold = build_scaffold(str(self._constraints(record).get("scaffold", "")))
            if scaffold is not None:
                implementation, test_writer = scaffold
                implementation_results = [implementation]
                test_writer_results = [test_writer]
                stage_results: list[dict[str, Any]] = []
                apply_transition(record, JobStatus.IMPLEMENTING)
                self._apply_patches(record, "implementer", implementation.patches)
                apply_transition(record, JobStatus.WRITING_TESTS)
                self._apply_patches(record, "test_writer", test_writer.patches)
                test_result = self._run_tests_with_fixes(record, primary_task)
            else:
                (
                    implementation_results,
                    test_writer_results,
                    test_result,
                    stage_results,
                ) = self._run_autonomous_task_loop(record, task_graph)
                implementation = self._combine_implementation_results(implementation_results)
                test_writer = self._combine_test_writer_results(test_writer_results)
            if self._has_pending_recovery_plan(record):
                return self.store.update(record)
            if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                self._recover_record(record, error=record.last_error)
                return self.store.update(record)
            if not self._constraint_flag(record, "skip_review"):
                review, security_review = self._run_review_cycle(record, primary_task)
                if self._has_pending_recovery_plan(record):
                    return self.store.update(record)
                if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                    self._recover_record(record, error=record.last_error)
                    return self.store.update(record)
                test_result = self._run_tests_with_fixes(record, primary_task)
                if self._has_pending_recovery_plan(record):
                    return self.store.update(record)
            else:
                if record.status != JobStatus.TESTING:
                    apply_transition(record, JobStatus.REVIEWING)
            if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                self._recover_record(record, error=record.last_error)
                return self.store.update(record)
            if not test_result.success:
                self._recover_record(record, error="max_attempts_exceeded")
                return self.store.update(record)
            if not self._validate_completion_integrity(record, task_graph, test_result):
                return self.store.update(record)
            if self._constraint_flag(record, "skip_release"):
                record.outputs["prd"] = prd.model_dump()
                record.outputs["architecture"] = architecture.model_dump()
                record.outputs["task_graph"] = task_graph.model_dump()
                record.outputs["implementation"] = implementation.model_dump()
                record.outputs["test_writer"] = test_writer.model_dump()
                record.outputs["implementation_task_count"] = len(implementation_results)
                record.outputs["test_writer_task_count"] = len(test_writer_results)
                record.outputs["autonomous_stages"] = stage_results
                record.outputs["test_run"] = test_result.model_dump()
                apply_transition(record, JobStatus.FINALIZING)
                record.last_error = None
                self._clear_active_recovery_state(record)
                apply_transition(record, JobStatus.DONE)
                return self.store.update(record)
            summary = self._run_structured_role(
                record,
                "summarizer",
                SummaryResult,
                "Summarize the completed job and memory",
                task=primary_task,
                logs=[test_result.output_excerpt],
            )
            self._write_memory_entries(record, summary)
            release = self._run_structured_role(
                record,
                "release_manager",
                ReleaseResult,
                "Prepare the final release artifact",
                task=primary_task,
            )
            self._release(record, release)
            record.outputs["prd"] = prd.model_dump()
            record.outputs["architecture"] = architecture.model_dump()
            record.outputs["task_graph"] = task_graph.model_dump()
            record.outputs["implementation"] = implementation.model_dump()
            record.outputs["test_writer"] = test_writer.model_dump()
            record.outputs["autonomous_stages"] = stage_results
            record.outputs["test_run"] = test_result.model_dump()
            record.outputs["summary"] = summary.model_dump()
            record.last_error = None
            self._clear_active_recovery_state(record)
            apply_transition(record, JobStatus.DONE)
            return self.store.update(record)
        except JobWaitingForApproval:
            return self.store.update(record)
        except QualityGateError as exc:
            self._recover_record(
                record,
                error=self._quality_gate_recovery_error(exc),
            )
            return self.store.update(record)
        except AdapterError as exc:
            return self._handle_provider_adapter_error(record, exc)
        except StructuredOutputError as exc:
            self._recover_record(record, error=str(exc))
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            self._recover_record(record, error=str(exc))
            return self.store.update(record)
        finally:
            self._active_record = None

    def _phase_for_role(self, role: str) -> JobStatus:
        return {
            "pm": JobStatus.ANALYZING,
            "architect": JobStatus.DESIGNING,
            "planner": JobStatus.PLANNING,
            "implementer": JobStatus.IMPLEMENTING,
            "scaffold": JobStatus.IMPLEMENTING,
            "test_writer": JobStatus.WRITING_TESTS,
            "diagnoser": JobStatus.DIAGNOSING,
            "reviewer": JobStatus.REVIEWING,
            "security_reviewer": JobStatus.REVIEWING,
            "fixer": JobStatus.FIXING,
            "release_manager": JobStatus.FINALIZING,
            "summarizer": JobStatus.FINALIZING,
        }[role]

    def _reset_blocked_planning_resume(
        self,
        record: JobRecord,
        *,
        target_status: JobStatus,
        last_error_prefix: str,
    ) -> None:
        if record.status != JobStatus.BLOCKED:
            return
        if not isinstance(record.last_error, str) or not record.last_error.startswith(
            last_error_prefix
        ):
            return
        record.status = target_status
        record.history.append(target_status)
        record.last_error = None
        record.updated_at = datetime.now(timezone.utc)
        self.store.update(record)

    def _is_terminal_status(self, status: JobStatus) -> bool:
        return is_hard_terminal_status(status)

    def _is_waiting_status(self, status: JobStatus) -> bool:
        return is_waiting_status(status)

    def _is_recoverable_status(self, status: JobStatus) -> bool:
        return is_recoverable_status(status)

    def _recover_record(
        self,
        record: JobRecord,
        *,
        error: str | None = None,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(self, "recovery_governor"):
            self.recovery_governor = RecoveryGovernor()
        self.recovery_governor.recover(
            record,
            error=error,
            runtime_state=runtime_state,
        )
        if not hasattr(self, "recovery_executor"):
            self.recovery_executor = RecoveryExecutor(self.store)
        self.recovery_executor.execute_until_ready(record)
        self.store.update(record)

    def _recover_record_if_needed(self, record: JobRecord) -> bool:
        if not self._is_recoverable_status(record.status):
            return False
        self._recover_record(record)
        return True

    @staticmethod
    def _has_pending_recovery_plan(record: JobRecord) -> bool:
        plan = record.runtime_state.get("recovery_plan")
        if not isinstance(plan, dict):
            return False
        if plan.get("status") == "completed" and plan.get("consumed_by_runner") is True:
            return False
        next_status = plan.get("next_status")
        return isinstance(next_status, str) and record.status.value == next_status

    @staticmethod
    def _consume_completed_recovery_plan(record: JobRecord) -> None:
        plan = record.runtime_state.get("recovery_plan")
        if isinstance(plan, dict) and plan.get("status") == "completed":
            plan["consumed_by_runner"] = True
            if JobRunner._completed_file_recovery_resolved(record, plan):
                JobRunner._clear_resolved_file_recovery_constraints(record)

    @staticmethod
    def _completed_file_recovery_resolved(
        record: JobRecord,
        plan: dict[str, Any],
    ) -> bool:
        plan_constraints = plan.get("constraints")
        if not isinstance(plan_constraints, dict):
            plan_constraints = {}
        metadata_constraints = record.spec.metadata.get("constraints")
        if not isinstance(metadata_constraints, dict):
            metadata_constraints = {}
        file_recovery_marker_keys = {
            "empty_artifacts",
            "missing_target_file",
            "patch_operation_hint",
            "failed_patch_path",
            "failed_patch_operation",
        }
        if not any(
            key in plan_constraints
            or key in metadata_constraints
            or key in record.runtime_state
            for key in file_recovery_marker_keys
        ):
            return False
        missing_artifacts = plan_constraints.get("missing_artifacts")
        empty_artifacts = plan_constraints.get("empty_artifacts")
        if isinstance(missing_artifacts, list) or isinstance(empty_artifacts, list):
            missing_resolved = (
                not isinstance(missing_artifacts, list) or len(missing_artifacts) == 0
            )
            empty_resolved = (
                not isinstance(empty_artifacts, list) or len(empty_artifacts) == 0
            )
            return missing_resolved and empty_resolved
        candidates: list[str] = []
        for source in (
            plan_constraints.get("missing_target_file"),
            metadata_constraints.get("missing_target_file"),
            record.runtime_state.get("missing_target_file"),
            record.runtime_state.get("failed_patch_path"),
        ):
            if isinstance(source, str) and source.strip():
                candidates.append(source.strip())
        if not candidates:
            return False
        root = record.spec.workspace_root or record.spec.repo_path
        return all(artifact_path_exists(path, workspace_root=root) for path in candidates)

    @staticmethod
    def _clear_resolved_file_recovery_constraints(record: JobRecord) -> None:
        record.last_error = None
        record.runtime_state.pop("current_recovery_event", None)
        record.runtime_state.pop("last_recoverable_error", None)
        record.outputs.pop("last_recoverable_error", None)
        for key in JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS:
            record.runtime_state.pop(key, None)
        constraints = record.spec.metadata.get("constraints")
        if not isinstance(constraints, dict):
            return
        for key in (
            *JobRunner.RECOVERY_METADATA_CONSTRAINT_KEYS,
            *JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS,
        ):
            constraints.pop(key, None)

    def _should_pause_for_recovery(self, record: JobRecord) -> bool:
        return (
            self._has_pending_recovery_plan(record)
            or self._is_recoverable_status(record.status)
            or self._is_terminal_status(record.status)
            or self._is_waiting_status(record.status)
        )

    @staticmethod
    def _quality_gate_recovery_error(exc: QualityGateError) -> str:
        message = str(exc)
        lowered = message.lower()
        if RecoveryGovernor.is_policy_hard_stop(message):
            return message
        if "weaken tests" in lowered:
            return f"test_patch_quality_failed:{message}"
        if "required artifact" in lowered or "artifact" in lowered:
            return f"required_artifacts_missing:{message}"
        if "target file" in lowered:
            return f"target_files_missing:{message}"
        if "reviewer did not approve" in lowered or "security review did not approve" in lowered:
            return f"reviews_rejected:{message}"
        return f"quality_gate_recoverable:{message}"

    def _load_or_run_role(
        self,
        record: JobRecord,
        role: str,
        response_model: type,
        objective: str,
        *,
        memory_key: str | None = None,
    ) -> Any:
        if role in record.outputs:
            return response_model.model_validate(record.outputs[role])
        if memory_key is not None and memory_key in record.outputs:
            return response_model.model_validate(record.outputs[memory_key])
        output = self._run_structured_role(record, role, response_model, objective)
        if memory_key is not None:
            self._write_memory_item(record, role, memory_key, output.model_dump_json())
        self.store.update(record)
        return output

    def _run_structured_role(
        self,
        record: JobRecord,
        role: str,
        response_model: type,
        objective: str,
        task: PlannedTask | None = None,
        logs: list[str] | None = None,
        security_sensitive: bool = False,
    ) -> Any:
        task = self._task_with_recovery_targets(record, role, task)
        objective = self._objective_with_recovery_operation_hint(record, role, objective)
        apply_transition(record, self._phase_for_role(role))
        record.runtime_state["active_role"] = role
        record.runtime_state["active_objective"] = objective
        record.runtime_state["active_started_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        if task is not None:
            record.runtime_state["active_task_id"] = task.id
        else:
            record.runtime_state.pop("active_task_id", None)
        self.store.update(record)
        try:
            agent_cfg = self.registry.get_agent(role)
            relevant_files = self._gather_relevant_files(role, record=record, task=task)
            diff = (
                self._call_tool(role, "git_server.diff").get("diff", "")
                if self.policy.is_tool_allowed(role, "git_server.diff")
                else ""
            )
            memory_summaries = self._read_memory(role)
            preselection = self.model_router.select_model(
                RoutingContext(
                    role=role,
                    failure_count=record.failure_count,
                    same_test_failure_count=record.same_test_failure_count,
                    changed_files_count=len(relevant_files),
                    security_sensitive=security_sensitive,
                    context_tokens=0,
                )
            )
            selected_model = self.registry.get_model(preselection.model_key)
            model_timeout_seconds = self._constraint_float(
                record,
                "model_timeout_seconds",
                0.0,
            )
            model_timeout_seconds = self._effective_model_timeout_seconds(
                record,
                model_timeout_seconds,
            )
            record.runtime_state["active_model"] = preselection.model_key
            if model_timeout_seconds > 0:
                record.runtime_state["active_model_timeout_seconds"] = model_timeout_seconds
            else:
                record.runtime_state.pop("active_model_timeout_seconds", None)
            self.store.update(record)
            effective_logs = [
                *self._recovery_guidance_logs(record, role),
                *self._pm_stall_guidance_logs(record, role),
                *self._planning_repair_guidance_logs(record, role),
                *self._recovery_history_logs(record, role),
                *(logs or []),
            ]
            packet = self.context_builder.build(
                job_id=record.job_id,
                role=role,
                objective=objective,
                repo_path=record.spec.repo_path,
                request_text=record.spec.request_text,
                constraints=self._context_constraints(record),
                relevant_files=relevant_files,
                diff=diff,
                memory_summaries=memory_summaries,
                logs=effective_logs,
                token_budget=agent_cfg.context_budget_tokens,
                agent_config=agent_cfg,
                selected_model=selected_model,
                task=task,
                metadata={
                    "output_schema": agent_cfg.output_schema,
                    "retrieval_trace": record.runtime_state.get("retrieval_trace", []),
                },
            )
            routing_context = RoutingContext(
                role=role,
                failure_count=record.failure_count,
                same_test_failure_count=record.same_test_failure_count,
                changed_files_count=len(relevant_files),
                security_sensitive=security_sensitive,
                context_tokens=estimate_tokens(packet.render_text()),
            )
            output, selection, model_record = self.agent_runner.run(
                role=role,
                response_model=response_model,
                context_packet=packet,
                routing_context=routing_context,
                allowed_tools=self._allowed_tools_for_role(role),
                require_json_schema=agent_cfg.require_json_schema,
                max_steps=self.max_steps_per_agent,
                audit_events=record.audit_events,
                request_timeout_seconds=(
                    model_timeout_seconds if model_timeout_seconds > 0 else None
                ),
            )
        except Exception:
            self._clear_active_role_state(record)
            self.store.update(record)
            raise
        output = self._result_with_rewritten_missing_target_patches(record, role, output)
        record.outputs[role] = output.model_dump()
        record.outputs[f"{role}_model_selection"] = selection.model_dump()
        self._clear_active_role_state(record)
        self.store.update(record)
        return output

    @staticmethod
    def _clear_active_role_state(record: JobRecord) -> None:
        for key in (
            "active_role",
            "active_objective",
            "active_task_id",
            "active_model",
            "active_model_timeout_seconds",
            "active_started_at",
        ):
            record.runtime_state.pop(key, None)

    def _gather_relevant_files(
        self,
        role: str,
        *,
        record: JobRecord | None = None,
        task: PlannedTask | None = None,
    ) -> dict[str, str]:
        files: dict[str, str] = {}
        trace: list[dict[str, str]] = []
        can_tree = self.policy.is_tool_allowed(role, "repo_server.repo_tree")
        can_read = self.policy.is_tool_allowed(role, "repo_server.read_file")
        repo_files: list[str] = []
        candidate_reasons: dict[str, list[str]] = {}

        def add_candidate(path: object, reason: str) -> None:
            normalized = self._normalize_context_path(path, record)
            if normalized is None:
                return
            candidate_reasons.setdefault(normalized, []).append(reason)

        if task is not None:
            for path in task.target_files:
                add_candidate(path, "task.target_files")
            for path in task.required_artifacts:
                if self._looks_like_context_file(path):
                    add_candidate(path, "task.required_artifacts")

        if record is not None:
            missing_target_file = self._recovery_missing_target_file(record)
            if role == "test_writer" and missing_target_file:
                add_candidate(missing_target_file, "recovery.missing_target_file")
            for path in self._failure_context_paths(record):
                add_candidate(path, "failure_log")

        if self.policy.is_tool_allowed("release_manager", "git_server.status"):
            try:
                status = self._call_tool("release_manager", "git_server.status")
            except Exception as exc:
                trace.append(
                    {
                        "path": "__git_status__",
                        "reason": "git.modified_files",
                        "action": f"skipped:{exc}",
                    }
                )
            else:
                for path in status.get("modified_files", []):
                    add_candidate(path, "git.modified_files")

        if can_tree:
            try:
                repo_files = [
                    str(path)
                    for path in self._call_tool(role, "repo_server.repo_tree").get("files", [])
                ]
            except Exception as exc:
                trace.append(
                    {
                        "path": "__repo_map__",
                        "reason": "repo_server.repo_tree",
                        "action": f"skipped:{exc}",
                    }
                )
            else:
                repo_file_set = set(repo_files)
                for path in list(candidate_reasons):
                    if path not in repo_file_set:
                        trace.append(
                            {
                                "path": path,
                                "reason": ",".join(candidate_reasons[path]),
                                "action": "candidate_not_in_repo_map",
                            }
                        )
                if len(candidate_reasons) < 3:
                    for path in self._prioritized_context_files(repo_files):
                        add_candidate(path, "repo_map.priority")

        if (
            record is not None
            and self._constraints(record).get("expand_context") is True
            and self.policy.is_tool_allowed(role, "repo_server.search_text")
        ):
            for query in self._context_search_queries(record)[:4]:
                try:
                    search_payload = self._call_tool(
                        role,
                        "repo_server.search_text",
                        query=query,
                        max_results=8,
                        context_lines=2,
                    )
                except Exception as exc:
                    trace.append(
                        {
                            "path": "__search__",
                            "reason": f"search:{query}",
                            "action": f"skipped:{exc}",
                        }
                    )
                    continue
                for match in search_payload.get("matches", []):
                    if isinstance(match, dict):
                        add_candidate(match.get("path"), f"search:{query}")
                        trace.append(
                            {
                                "path": str(match.get("path", "")),
                                "reason": f"search:{query}",
                                "action": f"search_hit:{match.get('line_number')}",
                            }
                        )

        agent_cfg = self.registry.get_agent(role)
        max_files = self._context_file_budget(agent_cfg.context_budget_tokens)
        candidates = self._prioritized_context_files(
            list(candidate_reasons),
            limit=max_files,
        )

        if can_read:
            for path in candidates:
                try:
                    payload = self._call_tool(role, "repo_server.read_file", path=path)
                except Exception as exc:
                    trace.append(
                        {
                            "path": path,
                            "reason": ",".join(candidate_reasons.get(path, [])),
                            "action": f"read_failed:{exc}",
                        }
                    )
                    continue
                files[path] = str(payload["content"])
                trace.append(
                    {
                        "path": path,
                        "reason": ",".join(candidate_reasons.get(path, [])),
                        "action": "read_file",
                    }
                )

        if repo_files:
            files["__repo_map__.txt"] = "\n".join(repo_files)
            trace.append(
                {
                    "path": "__repo_map__.txt",
                    "reason": "repo_server.repo_tree",
                    "action": "repo_map_only",
                }
            )
        elif candidates and not can_read:
            files["__repo_map__.txt"] = "\n".join(candidates)

        if record is not None and role == "test_writer":
            missing_target_file = self._recovery_missing_target_file(record)
            if missing_target_file and missing_target_file not in files:
                files[missing_target_file] = (
                    "[MISSING_TARGET_FILE]\n"
                    f"path={missing_target_file}\n"
                    "required_patch_operation=create\n"
                    "Do not return update for this file until it exists.\n"
                )
                trace.append(
                    {
                        "path": missing_target_file,
                        "reason": "recovery.missing_target_file",
                        "action": "missing_file_context",
                    }
                )

        if trace:
            files["__retrieval_trace__.txt"] = "\n".join(
                f"{item['action']}:{item['path']}:{item['reason']}" for item in trace
            )
            if record is not None:
                record.runtime_state["retrieval_trace"] = trace[-100:]
                record.outputs["retrieval_trace"] = trace[-100:]
        return files

    @staticmethod
    def _context_file_budget(context_budget_tokens: int) -> int:
        if context_budget_tokens <= 0:
            return 8
        return max(8, min(40, context_budget_tokens // 1200))

    @staticmethod
    def _looks_like_context_file(path: object) -> bool:
        if not isinstance(path, str):
            return False
        stripped = path.strip()
        if not stripped:
            return False
        return bool(
            re.search(
                r"\.(py|js|jsx|ts|tsx|json|ya?ml|toml|md|css|html|txt)$",
                stripped,
                flags=re.IGNORECASE,
            )
            or "/" in stripped
            or "\\" in stripped
        )

    @staticmethod
    def _normalize_context_path(
        path: object,
        record: JobRecord | None,
    ) -> str | None:
        if not isinstance(path, str):
            return None
        raw = path.strip().strip("'\"")
        if not raw:
            return None
        normalized = raw.replace("\\", "/")
        if record is not None:
            try:
                workspace = Path(record.spec.repo_path).resolve().as_posix()
            except OSError:
                workspace = ""
            if workspace and normalized.startswith(f"{workspace}/"):
                normalized = normalized[len(workspace) + 1 :]
        normalized = normalized.removeprefix("./")
        if (
            not normalized
            or normalized.startswith("../")
            or normalized == ".."
            or Path(normalized).is_absolute()
            or re.match(r"^[A-Za-z]:/", normalized)
        ):
            return None
        return normalized

    def _failure_context_paths(self, record: JobRecord) -> list[str]:
        paths: list[str] = []
        diagnosis = record.outputs.get("failure_diagnosis")
        if isinstance(diagnosis, dict):
            for path in diagnosis.get("failed_files", []):
                if isinstance(path, str):
                    paths.append(path)
        diagnoses = record.outputs.get("failure_diagnoses")
        if isinstance(diagnoses, list):
            for item in diagnoses[-3:]:
                if isinstance(item, dict):
                    for path in item.get("failed_files", []):
                        if isinstance(path, str):
                            paths.append(path)
        for text in self._failure_context_texts(record):
            paths.extend(self._extract_failed_files(text))
        return self._unique_paths(paths)

    @staticmethod
    def _failure_context_texts(record: JobRecord) -> list[str]:
        texts: list[str] = []
        if isinstance(record.last_error, str):
            texts.append(record.last_error)
        recoverable_error = record.runtime_state.get("last_recoverable_error")
        if isinstance(recoverable_error, str):
            texts.append(recoverable_error)
        recovery_event = record.runtime_state.get("current_recovery_event")
        if isinstance(recovery_event, dict):
            for key in ("error", "reason"):
                value = recovery_event.get(key)
                if isinstance(value, str):
                    texts.append(value)
        test_run = record.outputs.get("test_run")
        if isinstance(test_run, dict):
            output_excerpt = test_run.get("output_excerpt")
            if isinstance(output_excerpt, str):
                texts.append(output_excerpt)
        stages = record.outputs.get("autonomous_stages")
        if isinstance(stages, list):
            for stage in stages[-3:]:
                if not isinstance(stage, dict):
                    continue
                for key in ("test_run", "post_review_test_run"):
                    test_payload = stage.get(key)
                    if isinstance(test_payload, dict) and isinstance(
                        test_payload.get("output_excerpt"),
                        str,
                    ):
                        texts.append(str(test_payload["output_excerpt"]))
        return texts

    def _context_search_queries(self, record: JobRecord) -> list[str]:
        queries: list[str] = []
        diagnosis = record.outputs.get("failure_diagnosis")
        if isinstance(diagnosis, dict):
            for key in ("failure_signature", "root_cause", "recommended_fix_strategy"):
                value = diagnosis.get(key)
                if isinstance(value, str) and value.strip():
                    queries.extend(self._terms_from_failure_text(value))
        for text in self._failure_context_texts(record):
            queries.extend(self._terms_from_failure_text(text))
        return list(dict.fromkeys(query for query in queries if query.strip()))

    @staticmethod
    def _terms_from_failure_text(text: str) -> list[str]:
        terms: list[str] = []
        for pattern in (
            r"cannot import name ['\"]([^'\"]+)['\"]",
            r"No module named ['\"]([^'\"]+)['\"]",
            r"NameError: name ['\"]([^'\"]+)['\"]",
            r"AttributeError: .* has no attribute ['\"]([^'\"]+)['\"]",
            r"([A-Za-z_][A-Za-z0-9_]{2,})",
        ):
            for match in re.findall(pattern, text):
                if isinstance(match, tuple):
                    match = next((item for item in match if item), "")
                if isinstance(match, str) and len(match) >= 3:
                    terms.append(match)
        return terms[:8]

    @staticmethod
    def _prioritized_context_files(paths: list[str], limit: int = 40) -> list[str]:
        priority_names = {
            "pyproject.toml",
            "package.json",
            "requirements.txt",
            "README.md",
            "Makefile",
            "pytest.ini",
            "vite.config.ts",
            "tsconfig.json",
        }

        def score(path: str) -> tuple[int, str]:
            name = Path(path).name
            if name in priority_names:
                return (0, path)
            if "/tests/" in f"/{path}" or path.startswith("tests/"):
                return (1, path)
            if path.endswith((".py", ".ts", ".tsx", ".js", ".jsx")):
                return (2, path)
            if path.endswith((".md", ".yaml", ".yml", ".toml", ".json")):
                return (3, path)
            return (4, path)

        unique = list(dict.fromkeys(str(path) for path in paths if str(path).strip()))
        return sorted(unique, key=score)[:limit]

    def _allowed_tools_for_role(self, role: str) -> list[str]:
        agent_cfg = self.registry.get_agent(role)
        if not agent_cfg.allow_tools:
            return []
        if role in self.CONTEXT_ONLY_ROLES:
            return []
        return list(agent_cfg.allowed_tools)

    def _task_with_recovery_targets(
        self,
        record: JobRecord,
        role: str,
        task: PlannedTask | None,
    ) -> PlannedTask | None:
        if role != "test_writer":
            return task
        constraints = self._constraints(record)
        if constraints.get("patch_operation_hint") != "create":
            return task
        missing_target_file = self._recovery_missing_target_file(record)
        if not missing_target_file:
            return task
        if task is None:
            task_id = re.sub(r"[^a-z0-9-]+", "-", Path(missing_target_file).stem.lower()).strip("-")
            task = PlannedTask(
                id=f"recover-{task_id or 'missing-test-file'}",
                title=f"Create missing test file {missing_target_file}",
                description=(
                    "Recovery task: create the missing test target file. "
                    "The patch operation must be create."
                ),
                role="test_writer",
            )
        return task.model_copy(
            update={
                "target_files": self._unique_paths(
                    [*task.target_files, missing_target_file]
                ),
                "required_artifacts": self._unique_paths(
                    [*task.required_artifacts, missing_target_file]
                ),
                "acceptance_criteria": self._unique_paths(
                    [
                        *self._meaningful_planning_items(task.acceptance_criteria),
                        f"{missing_target_file} exists and was created with patch.operation=create.",
                    ]
                ),
            }
        )

    def _objective_with_recovery_operation_hint(
        self,
        record: JobRecord,
        role: str,
        objective: str,
    ) -> str:
        if role != "test_writer":
            return objective
        constraints = self._constraints(record)
        missing_target_file = self._recovery_missing_target_file(record)
        if constraints.get("patch_operation_hint") != "create" or not missing_target_file:
            return objective
        return (
            f"{objective}\n\n"
            "Recovery requirement: the target file is missing. Return a patch for "
            f"{missing_target_file} with operation=create. Do not use operation=update."
        )

    def _context_constraints(self, record: JobRecord) -> list[str]:
        constraints = [f"blocked_operation={item}" for item in self.policy.config.blocked_operations]
        job_constraints = self._constraints(record)
        for key in sorted(job_constraints):
            value = job_constraints[key]
            if isinstance(value, (str, int, float, bool)):
                constraints.append(f"job_constraint {key}={value}")
        return constraints

    @staticmethod
    def _clear_active_recovery_state(record: JobRecord) -> None:
        record.last_error = None
        for key in (
            "current_recovery_event",
            "last_recoverable_error",
            "recovery_plan",
            *JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS,
            "failed_stage_ids",
            "failed_stages",
            "failed_task_id",
            "missing_stage_test_patch_stage_ids",
            "missing_task_ids",
            "stages_missing_test_patches",
            "unmet_dependencies",
        ):
            record.runtime_state.pop(key, None)
        record.outputs.pop("last_recoverable_error", None)
        constraints = record.spec.metadata.get("constraints")
        if not isinstance(constraints, dict):
            return
        for key in (
            *JobRunner.RECOVERY_METADATA_CONSTRAINT_KEYS,
            *JobRunner.FILE_RECOVERY_CONSTRAINT_KEYS,
            "failed_stage_ids",
            "failed_stages",
            "failed_task_id",
            "missing_stage_test_patch_stage_ids",
            "missing_task_ids",
            "stages_missing_test_patches",
            "unmet_dependencies",
        ):
            constraints.pop(key, None)

    @staticmethod
    def _clear_planning_repair_constraints(record: JobRecord) -> None:
        constraints = record.spec.metadata.get("constraints")
        if not isinstance(constraints, dict):
            return
        clear_planning_pm_strategy = (
            constraints.get("planning_repair_strategy_change") is True
            or constraints.get("recovery_strategy")
            == "planning_repair_strategy_change"
            or constraints.get("pm_strategy") == "planning_repair_strategy_change"
        )
        clear_planning_recovery_metadata = (
            constraints.get("planning_repair_strategy_change") is True
            or constraints.get("recovery_mode")
            in {
                "planning_repair",
                "prd_quality_repair",
                "prd_quality_revision",
                "task_graph_repair",
                "task_graph_replanning",
            }
            or constraints.get("recovery_strategy")
            in {
                "planning_repair_strategy_change",
                "REVISE_PRD_AND_ARCHITECTURE",
                "REPLAN_TASK",
                "task_graph_replanning",
            }
        )
        stale_planning_context_keys = {
            "prd_quality_missing",
            "prd_quality_warnings",
            "prd_open_questions",
            "uncovered_acceptance_small_parts",
            "uncovered_smallest_working_core",
            "uncovered_incremental_milestone_small_parts",
            "uncovered_implementation_artifact_small_parts",
            "uncovered_implementation_artifact_domain_small_parts",
            "uncovered_test_artifact_domain_small_parts",
            "non_observable_acceptance_tests",
            "invalid_required_artifacts",
            "failed_stage_ids",
            "failed_stages",
            "missing_stage_test_patch_stage_ids",
            "missing_task_ids",
            "stages_missing_test_patches",
            "prd_required_artifacts",
            "required_incremental_milestone_count",
            "required_small_part_count",
            "source_required_artifacts",
            "implementation_required_artifacts",
            "test_required_artifacts",
            "task_graph_validation_errors",
            *TASK_GRAPH_VALIDATION_CONTEXT_KEYS_SOURCE,
        }
        for key in list(constraints):
            if key.startswith("planning_repair_") or key in stale_planning_context_keys:
                constraints.pop(key, None)
        if clear_planning_recovery_metadata:
            for key in (
                *JobRunner.RECOVERY_METADATA_CONSTRAINT_KEYS,
                "failed_task_id",
                "unmet_dependencies",
            ):
                constraints.pop(key, None)
        if clear_planning_pm_strategy:
            for key in (
                "pm_strategy_change",
                "pm_strategy",
                "pm_intervention_count",
            ):
                constraints.pop(key, None)

    def _recovery_guidance_logs(self, record: JobRecord, role: str) -> list[str]:
        constraints = self._constraints(record)
        strategy = constraints.get("recovery_strategy")
        mode = constraints.get("recovery_mode")
        if not isinstance(strategy, str) or not isinstance(mode, str):
            return []
        guidance = [
            (
                "recovery_context: "
                f"mode={mode}; strategy={strategy}; "
                f"attempt={constraints.get('recovery_attempt', 1)}; "
                f"failed_task_id={constraints.get('recovery_failed_task_id', 'unknown')}; "
                f"failed_stage={constraints.get('recovery_failed_stage', 'unknown')}; "
                f"reason={constraints.get('recovery_reason', 'unspecified')}"
            )
        ]
        role_guidance = self._role_recovery_guidance(strategy, role)
        if role_guidance is not None:
            guidance.append(role_guidance)
        if strategy == "diagnosis_guided_retry":
            root_cause = constraints.get("diagnosis_root_cause")
            fix_strategy = constraints.get("diagnosis_recommended_fix_strategy")
            retry_mode = constraints.get("diagnosis_retry_mode")
            should_retry = constraints.get("diagnosis_should_retry")
            if isinstance(root_cause, str):
                guidance.append(f"diagnosis_root_cause: {root_cause}")
            if isinstance(fix_strategy, str):
                guidance.append(
                    f"diagnosis_recommended_fix_strategy: {fix_strategy}"
                )
            if isinstance(retry_mode, str) or isinstance(should_retry, bool):
                guidance.append(
                    "diagnosis_retry_policy: "
                    f"retry_mode={retry_mode}; should_retry={should_retry}"
                )
        if strategy == "RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD":
            exceeded_role = constraints.get("max_steps_exceeded_role", "unknown")
            guidance.append(
                "structured_output_recovery: "
                f"previous_role={exceeded_role}; avoid_tool_loop=true; "
                "return_schema_first=true; retry_small_scope=true"
            )
        patch_operation_hint = constraints.get("patch_operation_hint")
        missing_target_file = constraints.get("missing_target_file")
        if patch_operation_hint == "create" and isinstance(missing_target_file, str):
            guidance.append(
                "patch_operation_recovery: "
                f"file does not exist; path={missing_target_file}; "
                "the next patch for this path MUST set operation=create; "
                "operation=update is forbidden until the file exists"
            )
        return guidance

    def _recovery_history_logs(self, record: JobRecord, role: str) -> list[str]:
        if role not in {
            "pm",
            "architect",
            "planner",
            "implementer",
            "test_writer",
            "diagnoser",
            "fixer",
            "reviewer",
        }:
            return []
        history = summarize_job_progress(record).get("recovery_history", [])
        if not isinstance(history, list):
            return []
        logs: list[str] = []
        for item in history[-3:]:
            if not isinstance(item, dict):
                continue
            logs.append(
                "recovered_failure: "
                f"task_id={item.get('task_id', 'unknown')}; "
                f"failed_stage={item.get('failed_stage', 'unknown')}; "
                f"resolved_by_stage={item.get('resolved_by_stage', 'unknown')}; "
                f"failed_files={','.join(str(path) for path in item.get('failed_changed_files', []))}; "
                f"resolved_files={','.join(str(path) for path in item.get('resolved_changed_files', []))}; "
                f"failed_patch_count={item.get('failed_patch_count', 0)}; "
                f"resolved_patch_count={item.get('resolved_patch_count', 0)}"
            )
        return logs

    def _planning_repair_guidance_logs(self, record: JobRecord, role: str) -> list[str]:
        if role not in {"pm", "planner"}:
            return []
        planning_quality = summarize_job_progress(record).get("planning_quality", {})
        if not isinstance(planning_quality, dict):
            return []
        planning_repair = planning_quality.get("planning_repair")
        if not isinstance(planning_repair, dict):
            return []
        if not planning_repair.get("strategy_change_recommended"):
            return []
        last_task_graph_attempt = planning_quality.get("last_task_graph_validation_attempt")
        if not isinstance(last_task_graph_attempt, dict):
            last_task_graph_attempt = planning_quality.get("task_graph_validation")
        repeated_prd_missing = [
            str(item) for item in planning_repair.get("repeated_prd_missing", [])
        ]
        repeated_task_graph_error_types = [
            str(item)
            for item in planning_repair.get("repeated_task_graph_error_types", [])
        ]
        logs = [
            (
                "planning_repair_context: "
                f"consecutive_prd_failures={planning_repair.get('consecutive_prd_failure_count', 0)}; "
                f"consecutive_task_graph_failures={planning_repair.get('consecutive_task_graph_failure_count', 0)}; "
                f"repeated_prd_missing={','.join(repeated_prd_missing) or 'none'}; "
                f"repeated_task_graph_error_types={','.join(repeated_task_graph_error_types) or 'none'}"
            )
        ]
        if role == "pm" and repeated_prd_missing:
            logs.append(
                "planning_repair_instruction: change the requirements strategy; explicitly fill "
                "the repeated missing PRD fields with concrete, testable details before moving on."
            )
        if role == "planner" and repeated_task_graph_error_types:
            logs.append(
                "planning_repair_instruction: change the task graph strategy; simplify or split "
                "the plan so repeated validation errors are removed instead of reusing the same graph."
            )
            if isinstance(last_task_graph_attempt, dict):
                logs.extend(
                    self._non_empty_task_graph_validation_detail_logs(
                        last_task_graph_attempt,
                        prefix="planning_repair_task_graph_detail",
                    )
                )
        return logs

    @classmethod
    def _non_empty_task_graph_validation_detail_logs(
        cls,
        validation: dict[str, Any],
        *,
        prefix: str,
    ) -> list[str]:
        logs: list[str] = []
        for key in cls.TASK_GRAPH_VALIDATION_DETAIL_KEYS:
            value = validation.get(key)
            if value:
                logs.append(f"{prefix}: {key}={value}")
        return logs

    @classmethod
    def _task_graph_validation_repair_logs(
        cls,
        prd: PRD,
        validation: dict[str, Any],
    ) -> list[str]:
        logs = [
            "The previous task graph failed autonomy validation.",
            f"Validation errors: {validation['errors']}",
            f"PRD small_parts: {cls._meaningful_prd_items(prd.small_parts)}",
        ]
        prd_required_artifacts = cls._valid_unique_planning_artifact_paths(
            cls._meaningful_prd_items(prd.required_artifacts)
        )
        if prd_required_artifacts:
            logs.append(f"PRD required_artifacts: {prd_required_artifacts}")
        logs.extend(
            cls._non_empty_task_graph_validation_detail_logs(
                validation,
                prefix="task_graph_validation_detail",
            )
        )
        return logs

    def _pm_stall_guidance_logs(self, record: JobRecord, role: str) -> list[str]:
        if role not in {
            "pm",
            "planner",
            "architect",
            "implementer",
            "test_writer",
            "diagnoser",
            "fixer",
        }:
            return []
        constraints = self._constraints(record)
        if constraints.get("pm_stall_recovery") is not True:
            return []
        strategy = constraints.get("pm_strategy", "unknown")
        focus_task_id = constraints.get("pm_focus_task_id", "unknown")
        reason = constraints.get("pm_reason", "same_progress_marker_repeated")
        next_actor = constraints.get("pm_next_actor", "unknown")
        playbook = constraints.get("pm_recovery_playbook")
        success_criteria = constraints.get("pm_success_criteria")
        logs = [
            (
                "pm_stall_recovery: "
                f"strategy={strategy}; focus_task_id={focus_task_id}; "
                f"next_actor={next_actor}; reason={reason}"
            )
        ]
        if isinstance(playbook, str):
            logs.append(f"pm_recovery_playbook: {playbook}")
        if isinstance(success_criteria, str):
            logs.append(f"pm_recovery_success_criteria: {success_criteria}")
        if strategy == "split_or_simplify_next_task":
            logs.append(
                "pm_stall_instruction: change approach now; split the focused task into a "
                "smaller verifiable change, avoid repeating the same broad patch, and produce "
                "fresh test evidence."
            )
        elif strategy == "planning_repair_strategy_change":
            logs.append(
                "pm_stall_instruction: revise requirements or task planning before further "
                "implementation; do not reuse the previous invalid planning shape."
            )
        elif strategy == "raise_stage_limit":
            logs.append(
                "pm_stall_instruction: continue with the raised stage limit, but keep the next "
                "stage narrowly scoped and verify it before expanding."
            )
        elif strategy == "diagnosis_guided_fix":
            logs.append(
                "pm_stall_instruction: the PM must change method based on the diagnosis; "
                "do not repeat the same fixer loop. Re-scope dependencies, tests, or the "
                "task boundary so the diagnosed root cause is removed first."
            )
        elif strategy in {
            "dependency_alignment_first",
            "import_wiring_repair_first",
            "syntax_minimal_rewrite",
            "contract_reconciliation",
            "frontend_build_repair_first",
            "runtime_trace_reproduction",
            "inspect_before_retry",
        }:
            logs.append(
                "pm_stall_instruction: execute the PM recovery playbook before normal "
                "feature work; success means the diagnosed signature is removed, not merely "
                "that a new patch was attempted."
            )
        return logs

    @staticmethod
    def _role_recovery_guidance(strategy: str, role: str) -> str | None:
        if strategy == "escalated_retry" and role in {"implementer", "fixer", "test_writer"}:
            return (
                "recovery_instruction: retry the failed stage with a narrower patch, "
                "preserve existing passing tests, and explain why the previous attempt failed."
            )
        if strategy == "replan_current_task" and role in {"planner", "architect", "implementer"}:
            return (
                "recovery_instruction: re-scope the failed task into a smaller, testable step "
                "before implementing more code."
            )
        if strategy == "split_or_clarify_task" and role in {"pm", "planner", "implementer"}:
            return (
                "recovery_instruction: identify the blocking ambiguity, split the task if possible, "
                "and avoid broad implementation until acceptance criteria are clear."
            )
        if strategy == "rewrite_tests" and role in {"test_writer", "fixer"}:
            return (
                "recovery_instruction: rewrite focused tests for the current behavior without "
                "weakening assertions or masking implementation defects."
            )
        if strategy == "split_or_clarify_tests" and role in {"planner", "test_writer"}:
            return (
                "recovery_instruction: split the blocked test work into concrete assertions and "
                "name any missing acceptance criteria."
            )
        if strategy == "completion_audit" and role in {
            "planner",
            "implementer",
            "test_writer",
            "reviewer",
        }:
            return (
                "recovery_instruction: audit completion evidence first, then fill only the missing "
                "tasks, tests, or stage proof needed by the integrity gate."
            )
        if strategy == "diagnosis_guided_retry" and role in {
            "pm",
            "planner",
            "architect",
            "implementer",
            "test_writer",
            "fixer",
        }:
            return (
                "recovery_instruction: treat the diagnosis as a PM strategy change. "
                "Change method before retrying: adjust dependency policy, split the failed "
                "task, or rewrite the smallest setup surface needed by the diagnosed root "
                "cause. Do not continue with the same patch loop."
            )
        if strategy == "RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD" and role in {
            "implementer",
            "fixer",
            "test_writer",
            "diagnoser",
        }:
            return (
                "recovery_instruction: the previous agent exhausted tool steps without returning "
                "valid JSON. Inspect only files already named in the diagnosis or retrieval trace, "
                "make the smallest necessary patch, then return the required structured JSON. "
                "Do not continue broad repository exploration."
            )
        return None

    def _apply_patches(self, record: JobRecord, role: str, patches: list[Any]) -> None:
        max_patches = self._constraint_int(record, "max_patches_per_agent_output", 0)
        if max_patches and len(patches) > max_patches:
            raise QualityGateError(
                f"patch_limit_exceeded:{role}:{len(patches)}>{max_patches}"
            )
        if role in {"fixer", "test_writer"}:
            ensure_test_patch_quality(
                patches,
                role=role,
                workspace_root=self._workspace_root(record),
            )
        for patch in patches:
            self.policy.assert_patch_target_allowed(role, patch.path)
            if role not in {"fixer", "test_writer"}:
                ensure_test_patch_quality(
                    [patch],
                    role=role,
                    workspace_root=self._workspace_root(record),
                )
            patch = self._patch_for_missing_target_operation(record, role, patch)
            if patch is None:
                return
            self._ensure_patch_approved_or_pause(record, role, patch)
            try:
                self._call_tool(
                    role,
                    "repo_server.apply_patch",
                    path=patch.path,
                    content=patch.content,
                    operation=patch.operation,
                    new_path=patch.new_path,
                    unified_diff=patch.unified_diff,
                    base_sha256=patch.base_sha256,
                    expected_old_content=patch.expected_old_content,
                    executable=patch.executable,
                )
            except RuntimeError as exc:
                if self._recover_missing_patch_target(record, role, patch, exc):
                    return
                raise
        self.store.update(record)

    def _patch_for_missing_target_operation(
        self,
        record: JobRecord,
        role: str,
        patch: Any,
    ) -> Any | None:
        if getattr(patch, "operation", None) != "update":
            return patch
        patch_path = str(getattr(patch, "path", ""))
        if not patch_path:
            return patch
        if artifact_path_exists(patch_path, workspace_root=self._workspace_root(record)):
            return patch
        invalid_paths = invalid_artifact_paths([patch_path])
        if invalid_paths:
            self._recover_record(
                record,
                error=f"target_files_invalid:{patch_path}",
                runtime_state={
                    **record.runtime_state,
                    "failed_patch_role": role,
                    "failed_patch_path": patch_path,
                    "failed_patch_operation": "update",
                    "invalid_artifacts": invalid_paths,
                    "required_artifacts": [patch_path],
                    "target_files": [patch_path],
                },
            )
            return None
        known_missing = self._is_known_missing_patch_target(record, patch_path)
        if known_missing and getattr(patch, "content", None) is not None:
            rewritten = patch.model_copy(
                update={
                    "operation": "create",
                    "base_sha256": None,
                    "expected_old_content": None,
                }
            )
            rewrites = record.outputs.setdefault("patch_operation_rewrites", [])
            if isinstance(rewrites, list):
                rewrites.append(
                    {
                        "role": role,
                        "path": patch_path,
                        "from": "update",
                        "to": "create",
                        "reason": "known_missing_target_file",
                    }
                )
            return rewritten
        self._recover_record(
            record,
            error=f"PATCH_OPERATION_MISMATCH:update_missing_target:{patch_path}",
            runtime_state={
                **record.runtime_state,
                "failed_patch_role": role,
                "failed_patch_path": patch_path,
                "failed_patch_operation": "update",
                "required_artifacts": [patch_path],
                "target_files": [patch_path],
                "missing_artifacts": [patch_path],
                "missing_target_file": patch_path,
                "patch_operation_hint": "create",
            },
        )
        return None

    def _result_with_rewritten_missing_target_patches(
        self,
        record: JobRecord,
        role: str,
        result: Any,
    ) -> Any:
        patches = getattr(result, "patches", None)
        if not isinstance(patches, list) or not patches:
            return result
        rewritten_patches: list[Any] = []
        changed = False
        for patch in patches:
            patch_path = str(getattr(patch, "path", ""))
            if (
                getattr(patch, "operation", None) == "update"
                and getattr(patch, "content", None) is not None
                and not invalid_artifact_paths([patch_path])
                and not artifact_path_exists(
                    patch_path,
                    workspace_root=self._workspace_root(record),
                )
                and (
                    self._is_known_missing_patch_target(record, patch_path)
                    or self._is_test_writer_declared_new_test_file(
                        role,
                        result,
                        patch_path,
                    )
                )
            ):
                reason = (
                    "known_missing_target_file"
                    if self._is_known_missing_patch_target(record, patch_path)
                    else "test_writer_declared_new_test_file"
                )
                patch = patch.model_copy(
                    update={
                        "operation": "create",
                        "base_sha256": None,
                        "expected_old_content": None,
                    }
                )
                rewrites = record.outputs.setdefault("patch_operation_rewrites", [])
                if isinstance(rewrites, list):
                    rewrites.append(
                        {
                            "role": role,
                            "path": patch.path,
                            "from": "update",
                            "to": "create",
                            "reason": reason,
                            "stage": "structured_output",
                        }
                    )
                changed = True
            rewritten_patches.append(patch)
        if not changed:
            return result
        changed_files = self._unique_paths(
            [
                *getattr(result, "changed_files", []),
                *[patch.path for patch in rewritten_patches if hasattr(patch, "path")],
            ]
        )
        return result.model_copy(
            update={
                "patches": rewritten_patches,
                "changed_files": changed_files,
            }
        )

    @classmethod
    def _is_test_writer_declared_new_test_file(
        cls,
        role: str,
        result: Any,
        path: str,
    ) -> bool:
        if role != "test_writer" or not cls._looks_like_test_path(path):
            return False
        normalized = path.replace("\\", "/").removeprefix("./")
        changed_files = getattr(result, "changed_files", [])
        if not isinstance(changed_files, list):
            return False
        return normalized in {
            str(item).replace("\\", "/").removeprefix("./")
            for item in changed_files
            if str(item).strip()
        }

    def _is_known_missing_patch_target(self, record: JobRecord, path: str) -> bool:
        normalized = path.replace("\\", "/").removeprefix("./")
        constraints = self._constraints(record)
        candidates: list[str] = []
        for key in (
            "missing_target_file",
            "failed_patch_path",
        ):
            value = constraints.get(key) or record.runtime_state.get(key)
            if isinstance(value, str):
                candidates.append(value)
        for key in ("missing_artifacts", "required_artifacts", "target_files"):
            for source in (constraints.get(key), record.runtime_state.get(key)):
                if isinstance(source, list):
                    candidates.extend(str(item) for item in source)
        return normalized in {
            candidate.replace("\\", "/").removeprefix("./")
            for candidate in candidates
            if str(candidate).strip()
        }

    def _recover_missing_patch_target(
        self,
        record: JobRecord,
        role: str,
        patch: Any,
        exc: RuntimeError,
    ) -> bool:
        message = str(exc)
        if "target_files_missing:update target does not exist:" not in message:
            return False
        missing_path = message.rsplit(":", 1)[-1].strip()
        if not missing_path:
            missing_path = str(getattr(patch, "path", ""))
        if self._record_missing_target_repeat(record, missing_path) >= 2 and self._looks_like_test_path(missing_path):
            self._create_deterministic_missing_test_file(record, missing_path)
            return True
        self._recover_record(
            record,
            error=f"target_files_missing:{message}",
            runtime_state={
                **record.runtime_state,
                "failed_patch_role": role,
                "failed_patch_path": missing_path,
                "failed_patch_operation": getattr(patch, "operation", "update"),
                "required_artifacts": [missing_path],
                "target_files": [missing_path],
                "missing_artifacts": [missing_path],
            },
        )
        return True

    @staticmethod
    def _looks_like_test_path(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        return (
            "/tests/" in f"/{normalized}"
            or "/test/" in f"/{normalized}"
            or name.startswith("test_")
            or ".test." in name
            or ".spec." in name
        )

    @staticmethod
    def _looks_like_test_work_item(text: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        return bool(
            tokens
            & {
                "assert",
                "asserts",
                "assertion",
                "assertions",
                "coverage",
                "pytest",
                "test",
                "tests",
                "testing",
                "vitest",
            }
        )

    @staticmethod
    def _looks_like_implementation_work_item(text: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        semantic_tokens = JobRunner._semantic_tokens(text)
        combined_tokens = tokens | semantic_tokens
        if not tokens:
            return False
        documentation_tokens = {
            "doc",
            "docs",
            "documentation",
            "guide",
            "manual",
            "readme",
        }
        implementation_tokens = {
            "api",
            "app",
            "backend",
            "client",
            "component",
            "crud",
            "endpoint",
            "feature",
            "frontend",
            "implement",
            "module",
            "page",
            "react",
            "route",
            "server",
            "service",
            "ui",
            "view",
        } | JobRunner.SEMANTIC_ANCHOR_TOKENS | JobRunner.CRUD_OPERATION_TOKENS
        return bool(combined_tokens & implementation_tokens) and not (
            combined_tokens <= documentation_tokens
        )

    @staticmethod
    def _looks_like_implementation_source_path(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        if name in {
            ".env",
            ".env.example",
            ".gitignore",
            "dockerfile",
            "package.json",
            "package-lock.json",
            "pnpm-lock.yaml",
            "readme.md",
            "requirements.txt",
            "vite.config.js",
            "vite.config.ts",
            "yarn.lock",
        }:
            return False
        if normalized.startswith(("docs/", "doc/")):
            return False
        source_extensions = {
            ".css",
            ".go",
            ".html",
            ".java",
            ".js",
            ".jsx",
            ".kt",
            ".php",
            ".py",
            ".rb",
            ".rs",
            ".scss",
            ".swift",
            ".ts",
            ".tsx",
        }
        return any(normalized.endswith(suffix) for suffix in source_extensions)

    @classmethod
    def _implementation_surfaces_for_work_item(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
        surfaces: set[str] = set()
        if tokens & cls.BACKEND_SURFACE_TOKENS:
            surfaces.add("backend")
        if tokens & cls.FRONTEND_SURFACE_TOKENS:
            surfaces.add("frontend")
        if tokens & cls.SHARED_SURFACE_TOKENS:
            surfaces.add("shared")
        if not surfaces:
            surfaces.add("implementation")
        return surfaces

    @classmethod
    def _implementation_surfaces_for_artifact(cls, path: str) -> set[str]:
        normalized = path.replace("\\", "/").lower().removeprefix("./")
        name = normalized.rsplit("/", 1)[-1]
        path_tokens = set(re.findall(r"[a-z0-9_]+", normalized))
        surfaces = {"implementation"}
        if (
            normalized.startswith(("backend/", "api/", "server/"))
            or "/backend/" in f"/{normalized}"
            or "/api/" in f"/{normalized}"
            or "/server/" in f"/{normalized}"
            or name in {"app.py", "main.py"}
            or path_tokens & {"fastapi", "flask", "django"}
        ):
            surfaces.add("backend")
        if (
            normalized.startswith(("frontend/", "web/", "client/", "static/"))
            or "/frontend/" in f"/{normalized}"
            or "/components/" in f"/{normalized}"
            or "/pages/" in f"/{normalized}"
            or name.endswith((".tsx", ".jsx", ".css", ".scss", ".html"))
            or name in {"app.tsx", "app.jsx", "main.tsx", "main.jsx"}
        ):
            surfaces.add("frontend")
        if (
            normalized.startswith(("shared/", "common/"))
            or "/shared/" in f"/{normalized}"
            or "/common/" in f"/{normalized}"
            or path_tokens & {"schema", "schemas", "types", "models", "contracts"}
        ):
            surfaces.add("shared")
        return surfaces

    @classmethod
    def _implementation_artifact_surface_coverage(
        cls,
        small_parts: list[str],
        implementation_required_artifacts: list[str],
    ) -> list[dict[str, Any]]:
        artifact_surface_entries = [
            (artifact, cls._implementation_surfaces_for_artifact(artifact))
            for artifact in implementation_required_artifacts
        ]
        all_artifact_surfaces: set[str] = set()
        for _, surfaces in artifact_surface_entries:
            all_artifact_surfaces.update(surfaces)
        coverage: list[dict[str, Any]] = []
        for index, part in enumerate(small_parts, start=1):
            if not cls._looks_like_implementation_work_item(part):
                continue
            required_surfaces = cls._implementation_surfaces_for_work_item(part)
            matched_artifacts = [
                artifact
                for artifact, surfaces in artifact_surface_entries
                if required_surfaces & surfaces
                or (
                    required_surfaces == {"implementation"}
                    and "implementation" in surfaces
                )
            ]
            missing_surfaces = sorted(required_surfaces - all_artifact_surfaces)
            coverage.append(
                {
                    "small_part_index": index,
                    "small_part": part,
                    "required_surfaces": sorted(required_surfaces),
                    "covered_surfaces": sorted(required_surfaces & all_artifact_surfaces),
                    "missing_surfaces": missing_surfaces,
                    "implementation_artifacts": matched_artifacts,
                    "covered": not missing_surfaces,
                }
            )
        return coverage

    @classmethod
    def _implementation_artifact_domain_coverage(
        cls,
        small_parts: list[str],
        implementation_required_artifacts: list[str],
    ) -> list[dict[str, Any]]:
        artifact_token_entries = [
            (
                artifact,
                cls._semantic_tokens(artifact, include_action_tokens=True),
            )
            for artifact in implementation_required_artifacts
        ]
        all_artifact_tokens: set[str] = set()
        for _, tokens in artifact_token_entries:
            all_artifact_tokens.update(tokens)
        coverage: list[dict[str, Any]] = []
        for index, part in enumerate(small_parts, start=1):
            if not cls._looks_like_implementation_work_item(part):
                continue
            part_tokens = cls._semantic_tokens(part)
            anchor_tokens = cls._semantic_anchor_tokens(part_tokens)
            domain_tokens = part_tokens - cls.IMPLEMENTATION_ARTIFACT_GENERIC_TOKENS
            if not part_tokens or (not anchor_tokens and not domain_tokens):
                coverage.append(
                    {
                        "small_part_index": index,
                        "small_part": part,
                        "required_anchor_tokens": sorted(anchor_tokens),
                        "required_domain_tokens": sorted(domain_tokens),
                        "covered_anchor_tokens": [],
                        "covered_domain_tokens": [],
                        "missing_anchor_tokens": [],
                        "implementation_artifacts": [],
                        "covered": True,
                    }
                )
                continue
            anchor_satisfied = bool(anchor_tokens) and cls._semantic_anchor_satisfied(
                anchor_tokens,
                all_artifact_tokens,
            )
            required_domain_score = cls._semantic_overlap_required(domain_tokens)
            covered_domain_tokens = domain_tokens & all_artifact_tokens
            domain_satisfied = (
                bool(domain_tokens)
                and len(covered_domain_tokens) >= required_domain_score
            )
            covered = anchor_satisfied or domain_satisfied
            matched_artifacts = [
                artifact
                for artifact, tokens in artifact_token_entries
                if (
                    bool(anchor_tokens)
                    and cls._semantic_anchor_satisfied(anchor_tokens, tokens)
                )
                or bool(domain_tokens & tokens)
            ]
            covered_anchor_tokens = (
                sorted(anchor_tokens)
                if anchor_satisfied
                else sorted(anchor_tokens & all_artifact_tokens)
            )
            missing_anchor_tokens = [] if covered else sorted(
                anchor_tokens - all_artifact_tokens
            )
            coverage.append(
                {
                    "small_part_index": index,
                    "small_part": part,
                    "required_anchor_tokens": sorted(anchor_tokens),
                    "required_domain_tokens": sorted(domain_tokens),
                    "covered_anchor_tokens": covered_anchor_tokens,
                    "covered_domain_tokens": sorted(covered_domain_tokens),
                    "missing_anchor_tokens": missing_anchor_tokens,
                    "implementation_artifacts": matched_artifacts,
                    "covered": covered,
                }
            )
        return coverage

    @classmethod
    def _test_artifact_domain_coverage(
        cls,
        small_parts: list[str],
        test_required_artifacts: list[str],
    ) -> list[dict[str, Any]]:
        artifact_token_entries = [
            (
                artifact,
                cls._semantic_tokens(artifact, include_action_tokens=True),
            )
            for artifact in test_required_artifacts
        ]
        all_artifact_tokens: set[str] = set()
        for _, tokens in artifact_token_entries:
            all_artifact_tokens.update(tokens)
        coverage: list[dict[str, Any]] = []
        for index, part in enumerate(small_parts, start=1):
            if not cls._looks_like_implementation_work_item(part):
                continue
            part_tokens = cls._semantic_tokens(part)
            anchor_tokens = cls._semantic_anchor_tokens(part_tokens)
            domain_tokens = part_tokens - cls.IMPLEMENTATION_ARTIFACT_GENERIC_TOKENS
            if not part_tokens or (not anchor_tokens and not domain_tokens):
                coverage.append(
                    {
                        "small_part_index": index,
                        "small_part": part,
                        "required_anchor_tokens": sorted(anchor_tokens),
                        "required_domain_tokens": sorted(domain_tokens),
                        "covered_anchor_tokens": [],
                        "covered_domain_tokens": [],
                        "missing_anchor_tokens": [],
                        "test_artifacts": [],
                        "covered": True,
                    }
                )
                continue
            anchor_satisfied = bool(anchor_tokens) and cls._semantic_anchor_satisfied(
                anchor_tokens,
                all_artifact_tokens,
            )
            required_domain_score = cls._semantic_overlap_required(domain_tokens)
            covered_domain_tokens = domain_tokens & all_artifact_tokens
            domain_satisfied = (
                bool(domain_tokens)
                and len(covered_domain_tokens) >= required_domain_score
            )
            covered = anchor_satisfied or domain_satisfied
            matched_artifacts = [
                artifact
                for artifact, tokens in artifact_token_entries
                if (
                    bool(anchor_tokens)
                    and cls._semantic_anchor_satisfied(anchor_tokens, tokens)
                )
                or bool(domain_tokens & tokens)
            ]
            covered_anchor_tokens = (
                sorted(anchor_tokens)
                if anchor_satisfied
                else sorted(anchor_tokens & all_artifact_tokens)
            )
            missing_anchor_tokens = [] if covered else sorted(
                anchor_tokens - all_artifact_tokens
            )
            coverage.append(
                {
                    "small_part_index": index,
                    "small_part": part,
                    "required_anchor_tokens": sorted(anchor_tokens),
                    "required_domain_tokens": sorted(domain_tokens),
                    "covered_anchor_tokens": covered_anchor_tokens,
                    "covered_domain_tokens": sorted(covered_domain_tokens),
                    "missing_anchor_tokens": missing_anchor_tokens,
                    "test_artifacts": matched_artifacts,
                    "covered": covered,
                }
            )
        return coverage

    def _record_missing_target_repeat(self, record: JobRecord, path: str) -> int:
        repeats = record.runtime_state.setdefault("missing_target_file_repeats", {})
        if not isinstance(repeats, dict):
            repeats = {}
            record.runtime_state["missing_target_file_repeats"] = repeats
        normalized = path.replace("\\", "/").removeprefix("./")
        repeats[normalized] = int(repeats.get(normalized, 0)) + 1
        return int(repeats[normalized])

    def _create_deterministic_missing_test_file(
        self,
        record: JobRecord,
        path: str,
    ) -> None:
        normalized = path.replace("\\", "/").removeprefix("./")
        patch = FilePatch(
            path=normalized,
            operation="create",
            content=self._minimal_test_scaffold_content(normalized),
        )
        self.policy.assert_patch_target_allowed("test_writer", patch.path)
        self._call_tool(
            "test_writer",
            "repo_server.apply_patch",
            path=patch.path,
            content=patch.content,
            operation=patch.operation,
            new_path=patch.new_path,
            unified_diff=patch.unified_diff,
            base_sha256=patch.base_sha256,
            expected_old_content=patch.expected_old_content,
            executable=patch.executable,
        )
        created = record.outputs.setdefault("deterministic_test_scaffolds", [])
        if isinstance(created, list):
            created.append(
                {
                    "path": normalized,
                    "reason": "repeated_missing_target_file",
                }
            )
        record.runtime_state["deterministic_test_scaffold_created"] = normalized
        self.store.update(record)

    @staticmethod
    def _minimal_test_scaffold_content(path: str) -> str:
        normalized = path.replace("\\", "/").removeprefix("./")
        suffix = Path(path).suffix.lower()
        if suffix in {".ts", ".tsx", ".js", ".jsx"}:
            js_path = normalized.replace("\\", "\\\\").replace("'", "\\'")
            return (
                "import { describe, expect, it } from 'vitest'\n\n"
                "describe('project scaffold', () => {\n"
                "  it('has a deterministic test scaffold', () => {\n"
                f"    // fallback target: {js_path}\n"
                "    const url = import.meta.url\n"
                "    expect(url).toMatch(/(^|\\/)(test|tests)\\//)\n"
                "    expect(url).toMatch(/(^|\\/)test_|\\.(test|spec)\\./)\n"
                "  })\n"
                "})\n"
            )
        py_path = repr(normalized)
        return (
            "from pathlib import Path\n\n\n"
            "def test_project_scaffold_placeholder() -> None:\n"
            f"    # fallback target: {py_path}\n"
            "    current_path = Path(__file__).as_posix()\n"
            "    name = Path(__file__).name\n"
            "    normalized = current_path.replace('\\\\', '/')\n"
            "    assert '/test' in f'/{normalized}' or name.startswith('test_')\n"
        )

    def _resume_approval_if_ready(self, record: JobRecord) -> bool:
        if record.status != JobStatus.WAITING_APPROVAL or not record.pending_approval_id:
            return False
        if self.approval_gateway is None:
            return False
        approval = self.approval_gateway.get(record.pending_approval_id)
        if approval.status == ApprovalStatus.APPROVED:
            record.runtime_state["approved_approval_id"] = approval.id
            pending_patch = record.runtime_state.get("pending_approval_patch")
            pending_patch_role = str(
                record.runtime_state.get("pending_approval_patch_role")
                or "approved_patch"
            )
            if isinstance(pending_patch, dict):
                try:
                    self._ensure_pending_approval_patch_quality(
                        record,
                        pending_patch,
                        role=pending_patch_role,
                    )
                except QualityGateError:
                    record.runtime_state.pop("pending_approval_patch", None)
                    record.runtime_state.pop("pending_approval_patch_role", None)
                    record.pending_approval_id = None
                    raise
                result = self.router.call("repo_server.apply_patch", **pending_patch)
                if not result.ok:
                    raise RuntimeError(result.error or "approved patch application failed")
            record.runtime_state.pop("pending_approval_patch", None)
            record.runtime_state.pop("pending_approval_patch_role", None)
            record.pending_approval_id = None
            if record.status != JobStatus.RESUMING:
                record.status = JobStatus.RESUMING
                record.history.append(JobStatus.RESUMING)
            return True
        if approval.status == ApprovalStatus.REJECTED:
            record.last_error = approval.resolution_reason or "approval rejected"
            record.pending_approval_id = None
            record.runtime_state.pop("pending_approval_patch", None)
            record.runtime_state.pop("pending_approval_patch_role", None)
            if record.status != JobStatus.BLOCKED:
                record.status = JobStatus.BLOCKED
                record.history.append(JobStatus.BLOCKED)
            return True
        return False

    def _ensure_pending_approval_patch_quality(
        self,
        record: JobRecord,
        pending_patch: dict[str, Any],
        *,
        role: str,
    ) -> None:
        patch_payload = {
            key: pending_patch[key]
            for key in FilePatch.model_fields
            if key in pending_patch
        }
        try:
            patch = FilePatch.model_validate(patch_payload)
        except ValueError as exc:
            raise QualityGateError(f"approved_patch_invalid:{exc}") from exc
        ensure_test_patch_quality(
            [patch],
            role=role,
            workspace_root=self._workspace_root(record),
        )

    def _ensure_patch_approved_or_pause(
        self,
        record: JobRecord,
        role: str,
        patch: Any,
    ) -> None:
        if self.approval_gateway is None:
            return
        if self._resume_approval_if_ready(record):
            if record.status == JobStatus.BLOCKED:
                raise JobWaitingForApproval(record.last_error or "approval rejected")
            return
        if record.pending_approval_id:
            raise JobWaitingForApproval("approval pending")
        try:
            decision = self.policy.classify_tool_call(
                role=role,
                tool_name="repo_server.apply_patch",
                arguments={
                    "path": patch.path,
                    "content": patch.content or patch.unified_diff or "",
                    "changed_files": 1,
                },
                workspace_root=record.spec.workspace_root or record.spec.repo_path,
                job_metadata=record.spec.metadata,
            )
        except PermissionError as exc:
            raise QualityGateError(f"policy_deny:{exc}") from exc
        if decision.policy_action == PolicyAction.DENY:
            raise QualityGateError(f"policy_deny:{decision.reason}")
        if decision.policy_action != PolicyAction.REQUIRE_APPROVAL:
            return
        challenge = self.approval_gateway.create_challenge(
            job_id=record.job_id,
            task_id=record.current_task_id,
            role=role,
            requested_by=role,
            operation=decision.operation,
            risk_level=decision.risk_level,
            reason=decision.reason,
            proposed_action={
                "tool_name": "repo_server.apply_patch",
                "path": patch.path,
                "operation": patch.operation,
            },
        )
        record.pending_approval_id = challenge.request.id
        record.runtime_state["pending_approval_patch"] = {
            "path": patch.path,
            "content": patch.content,
            "operation": patch.operation,
            "new_path": patch.new_path,
            "unified_diff": patch.unified_diff,
            "base_sha256": patch.base_sha256,
            "expected_old_content": patch.expected_old_content,
            "executable": patch.executable,
        }
        record.runtime_state["pending_approval_patch_role"] = role
        if record.status != JobStatus.WAITING_APPROVAL:
            record.status = JobStatus.WAITING_APPROVAL
            record.history.append(JobStatus.WAITING_APPROVAL)
        self.store.update(record)
        self.router.call(
            "notify_server.send_notification",
            body=f"Approval required for {decision.operation}: {patch.path}",
            kind="approval_required",
            job_id=record.job_id,
            approval_id=challenge.request.id,
            approve_url=challenge.approve_url,
            reject_url=challenge.reject_url,
        )
        raise JobWaitingForApproval("approval pending")

    def _handle_provider_adapter_error(
        self,
        record: JobRecord,
        exc: AdapterError,
    ) -> JobRecord:
        if self.runtime_manager is None:
            self._recover_record(record, error=str(exc))
            return self.store.update(record)
        code = (exc.code or "").lower()
        issue_type = RuntimeIssueType.CONNECTION_ERROR
        if "timeout" in code or "timeout" in str(exc).lower():
            issue_type = RuntimeIssueType.TIMEOUT
        elif "auth" in code or "unauthorized" in str(exc).lower():
            issue_type = RuntimeIssueType.AUTH_ERROR
        elif "model" in code and "not" in code:
            issue_type = RuntimeIssueType.MODEL_NOT_FOUND
        provider_key = "unknown"
        model_key = None
        try:
            selection = self.model_router.select_model(
                RoutingContext(
                    role=record.current_role or "pm",
                    failure_count=record.failure_count,
                    same_test_failure_count=record.same_test_failure_count,
                    changed_files_count=0,
                    security_sensitive=False,
                    context_tokens=0,
                )
            )
            provider_key = selection.provider_key
            model_key = selection.model_key
        except Exception:
            pass
        issue = self.runtime_manager.handle_provider_error(
            record=record,
            provider_key=provider_key,
            model_key=model_key,
            issue_type=issue_type,
            message=str(exc),
        )
        self.router.call(
            "notify_server.send_notification",
            body=f"Runtime provider wait: {issue.message}",
            kind="runtime_wait",
            job_id=record.job_id,
            runtime_issue_id=issue.id,
        )
        return self.store.update(record)

    def _run_tests(self, record: JobRecord) -> TestRunResult:
        self._transition_to_testing(record)
        constraints = self._constraints(record)
        command_name = str(constraints.get("test_command_name", "pytest"))
        timeout_seconds = int(constraints.get("test_timeout_seconds", 120))
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name=command_name,
            timeout_seconds=timeout_seconds,
        )
        result = TestRunResult.model_validate(payload)
        record.outputs["test_run"] = result.model_dump()
        if result.success:
            runtime_result = self._run_runtime_contract_checks(
                record,
                timeout_seconds=timeout_seconds,
            )
            if runtime_result is not None and not runtime_result.success:
                return runtime_result
        return result

    def _run_runtime_contract_checks(
        self,
        record: JobRecord,
        *,
        timeout_seconds: int,
    ) -> TestRunResult | None:
        runtime_contract = self._runtime_contract(record)
        acceptance_checks = self._runtime_acceptance_checks(record)
        if not runtime_contract and not acceptance_checks:
            return None
        runtime_timeout_seconds = self._runtime_timeout_seconds(
            runtime_contract,
            fallback=timeout_seconds,
        )
        start_command = self._runtime_start_command(runtime_contract)
        if start_command:
            payload = self._call_tool(
                "runner",
                "test_server.run_command",
                argv=start_command,
                timeout_seconds=runtime_timeout_seconds,
                mode="server",
                http_path=self._runtime_http_path(runtime_contract),
                http_checks=acceptance_checks or None,
            )
        else:
            payload = self._call_tool(
                "runner",
                "test_server.run_test",
                command_name="runtime-smoke-auto",
                timeout_seconds=runtime_timeout_seconds,
                http_checks=acceptance_checks or None,
            )
        result = TestRunResult.model_validate(payload)
        result_payload = result.model_dump()
        record.outputs["runtime_smoke"] = result_payload
        if acceptance_checks:
            record.outputs["acceptance_checks"] = result_payload
        return result

    @staticmethod
    def _runtime_contract(record: JobRecord) -> dict[str, Any]:
        metadata = record.spec.metadata if isinstance(record.spec.metadata, dict) else {}
        constraints = metadata.get("constraints")
        for source in (metadata, constraints if isinstance(constraints, dict) else {}):
            value = source.get("runtime")
            if isinstance(value, dict) and value:
                return value
        return {}

    @staticmethod
    def _runtime_acceptance_checks(record: JobRecord) -> list[dict[str, Any]]:
        metadata = record.spec.metadata if isinstance(record.spec.metadata, dict) else {}
        constraints = metadata.get("constraints")
        checks: list[dict[str, Any]] = []
        for source in (metadata, constraints if isinstance(constraints, dict) else {}):
            runtime = source.get("runtime")
            if isinstance(runtime, dict):
                http_checks = runtime.get("http_checks")
                if isinstance(http_checks, list):
                    checks.extend(item for item in http_checks if isinstance(item, dict))
            acceptance_checks = source.get("acceptance_checks")
            if isinstance(acceptance_checks, list):
                checks.extend(item for item in acceptance_checks if isinstance(item, dict))
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for check in checks:
            key = repr(sorted(check.items()))
            if key not in seen:
                deduped.append(check)
                seen.add(key)
        return deduped

    @staticmethod
    def _runtime_timeout_seconds(
        runtime_contract: dict[str, Any],
        *,
        fallback: int,
    ) -> int:
        for key in ("startup_timeout_seconds", "prepare_timeout_seconds"):
            value = runtime_contract.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return fallback

    @staticmethod
    def _runtime_start_command(runtime_contract: dict[str, Any]) -> list[str]:
        value = runtime_contract.get("start_command")
        if isinstance(value, str) and value.strip():
            return shlex.split(value)
        if isinstance(value, list) and all(
            isinstance(item, str) and item.strip() for item in value
        ):
            return [str(item) for item in value]
        return []

    @staticmethod
    def _runtime_http_path(runtime_contract: dict[str, Any]) -> str:
        value = runtime_contract.get("http_probe_path")
        if isinstance(value, str) and value.startswith("/"):
            return value
        return "/"

    def _transition_to_testing(self, record: JobRecord) -> None:
        if record.status == JobStatus.IMPLEMENTING:
            apply_transition(record, JobStatus.WRITING_TESTS)
        if record.status == JobStatus.WRITING_TESTS:
            apply_transition(record, JobStatus.REVIEWING)
        apply_transition(record, JobStatus.TESTING)

    def _load_or_refine_prd_for_autonomy(self, record: JobRecord) -> PRD | None:
        prd = self._load_or_run_role(
            record,
            "pm",
            PRD,
            "Produce the product requirements",
            memory_key="prd",
        )
        refined = self._refine_prd_quality_for_autonomy(record, prd)
        if refined is not None:
            self._sync_execution_contract_metadata(record, refined)
        return refined

    def _sync_execution_contract_metadata(self, record: JobRecord, prd: PRD) -> None:
        merged = synthesize_job_metadata_from_prd(
            prd,
            record.spec.metadata,
            workspace_root=self._workspace_root(record),
        )
        if merged == record.spec.metadata:
            return
        record.spec.metadata = merged
        record.outputs["execution_contracts"] = {
            "runtime": bool(merged.get("runtime")),
            "acceptance_checks": bool(merged.get("acceptance_checks")),
            "required_artifacts": list(merged.get("required_artifacts", []))
            if isinstance(merged.get("required_artifacts"), list)
            else [],
            "framework_profile": merged.get("framework_profile"),
        }
        self.store.update(record)

    def _refine_prd_quality_for_autonomy(self, record: JobRecord, prd: PRD) -> PRD | None:
        min_small_parts = self._prd_quality_min_small_parts(record)
        report = self._build_prd_quality_report(
            prd,
            min_small_parts=min_small_parts,
        )
        record.outputs["prd_quality"] = report
        self._record_prd_quality_attempt(record, attempt=0, action="initial", report=report)
        if report["passed"] or not self._constraint_flag(record, "require_prd_quality"):
            if report["passed"]:
                self._clear_planning_repair_constraints(record)
            self.store.update(record)
            return prd

        refinement_attempts = self._constraint_int(record, "prd_quality_refinement_attempts", 2)
        current_prd = prd
        self._reset_blocked_planning_resume(
            record,
            target_status=JobStatus.ANALYZING,
            last_error_prefix="prd_quality_gate_failed:",
        )
        attempt_offset = 0
        deterministic_prd = (
            self._deterministically_repair_prd_quality(
                current_prd,
                report,
            )
            if refinement_attempts > 0
            else None
        )
        if deterministic_prd is not None:
            previous_acceptance_tests = set(
                self._meaningful_prd_items(current_prd.acceptance_tests)
            )
            current_prd = deterministic_prd
            added_acceptance_tests = [
                item
                for item in self._meaningful_prd_items(current_prd.acceptance_tests)
                if item not in previous_acceptance_tests
            ]
            record.outputs["prd_quality_deterministic_repair"] = {
                "applied": True,
            }
            duplicate_prd_items = report.get("duplicate_prd_items")
            if isinstance(duplicate_prd_items, dict) and duplicate_prd_items:
                record.outputs["prd_quality_deterministic_repair"][
                    "removed_duplicate_prd_items"
                ] = duplicate_prd_items
            if added_acceptance_tests:
                record.outputs["prd_quality_deterministic_repair"][
                    "added_acceptance_tests"
                ] = added_acceptance_tests
            previous_milestones = set(
                self._meaningful_prd_items(prd.incremental_milestones)
            )
            added_milestones = [
                item
                for item in self._meaningful_prd_items(
                    current_prd.incremental_milestones
                )
                if item not in previous_milestones
            ]
            if added_milestones:
                record.outputs["prd_quality_deterministic_repair"][
                    "added_incremental_milestones"
                ] = added_milestones
            self._write_memory_item(record, "pm", "prd", current_prd.model_dump_json())
            record.outputs["pm"] = current_prd.model_dump()
            report = self._build_prd_quality_report(
                current_prd,
                min_small_parts=min_small_parts,
            )
            record.outputs["prd_quality"] = report
            self._record_prd_quality_attempt(
                record,
                attempt=1,
                action="deterministic_repair",
                report=report,
            )
            attempt_offset = 1
            if report["passed"]:
                self._clear_planning_repair_constraints(record)
                self.store.update(record)
                return current_prd

        for attempt in range(attempt_offset + 1, attempt_offset + refinement_attempts + 1):
            current_prd = self._run_structured_role(
                record,
                "pm",
                PRD,
                (
                    "Refine the product requirements before implementation. "
                    "Fill every missing PRD quality field for autonomous large-scale execution: "
                    f"{', '.join(report['missing'])}. "
                    "When acceptance coverage is missing, add or rewrite acceptance_tests "
                    "so every small_part has a direct observable test using the same domain terms."
                ),
                logs=self._prd_quality_repair_logs(current_prd, report),
            )
            self._write_memory_item(record, "pm", "prd", current_prd.model_dump_json())
            report = self._build_prd_quality_report(
                current_prd,
                min_small_parts=min_small_parts,
            )
            record.outputs["prd_quality"] = report
            self._record_prd_quality_attempt(
                record,
                attempt=attempt,
                action="refine",
                report=report,
            )
            if report["passed"]:
                self._clear_planning_repair_constraints(record)
                self.store.update(record)
                return current_prd

        self._recover_record(
            record,
            error="prd_quality_gate_failed:" + ",".join(report["missing"]),
            runtime_state=self._prd_quality_recovery_state(
                record,
                current_prd,
                report,
            ),
        )
        self.store.update(record)
        return None

    def _prd_quality_min_small_parts(self, record: JobRecord) -> int:
        explicit = self._constraint_int(record, "min_prd_small_parts", 0)
        if explicit > 0:
            return explicit
        if (
            self._constraint_flag(record, "require_prd_quality")
            and self._constraint_int(record, "max_autonomous_stages", 0) > 0
        ):
            return 2
        return 0

    @staticmethod
    def _prd_quality_recovery_state(
        record: JobRecord,
        prd: PRD,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_state = dict(record.runtime_state)
        for key in (
            "prd_quality_missing",
            "prd_quality_warnings",
            "prd_open_questions",
            "uncovered_acceptance_small_parts",
            "uncovered_smallest_working_core",
            "uncovered_incremental_milestone_small_parts",
            "uncovered_implementation_artifact_small_parts",
            "uncovered_implementation_artifact_domain_small_parts",
            "uncovered_test_artifact_domain_small_parts",
            "non_observable_acceptance_tests",
            "duplicate_prd_items",
            "invalid_required_artifacts",
            "prd_required_artifacts",
            "required_incremental_milestone_count",
            "required_small_part_count",
            "source_required_artifacts",
            "implementation_required_artifacts",
            "test_required_artifacts",
        ):
            runtime_state.pop(key, None)
        missing = JobRunner._non_empty_items(
            [str(item) for item in report.get("missing", [])]
        )
        warnings = JobRunner._non_empty_items(
            [str(item) for item in report.get("warnings", [])]
        )
        open_questions = JobRunner._meaningful_prd_items(prd.open_questions)
        invalid_required_artifacts = JobRunner._non_empty_items(
            [str(item) for item in report.get("invalid_required_artifacts", [])]
        )
        prd_required_artifacts = JobRunner._non_empty_items(
            [str(item) for item in report.get("required_artifacts", [])]
        )
        required_incremental_milestone_count = report.get(
            "required_incremental_milestone_count"
        )
        required_small_part_count = report.get("required_small_part_count")
        source_required_artifacts = JobRunner._non_empty_items(
            [str(item) for item in report.get("source_required_artifacts", [])]
        )
        implementation_required_artifacts = JobRunner._non_empty_items(
            [str(item) for item in report.get("implementation_required_artifacts", [])]
        )
        test_required_artifacts = JobRunner._non_empty_items(
            [str(item) for item in report.get("test_required_artifacts", [])]
        )
        uncovered = report.get("uncovered_acceptance_small_parts")
        uncovered_core = report.get("uncovered_smallest_working_core")
        uncovered_milestones = report.get("uncovered_incremental_milestone_small_parts")
        uncovered_implementation_artifacts = report.get(
            "uncovered_implementation_artifact_small_parts"
        )
        uncovered_implementation_artifact_domains = report.get(
            "uncovered_implementation_artifact_domain_small_parts"
        )
        uncovered_test_artifact_domains = report.get(
            "uncovered_test_artifact_domain_small_parts"
        )
        non_observable = report.get("non_observable_acceptance_tests")
        duplicate_prd_items = report.get("duplicate_prd_items")
        if missing:
            runtime_state["prd_quality_missing"] = missing
        if warnings:
            runtime_state["prd_quality_warnings"] = warnings
        if open_questions:
            runtime_state["prd_open_questions"] = open_questions
        if isinstance(uncovered, list) and uncovered:
            runtime_state["uncovered_acceptance_small_parts"] = uncovered
        if isinstance(uncovered_core, list) and uncovered_core:
            runtime_state["uncovered_smallest_working_core"] = uncovered_core
        if isinstance(uncovered_milestones, list) and uncovered_milestones:
            runtime_state["uncovered_incremental_milestone_small_parts"] = (
                uncovered_milestones
            )
        if (
            isinstance(uncovered_implementation_artifacts, list)
            and uncovered_implementation_artifacts
        ):
            runtime_state["uncovered_implementation_artifact_small_parts"] = (
                uncovered_implementation_artifacts
            )
        if (
            isinstance(uncovered_implementation_artifact_domains, list)
            and uncovered_implementation_artifact_domains
        ):
            runtime_state[
                "uncovered_implementation_artifact_domain_small_parts"
            ] = uncovered_implementation_artifact_domains
        if (
            isinstance(uncovered_test_artifact_domains, list)
            and uncovered_test_artifact_domains
        ):
            runtime_state["uncovered_test_artifact_domain_small_parts"] = (
                uncovered_test_artifact_domains
            )
        if isinstance(non_observable, list) and non_observable:
            runtime_state["non_observable_acceptance_tests"] = non_observable
        if isinstance(duplicate_prd_items, dict) and duplicate_prd_items:
            runtime_state["duplicate_prd_items"] = duplicate_prd_items
        if invalid_required_artifacts:
            runtime_state["invalid_required_artifacts"] = invalid_required_artifacts
        if prd_required_artifacts:
            runtime_state["prd_required_artifacts"] = prd_required_artifacts
        if (
            isinstance(required_incremental_milestone_count, int)
            and required_incremental_milestone_count > 0
        ):
            runtime_state["required_incremental_milestone_count"] = (
                required_incremental_milestone_count
            )
        if isinstance(required_small_part_count, int) and required_small_part_count > 0:
            runtime_state["required_small_part_count"] = required_small_part_count
        if source_required_artifacts:
            runtime_state["source_required_artifacts"] = source_required_artifacts
        if implementation_required_artifacts:
            runtime_state["implementation_required_artifacts"] = (
                implementation_required_artifacts
            )
        if test_required_artifacts:
            runtime_state["test_required_artifacts"] = test_required_artifacts
        return runtime_state

    @classmethod
    def _deterministically_repair_prd_quality(
        cls,
        prd: PRD,
        report: dict[str, Any],
    ) -> PRD | None:
        repairable_missing = {
            "acceptance_tests",
            "acceptance_tests_cover_small_parts",
            "acceptance_tests_semantically_cover_small_parts",
            "acceptance_tests_observable",
            "incremental_milestones",
            "incremental_milestones_cover_small_parts",
            "incremental_milestones_semantically_cover_small_parts",
            "prd_items_unique",
        }
        missing = set(report.get("missing") or [])
        if not missing or not missing.issubset(repairable_missing):
            return None
        original_smallest_working_core = cls._meaningful_prd_items(
            prd.smallest_working_core
        )
        original_small_parts = cls._meaningful_prd_items(prd.small_parts)
        original_incremental_milestones = cls._meaningful_prd_items(
            prd.incremental_milestones
        )
        original_acceptance_tests = cls._meaningful_prd_items(prd.acceptance_tests)
        original_definition_of_done = cls._meaningful_prd_items(prd.definition_of_done)
        original_required_artifacts = cls._meaningful_prd_items(prd.required_artifacts)

        smallest_working_core = cls._dedupe_planning_items(
            original_smallest_working_core
        )
        small_parts = cls._dedupe_planning_items(original_small_parts)
        incremental_milestones = cls._dedupe_planning_items(
            original_incremental_milestones
        )
        acceptance_tests = cls._dedupe_planning_items(original_acceptance_tests)
        definition_of_done = cls._dedupe_planning_items(original_definition_of_done)
        required_artifacts = cls._dedupe_planning_items(original_required_artifacts)
        changed = False
        added_tests: list[str] = []
        added_milestones: list[str] = []
        if (
            smallest_working_core != original_smallest_working_core
            or small_parts != original_small_parts
            or incremental_milestones != original_incremental_milestones
            or acceptance_tests != original_acceptance_tests
            or definition_of_done != original_definition_of_done
            or required_artifacts != original_required_artifacts
        ):
            changed = True
        if {
            "incremental_milestones",
            "incremental_milestones_cover_small_parts",
        } & missing and small_parts:
            for index, part in enumerate(small_parts, start=1):
                if index <= len(incremental_milestones):
                    continue
                milestone = cls._incremental_milestone_for_small_part(part)
                incremental_milestones.append(milestone)
                added_milestones.append(milestone)
                changed = True
        milestone_coverage = report.get("incremental_milestone_small_part_coverage")
        if isinstance(milestone_coverage, list):
            for item in milestone_coverage:
                if not isinstance(item, dict) or item.get("covered") is True:
                    continue
                part = item.get("small_part")
                if not isinstance(part, str) or not part.strip():
                    continue
                milestone = cls._incremental_milestone_for_small_part(part)
                if milestone not in incremental_milestones:
                    incremental_milestones.append(milestone)
                    added_milestones.append(milestone)
                    changed = True
        coverage = report.get("acceptance_test_small_part_coverage")
        small_part_by_acceptance_index: dict[int, str] = {}
        if isinstance(coverage, list):
            for item in coverage:
                if not isinstance(item, dict):
                    continue
                acceptance_index = item.get("acceptance_test_index")
                small_part = item.get("small_part")
                if isinstance(acceptance_index, int) and isinstance(small_part, str):
                    small_part_by_acceptance_index[acceptance_index] = small_part
        non_observable = report.get("non_observable_acceptance_tests")
        if isinstance(non_observable, list):
            for item in non_observable:
                if not isinstance(item, dict):
                    continue
                acceptance_index = item.get("acceptance_test_index")
                if (
                    not isinstance(acceptance_index, int)
                    or acceptance_index < 1
                    or acceptance_index > len(acceptance_tests)
                ):
                    continue
                source_text = small_part_by_acceptance_index.get(
                    acceptance_index,
                    acceptance_tests[acceptance_index - 1],
                )
                replacement = cls._acceptance_test_for_small_part(source_text)
                if acceptance_tests[acceptance_index - 1] != replacement:
                    acceptance_tests[acceptance_index - 1] = replacement
                    changed = True
                    added_tests.append(replacement)
        uncovered = report.get("uncovered_acceptance_small_parts")
        if not isinstance(uncovered, list):
            uncovered = []
        for item in uncovered:
            if not isinstance(item, dict):
                continue
            part = item.get("small_part")
            if not isinstance(part, str) or not part.strip():
                continue
            acceptance_test = cls._acceptance_test_for_small_part(part)
            if acceptance_test not in acceptance_tests:
                acceptance_tests.append(acceptance_test)
                added_tests.append(acceptance_test)
                changed = True
        if not changed:
            return None
        updates: dict[str, Any] = {}
        if smallest_working_core != original_smallest_working_core:
            updates["smallest_working_core"] = smallest_working_core
        if small_parts != original_small_parts:
            updates["small_parts"] = small_parts
        if incremental_milestones != original_incremental_milestones or added_milestones:
            updates["incremental_milestones"] = incremental_milestones
        if acceptance_tests != original_acceptance_tests or added_tests:
            updates["acceptance_tests"] = acceptance_tests
        if definition_of_done != original_definition_of_done:
            updates["definition_of_done"] = definition_of_done
        if required_artifacts != original_required_artifacts:
            updates["required_artifacts"] = required_artifacts
        return prd.model_copy(update=updates)

    @staticmethod
    def _incremental_milestone_for_small_part(small_part: str) -> str:
        part = " ".join(small_part.split())
        lower = part.lower()
        if "test" in lower or "tests" in lower:
            return f"Automated checks for {part} exist and pass"
        if "readme" in lower or "doc" in lower:
            return f"{part} documentation exists and is verified"
        return f"{part} works and is ready for focused tests"

    @staticmethod
    def _acceptance_test_for_small_part(small_part: str) -> str:
        part = " ".join(small_part.split())
        lower = part.lower()
        if "readme" in lower:
            return f"{part} exists and contains setup, run, and test instructions."
        if ".env" in lower or "env example" in lower:
            return f"{part} exists and lists required environment variables."
        if "test" in lower or "tests" in lower:
            return f"Automated checks for {part} exist and pass."
        return f"{part} works and can be verified by an observable app or API check."

    @staticmethod
    def _looks_like_observable_acceptance_test(text: str) -> bool:
        stripped = " ".join(text.split())
        lowered = stripped.lower()
        if re.search(r"\b(get|post|put|patch|delete)\s+/", lowered):
            return True
        tokens = set(re.findall(r"[a-z0-9_]+", lowered))
        if JobRunner._looks_like_generic_acceptance_test(tokens):
            return False
        observable_tokens = {
            "accepts",
            "assert",
            "asserts",
            "available",
            "can",
            "contains",
            "created",
            "delete",
            "deletes",
            "display",
            "displayed",
            "displays",
            "equal",
            "equals",
            "exist",
            "exists",
            "fail",
            "fails",
            "initializes",
            "initialized",
            "listed",
            "lists",
            "pass",
            "passes",
            "persist",
            "persists",
            "read",
            "rejects",
            "render",
            "rendered",
            "renders",
            "return",
            "returns",
            "saved",
            "shows",
            "stores",
            "successfully",
            "supports",
            "tracks",
            "update",
            "updates",
            "validates",
            "verified",
            "verify",
            "visible",
        }
        if tokens & observable_tokens:
            return True
        return False

    @staticmethod
    def _looks_like_generic_acceptance_test(tokens: set[str]) -> bool:
        if not tokens:
            return True
        generic_tokens = {
            "a",
            "all",
            "an",
            "app",
            "application",
            "as",
            "acceptance",
            "automated",
            "be",
            "behavior",
            "behaviour",
            "check",
            "checks",
            "code",
            "complete",
            "completed",
            "correctly",
            "criteria",
            "done",
            "expected",
            "feature",
            "functionality",
            "generated",
            "implementation",
            "is",
            "it",
            "module",
            "pass",
            "passes",
            "passing",
            "properly",
            "should",
            "screen",
            "service",
            "system",
            "task",
            "test",
            "tests",
            "the",
            "work",
            "working",
            "works",
        }
        return tokens <= generic_tokens

    @classmethod
    def _looks_like_generic_task_acceptance_criterion(cls, criterion: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9_]+", criterion.lower()))
        return cls._looks_like_generic_acceptance_test(tokens)

    @classmethod
    def _meaningful_task_acceptance_criteria(cls, items: list[str]) -> list[str]:
        return [
            item
            for item in cls._meaningful_planning_items(items)
            if not cls._looks_like_generic_task_acceptance_criterion(item)
        ]

    @classmethod
    def _smallest_working_core_coverage(
        cls,
        core_items: list[str],
        small_parts: list[str],
    ) -> list[dict[str, Any]]:
        small_part_tokens: set[str] = set()
        for part in small_parts:
            small_part_tokens.update(
                cls._semantic_tokens(part, include_action_tokens=True)
            )
        coverage: list[dict[str, Any]] = []
        for index, core in enumerate(core_items, start=1):
            core_tokens = cls._semantic_tokens(core, include_action_tokens=True)
            anchor_tokens = cls._semantic_anchor_tokens(core_tokens)
            if not core_tokens or not anchor_tokens:
                coverage.append(
                    {
                        "core_index": index,
                        "smallest_working_core": core,
                        "required_anchor_tokens": sorted(anchor_tokens),
                        "covered_anchor_tokens": [],
                        "missing_anchor_tokens": [],
                        "covered": True,
                    }
                )
                continue
            covered_anchor_tokens = sorted(anchor_tokens & small_part_tokens)
            missing_anchor_tokens = sorted(anchor_tokens - small_part_tokens)
            coverage.append(
                {
                    "core_index": index,
                    "smallest_working_core": core,
                    "required_anchor_tokens": sorted(anchor_tokens),
                    "covered_anchor_tokens": covered_anchor_tokens,
                    "missing_anchor_tokens": missing_anchor_tokens,
                    "covered": not missing_anchor_tokens,
                }
            )
        return coverage

    @classmethod
    def _incremental_milestone_small_part_coverage(
        cls,
        small_parts: list[str],
        incremental_milestones: list[str],
    ) -> list[dict[str, Any]]:
        milestone_tokens: set[str] = set()
        for milestone in incremental_milestones:
            milestone_tokens.update(
                cls._semantic_tokens(milestone, include_action_tokens=True)
            )
        coverage: list[dict[str, Any]] = []
        for index, part in enumerate(small_parts, start=1):
            part_tokens = cls._semantic_tokens(part)
            anchor_tokens = cls._semantic_anchor_tokens(part_tokens)
            if not part_tokens or not anchor_tokens:
                coverage.append(
                    {
                        "small_part_index": index,
                        "small_part": part,
                        "required_anchor_tokens": sorted(anchor_tokens),
                        "covered_anchor_tokens": [],
                        "missing_anchor_tokens": [],
                        "covered": True,
                    }
                )
                continue
            covered = cls._semantic_anchor_satisfied(anchor_tokens, milestone_tokens)
            covered_anchor_tokens = (
                sorted(anchor_tokens)
                if covered
                else sorted(anchor_tokens & milestone_tokens)
            )
            missing_anchor_tokens = (
                [] if covered else sorted(anchor_tokens - milestone_tokens)
            )
            coverage.append(
                {
                    "small_part_index": index,
                    "small_part": part,
                    "required_anchor_tokens": sorted(anchor_tokens),
                    "covered_anchor_tokens": covered_anchor_tokens,
                    "missing_anchor_tokens": missing_anchor_tokens,
                    "covered": covered,
                }
            )
        return coverage

    @staticmethod
    def _prd_quality_repair_logs(prd: PRD, report: dict[str, Any]) -> list[str]:
        logs = [
            "The previous PRD was not specific enough for autonomous execution.",
            f"Missing fields: {', '.join(report['missing'])}",
            f"Warnings: {', '.join(report['warnings'])}",
        ]
        uncovered = report.get("uncovered_acceptance_small_parts")
        if isinstance(uncovered, list) and uncovered:
            summaries: list[str] = []
            for item in uncovered:
                if not isinstance(item, dict):
                    continue
                index = item.get("small_part_index")
                part = item.get("small_part")
                if isinstance(part, str) and part.strip():
                    summaries.append(f"{index}: {part}" if index else part)
            if summaries:
                logs.append(
                    "Uncovered PRD small_parts needing direct acceptance_tests: "
                    + " | ".join(summaries)
                )
                logs.append(
                    "Repair acceptance_tests by adding or rewriting one observable check "
                    "for each uncovered small_part, reusing distinctive nouns and verbs from that small_part."
                )
        uncovered_core = report.get("uncovered_smallest_working_core")
        if isinstance(uncovered_core, list) and uncovered_core:
            summaries = []
            for item in uncovered_core:
                if not isinstance(item, dict):
                    continue
                index = item.get("core_index")
                core = item.get("smallest_working_core")
                missing_anchors = item.get("missing_anchor_tokens")
                if not isinstance(core, str) or not core.strip():
                    continue
                suffix = ""
                if isinstance(missing_anchors, list) and missing_anchors:
                    suffix = " missing anchors " + ", ".join(
                        str(token) for token in missing_anchors
                    )
                summaries.append(
                    f"{index}: {core}{suffix}" if index else f"{core}{suffix}"
                )
            if summaries:
                logs.append(
                    "Smallest working core items not represented in small_parts: "
                    + " | ".join(summaries)
                )
                logs.append(
                    "Rewrite small_parts so they carry the distinctive domain anchors "
                    "from smallest_working_core before planning implementation tasks."
                )
        uncovered_implementation = report.get(
            "uncovered_implementation_artifact_small_parts"
        )
        if isinstance(uncovered_implementation, list) and uncovered_implementation:
            summaries = []
            for item in uncovered_implementation:
                if not isinstance(item, dict):
                    continue
                index = item.get("small_part_index")
                part = item.get("small_part")
                missing_surfaces = item.get("missing_surfaces")
                if not isinstance(part, str) or not part.strip():
                    continue
                surfaces_text = ""
                if isinstance(missing_surfaces, list) and missing_surfaces:
                    surface_names = ", ".join(
                        str(surface) for surface in missing_surfaces
                    )
                    surfaces_text = f" missing {surface_names}"
                summaries.append(
                    f"{index}: {part}{surfaces_text}"
                    if index
                    else f"{part}{surfaces_text}"
                )
            if summaries:
                logs.append(
                    "Implementation small_parts without matching required_artifacts: "
                    + " | ".join(summaries)
                )
                logs.append(
                    "Add implementation required_artifacts for each missing surface, such as "
                    "backend/main.py for backend API work, frontend/src/App.tsx for frontend UI work, "
                    "or shared schema/type files for shared contracts."
                )
        non_observable = report.get("non_observable_acceptance_tests")
        if isinstance(non_observable, list) and non_observable:
            summaries = []
            for item in non_observable:
                if not isinstance(item, dict):
                    continue
                index = item.get("acceptance_test_index")
                acceptance_test = item.get("acceptance_test")
                if isinstance(acceptance_test, str) and acceptance_test.strip():
                    summaries.append(
                        f"{index}: {acceptance_test}" if index else acceptance_test
                    )
            if summaries:
                logs.append(
                    "Non-observable acceptance_tests that restate work instead of proving behavior: "
                    + " | ".join(summaries)
                )
                logs.append(
                    "Rewrite each acceptance_test as an observable result, for example exists, "
                    "returns, renders, contains, validates, persists, or passes."
                )
        duplicate_prd_items = report.get("duplicate_prd_items")
        if isinstance(duplicate_prd_items, dict) and duplicate_prd_items:
            summaries = []
            for section, duplicates in duplicate_prd_items.items():
                if not isinstance(duplicates, list):
                    continue
                items = []
                for duplicate in duplicates:
                    if not isinstance(duplicate, dict):
                        continue
                    item = duplicate.get("item")
                    duplicate_indices = duplicate.get("duplicate_indices")
                    if not isinstance(item, str) or not item.strip():
                        continue
                    indices_text = ""
                    if isinstance(duplicate_indices, list) and duplicate_indices:
                        indices_text = (
                            " duplicated at "
                            + ", ".join(str(index) for index in duplicate_indices)
                        )
                    items.append(f"{item}{indices_text}")
                if items:
                    summaries.append(f"{section}: " + " | ".join(items))
            if summaries:
                logs.append(
                    "Duplicate PRD items are not valid coverage: "
                    + " ; ".join(summaries)
                )
                logs.append(
                    "Remove duplicate PRD entries and replace any padding with distinct, "
                    "independently verifiable work, milestone, or acceptance-test items."
                )
        if prd.acceptance_tests:
            logs.append(
                "Current acceptance_tests: "
                + " | ".join(self_item for self_item in prd.acceptance_tests)
            )
        required_artifacts = JobRunner._valid_unique_planning_artifact_paths(
            prd.required_artifacts
        )
        if required_artifacts:
            logs.append("Current required_artifacts: " + " | ".join(required_artifacts))
        missing = set(report.get("missing") or [])
        if "incremental_milestones" in missing:
            logs.append(
                "Add incremental_milestones that show the build order from smallest working core "
                "through each independently testable small_part."
            )
        if "incremental_milestones_cover_small_parts" in missing:
            required_count = report.get("required_incremental_milestone_count")
            required_text = (
                str(required_count)
                if isinstance(required_count, int) and required_count > 0
                else "one per small_part"
            )
            logs.append(
                "Add at least "
                f"{required_text} incremental_milestones so the milestone sequence covers every small_part."
            )
        if "incremental_milestones_semantically_cover_small_parts" in missing:
            uncovered_milestones = report.get(
                "uncovered_incremental_milestone_small_parts"
            )
            if isinstance(uncovered_milestones, list) and uncovered_milestones:
                summaries = []
                for item in uncovered_milestones:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("small_part_index")
                    part = item.get("small_part")
                    missing_anchors = item.get("missing_anchor_tokens")
                    if not isinstance(part, str) or not part.strip():
                        continue
                    suffix = ""
                    if isinstance(missing_anchors, list) and missing_anchors:
                        suffix = " missing anchors " + ", ".join(
                            str(token) for token in missing_anchors
                        )
                    summaries.append(
                        f"{index}: {part}{suffix}" if index else f"{part}{suffix}"
                    )
                if summaries:
                    logs.append(
                        "Incremental milestones not tied to small_parts: "
                        + " | ".join(summaries)
                    )
            logs.append(
                "Rewrite incremental_milestones so each domain-specific small_part "
                "has a milestone carrying the same distinctive anchors."
            )
        if "small_parts_split_for_autonomy" in missing:
            required_count = report.get("required_small_part_count")
            required_text = (
                str(required_count)
                if isinstance(required_count, int) and required_count > 0
                else "multiple"
            )
            logs.append(
                "Split small_parts into at least "
                f"{required_text} independently executable implementation/test steps; "
                "a single broad work item is not specific enough for large autonomous execution."
            )
        if "required_source_artifacts" in missing:
            logs.append(
                "Add at least one non-test required_artifact for the implementation or app surface; "
                "tests alone are not enough for autonomous implementation."
            )
        if "required_implementation_artifacts" in missing:
            logs.append(
                "Add at least one implementation source required_artifact such as backend/main.py, "
                "frontend/src/App.tsx, src/server.ts, or app.py; README, .env, and package manifests "
                "alone are not enough for app implementation."
            )
        if "implementation_artifacts_cover_small_parts" in missing:
            logs.append(
                "Ensure required_artifacts includes implementation source files for every "
                "backend, frontend, and shared small_part surface described in the PRD."
            )
        if "implementation_artifacts_semantically_cover_small_parts" in missing:
            uncovered_domains = report.get(
                "uncovered_implementation_artifact_domain_small_parts"
            )
            if isinstance(uncovered_domains, list) and uncovered_domains:
                summaries = []
                for item in uncovered_domains:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("small_part_index")
                    part = item.get("small_part")
                    domain_tokens = item.get("required_domain_tokens")
                    if not isinstance(part, str) or not part.strip():
                        continue
                    suffix = ""
                    if isinstance(domain_tokens, list) and domain_tokens:
                        suffix = " required domain tokens " + ", ".join(
                            str(token) for token in domain_tokens
                        )
                    summaries.append(
                        f"{index}: {part}{suffix}" if index else f"{part}{suffix}"
                    )
                if summaries:
                    logs.append(
                        "Implementation artifacts are too generic for small_parts: "
                        + " | ".join(summaries)
                    )
            logs.append(
                "Rewrite required_artifacts to include domain-specific implementation files "
                "for each app behavior, not only generic entrypoints such as backend/main.py."
            )
        if "required_test_artifacts" in missing:
            logs.append(
                "Add at least one test required_artifact such as tests/test_*.py or frontend test/*.test.tsx."
            )
        if "test_artifacts_semantically_cover_small_parts" in missing:
            uncovered_tests = report.get("uncovered_test_artifact_domain_small_parts")
            if isinstance(uncovered_tests, list) and uncovered_tests:
                summaries = []
                for item in uncovered_tests:
                    if not isinstance(item, dict):
                        continue
                    index = item.get("small_part_index")
                    part = item.get("small_part")
                    domain_tokens = item.get("required_domain_tokens")
                    if not isinstance(part, str) or not part.strip():
                        continue
                    suffix = ""
                    if isinstance(domain_tokens, list) and domain_tokens:
                        suffix = " required domain tokens " + ", ".join(
                            str(token) for token in domain_tokens
                        )
                    summaries.append(
                        f"{index}: {part}{suffix}" if index else f"{part}{suffix}"
                    )
                if summaries:
                    logs.append(
                        "Test artifacts are too generic for small_parts: "
                        + " | ".join(summaries)
                    )
            logs.append(
                "Rewrite test required_artifacts to include domain-specific test files "
                "for each app behavior, not only generic project setup tests."
            )
        open_questions = JobRunner._meaningful_prd_items(prd.open_questions)
        if open_questions:
            logs.append("Open questions blocking autonomy: " + " | ".join(open_questions))
            logs.append(
                "Resolve open_questions before implementation by converting each one "
                "into an explicit assumption, constraint, non_goal, or acceptance test; "
                "then return open_questions as an empty list."
            )
        return logs

    def _record_prd_quality_attempt(
        self,
        record: JobRecord,
        *,
        attempt: int,
        action: str,
        report: dict[str, Any],
    ) -> None:
        attempts = record.outputs.setdefault("prd_quality_attempts", [])
        if not isinstance(attempts, list):
            attempts = []
            record.outputs["prd_quality_attempts"] = attempts
        attempts.append(
            {
                "attempt": attempt,
                "action": action,
                "passed": report["passed"],
                "missing": list(report["missing"]),
                "warnings": list(report["warnings"]),
            }
        )

    @staticmethod
    def _build_prd_quality_report(
        prd: PRD,
        *,
        min_small_parts: int = 0,
    ) -> dict[str, Any]:
        missing: list[str] = []
        warnings: list[str] = []
        if (
            not prd.title.strip()
            or JobRunner._looks_like_placeholder_prd_item(prd.title)
        ):
            missing.append("title")
        if (
            not prd.problem_statement.strip()
            or JobRunner._looks_like_placeholder_prd_item(prd.problem_statement)
        ):
            missing.append("problem_statement")
        smallest_working_core = JobRunner._meaningful_prd_items(
            prd.smallest_working_core
        )
        duplicate_smallest_working_core = JobRunner._duplicate_planning_items(
            smallest_working_core
        )
        if not smallest_working_core:
            missing.append("smallest_working_core")
        small_parts = JobRunner._meaningful_prd_items(prd.small_parts)
        duplicate_small_parts = JobRunner._duplicate_planning_items(small_parts)
        if not small_parts:
            missing.append("small_parts")
        elif len(small_parts) == 1:
            warnings.append("small_parts_has_single_item")
        smallest_working_core_coverage = JobRunner._smallest_working_core_coverage(
            smallest_working_core,
            small_parts,
        )
        uncovered_smallest_working_core = [
            item for item in smallest_working_core_coverage if not item["covered"]
        ]
        if (
            smallest_working_core
            and small_parts
            and uncovered_smallest_working_core
        ):
            missing.append("smallest_working_core_covered_by_small_parts")
        if min_small_parts > 0 and small_parts and len(small_parts) < min_small_parts:
            missing.append("small_parts_split_for_autonomy")
        incremental_milestones = JobRunner._meaningful_prd_items(
            prd.incremental_milestones
        )
        duplicate_incremental_milestones = JobRunner._duplicate_planning_items(
            incremental_milestones
        )
        if not incremental_milestones:
            missing.append("incremental_milestones")
        elif small_parts and len(incremental_milestones) < len(small_parts):
            missing.append("incremental_milestones_cover_small_parts")
        incremental_milestone_small_part_coverage = (
            JobRunner._incremental_milestone_small_part_coverage(
                small_parts,
                incremental_milestones,
            )
        )
        uncovered_incremental_milestone_small_parts = [
            item
            for item in incremental_milestone_small_part_coverage
            if not item["covered"]
        ]
        if (
            small_parts
            and incremental_milestones
            and len(incremental_milestones) >= len(small_parts)
            and uncovered_incremental_milestone_small_parts
        ):
            missing.append("incremental_milestones_semantically_cover_small_parts")
        acceptance_tests = JobRunner._meaningful_prd_items(prd.acceptance_tests)
        duplicate_acceptance_tests = JobRunner._duplicate_planning_items(
            acceptance_tests
        )
        acceptance_test_small_part_coverage = JobRunner._semantic_item_coverage(
            small_parts,
            acceptance_tests,
            item_key="small_part",
            index_key="small_part_index",
            candidate_key="acceptance_test",
            candidate_index_key="acceptance_test_index",
        )
        uncovered_acceptance_small_parts = [
            item for item in acceptance_test_small_part_coverage if not item["covered"]
        ]
        non_observable_acceptance_tests = [
            {
                "acceptance_test_index": index,
                "acceptance_test": acceptance_test,
            }
            for index, acceptance_test in enumerate(acceptance_tests, start=1)
            if not JobRunner._looks_like_observable_acceptance_test(acceptance_test)
        ]
        if not acceptance_tests:
            missing.append("acceptance_tests")
        elif small_parts and len(acceptance_tests) < len(small_parts):
            missing.append("acceptance_tests_cover_small_parts")
        elif small_parts and uncovered_acceptance_small_parts:
            missing.append("acceptance_tests_semantically_cover_small_parts")
        if acceptance_tests and non_observable_acceptance_tests:
            missing.append("acceptance_tests_observable")
        definition_of_done = JobRunner._meaningful_prd_items(prd.definition_of_done)
        duplicate_definition_of_done = JobRunner._duplicate_planning_items(
            definition_of_done
        )
        if not definition_of_done:
            missing.append("definition_of_done")
        required_artifact_items = JobRunner._meaningful_prd_items(
            prd.required_artifacts
        )
        duplicate_required_artifacts = JobRunner._duplicate_planning_items(
            required_artifact_items
        )
        required_artifacts = set(
            JobRunner._valid_unique_planning_artifact_paths(required_artifact_items)
        )
        invalid_required_artifacts = JobRunner._invalid_planning_artifact_paths(
            required_artifact_items
        )
        if not required_artifacts:
            missing.append("required_artifacts")
        if invalid_required_artifacts:
            missing.append("required_artifacts_valid_paths")
        source_required_artifacts = sorted(
            path
            for path in required_artifacts
            if not JobRunner._looks_like_test_path(path)
        )
        implementation_required_artifacts = sorted(
            path
            for path in source_required_artifacts
            if JobRunner._looks_like_implementation_source_path(path)
        )
        test_required_artifacts = sorted(
            path for path in required_artifacts if JobRunner._looks_like_test_path(path)
        )
        if acceptance_tests and required_artifacts and not source_required_artifacts:
            missing.append("required_source_artifacts")
        implementation_small_parts = [
            part
            for part in small_parts
            if JobRunner._looks_like_implementation_work_item(part)
        ]
        implementation_artifact_small_part_coverage = (
            JobRunner._implementation_artifact_surface_coverage(
                small_parts,
                implementation_required_artifacts,
            )
        )
        uncovered_implementation_artifact_small_parts = [
            item
            for item in implementation_artifact_small_part_coverage
            if not item["covered"]
        ]
        implementation_artifact_domain_coverage = (
            JobRunner._implementation_artifact_domain_coverage(
                small_parts,
                implementation_required_artifacts,
            )
        )
        uncovered_implementation_artifact_domain_small_parts = [
            item
            for item in implementation_artifact_domain_coverage
            if not item["covered"]
        ]
        test_artifact_domain_coverage = JobRunner._test_artifact_domain_coverage(
            small_parts,
            test_required_artifacts,
        )
        uncovered_test_artifact_domain_small_parts = [
            item for item in test_artifact_domain_coverage if not item["covered"]
        ]
        if (
            acceptance_tests
            and source_required_artifacts
            and implementation_small_parts
            and not implementation_required_artifacts
        ):
            missing.append("required_implementation_artifacts")
        if (
            acceptance_tests
            and implementation_small_parts
            and implementation_required_artifacts
            and uncovered_implementation_artifact_small_parts
        ):
            missing.append("implementation_artifacts_cover_small_parts")
        if (
            acceptance_tests
            and implementation_small_parts
            and implementation_required_artifacts
            and uncovered_implementation_artifact_domain_small_parts
        ):
            missing.append("implementation_artifacts_semantically_cover_small_parts")
        if acceptance_tests and required_artifacts and not test_required_artifacts:
            missing.append("required_test_artifacts")
        if (
            acceptance_tests
            and implementation_small_parts
            and test_required_artifacts
            and uncovered_test_artifact_domain_small_parts
        ):
            missing.append("test_artifacts_semantically_cover_small_parts")
        duplicate_prd_items = {
            key: duplicates
            for key, duplicates in {
                "smallest_working_core": duplicate_smallest_working_core,
                "small_parts": duplicate_small_parts,
                "incremental_milestones": duplicate_incremental_milestones,
                "acceptance_tests": duplicate_acceptance_tests,
                "definition_of_done": duplicate_definition_of_done,
                "required_artifacts": duplicate_required_artifacts,
            }.items()
            if duplicates
        }
        if duplicate_prd_items:
            missing.append("prd_items_unique")
        if JobRunner._meaningful_prd_items(prd.open_questions):
            missing.append("open_questions_resolved")
            warnings.append("open_questions_present")
        missing_acceptance_test_count = max(0, len(small_parts) - len(acceptance_tests))
        missing_incremental_milestone_count = max(
            0,
            len(small_parts) - len(incremental_milestones),
        )
        report = {
            "passed": not missing,
            "missing": missing,
            "warnings": warnings,
            "small_part_count": len(small_parts),
            "smallest_working_core_covered_by_small_parts": (
                not uncovered_smallest_working_core
            ),
            "smallest_working_core_coverage": smallest_working_core_coverage,
            "uncovered_smallest_working_core": uncovered_smallest_working_core,
            "incremental_milestone_count": len(incremental_milestones),
            "incremental_milestones_cover_small_parts": (
                missing_incremental_milestone_count == 0
            ),
            "incremental_milestones_semantically_cover_small_parts": (
                not uncovered_incremental_milestone_small_parts
            ),
            "incremental_milestone_small_part_coverage": (
                incremental_milestone_small_part_coverage
            ),
            "uncovered_incremental_milestone_small_parts": (
                uncovered_incremental_milestone_small_parts
            ),
            "missing_incremental_milestone_count": missing_incremental_milestone_count,
            "required_incremental_milestone_count": len(small_parts),
            "acceptance_test_count": len(acceptance_tests),
            "acceptance_tests_cover_small_parts": missing_acceptance_test_count == 0,
            "missing_acceptance_test_count": missing_acceptance_test_count,
            "acceptance_tests_semantically_cover_small_parts": (
                not uncovered_acceptance_small_parts
            ),
            "acceptance_tests_observable": not non_observable_acceptance_tests,
            "acceptance_test_small_part_coverage": acceptance_test_small_part_coverage,
            "uncovered_acceptance_small_parts": uncovered_acceptance_small_parts,
            "non_observable_acceptance_tests": non_observable_acceptance_tests,
            "definition_of_done_count": len(
                definition_of_done
            ),
            "required_artifact_count": len(required_artifacts),
            "required_artifacts": sorted(required_artifacts),
            "source_required_artifact_count": len(source_required_artifacts),
            "source_required_artifacts": source_required_artifacts,
            "implementation_required_artifact_count": len(
                implementation_required_artifacts
            ),
            "implementation_required_artifacts": implementation_required_artifacts,
            "implementation_artifacts_cover_small_parts": (
                not uncovered_implementation_artifact_small_parts
            ),
            "implementation_artifact_small_part_coverage": (
                implementation_artifact_small_part_coverage
            ),
            "uncovered_implementation_artifact_small_parts": (
                uncovered_implementation_artifact_small_parts
            ),
            "implementation_artifacts_semantically_cover_small_parts": (
                not uncovered_implementation_artifact_domain_small_parts
            ),
            "implementation_artifact_domain_coverage": (
                implementation_artifact_domain_coverage
            ),
            "uncovered_implementation_artifact_domain_small_parts": (
                uncovered_implementation_artifact_domain_small_parts
            ),
            "test_required_artifact_count": len(test_required_artifacts),
            "test_required_artifacts": test_required_artifacts,
            "test_artifacts_semantically_cover_small_parts": (
                not uncovered_test_artifact_domain_small_parts
            ),
            "test_artifact_domain_coverage": test_artifact_domain_coverage,
            "uncovered_test_artifact_domain_small_parts": (
                uncovered_test_artifact_domain_small_parts
            ),
            "invalid_required_artifacts": invalid_required_artifacts,
        }
        if duplicate_prd_items:
            report["duplicate_prd_items"] = duplicate_prd_items
        if min_small_parts > 0:
            report["required_small_part_count"] = min_small_parts
            report["small_parts_split_for_autonomy"] = len(small_parts) >= min_small_parts
        return report

    @staticmethod
    def _non_empty_items(items: list[str]) -> list[str]:
        return [item.strip() for item in items if item.strip()]

    @classmethod
    def _meaningful_prd_items(cls, items: list[str]) -> list[str]:
        return cls._meaningful_planning_items(items)

    @classmethod
    def _meaningful_planning_items(cls, items: list[str]) -> list[str]:
        return [
            item
            for item in cls._non_empty_items(items)
            if not cls._looks_like_placeholder_prd_item(item)
        ]

    @staticmethod
    def _planning_item_identity(item: str) -> str:
        return " ".join(str(item).split()).lower()

    @classmethod
    def _duplicate_planning_items(cls, items: list[str]) -> list[dict[str, Any]]:
        first_seen: dict[str, tuple[int, str]] = {}
        duplicates_by_key: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(items, start=1):
            key = cls._planning_item_identity(item)
            if not key:
                continue
            if key not in first_seen:
                first_seen[key] = (index, item)
                continue
            first_index, first_item = first_seen[key]
            duplicate = duplicates_by_key.setdefault(
                key,
                {
                    "item": first_item,
                    "first_index": first_index,
                    "duplicate_indices": [],
                },
            )
            duplicate["duplicate_indices"].append(index)
        return list(duplicates_by_key.values())

    @classmethod
    def _dedupe_planning_items(cls, items: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            key = cls._planning_item_identity(item)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _looks_like_placeholder_prd_item(item: str) -> bool:
        value = " ".join(str(item).split())
        if not value:
            return True
        lowered = value.lower().strip(" .:-_[]()")
        compact = re.sub(r"[^a-z0-9]+", "", lowered)
        if not compact:
            return True
        exact_placeholders = {
            "coming soon",
            "fill in later",
            "fill me in",
            "fixme",
            "n/a",
            "na",
            "none",
            "none known",
            "no open questions",
            "not applicable",
            "not specified",
            "placeholder",
            "tbd",
            "to be decided",
            "to be defined",
            "to be determined",
            "todo",
            "unknown",
            "unspecified",
        }
        compact_placeholders = {
            "comingsoon",
            "fillinlater",
            "fillmein",
            "fixme",
            "loremipsum",
            "na",
            "none",
            "noneknown",
            "noopenquestions",
            "notapplicable",
            "notspecified",
            "placeholder",
            "tbd",
            "tobedecided",
            "tobedefined",
            "tobedetermined",
            "todo",
            "unknown",
            "unspecified",
        }
        return lowered in exact_placeholders or compact in compact_placeholders

    def _refine_task_graph_for_autonomy(
        self,
        record: JobRecord,
        prd: PRD,
        task_graph: TaskGraph,
    ) -> TaskGraph:
        if self._constraint_flag(record, "disable_task_graph_refinement"):
            return task_graph
        implementation_tasks = self._tasks_for_roles(task_graph, self.IMPLEMENTATION_TASK_ROLES)
        small_parts = [item.strip() for item in prd.small_parts if item.strip()]
        if len(implementation_tasks) > 1 or len(small_parts) < 2:
            task_graph = self._enrich_task_graph_acceptance_criteria(record, prd, task_graph)
            record.outputs["task_graph"] = task_graph.model_dump()
            self.store.update(record)
            return task_graph

        tasks: list[PlannedTask] = []
        previous_id: str | None = None
        acceptance_tests = [item.strip() for item in prd.acceptance_tests if item.strip()]
        raw_source_target_files = [
            path
            for task in implementation_tasks
            for path in task.target_files
        ]
        raw_source_required_artifacts = [
            *[
                path
                for task in implementation_tasks
                for path in task.required_artifacts
            ],
            *prd.required_artifacts,
        ]
        source_target_files = self._valid_unique_planning_artifact_paths(
            raw_source_target_files
        )
        source_required_artifacts = self._valid_unique_planning_artifact_paths(
            raw_source_required_artifacts
        )
        source_implementation_artifacts = [
            path
            for path in source_required_artifacts
            if not self._looks_like_test_path(path)
        ]
        source_implementation_targets = [
            path
            for path in source_target_files
            if not self._looks_like_test_path(path)
        ] or source_implementation_artifacts
        source_test_artifacts = [
            path for path in source_required_artifacts if self._looks_like_test_path(path)
        ]
        invalid_inherited_artifacts = self._unique_paths(
            self._invalid_planning_artifact_paths(
                [*raw_source_target_files, *raw_source_required_artifacts]
            )
        )
        for index, part in enumerate(small_parts, start=1):
            task_id = f"part-{index:02d}"
            role = (
                "test_writer"
                if self._looks_like_test_work_item(part)
                else "implementer"
            )
            criteria = (
                [acceptance_tests[index - 1]]
                if index <= len(acceptance_tests)
                else [f"{part} works and existing behavior remains covered by tests."]
            )
            task = PlannedTask(
                id=task_id,
                title=self._task_title_from_part(part),
                description=(
                    (
                        "Write only the focused tests for this small part before moving on: "
                        if role == "test_writer"
                        else "Implement only this small part before moving on: "
                    )
                    + f"{part}. Keep the change narrow enough to test immediately."
                ),
                role=role,
                complexity=TaskComplexity.MEDIUM,
                depends_on=[previous_id] if previous_id is not None else [],
                acceptance_criteria=criteria,
                target_files=(
                    source_test_artifacts
                    if role == "test_writer"
                    else source_implementation_targets
                ),
                required_artifacts=(
                    source_test_artifacts
                    if role == "test_writer"
                    else source_implementation_artifacts or source_implementation_targets
                ),
            )
            tasks.append(task)
            previous_id = task_id
            if role != "test_writer" and source_test_artifacts:
                test_task_id = f"{task_id}-tests"
                tasks.append(
                    PlannedTask(
                        id=test_task_id,
                        title=f"{task.title} tests",
                        description=(
                            "Write focused tests for this completed small part before "
                            f"the next implementation task: {part}."
                        ),
                        role="test_writer",
                        complexity=TaskComplexity.MEDIUM,
                        depends_on=[task_id],
                        acceptance_criteria=criteria,
                        target_files=source_test_artifacts,
                        required_artifacts=source_test_artifacts,
                    )
                )
                previous_id = test_task_id

        refined = TaskGraph(
            goal=task_graph.goal,
            tasks=tasks,
            notes=[
                *task_graph.notes,
                "ACOS refined a coarse planner graph using PM small_parts for autonomous execution.",
            ],
        )
        record.outputs["planner_task_graph_raw"] = task_graph.model_dump()
        record.outputs["task_graph"] = refined.model_dump()
        record.outputs["task_graph_refinement"] = {
            "applied": True,
            "reason": "coarse_planner_graph_with_pm_small_parts",
            "original_task_count": len(task_graph.tasks),
            "refined_task_count": len(refined.tasks),
            "inherited_target_files": source_target_files,
            "inherited_required_artifacts": source_required_artifacts,
            "inherited_implementation_artifacts": source_implementation_artifacts,
            "inherited_implementation_targets": source_implementation_targets,
            "inherited_test_artifacts": source_test_artifacts,
            "invalid_inherited_artifacts": invalid_inherited_artifacts,
            "paired_test_task_count": len(
                [
                    task
                    for task in refined.tasks
                    if task.role == "test_writer"
                    and task.id.endswith("-tests")
                ]
            ),
        }
        self.store.update(record)
        return refined

    def _enrich_task_graph_acceptance_criteria(
        self,
        record: JobRecord,
        prd: PRD,
        task_graph: TaskGraph,
    ) -> TaskGraph:
        acceptance_tests = self._meaningful_prd_items(prd.acceptance_tests)
        definition_of_done = self._meaningful_prd_items(prd.definition_of_done)
        source_required_artifacts = self._valid_unique_planning_artifact_paths(
            self._meaningful_prd_items(prd.required_artifacts)
        )
        source_implementation_artifacts = [
            path
            for path in source_required_artifacts
            if not self._looks_like_test_path(path)
        ]
        source_test_artifacts = [
            path
            for path in source_required_artifacts
            if self._looks_like_test_path(path)
        ]
        implementation_task_count = len(
            [
                task
                for task in task_graph.tasks
                if task.role in self.IMPLEMENTATION_TASK_ROLES
            ]
        )
        test_writer_task_count = len(
            [
                task
                for task in task_graph.tasks
                if task.role in self.TEST_TASK_ROLES
            ]
        )
        project_setup_scaffold_covers_test_artifacts = (
            bool(source_test_artifacts)
            and all(
                path in self.PROJECT_SETUP_REQUIRED_ARTIFACTS
                for path in source_test_artifacts
            )
            and any(
                task.role == "scaffold"
                and self._is_project_setup_task(task)
                and set(source_test_artifacts).issubset(
                    self._valid_unique_artifact_paths(task.target_files)
                )
                for task in task_graph.tasks
            )
        )
        should_synthesize_test_writer_tasks = (
            test_writer_task_count == 0
            and implementation_task_count > 0
            and bool(source_test_artifacts)
            and not project_setup_scaffold_covers_test_artifacts
        )
        if (
            not acceptance_tests
            and not definition_of_done
            and not source_implementation_artifacts
            and not source_test_artifacts
        ):
            record.outputs["task_graph_acceptance_enrichment"] = {
                "applied": False,
                "reason": "no_prd_acceptance_sources",
                "updated_task_ids": [],
            }
            return task_graph

        updated_task_ids: list[str] = []
        artifact_updated_task_ids: list[str] = []
        implementation_index = 0
        test_writer_index = 0
        synthesized_test_writer_task_ids: list[str] = []
        used_task_ids = {task.id for task in task_graph.tasks}
        implementation_dependency_ids: list[str] = []
        assigned_synthetic_test_artifacts: set[str] = set()
        tasks: list[PlannedTask] = []
        for task in task_graph.tasks:
            if task.role in self.IMPLEMENTATION_TASK_ROLES:
                task_artifacts = self._prd_artifacts_for_task(
                    task,
                    source_implementation_artifacts,
                    task_count=implementation_task_count,
                )
                criteria_index = implementation_index
                implementation_index += 1
                implementation_dependency_ids.append(task.id)
            elif task.role in self.TEST_TASK_ROLES:
                task_artifacts = self._prd_artifacts_for_task(
                    task,
                    source_test_artifacts,
                    task_count=test_writer_task_count,
                )
                criteria_index = test_writer_index
                test_writer_index += 1
            else:
                tasks.append(task)
                continue
            updates: dict[str, Any] = {}
            if (
                not self._meaningful_planning_items(task.acceptance_criteria)
                and (acceptance_tests or definition_of_done)
            ):
                criteria = self._criteria_for_task_from_prd(
                    task,
                    acceptance_tests,
                    definition_of_done,
                    criteria_index,
                )
                updates["acceptance_criteria"] = criteria
                updated_task_ids.append(task.id)
            if task_artifacts:
                target_files = self._unique_paths([*task.target_files, *task_artifacts])
                required_artifacts = self._unique_paths(
                    [*task.required_artifacts, *task_artifacts]
                )
                if target_files != task.target_files:
                    updates["target_files"] = target_files
                if required_artifacts != task.required_artifacts:
                    updates["required_artifacts"] = required_artifacts
                if "target_files" in updates or "required_artifacts" in updates:
                    artifact_updated_task_ids.append(task.id)
            updated_task = task.model_copy(update=updates) if updates else task
            tasks.append(updated_task)
            if (
                should_synthesize_test_writer_tasks
                and task.role in self.IMPLEMENTATION_TASK_ROLES
            ):
                test_task_id = self._unique_task_id(f"{task.id}-tests", used_task_ids)
                used_task_ids.add(test_task_id)
                criteria = self._criteria_for_task_from_prd(
                    task,
                    acceptance_tests,
                    definition_of_done,
                    criteria_index,
                )
                synthetic_test_task = PlannedTask(
                    id=test_task_id,
                    title=f"{task.title} tests",
                    description=(
                        "Write focused tests for this implementation task before "
                        f"the next stage: {task.title}."
                    ),
                    role="test_writer",
                    complexity=TaskComplexity.MEDIUM,
                    depends_on=[task.id],
                    acceptance_criteria=criteria,
                )
                matched_test_artifacts = self._prd_artifacts_for_task(
                    synthetic_test_task,
                    source_test_artifacts,
                    task_count=implementation_task_count,
                )
                test_artifacts = [
                    artifact
                    for artifact in matched_test_artifacts
                    if artifact not in assigned_synthetic_test_artifacts
                ]
                if not test_artifacts:
                    continue
                tasks.append(
                    synthetic_test_task.model_copy(
                        update={
                            "target_files": test_artifacts,
                            "required_artifacts": test_artifacts,
                        }
                    )
                )
                synthesized_test_writer_task_ids.append(test_task_id)
                artifact_updated_task_ids.append(test_task_id)
                assigned_synthetic_test_artifacts.update(test_artifacts)

        if should_synthesize_test_writer_tasks:
            unassigned_test_artifacts = [
                artifact
                for artifact in source_test_artifacts
                if artifact not in assigned_synthetic_test_artifacts
            ]
            if unassigned_test_artifacts:
                test_task_id = self._unique_task_id("prd-tests", used_task_ids)
                aggregate_criteria = (
                    list(acceptance_tests)
                    if acceptance_tests
                    else [
                        "Generated tests cover the PRD-required test artifacts."
                    ]
                )
                aggregate_test_task = PlannedTask(
                    id=test_task_id,
                    title="PRD acceptance tests",
                    description=(
                        "Write the required tests for PRD test artifacts that are "
                        "not domain-specific enough to attach to one implementation task."
                    ),
                    role="test_writer",
                    complexity=TaskComplexity.MEDIUM,
                    depends_on=list(implementation_dependency_ids),
                    acceptance_criteria=aggregate_criteria,
                    target_files=unassigned_test_artifacts,
                    required_artifacts=unassigned_test_artifacts,
                )
                tasks.append(aggregate_test_task)
                synthesized_test_writer_task_ids.append(test_task_id)
                artifact_updated_task_ids.append(test_task_id)

        tasks, supplemental_artifact_assignments = (
            self._assign_missing_prd_implementation_artifacts(
                tasks,
                source_implementation_artifacts,
            )
        )
        if supplemental_artifact_assignments:
            artifact_updated_task_ids.extend(
                item["task_id"] for item in supplemental_artifact_assignments
            )

        if not updated_task_ids and not artifact_updated_task_ids:
            record.outputs["task_graph_acceptance_enrichment"] = {
                "applied": False,
                "reason": "all_executable_tasks_already_have_prd_criteria_and_artifacts",
                "updated_task_ids": [],
            }
            return task_graph

        enrichment = {
            "applied": True,
            "reason": "filled_missing_task_acceptance_criteria_from_prd",
            "updated_task_ids": updated_task_ids,
        }
        if synthesized_test_writer_task_ids:
            enrichment["synthesized_test_writer_task_ids"] = (
                synthesized_test_writer_task_ids
            )
        if artifact_updated_task_ids:
            enrichment["artifact_updated_task_ids"] = self._unique_paths(
                artifact_updated_task_ids
            )
            if source_implementation_artifacts:
                enrichment["inherited_required_artifacts"] = (
                    source_implementation_artifacts
                )
            if source_test_artifacts:
                enrichment["inherited_test_artifacts"] = source_test_artifacts
        if supplemental_artifact_assignments:
            enrichment["supplemental_artifact_assignments"] = (
                supplemental_artifact_assignments
            )
        record.outputs["task_graph_acceptance_enrichment"] = enrichment
        return TaskGraph(
            goal=task_graph.goal,
            tasks=tasks,
            notes=[
                *task_graph.notes,
                "ACOS filled missing task acceptance_criteria from the PRD.",
            ],
        )

    @staticmethod
    def _unique_task_id(base: str, used_task_ids: set[str]) -> str:
        candidate = re.sub(r"[^a-zA-Z0-9_-]+", "-", base).strip("-") or "task"
        if candidate not in used_task_ids:
            return candidate
        index = 2
        while f"{candidate}-{index}" in used_task_ids:
            index += 1
        return f"{candidate}-{index}"

    @classmethod
    def _assign_missing_prd_implementation_artifacts(
        cls,
        tasks: list[PlannedTask],
        artifacts: list[str],
    ) -> tuple[list[PlannedTask], list[dict[str, str]]]:
        if not artifacts:
            return tasks, []
        updated_tasks = list(tasks)
        assignments: list[dict[str, str]] = []

        def implementation_indexes() -> list[int]:
            return [
                index
                for index, task in enumerate(updated_tasks)
                if task.role in cls.IMPLEMENTATION_TASK_ROLES
            ]

        for artifact in artifacts:
            target_owners = [
                index
                for index in implementation_indexes()
                if artifact
                in cls._valid_unique_planning_artifact_paths(
                    updated_tasks[index].target_files
                )
            ]
            required_owners = [
                index
                for index in implementation_indexes()
                if artifact
                in cls._valid_unique_planning_artifact_paths(
                    updated_tasks[index].required_artifacts
                )
            ]
            owner_index = (
                target_owners[0]
                if target_owners
                else required_owners[0]
                if required_owners
                else cls._best_prd_artifact_owner_task_index(updated_tasks, artifact)
            )
            if owner_index is None:
                continue
            task = updated_tasks[owner_index]
            target_files = cls._unique_paths([*task.target_files, artifact])
            required_artifacts = cls._unique_paths([*task.required_artifacts, artifact])
            if (
                target_files == task.target_files
                and required_artifacts == task.required_artifacts
            ):
                continue
            updated_tasks[owner_index] = task.model_copy(
                update={
                    "target_files": target_files,
                    "required_artifacts": required_artifacts,
                }
            )
            assignments.append({"task_id": task.id, "path": artifact})
        return updated_tasks, assignments

    @classmethod
    def _best_prd_artifact_owner_task_index(
        cls,
        tasks: list[PlannedTask],
        artifact: str,
    ) -> int | None:
        implementation_indexes = [
            index
            for index, task in enumerate(tasks)
            if task.role in cls.IMPLEMENTATION_TASK_ROLES
        ]
        if not implementation_indexes:
            return None
        artifact_tokens = cls._artifact_semantic_tokens(artifact)
        best_index: int | None = None
        best_score = 0
        if artifact_tokens:
            for index in implementation_indexes:
                task_tokens = cls._semantic_tokens(cls._task_semantic_text(tasks[index]))
                score = cls._semantic_overlap_score(artifact_tokens, task_tokens)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score > 0:
                return best_index

        category_tokens = cls._artifact_category_tokens(artifact)
        if category_tokens:
            category_matches: list[int] = []
            for index in implementation_indexes:
                task_tokens = cls._semantic_tokens(cls._task_semantic_text(tasks[index]))
                if category_tokens & task_tokens:
                    category_matches.append(index)
            if len(category_matches) == 1:
                return category_matches[0]
        if len(implementation_indexes) == 1:
            return implementation_indexes[0]
        return None

    @staticmethod
    def _artifact_category_tokens(path: str) -> set[str]:
        normalized = path.replace("\\", "/").lower()
        parts = {part for part in re.split(r"[./_-]+", normalized) if part}
        categories: set[str] = set()
        if parts & {"frontend", "client", "ui", "web"}:
            categories.update({"frontend", "client", "react"})
        if parts & {"backend", "server", "api"}:
            categories.update({"backend", "server", "api"})
        if parts & {"shared", "common", "types"}:
            categories.update({"shared", "type"})
        return categories

    def _normalize_project_setup_task_graph(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
    ) -> TaskGraph:
        tasks: list[PlannedTask] = []
        normalized_task_ids: list[str] = []
        project_setup_task_ids: list[str] = []
        role_normalized_task_ids: list[str] = []
        ignored_project_setup_artifacts: list[dict[str, Any]] = []
        for task in task_graph.tasks:
            if self._is_project_setup_task(task):
                declared_artifacts = self._unique_paths(
                    [*task.target_files, *task.required_artifacts]
                )
                ignored_artifacts = [
                    artifact
                    for artifact in declared_artifacts
                    if artifact not in self.PROJECT_SETUP_REQUIRED_ARTIFACTS
                ]
                if ignored_artifacts:
                    ignored_project_setup_artifacts.append(
                        {"task_id": task.id, "paths": ignored_artifacts}
                    )
                artifacts = list(self.PROJECT_SETUP_REQUIRED_ARTIFACTS)
                tasks.append(
                    task.model_copy(
                        update={
                            "role": "scaffold",
                            "target_files": artifacts,
                            "required_artifacts": artifacts,
                            "acceptance_criteria": self._meaningful_planning_items(
                                task.acceptance_criteria
                            )
                            or [
                                "Backend, frontend, shared, root manifest, README, gitignore, and env example scaffold files exist.",
                                "Project setup smoke test exists before test_writer tries to update it.",
                            ],
                        }
                    )
                )
                normalized_task_ids.append(task.id)
                project_setup_task_ids.append(task.id)
                continue
            if task.role == "architect":
                tasks.append(task.model_copy(update={"role": "implementer"}))
                normalized_task_ids.append(task.id)
                role_normalized_task_ids.append(task.id)
                continue
            tasks.append(task)

        if not normalized_task_ids:
            if "task_graph_normalization" in record.outputs:
                record.outputs.pop("task_graph_normalization", None)
                self.store.update(record)
            return task_graph

        normalized = TaskGraph(
            goal=task_graph.goal,
            tasks=tasks,
            notes=[
                *task_graph.notes,
                "ACOS normalized executable architect/project-setup tasks into deterministic scaffold/implementer tasks.",
            ],
        )
        record.outputs["task_graph"] = normalized.model_dump()
        record.outputs["task_graph_normalization"] = {
            "applied": True,
            "normalized_task_ids": normalized_task_ids,
            "project_setup_task_ids": project_setup_task_ids,
            "role_normalized_task_ids": role_normalized_task_ids,
            "required_artifacts": (
                list(self.PROJECT_SETUP_REQUIRED_ARTIFACTS)
                if project_setup_task_ids
                else []
            ),
            "ignored_project_setup_artifacts": ignored_project_setup_artifacts,
        }
        self.store.update(record)
        return normalized

    @classmethod
    def _is_project_setup_task(cls, task: PlannedTask | None) -> bool:
        if task is None:
            return False
        if task.role == "test_writer":
            return False
        identity = " ".join([task.id, task.title]).lower()
        description = task.description.lower()
        artifacts = cls._unique_paths([*task.target_files, *task.required_artifacts])
        artifact_text = " ".join(artifacts).lower()
        haystack = " ".join([identity, description, artifact_text])
        canonical_artifacts = set(cls.PROJECT_SETUP_REQUIRED_ARTIFACTS)
        declares_project_setup_artifacts = any(
            artifact in canonical_artifacts for artifact in artifacts
        )
        has_no_declared_artifacts = not artifacts
        strong_identity = any(
            keyword in identity
            for keyword in (
                "project-scaffold",
                "project scaffold",
                "project-setup",
                "project setup",
                "verify-project-setup",
            )
        )
        has_backend = "backend" in haystack
        has_frontend = "frontend" in haystack
        has_shared = "shared" in haystack
        structural_setup = (
            "monorepo" in haystack
            or "backend/frontend/shared" in haystack
            or (has_backend and has_frontend and has_shared)
        )
        if strong_identity and (
            has_no_declared_artifacts
            or declares_project_setup_artifacts
            or structural_setup
        ):
            return True
        return structural_setup and (
            has_no_declared_artifacts or declares_project_setup_artifacts
        )

    def _project_setup_artifacts_ready(
        self,
        record: JobRecord,
        task: PlannedTask | None,
    ) -> bool:
        return not self._missing_project_setup_artifacts(record, task)

    def _missing_project_setup_artifacts(
        self,
        record: JobRecord,
        task: PlannedTask | None,
    ) -> list[str]:
        if not self._is_project_setup_task(task):
            return []
        artifacts = self._unique_paths(
            [
                *self.PROJECT_SETUP_REQUIRED_ARTIFACTS,
            ]
        )
        root = self._workspace_root(record)
        return [
            artifact
            for artifact in artifacts
            if not artifact_path_exists(artifact, workspace_root=root)
        ]

    def _ensure_project_setup_ready_before_test_writer(
        self,
        record: JobRecord,
        task: PlannedTask | None,
    ) -> bool:
        missing = self._missing_project_setup_artifacts(record, task)
        if not missing:
            return True
        self._recover_record(
            record,
            error="required_artifacts_missing:project_setup_artifacts_missing",
            runtime_state={
                "required_artifacts": list(task.required_artifacts if task else missing),
                "target_files": list(task.target_files if task else missing),
                "missing_artifacts": missing,
                "failed_task_id": task.id if task is not None else "project-setup",
                "force_project_setup_scaffold": True,
            },
        )
        return False

    def _run_project_setup_scaffold(
        self,
        record: JobRecord,
        task: PlannedTask,
    ) -> ImplementationResult:
        apply_transition(record, JobStatus.IMPLEMENTING)
        preflight_evidence = self._project_setup_artifact_evidence(record, task)
        blocking_artifacts = [
            item["path"]
            for item in preflight_evidence
            if item["path_exists"] and not item["is_file"]
        ]
        if blocking_artifacts:
            missing = [
                item["path"] for item in preflight_evidence if not item["exists"]
            ]
            record.outputs["project_setup_scaffold"] = {
                "task_id": task.id,
                "artifact_evidence": preflight_evidence,
                "missing_artifacts": missing,
                "non_file_artifacts": blocking_artifacts,
            }
            self.store.update(record)
            self._recover_record(
                record,
                error="required_artifacts_missing:project_setup_artifact_blocked_by_non_file",
                runtime_state={
                    "required_artifacts": list(self.PROJECT_SETUP_REQUIRED_ARTIFACTS),
                    "target_files": list(self.PROJECT_SETUP_REQUIRED_ARTIFACTS),
                    "missing_artifacts": missing,
                    "non_file_artifacts": blocking_artifacts,
                    "failed_task_id": task.id,
                    "force_project_setup_scaffold": True,
                },
            )
            return ImplementationResult(
                status=ImplementationStatus.BLOCKED,
                summary="Project setup scaffold is blocked by non-file artifact paths.",
                changed_files=[],
                patches=[],
                risks=[
                    "Non-file paths must be removed or renamed before deterministic scaffold can write project setup artifacts."
                ],
            )
        workspace_root = self._workspace_root(record)
        missing_artifacts = [
            artifact
            for artifact in self.PROJECT_SETUP_REQUIRED_ARTIFACTS
            if not artifact_path_exists(artifact, workspace_root=workspace_root)
        ]
        patches = [
            FilePatch(
                path=artifact,
                operation="create",
                content=self._project_setup_file_content(artifact, record),
            )
            for artifact in missing_artifacts
        ]
        result = ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Created deterministic project setup scaffold.",
            changed_files=[patch.path for patch in patches],
            patches=patches,
        )
        self._apply_project_setup_scaffold_patches(record, patches)
        evidence = self._project_setup_artifact_evidence(record, task)
        record.outputs["project_setup_scaffold"] = {
            "task_id": task.id,
            "artifact_evidence": evidence,
            "missing_artifacts": [
                item["path"] for item in evidence if not item["exists"]
            ],
        }
        self.store.update(record)
        missing = [item["path"] for item in evidence if not item["exists"]]
        if missing:
            self._recover_record(
                record,
                error="required_artifacts_missing:project_setup_scaffold_incomplete",
                runtime_state={
                    "required_artifacts": list(task.required_artifacts),
                    "target_files": list(task.target_files),
                    "missing_artifacts": missing,
                    "failed_task_id": task.id,
                    "force_project_setup_scaffold": True,
                },
            )
        else:
            self._record_task_output(record, "implementation_tasks", task, result)
            self.recovery_executor.execute_until_ready(record)
            self._consume_completed_recovery_plan(record)
        return result

    def _apply_project_setup_scaffold_patches(
        self,
        record: JobRecord,
        patches: list[FilePatch],
    ) -> None:
        allowed = set(self.PROJECT_SETUP_REQUIRED_ARTIFACTS)
        for patch in patches:
            if patch.path not in allowed:
                raise QualityGateError(
                    f"policy_denied:unexpected_project_setup_scaffold_path:{patch.path}"
                )
            result = self.router.call(
                "repo_server.apply_patch",
                path=patch.path,
                content=patch.content,
                operation=patch.operation,
                new_path=patch.new_path,
                unified_diff=patch.unified_diff,
                base_sha256=patch.base_sha256,
                expected_old_content=patch.expected_old_content,
                executable=patch.executable,
            )
            event = self.audit.tool_event(
                role="orchestrator",
                tool_name="repo_server.apply_patch",
                input_payload={
                    "path": patch.path,
                    "operation": patch.operation,
                    "deterministic_scaffold": True,
                },
                output_payload=result.data,
                status="success" if result.ok else "failed",
            )
            record.audit_events.append(event)
            if not result.ok:
                raise RuntimeError(result.error or "project setup scaffold patch failed")
        self.store.update(record)

    def _project_setup_artifact_evidence(
        self,
        record: JobRecord,
        task: PlannedTask,
    ) -> list[dict[str, Any]]:
        root = self._workspace_root(record)
        artifacts = list(self.PROJECT_SETUP_REQUIRED_ARTIFACTS)
        return [
            {
                "path": artifact,
                "exists": artifact_path_exists(artifact, workspace_root=root),
                "path_exists": (root / artifact).exists(),
                "is_file": (root / artifact).is_file(),
                "size": (root / artifact).stat().st_size
                if (root / artifact).is_file()
                else 0,
            }
            for artifact in artifacts
        ]

    @staticmethod
    def _project_setup_file_content(path: str, record: JobRecord) -> str:
        title = record.spec.title or record.job_id
        contents = {
            "backend/main.py": (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI(title=\"ACOS generated app\")\n\n\n"
                "@app.get(\"/health\")\n"
                "def health() -> dict[str, str]:\n"
                "    return {\"status\": \"ok\"}\n"
            ),
            "backend/requirements.txt": "fastapi\nuvicorn\npytest\n",
            "backend/tests/test_project_setup.py": (
                "from pathlib import Path\n\n\n"
                "def test_project_setup_artifacts_exist() -> None:\n"
                "    root = Path(__file__).resolve().parents[2]\n"
                "    assert (root / \"backend\" / \"main.py\").exists()\n"
                "    assert (root / \"frontend\" / \"package.json\").exists()\n"
                "    assert (root / \"shared\" / \".gitkeep\").exists()\n"
            ),
            "frontend/package.json": (
                "{\n"
                f"  \"name\": \"{JobRunner._safe_package_name(title)}-frontend\",\n"
                "  \"private\": true,\n"
                "  \"version\": \"0.1.0\",\n"
                "  \"type\": \"module\",\n"
                "  \"scripts\": {\n"
                "    \"dev\": \"vite --host 0.0.0.0\",\n"
                "    \"build\": \"tsc -b && vite build\",\n"
                "    \"preview\": \"vite preview --host 0.0.0.0\"\n"
                "  },\n"
                "  \"dependencies\": {\n"
                "    \"@vitejs/plugin-react\": \"latest\",\n"
                "    \"typescript\": \"latest\",\n"
                "    \"vite\": \"latest\",\n"
                "    \"react\": \"latest\",\n"
                "    \"react-dom\": \"latest\"\n"
                "  },\n"
                "  \"devDependencies\": {}\n"
                "}\n"
            ),
            "frontend/vite.config.js": (
                "import react from '@vitejs/plugin-react'\n"
                "import { defineConfig } from 'vite'\n\n"
                "export default defineConfig({\n"
                "  plugins: [react()],\n"
                "})\n"
            ),
            "frontend/src/main.tsx": (
                "import React from 'react'\n"
                "import ReactDOM from 'react-dom/client'\n"
                "import App from './App'\n\n"
                "ReactDOM.createRoot(document.getElementById('root')!).render(\n"
                "  <React.StrictMode>\n"
                "    <App />\n"
                "  </React.StrictMode>,\n"
                ")\n"
            ),
            "frontend/src/App.tsx": (
                "function App() {\n"
                "  return <main>ACOS project scaffold is ready.</main>\n"
                "}\n\n"
                "export default App\n"
            ),
            "shared/.gitkeep": "",
            ".gitignore": (
                ".venv/\n"
                "__pycache__/\n"
                ".pytest_cache/\n"
                "node_modules/\n"
                "dist/\n"
                ".env\n"
            ),
            "package.json": (
                "{\n"
                f"  \"name\": \"{JobRunner._safe_package_name(title)}\",\n"
                "  \"private\": true,\n"
                "  \"version\": \"0.1.0\",\n"
                "  \"scripts\": {\n"
                "    \"dev\": \"npm --prefix frontend run dev\",\n"
                "    \"build\": \"npm --prefix frontend run build\"\n"
                "  }\n"
                "}\n"
            ),
            "README.md": f"# {title}\n\nACOS deterministic project scaffold.\n",
            ".env.example": "LOCAL_ORNITH_BASE_URL=http://127.0.0.1:8000/v1\n",
        }
        return contents.get(path, "")

    @staticmethod
    def _safe_package_name(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
        return normalized or "acos-generated-app"

    @staticmethod
    def _workspace_root(record: JobRecord) -> Path:
        return Path(record.spec.workspace_root or record.spec.repo_path).resolve()

    @staticmethod
    def _criteria_for_task_from_prd(
        task: PlannedTask,
        acceptance_tests: list[str],
        definition_of_done: list[str],
        implementation_index: int,
    ) -> list[str]:
        if implementation_index < len(acceptance_tests):
            return [acceptance_tests[implementation_index]]
        if definition_of_done:
            return [f"{task.title} satisfies definition of done: {definition_of_done[0]}"]
        return [f"{task.title} satisfies the PRD acceptance tests."]

    @staticmethod
    def _task_title_from_part(part: str) -> str:
        cleaned = " ".join(part.split())
        if len(cleaned) > 80:
            cleaned = f"{cleaned[:77].rstrip()}..."
        return cleaned or "Implement small part"

    @classmethod
    def _prd_artifacts_for_task(
        cls,
        task: PlannedTask,
        artifacts: list[str],
        *,
        task_count: int,
    ) -> list[str]:
        if not artifacts:
            return []
        if task_count <= 1:
            return list(artifacts)
        task_tokens = cls._semantic_tokens(cls._task_semantic_text(task))
        matched: list[str] = []
        for artifact in artifacts:
            artifact_tokens = cls._artifact_semantic_tokens(artifact)
            if task_tokens & artifact_tokens:
                matched.append(artifact)
        return matched

    @classmethod
    def _artifact_semantic_tokens(cls, path: str) -> set[str]:
        normalized = path.replace("\\", "/")
        stem_text = re.sub(r"[./_-]+", " ", normalized)
        tokens = cls._semantic_tokens(stem_text)
        return tokens - {
            "backend",
            "component",
            "components",
            "frontend",
            "index",
            "main",
            "page",
            "pages",
            "py",
            "server",
            "src",
            "tsx",
            "ts",
            "view",
            "views",
        }

    def _load_or_repair_task_graph_for_autonomy(
        self,
        record: JobRecord,
        prd: PRD,
    ) -> TaskGraph | None:
        task_graph = self._load_or_run_role(
            record,
            "planner",
            TaskGraph,
            "Create the implementation task graph",
            memory_key="task_graph",
        )
        task_graph = self._refine_task_graph_for_autonomy(record, prd, task_graph)
        task_graph = self._normalize_project_setup_task_graph(record, task_graph)
        ignored_project_setup_artifacts = self._ignored_project_setup_artifacts(record)
        validation = self._build_task_graph_validation(
            task_graph,
            prd=prd,
            require_acceptance_criteria=self._constraint_flag(
                record,
                "require_task_acceptance_criteria",
            ),
            require_executable_task_roles=self._constraint_flag(
                record,
                "require_completion_integrity",
            ),
            require_task_artifacts=self._constraint_flag(
                record,
                "require_task_artifacts",
            ),
            ignored_project_setup_artifacts=ignored_project_setup_artifacts,
        )
        record.outputs["task_graph_validation"] = validation
        self._record_task_graph_validation_attempt(
            record,
            attempt=0,
            action="initial",
            validation=validation,
        )
        if validation["valid"]:
            self._clear_planning_repair_constraints(record)
            self.store.update(record)
            return task_graph

        repair_attempts = self._constraint_int(
            record,
            "task_graph_validation_refinement_attempts",
            1,
        )
        self._reset_blocked_planning_resume(
            record,
            target_status=JobStatus.PLANNING,
            last_error_prefix="invalid_task_graph",
        )
        for attempt in range(1, repair_attempts + 1):
            task_graph = self._run_structured_role(
                record,
                "planner",
                TaskGraph,
                (
                    "Repair the implementation task graph before coding. "
                    "Return a valid graph with at least one implementer task, "
                    "known dependencies, no duplicate ids, no dependency cycles, "
                    "implementation task coverage for every PRD small_part, "
                    "testable acceptance_criteria on every executable task, "
                    "at least one test_writer task whenever the PRD has acceptance_tests "
                    "or test required_artifacts, "
                    "test_writer acceptance_criteria that directly cover every PRD "
                    "acceptance_test, "
                    "target_files on every test_writer task, "
                    "required_artifacts on every executable task, "
                    "depends_on from every test_writer task to the implementer/scaffold task it verifies, "
                    "test_writer dependencies that semantically match the behavior "
                    "being tested, "
                    "test_writer acceptance_criteria that are semantically covered "
                    "by the implementer/scaffold dependencies, "
                    "repo source target_files on implementer/scaffold tasks, "
                    "test target_files on test_writer tasks, "
                    "matching target_files and required_artifacts on every executable task, "
                    "dependencies that are satisfiable in the autonomous executor order, "
                    "and PRD required_artifacts assigned to their owning role target_files, "
                    "and only autonomous-executable task roles."
                ),
                logs=self._task_graph_validation_repair_logs(prd, validation),
            )
            self._write_memory_item(record, "planner", "task_graph", task_graph.model_dump_json())
            task_graph = self._refine_task_graph_for_autonomy(record, prd, task_graph)
            task_graph = self._normalize_project_setup_task_graph(record, task_graph)
            ignored_project_setup_artifacts = self._ignored_project_setup_artifacts(record)
            validation = self._build_task_graph_validation(
                task_graph,
                prd=prd,
                require_acceptance_criteria=self._constraint_flag(
                    record,
                    "require_task_acceptance_criteria",
                ),
                require_executable_task_roles=self._constraint_flag(
                    record,
                    "require_completion_integrity",
                ),
                require_task_artifacts=self._constraint_flag(
                    record,
                    "require_task_artifacts",
                ),
                ignored_project_setup_artifacts=ignored_project_setup_artifacts,
            )
            record.outputs["task_graph_validation"] = validation
            self._record_task_graph_validation_attempt(
                record,
                attempt=attempt,
                action="repair",
                validation=validation,
            )
            if validation["valid"]:
                self._clear_planning_repair_constraints(record)
                self.store.update(record)
                return task_graph

        self._recover_record(
            record,
            error="invalid_task_graph",
            runtime_state=self._task_graph_validation_recovery_state(
                record,
                validation,
            ),
        )
        self.store.update(record)
        return None

    @classmethod
    def _task_graph_validation_recovery_state(
        cls,
        record: JobRecord,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_state = dict(record.runtime_state)
        for key in (
            "task_graph_validation_errors",
            "uncovered_small_parts",
            "uncovered_acceptance_tests",
            *cls.TASK_GRAPH_VALIDATION_DETAIL_KEYS,
        ):
            runtime_state.pop(key, None)
        error_types = [
            str(item.get("type"))
            for item in validation.get("errors", [])
            if isinstance(item, dict) and str(item.get("type", "")).strip()
        ]
        if error_types:
            runtime_state["task_graph_validation_errors"] = list(
                dict.fromkeys(error_types)
            )
        for key in cls.TASK_GRAPH_VALIDATION_CONTEXT_KEYS:
            value = validation.get(key)
            if isinstance(value, list) and value:
                runtime_state[key] = value
        return runtime_state

    def _record_task_graph_validation_attempt(
        self,
        record: JobRecord,
        *,
        attempt: int,
        action: str,
        validation: dict[str, Any],
    ) -> None:
        attempts = record.outputs.setdefault("task_graph_validation_attempts", [])
        if not isinstance(attempts, list):
            attempts = []
            record.outputs["task_graph_validation_attempts"] = attempts
        attempt_record = {
            "attempt": attempt,
            "action": action,
            "valid": validation["valid"],
            "errors": list(validation["errors"]),
            "small_part_coverage": list(validation.get("small_part_coverage", [])),
            "uncovered_small_parts": list(validation.get("uncovered_small_parts", [])),
            "acceptance_test_coverage": list(
                validation.get("acceptance_test_coverage", [])
            ),
            "uncovered_acceptance_tests": list(
                validation.get("uncovered_acceptance_tests", [])
            ),
            "test_writer_acceptance_test_coverage": list(
                validation.get("test_writer_acceptance_test_coverage", [])
            ),
            "uncovered_test_writer_acceptance_tests": list(
                validation.get("uncovered_test_writer_acceptance_tests", [])
            ),
        }
        for key in self.TASK_GRAPH_VALIDATION_DETAIL_KEYS:
            attempt_record[key] = list(validation.get(key, []))
        attempts.append(attempt_record)

    def _validate_task_graph_for_autonomy(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
    ) -> bool:
        validation = self._build_task_graph_validation(
            task_graph,
            ignored_project_setup_artifacts=self._ignored_project_setup_artifacts(record),
            require_acceptance_criteria=self._constraint_flag(
                record,
                "require_task_acceptance_criteria",
            ),
            require_executable_task_roles=self._constraint_flag(
                record,
                "require_completion_integrity",
            ),
            require_task_artifacts=self._constraint_flag(
                record,
                "require_task_artifacts",
            ),
        )
        record.outputs["task_graph_validation"] = validation
        if validation["valid"]:
            self.store.update(record)
            return True
        self._recover_record(
            record,
            error="invalid_task_graph",
            runtime_state=self._task_graph_validation_recovery_state(
                record,
                validation,
            ),
        )
        self.store.update(record)
        return False

    @staticmethod
    def _ignored_project_setup_artifacts(record: JobRecord) -> list[dict[str, Any]]:
        normalization = record.outputs.get("task_graph_normalization")
        if not isinstance(normalization, dict):
            return []
        ignored = normalization.get("ignored_project_setup_artifacts")
        if not isinstance(ignored, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in ignored:
            if not isinstance(item, dict):
                continue
            raw_paths = item.get("paths")
            if not isinstance(raw_paths, list):
                continue
            paths = [str(path).strip() for path in raw_paths if str(path).strip()]
            if not paths:
                continue
            cleaned.append(
                {
                    "task_id": str(item.get("task_id") or "").strip(),
                    "paths": paths,
                }
            )
        return cleaned

    @staticmethod
    def _build_task_graph_validation(
        task_graph: TaskGraph,
        prd: PRD | None = None,
        require_acceptance_criteria: bool = False,
        require_executable_task_roles: bool = False,
        require_task_artifacts: bool = False,
        ignored_project_setup_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ids = [task.id for task in task_graph.tasks]
        duplicate_ids = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        id_set = set(ids)
        implementation_task_ids = [
            task.id for task in task_graph.tasks if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
        ]
        test_writer_task_ids = [
            task.id for task in task_graph.tasks if task.role in JobRunner.TEST_TASK_ROLES
        ]
        executable_roles = JobRunner.IMPLEMENTATION_TASK_ROLES | JobRunner.TEST_TASK_ROLES
        executable_task_ids = [
            task.id for task in task_graph.tasks if task.role in executable_roles
        ]
        small_parts = (
            JobRunner._meaningful_prd_items(prd.small_parts) if prd is not None else []
        )
        implementation_small_parts = [
            part for part in small_parts if not JobRunner._looks_like_test_work_item(part)
        ]
        test_focused_small_parts = [
            {"small_part_index": index, "small_part": part}
            for index, part in enumerate(small_parts, start=1)
            if JobRunner._looks_like_test_work_item(part)
        ]
        acceptance_tests = (
            JobRunner._meaningful_prd_items(prd.acceptance_tests)
            if prd is not None
            else []
        )
        implementation_tasks = [
            task for task in task_graph.tasks if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
        ]
        test_writer_tasks = [
            task for task in task_graph.tasks if task.role in JobRunner.TEST_TASK_ROLES
        ]
        executable_tasks = [
            task for task in task_graph.tasks if task.role in executable_roles
        ]
        prd_required_artifacts = (
            set(JobRunner._valid_unique_planning_artifact_paths(prd.required_artifacts))
            if prd is not None
            else set()
        )
        invalid_prd_required_artifacts = (
            JobRunner._invalid_planning_artifact_paths(prd.required_artifacts)
            if prd is not None
            else []
        )
        prd_test_required_artifacts = sorted(
            path for path in prd_required_artifacts if JobRunner._looks_like_test_path(path)
        )
        assigned_artifacts = set(
            JobRunner._valid_unique_planning_artifact_paths(
                [
                    path
                    for task in executable_tasks
                    for path in [*task.target_files, *task.required_artifacts]
                ]
            )
        )
        unassigned_required_artifacts = sorted(
            prd_required_artifacts - assigned_artifacts
        )
        target_artifacts_by_role: dict[str, set[str]] = {}
        for task in executable_tasks:
            target_artifacts_by_role.setdefault(task.role, set()).update(
                JobRunner._valid_unique_planning_artifact_paths(task.target_files)
            )
        unowned_required_artifacts: list[dict[str, Any]] = []
        for artifact in sorted(prd_required_artifacts):
            expected_roles = JobRunner._artifact_owner_roles(artifact)
            owned_targets = set().union(
                *[
                    target_artifacts_by_role.get(role, set())
                    for role in expected_roles
                ]
            )
            if artifact not in owned_targets:
                unowned_required_artifacts.append(
                    {
                        "path": artifact,
                        "expected_roles": sorted(expected_roles),
                    }
                )
        role_mismatched_target_files: list[dict[str, Any]] = []
        role_mismatched_required_artifacts: list[dict[str, Any]] = []
        required_artifacts_missing_target_files: list[dict[str, Any]] = []
        target_files_missing_required_artifacts: list[dict[str, Any]] = []
        for task in executable_tasks:
            task_target_files = set(
                JobRunner._valid_unique_planning_artifact_paths(task.target_files)
            )
            task_required_artifacts = set(
                JobRunner._valid_unique_planning_artifact_paths(
                    task.required_artifacts
                )
            )
            missing_target_files = sorted(task_required_artifacts - task_target_files)
            if missing_target_files:
                required_artifacts_missing_target_files.append(
                    {
                        "task_id": task.id,
                        "role": task.role,
                        "paths": missing_target_files,
                    }
                )
            missing_required_artifacts = sorted(task_target_files - task_required_artifacts)
            if missing_required_artifacts:
                target_files_missing_required_artifacts.append(
                    {
                        "task_id": task.id,
                        "role": task.role,
                        "paths": missing_required_artifacts,
                    }
                )
            for path in sorted(task_target_files):
                expected_roles = JobRunner._artifact_owner_roles(path)
                if task.role not in expected_roles:
                    role_mismatched_target_files.append(
                        {
                            "task_id": task.id,
                            "role": task.role,
                            "path": path,
                            "expected_roles": sorted(expected_roles),
                        }
                    )
            for path in sorted(task_required_artifacts):
                expected_roles = JobRunner._artifact_owner_roles(path)
                if task.role not in expected_roles:
                    role_mismatched_required_artifacts.append(
                        {
                            "task_id": task.id,
                            "role": task.role,
                            "path": path,
                            "expected_roles": sorted(expected_roles),
                        }
                    )
        unordered_target_file_owner_conflicts = (
            JobRunner._unordered_target_file_owner_conflicts(
                task_graph,
                executable_tasks,
            )
        )
        invalid_task_titles = [
            {
                "task_id": task.id,
                "role": task.role,
                "title": task.title,
            }
            for task in executable_tasks
            if JobRunner._looks_like_placeholder_prd_item(task.title)
        ]
        invalid_task_descriptions = [
            {
                "task_id": task.id,
                "role": task.role,
                "description": task.description,
            }
            for task in executable_tasks
            if JobRunner._looks_like_placeholder_prd_item(task.description)
        ]
        invalid_task_ids = [
            {
                "task_id": task.id,
                "role": task.role,
                "reason": JobRunner._invalid_task_id_reason(task.id),
            }
            for task in task_graph.tasks
            if JobRunner._invalid_task_id_reason(task.id)
        ]
        duplicate_task_acceptance_criteria = [
            {
                "task_id": task.id,
                "role": task.role,
                "duplicates": duplicates,
            }
            for task in executable_tasks
            if (
                duplicates := JobRunner._duplicate_planning_items(
                    JobRunner._meaningful_planning_items(task.acceptance_criteria)
                )
            )
        ]
        generic_task_acceptance_criteria = [
            {
                "task_id": task.id,
                "role": task.role,
                "acceptance_criteria": criterion,
            }
            for task in executable_tasks
            for criterion in JobRunner._meaningful_planning_items(
                task.acceptance_criteria
            )
            if JobRunner._looks_like_generic_task_acceptance_criterion(criterion)
        ]
        small_part_coverage = JobRunner._semantic_task_coverage(
            implementation_small_parts,
            implementation_tasks,
            item_key="small_part",
            index_key="small_part_index",
        )
        acceptance_test_coverage = JobRunner._semantic_task_coverage(
            acceptance_tests,
            implementation_tasks,
            item_key="acceptance_test",
            index_key="acceptance_test_index",
            allow_reuse=True,
            include_action_tokens=True,
        )
        test_writer_acceptance_test_coverage = JobRunner._semantic_task_coverage(
            acceptance_tests,
            test_writer_tasks,
            item_key="acceptance_test",
            index_key="acceptance_test_index",
            allow_reuse=True,
            include_action_tokens=True,
        )
        uncovered_small_parts = [
            item for item in small_part_coverage if not item["covered"]
        ]
        uncovered_acceptance_tests = [
            item for item in acceptance_test_coverage if not item["covered"]
        ]
        uncovered_test_writer_acceptance_tests = [
            item for item in test_writer_acceptance_test_coverage if not item["covered"]
        ]
        unknown_dependencies = [
            {"task_id": task.id, "dependency": dependency}
            for task in task_graph.tasks
            for dependency in task.depends_on
            if dependency not in id_set
        ]
        cycle = JobRunner._find_task_graph_cycle(task_graph)
        executor_order_dependency_violations = (
            JobRunner._executor_order_dependency_violations(task_graph)
            if not cycle
            else []
        )
        implementation_task_id_set = set(implementation_task_ids)
        test_writer_missing_implementation_dependencies = [
            {
                "task_id": task.id,
                "depends_on": list(task.depends_on),
                "required_dependency_roles": sorted(JobRunner.IMPLEMENTATION_TASK_ROLES),
            }
            for task in task_graph.tasks
            if task.role in JobRunner.TEST_TASK_ROLES
            and implementation_task_ids
            and not any(
                dependency in implementation_task_id_set
                for dependency in task.depends_on
            )
        ]
        test_writer_dependency_semantic_mismatches = (
            JobRunner._test_writer_dependency_semantic_mismatches(task_graph)
        )
        test_writer_acceptance_dependency_mismatches = (
            JobRunner._test_writer_acceptance_dependency_mismatches(task_graph)
        )
        cleaned_ignored_project_setup_artifacts: list[dict[str, Any]] = []
        for item in ignored_project_setup_artifacts or []:
            if not isinstance(item, dict):
                continue
            raw_paths = item.get("paths")
            if not isinstance(raw_paths, list):
                continue
            paths = [str(path).strip() for path in raw_paths if str(path).strip()]
            if paths:
                cleaned_ignored_project_setup_artifacts.append(
                    {
                        "task_id": str(item.get("task_id") or "").strip(),
                        "paths": paths,
                    }
                )
        ignored_project_setup_artifacts = cleaned_ignored_project_setup_artifacts
        project_setup_scaffold_covers_test_artifacts = (
            bool(prd_test_required_artifacts)
            and all(
                path in JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS
                for path in prd_test_required_artifacts
            )
            and any(
                task.role == "scaffold"
                and JobRunner._is_project_setup_task(task)
                and set(prd_test_required_artifacts).issubset(
                    valid_artifact_paths(task.target_files)
                )
                for task in task_graph.tasks
            )
        )
        test_writer_required = bool(
            acceptance_tests or test_focused_small_parts or prd_test_required_artifacts
        )
        missing_test_writer_tasks = (
            require_task_artifacts
            and test_writer_required
            and not test_writer_task_ids
            and not project_setup_scaffold_covers_test_artifacts
        )
        missing_test_writer_task_requirements = (
            [
                {
                    "acceptance_tests": bool(acceptance_tests),
                    "test_focused_small_parts": bool(test_focused_small_parts),
                    "prd_test_required_artifacts": prd_test_required_artifacts,
                }
            ]
            if missing_test_writer_tasks
            else []
        )
        errors: list[dict[str, Any]] = []
        if not task_graph.tasks:
            errors.append({"type": "empty_task_graph"})
        elif not implementation_task_ids:
            errors.append({"type": "missing_implementation_tasks"})
        elif implementation_small_parts and len(implementation_task_ids) < len(implementation_small_parts):
            errors.append(
                {
                    "type": "undercovered_small_parts",
                    "small_part_count": len(small_parts),
                    "implementation_small_part_count": len(implementation_small_parts),
                    "implementation_task_count": len(implementation_task_ids),
                    "uncovered_small_parts": uncovered_small_parts,
                }
            )
        elif uncovered_small_parts:
            errors.append(
                {
                    "type": "semantic_small_part_mismatch",
                    "small_part_count": len(small_parts),
                    "implementation_small_part_count": len(implementation_small_parts),
                    "implementation_task_count": len(implementation_task_ids),
                    "uncovered_small_parts": uncovered_small_parts,
                }
            )
        if acceptance_tests and uncovered_acceptance_tests:
            errors.append(
                {
                    "type": "semantic_acceptance_test_mismatch",
                    "acceptance_test_count": len(acceptance_tests),
                    "implementation_task_count": len(implementation_task_ids),
                    "uncovered_acceptance_tests": uncovered_acceptance_tests,
                }
            )
        if (
            require_acceptance_criteria
            and acceptance_tests
            and test_writer_task_ids
            and uncovered_test_writer_acceptance_tests
        ):
            errors.append(
                {
                    "type": "semantic_test_writer_acceptance_mismatch",
                    "acceptance_test_count": len(acceptance_tests),
                    "test_writer_task_count": len(test_writer_task_ids),
                    "uncovered_test_writer_acceptance_tests": (
                        uncovered_test_writer_acceptance_tests
                    ),
                }
            )
        if invalid_prd_required_artifacts:
            errors.append(
                {
                    "type": "invalid_prd_required_artifacts",
                    "paths": invalid_prd_required_artifacts,
                }
            )
        if unassigned_required_artifacts:
            errors.append(
                {
                    "type": "unassigned_required_artifacts",
                    "paths": unassigned_required_artifacts,
                }
            )
        if require_task_artifacts and unowned_required_artifacts:
            errors.append(
                {
                    "type": "unowned_required_artifacts",
                    "items": unowned_required_artifacts,
                }
            )
        if require_task_artifacts and role_mismatched_target_files:
            errors.append(
                {
                    "type": "role_mismatched_target_files",
                    "items": role_mismatched_target_files,
                }
            )
        if require_task_artifacts and role_mismatched_required_artifacts:
            errors.append(
                {
                    "type": "role_mismatched_required_artifacts",
                    "items": role_mismatched_required_artifacts,
                }
            )
        if require_task_artifacts and required_artifacts_missing_target_files:
            errors.append(
                {
                    "type": "required_artifacts_missing_target_files",
                    "items": required_artifacts_missing_target_files,
                }
            )
        if require_task_artifacts and target_files_missing_required_artifacts:
            errors.append(
                {
                    "type": "target_files_missing_required_artifacts",
                    "items": target_files_missing_required_artifacts,
                }
            )
        if require_task_artifacts and unordered_target_file_owner_conflicts:
            errors.append(
                {
                    "type": "unordered_target_file_owner_conflicts",
                    "items": unordered_target_file_owner_conflicts,
                }
            )
        strict_executable_task_validation = (
            require_acceptance_criteria
            or require_task_artifacts
            or require_executable_task_roles
        )
        if strict_executable_task_validation and invalid_task_titles:
            errors.append(
                {
                    "type": "invalid_task_titles",
                    "items": invalid_task_titles,
                }
            )
        if strict_executable_task_validation and invalid_task_descriptions:
            errors.append(
                {
                    "type": "invalid_task_descriptions",
                    "items": invalid_task_descriptions,
                }
            )
        if strict_executable_task_validation and invalid_task_ids:
            errors.append(
                {
                    "type": "invalid_task_ids",
                    "items": invalid_task_ids,
                }
            )
        if duplicate_ids:
            errors.append({"type": "duplicate_task_ids", "task_ids": duplicate_ids})
        if unknown_dependencies:
            errors.append({"type": "unknown_dependencies", "items": unknown_dependencies})
        if executor_order_dependency_violations:
            errors.append(
                {
                    "type": "executor_order_dependency_violations",
                    "items": executor_order_dependency_violations,
                }
            )
        if cycle:
            errors.append({"type": "dependency_cycle", "task_ids": cycle})
        tasks_missing_acceptance_criteria = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
            and not JobRunner._meaningful_task_acceptance_criteria(
                task.acceptance_criteria
            )
        ]
        if require_acceptance_criteria and tasks_missing_acceptance_criteria:
            errors.append(
                {
                    "type": "missing_acceptance_criteria",
                    "task_ids": tasks_missing_acceptance_criteria,
                }
            )
        test_writer_tasks_missing_acceptance_criteria = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.TEST_TASK_ROLES
            and not JobRunner._meaningful_task_acceptance_criteria(
                task.acceptance_criteria
            )
        ]
        if require_acceptance_criteria and test_writer_tasks_missing_acceptance_criteria:
            errors.append(
                {
                    "type": "missing_test_writer_acceptance_criteria",
                    "task_ids": test_writer_tasks_missing_acceptance_criteria,
                }
            )
        if require_acceptance_criteria and duplicate_task_acceptance_criteria:
            errors.append(
                {
                    "type": "duplicate_task_acceptance_criteria",
                    "items": duplicate_task_acceptance_criteria,
                }
            )
        if require_acceptance_criteria and generic_task_acceptance_criteria:
            errors.append(
                {
                    "type": "generic_task_acceptance_criteria",
                    "items": generic_task_acceptance_criteria,
                }
            )
        tasks_missing_artifacts = [
            task.id
            for task in task_graph.tasks
            if task.role in executable_roles
            and not JobRunner._valid_unique_planning_artifact_paths(
                [*task.target_files, *task.required_artifacts]
            )
        ]
        if require_task_artifacts and tasks_missing_artifacts:
            errors.append(
                {
                    "type": "missing_task_artifacts",
                    "task_ids": tasks_missing_artifacts,
                }
            )
        executable_tasks_missing_required_artifacts = [
            task.id
            for task in task_graph.tasks
            if task.role in executable_roles
            and not JobRunner._valid_unique_planning_artifact_paths(
                task.required_artifacts
            )
        ]
        if require_task_artifacts and executable_tasks_missing_required_artifacts:
            errors.append(
                {
                    "type": "missing_required_artifacts",
                    "task_ids": executable_tasks_missing_required_artifacts,
                }
            )
        test_writer_tasks_missing_target_files = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.TEST_TASK_ROLES
            and not JobRunner._valid_unique_planning_artifact_paths(task.target_files)
        ]
        if require_task_artifacts and test_writer_tasks_missing_target_files:
            errors.append(
                {
                    "type": "missing_test_writer_target_files",
                    "task_ids": test_writer_tasks_missing_target_files,
                }
            )
        if missing_test_writer_tasks:
            errors.append(
                {
                    "type": "missing_test_writer_tasks",
                    "required_by": missing_test_writer_task_requirements[0],
                }
            )
        implementation_tasks_missing_target_files = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
            and not JobRunner._valid_unique_planning_artifact_paths(task.target_files)
        ]
        if require_task_artifacts and implementation_tasks_missing_target_files:
            errors.append(
                {
                    "type": "missing_implementation_target_files",
                    "task_ids": implementation_tasks_missing_target_files,
                }
            )
        if require_task_artifacts and test_writer_missing_implementation_dependencies:
            errors.append(
                {
                    "type": "test_writer_missing_implementation_dependency",
                    "items": test_writer_missing_implementation_dependencies,
                }
            )
        if require_task_artifacts and test_writer_dependency_semantic_mismatches:
            errors.append(
                {
                    "type": "test_writer_dependency_semantic_mismatch",
                    "items": test_writer_dependency_semantic_mismatches,
                }
            )
        if require_task_artifacts and test_writer_acceptance_dependency_mismatches:
            errors.append(
                {
                    "type": "test_writer_acceptance_dependency_mismatch",
                    "items": test_writer_acceptance_dependency_mismatches,
                }
            )
        implementation_tasks_missing_artifacts = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
            and not JobRunner._valid_unique_planning_artifact_paths(
                [*task.target_files, *task.required_artifacts]
            )
        ]
        invalid_task_artifacts = []
        for task in task_graph.tasks:
            if task.role not in executable_roles:
                continue
            invalid_paths = JobRunner._invalid_planning_artifact_paths(
                [*task.target_files, *task.required_artifacts]
            )
            if invalid_paths:
                invalid_task_artifacts.append(
                    {"task_id": task.id, "paths": invalid_paths}
                )
        if require_task_artifacts and invalid_task_artifacts:
            errors.append(
                {
                    "type": "invalid_task_artifacts",
                    "items": invalid_task_artifacts,
                }
            )
        if require_task_artifacts and ignored_project_setup_artifacts:
            errors.append(
                {
                    "type": "ignored_project_setup_artifacts",
                    "items": ignored_project_setup_artifacts,
                }
            )
        unsupported_task_roles = [
            {"task_id": task.id, "role": task.role}
            for task in task_graph.tasks
            if task.role not in executable_roles
        ]
        if strict_executable_task_validation and unsupported_task_roles:
            errors.append(
                {
                    "type": "unsupported_autonomous_task_roles",
                    "items": unsupported_task_roles,
                    "allowed_roles": sorted(executable_roles),
                }
            )
        return {
            "valid": not errors,
            "task_count": len(task_graph.tasks),
            "implementation_task_count": len(implementation_task_ids),
            "test_writer_task_count": len(test_writer_task_ids),
            "executable_task_count": len(executable_task_ids),
            "implementation_task_acceptance_criteria_count": (
                len(implementation_task_ids) - len(tasks_missing_acceptance_criteria)
            ),
            "test_writer_task_acceptance_criteria_count": (
                len(test_writer_task_ids)
                - len(test_writer_tasks_missing_acceptance_criteria)
            ),
            "executable_task_acceptance_criteria_count": (
                len(executable_task_ids)
                - len(tasks_missing_acceptance_criteria)
                - len(test_writer_tasks_missing_acceptance_criteria)
            ),
            "require_acceptance_criteria": require_acceptance_criteria,
            "require_executable_task_roles": require_executable_task_roles,
            "require_task_artifacts": require_task_artifacts,
            "implementation_task_artifact_count": (
                len(implementation_task_ids) - len(implementation_tasks_missing_artifacts)
            ),
            "executable_task_artifact_count": (
                len(executable_task_ids) - len(tasks_missing_artifacts)
            ),
            "invalid_task_artifact_count": len(invalid_task_artifacts),
            "invalid_task_artifacts": invalid_task_artifacts,
            "ignored_project_setup_artifact_count": len(ignored_project_setup_artifacts),
            "ignored_project_setup_artifacts": ignored_project_setup_artifacts,
            "prd_required_artifact_count": len(prd_required_artifacts),
            "assigned_required_artifact_count": len(
                prd_required_artifacts & assigned_artifacts
            ),
            "unassigned_required_artifacts": unassigned_required_artifacts,
            "invalid_prd_required_artifacts": invalid_prd_required_artifacts,
            "unowned_required_artifacts": unowned_required_artifacts,
            "role_mismatched_target_files": role_mismatched_target_files,
            "role_mismatched_required_artifacts": role_mismatched_required_artifacts,
            "required_artifacts_missing_target_files": (
                required_artifacts_missing_target_files
            ),
            "target_files_missing_required_artifacts": (
                target_files_missing_required_artifacts
            ),
            "unordered_target_file_owner_conflicts": (
                unordered_target_file_owner_conflicts
            ),
            "duplicate_task_ids": duplicate_ids,
            "unknown_dependencies": unknown_dependencies,
            "dependency_cycle_task_ids": cycle,
            "prd_test_required_artifacts": prd_test_required_artifacts,
            "missing_test_writer_tasks": missing_test_writer_tasks,
            "missing_test_writer_task_requirements": (
                missing_test_writer_task_requirements
            ),
            "project_setup_scaffold_covers_test_artifacts": (
                project_setup_scaffold_covers_test_artifacts
            ),
            "executable_tasks_missing_required_artifacts": (
                executable_tasks_missing_required_artifacts
            ),
            "test_writer_tasks_missing_acceptance_criteria": (
                test_writer_tasks_missing_acceptance_criteria
            ),
            "duplicate_task_acceptance_criteria": (
                duplicate_task_acceptance_criteria
            ),
            "generic_task_acceptance_criteria": generic_task_acceptance_criteria,
            "implementation_tasks_missing_target_files": (
                implementation_tasks_missing_target_files
            ),
            "test_writer_tasks_missing_target_files": (
                test_writer_tasks_missing_target_files
            ),
            "test_writer_missing_implementation_dependencies": (
                test_writer_missing_implementation_dependencies
            ),
            "test_writer_dependency_semantic_mismatches": (
                test_writer_dependency_semantic_mismatches
            ),
            "test_writer_acceptance_dependency_mismatches": (
                test_writer_acceptance_dependency_mismatches
            ),
            "executor_order_dependency_violations": (
                executor_order_dependency_violations
            ),
            "invalid_task_titles": invalid_task_titles,
            "invalid_task_descriptions": invalid_task_descriptions,
            "invalid_task_ids": invalid_task_ids,
            "unsupported_task_role_count": len(unsupported_task_roles),
            "unsupported_task_roles": unsupported_task_roles,
            "small_part_count": len(small_parts),
            "implementation_small_part_count": len(implementation_small_parts),
            "test_focused_small_parts": test_focused_small_parts,
            "small_part_coverage": small_part_coverage,
            "uncovered_small_parts": uncovered_small_parts,
            "acceptance_test_count": len(acceptance_tests),
            "acceptance_test_coverage": acceptance_test_coverage,
            "uncovered_acceptance_tests": uncovered_acceptance_tests,
            "test_writer_acceptance_test_coverage": test_writer_acceptance_test_coverage,
            "uncovered_test_writer_acceptance_tests": (
                uncovered_test_writer_acceptance_tests
            ),
            "errors": errors,
        }

    @staticmethod
    def _unordered_target_file_owner_conflicts(
        task_graph: TaskGraph,
        executable_tasks: list[PlannedTask],
    ) -> list[dict[str, Any]]:
        target_owners: dict[str, list[PlannedTask]] = {}
        for task in executable_tasks:
            for path in JobRunner._valid_unique_planning_artifact_paths(
                task.target_files
            ):
                target_owners.setdefault(path, []).append(task)

        conflicts: list[dict[str, Any]] = []
        for path, owners in sorted(target_owners.items()):
            if len(owners) < 2:
                continue
            unordered_pairs: list[dict[str, str]] = []
            for left_index, left in enumerate(owners):
                for right in owners[left_index + 1 :]:
                    if left.id == right.id:
                        continue
                    if (
                        JobRunner._task_depends_on(task_graph, left.id, right.id)
                        or JobRunner._task_depends_on(task_graph, right.id, left.id)
                    ):
                        continue
                    unordered_pairs.append(
                        {"first_task_id": left.id, "second_task_id": right.id}
                    )
            if unordered_pairs:
                conflicts.append(
                    {
                        "path": path,
                        "task_ids": [task.id for task in owners],
                        "unordered_task_pairs": unordered_pairs,
                    }
                )
        return conflicts

    @staticmethod
    def _task_depends_on(
        task_graph: TaskGraph,
        task_id: str,
        dependency_id: str,
    ) -> bool:
        task_by_id = {task.id: task for task in task_graph.tasks}
        visited: set[str] = set()

        def visit(current_id: str) -> bool:
            if current_id in visited:
                return False
            visited.add(current_id)
            current = task_by_id.get(current_id)
            if current is None:
                return False
            for dependency in current.depends_on:
                if dependency == dependency_id:
                    return True
                if dependency in task_by_id and visit(dependency):
                    return True
            return False

        return visit(task_id)

    @staticmethod
    def _invalid_task_id_reason(task_id: str) -> str | None:
        raw = str(task_id)
        value = raw.strip()
        if JobRunner._looks_like_placeholder_prd_item(value):
            return "placeholder"
        if raw != value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            return "unsafe_task_id_format"
        return None

    @staticmethod
    def _test_writer_dependency_semantic_mismatches(
        task_graph: TaskGraph,
    ) -> list[dict[str, Any]]:
        task_by_id = {task.id: task for task in task_graph.tasks}
        mismatches: list[dict[str, Any]] = []
        for task in task_graph.tasks:
            if task.role not in JobRunner.TEST_TASK_ROLES:
                continue
            implementation_dependencies = [
                task_by_id[dependency_id]
                for dependency_id in task.depends_on
                if dependency_id in task_by_id
                and task_by_id[dependency_id].role in JobRunner.IMPLEMENTATION_TASK_ROLES
            ]
            if not implementation_dependencies:
                continue
            task_tokens = JobRunner._semantic_tokens(JobRunner._task_semantic_text(task))
            if not task_tokens:
                continue
            required_score = JobRunner._semantic_overlap_required(task_tokens)
            anchor_tokens = JobRunner._semantic_anchor_tokens(task_tokens)
            matching_dependencies: list[str] = []
            combined_dependency_tokens: set[str] = set()
            for dependency in implementation_dependencies:
                dependency_tokens = JobRunner._semantic_tokens(
                    JobRunner._task_semantic_text(dependency)
                )
                combined_dependency_tokens.update(dependency_tokens)
                score = JobRunner._semantic_overlap_score(
                    task_tokens,
                    dependency_tokens,
                )
                if (
                    score >= required_score
                    and JobRunner._semantic_anchor_satisfied(
                        anchor_tokens,
                        dependency_tokens,
                    )
                ):
                    matching_dependencies.append(dependency.id)
            if not matching_dependencies and combined_dependency_tokens:
                combined_score = JobRunner._semantic_overlap_score(
                    task_tokens,
                    combined_dependency_tokens,
                )
                if (
                    combined_score >= required_score
                    and JobRunner._semantic_anchor_satisfied(
                        anchor_tokens,
                        combined_dependency_tokens,
                    )
                ):
                    matching_dependencies = [
                        dependency.id for dependency in implementation_dependencies
                    ]
            if not matching_dependencies:
                mismatches.append(
                    {
                        "task_id": task.id,
                        "depends_on": [
                            dependency.id for dependency in implementation_dependencies
                        ],
                        "required_dependency_roles": sorted(
                            JobRunner.IMPLEMENTATION_TASK_ROLES
                        ),
                    }
                )
        return mismatches

    @staticmethod
    def _test_writer_acceptance_dependency_mismatches(
        task_graph: TaskGraph,
    ) -> list[dict[str, Any]]:
        task_by_id = {task.id: task for task in task_graph.tasks}
        mismatches: list[dict[str, Any]] = []
        for task in task_graph.tasks:
            if task.role not in JobRunner.TEST_TASK_ROLES:
                continue
            implementation_dependencies = [
                task_by_id[dependency_id]
                for dependency_id in task.depends_on
                if dependency_id in task_by_id
                and task_by_id[dependency_id].role in JobRunner.IMPLEMENTATION_TASK_ROLES
            ]
            if not implementation_dependencies:
                continue
            dependency_tokens = [
                JobRunner._semantic_tokens(
                    JobRunner._task_semantic_text(dependency),
                    include_action_tokens=True,
                )
                for dependency in implementation_dependencies
            ]
            uncovered_criteria: list[dict[str, Any]] = []
            for index, criterion in enumerate(
                JobRunner._meaningful_planning_items(task.acceptance_criteria),
                start=1,
            ):
                criterion_tokens = JobRunner._test_writer_acceptance_dependency_tokens(
                    criterion
                )
                if not criterion_tokens:
                    continue
                required_score = JobRunner._semantic_overlap_required(criterion_tokens)
                anchor_tokens = JobRunner._semantic_anchor_tokens(criterion_tokens)
                covered = any(
                    JobRunner._semantic_overlap_score(
                        criterion_tokens,
                        candidate_tokens,
                    )
                    >= required_score
                    and JobRunner._semantic_anchor_satisfied(
                        anchor_tokens,
                        candidate_tokens,
                    )
                    for candidate_tokens in dependency_tokens
                )
                if not covered:
                    uncovered_criteria.append(
                        {
                            "acceptance_criteria_index": index,
                            "acceptance_criteria": criterion,
                            "covered": False,
                        }
                    )
            if uncovered_criteria:
                mismatches.append(
                    {
                        "task_id": task.id,
                        "depends_on": [
                            dependency.id for dependency in implementation_dependencies
                        ],
                        "required_dependency_roles": sorted(
                            JobRunner.IMPLEMENTATION_TASK_ROLES
                        ),
                        "uncovered_acceptance_criteria": uncovered_criteria,
                    }
                )
        return mismatches

    @staticmethod
    def _test_writer_acceptance_dependency_tokens(criterion: str) -> set[str]:
        generic_test_tokens = {
            "artifact",
            "artifacts",
            "by",
            "cover",
            "covered",
            "coverage",
            "exist",
            "exists",
            "file",
            "files",
            "generated",
            "pass",
            "passe",
            "passed",
            "passing",
            "regression",
            "smoke",
        }
        return (
            JobRunner._semantic_tokens(criterion, include_action_tokens=True)
            - generic_test_tokens
        )

    @staticmethod
    def _executor_order_dependency_violations(
        task_graph: TaskGraph,
    ) -> list[dict[str, Any]]:
        task_by_id = {task.id: task for task in task_graph.tasks}
        implementation_tasks = JobRunner._order_tasks_by_dependencies(
            [
                task
                for task in task_graph.tasks
                if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
            ]
        )
        implementation_tasks = [
            task
            for _index, task in sorted(
                enumerate(implementation_tasks),
                key=lambda item: (
                    0 if JobRunner._is_project_setup_task(item[1]) else 1,
                    item[0],
                ),
            )
        ]
        pending_test_tasks = JobRunner._order_tasks_by_dependencies(
            [
                task
                for task in task_graph.tasks
                if task.role in JobRunner.TEST_TASK_ROLES
            ]
        )
        completed_task_ids: set[str] = set()
        violations: list[dict[str, Any]] = []

        def known_unmet_dependencies(task: PlannedTask) -> list[str]:
            return [
                dependency
                for dependency in task.depends_on
                if dependency in task_by_id and dependency not in completed_task_ids
            ]

        def append_violation(task: PlannedTask, phase: str, unmet: list[str]) -> None:
            violations.append(
                {
                    "task_id": task.id,
                    "role": task.role,
                    "executor_phase": phase,
                    "unmet_dependencies": unmet,
                    "dependency_roles": [
                        {
                            "task_id": dependency,
                            "role": task_by_id[dependency].role,
                        }
                        for dependency in unmet
                    ],
                }
            )

        def complete_ready_tests() -> None:
            while True:
                ready_tasks: list[PlannedTask] = []
                for task in list(pending_test_tasks):
                    local_dependencies = [
                        dependency
                        for dependency in task.depends_on
                        if any(
                            dependency == pending_task.id
                            for pending_task in pending_test_tasks
                        )
                        or dependency not in completed_task_ids
                    ]
                    if not local_dependencies or all(
                        dependency in completed_task_ids
                        for dependency in task.depends_on
                    ):
                        ready_tasks.append(task)
                        pending_test_tasks.remove(task)
                if not ready_tasks:
                    return
                completed_task_ids.update(task.id for task in ready_tasks)

        for task in implementation_tasks:
            unmet = known_unmet_dependencies(task)
            if unmet:
                append_violation(task, "implementation", unmet)
                continue
            completed_task_ids.add(task.id)
            complete_ready_tests()

        for task in pending_test_tasks:
            unmet = known_unmet_dependencies(task)
            if unmet:
                append_violation(task, "test_writer", unmet)

        return violations

    @staticmethod
    def _artifact_owner_roles(path: str) -> set[str]:
        if JobRunner._looks_like_test_path(path):
            if path in JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS:
                return {"scaffold", "test_writer"}
            return set(JobRunner.TEST_TASK_ROLES)
        if path in JobRunner.PROJECT_SETUP_REQUIRED_ARTIFACTS:
            return set(JobRunner.IMPLEMENTATION_TASK_ROLES)
        return {"implementer"}

    @classmethod
    def _semantic_item_coverage(
        cls,
        items: list[str],
        candidates: list[str],
        *,
        item_key: str,
        index_key: str,
        candidate_key: str,
        candidate_index_key: str,
    ) -> list[dict[str, Any]]:
        remaining_candidates = list(enumerate(candidates, start=1))
        coverage: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            item_tokens = cls._semantic_tokens(item)
            if not item_tokens:
                fallback = candidates[index - 1] if index <= len(candidates) else None
                coverage.append(
                    {
                        index_key: index,
                        item_key: item,
                        candidate_index_key: index if fallback is not None else None,
                        candidate_key: fallback,
                        "covered": fallback is not None,
                    }
                )
                if fallback is not None:
                    remaining_candidates = [
                        candidate
                        for candidate in remaining_candidates
                        if candidate[0] != index
                    ]
                continue

            required_score = cls._semantic_overlap_required(item_tokens)
            anchor_tokens = cls._semantic_anchor_tokens(item_tokens)
            best_candidate: tuple[int, str] | None = None
            best_score = 0
            for candidate in remaining_candidates:
                candidate_tokens = cls._semantic_tokens(candidate[1])
                score = cls._semantic_overlap_score(
                    item_tokens,
                    candidate_tokens,
                )
                if (
                    score >= required_score
                    and cls._semantic_anchor_satisfied(anchor_tokens, candidate_tokens)
                    and score > best_score
                ):
                    best_score = score
                    best_candidate = candidate

            covered = best_candidate is not None
            coverage.append(
                {
                    index_key: index,
                    item_key: item,
                    candidate_index_key: best_candidate[0] if covered and best_candidate else None,
                    candidate_key: best_candidate[1] if covered and best_candidate else None,
                    "covered": covered,
                }
            )
            if covered and best_candidate is not None:
                remaining_candidates.remove(best_candidate)
        return coverage

    @classmethod
    def _semantic_task_coverage(
        cls,
        items: list[str],
        tasks: list[PlannedTask],
        *,
        item_key: str,
        index_key: str,
        allow_reuse: bool = False,
        include_action_tokens: bool = False,
    ) -> list[dict[str, Any]]:
        remaining_tasks = list(tasks)
        coverage: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            item_tokens = cls._semantic_tokens(
                item,
                include_action_tokens=include_action_tokens,
            )
            if not item_tokens:
                fallback_task = tasks[index - 1] if index <= len(tasks) else None
                coverage.append(
                    {
                        index_key: index,
                        item_key: item,
                        "task_id": fallback_task.id if fallback_task is not None else None,
                        "covered": fallback_task is not None,
                    }
                )
                continue
            required_score = cls._semantic_overlap_required(item_tokens)
            anchor_tokens = cls._semantic_anchor_tokens(item_tokens)
            best_task: PlannedTask | None = None
            best_score = 0
            for task in remaining_tasks:
                task_tokens = cls._semantic_tokens(
                    cls._task_semantic_text(task),
                    include_action_tokens=include_action_tokens,
                )
                score = cls._semantic_overlap_score(
                    item_tokens,
                    task_tokens,
                )
                if (
                    score >= required_score
                    and cls._semantic_anchor_satisfied(anchor_tokens, task_tokens)
                    and score > best_score
                ):
                    best_score = score
                    best_task = task
            covered = best_task is not None
            coverage.append(
                {
                    index_key: index,
                    item_key: item,
                    "task_id": best_task.id if covered and best_task is not None else None,
                    "covered": covered,
                }
            )
            if covered and best_task is not None and not allow_reuse:
                remaining_tasks.remove(best_task)
        return coverage

    @staticmethod
    def _semantic_overlap_score(item_tokens: set[str], task_tokens: set[str]) -> int:
        return len(item_tokens & task_tokens)

    @staticmethod
    def _semantic_overlap_required(item_tokens: set[str]) -> int:
        return 1 if len(item_tokens) <= 1 else 2

    @classmethod
    def _semantic_anchor_tokens(cls, item_tokens: set[str]) -> set[str]:
        return item_tokens & (cls.SEMANTIC_ANCHOR_TOKENS | cls.CRUD_OPERATION_TOKENS)

    @classmethod
    def _semantic_anchor_satisfied(
        cls,
        anchor_tokens: set[str],
        candidate_tokens: set[str],
    ) -> bool:
        required_tokens = set(anchor_tokens)
        required_operations = required_tokens & cls.CRUD_OPERATION_TOKENS
        if required_operations:
            candidate_operations = set(candidate_tokens & cls.CRUD_OPERATION_TOKENS)
            if "crud" in candidate_tokens:
                candidate_operations.update(cls.CRUD_OPERATION_TOKENS)
            if not required_operations.issubset(candidate_operations):
                return False
            required_tokens -= required_operations
        if "crud" in required_tokens:
            required_tokens.remove("crud")
            has_crud_coverage = (
                "crud" in candidate_tokens
                or cls.CRUD_OPERATION_TOKENS.issubset(candidate_tokens)
            )
            if not has_crud_coverage:
                return False
        return required_tokens.issubset(candidate_tokens)

    @classmethod
    def _semantic_tokens(
        cls,
        text: str,
        *,
        include_action_tokens: bool = False,
    ) -> set[str]:
        stopwords = {
            "a",
            "add",
            "added",
            "adding",
            "adds",
            "an",
            "and",
            "app",
            "application",
            "build",
            "can",
            "check",
            "checks",
            "core",
            "create",
            "created",
            "creates",
            "creating",
            "do",
            "does",
            "feature",
            "for",
            "from",
            "has",
            "have",
            "helper",
            "implement",
            "implemented",
            "implements",
            "in",
            "initial",
            "is",
            "it",
            "manage",
            "module",
            "of",
            "operation",
            "operations",
            "part",
            "return",
            "returns",
            "setup",
            "focused",
            "should",
            "task",
            "test",
            "tests",
            "that",
            "the",
            "to",
            "with",
            "works",
        }
        aliases = {
            "authenticate": "auth",
            "authenticated": "auth",
            "authentication": "auth",
            "log": "auth",
            "login": "auth",
            "sign": "auth",
            "signin": "auth",
            "signup": "auth",
            "registration": "register",
            "quiz": "quiz",
            "quizzes": "quiz",
            "quizze": "quiz",
            "add": "create",
            "added": "create",
            "adding": "create",
            "adds": "create",
            "create": "create",
            "created": "create",
            "creates": "create",
            "creating": "create",
            "read": "read",
            "reads": "read",
            "list": "read",
            "lists": "read",
            "listed": "read",
            "listing": "read",
            "update": "update",
            "updates": "update",
            "updated": "update",
            "updating": "update",
            "edit": "update",
            "edits": "update",
            "edited": "update",
            "editing": "update",
            "delete": "delete",
            "deletes": "delete",
            "deleted": "delete",
            "deleting": "delete",
            "remove": "delete",
            "removes": "delete",
            "removed": "delete",
            "removing": "delete",
            "student": "user",
            "students": "user",
            "teacher": "role",
            "teachers": "role",
            "vocab": "vocabulary",
        }
        crud_operation_aliases = {
            "create": {
                "add",
                "added",
                "adding",
                "adds",
                "create",
                "created",
                "creates",
                "creating",
                "post",
                "posts",
            },
            "read": {
                "get",
                "gets",
                "read",
                "reads",
                "list",
                "lists",
                "listed",
                "listing",
            },
            "update": {
                "patch",
                "patches",
                "put",
                "puts",
                "update",
                "updates",
                "updated",
                "updating",
                "edit",
                "edits",
                "edited",
                "editing",
            },
            "delete": {
                "delete",
                "deletes",
                "deleted",
                "deleting",
                "remove",
                "removes",
                "removed",
                "removing",
            },
        }
        action_aliases = {
            raw_alias: operation
            for operation, raw_aliases in crud_operation_aliases.items()
            for raw_alias in raw_aliases
        }
        active_aliases = {**aliases, **action_aliases} if include_action_tokens else aliases
        expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        raw_tokens = re.findall(r"[a-z0-9_]+", expanded.lower())
        tokens: set[str] = set()
        raw_pieces: set[str] = set()
        for token in raw_tokens:
            pieces = [token]
            if "_" in token:
                token_parts = [part for part in token.split("_") if part]
                pieces.extend(token_parts)
                pieces.extend(
                    f"{left}_{right}"
                    for left, right in zip(token_parts, token_parts[1:])
                    if left and right
                )
            if "-" in token:
                token_parts = [part for part in token.split("-") if part]
                pieces.extend(token_parts)
                pieces.extend(
                    f"{left}_{right}"
                    for left, right in zip(token_parts, token_parts[1:])
                    if left and right
                )
            for piece in pieces:
                raw_pieces.add(piece)
                if len(piece) < 2 or piece.isdigit() or piece in stopwords:
                    continue
                normalized = active_aliases.get(piece, piece)
                if (
                    len(normalized) > 3
                    and normalized.endswith("s")
                    and not normalized.endswith("ss")
                ):
                    normalized = normalized[:-1]
                tokens.add(normalized)
        if include_action_tokens:
            for operation, raw_aliases in crud_operation_aliases.items():
                if raw_pieces & raw_aliases:
                    tokens.add(operation)
        if all(raw_pieces & aliases for aliases in crud_operation_aliases.values()):
            tokens.update(cls.CRUD_OPERATION_TOKENS)
        return tokens

    @staticmethod
    def _task_semantic_text(task: PlannedTask) -> str:
        return " ".join(
            [
                task.id,
                task.title,
                task.description,
                " ".join(
                    JobRunner._meaningful_planning_items(task.acceptance_criteria)
                ),
                " ".join(task.target_files),
                " ".join(task.required_artifacts),
            ]
        )

    @staticmethod
    def _find_task_graph_cycle(task_graph: TaskGraph) -> list[str]:
        dependencies = {task.id: list(task.depends_on) for task in task_graph.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def visit(task_id: str) -> list[str] | None:
            if task_id in visiting:
                cycle_start = stack.index(task_id) if task_id in stack else 0
                return [*stack[cycle_start:], task_id]
            if task_id in visited:
                return None
            visiting.add(task_id)
            stack.append(task_id)
            for dependency in dependencies.get(task_id, []):
                if dependency not in dependencies:
                    continue
                cycle = visit(dependency)
                if cycle is not None:
                    return cycle
            visiting.remove(task_id)
            visited.add(task_id)
            stack.pop()
            return None

        for task_id in dependencies:
            cycle = visit(task_id)
            if cycle is not None:
                return cycle
        return []

    def _run_autonomous_task_loop(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
    ) -> tuple[
        list[ImplementationResult],
        list[TestWriterResult],
        TestRunResult,
        list[dict[str, Any]],
    ]:
        implementation_tasks = self._prioritize_project_setup_tasks(
            self._tasks_for_roles(task_graph, self.IMPLEMENTATION_TASK_ROLES)
        )
        pending_test_tasks = self._tasks_for_roles(task_graph, self.TEST_TASK_ROLES)
        implementation_results = self._load_recorded_task_results(
            record,
            "implementation_tasks",
            ImplementationResult,
            allowed_result_statuses={ImplementationStatus.IMPLEMENTED.value},
        )
        test_writer_results = self._load_recorded_task_results(
            record,
            "test_writer_tasks",
            TestWriterResult,
            allowed_result_statuses={TestWriterStatus.TESTS_WRITTEN.value},
        )
        raw_stage_results = record.outputs.get("autonomous_stages", [])
        stage_results: list[dict[str, Any]] = (
            list(raw_stage_results) if isinstance(raw_stage_results, list) else []
        )
        completed_task_ids: set[str] = set(record.completed_task_ids)
        recorded_implementation_task_ids = self._recorded_task_ids(
            record,
            "implementation_tasks",
            allowed_result_statuses={ImplementationStatus.IMPLEMENTED.value},
        )
        last_test_result = self._synthetic_test_result(success=True, output="No tests run yet.")
        if not self._ensure_completed_task_artifacts_available(
            record,
            task_graph,
            completed_task_ids,
        ):
            return implementation_results, test_writer_results, last_test_result, stage_results
        ready_task_ids: set[str] = set(completed_task_ids)
        pending_test_tasks = [task for task in pending_test_tasks if task.id not in completed_task_ids]

        if not implementation_tasks and not pending_test_tasks:
            implementation_results = self._run_implementation_tasks(record, task_graph)
            test_writer_results = self._run_test_writer_tasks(
                record,
                task_graph,
                self._choose_primary_task(task_graph),
                implementation_results,
            )
            last_test_result = self._run_tests_with_fixes(
                record,
                self._choose_primary_task(task_graph),
            )
            return implementation_results, test_writer_results, last_test_result, stage_results

        implementation_summaries: list[str] = []
        for task in implementation_tasks:
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            if task.id in completed_task_ids:
                unmet_dependencies = self._unmet_dependencies_for_task(
                    task,
                    completed_task_ids,
                )
                if unmet_dependencies:
                    self._recover_unmet_task_dependencies(
                        record,
                        task,
                        unmet_dependencies,
                    )
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_test_pairs = self._run_ready_test_tasks(
                    record=record,
                    pending_test_tasks=pending_test_tasks,
                    ready_task_ids=ready_task_ids,
                    implementation_results=implementation_results,
                    test_writer_results=test_writer_results,
                )
                if self._should_pause_for_recovery(record):
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_test_results = [item[1] for item in stage_test_pairs]
                if stage_test_results:
                    last_test_result = self._run_tests_with_fixes(
                        record,
                        task,
                        logs=[f"resumed after completed task {task.id}"],
                    )
                    if self._autonomous_stage_limit_reached(record, stage_results):
                        return (
                            implementation_results,
                            test_writer_results,
                            last_test_result,
                            stage_results,
                        )
                    stage_result = {
                        "stage": len(stage_results) + 1,
                        "task": task.model_dump(),
                        "implementation": None,
                        "test_writer_results": [item.model_dump() for item in stage_test_results],
                        "change_summary": self._build_stage_change_summary(
                            None,
                            stage_test_results,
                        ),
                        "test_run": last_test_result.model_dump(),
                    }
                    stage_results.append(stage_result)
                    self._record_stage_checkpoint(record, stage_result)
                    if self._should_pause_for_recovery(record) or not last_test_result.success:
                        return (
                            implementation_results,
                            test_writer_results,
                            last_test_result,
                            stage_results,
                        )
                    self._mark_tasks_completed(record, [task.id for task, _ in stage_test_pairs])
                    completed_task_ids.update(task.id for task, _ in stage_test_pairs)
                    ready_task_ids.update(task.id for task, _ in stage_test_pairs)
                continue
            if task.id in recorded_implementation_task_ids:
                unmet_dependencies = self._unmet_dependencies_for_task(
                    task,
                    completed_task_ids,
                )
                if unmet_dependencies:
                    self._recover_unmet_task_dependencies(
                        record,
                        task,
                        unmet_dependencies,
                    )
                    return implementation_results, test_writer_results, last_test_result, stage_results
                if not self._ensure_resumed_task_artifacts_available(
                    record,
                    task,
                    source="recorded_implementation_task",
                ):
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_test_pairs: list[tuple[PlannedTask, TestWriterResult]] = []
                ready_task_ids.add(task.id)
                stage_test_pairs = self._run_ready_test_tasks(
                    record=record,
                    pending_test_tasks=pending_test_tasks,
                    ready_task_ids=ready_task_ids,
                    implementation_results=implementation_results,
                    test_writer_results=test_writer_results,
                )
                if self._should_pause_for_recovery(record):
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_test_results = [item[1] for item in stage_test_pairs]
                if not stage_test_results:
                    test_writer = self._run_stage_test_gate(
                        record,
                        task,
                        implementation_results,
                        test_writer_results,
                    )
                    test_writer_results.append(test_writer)
                    stage_test_results.append(test_writer)
                    if self._should_pause_for_recovery(record):
                        return (
                            implementation_results,
                            test_writer_results,
                            last_test_result,
                            stage_results,
                        )
                last_test_result = self._run_tests_with_fixes(
                    record,
                    task,
                    logs=[f"resumed validation for implemented task {task.id}"],
                )
                if self._autonomous_stage_limit_reached(record, stage_results):
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_result = {
                    "stage": len(stage_results) + 1,
                    "task": task.model_dump(),
                    "implementation": None,
                    "test_writer_results": [item.model_dump() for item in stage_test_results],
                    "change_summary": self._build_stage_change_summary(
                        None,
                        stage_test_results,
                    ),
                    "test_run": last_test_result.model_dump(),
                }
                stage_results.append(stage_result)
                self._record_stage_checkpoint(record, stage_result)
                if self._should_pause_for_recovery(record) or not last_test_result.success:
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_review = self._run_stage_review_gate(record, task)
                if stage_review is not None:
                    stage_result["stage_review"] = stage_review
                    self._annotate_stage_status_for_recovery(record, stage_result)
                    self._recover_failed_stage_if_needed(record, stage_result)
                    self.store.update(record)
                    if self._should_pause_for_recovery(record):
                        return implementation_results, test_writer_results, last_test_result, stage_results
                    last_test_result = self._run_tests_with_fixes(
                        record,
                        task,
                        logs=["stage review applied fixes"],
                    )
                    stage_result["post_review_test_run"] = last_test_result.model_dump()
                    self.store.update(record)
                    if (
                        self._should_pause_for_recovery(record)
                        or not last_test_result.success
                    ):
                        return implementation_results, test_writer_results, last_test_result, stage_results
                self._mark_task_completed(record, task.id)
                completed_task_ids.add(task.id)
                ready_task_ids.add(task.id)
                self._mark_tasks_completed(record, [task.id for task, _ in stage_test_pairs])
                completed_task_ids.update(task.id for task, _ in stage_test_pairs)
                ready_task_ids.update(task.id for task, _ in stage_test_pairs)
                continue
            unmet_dependencies = self._unmet_dependencies_for_task(
                task,
                completed_task_ids,
            )
            if unmet_dependencies:
                self._recover_unmet_task_dependencies(
                    record,
                    task,
                    unmet_dependencies,
                )
                return implementation_results, test_writer_results, last_test_result, stage_results
            if self._is_project_setup_task(task):
                implementation = self._run_project_setup_scaffold(record, task)
            else:
                implementation_role = "scaffold" if task.role == "scaffold" else "implementer"
                implementation = self._run_structured_role(
                    record,
                    implementation_role,
                    ImplementationResult,
                    f"Implement the next autonomous stage task {task.id}: {task.title}",
                    task=task,
                    logs=implementation_summaries,
                )
                self._record_task_output(record, "implementation_tasks", task, implementation)
            implementation_results.append(implementation)
            implementation_summaries.append(f"{task.id}: {implementation.summary}")
            if not self._implementation_allows_progress(record, task, implementation):
                return implementation_results, test_writer_results, last_test_result, stage_results
            if not self._is_project_setup_task(task):
                self._apply_patches(record, implementation_role, implementation.patches)
            if self._should_pause_for_recovery(record):
                return implementation_results, test_writer_results, last_test_result, stage_results
            ready_task_ids.add(task.id)

            stage_test_pairs = self._run_ready_test_tasks(
                record=record,
                pending_test_tasks=pending_test_tasks,
                ready_task_ids=ready_task_ids,
                implementation_results=implementation_results,
                test_writer_results=test_writer_results,
            )
            if self._should_pause_for_recovery(record):
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_test_results = [item[1] for item in stage_test_pairs]
            if not stage_test_results:
                test_writer = self._run_stage_test_gate(
                    record,
                    task,
                    implementation_results,
                    test_writer_results,
                )
                test_writer_results.append(test_writer)
                stage_test_results.append(test_writer)
                if self._should_pause_for_recovery(record):
                    return implementation_results, test_writer_results, last_test_result, stage_results
            last_test_result = self._run_tests_with_fixes(
                record,
                task,
                logs=[implementation.summary],
            )
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_result = {
                "stage": len(stage_results) + 1,
                "task": task.model_dump(),
                "implementation": implementation.model_dump(),
                "test_writer_results": [item.model_dump() for item in stage_test_results],
                "change_summary": self._build_stage_change_summary(
                    implementation,
                    stage_test_results,
                ),
                "test_run": last_test_result.model_dump(),
            }
            stage_results.append(stage_result)
            self._record_stage_checkpoint(record, stage_result)
            if self._should_pause_for_recovery(record) or not last_test_result.success:
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_review = self._run_stage_review_gate(record, task)
            if stage_review is not None:
                stage_result["stage_review"] = stage_review
                self._annotate_stage_status_for_recovery(record, stage_result)
                self._recover_failed_stage_if_needed(record, stage_result)
                self.store.update(record)
                if self._should_pause_for_recovery(record):
                    return implementation_results, test_writer_results, last_test_result, stage_results
                last_test_result = self._run_tests_with_fixes(
                    record,
                    task,
                    logs=["stage review applied fixes"],
                )
                stage_result["post_review_test_run"] = last_test_result.model_dump()
                self.store.update(record)
                if (
                    self._should_pause_for_recovery(record)
                    or not last_test_result.success
                ):
                    return implementation_results, test_writer_results, last_test_result, stage_results
            self._mark_task_completed(record, task.id)
            completed_task_ids.add(task.id)
            ready_task_ids.add(task.id)
            self._mark_tasks_completed(record, [task.id for task, _ in stage_test_pairs])
            completed_task_ids.update(task.id for task, _ in stage_test_pairs)
            ready_task_ids.update(task.id for task, _ in stage_test_pairs)

        while pending_test_tasks:
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            task = pending_test_tasks.pop(0)
            unmet_dependencies = self._unmet_dependencies_for_task(
                task,
                completed_task_ids,
            )
            if unmet_dependencies:
                self._recover_unmet_task_dependencies(
                    record,
                    task,
                    unmet_dependencies,
                )
                return implementation_results, test_writer_results, last_test_result, stage_results
            test_writer = self._run_test_writer_task(
                record,
                task,
                implementation_results,
                test_writer_results,
            )
            test_writer_results.append(test_writer)
            if self._should_pause_for_recovery(record):
                return implementation_results, test_writer_results, last_test_result, stage_results
            last_test_result = self._run_tests_with_fixes(record, task, logs=[test_writer.summary])
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_result = {
                "stage": len(stage_results) + 1,
                "task": task.model_dump(),
                "implementation": None,
                "test_writer_results": [test_writer.model_dump()],
                "change_summary": self._build_stage_change_summary(
                    None,
                    [test_writer],
                ),
                "test_run": last_test_result.model_dump(),
            }
            stage_results.append(stage_result)
            self._record_stage_checkpoint(record, stage_result)
            if self._should_pause_for_recovery(record) or not last_test_result.success:
                return implementation_results, test_writer_results, last_test_result, stage_results
            self._mark_task_completed(record, task.id)
            completed_task_ids.add(task.id)
            ready_task_ids.add(task.id)

        if not test_writer_results:
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            primary_task = self._choose_primary_task(task_graph)
            test_writer = self._run_structured_role(
                record,
                "test_writer",
                TestWriterResult,
                "Add smoke tests for the smallest working core before continuing",
                task=primary_task,
                logs=[item.summary for item in implementation_results],
            )
            test_writer_results.append(test_writer)
            self._record_task_output(record, "test_writer_tasks", primary_task, test_writer)
            if not self._test_writer_allows_progress(record, primary_task, test_writer):
                return implementation_results, test_writer_results, last_test_result, stage_results
            self._apply_patches(record, "test_writer", test_writer.patches)
            last_test_result = self._run_tests_with_fixes(
                record,
                primary_task,
                logs=[test_writer.summary],
            )
            if self._autonomous_stage_limit_reached(record, stage_results):
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_result = {
                "stage": len(stage_results) + 1,
                "task": primary_task.model_dump() if primary_task is not None else None,
                "implementation": None,
                "test_writer_results": [test_writer.model_dump()],
                "change_summary": self._build_stage_change_summary(
                    None,
                    [test_writer],
                ),
                "test_run": last_test_result.model_dump(),
            }
            stage_results.append(stage_result)
            self._record_stage_checkpoint(record, stage_result)
            if self._should_pause_for_recovery(record) or not last_test_result.success:
                return implementation_results, test_writer_results, last_test_result, stage_results
            if primary_task is not None:
                self._mark_task_completed(record, primary_task.id)

        return implementation_results, test_writer_results, last_test_result, stage_results

    @staticmethod
    def _unmet_dependencies_for_task(
        task: PlannedTask,
        completed_task_ids: set[str],
    ) -> list[str]:
        return [
            dependency
            for dependency in task.depends_on
            if dependency not in completed_task_ids
        ]

    def _recover_unmet_task_dependencies(
        self,
        record: JobRecord,
        task: PlannedTask,
        unmet_dependencies: list[str],
    ) -> None:
        self._recover_record(
            record,
            error=f"unmet_task_dependencies:{','.join(unmet_dependencies)}",
            runtime_state={
                "failed_task_id": task.id,
                "unmet_dependencies": list(unmet_dependencies),
            },
        )

    def _ensure_completed_task_artifacts_available(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
        completed_task_ids: set[str],
    ) -> bool:
        if not completed_task_ids:
            return True
        for task in task_graph.tasks:
            if task.id not in completed_task_ids:
                continue
            if self._unmet_dependencies_for_task(task, completed_task_ids):
                continue
            if not self._ensure_resumed_task_artifacts_available(
                record,
                task,
                source="completed_task_id",
            ):
                return False
        return True

    def _ensure_resumed_task_artifacts_available(
        self,
        record: JobRecord,
        task: PlannedTask,
        *,
        source: str,
    ) -> bool:
        report = self._resumed_task_artifact_report(record, task)
        if not (
            report["missing_artifacts"]
            or report["non_file_artifacts"]
            or report["empty_artifacts"]
            or report["invalid_artifacts"]
        ):
            return True
        checks = record.outputs.setdefault("resumed_task_artifact_checks", [])
        if isinstance(checks, list):
            checks.append(
                {
                    "task_id": task.id,
                    "role": task.role,
                    "source": source,
                    **report,
                }
            )
        runtime_state: dict[str, Any] = {
            **record.runtime_state,
            "failed_task_id": task.id,
            "resumed_task_artifact_source": source,
        }
        if report["required_artifacts"]:
            runtime_state["required_artifacts"] = report["required_artifacts"]
        if report["target_files"]:
            runtime_state["target_files"] = report["target_files"]
        if report["missing_artifacts"]:
            runtime_state["missing_artifacts"] = report["missing_artifacts"]
        if report["non_file_artifacts"]:
            runtime_state["non_file_artifacts"] = report["non_file_artifacts"]
        if report["empty_artifacts"]:
            runtime_state["empty_artifacts"] = report["empty_artifacts"]
        if report["invalid_artifacts"]:
            runtime_state["invalid_artifacts"] = report["invalid_artifacts"]
        artifact_failures = self._unique_paths(
            [
                *report["missing_artifacts"],
                *report["non_file_artifacts"],
                *report["empty_artifacts"],
                *report["invalid_artifacts"],
            ]
        )
        if self._is_project_setup_task(task) or (
            artifact_failures
            and all(path in self.PROJECT_SETUP_REQUIRED_ARTIFACTS for path in artifact_failures)
        ):
            runtime_state["force_project_setup_scaffold"] = True
        self._recover_record(
            record,
            error="required_artifacts_missing:resumed_task_artifacts_missing",
            runtime_state=runtime_state,
        )
        return False

    def _resumed_task_artifact_report(
        self,
        record: JobRecord,
        task: PlannedTask,
    ) -> dict[str, list[str]]:
        required_artifacts = self._clean_declared_artifact_paths(task.required_artifacts)
        target_files = self._clean_declared_artifact_paths(task.target_files)
        declared = self._unique_paths([*required_artifacts, *target_files])
        invalid_artifacts = self._invalid_planning_artifact_paths(declared)
        invalid_set = set(invalid_artifacts)
        root = self._workspace_root(record)
        missing_artifacts: list[str] = []
        non_file_artifacts: list[str] = []
        empty_artifacts: list[str] = []
        for path in declared:
            normalized = path.replace("\\", "/").removeprefix("./")
            if path in invalid_set or normalized in invalid_set:
                continue
            target = root / normalized
            if not target.exists():
                missing_artifacts.append(normalized)
                continue
            if not target.is_file():
                non_file_artifacts.append(normalized)
                continue
            if (
                target.stat().st_size == 0
                and Path(normalized).name not in RecoveryExecutor.ALLOW_EMPTY_ARTIFACT_NAMES
            ):
                empty_artifacts.append(normalized)
        return {
            "required_artifacts": required_artifacts,
            "target_files": target_files,
            "missing_artifacts": self._unique_paths(missing_artifacts),
            "non_file_artifacts": self._unique_paths(non_file_artifacts),
            "empty_artifacts": self._unique_paths(empty_artifacts),
            "invalid_artifacts": invalid_artifacts,
        }

    @staticmethod
    def _clean_declared_artifact_paths(paths: list[str]) -> list[str]:
        return JobRunner._unique_paths(
            [
                str(path).replace("\\", "/").removeprefix("./").strip()
                for path in paths
                if str(path).strip()
            ]
        )

    def _run_ready_test_tasks(
        self,
        *,
        record: JobRecord,
        pending_test_tasks: list[PlannedTask],
        ready_task_ids: set[str],
        implementation_results: list[ImplementationResult],
        test_writer_results: list[TestWriterResult],
    ) -> list[tuple[PlannedTask, TestWriterResult]]:
        ready_tasks: list[PlannedTask] = []
        for task in list(pending_test_tasks):
            local_dependencies = [
                dependency
                for dependency in task.depends_on
                if any(dependency == item.id for item in pending_test_tasks)
                or dependency not in ready_task_ids
            ]
            if not local_dependencies or all(
                dependency in ready_task_ids for dependency in task.depends_on
            ):
                ready_tasks.append(task)
                pending_test_tasks.remove(task)

        results: list[tuple[PlannedTask, TestWriterResult]] = []
        for task in ready_tasks:
            test_writer = self._run_test_writer_task(
                record,
                task,
                implementation_results,
                test_writer_results,
            )
            test_writer_results.append(test_writer)
            results.append((task, test_writer))
            if self._should_pause_for_recovery(record):
                break
            ready_task_ids.add(task.id)
        return results

    def _run_test_writer_task(
        self,
        record: JobRecord,
        task: PlannedTask,
        implementation_results: list[ImplementationResult],
        previous_test_writer_results: list[TestWriterResult],
    ) -> TestWriterResult:
        task = self._task_with_recovery_targets(record, "test_writer", task) or task
        if not self._ensure_project_setup_ready_before_test_writer(record, task):
            return TestWriterResult(
                status=TestWriterStatus.BLOCKED,
                summary="Project setup artifacts are missing; deterministic scaffold must run before test_writer.",
            )
        logs = [f"implementation: {item.summary}" for item in implementation_results]
        logs.extend(f"existing tests: {item.summary}" for item in previous_test_writer_results)
        test_writer = self._run_structured_role(
            record,
            "test_writer",
            TestWriterResult,
            f"Add or update tests for autonomous stage task {task.id}: {task.title}",
            task=task,
            logs=logs,
        )
        self._record_task_output(record, "test_writer_tasks", task, test_writer)
        if not self._test_writer_allows_progress(record, task, test_writer):
            return test_writer
        self._apply_patches(record, "test_writer", test_writer.patches)
        return test_writer

    def _run_stage_test_gate(
        self,
        record: JobRecord,
        task: PlannedTask,
        implementation_results: list[ImplementationResult],
        previous_test_writer_results: list[TestWriterResult],
    ) -> TestWriterResult:
        task = self._task_with_recovery_targets(record, "test_writer", task) or task
        if not self._ensure_project_setup_ready_before_test_writer(record, task):
            return TestWriterResult(
                status=TestWriterStatus.BLOCKED,
                summary="Project setup artifacts are missing; deterministic scaffold must run before test_writer.",
            )
        logs = [f"implementation: {item.summary}" for item in implementation_results]
        logs.extend(f"existing tests: {item.summary}" for item in previous_test_writer_results)
        test_writer = self._run_structured_role(
            record,
            "test_writer",
            TestWriterResult,
            (
                "Add focused tests for the current autonomous stage because "
                f"the planner did not provide a ready test task for {task.id}: {task.title}"
            ),
            task=task,
            logs=logs,
        )
        self._record_task_output(record, "test_writer_tasks", task, test_writer)
        if not self._test_writer_allows_progress(record, task, test_writer):
            return test_writer
        self._apply_patches(record, "test_writer", test_writer.patches)
        return test_writer

    def _run_tests_with_fixes(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        logs: list[str] | None = None,
    ) -> TestRunResult:
        test_result = self._run_tests(record)
        attempts = 0
        same_failure_repeats = 0
        previous_failure_signature: str | None = None
        while not test_result.success and attempts < self.max_attempts_per_task:
            diagnosis = self._diagnose_test_failure(
                record,
                task,
                test_result,
                previous_failure_signature=previous_failure_signature,
                same_failure_repeats=same_failure_repeats,
            )
            failure_signature = diagnosis.failure_signature or self._failure_signature(
                test_result
            )
            if previous_failure_signature == failure_signature:
                same_failure_repeats += 1
            else:
                same_failure_repeats = 1
                previous_failure_signature = failure_signature
            diagnosis_logs = self._diagnosis_logs(
                diagnosis,
                same_failure_repeats=same_failure_repeats,
                repeated=record.same_test_failure_count > 0,
            )
            fix = self._run_structured_role(
                record,
                "fixer",
                FixResult,
                "Fix only the current autonomous stage test failures",
                task=task,
                logs=[*(logs or []), *diagnosis_logs, test_result.output_excerpt],
            )
            attempts += 1
            record.failure_count += 1
            if failure_signature:
                record.same_test_failure_count += 1
            else:
                same_failure_repeats = 0
                record.same_test_failure_count = 0
            self.store.update(record)
            if not self._fixer_allows_progress(record, task, fix):
                return test_result
            ensure_fixer_safe(
                fix.patches,
                workspace_root=self._workspace_root(record),
            )
            self._apply_patches(record, "fixer", fix.patches)
            if same_failure_repeats >= self.max_same_failure_repeats:
                self._store_failure_diagnosis(record, diagnosis)
                record.last_error = "same_failure_threshold_reached"
                record.outputs["recovery_ready"] = {
                    "reason": "same_failure_threshold_reached",
                    "classification": diagnosis.classification.value,
                    "root_cause": diagnosis.root_cause,
                    "recommended_fix_strategy": diagnosis.recommended_fix_strategy,
                    "retry_mode": diagnosis.retry_mode.value,
                    "should_retry": diagnosis.should_retry,
                    "failure_signature": diagnosis.failure_signature,
                    "diagnosed_last_error": (
                        "diagnosed_repeated_failure:"
                        f"{diagnosis.classification.value}"
                    ),
                }
                self._recover_record(record, error="same_failure_threshold_reached")
                return test_result
            test_result = self._run_tests(record)
        if not test_result.success and attempts >= self.max_attempts_per_task:
            if not (
                self._is_recoverable_status(record.status)
                or self._is_terminal_status(record.status)
                or self._is_waiting_status(record.status)
            ):
                self._recover_record(record, error="max_attempts_exceeded")
        return test_result

    def _diagnose_test_failure(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        test_result: TestRunResult,
        *,
        previous_failure_signature: str | None,
        same_failure_repeats: int,
    ) -> FailureDiagnosis:
        fallback = self._deterministic_failure_diagnosis(test_result)
        logs = [
            "diagnosis_seed: "
            f"classification={fallback.classification.value}; "
            f"signature={fallback.failure_signature or 'unknown'}; "
            f"root_cause={fallback.root_cause}; "
            f"recommended_fix_strategy={fallback.recommended_fix_strategy}",
            test_result.output_excerpt,
        ]
        if previous_failure_signature and previous_failure_signature == fallback.failure_signature:
            logs.append(
                "repeated_failure_context: this failure signature has repeated; "
                "change strategy instead of repeating the prior fix."
            )
        try:
            diagnosis = self._run_structured_role(
                record,
                "diagnoser",
                FailureDiagnosis,
                "Diagnose the deterministic test failure before fixing",
                task=task,
                logs=logs,
            )
        except Exception as exc:
            diagnosis = fallback.model_copy(
                update={
                    "root_cause": (
                        f"{fallback.root_cause} Diagnoser unavailable: {exc}"
                    ),
                    "confidence": min(fallback.confidence, 0.55),
                }
            )
        if not diagnosis.failure_signature:
            diagnosis = diagnosis.model_copy(
                update={"failure_signature": fallback.failure_signature}
            )
        if not diagnosis.failed_tests:
            diagnosis = diagnosis.model_copy(
                update={"failed_tests": fallback.failed_tests}
            )
        self._store_failure_diagnosis(
            record,
            diagnosis,
            same_failure_repeats=same_failure_repeats,
        )
        return diagnosis

    @staticmethod
    def _diagnosis_logs(
        diagnosis: FailureDiagnosis,
        *,
        same_failure_repeats: int,
        repeated: bool,
    ) -> list[str]:
        logs = [
            "failure_diagnosis: "
            f"classification={diagnosis.classification.value}; "
            f"retry_mode={diagnosis.retry_mode.value}; "
            f"confidence={diagnosis.confidence}; "
            f"should_retry={diagnosis.should_retry}; "
            f"signature={diagnosis.failure_signature or 'unknown'}; "
            f"root_cause={diagnosis.root_cause}; "
            f"recommended_fix_strategy={diagnosis.recommended_fix_strategy}; "
            f"failed_files={','.join(diagnosis.failed_files)}; "
            f"failed_tests={','.join(diagnosis.failed_tests)}"
        ]
        if repeated or same_failure_repeats > 1:
            logs.append(
                "repeated_failure_instruction: the same failure signature is still present; "
                "do not repeat the previous patch strategy. Focus only on the diagnosed "
                "root cause and inspect the named files first when needed."
            )
        if diagnosis.retry_mode == FailureRetryMode.INSPECT_FILES_FIRST:
            logs.append(
                "diagnosis_instruction: inspect the relevant files before editing; avoid "
                "guessing from the test output alone."
            )
        elif diagnosis.retry_mode == FailureRetryMode.REWRITE_SMALL_SCOPE:
            logs.append(
                "diagnosis_instruction: replace only the smallest coherent scope needed "
                "to address the diagnosed failure."
            )
        return logs

    def _store_failure_diagnosis(
        self,
        record: JobRecord,
        diagnosis: FailureDiagnosis,
        *,
        same_failure_repeats: int | None = None,
    ) -> None:
        payload = diagnosis.model_dump(mode="json")
        if same_failure_repeats is not None:
            payload["same_failure_repeats"] = same_failure_repeats
        record.outputs["failure_diagnosis"] = payload
        history = record.outputs.setdefault("failure_diagnoses", [])
        if isinstance(history, list):
            history.append(payload)
            del history[:-10]
        self.store.update(record)

    def _deterministic_failure_diagnosis(
        self,
        test_result: TestRunResult,
    ) -> FailureDiagnosis:
        output = test_result.output_excerpt or ""
        signature = self._failure_signature(test_result)
        failed_files = self._extract_failed_files(output)
        if "no tests ran" in output.lower() or test_result.executed_test_count == 0:
            return FailureDiagnosis(
                classification=FailureClassification.TEST_EXPECTATION_MISMATCH,
                root_cause=(
                    "Pytest did not discover any tests. The project likely has a "
                    "pytest.ini/testpaths configuration that excludes existing test "
                    "files, or the generated tests are outside the configured test "
                    "directory."
                ),
                failed_files=[*failed_files, "pytest.ini", "tests/"][:8],
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Inspect pytest.ini and the actual test file layout; update "
                    "testpaths or move tests so pytest discovers the generated tests."
                ),
                confidence=0.85,
                should_retry=True,
                retry_mode=FailureRetryMode.INSPECT_FILES_FIRST,
                failure_signature=signature,
            )
        if "ModuleNotFoundError" in output or "No module named" in output:
            return FailureDiagnosis(
                classification=FailureClassification.MISSING_DEPENDENCY,
                root_cause="A required Python module or package cannot be imported.",
                failed_files=failed_files,
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Add the missing dependency or correct the import/module path without "
                    "weakening tests."
                ),
                confidence=0.8,
                should_retry=True,
                retry_mode=FailureRetryMode.TARGETED_FIX,
                failure_signature=signature,
            )
        if "ImportError" in output:
            return FailureDiagnosis(
                classification=FailureClassification.IMPORT_ERROR,
                root_cause=self._import_error_root_cause(output),
                failed_files=failed_files,
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Correct the import wiring so symbols are imported from the module "
                    "where they are actually defined."
                ),
                confidence=0.85,
                should_retry=True,
                retry_mode=FailureRetryMode.TARGETED_FIX,
                failure_signature=signature,
            )
        if "SyntaxError" in output:
            return FailureDiagnosis(
                classification=FailureClassification.SYNTAX_ERROR,
                root_cause="Generated code contains invalid Python syntax.",
                failed_files=failed_files,
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Fix the syntax error in the named file and preserve existing behavior."
                ),
                confidence=0.85,
                should_retry=True,
                retry_mode=FailureRetryMode.TARGETED_FIX,
                failure_signature=signature,
            )
        if self._looks_like_frontend_build_error(output):
            return FailureDiagnosis(
                classification=FailureClassification.FRONTEND_BUILD_ERROR,
                root_cause="Frontend build or package tooling failed.",
                failed_files=failed_files,
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Fix the smallest frontend source or package configuration issue shown "
                    "by the build output."
                ),
                confidence=0.75,
                should_retry=True,
                retry_mode=FailureRetryMode.TARGETED_FIX,
                failure_signature=signature,
            )
        if "E   AssertionError" in output or "assert " in output:
            return FailureDiagnosis(
                classification=FailureClassification.TEST_EXPECTATION_MISMATCH,
                root_cause="Implementation behavior does not match a test expectation.",
                failed_files=failed_files,
                failed_tests=test_result.failed_tests,
                recommended_fix_strategy=(
                    "Adjust the implementation to satisfy the asserted behavior without "
                    "loosening the test."
                ),
                confidence=0.75,
                should_retry=True,
                retry_mode=FailureRetryMode.NORMAL_FIX,
                failure_signature=signature,
            )
        return FailureDiagnosis(
            classification=FailureClassification.RUNTIME_ERROR,
            root_cause="Tests failed with a runtime error that needs focused inspection.",
            failed_files=failed_files,
            failed_tests=test_result.failed_tests,
            recommended_fix_strategy=(
                "Inspect the failing traceback and make the smallest code change that "
                "addresses the runtime failure."
            ),
            confidence=0.45,
            should_retry=True,
            retry_mode=FailureRetryMode.INSPECT_FILES_FIRST,
            failure_signature=signature,
        )

    @staticmethod
    def _failure_signature(test_result: TestRunResult) -> str:
        output = test_result.output_excerpt or ""
        patterns = [
            r"(ImportError:\s*[^\n]+)",
            r"(ModuleNotFoundError:\s*[^\n]+)",
            r"(SyntaxError:\s*[^\n]+)",
            r"(AssertionError:\s*[^\n]*)",
            r"(TypeError:\s*[^\n]+)",
            r"(ValueError:\s*[^\n]+)",
            r"(RuntimeError:\s*[^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                message = re.sub(r"\s+", " ", match.group(1)).strip()
                return message.replace("'", "")
        if test_result.failed_tests:
            return "|".join(test_result.failed_tests)
        return re.sub(r"\s+", " ", output[:240]).strip()

    @staticmethod
    def _extract_failed_files(output: str) -> list[str]:
        paths: list[str] = []
        suffixes = r"py|js|jsx|ts|tsx|json|ya?ml|toml|md|css|html|txt"
        for match in re.finditer(rf"([A-Za-z]:\\[^\n:]+?\.(?:{suffixes}))", output):
            path = match.group(1)
            if path not in paths:
                paths.append(path)
        for match in re.finditer(rf"\b([\w./-]+\.(?:{suffixes}))(?::(?:\d+))?", output):
            path = match.group(1)
            if path not in paths:
                paths.append(path)
        return paths[:8]

    @staticmethod
    def _import_error_root_cause(output: str) -> str:
        match = re.search(
            r"ImportError:\s*cannot import name '([^']+)' from '([^']+)'", output
        )
        if match:
            symbol, module = match.groups()
            return (
                f"{symbol} is imported from {module}, but that module does not "
                "export the requested symbol."
            )
        match = re.search(r"(ImportError:\s*[^\n]+)", output)
        if match:
            return match.group(1).strip()
        return "A Python import failed because the generated module wiring is inconsistent."

    @staticmethod
    def _looks_like_frontend_build_error(output: str) -> bool:
        lowered = output.lower()
        return any(
            marker in lowered
            for marker in (
                "npm err!",
                "vite",
                "webpack",
                "eslint",
                "typescript",
                "tsc",
            )
        )

    @staticmethod
    def _synthetic_test_result(*, success: bool, output: str) -> TestRunResult:
        return TestRunResult(
            success=success,
            command=[],
            failed_tests=[],
            output_excerpt=output,
            exit_code=0 if success else 1,
        )

    def _run_implementation_tasks(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
    ) -> list[ImplementationResult]:
        results: list[ImplementationResult] = []
        summaries: list[str] = []
        for task in self._prioritize_project_setup_tasks(
            self._tasks_for_roles(task_graph, self.IMPLEMENTATION_TASK_ROLES)
        ):
            objective = f"Implement planned task {task.id}: {task.title}"
            if self._is_project_setup_task(task):
                implementation = self._run_project_setup_scaffold(record, task)
            else:
                implementation_role = "scaffold" if task.role == "scaffold" else "implementer"
                implementation = self._run_structured_role(
                    record,
                    implementation_role,
                    ImplementationResult,
                    objective,
                    task=task,
                    logs=summaries,
                )
                self._record_task_output(record, "implementation_tasks", task, implementation)
            results.append(implementation)
            summaries.append(f"{task.id}: {implementation.summary}")
            if not self._implementation_allows_progress(record, task, implementation):
                return results
            if not self._is_project_setup_task(task):
                self._apply_patches(record, implementation_role, implementation.patches)
            if self._should_pause_for_recovery(record):
                return results
        if results:
            return results
        if task_graph.tasks:
            return []
        implementation = self._run_structured_role(
            record,
            "implementer",
            ImplementationResult,
            "Implement the planned feature",
            task=self._choose_primary_task(task_graph),
        )
        self._record_task_output(record, "implementation_tasks", None, implementation)
        if not self._implementation_allows_progress(record, None, implementation):
            return [implementation]
        self._apply_patches(record, "implementer", implementation.patches)
        return [implementation]

    def _run_test_writer_tasks(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
        primary_task: PlannedTask | None,
        implementation_results: list[ImplementationResult],
    ) -> list[TestWriterResult]:
        results: list[TestWriterResult] = []
        logs = [f"implementation: {item.summary}" for item in implementation_results]
        test_tasks = self._tasks_for_roles(task_graph, self.TEST_TASK_ROLES)
        if not test_tasks:
            test_tasks = [primary_task] if primary_task is not None else []
        if not test_tasks:
            test_writer = self._run_structured_role(
                record,
                "test_writer",
                TestWriterResult,
                "Add tests for the implementation",
                logs=logs,
            )
            self._record_task_output(record, "test_writer_tasks", None, test_writer)
            if not self._test_writer_allows_progress(record, None, test_writer):
                return [test_writer]
            self._apply_patches(record, "test_writer", test_writer.patches)
            return [test_writer]
        for task in test_tasks:
            test_writer = self._run_test_writer_task(
                record,
                task,
                implementation_results,
                results,
            )
            results.append(test_writer)
            logs.append(f"{task.id}: {test_writer.summary}")
            if self._should_pause_for_recovery(record):
                return results
        return results

    def _implementation_allows_progress(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        implementation: ImplementationResult,
    ) -> bool:
        if implementation.status == ImplementationStatus.IMPLEMENTED:
            return True
        if self._has_pending_recovery_plan(record):
            self.store.update(record)
            return False
        task_id = task.id if task is not None else "unplanned"
        if implementation.status == ImplementationStatus.BLOCKED:
            self._recover_record(record, error=f"implementation_blocked:{task_id}")
        else:
            self._recover_record(record, error=f"implementation_failed:{task_id}")
        self.store.update(record)
        return False

    def _test_writer_allows_progress(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        test_writer: TestWriterResult,
    ) -> bool:
        if test_writer.status == TestWriterStatus.TESTS_WRITTEN:
            return True
        task_id = task.id if task is not None else "unplanned"
        if test_writer.status == TestWriterStatus.BLOCKED:
            self._recover_record(record, error=f"test_writer_blocked:{task_id}")
        else:
            self._recover_record(record, error=f"test_writer_failed:{task_id}")
        self.store.update(record)
        return False

    def _fixer_allows_progress(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        fix: FixResult,
    ) -> bool:
        if fix.status == FixStatus.FIXED:
            return True
        task_id = task.id if task is not None else "unplanned"
        if fix.status == FixStatus.STUCK:
            self._recover_record(record, error=f"fixer_stuck:{task_id}")
        else:
            self._recover_record(record, error=f"fixer_failed:{task_id}")
        self.store.update(record)
        return False

    def _record_task_output(
        self,
        record: JobRecord,
        output_key: str,
        task: PlannedTask | None,
        result: Any,
    ) -> None:
        task_outputs = record.outputs.setdefault(output_key, [])
        if isinstance(task_outputs, list):
            task_outputs.append(
                {
                    "task": task.model_dump() if task is not None else None,
                    "result": result.model_dump(),
                }
            )
            self.store.update(record)

    def _build_stage_change_summary(
        self,
        implementation: ImplementationResult | None,
        test_writer_results: list[TestWriterResult],
    ) -> dict[str, Any]:
        implementation_files: list[str] = []
        implementation_patch_count = 0
        if implementation is not None:
            implementation_files = self._unique_paths(
                [
                    *implementation.changed_files,
                    *[patch.path for patch in implementation.patches],
                ]
            )
            implementation_patch_count = len(implementation.patches)
        test_writer_files = self._unique_paths(
            [
                path
                for result in test_writer_results
                for path in [
                    *result.changed_files,
                    *[patch.path for patch in result.patches],
                ]
            ]
        )
        test_files = [
            path
            for path in test_writer_files
            if self._looks_like_test_path(path)
        ]
        test_writer_patch_count = sum(
            len(result.patches) for result in test_writer_results
        )
        test_patch_count = sum(
            1
            for result in test_writer_results
            for patch in result.patches
            if self._looks_like_test_path(patch.path)
        )
        changed_files = self._unique_paths([*implementation_files, *test_writer_files])
        return {
            "changed_files": changed_files,
            "implementation_files": implementation_files,
            "test_files": test_files,
            "test_writer_files": test_writer_files,
            "implementation_patch_count": implementation_patch_count,
            "test_patch_count": test_patch_count,
            "test_writer_patch_count": test_writer_patch_count,
            "patch_count": implementation_patch_count + test_writer_patch_count,
        }

    @staticmethod
    def _load_recorded_task_results(
        record: JobRecord,
        output_key: str,
        response_model: type,
        allowed_result_statuses: set[str] | None = None,
    ) -> list[Any]:
        raw_items = record.outputs.get(output_key, [])
        if not isinstance(raw_items, list):
            return []
        results: list[Any] = []
        for item in raw_items:
            if not isinstance(item, dict) or "result" not in item:
                continue
            result = item["result"]
            if allowed_result_statuses is not None:
                if not isinstance(result, dict) or result.get("status") not in allowed_result_statuses:
                    continue
            results.append(response_model.model_validate(result))
        return results

    def _mark_task_completed(self, record: JobRecord, task_id: str) -> None:
        if task_id not in record.completed_task_ids:
            record.completed_task_ids.append(task_id)
            self.store.update(record)

    def _mark_tasks_completed(self, record: JobRecord, task_ids: list[str]) -> None:
        for task_id in task_ids:
            if task_id not in record.completed_task_ids:
                record.completed_task_ids.append(task_id)
        self.store.update(record)

    @staticmethod
    def _recorded_task_ids(
        record: JobRecord,
        output_key: str,
        allowed_result_statuses: set[str] | None = None,
    ) -> set[str]:
        raw_items = record.outputs.get(output_key, [])
        if not isinstance(raw_items, list):
            return set()
        task_ids: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            if allowed_result_statuses is not None:
                if not isinstance(result, dict) or result.get("status") not in allowed_result_statuses:
                    continue
            task = item.get("task")
            if isinstance(task, dict) and isinstance(task.get("id"), str):
                task_ids.add(task["id"])
        return task_ids

    def _record_stage_checkpoint(
        self,
        record: JobRecord,
        stage_result: dict[str, Any],
    ) -> None:
        self._annotate_stage_status_for_recovery(record, stage_result)
        stages = record.outputs.setdefault("autonomous_stages", [])
        if isinstance(stages, list):
            stages.append(stage_result)
        test_run = stage_result.get("test_run")
        record.checkpoints.append(
            {
                "kind": "autonomous_stage",
                "stage": stage_result.get("stage"),
                "task_id": (stage_result.get("task") or {}).get("id"),
                "test_success": test_run.get("success") if isinstance(test_run, dict) else None,
            }
        )
        self._recover_failed_stage_if_needed(record, stage_result)
        self.store.update(record)

    def _recover_failed_stage_if_needed(
        self,
        record: JobRecord,
        stage_result: dict[str, Any],
    ) -> None:
        if stage_result.get("status") != "failed_for_recovery":
            return
        if self._has_pending_recovery_plan(record) or self._is_recoverable_status(record.status):
            return
        task = stage_result.get("task")
        task_id = (
            task.get("id")
            if isinstance(task, dict) and isinstance(task.get("id"), str)
            else "unknown"
        )
        failure_reason = str(stage_result.get("failure_reason") or "stage_failed")
        runtime_state = {
            "failed_stage": stage_result.get("stage"),
            "failed_task_id": task_id,
            "stage_failure_reason": failure_reason,
        }
        for key in ("missing_artifacts", "invalid_artifacts"):
            value = stage_result.get(key)
            if isinstance(value, list):
                runtime_state[key] = value
        if isinstance(task, dict):
            for key in ("required_artifacts", "target_files"):
                value = task.get(key)
                if isinstance(value, list):
                    runtime_state[key] = value
        record.outputs["failed_stage"] = stage_result.get("stage")
        self._recover_record(
            record,
            error=f"{failure_reason}:stage:{task_id}",
            runtime_state=runtime_state,
        )

    def _annotate_stage_status_for_recovery(
        self,
        record: JobRecord,
        stage_result: dict[str, Any],
    ) -> None:
        task = stage_result.get("task")
        task_role = task.get("role") if isinstance(task, dict) else None
        change_summary = stage_result.get("change_summary")
        if not isinstance(change_summary, dict):
            change_summary = {}
        implementation = stage_result.get("implementation")
        if (
            isinstance(implementation, dict)
            and task_role in self.IMPLEMENTATION_TASK_ROLES
            and int(change_summary.get("implementation_patch_count") or 0) == 0
            and not change_summary.get("implementation_files")
        ):
            stage_result["status"] = "failed_for_recovery"
            stage_result["failure_reason"] = "implementation_produced_no_changes"
        missing_artifacts = self._missing_artifacts_for_stage(record, task)
        if missing_artifacts:
            invalid_artifacts = invalid_artifact_paths(missing_artifacts)
            stage_result["status"] = "failed_for_recovery"
            stage_result["failure_reason"] = "required_artifacts_missing"
            stage_result["missing_artifacts"] = missing_artifacts
            if invalid_artifacts:
                stage_result["invalid_artifacts"] = invalid_artifacts
        review = stage_result.get("stage_review")
        if isinstance(review, dict) and review.get("decision") in {
            ReviewDecision.REJECT.value,
            ReviewDecision.REQUEST_CHANGES.value,
            "reject",
            "request_changes",
        }:
            stage_result["status"] = "failed_for_recovery"
            stage_result["failure_reason"] = "review_rejected"
        if "status" not in stage_result:
            test_run = stage_result.get("test_run")
            if isinstance(test_run, dict) and test_run.get("success") is True:
                stage_result["status"] = "passed"
            else:
                stage_result["status"] = "failed_for_recovery"
                stage_result["failure_reason"] = "tests_failed"

    def _missing_artifacts_for_stage(
        self,
        record: JobRecord,
        task: Any,
    ) -> list[str]:
        if not isinstance(task, dict):
            return []
        artifacts = self._unique_paths(
            [
                str(item)
                for key in ("required_artifacts", "target_files")
                for item in task.get(key, [])
                if str(item).strip()
            ]
        )
        if not artifacts:
            return []
        root = self._workspace_root(record)
        return [
            artifact
            for artifact in artifacts
            if not artifact_path_exists(artifact, workspace_root=root)
        ]

    def _validate_completion_integrity(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
        test_result: TestRunResult,
    ) -> bool:
        record.outputs["task_graph"] = task_graph.model_dump()
        record.outputs["test_run"] = test_result.model_dump()
        if (
            not self._constraint_flag(record, "require_completion_integrity")
            and not self._constraint_flag(record, "require_test_evidence")
            and not self._constraint_flag(record, "require_stage_test_patches")
        ):
            return True
        report = self._build_completion_integrity_report(
            record,
            task_graph,
            test_result,
            require_completion_integrity=self._constraint_flag(
                record,
                "require_completion_integrity",
            ),
            require_test_evidence=self._constraint_flag(record, "require_test_evidence"),
            require_stage_test_patches=self._constraint_flag(
                record,
                "require_stage_test_patches",
            ),
        )
        record.outputs["completion_integrity"] = report
        dod = self.completion_verifier.verify(record)
        if not dod.passed:
            report["passed"] = False
            report["failure_reasons"].extend(dod.missing_evidence)
            report["unresolved_findings"] = dod.unresolved_findings
        if report["passed"]:
            self.store.update(record)
            return True
        runtime_state = self._completion_integrity_recovery_state(
            record,
            report["failure_reasons"],
        )
        self._recover_record(
            record,
            error="completion_integrity_failed:" + ",".join(report["failure_reasons"]),
            runtime_state=runtime_state,
        )
        return False

    @staticmethod
    def _completion_integrity_recovery_state(
        record: JobRecord,
        failure_reasons: list[str],
    ) -> dict[str, Any]:
        runtime_state = dict(record.runtime_state)
        parsed: dict[str, list[str]] = {
            "failed_stage_ids": [],
            "missing_task_ids": [],
            "missing_stage_test_patch_stage_ids": [],
            "required_artifacts": [],
            "target_files": [],
            "missing_artifacts": [],
            "non_file_artifacts": [],
            "invalid_artifacts": [],
            "empty_artifacts": [],
        }
        for key in (
            *parsed.keys(),
            "completion_integrity_failure_reasons",
            "failed_stages",
            "stages_missing_test_patches",
        ):
            runtime_state.pop(key, None)
        prefix_map = {
            "required_artifact_missing": ("required_artifacts", "missing_artifacts"),
            "target_file_missing": ("target_files", "missing_artifacts"),
            "required_artifact_non_file": ("required_artifacts", "non_file_artifacts"),
            "target_file_non_file": ("target_files", "non_file_artifacts"),
            "required_artifact_invalid": ("required_artifacts", "invalid_artifacts"),
            "target_file_invalid": ("target_files", "invalid_artifacts"),
            "required_artifact_empty": ("required_artifacts", "empty_artifacts"),
            "target_file_empty": ("target_files", "empty_artifacts"),
        }
        for reason in failure_reasons:
            if not isinstance(reason, str) or ":" not in reason:
                continue
            prefix, artifact = reason.split(":", 1)
            artifact = artifact.strip()
            if prefix == "missing_tasks":
                parsed["missing_task_ids"].extend(
                    item.strip() for item in artifact.split("|") if item.strip()
                )
                continue
            if prefix == "missing_stage_test_patches":
                parsed["missing_stage_test_patch_stage_ids"].extend(
                    item.strip() for item in artifact.split("|") if item.strip()
                )
                continue
            if prefix == "failed_stages":
                parsed["failed_stage_ids"].extend(
                    item.strip() for item in artifact.split("|") if item.strip()
                )
                continue
            if not artifact or prefix not in prefix_map:
                continue
            owner_key, evidence_key = prefix_map[prefix]
            parsed[owner_key].append(artifact)
            parsed[evidence_key].append(artifact)

        for key, values in parsed.items():
            deduped = list(dict.fromkeys(value for value in values if value.strip()))
            if deduped:
                runtime_state[key] = deduped
        completion_report = record.outputs.get("completion_integrity")
        if isinstance(completion_report, dict):
            for key in ("failed_stages", "stages_missing_test_patches"):
                value = completion_report.get(key)
                if isinstance(value, list) and value:
                    runtime_state[key] = value
        if failure_reasons:
            runtime_state["completion_integrity_failure_reasons"] = [
                str(reason) for reason in failure_reasons
            ]
        return runtime_state

    @staticmethod
    def _build_completion_integrity_report(
        record: JobRecord,
        task_graph: TaskGraph,
        test_result: TestRunResult,
        *,
        require_completion_integrity: bool,
        require_test_evidence: bool,
        require_stage_test_patches: bool,
    ) -> dict[str, Any]:
        planned_task_ids = [task.id for task in task_graph.tasks if task.id]
        completed_ids = list(record.completed_task_ids)
        completed_set = set(completed_ids)
        missing_task_ids = [task_id for task_id in planned_task_ids if task_id not in completed_set]
        missing_test_evidence = (
            require_test_evidence
            and (test_result.executed_test_count is None or test_result.executed_test_count < 1)
        )
        failure_reasons: list[str] = []
        if require_completion_integrity and missing_task_ids:
            failure_reasons.append("missing_tasks:" + "|".join(missing_task_ids))
        if not test_result.success:
            failure_reasons.append("test_failed")
        if missing_test_evidence:
            failure_reasons.append("missing_test_evidence")
        stages_missing_test_patches = JobRunner._stages_missing_test_patches(record)
        if require_stage_test_patches and stages_missing_test_patches:
            failure_reasons.append(
                "missing_stage_test_patches:"
                + "|".join(str(stage["stage"]) for stage in stages_missing_test_patches)
            )
        failed_stages = JobRunner._failed_autonomous_stages(record)
        if require_completion_integrity and failed_stages:
            failure_reasons.append(
                "failed_stages:"
                + "|".join(str(stage["stage"]) for stage in failed_stages)
            )
        return {
            "passed": (
                (not require_completion_integrity or not missing_task_ids)
                and (not require_test_evidence or not missing_test_evidence)
                and (not require_stage_test_patches or not stages_missing_test_patches)
                and (not require_completion_integrity or not failed_stages)
                and test_result.success
            ),
            "failure_reasons": failure_reasons,
            "require_completion_integrity": require_completion_integrity,
            "require_test_evidence": require_test_evidence,
            "require_stage_test_patches": require_stage_test_patches,
            "planned_task_count": len(planned_task_ids),
            "completed_task_count": len([task_id for task_id in planned_task_ids if task_id in completed_set]),
            "planned_task_ids": planned_task_ids,
            "completed_task_ids": completed_ids,
            "missing_task_ids": missing_task_ids,
            "test_success": test_result.success,
            "executed_test_count": test_result.executed_test_count,
            "stages_missing_test_patches": stages_missing_test_patches,
            "failed_stages": failed_stages,
        }

    @staticmethod
    def _failed_autonomous_stages(record: JobRecord) -> list[dict[str, Any]]:
        stages = record.outputs.get("autonomous_stages", [])
        if not isinstance(stages, list):
            return []
        failed: list[dict[str, Any]] = []
        later_passed_task_ids: set[str] = set()
        for stage in reversed(stages):
            if not isinstance(stage, dict):
                continue
            task = stage.get("task")
            task_id = task.get("id") if isinstance(task, dict) else None
            status = str(stage.get("status") or "").strip().lower()
            test_run = stage.get("test_run")
            test_success = test_run.get("success") if isinstance(test_run, dict) else None
            post_review_test_run = stage.get("post_review_test_run")
            post_review_success = (
                post_review_test_run.get("success")
                if isinstance(post_review_test_run, dict)
                else None
            )
            failed_status = status in {"failed", "failed_for_recovery"}
            if (
                not failed_status
                and (
                    status == "passed"
                    or (test_success is True and post_review_success is not False)
                )
            ):
                if isinstance(task_id, str):
                    later_passed_task_ids.add(task_id)
                continue
            stage_failed = (
                failed_status
                or test_success is False
                or post_review_success is False
            )
            if not stage_failed:
                continue
            if isinstance(task_id, str) and task_id in later_passed_task_ids:
                continue
            failure_reason = stage.get("failure_reason")
            if not isinstance(failure_reason, str) or not failure_reason.strip():
                if post_review_success is False:
                    failure_reason = "post_review_tests_failed"
                elif test_success is False:
                    failure_reason = "tests_failed"
                else:
                    failure_reason = "stage_failed"
            failed.append(
                {
                    "stage": stage.get("stage"),
                    "task_id": task_id,
                    "failure_reason": failure_reason,
                }
            )
        return list(reversed(failed))

    @staticmethod
    def _stages_missing_test_patches(record: JobRecord) -> list[dict[str, Any]]:
        stages = record.outputs.get("autonomous_stages", [])
        if not isinstance(stages, list):
            return []
        missing: list[dict[str, Any]] = []
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            summary = stage.get("change_summary")
            task = stage.get("task")
            if not isinstance(summary, dict):
                continue
            implementation_patch_count = summary.get("implementation_patch_count")
            test_patch_count = summary.get("test_patch_count")
            if not isinstance(implementation_patch_count, int):
                implementation_patch_count = 0
            if not isinstance(test_patch_count, int):
                test_patch_count = 0
            if implementation_patch_count > 0 and test_patch_count < 1:
                missing.append(
                    {
                        "stage": stage.get("stage"),
                        "task_id": task.get("id") if isinstance(task, dict) else None,
                        "implementation_patch_count": implementation_patch_count,
                        "test_patch_count": test_patch_count,
                    }
                )
        return missing

    @staticmethod
    def _combine_implementation_results(
        results: list[ImplementationResult],
    ) -> ImplementationResult:
        if not results:
            return ImplementationResult(
                status=ImplementationStatus.IMPLEMENTED,
                summary="No implementation tasks were generated.",
            )
        status = ImplementationStatus.IMPLEMENTED
        if any(item.status == ImplementationStatus.FAILED for item in results):
            status = ImplementationStatus.FAILED
        elif any(item.status == ImplementationStatus.BLOCKED for item in results):
            status = ImplementationStatus.BLOCKED
        changed_files = JobRunner._unique_paths(
            [path for item in results for path in item.changed_files]
            + [patch.path for item in results for patch in item.patches]
        )
        return ImplementationResult(
            status=status,
            summary="\n".join(item.summary for item in results if item.summary),
            changed_files=changed_files,
            patches=[patch for item in results for patch in item.patches],
            risks=[risk for item in results for risk in item.risks],
        )

    @staticmethod
    def _combine_test_writer_results(results: list[TestWriterResult]) -> TestWriterResult:
        if not results:
            return TestWriterResult(summary="No test writer tasks were generated.")
        changed_files = JobRunner._unique_paths(
            [path for item in results for path in item.changed_files]
            + [patch.path for item in results for patch in item.patches]
        )
        return TestWriterResult(
            summary="\n".join(item.summary for item in results if item.summary),
            changed_files=changed_files,
            patches=[patch for item in results for patch in item.patches],
            test_strategy=[strategy for item in results for strategy in item.test_strategy],
        )

    @staticmethod
    def _unique_paths(paths: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            if path and path not in seen:
                unique.append(path)
                seen.add(path)
        return unique

    @staticmethod
    def _valid_unique_artifact_paths(paths: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            normalized_paths = valid_artifact_paths([path])
            if not normalized_paths:
                continue
            normalized = next(iter(normalized_paths))
            if normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        return unique

    @classmethod
    def _valid_unique_planning_artifact_paths(cls, paths: list[str]) -> list[str]:
        return cls._valid_unique_artifact_paths(cls._meaningful_artifact_items(paths))

    @classmethod
    def _invalid_planning_artifact_paths(cls, paths: list[str]) -> list[str]:
        invalid = [
            *invalid_artifact_paths(paths),
            *cls._placeholder_artifact_paths(paths),
        ]
        return cls._unique_paths(invalid)

    @classmethod
    def _meaningful_artifact_items(cls, paths: list[str]) -> list[str]:
        return [
            item
            for item in cls._non_empty_items(paths)
            if not cls._looks_like_placeholder_artifact_path(item)
        ]

    @classmethod
    def _placeholder_artifact_paths(cls, paths: list[str]) -> list[str]:
        return [
            item
            for item in cls._non_empty_items(paths)
            if cls._looks_like_placeholder_artifact_path(item)
        ]

    @staticmethod
    def _looks_like_placeholder_artifact_path(path: str) -> bool:
        value = str(path).replace("\\", "/").strip()
        if not value:
            return False
        if JobRunner._looks_like_placeholder_prd_item(value):
            return True
        name = value.rsplit("/", 1)[-1].strip()
        if not name:
            return False
        stem = name.split(".", 1)[0]
        compact = re.sub(r"[^a-z0-9]+", "", stem.lower())
        if compact in {
            "fixme",
            "placeholder",
            "tbd",
            "tobedecided",
            "tobedefined",
            "tobedetermined",
            "unknown",
            "unspecified",
        }:
            return True
        parts = set(re.findall(r"[a-z0-9]+", stem.lower()))
        strong_placeholder_parts = {
            "fixme",
            "placeholder",
            "tbd",
            "unspecified",
        }
        return bool(parts & strong_placeholder_parts)

    @staticmethod
    def _constraints(record: JobRecord) -> dict[str, Any]:
        constraints = record.spec.metadata.get("constraints", {})
        return constraints if isinstance(constraints, dict) else {}

    def _recovery_missing_target_file(self, record: JobRecord) -> str:
        constraints = self._constraints(record)
        for source in (
            constraints.get("missing_target_file"),
            record.runtime_state.get("missing_target_file"),
            record.runtime_state.get("failed_patch_path"),
        ):
            if isinstance(source, str) and source.strip():
                return self._normalize_context_path(source, record) or source.strip()
        return ""

    def _constraint_flag(self, record: JobRecord, key: str) -> bool:
        return bool(self._constraints(record).get(key, False))

    def _constraint_int(self, record: JobRecord, key: str, default: int) -> int:
        value = self._constraints(record).get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _constraint_float(self, record: JobRecord, key: str, default: float) -> float:
        value = self._constraints(record).get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _effective_model_timeout_seconds(
        self,
        record: JobRecord,
        base_timeout_seconds: float,
    ) -> float:
        deadline_epoch = self._constraint_float(
            record,
            "model_timeout_deadline_epoch",
            0.0,
        )
        if deadline_epoch <= 0:
            return base_timeout_seconds
        remaining_seconds = deadline_epoch - datetime.now(timezone.utc).timestamp()
        if remaining_seconds <= 0:
            raise AdapterError(
                "model runtime deadline exceeded before the next model call",
                code="timeout",
            )
        if base_timeout_seconds <= 0:
            return remaining_seconds
        return min(base_timeout_seconds, remaining_seconds)

    def _autonomous_stage_limit_reached(
        self,
        record: JobRecord,
        stage_results: list[dict[str, Any]],
    ) -> bool:
        max_stages = self._constraint_int(record, "max_autonomous_stages", 0)
        if not max_stages or len(stage_results) < max_stages:
            return False
        if not self._constraint_flag(record, "max_autonomous_stages_hard"):
            constraints = record.spec.metadata.setdefault("constraints", {})
            if isinstance(constraints, dict):
                bumped = max(max_stages + 16, max_stages * 2, 64)
                constraints["max_autonomous_stages"] = bumped
                record.outputs["autonomous_stage_limit"] = {
                    "max_autonomous_stages": max_stages,
                    "bumped_to": bumped,
                    "completed_stage_count": len(stage_results),
                    "recovery_action": "auto_bump_stage_limit",
                }
                self.store.update(record)
                return False
        self._recover_record(record, error="autonomous_stage_limit_reached")
        record.outputs["autonomous_stage_limit"] = {
            "max_autonomous_stages": max_stages,
            "completed_stage_count": len(stage_results),
        }
        self.store.update(record)
        return True

    def _read_memory(self, role: str) -> list[str]:
        if not self.policy.is_tool_allowed(role, "memory_server.read_memory"):
            return []
        payload = self._call_tool(role, "memory_server.read_memory", limit=5)
        return [str(item["value"]) for item in payload.get("entries", [])]

    def _write_memory_item(self, record: JobRecord, role: str, key: str, value: str) -> None:
        if not self.policy.is_tool_allowed(role, "memory_server.write_memory"):
            return
        self._call_tool(
            role,
            "memory_server.write_memory",
            uri=f"memory://{record.job_id}/{key}",
            content=value,
        )

    def _write_memory_entries(self, record: JobRecord, summary: SummaryResult) -> None:
        if not self.policy.is_tool_allowed("summarizer", "memory_server.write_memory"):
            return
        for index, entry in enumerate(summary.memory_entries):
            self._call_tool(
                "summarizer",
                "memory_server.write_memory",
                uri=f"memory://{record.job_id}/summary_{index}",
                content=entry,
            )

    def _release(self, record: JobRecord, release: ReleaseResult) -> None:
        self.policy.assert_release_commit_allowed("release_manager")
        self.policy.assert_branch_allowed(record.spec.target_branch)
        commit_message = (
            release.commit_message
            if release.commit_message.startswith("acos:")
            else f"acos: {release.commit_message}"
        )
        self._call_tool(
            "release_manager",
            "git_server.commit",
            message=commit_message,
            branch=record.spec.target_branch,
        )
        self._call_tool(
            "release_manager",
            "notify_server.send_notification",
            body=release.notify_message,
        )

    def _call_tool(self, role: str, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        self.policy.assert_tool_allowed(role, tool_name)
        result = self.router.call(tool_name, **kwargs)
        status = "success" if result.ok else "failed"
        event = self.audit.tool_event(
            role=role,
            tool_name=tool_name,
            input_payload=kwargs,
            output_payload=result.data,
            status=status,
        )
        if self._active_record is not None:
            self._active_record.audit_events.append(event)
        if result.ok:
            return result.data
        raise RuntimeError(result.error or f"tool call failed: {tool_name}")

    def _prepare_branch(self, record: JobRecord) -> None:
        result = self.router.call("git_server.create_branch", branch=record.spec.target_branch)
        event = self.audit.tool_event(
            role="orchestrator",
            tool_name="git_server.create_branch",
            input_payload={"branch": record.spec.target_branch},
            output_payload=result.data,
            status="success" if result.ok else "failed",
        )
        record.audit_events.append(event)
        if not result.ok:
            raise RuntimeError(result.error or "failed to create branch")

    def _run_review_cycle(
        self, record: JobRecord, primary_task: PlannedTask | None
    ) -> tuple[ReviewResult, SecurityReviewResult]:
        attempts = 0
        while True:
            review = self._run_structured_role(
                record,
                "reviewer",
                ReviewResult,
                "Review the changed code and tests",
                task=primary_task,
            )
            security_review = self._run_structured_role(
                record,
                "security_reviewer",
                SecurityReviewResult,
                "Review the changes for security risks",
                task=primary_task,
                security_sensitive=True,
            )
            try:
                ensure_reviews_pass(review, security_review)
                return review, security_review
            except QualityGateError as exc:
                attempts += 1
                if attempts >= self.max_attempts_per_task:
                    raise QualityGateError(
                        f"acceptance_review_max_attempts_exceeded:{exc}"
                    ) from exc
                findings = [
                    review.summary,
                    security_review.summary,
                    *[item.description for item in review.findings],
                    *[item.description for item in security_review.findings],
                ]
                fix = self._run_structured_role(
                    record,
                    "fixer",
                    FixResult,
                    "Address review findings without weakening tests",
                    task=primary_task,
                    logs=findings,
                )
                if not self._fixer_allows_progress(record, primary_task, fix):
                    return review, security_review
                ensure_fixer_safe(
                    fix.patches,
                    workspace_root=self._workspace_root(record),
                )
                self._apply_patches(record, "fixer", fix.patches)

    def _run_stage_review_gate(
        self,
        record: JobRecord,
        task: PlannedTask | None,
    ) -> dict[str, Any] | None:
        if not self._constraint_flag(record, "stage_review"):
            return None
        review, security_review = self._run_review_cycle(record, task)
        return {
            "review": review.model_dump(),
            "security_review": security_review.model_dump(),
        }

    @staticmethod
    def _choose_primary_task(task_graph: TaskGraph) -> PlannedTask | None:
        return task_graph.tasks[0] if task_graph.tasks else None

    def _tasks_for_roles(self, task_graph: TaskGraph, roles: set[str]) -> list[PlannedTask]:
        tasks = [task for task in task_graph.tasks if task.role in roles]
        return self._order_tasks_by_dependencies(tasks)

    def _prioritize_project_setup_tasks(
        self,
        tasks: list[PlannedTask],
    ) -> list[PlannedTask]:
        return [
            task
            for _index, task in sorted(
                enumerate(tasks),
                key=lambda item: (
                    0 if self._is_project_setup_task(item[1]) else 1,
                    item[0],
                ),
            )
        ]

    @staticmethod
    def _order_tasks_by_dependencies(tasks: list[PlannedTask]) -> list[PlannedTask]:
        if len(tasks) < 2:
            return tasks
        remaining = list(tasks)
        remaining_ids = {task.id for task in remaining}
        completed: set[str] = set()
        ordered: list[PlannedTask] = []
        while remaining:
            progressed = False
            for task in list(remaining):
                local_dependencies = [
                    dependency for dependency in task.depends_on if dependency in remaining_ids
                ]
                if all(dependency in completed for dependency in local_dependencies):
                    ordered.append(task)
                    completed.add(task.id)
                    remaining.remove(task)
                    progressed = True
            if not progressed:
                ordered.extend(remaining)
                break
        return ordered


def build_default_runner(
    config_dir: str | Path = "configs",
    workspace_root: str | Path = ".",
    memory_db_path: str | Path | None = None,
    store: InMemoryJobStore | None = None,
    allow_mock_fallback: bool = False,
) -> tuple[JobRunner, FakeMCPEnvironment]:
    """Build a JobRunner wired to the local config directory and fake MCP tools."""
    config_path = Path(config_dir)
    workspace_path = Path(workspace_root).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    if memory_db_path is None:
        workspace_hash = hashlib.sha256(str(workspace_path).encode("utf-8")).hexdigest()[:16]
        memory_db = config_path.parent / ".acos" / "memory" / f"{workspace_hash}.sqlite3"
    else:
        memory_db = Path(memory_db_path)
    memory_db.parent.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry.from_paths(
        provider_path=config_path / "model_providers.yaml",
        agents_path=config_path / "agents.yaml",
        routing_path=config_path / "model_routing.yaml",
    )
    if not allow_mock_fallback:
        _disable_mock_fallback_models(registry)
    policy = PolicyEngine.from_path(config_path / "policies.yaml")
    registry.validate_or_raise(policy=policy)
    env = FakeMCPEnvironment(
        workspace_root=workspace_path,
        memory_db_path=memory_db,
        workspace_policy=policy.build_workspace_policy(workspace_path),
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=env.build_router(),
        store=store,
    )
    return runner, env
