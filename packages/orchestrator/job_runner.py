"""ACOS job orchestration engine."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import yaml

from packages.agents.runner import AgentRunner
from packages.llm.budget import TokenBudgetPolicy, estimate_tokens
from packages.llm.client import LLMClient
from packages.llm.errors import AdapterError
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext, RoutingError
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.mcp_client.router import MCPRouter
from packages.memory.redaction import redact_text
from packages.orchestrator.checkpoint import CheckpointStore
from packages.orchestrator.approval import (
    ApprovalGateway,
    ApprovalRequiredError,
    SQLiteApprovalStore,
)
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.context_builder import ContextBuilder
from packages.orchestrator.execution_contracts import synthesize_job_metadata_from_prd
from packages.orchestrator.framework_profiles import (
    ResolvedFrameworkProfile,
    resolve_framework_profile,
)
from packages.orchestrator.framework_scaffolds import (
    ResolvedFrameworkScaffold,
    resolve_framework_scaffold,
)
from packages.orchestrator.job_store import InMemoryJobStore, JobStore, SQLiteJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.orchestrator.quality_gates import (
    QualityGateError,
    ensure_required_artifacts_assigned_to_tasks,
    ensure_required_artifacts_exist,
    ensure_fixer_safe,
    ensure_reviews_pass,
    ensure_task_required_artifacts_exist,
    ensure_task_target_files_exist,
    ensure_test_patch_quality,
)
from packages.orchestrator.runtime import ProviderUnavailableError, RuntimeManager
from packages.orchestrator.states import apply_transition
from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FixResult,
    ImplementationResult,
    PMReviewResult,
    PRD,
    ReleaseResult,
    ReviewResult,
    SecurityReviewResult,
    SummaryResult,
    TestRunResult,
    TestWriterResult,
)
from packages.schemas.approvals import PolicyAction
from packages.schemas.audit import AuditEvent
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import FixStatus, JobStatus, TaskStatus
from packages.schemas.models import ReviewDecision
from packages.schemas.runtime import RuntimeConfig, RuntimeHttpCheck, RuntimeIssueType
from packages.schemas.tasks import PlannedTask, TaskGraph, TaskRecord

T = TypeVar("T")
MISSING_MODULE_PATTERN = re.compile(r"No module named ['\"]([A-Za-z0-9_.-]+)['\"]")
ALLOWLISTED_DEPENDENCY_PACKAGES = {
    "django": "django",
    "fastapi": "fastapi",
    "flask": "flask",
    "jinja2": "jinja2",
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "uvicorn": "uvicorn",
}


class JobRunner:
    """Run ACOS jobs across explicit role phases."""

    def __init__(
        self,
        registry: ModelRegistry,
        policy: PolicyEngine,
        router: MCPRouter,
        store: JobStore | None = None,
        model_router: ModelRouter | None = None,
        agent_runner: AgentRunner | None = None,
        approval_gateway: ApprovalGateway | None = None,
        runtime_manager: RuntimeManager | None = None,
        checkpoint_store: CheckpointStore | None = None,
        token_budget_policy: TokenBudgetPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.router = router
        self.store = store or InMemoryJobStore()
        self.audit = AuditRecorder()
        self.token_budget_policy = token_budget_policy or (
            runtime_manager.config.token_budget
            if runtime_manager is not None
            else TokenBudgetPolicy()
        )
        self.context_builder = ContextBuilder(self.token_budget_policy)
        self.model_router = model_router or ModelRouter(
            registry,
            token_budget_policy=self.token_budget_policy,
        )
        self.llm_client = LLMClient(
            registry,
            self.model_router,
            token_budget_policy=self.token_budget_policy,
        )
        self.approval_gateway = approval_gateway
        self.agent_runner = agent_runner or AgentRunner(
            llm_client=self.llm_client,
            registry=registry,
            mcp_router=router,
            policy_engine=policy,
            audit_recorder=self.audit,
            token_budget_policy=self.token_budget_policy,
        )
        self.max_attempts_per_task = 3
        self.max_same_failure_repeats = 2
        self.max_steps_per_agent = 6
        self.runtime_manager = runtime_manager
        self.checkpoints = checkpoint_store or CheckpointStore(self.store)
        self._active_record: JobRecord | None = None

    def submit(self, spec: JobSpec) -> JobRecord:
        return self.store.create(spec, status=JobStatus.QUEUED)

    def get(self, job_id: str) -> JobRecord:
        return self.store.get(job_id)

    def list_jobs(self, *, statuses: list[JobStatus] | None = None) -> list[JobRecord]:
        return self.store.list_jobs(statuses=statuses)

    def pause_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if self._is_terminal_status(record.status):
            return record
        record.status = JobStatus.PAUSED
        return self.store.update(record)

    def cancel_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if self._is_terminal_status(record.status):
            return record
        record.status = JobStatus.CANCELLED
        record.completed_at = record.updated_at
        return self.store.update(record)

    def get_events(self, job_id: str) -> list[Any]:
        return list(self.store.get(job_id).audit_events)

    def get_notifications(self, job_id: str) -> list[dict[str, Any]]:
        return self.store.list_notifications(job_id=job_id)

    def list_approvals(self, job_id: str | None = None) -> list[Any]:
        if self.approval_gateway is None:
            return []
        return self.approval_gateway.list_all(job_id=job_id)

    def run_job(self, spec: JobSpec) -> JobRecord:
        record = self.store.create(spec, status=JobStatus.QUEUED)
        return self._run_until_pause_or_done(record.job_id)

    def resume_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if record.pending_approval_id and self.approval_gateway is not None:
            approval = self.approval_gateway.get(record.pending_approval_id)
            if approval.status.value == "pending":
                return self.store.update(record)
            record.audit_events.append(
                self.audit.approval_event(
                    role=record.current_role or approval.requested_by,
                    action=approval.status.value,
                    approval=approval,
                )
            )
            if approval.status.value in {"rejected", "expired", "cancelled"}:
                record.status = JobStatus.BLOCKED
                record.last_error = redact_text(
                    approval.resolution_reason or f"approval_{approval.status}"
                )
                self._set_task_status(
                    record,
                    record.current_task_id,
                    TaskStatus.BLOCKED,
                )
                record.pending_approval_id = None
                record.runtime_state.pop("pending_operation", None)
                return self.store.update(record)
            resume_status = record.runtime_state.pop("resume_status", None)
            pending_operation = record.runtime_state.pop("pending_operation", None)
            if pending_operation is not None:
                record.runtime_state["approved_operation"] = {
                    **pending_operation,
                    "approval_id": approval.id,
                }
            record.pending_approval_id = None
            record.last_error = None
            self._set_task_status(
                record,
                record.current_task_id,
                TaskStatus.IN_PROGRESS,
            )
            if record.status == JobStatus.WAITING_APPROVAL and resume_status is not None:
                apply_transition(record, JobStatus(resume_status))
        if record.status in {
            JobStatus.WAITING_RUNTIME,
            JobStatus.PROVIDER_UNAVAILABLE,
            JobStatus.RETRYING_PROVIDER,
        }:
            record.status = JobStatus.RESUMING
            record.pending_runtime_issue_id = None
            record.runtime_error = None
            self.store.update(record)
        return self._run_until_pause_or_done(job_id)

    def run_next_step(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
            return record
        self._active_record = record
        try:
            if record.started_at is None:
                record.started_at = record.updated_at
            if record.status in {JobStatus.QUEUED, JobStatus.SUBMITTED, JobStatus.RESUMING, JobStatus.RECOVERING}:
                record.status = JobStatus.RUNNING
                self.store.update(record)
            if not self.checkpoints.has_completed(job_id=record.job_id, checkpoint_key="branch_prepared"):
                return self.run_prepare_branch_step(record)
            if "pm" not in record.outputs:
                return self.run_pm_step(record)
            if "architect" not in record.outputs or record.runtime_state.get("needs_architecture_revision"):
                return self.run_architect_step(record)
            if "planner" not in record.outputs or record.runtime_state.get("needs_plan_revision"):
                return self.run_planner_step(record)
            if (
                not self.checkpoints.has_completed(
                    job_id=record.job_id,
                    checkpoint_key="design_review_completed",
                )
                or record.runtime_state.get("needs_design_review")
            ):
                return self.run_design_review_step(record)
            task_graph = (
                TaskGraph.model_validate(record.outputs["planner"])
                if "planner" in record.outputs
                else TaskGraph.model_validate(record.outputs["task_graph"])
            )
            active_task = self._choose_active_task(task_graph)
            if active_task is not None:
                if record.runtime_state.get("needs_product_fix"):
                    return self.run_task_product_fix_step(record, active_task)
                if (
                    self._framework_scaffold(record) is not None
                    and not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:scaffold_completed",
                    )
                ):
                    return self.run_task_scaffold_step(record, active_task)
                if not self.checkpoints.has_completed(
                    job_id=record.job_id,
                    task_id=active_task.id,
                    checkpoint_key=f"task:{active_task.id}:implementer_completed",
                ):
                    return self.run_task_implementer_step(record, active_task)
                if not self.checkpoints.has_completed(
                    job_id=record.job_id,
                    task_id=active_task.id,
                    checkpoint_key=f"task:{active_task.id}:test_writer_completed",
                ):
                    return self.run_task_test_writer_step(record, active_task)
                if not self.checkpoints.has_completed(
                    job_id=record.job_id,
                    task_id=active_task.id,
                    checkpoint_key=f"task:{active_task.id}:review_completed",
                ) or record.runtime_state.get("needs_rereview"):
                    return self.run_task_review_step(record, active_task)
                if (
                    not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:tests_completed",
                    )
                    or record.runtime_state.get("needs_retest")
                ):
                    return self.run_task_test_step(record, active_task)
                test_result = TestRunResult.model_validate(record.outputs["test_run"])
                if not test_result.success:
                    if record.failure_count >= self.max_attempts_per_task:
                        record.status = JobStatus.STUCK
                        record.last_error = "max_attempts_exceeded"
                        return self.store.update(record)
                    return self.run_task_fixer_step(record, active_task, test_result)
                if (
                    not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:runtime_prepare_completed",
                    )
                    or record.runtime_state.get("needs_runtime_prepare")
                ):
                    return self.run_task_runtime_prepare_step(record, active_task)
                runtime_prepare = TestRunResult.model_validate(record.outputs["runtime_prepare"])
                if not runtime_prepare.success:
                    if record.failure_count >= self.max_attempts_per_task:
                        record.status = JobStatus.STUCK
                        record.last_error = "max_attempts_exceeded"
                        return self.store.update(record)
                    return self.run_task_fixer_step(record, active_task, runtime_prepare)
                if (
                    not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:runtime_smoke_completed",
                    )
                    or record.runtime_state.get("needs_runtime_smoke")
                ):
                    return self.run_task_runtime_smoke_step(record, active_task)
                runtime_smoke = TestRunResult.model_validate(record.outputs["runtime_smoke"])
                if not runtime_smoke.success:
                    if record.failure_count >= self.max_attempts_per_task:
                        record.status = JobStatus.STUCK
                        record.last_error = "max_attempts_exceeded"
                        return self.store.update(record)
                    return self.run_task_fixer_step(record, active_task, runtime_smoke)
                if (
                    not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:acceptance_checks_completed",
                    )
                    or record.runtime_state.get("needs_acceptance_checks")
                ):
                    return self.run_task_acceptance_checks_step(record, active_task)
                acceptance_checks = TestRunResult.model_validate(record.outputs["acceptance_checks"])
                if not acceptance_checks.success:
                    if record.failure_count >= self.max_attempts_per_task:
                        record.status = JobStatus.STUCK
                        record.last_error = "max_attempts_exceeded"
                        return self.store.update(record)
                    return self.run_task_fixer_step(record, active_task, acceptance_checks)
                if (
                    not self.checkpoints.has_completed(
                        job_id=record.job_id,
                        task_id=active_task.id,
                        checkpoint_key=f"task:{active_task.id}:acceptance_review_completed",
                    )
                    or record.runtime_state.get("needs_acceptance_review")
                ):
                    return self.run_task_acceptance_review_step(record, active_task)
            if not self.checkpoints.has_completed(
                job_id=record.job_id,
                checkpoint_key="final_quality_gates_completed",
                task_id=None,
            ):
                return self.run_final_quality_gates_step(record, None)
            if not self.checkpoints.has_completed(
                job_id=record.job_id,
                checkpoint_key="release_completed",
                task_id=None,
            ):
                return self.run_release_step(record, None)
            if record.status != JobStatus.DONE:
                record.status = JobStatus.DONE
                record.completed_at = record.updated_at
                self.store.update(record)
            return record
        except ApprovalRequiredError as exc:
            return self._pause_for_approval(record, exc)
        except ProviderUnavailableError as exc:
            return self._pause_for_runtime(record, exc)
        except RoutingError as exc:
            issue_type = self._classify_runtime_issue_message(str(exc))
            if issue_type is not None:
                current_model = self.registry.get_agent(record.current_role or "pm").primary_model
                provider_key = self.registry.get_model(current_model).provider
                return self._pause_for_runtime(
                    record,
                    ProviderUnavailableError(
                        provider_key=provider_key,
                        model_key=current_model,
                        issue_type=issue_type,
                        message=redact_text(str(exc)),
                    ),
                )
            record.status = JobStatus.FAILED
            record.last_error = redact_text(str(exc))
            return self.store.update(record)
        except AdapterError as exc:
            issue_type = self._classify_runtime_issue_type(exc.code)
            if issue_type is not None:
                current_model = self.registry.get_agent(record.current_role or "pm").primary_model
                provider_key = self.registry.get_model(current_model).provider
                return self._pause_for_runtime(
                    record,
                    ProviderUnavailableError(
                        provider_key=provider_key,
                        model_key=current_model,
                        issue_type=issue_type,
                        message=redact_text(str(exc)),
                    ),
                )
            record.status = JobStatus.FAILED
            record.last_error = redact_text(str(exc))
            return self.store.update(record)
        except QualityGateError as exc:
            record.status = JobStatus.BLOCKED
            record.last_error = redact_text(str(exc))
            self._set_task_status(record, record.current_task_id, TaskStatus.BLOCKED)
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            record.status = JobStatus.FAILED
            record.last_error = redact_text(str(exc))
            return self.store.update(record)
        finally:
            self._active_record = None

    def _run_until_pause_or_done(self, job_id: str) -> JobRecord:
        while True:
            record = self.run_next_step(job_id)
            if self._is_terminal_status(record.status) or self._is_waiting_status(record.status):
                return record

    @staticmethod
    def _is_terminal_status(status: JobStatus) -> bool:
        return status in {
            JobStatus.DONE,
            JobStatus.BLOCKED,
            JobStatus.STUCK,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    @staticmethod
    def _is_waiting_status(status: JobStatus) -> bool:
        return status in {
            JobStatus.WAITING_APPROVAL,
            JobStatus.WAITING_RUNTIME,
            JobStatus.PROVIDER_UNAVAILABLE,
            JobStatus.PAUSED,
        }

    @staticmethod
    def _classify_runtime_issue_type(error_code: str | None) -> RuntimeIssueType | None:
        mapping = {
            "timeout": RuntimeIssueType.TIMEOUT,
            "provider_error": RuntimeIssueType.PROVIDER_UNAVAILABLE,
            "auth_error": RuntimeIssueType.AUTH_ERROR,
            "model_not_found": RuntimeIssueType.MODEL_NOT_FOUND,
            "connection_error": RuntimeIssueType.CONNECTION_ERROR,
        }
        return mapping.get(error_code or "")

    @classmethod
    def _classify_runtime_issue_message(cls, message: str) -> RuntimeIssueType | None:
        lowered = message.lower()
        for token, issue_type in {
            "timeout": RuntimeIssueType.TIMEOUT,
            "provider_error": RuntimeIssueType.PROVIDER_UNAVAILABLE,
            "auth_error": RuntimeIssueType.AUTH_ERROR,
            "model_not_found": RuntimeIssueType.MODEL_NOT_FOUND,
            "connection_error": RuntimeIssueType.CONNECTION_ERROR,
        }.items():
            if token in lowered:
                return issue_type
        return None

    def _mark_step_started(
        self,
        *,
        record: JobRecord,
        checkpoint_key: str,
        step_name: str,
        task_id: str | None = None,
        phase: str | None = None,
    ) -> None:
        record.current_phase = phase or step_name
        record.current_task_id = task_id
        self.checkpoints.mark_started(
            job_id=record.job_id,
            task_id=task_id,
            checkpoint_key=checkpoint_key,
            step_name=step_name,
            idempotency_key=checkpoint_key,
        )
        self.store.update(record)

    def _mark_step_completed(
        self,
        *,
        record: JobRecord,
        checkpoint_key: str,
        step_name: str,
        task_id: str | None = None,
        result_json: dict[str, object] | None = None,
    ) -> JobRecord:
        self.checkpoints.mark_completed(
            job_id=record.job_id,
            task_id=task_id,
            checkpoint_key=checkpoint_key,
            step_name=step_name,
            idempotency_key=checkpoint_key,
            result_json=result_json,
        )
        return self.store.update(record)

    def run_prepare_branch_step(self, record: JobRecord) -> JobRecord:
        self._mark_step_started(
            record=record,
            checkpoint_key="branch_prepared",
            step_name="prepare_branch",
            phase="prepare_branch",
        )
        self._prepare_branch(record)
        return self._mark_step_completed(
            record=record,
            checkpoint_key="branch_prepared",
            step_name="prepare_branch",
        )

    def run_pm_step(self, record: JobRecord) -> JobRecord:
        self._mark_step_started(record=record, checkpoint_key="pm_started", step_name="pm", phase="pm")
        prd = self._run_structured_role(record, "pm", PRD, "Produce the product requirements", reuse_existing=True)
        record.spec.metadata = synthesize_job_metadata_from_prd(
            prd,
            record.spec.metadata,
            workspace_root=record.spec.workspace_root or record.spec.repo_path,
        )
        record.outputs["prd"] = prd.model_dump()
        self._write_memory_item(record, "pm", "prd", prd.model_dump_json())
        return self._mark_step_completed(
            record=record,
            checkpoint_key="pm_completed",
            step_name="pm",
            result_json={"title": prd.title},
        )

    def run_architect_step(self, record: JobRecord) -> JobRecord:
        design_feedback = self._design_feedback_logs(record)
        revising = bool(record.runtime_state.get("needs_architecture_revision"))
        self._mark_step_started(
            record=record,
            checkpoint_key="architecture_started",
            step_name="architect",
            phase="architect",
        )
        architecture = self._run_structured_role(
            record,
            "architect",
            ArchitecturePlan,
            "Revise the system architecture to address PM design review findings"
            if revising
            else "Design the system architecture",
            logs=design_feedback,
            reuse_existing=not revising,
        )
        record.outputs["architecture"] = architecture.model_dump()
        record.runtime_state.pop("needs_architecture_revision", None)
        self._write_memory_item(record, "architect", "architecture", architecture.model_dump_json())
        return self._mark_step_completed(
            record=record,
            checkpoint_key="architecture_completed",
            step_name="architect",
            result_json={"summary": architecture.summary},
        )

    def run_planner_step(self, record: JobRecord) -> JobRecord:
        design_feedback = self._design_feedback_logs(record)
        replanning = bool(record.runtime_state.get("needs_plan_revision"))
        self._mark_step_started(
            record=record,
            checkpoint_key="planning_started",
            step_name="planner",
            phase="planner",
        )
        task_graph = self._run_structured_role(
            record,
            "planner",
            TaskGraph,
            "Revise the implementation task graph to address PM design review findings"
            if replanning
            else "Create the implementation task graph",
            logs=design_feedback,
            reuse_existing=not replanning,
        )
        record.outputs["task_graph"] = task_graph.model_dump()
        record.runtime_state.pop("needs_plan_revision", None)
        self._ensure_runtime_artifacts_are_assigned(task_graph, record)
        self._write_memory_item(record, "planner", "task_graph", task_graph.model_dump_json())
        self.store.save_tasks(
            record.job_id,
            [
                TaskRecord.from_planned_task(job_id=record.job_id, task=task)
                for task in task_graph.tasks
            ],
        )
        return self._mark_step_completed(
            record=record,
            checkpoint_key="planning_completed",
            step_name="planner",
            result_json={"task_count": len(task_graph.tasks)},
        )

    def run_design_review_step(self, record: JobRecord) -> JobRecord:
        self._mark_step_started(
            record=record,
            checkpoint_key="design_review_started",
            step_name="design_review",
            phase="design_review",
        )
        review = self._run_structured_role(
            record,
            "pm",
            PMReviewResult,
            (
                "Review the PRD, architecture, and task graph. Reject plans that omit "
                "required bootstrap artifacts, runtime verification, or clear coverage "
                "of the user's requested outcome."
            ),
            reuse_existing=False,
            output_key="pm_design_review",
            phase_status=JobStatus.REVIEWING,
        )
        record.outputs["pm_design_review"] = review.model_dump()
        if review.decision == ReviewDecision.APPROVE:
            record.runtime_state.pop("needs_design_review", None)
            record.runtime_state.pop("design_feedback", None)
            record.runtime_state.pop("design_review_attempts", None)
            self._ensure_design_review_artifacts_are_assigned(record, review)
            record.runtime_state["required_artifacts"] = self._required_artifacts(record)
            return self._mark_step_completed(
                record=record,
                checkpoint_key="design_review_completed",
                step_name="design_review",
                result_json={"decision": review.decision.value},
            )
        attempts = int(record.runtime_state.get("design_review_attempts", 0)) + 1
        record.runtime_state["design_review_attempts"] = attempts
        if attempts >= self.max_attempts_per_task:
            record.status = JobStatus.STUCK
            record.last_error = "design_review_max_attempts_exceeded"
            return self.store.update(record)
        record.runtime_state["design_feedback"] = self._review_feedback_lines(review)
        record.runtime_state["needs_architecture_revision"] = True
        record.runtime_state["needs_plan_revision"] = True
        record.runtime_state["needs_design_review"] = True
        return self._mark_step_completed(
            record=record,
            checkpoint_key="design_review_completed",
            step_name="design_review",
            result_json={"decision": review.decision.value},
        )

    def run_task_scaffold_step(self, record: JobRecord, task: PlannedTask) -> JobRecord:
        checkpoint_key = f"task:{task.id}:scaffold_completed"
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task.id}:scaffold_started",
            step_name="scaffold",
            task_id=task.id,
            phase="scaffold",
        )
        if record.status != JobStatus.IMPLEMENTING:
            apply_transition(record, JobStatus.IMPLEMENTING)
        self._set_task_status(record, task.id, TaskStatus.IN_PROGRESS)
        changed_files: list[str] = []
        scaffold = self._framework_scaffold(record)
        if scaffold is not None:
            patches = [
                patch
                for patch in scaffold.patches
                if not self._workspace_file_exists(record, patch.path)
            ]
            if patches:
                self._apply_patches(record, "implementer", patches, task=task)
                changed_files = [patch.path for patch in patches]
            ensure_required_artifacts_exist(
                scaffold.required_artifacts,
                workspace_root=record.spec.workspace_root or record.spec.repo_path,
                label=f"framework scaffold {scaffold.key} required_artifacts",
            )
        return self._mark_step_completed(
            record=record,
            checkpoint_key=checkpoint_key,
            step_name="scaffold",
            task_id=task.id,
            result_json={"changed_files": changed_files},
        )

    def run_task_implementer_step(self, record: JobRecord, task: PlannedTask) -> JobRecord:
        checkpoint_key = f"task:{task.id}:implementer_completed"
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task.id}:implementer_started",
            step_name="implementer",
            task_id=task.id,
            phase="implementer",
        )
        implementation = self._run_structured_role(
            record,
            "implementer",
            ImplementationResult,
            "Implement the planned feature",
            task=task,
            reuse_existing=True,
        )
        self._apply_patches(record, "implementer", implementation.patches, task=task)
        self._ensure_runtime_bootstrap_artifacts_exist(record, task)
        self._set_task_status(record, task.id, TaskStatus.IMPLEMENTED)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=checkpoint_key,
            step_name="implementer",
            task_id=task.id,
            result_json={"changed_files": implementation.changed_files},
        )

    def run_task_test_writer_step(self, record: JobRecord, task: PlannedTask) -> JobRecord:
        checkpoint_key = f"task:{task.id}:test_writer_completed"
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task.id}:test_writer_started",
            step_name="test_writer",
            task_id=task.id,
            phase="test_writer",
        )
        output = self._run_structured_role(
            record,
            "test_writer",
            TestWriterResult,
            "Add tests for the implementation",
            task=task,
            reuse_existing=True,
        )
        self._apply_patches(record, "test_writer", output.patches, task=task)
        ensure_task_required_artifacts_exist(
            task,
            workspace_root=record.spec.workspace_root or record.spec.repo_path,
        )
        self._set_task_status(record, task.id, TaskStatus.TESTS_WRITTEN)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=checkpoint_key,
            step_name="test_writer",
            task_id=task.id,
            result_json={"changed_files": output.changed_files},
        )

    def run_task_review_step(self, record: JobRecord, task: PlannedTask) -> JobRecord:
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task.id}:review_started",
            step_name="review",
            task_id=task.id,
            phase="review",
        )
        review = self._run_structured_role(
            record,
            "reviewer",
            ReviewResult,
            "Review the changed code and tests",
            task=task,
            reuse_existing=False,
        )
        security_review = self._run_structured_role(
            record,
            "security_reviewer",
            SecurityReviewResult,
            "Review the changes for security risks",
            task=task,
            security_sensitive=True,
            reuse_existing=False,
        )
        try:
            ensure_reviews_pass(review, security_review)
        except QualityGateError:
            record.runtime_state["pending_fix_request"] = {
                "objective": "Address reviewer findings without weakening tests",
                "logs": [
                    *self._review_feedback_lines(review),
                    *self._review_feedback_lines(security_review),
                ],
                "source": "review",
            }
            record.runtime_state["needs_product_fix"] = True
            record.runtime_state["needs_rereview"] = True
            self._set_task_status(record, task.id, TaskStatus.CHANGES_REQUESTED)
            self._mark_step_completed(
                record=record,
                checkpoint_key=f"task:{task.id}:security_review_completed",
                step_name="security_review",
                task_id=task.id,
                result_json={"summary": security_review.summary},
            )
            return self._mark_step_completed(
                record=record,
                checkpoint_key=f"task:{task.id}:review_completed",
                step_name="review",
                task_id=task.id,
                result_json={"summary": review.summary},
            )
        record.runtime_state.pop("needs_rereview", None)
        self._set_task_status(record, task.id, TaskStatus.UNDER_REVIEW)
        self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task.id}:security_review_completed",
            step_name="security_review",
            task_id=task.id,
            result_json={"summary": security_review.summary},
        )
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task.id}:review_completed",
            step_name="review",
            task_id=task.id,
            result_json={"summary": review.summary},
        )

    def run_task_product_fix_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        request = record.runtime_state.get("pending_fix_request")
        if not isinstance(request, dict):
            raise RuntimeError("pending product fix requested without fix context")
        objective = str(request.get("objective") or "Address product review findings")
        logs = [str(item) for item in request.get("logs", []) if str(item).strip()]
        return self.run_task_fixer_step(record, task, None, objective=objective, logs=logs)

    def run_task_test_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        checkpoint_task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{checkpoint_task_id or 'job'}:tests_started",
            step_name="tests",
            task_id=checkpoint_task_id,
            phase="tests",
        )
        record.runtime_state.pop("needs_retest", None)
        test_result = self._run_tests(record)
        if not test_result.success:
            self._record_test_failure(record, test_result)
        record.outputs["test_run"] = test_result.model_dump()
        record.current_task_id = checkpoint_task_id
        self._set_task_status(
            record,
            checkpoint_task_id,
            TaskStatus.RUNNING if test_result.success else TaskStatus.TEST_FAILED,
        )
        if test_result.success:
            record.runtime_state["needs_runtime_prepare"] = True
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{checkpoint_task_id or 'job'}:tests_completed",
            step_name="tests",
            task_id=checkpoint_task_id,
            result_json={"success": test_result.success},
        )

    def run_task_runtime_prepare_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:runtime_prepare_started",
            step_name="runtime_prepare",
            task_id=task_id,
            phase="runtime_prepare",
        )
        prepare_result = self._run_runtime_prepare(record)
        record.outputs["runtime_prepare"] = prepare_result.model_dump()
        record.current_task_id = task_id
        record.runtime_state.pop("needs_runtime_prepare", None)
        if prepare_result.success:
            record.runtime_state["needs_runtime_smoke"] = True
            self._set_task_status(record, task_id, TaskStatus.RUNNING)
        else:
            self._record_test_failure(record, prepare_result)
            self._set_task_status(record, task_id, TaskStatus.TEST_FAILED)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:runtime_prepare_completed",
            step_name="runtime_prepare",
            task_id=task_id,
            result_json={"success": prepare_result.success},
        )

    def run_task_runtime_smoke_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:runtime_smoke_started",
            step_name="runtime_smoke",
            task_id=task_id,
            phase="runtime_smoke",
        )
        smoke_result = self._run_runtime_smoke(record)
        record.outputs["runtime_smoke"] = smoke_result.model_dump()
        record.current_task_id = task_id
        record.runtime_state.pop("needs_runtime_smoke", None)
        if smoke_result.success:
            required_artifacts = self._required_artifacts(record, task)
            ensure_task_target_files_exist(
                task,
                workspace_root=record.spec.workspace_root or record.spec.repo_path,
            )
            ensure_required_artifacts_exist(
                required_artifacts,
                workspace_root=record.spec.workspace_root or record.spec.repo_path,
                label="required_artifacts",
            )
            apply_transition(record, JobStatus.RUNNING)
            record.runtime_state["needs_acceptance_checks"] = True
            self._set_task_status(record, task_id, TaskStatus.RUNNING)
        else:
            self._record_test_failure(record, smoke_result)
            self._set_task_status(record, task_id, TaskStatus.TEST_FAILED)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:runtime_smoke_completed",
            step_name="runtime_smoke",
            task_id=task_id,
            result_json={"success": smoke_result.success},
        )

    def run_task_acceptance_checks_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:acceptance_checks_started",
            step_name="acceptance_checks",
            task_id=task_id,
            phase="acceptance_checks",
        )
        acceptance_result = self._run_acceptance_checks(record)
        record.outputs["acceptance_checks"] = acceptance_result.model_dump()
        record.current_task_id = task_id
        record.runtime_state.pop("needs_acceptance_checks", None)
        if acceptance_result.success:
            apply_transition(record, JobStatus.RUNNING)
            record.runtime_state["needs_acceptance_review"] = True
            self._set_task_status(record, task_id, TaskStatus.RUNNING)
        else:
            self._record_test_failure(record, acceptance_result)
            self._set_task_status(record, task_id, TaskStatus.TEST_FAILED)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:acceptance_checks_completed",
            step_name="acceptance_checks",
            task_id=task_id,
            result_json={"success": acceptance_result.success},
        )

    def run_task_acceptance_review_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:acceptance_review_started",
            step_name="acceptance_review",
            task_id=task_id,
            phase="acceptance_review",
        )
        logs = []
        for key in ("test_run", "runtime_prepare", "runtime_smoke", "acceptance_checks"):
            payload = record.outputs.get(key)
            if isinstance(payload, dict):
                excerpt = str(payload.get("output_excerpt", "")).strip()
                if excerpt:
                    logs.append(excerpt)
        review = self._run_structured_role(
            record,
            "pm",
            PMReviewResult,
            (
                "Review the delivered workspace against the PRD, architecture, tests, "
                "and runtime evidence. Reject if the implemented outcome still differs "
                "from the requested product behavior."
            ),
            task=task,
            logs=logs,
            reuse_existing=False,
            output_key="pm_acceptance_review",
            phase_status=JobStatus.REVIEWING,
        )
        record.outputs["pm_acceptance_review"] = review.model_dump()
        if review.decision == ReviewDecision.APPROVE:
            record.runtime_state.pop("needs_acceptance_review", None)
            record.runtime_state.pop("acceptance_review_attempts", None)
            self._set_task_status(record, task_id, TaskStatus.DONE)
            return self._mark_step_completed(
                record=record,
                checkpoint_key=f"task:{task_id or 'job'}:acceptance_review_completed",
                step_name="acceptance_review",
                task_id=task_id,
                result_json={"decision": review.decision.value},
            )
        attempts = int(record.runtime_state.get("acceptance_review_attempts", 0)) + 1
        record.runtime_state["acceptance_review_attempts"] = attempts
        if attempts >= self.max_attempts_per_task:
            record.status = JobStatus.STUCK
            record.last_error = "acceptance_review_max_attempts_exceeded"
            self._set_task_status(record, task_id, TaskStatus.STUCK)
            return self.store.update(record)
        record.runtime_state["pending_fix_request"] = {
            "objective": "Address PM acceptance findings while preserving tests and runtime behavior",
            "logs": self._review_feedback_lines(review),
            "source": "pm_acceptance",
        }
        record.runtime_state["needs_product_fix"] = True
        record.runtime_state["needs_acceptance_review"] = True
        self._set_task_status(record, task_id, TaskStatus.CHANGES_REQUESTED)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:acceptance_review_completed",
            step_name="acceptance_review",
            task_id=task_id,
            result_json={"decision": review.decision.value},
        )

    def run_task_fixer_step(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        test_result: TestRunResult | None,
        *,
        objective: str | None = None,
        logs: list[str] | None = None,
    ) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:fixer_started",
            step_name="fixer",
            task_id=task_id,
            phase="fixer",
        )
        fix = self._run_structured_role(
            record,
            "fixer",
            FixResult,
            objective or "Fix the deterministic test failures",
            task=task,
            logs=logs if logs is not None else ([test_result.output_excerpt] if test_result is not None else []),
            reuse_existing=not self._should_force_fixer_rerun(record),
        )
        ensure_fixer_safe(fix.patches)
        self._apply_patches(record, "fixer", fix.patches, task=task)
        ensure_task_required_artifacts_exist(
            task,
            workspace_root=record.spec.workspace_root or record.spec.repo_path,
        )
        self._mark_fixer_consumed(record)
        review_driven_fix = test_result is None
        record.failure_count += 1
        if test_result is not None:
            record.same_test_failure_count += 1 if test_result.failed_tests else 0
        if fix.status == FixStatus.STUCK:
            record.status = JobStatus.STUCK
            return self.store.update(record)
        if test_result is not None and record.same_test_failure_count >= self.max_same_failure_repeats:
            record.status = JobStatus.STUCK
            record.last_error = "same_failure_threshold_reached"
            return self.store.update(record)
        record.runtime_state.pop("pending_fix_request", None)
        record.runtime_state.pop("needs_product_fix", None)
        if review_driven_fix:
            record.runtime_state["needs_rereview"] = True
        record.runtime_state["needs_retest"] = True
        record.outputs.pop("test_run", None)
        record.outputs.pop("pm_acceptance_review", None)
        record.outputs.pop("runtime_prepare", None)
        record.outputs.pop("runtime_smoke", None)
        record.outputs.pop("acceptance_checks", None)
        if review_driven_fix:
            record.outputs.pop("reviewer", None)
            record.outputs.pop("security_reviewer", None)
        record.runtime_state.pop("needs_runtime_prepare", None)
        record.runtime_state.pop("needs_runtime_smoke", None)
        record.runtime_state.pop("needs_acceptance_checks", None)
        self._set_task_status(record, task_id, TaskStatus.RUNNING)
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{task_id or 'job'}:fixer_completed",
            step_name="fixer",
            task_id=task_id,
            result_json={"status": fix.status.value},
        )

    def run_final_quality_gates_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key="final_quality_gates_started",
            step_name="final_quality_gates",
            task_id=task_id,
            phase="final_quality_gates",
        )
        test_result = TestRunResult.model_validate(record.outputs["test_run"])
        summary = self._run_structured_role(
            record,
            "summarizer",
            SummaryResult,
            "Summarize the completed job and memory",
            task=task,
            logs=[test_result.output_excerpt],
            reuse_existing=True,
        )
        self._write_memory_entries(record, summary)
        record.outputs["summary"] = summary.model_dump()
        return self._mark_step_completed(
            record=record,
            checkpoint_key="final_quality_gates_completed",
            step_name="final_quality_gates",
            task_id=task_id,
            result_json={"summary": summary.summary},
        )

    def run_release_step(self, record: JobRecord, task: PlannedTask | None) -> JobRecord:
        task_id = task.id if task is not None else None
        self._mark_step_started(
            record=record,
            checkpoint_key="release_started",
            step_name="release",
            task_id=task_id,
            phase="release",
        )
        release = self._run_structured_role(
            record,
            "release_manager",
            ReleaseResult,
            "Prepare the final release artifact",
            task=task,
            reuse_existing=True,
        )
        self._release(record, release)
        record.status = JobStatus.DONE
        record.completed_at = record.updated_at
        return self._mark_step_completed(
            record=record,
            checkpoint_key="release_completed",
            step_name="release",
            task_id=task_id,
            result_json={"commit_message": release.commit_message},
        )

    def _execute(self, record: JobRecord) -> JobRecord:
        self._active_record = record
        try:
            self._prepare_branch(record)
            prd = self._run_structured_role(
                record,
                "pm",
                PRD,
                "Produce the product requirements",
                reuse_existing=True,
            )
            self._write_memory_item(record, "pm", "prd", prd.model_dump_json())
            architecture = self._run_structured_role(
                record,
                "architect",
                ArchitecturePlan,
                "Design the system architecture",
                reuse_existing=True,
            )
            self._write_memory_item(
                record,
                "architect",
                "architecture",
                architecture.model_dump_json(),
            )
            task_graph = self._run_structured_role(
                record,
                "planner",
                TaskGraph,
                "Create the implementation task graph",
                reuse_existing=True,
            )
            self._write_memory_item(
                record,
                "planner",
                "task_graph",
                task_graph.model_dump_json(),
            )
            primary_task = self._choose_primary_task(task_graph)
            implementation = self._run_structured_role(
                record,
                "implementer",
                ImplementationResult,
                "Implement the planned feature",
                task=primary_task,
                reuse_existing=True,
            )
            self._apply_patches(record, "implementer", implementation.patches, task=primary_task)
            test_writer = self._run_structured_role(
                record,
                "test_writer",
                TestWriterResult,
                "Add tests for the implementation",
                task=primary_task,
                reuse_existing=True,
            )
            self._apply_patches(record, "test_writer", test_writer.patches, task=primary_task)
            self._run_review_cycle(record, primary_task)
            test_result = self._run_tests(record)
            if not test_result.success:
                self._record_test_failure(record, test_result)
            while not test_result.success and record.failure_count < self.max_attempts_per_task:
                fix = self._run_structured_role(
                    record,
                    "fixer",
                    FixResult,
                    "Fix the deterministic test failures",
                    task=primary_task,
                    logs=[test_result.output_excerpt],
                    reuse_existing=not self._should_force_fixer_rerun(record),
                )
                ensure_fixer_safe(fix.patches)
                self._apply_patches(record, "fixer", fix.patches, task=primary_task)
                self._mark_fixer_consumed(record)
                record.failure_count += 1
                record.same_test_failure_count += 1 if test_result.failed_tests else 0
                self.store.update(record)
                if fix.status == FixStatus.STUCK:
                    record.status = JobStatus.STUCK
                    return self.store.update(record)
                if record.same_test_failure_count >= self.max_same_failure_repeats:
                    record.status = JobStatus.STUCK
                    record.last_error = "same_failure_threshold_reached"
                    return self.store.update(record)
                test_result = self._run_tests(record)
                if not test_result.success:
                    self._record_test_failure(record, test_result)
            if not test_result.success:
                if record.failure_count >= self.max_attempts_per_task:
                    record.status = JobStatus.STUCK
                    record.last_error = "max_attempts_exceeded"
                else:
                    record.status = JobStatus.FAILED
                    record.last_error = "tests_failed_after_retries"
                return self.store.update(record)
            record.current_task_id = primary_task.id if primary_task is not None else None
            ensure_task_target_files_exist(
                primary_task,
                workspace_root=record.spec.workspace_root or record.spec.repo_path,
            )
            summary = self._run_structured_role(
                record,
                "summarizer",
                SummaryResult,
                "Summarize the completed job and memory",
                task=primary_task,
                logs=[test_result.output_excerpt],
                reuse_existing=True,
            )
            self._write_memory_entries(record, summary)
            release = self._run_structured_role(
                record,
                "release_manager",
                ReleaseResult,
                "Prepare the final release artifact",
                task=primary_task,
                reuse_existing=True,
            )
            self._release(record, release)
            record.outputs["prd"] = prd.model_dump()
            record.outputs["architecture"] = architecture.model_dump()
            record.outputs["task_graph"] = task_graph.model_dump()
            record.outputs["test_run"] = test_result.model_dump()
            record.outputs["summary"] = summary.model_dump()
            if record.status != JobStatus.DONE:
                apply_transition(record, JobStatus.DONE)
            return self.store.update(record)
        except ApprovalRequiredError as exc:
            return self._pause_for_approval(record, exc)
        except QualityGateError as exc:
            record.status = JobStatus.BLOCKED
            record.last_error = redact_text(str(exc))
            self._set_task_status(record, record.current_task_id, TaskStatus.BLOCKED)
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            record.status = JobStatus.FAILED
            record.last_error = redact_text(str(exc))
            return self.store.update(record)
        finally:
            self._active_record = None

    def _pause_for_approval(self, record: JobRecord, error: ApprovalRequiredError) -> JobRecord:
        if self.approval_gateway is None:
            raise RuntimeError("approval gateway is not configured") from error
        challenge = self.approval_gateway.create_challenge(
            job_id=record.job_id,
            task_id=error.task_id or record.current_task_id,
            role=record.current_role,
            requested_by=error.requested_by,
            operation=error.operation,
            risk_level=error.decision.risk_level,
            reason=error.decision.reason,
            proposed_action=error.proposed_action,
        )
        if self._active_record is not None:
            self._active_record.audit_events.append(
                self.audit.approval_event(
                    role=record.current_role or error.requested_by,
                    action="requested",
                    approval=challenge.request,
                )
            )
        record.runtime_state["resume_status"] = record.status.value
        record.runtime_state["pending_operation"] = {
            "operation": error.operation,
            "requested_by": error.requested_by,
            "details": error.proposed_action,
        }
        record.pending_approval_id = challenge.request.id
        record.last_error = redact_text(error.decision.reason)
        self._set_task_status(
            record,
            record.current_task_id,
            TaskStatus.WAITING_APPROVAL,
            approval_id=challenge.request.id,
        )
        if record.status != JobStatus.WAITING_APPROVAL:
            apply_transition(record, JobStatus.WAITING_APPROVAL)
        self.store.update(record)
        self._notify_approval_required(record, challenge)
        return record

    def _pause_for_runtime(
        self,
        record: JobRecord,
        error: ProviderUnavailableError,
    ) -> JobRecord:
        if self.runtime_manager is None:
            record.status = JobStatus.FAILED
            record.last_error = redact_text(str(error))
            return self.store.update(record)
        issue = self.runtime_manager.handle_provider_error(
            record=record,
            provider_key=error.provider_key,
            model_key=error.model_key,
            issue_type=error.issue_type,
            message=redact_text(str(error)),
        )
        self._notify_runtime_wait(record, issue)
        return self.store.get(record.job_id)

    def _notify_approval_required(self, record: JobRecord, challenge: Any) -> None:
        payload = {
            "approval_id": challenge.request.id,
            "job_id": record.job_id,
            "risk_level": challenge.request.risk_level.value,
            "operation": challenge.request.operation,
            "reason": challenge.request.reason,
            "approve_url": challenge.approve_url,
            "reject_url": challenge.reject_url,
            "cli_command": f"acos approvals approve {challenge.request.id} --workspace {record.spec.repo_path}",
        }
        result = self.router.call("notify_server.send_approval_request", **payload)
        event = self.audit.tool_event(
            role="orchestrator",
            tool_name="notify_server.send_approval_request",
            input_payload=payload,
            output_payload=result.data,
            status="success" if result.ok else "failed",
        )
        record.audit_events.append(event)

    def _notify_runtime_wait(self, record: JobRecord, issue: Any) -> None:
        payload = {
            "job_id": record.job_id,
            "provider_key": issue.provider_key,
            "model_key": issue.model_key,
            "reason": issue.message,
            "kind": "runtime_wait",
            "channel": "console",
            "cli_command": "\n".join(
                [
                    f"acos check-provider --provider {issue.provider_key}",
                    f"acos check-model --model {issue.model_key}" if issue.model_key else "",
                    f"acos jobs resume {record.job_id}",
                ]
            ).strip(),
        }
        if hasattr(self.store, "record_notification"):
            self.store.record_notification(payload)
        result = self.router.call("notify_server.send_runtime_wait", **payload)
        event = self.audit.tool_event(
            role="orchestrator",
            tool_name="notify_server.send_runtime_wait",
            input_payload=payload,
            output_payload=result.data,
            status="success" if result.ok else "failed",
        )
        record.audit_events.append(event)
        self.store.update(record)

    def _phase_for_role(self, role: str) -> JobStatus:
        return {
            "pm": JobStatus.ANALYZING,
            "architect": JobStatus.DESIGNING,
            "planner": JobStatus.PLANNING,
            "implementer": JobStatus.IMPLEMENTING,
            "test_writer": JobStatus.WRITING_TESTS,
            "reviewer": JobStatus.REVIEWING,
            "security_reviewer": JobStatus.REVIEWING,
            "fixer": JobStatus.FIXING,
            "release_manager": JobStatus.FINALIZING,
            "summarizer": JobStatus.FINALIZING,
        }[role]

    def _run_structured_role(
        self,
        record: JobRecord,
        role: str,
        response_model: type[T],
        objective: str,
        task: PlannedTask | None = None,
        logs: list[str] | None = None,
        security_sensitive: bool = False,
        *,
        reuse_existing: bool,
        output_key: str | None = None,
        phase_status: JobStatus | None = None,
    ) -> T:
        cache_key = output_key or self._role_output_cache_key(role, task)
        if reuse_existing and cache_key in record.outputs:
            return response_model.model_validate(record.outputs[cache_key])
        if reuse_existing and output_key is None and task is None and role in record.outputs:
            return response_model.model_validate(record.outputs[role])
        target_phase = phase_status or self._phase_for_role(role)
        if record.status != target_phase:
            apply_transition(record, target_phase)
        record.current_role = role
        record.current_task_id = task.id if task is not None else None
        agent_cfg = self.registry.get_agent(role)
        relevant_files = self._gather_relevant_files(role)
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
        packet = self.context_builder.build(
            job_id=record.job_id,
            role=role,
            objective=objective,
            repo_path=record.spec.workspace_root or record.spec.repo_path,
            request_text=record.spec.request_text,
            constraints=list(self.policy.config.risk_rules.deny),
            relevant_files=relevant_files,
            diff=diff,
            memory_summaries=memory_summaries,
            logs=logs or [],
            token_budget=agent_cfg.context_budget_tokens,
            agent_config=agent_cfg,
            selected_model=selected_model,
            task=task,
            metadata={
                "output_schema": agent_cfg.output_schema,
                "job_metadata": record.spec.metadata,
            },
        )
        if packet.metadata.get("context_truncated"):
            record.audit_events.append(
                AuditEvent(
                    event_type="context_build",
                    role=role,
                    action="truncate_context",
                    status="truncated",
                    job_id=record.job_id,
                    task_id=task.id if task is not None else None,
                    metadata={
                        "selected_model_hint": packet.selected_model_hint,
                        "estimated_input_tokens": packet.metadata.get("estimated_input_tokens"),
                        "context_budget_tokens": packet.metadata.get("context_budget_tokens"),
                        "effective_context_budget_tokens": packet.metadata.get(
                            "effective_context_budget_tokens"
                        ),
                        "safety_margin_tokens": packet.metadata.get("safety_margin_tokens"),
                        "context_truncation_notes": packet.metadata.get(
                            "context_truncation_notes",
                            [],
                        ),
                    },
                )
            )
        routing_context = RoutingContext(
            role=role,
            failure_count=record.failure_count,
            same_test_failure_count=record.same_test_failure_count,
            changed_files_count=len(relevant_files),
            security_sensitive=security_sensitive,
            context_tokens=estimate_tokens(packet.render_text()),
        )
        output, selection, _model_record = self.agent_runner.run(
            role=role,
            response_model=response_model,
            context_packet=packet,
            routing_context=routing_context,
            allowed_tools=agent_cfg.allowed_tools if agent_cfg.allow_tools else [],
            require_json_schema=agent_cfg.require_json_schema,
            max_steps=self.max_steps_per_agent,
            audit_events=record.audit_events,
        )
        record.outputs[cache_key] = output.model_dump()
        record.outputs[f"{cache_key}_model_selection"] = selection.model_dump()
        if output_key is None:
            record.outputs[role] = output.model_dump()
            record.outputs[f"{role}_model_selection"] = selection.model_dump()
        self.store.update(record)
        return output

    @staticmethod
    def _role_output_cache_key(role: str, task: PlannedTask | None) -> str:
        if task is None:
            return role
        return f"{role}__{task.id}"

    def _gather_relevant_files(self, role: str) -> dict[str, str]:
        files: dict[str, str] = {}
        can_tree = self.policy.is_tool_allowed(role, "repo_server.repo_tree")
        can_read = self.policy.is_tool_allowed(role, "repo_server.read_file")
        candidates: list[str] = []
        if can_tree:
            candidates = list(self._call_tool(role, "repo_server.repo_tree").get("files", []))[:5]
        else:
            status = self._call_tool("release_manager", "git_server.status")
            candidates = list(status.get("modified_files", []))[:5]
        if can_read:
            for path in candidates:
                payload = self._call_tool(role, "repo_server.read_file", path=path)
                files[path] = str(payload["content"])
        return files

    def _apply_patches(
        self,
        record: JobRecord,
        role: str,
        patches: list[Any],
        *,
        task: PlannedTask | None = None,
    ) -> None:
        previous_active_record = self._active_record
        if previous_active_record is None:
            self._active_record = record
        normalized_patches = []
        try:
            for patch in patches:
                normalized_path = self._normalize_patch_path(record, patch.path)
                normalized_patches.append(
                    patch.model_copy(update={"path": normalized_path})
                )
            if role == "test_writer":
                ensure_test_patch_quality(normalized_patches, role=role)
            applied = set(record.runtime_state.get("applied_patches", []))
            for index, patch in enumerate(normalized_patches):
                patch_key = f"{role}:{task.id if task else 'job'}:{index}:{patch.path}"
                if patch_key in applied:
                    continue
                record.current_role = role
                record.current_task_id = task.id if task else None
                self.policy.assert_patch_target_allowed(role, patch.path)
                self._call_tool(
                    role,
                    "repo_server.apply_patch",
                    path=patch.path,
                    content=patch.content,
                    operation=patch.operation,
                )
                applied.add(patch_key)
            record.runtime_state["applied_patches"] = sorted(applied)
            self.store.update(record)
        finally:
            if previous_active_record is None:
                self._active_record = None

    @staticmethod
    def _normalize_patch_path(record: JobRecord, path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return path
        workspace_root = Path(record.spec.workspace_root or record.spec.repo_path).resolve()
        resolved = candidate.resolve()
        if workspace_root not in [resolved, *resolved.parents]:
            raise PermissionError("patch path escapes workspace")
        relative = resolved.relative_to(workspace_root).as_posix()
        if relative in {"", "."}:
            raise PermissionError("patch path must target a file inside the workspace")
        return relative

    def _run_tests(self, record: JobRecord) -> TestRunResult:
        if record.status != JobStatus.TESTING:
            apply_transition(record, JobStatus.TESTING)
        record.current_role = "runner"
        record.current_task_id = None
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="auto",
            timeout_seconds=120,
        )
        result = TestRunResult.model_validate(payload)
        result = self._retry_after_allowlisted_dependency_install(
            record,
            result,
            command_name="auto",
            timeout_seconds=120,
        )
        record.outputs["test_run"] = result.model_dump()
        self.store.update(record)
        return result

    def _run_runtime_prepare(self, record: JobRecord) -> TestRunResult:
        if record.status != JobStatus.TESTING:
            apply_transition(record, JobStatus.TESTING)
        record.current_role = "runner"
        runtime_commands = self._runtime_prepare_commands(record)
        if runtime_commands is not None:
            return self._run_runtime_prepare_commands(record, runtime_commands)
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="prepare-runtime-auto",
            timeout_seconds=120,
        )
        result = TestRunResult.model_validate(payload)
        result = self._retry_after_allowlisted_dependency_install(
            record,
            result,
            command_name="prepare-runtime-auto",
            timeout_seconds=120,
        )
        record.outputs["runtime_prepare"] = result.model_dump()
        self.store.update(record)
        return result

    def _run_runtime_smoke(self, record: JobRecord) -> TestRunResult:
        if record.status != JobStatus.TESTING:
            apply_transition(record, JobStatus.TESTING)
        record.current_role = "runner"
        http_checks = self._runtime_http_checks(record)
        start_command = self._runtime_start_command(record)
        if start_command is not None:
            return self._run_runtime_start_command(record, start_command, http_checks=http_checks)
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="runtime-smoke-auto",
            timeout_seconds=60,
            http_checks=http_checks,
        )
        result = TestRunResult.model_validate(payload)
        result = self._retry_after_allowlisted_dependency_install(
            record,
            result,
            command_name="runtime-smoke-auto",
            timeout_seconds=60,
            extra_tool_kwargs={"http_checks": http_checks} if http_checks is not None else None,
        )
        record.outputs["runtime_smoke"] = result.model_dump()
        self.store.update(record)
        return result

    def _run_acceptance_checks(self, record: JobRecord) -> TestRunResult:
        checks = self._acceptance_http_checks(record)
        if checks is None:
            return TestRunResult(
                success=True,
                command=["acceptance-checks", "skipped"],
                failed_tests=[],
                output_excerpt="no acceptance checks configured",
                exit_code=0,
            )
        if record.status != JobStatus.TESTING:
            apply_transition(record, JobStatus.TESTING)
        record.current_role = "runner"
        start_command = self._runtime_start_command(record)
        if start_command is not None:
            return self._run_runtime_start_command(
                record,
                start_command,
                output_key="acceptance_checks",
                http_checks=checks,
            )
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="runtime-smoke-auto",
            timeout_seconds=60,
            http_checks=checks,
        )
        result = TestRunResult.model_validate(payload)
        result = self._retry_after_allowlisted_dependency_install(
            record,
            result,
            command_name="runtime-smoke-auto",
            timeout_seconds=60,
            extra_tool_kwargs={"http_checks": checks},
        )
        record.outputs["acceptance_checks"] = result.model_dump()
        self.store.update(record)
        return result

    def _run_runtime_prepare_commands(
        self,
        record: JobRecord,
        commands: list[list[str]],
    ) -> TestRunResult:
        timeout_seconds = self._runtime_prepare_timeout_seconds(record)
        outputs: list[str] = []
        last_result: TestRunResult | None = None
        for argv in commands:
            payload = self._call_tool(
                "runner",
                "test_server.run_command",
                argv=argv,
                timeout_seconds=timeout_seconds,
                mode="oneshot",
            )
            result = TestRunResult.model_validate(payload)
            result = self._retry_runtime_command_after_allowlisted_dependency_install(
                record,
                result,
                argv=argv,
                timeout_seconds=timeout_seconds,
                mode="oneshot",
            )
            outputs.append(result.output_excerpt)
            last_result = result
            if not result.success:
                aggregated = result.model_copy(update={"output_excerpt": "\n\n".join(item for item in outputs if item)[-20000:]})
                record.outputs["runtime_prepare"] = aggregated.model_dump()
                self.store.update(record)
                return aggregated
        if last_result is None:
            last_result = TestRunResult(
                success=True,
                command=["runtime-prepare", "skipped"],
                failed_tests=[],
                output_excerpt="no runtime preparation commands configured",
                exit_code=0,
            )
        aggregated = last_result.model_copy(update={"output_excerpt": "\n\n".join(item for item in outputs if item)[-20000:]})
        record.outputs["runtime_prepare"] = aggregated.model_dump()
        self.store.update(record)
        return aggregated

    def _run_runtime_start_command(
        self,
        record: JobRecord,
        argv: list[str],
        *,
        output_key: str = "runtime_smoke",
        http_checks: list[dict[str, Any]] | None = None,
    ) -> TestRunResult:
        timeout_seconds = self._runtime_start_timeout_seconds(record)
        http_path = self._runtime_http_probe_path(record)
        payload = self._call_tool(
            "runner",
            "test_server.run_command",
            argv=argv,
            timeout_seconds=timeout_seconds,
            mode="server",
            http_path=http_path,
            http_checks=http_checks,
        )
        result = TestRunResult.model_validate(payload)
        result = self._retry_runtime_command_after_allowlisted_dependency_install(
            record,
            result,
            argv=argv,
            timeout_seconds=timeout_seconds,
            mode="server",
            http_path=http_path,
            http_checks=http_checks,
        )
        record.outputs[output_key] = result.model_dump()
        self.store.update(record)
        return result

    def _retry_after_allowlisted_dependency_install(
        self,
        record: JobRecord,
        result: TestRunResult,
        *,
        command_name: str,
        timeout_seconds: int,
        extra_tool_kwargs: dict[str, Any] | None = None,
    ) -> TestRunResult:
        package = self._infer_allowlisted_dependency_package(result.output_excerpt)
        if result.success or package is None or not self._job_allows_dependency_addition(record):
            return result
        installed = set(record.runtime_state.get("installed_dependencies", []))
        if package in installed:
            return result
        install_result = self._call_tool(
            "runner",
            "test_server.install_package",
            package=package,
            timeout_seconds=600,
        )
        if not bool(install_result.get("success")):
            return result
        installed.add(package)
        record.runtime_state["installed_dependencies"] = sorted(installed)
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name=command_name,
            timeout_seconds=timeout_seconds,
            **(extra_tool_kwargs or {}),
        )
        return TestRunResult.model_validate(payload)

    def _retry_runtime_command_after_allowlisted_dependency_install(
        self,
        record: JobRecord,
        result: TestRunResult,
        *,
        argv: list[str],
        timeout_seconds: int,
        mode: str,
        http_path: str = "/",
        http_checks: list[dict[str, Any]] | None = None,
    ) -> TestRunResult:
        package = self._infer_allowlisted_dependency_package(result.output_excerpt)
        if result.success or package is None or not self._job_allows_dependency_addition(record):
            return result
        installed = set(record.runtime_state.get("installed_dependencies", []))
        if package in installed:
            return result
        install_result = self._call_tool(
            "runner",
            "test_server.install_package",
            package=package,
            timeout_seconds=600,
        )
        if not bool(install_result.get("success")):
            return result
        installed.add(package)
        record.runtime_state["installed_dependencies"] = sorted(installed)
        payload = self._call_tool(
            "runner",
            "test_server.run_command",
            argv=argv,
            timeout_seconds=timeout_seconds,
            mode=mode,
            http_path=http_path,
            http_checks=http_checks,
        )
        return TestRunResult.model_validate(payload)

    def _runtime_metadata(self, record: JobRecord) -> dict[str, Any]:
        runtime = record.spec.metadata.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("metadata.runtime must be a mapping")
        return runtime

    def _framework_profile(self, record: JobRecord) -> ResolvedFrameworkProfile | None:
        metadata = record.spec.metadata
        if not isinstance(metadata, dict):
            return None
        return resolve_framework_profile(metadata)

    def _framework_scaffold(self, record: JobRecord) -> ResolvedFrameworkScaffold | None:
        metadata = record.spec.metadata
        if not isinstance(metadata, dict):
            return None
        return resolve_framework_scaffold(
            metadata,
            workspace_root=record.spec.workspace_root or record.spec.repo_path,
        )

    def _framework_profile_required_artifacts(self, record: JobRecord) -> list[str]:
        scaffold = self._framework_scaffold(record)
        if scaffold is not None:
            return list(scaffold.required_artifacts)
        profile = self._framework_profile(record)
        if profile is None:
            return []
        return list(profile.required_artifacts)

    def _runtime_prepare_commands(self, record: JobRecord) -> list[list[str]] | None:
        runtime = self._runtime_metadata(record)
        commands = runtime.get("prepare_commands")
        if commands is not None:
            return self._validate_runtime_command_list(commands, field_name="metadata.runtime.prepare_commands")
        profile = self._framework_profile(record)
        if profile is None or profile.runtime_prepare_commands is None:
            return None
        return [list(command) for command in profile.runtime_prepare_commands]

    def _runtime_start_command(self, record: JobRecord) -> list[str] | None:
        runtime = self._runtime_metadata(record)
        command = runtime.get("start_command")
        if command is not None:
            if not isinstance(command, list) or not command or not all(isinstance(item, str) and item.strip() for item in command):
                raise ValueError("metadata.runtime.start_command must be a non-empty list of strings")
            return list(command)
        profile = self._framework_profile(record)
        if profile is None or profile.runtime_start_command is None:
            return None
        return list(profile.runtime_start_command)

    def _runtime_http_probe_path(self, record: JobRecord) -> str:
        runtime = self._runtime_metadata(record)
        http_path = runtime.get("http_probe_path")
        if http_path is not None:
            if not isinstance(http_path, str) or not http_path.startswith("/"):
                raise ValueError("metadata.runtime.http_probe_path must start with '/'")
            return http_path
        profile = self._framework_profile(record)
        if profile is None:
            return "/"
        return profile.runtime_http_probe_path

    def _runtime_http_checks(self, record: JobRecord) -> list[dict[str, Any]] | None:
        runtime = self._runtime_metadata(record)
        checks = runtime.get("http_checks")
        if checks is None:
            return None
        if not isinstance(checks, list):
            raise ValueError("metadata.runtime.http_checks must be a list")
        return [
            RuntimeHttpCheck.model_validate(item).model_dump(exclude_none=True)
            for item in checks
        ]

    def _acceptance_http_checks(self, record: JobRecord) -> list[dict[str, Any]] | None:
        checks = record.spec.metadata.get("acceptance_checks")
        if checks is None:
            return None
        if not isinstance(checks, list):
            raise ValueError("metadata.acceptance_checks must be a list")
        return [
            RuntimeHttpCheck.model_validate(item).model_dump(exclude_none=True)
            for item in checks
        ]

    def _runtime_prepare_timeout_seconds(self, record: JobRecord) -> int:
        runtime = self._runtime_metadata(record)
        return self._coerce_runtime_timeout(runtime.get("prepare_timeout_seconds"), default=120)

    def _runtime_start_timeout_seconds(self, record: JobRecord) -> int:
        runtime = self._runtime_metadata(record)
        return self._coerce_runtime_timeout(runtime.get("startup_timeout_seconds"), default=60)

    @staticmethod
    def _coerce_runtime_timeout(value: Any, *, default: int) -> int:
        if value is None:
            return default
        if not isinstance(value, int) or value <= 0:
            raise ValueError("runtime timeout values must be positive integers")
        return value

    @staticmethod
    def _validate_runtime_command_list(value: Any, *, field_name: str) -> list[list[str]]:
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be a list of argument lists")
        normalized: list[list[str]] = []
        for item in value:
            if not isinstance(item, list) or not item or not all(isinstance(part, str) and part.strip() for part in item):
                raise ValueError(f"{field_name} must be a list of non-empty string argument lists")
            normalized.append(list(item))
        return normalized

    @staticmethod
    def _review_feedback_lines(result: ReviewResult | SecurityReviewResult | PMReviewResult) -> list[str]:
        lines = [result.summary]
        lines.extend(item.description for item in result.findings)
        return [line for line in lines if line]

    @staticmethod
    def _design_feedback_logs(record: JobRecord) -> list[str]:
        feedback = record.runtime_state.get("design_feedback")
        if not isinstance(feedback, list):
            return []
        return [str(item) for item in feedback if str(item).strip()]

    def _ensure_runtime_artifacts_are_assigned(
        self,
        task_graph: TaskGraph,
        record: JobRecord,
    ) -> None:
        runtime_artifacts = sorted(
            set(self._metadata_required_artifacts(record))
            | set(self._framework_profile_required_artifacts(record))
            | set(self._runtime_command_artifacts(record))
        )
        ensure_required_artifacts_assigned_to_tasks(
            task_graph.tasks,
            runtime_artifacts,
            label="runtime required_artifacts",
        )

    def _metadata_required_artifacts(self, record: JobRecord) -> list[str]:
        raw = record.spec.metadata.get("required_artifacts")
        if not isinstance(raw, list):
            return []
        artifacts: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                artifacts.append(item.strip())
        return artifacts

    def _ensure_design_review_artifacts_are_assigned(
        self,
        record: JobRecord,
        review: PMReviewResult,
    ) -> None:
        task_graph_payload = record.outputs.get("task_graph") or record.outputs.get("planner")
        if not isinstance(task_graph_payload, dict):
            return
        task_graph = TaskGraph.model_validate(task_graph_payload)
        ensure_required_artifacts_assigned_to_tasks(
            task_graph.tasks,
            [*self._metadata_required_artifacts(record), *review.required_artifacts],
            label="design review required_artifacts",
        )

    def _ensure_runtime_bootstrap_artifacts_exist(
        self,
        record: JobRecord,
        task: PlannedTask | None,
    ) -> None:
        if task is None:
            return
        runtime_artifacts = self._runtime_command_artifacts(record)
        if not runtime_artifacts:
            return
        declared = set(task.required_artifacts) | set(task.target_files)
        bootstrap_artifacts = sorted(runtime_artifacts & declared)
        if not bootstrap_artifacts:
            return
        ensure_required_artifacts_exist(
            bootstrap_artifacts,
            workspace_root=record.spec.workspace_root or record.spec.repo_path,
            label=f"task {task.id} runtime bootstrap artifacts",
        )

    def _required_artifacts(self, record: JobRecord, task: PlannedTask | None = None) -> list[str]:
        artifacts: set[str] = set()
        artifacts.update(self._metadata_required_artifacts(record))
        artifacts.update(self._framework_profile_required_artifacts(record))
        if task is not None:
            artifacts.update(item for item in task.target_files if item)
            artifacts.update(item for item in task.required_artifacts if item)
        design_review = record.outputs.get("pm_design_review")
        if isinstance(design_review, dict):
            for item in design_review.get("required_artifacts", []):
                if isinstance(item, str) and item.strip():
                    artifacts.add(item.strip())
        artifacts.update(self._runtime_command_artifacts(record))
        return sorted(artifacts)

    def _runtime_command_artifacts(self, record: JobRecord) -> set[str]:
        artifacts: set[str] = set()
        try:
            commands = []
            prepare = self._runtime_prepare_commands(record)
            if prepare is not None:
                commands.extend(prepare)
            start = self._runtime_start_command(record)
            if start is not None:
                commands.append(start)
        except ValueError:
            return artifacts
        path_suffixes = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".sh",
            ".rb",
            ".php",
            ".go",
            ".rs",
            ".java",
            ".kt",
        }
        for argv in commands:
            for part in argv:
                candidate = str(part).strip().replace("\\", "/")
                if (
                    not candidate
                    or candidate.startswith("-")
                    or "{" in candidate
                    or "}" in candidate
                    or ":" in candidate
                ):
                    continue
                path = Path(candidate)
                suffix = path.suffix.lower()
                if suffix in path_suffixes and not path.is_absolute():
                    artifacts.add(candidate)
        return artifacts

    @staticmethod
    def _job_allows_dependency_addition(record: JobRecord) -> bool:
        constraints = record.spec.metadata.get("constraints", {})
        return isinstance(constraints, dict) and bool(constraints.get("allow_dependency_addition"))

    @staticmethod
    def _infer_allowlisted_dependency_package(output_excerpt: str) -> str | None:
        match = MISSING_MODULE_PATTERN.search(output_excerpt)
        if match is None:
            return None
        module_name = match.group(1).split(".", 1)[0].lower()
        return ALLOWLISTED_DEPENDENCY_PACKAGES.get(module_name)

    @staticmethod
    def _workspace_file_exists(record: JobRecord, relative_path: str) -> bool:
        normalized = Path(*PurePosixPath(relative_path.replace("\\", "/")).parts)
        workspace_root = Path(record.spec.workspace_root or record.spec.repo_path).resolve()
        target = (workspace_root / normalized).resolve()
        return workspace_root in [target, *target.parents] and target.is_file()

    def _record_test_failure(self, record: JobRecord, test_result: TestRunResult) -> None:
        if not self.policy.is_tool_allowed("fixer", "memory_server.write_memory"):
            return
        failure_index = record.failure_count + 1
        self._call_tool(
            "fixer",
            "memory_server.write_memory",
            uri=f"memory://{record.job_id}/test_failure_{failure_index}",
            content=json.dumps(
                {
                    "failed_tests": test_result.failed_tests,
                    "exit_code": test_result.exit_code,
                    "output_excerpt": test_result.output_excerpt,
                },
                sort_keys=True,
            ),
        )

    def _read_memory(self, role: str) -> list[str]:
        if not self.policy.is_tool_allowed(role, "memory_server.read_memory"):
            return []
        payload = self._call_tool(role, "memory_server.read_memory", limit=5)
        return [str(item["value"]) for item in payload.get("entries", [])]

    def _write_memory_item(self, record: JobRecord, role: str, key: str, value: str) -> None:
        if not self.policy.is_tool_allowed(role, "memory_server.write_memory"):
            return
        written = set(record.runtime_state.get("memory_writes", []))
        write_key = f"{role}:{key}"
        if write_key in written:
            return
        self._call_tool(
            role,
            "memory_server.write_memory",
            uri=f"memory://{record.job_id}/{key}",
            content=value,
        )
        written.add(write_key)
        record.runtime_state["memory_writes"] = sorted(written)
        self.store.update(record)

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
        if record.runtime_state.get("released"):
            return
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
        record.runtime_state["released"] = True
        self.store.update(record)

    def _call_tool(self, role: str, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        self.policy.assert_tool_allowed(role, tool_name)
        if self._active_record is None:
            raise RuntimeError("no active record for tool call")
        workspace_root = (
            self._active_record.spec.workspace_root or self._active_record.spec.repo_path
        )
        decision = self.policy.classify_tool_call(
            role=role,
            tool_name=tool_name,
            arguments=kwargs,
            workspace_root=workspace_root,
            job_metadata=self._active_record.spec.metadata,
        )
        if (
            decision.policy_action == PolicyAction.REQUIRE_APPROVAL
            and self._consume_approved_operation(role, tool_name, kwargs)
        ):
            decision = decision.model_copy(
                update={
                    "policy_action": PolicyAction.ALLOW_AND_AUDIT,
                    "reason": "previously approved operation matched and was resumed",
                    "details": {
                        **decision.details,
                        "approval_resumed": True,
                    },
                }
            )
        self._active_record.audit_events.append(
            self.audit.policy_event(
                role=role,
                job_id=self._active_record.job_id,
                task_id=self._active_record.current_task_id,
                decision=decision,
            )
        )
        if decision.policy_action == PolicyAction.DENY:
            raise PermissionError(decision.reason)
        if decision.policy_action == PolicyAction.REQUIRE_APPROVAL:
            raise ApprovalRequiredError(
                requested_by=role,
                operation=decision.operation,
                decision=decision,
                proposed_action={"tool_name": tool_name, "arguments": kwargs},
                task_id=self._active_record.current_task_id,
            )
        result = self.router.call(tool_name, **kwargs)
        status = "success" if result.ok else "failed"
        event = self.audit.tool_event(
            role=role,
            tool_name=tool_name,
            input_payload=kwargs,
            output_payload=result.data,
            status=status,
        )
        self._active_record.audit_events.append(event)
        if result.ok:
            return result.data
        raise RuntimeError(result.error or f"tool call failed: {tool_name}")

    def _consume_approved_operation(
        self,
        role: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        if self._active_record is None:
            return False
        approved_operation = self._active_record.runtime_state.get("approved_operation")
        if not isinstance(approved_operation, dict):
            return False
        details = approved_operation.get("details")
        if not isinstance(details, dict):
            return False
        approved_tool = details.get("tool_name")
        approved_arguments = details.get("arguments")
        if approved_tool != tool_name or not isinstance(approved_arguments, dict):
            return False
        if approved_operation.get("requested_by") != role:
            return False
        if json.dumps(approved_arguments, sort_keys=True, default=str) != json.dumps(
            arguments,
            sort_keys=True,
            default=str,
        ):
            return False
        self._active_record.runtime_state.pop("approved_operation", None)
        return True

    def _prepare_branch(self, record: JobRecord) -> None:
        if record.runtime_state.get("branch_prepared"):
            return
        record.current_role = "orchestrator"
        record.current_task_id = None
        self._call_tool("orchestrator", "git_server.create_branch", branch=record.spec.target_branch)
        record.runtime_state["branch_prepared"] = True
        self.store.update(record)

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
                reuse_existing=False,
            )
            security_review = self._run_structured_role(
                record,
                "security_reviewer",
                SecurityReviewResult,
                "Review the changes for security risks",
                task=primary_task,
                security_sensitive=True,
                reuse_existing=False,
            )
            try:
                ensure_reviews_pass(review, security_review)
                return review, security_review
            except QualityGateError:
                attempts += 1
                if attempts >= self.max_attempts_per_task:
                    raise
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
                    reuse_existing=not self._should_force_fixer_rerun(record),
                )
                ensure_fixer_safe(fix.patches)
                self._apply_patches(record, "fixer", fix.patches, task=primary_task)
                self._mark_fixer_consumed(record)

    @staticmethod
    def _choose_active_task(task_graph: TaskGraph) -> PlannedTask | None:
        if not task_graph.tasks:
            return None
        terminal_statuses = {
            TaskStatus.DONE,
            TaskStatus.BLOCKED,
            TaskStatus.STUCK,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
        }
        active_statuses = {
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.WAITING_RUNTIME,
            TaskStatus.PAUSED,
            TaskStatus.RESUMING,
            TaskStatus.IN_PROGRESS,
            TaskStatus.IMPLEMENTED,
            TaskStatus.TESTS_WRITTEN,
            TaskStatus.UNDER_REVIEW,
            TaskStatus.CHANGES_REQUESTED,
            TaskStatus.TEST_RUNNING,
            TaskStatus.TEST_FAILED,
        }
        task_by_id = {task.id: task for task in task_graph.tasks}
        for task in task_graph.tasks:
            if task.status in active_statuses:
                return task
        for task in task_graph.tasks:
            if task.status not in {TaskStatus.TODO, TaskStatus.QUEUED}:
                continue
            dependency_ids = list(dict.fromkeys([*task.dependencies, *task.depends_on]))
            if all(
                dependency_id in task_by_id
                and task_by_id[dependency_id].status == TaskStatus.DONE
                for dependency_id in dependency_ids
            ):
                return task
        if all(task.status in terminal_statuses for task in task_graph.tasks):
            return None
        return next((task for task in task_graph.tasks if task.status not in terminal_statuses), None)

    @staticmethod
    def _should_force_fixer_rerun(record: JobRecord) -> bool:
        return bool(record.runtime_state.get("fixer_consumed"))

    @staticmethod
    def _mark_fixer_consumed(record: JobRecord) -> None:
        record.runtime_state["fixer_consumed"] = True

    def _set_task_status(
        self,
        record: JobRecord,
        task_id: str | None,
        status: TaskStatus,
        approval_id: str | None = None,
    ) -> None:
        if task_id is None:
            return
        task_graph = record.outputs.get("planner") or record.outputs.get("task_graph")
        if not isinstance(task_graph, dict):
            return
        tasks = task_graph.get("tasks")
        if not isinstance(tasks, list):
            return
        changed = False
        for item in tasks:
            if not isinstance(item, dict) or item.get("id") != task_id:
                continue
            item["status"] = status.value
            item["approval_id"] = approval_id
            changed = True
            break
        if changed:
            if "planner" in record.outputs and isinstance(record.outputs["planner"], dict):
                record.outputs["planner"] = task_graph
            record.outputs["task_graph"] = task_graph
            try:
                task_record = self.store.get_task(record.job_id, task_id)
            except KeyError:
                return
            task_record.status = status
            task_record.pending_approval_id = approval_id
            task_record.updated_at = record.updated_at
            self.store.upsert_task(task_record)


def build_default_runner(
    config_dir: str | Path = "configs",
    workspace_root: str | Path = ".",
    memory_db_path: str | Path | None = None,
    approval_db_path: str | Path | None = None,
    job_store_path: str | Path | None = None,
) -> tuple[JobRunner, FakeMCPEnvironment]:
    """Build a JobRunner wired to the local config directory and fake MCP tools."""
    config_path = Path(config_dir)
    workspace_path = Path(workspace_root).resolve()
    if workspace_path.exists() and not workspace_path.is_dir():
        raise ValueError(f"workspace_root must be a directory: {workspace_path}")
    workspace_path.mkdir(parents=True, exist_ok=True)
    acos_dir = workspace_path / ".acos"
    acos_dir.mkdir(parents=True, exist_ok=True)
    memory_db = Path(memory_db_path or (workspace_path / ".acos_memory.sqlite3"))
    runtime_db = Path(job_store_path or approval_db_path or (acos_dir / "acos.sqlite3"))
    legacy_approval_db = workspace_path / ".acos_approvals.sqlite3"
    if not legacy_approval_db.exists():
        legacy_approval_db.touch()
    registry = ModelRegistry.from_paths(
        provider_path=config_path / "model_providers.yaml",
        agents_path=config_path / "agents.yaml",
        routing_path=config_path / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_path / "policies.yaml")
    registry.validate_or_raise(policy=policy)
    workspace_policy = policy.build_workspace_policy(workspace_path)
    env = FakeMCPEnvironment(
        workspace_root=workspace_path,
        memory_db_path=memory_db,
        workspace_policy=workspace_policy,
    )
    approval_gateway = ApprovalGateway(
        SQLiteApprovalStore(approval_db_path or runtime_db),
        request_ttl_minutes=policy.config.approval.request_ttl_minutes,
        allow_cli_approval=policy.config.approval.allow_cli_approval,
        allow_http_approval=policy.config.approval.allow_http_approval,
        allow_notification_links=policy.config.approval.allow_notification_links,
        require_signed_tokens=policy.config.approval.require_signed_tokens,
    )
    runtime_payload = yaml.safe_load(
        (config_path / "runtime.yaml").read_text(encoding="utf-8")
    ) if (config_path / "runtime.yaml").exists() else {}
    runtime_config = RuntimeConfig(**(runtime_payload.get("runtime") or {}))
    health_checker = ProviderHealthChecker(
        registry,
        config=runtime_config.provider_health_check,
    )
    store: JobStore
    if Path(runtime_db).suffix == ".json":
        store = InMemoryJobStore(runtime_db)
    else:
        store = SQLiteJobStore(runtime_db)
    runtime_manager = RuntimeManager(
        store=store,
        health_checker=health_checker,
        config=runtime_config,
    )
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=env.build_router(),
        store=store,
        approval_gateway=approval_gateway,
        runtime_manager=runtime_manager,
        token_budget_policy=runtime_config.token_budget,
    )
    return runner, env
