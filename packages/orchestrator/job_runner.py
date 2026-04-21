"""ACOS job orchestration engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypeVar

import yaml

from packages.agents.runner import AgentRunner
from packages.llm.budget import estimate_tokens
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
from packages.orchestrator.job_store import InMemoryJobStore, JobStore, SQLiteJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.orchestrator.quality_gates import (
    QualityGateError,
    ensure_fixer_safe,
    ensure_reviews_pass,
    ensure_test_patch_quality,
)
from packages.orchestrator.runtime import ProviderUnavailableError, RuntimeManager
from packages.orchestrator.states import apply_transition
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
from packages.schemas.approvals import PolicyAction
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import FixStatus, JobStatus, TaskStatus
from packages.schemas.runtime import RuntimeConfig, RuntimeIssueType
from packages.schemas.tasks import PlannedTask, TaskGraph, TaskRecord

T = TypeVar("T")


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
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.router = router
        self.store = store or InMemoryJobStore()
        self.audit = AuditRecorder()
        self.context_builder = ContextBuilder()
        self.model_router = model_router or ModelRouter(registry)
        self.llm_client = LLMClient(registry, self.model_router)
        self.approval_gateway = approval_gateway
        self.agent_runner = agent_runner or AgentRunner(
            llm_client=self.llm_client,
            registry=registry,
            mcp_router=router,
            policy_engine=policy,
            audit_recorder=self.audit,
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
            if "architect" not in record.outputs:
                return self.run_architect_step(record)
            if "planner" not in record.outputs:
                return self.run_planner_step(record)
            primary_task = self._choose_primary_task(
                TaskGraph.model_validate(record.outputs["planner"])
                if "planner" in record.outputs
                else TaskGraph.model_validate(record.outputs["task_graph"])
            )
            if primary_task is not None and not self.checkpoints.has_completed(
                job_id=record.job_id,
                task_id=primary_task.id,
                checkpoint_key=f"task:{primary_task.id}:implementer_completed",
            ):
                return self.run_task_implementer_step(record, primary_task)
            if primary_task is not None and not self.checkpoints.has_completed(
                job_id=record.job_id,
                task_id=primary_task.id,
                checkpoint_key=f"task:{primary_task.id}:test_writer_completed",
            ):
                return self.run_task_test_writer_step(record, primary_task)
            if primary_task is not None and not self.checkpoints.has_completed(
                job_id=record.job_id,
                task_id=primary_task.id,
                checkpoint_key=f"task:{primary_task.id}:review_completed",
            ):
                return self.run_task_review_step(record, primary_task)
            if "test_run" not in record.outputs or record.runtime_state.get("needs_retest"):
                return self.run_task_test_step(record, primary_task)
            test_result = TestRunResult.model_validate(record.outputs["test_run"])
            if not test_result.success:
                if record.failure_count >= self.max_attempts_per_task:
                    record.status = JobStatus.STUCK
                    record.last_error = "max_attempts_exceeded"
                    return self.store.update(record)
                return self.run_task_fixer_step(record, primary_task, test_result)
            if not self.checkpoints.has_completed(
                job_id=record.job_id,
                checkpoint_key="final_quality_gates_completed",
                task_id=primary_task.id if primary_task is not None else None,
            ):
                return self.run_final_quality_gates_step(record, primary_task)
            if not self.checkpoints.has_completed(
                job_id=record.job_id,
                checkpoint_key="release_completed",
                task_id=primary_task.id if primary_task is not None else None,
            ):
                return self.run_release_step(record, primary_task)
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
        record.outputs["prd"] = prd.model_dump()
        self._write_memory_item(record, "pm", "prd", prd.model_dump_json())
        return self._mark_step_completed(
            record=record,
            checkpoint_key="pm_completed",
            step_name="pm",
            result_json={"title": prd.title},
        )

    def run_architect_step(self, record: JobRecord) -> JobRecord:
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
            "Design the system architecture",
            reuse_existing=True,
        )
        record.outputs["architecture"] = architecture.model_dump()
        self._write_memory_item(record, "architect", "architecture", architecture.model_dump_json())
        return self._mark_step_completed(
            record=record,
            checkpoint_key="architecture_completed",
            step_name="architect",
            result_json={"summary": architecture.summary},
        )

    def run_planner_step(self, record: JobRecord) -> JobRecord:
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
            "Create the implementation task graph",
            reuse_existing=True,
        )
        record.outputs["task_graph"] = task_graph.model_dump()
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
        review, security_review = self._run_review_cycle(record, task)
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
        self._set_task_status(
            record,
            checkpoint_task_id,
            TaskStatus.DONE if test_result.success else TaskStatus.TEST_FAILED,
        )
        return self._mark_step_completed(
            record=record,
            checkpoint_key=f"task:{checkpoint_task_id or 'job'}:tests_completed",
            step_name="tests",
            task_id=checkpoint_task_id,
            result_json={"success": test_result.success},
        )

    def run_task_fixer_step(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        test_result: TestRunResult,
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
            "Fix the deterministic test failures",
            task=task,
            logs=[test_result.output_excerpt],
            reuse_existing=not self._should_force_fixer_rerun(record),
        )
        ensure_fixer_safe(fix.patches)
        self._apply_patches(record, "fixer", fix.patches, task=task)
        self._mark_fixer_consumed(record)
        record.failure_count += 1
        record.same_test_failure_count += 1 if test_result.failed_tests else 0
        if fix.status == FixStatus.STUCK:
            record.status = JobStatus.STUCK
            return self.store.update(record)
        if record.same_test_failure_count >= self.max_same_failure_repeats:
            record.status = JobStatus.STUCK
            record.last_error = "same_failure_threshold_reached"
            return self.store.update(record)
        record.runtime_state["needs_retest"] = True
        record.outputs.pop("test_run", None)
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
    ) -> T:
        if reuse_existing and role in record.outputs:
            return response_model.model_validate(record.outputs[role])
        if record.status != self._phase_for_role(role):
            apply_transition(record, self._phase_for_role(role))
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
            metadata={"output_schema": agent_cfg.output_schema},
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
        record.outputs[role] = output.model_dump()
        record.outputs[f"{role}_model_selection"] = selection.model_dump()
        self.store.update(record)
        return output

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
        if role == "test_writer":
            ensure_test_patch_quality(patches, role=role)
        applied = set(record.runtime_state.get("applied_patches", []))
        for index, patch in enumerate(patches):
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

    def _run_tests(self, record: JobRecord) -> TestRunResult:
        if record.status != JobStatus.TESTING:
            apply_transition(record, JobStatus.TESTING)
        record.current_role = "runner"
        record.current_task_id = None
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="pytest",
            timeout_seconds=120,
        )
        result = TestRunResult.model_validate(payload)
        record.outputs["test_run"] = result.model_dump()
        self.store.update(record)
        return result

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
    def _choose_primary_task(task_graph: TaskGraph) -> PlannedTask | None:
        return task_graph.tasks[0] if task_graph.tasks else None

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
    )
    return runner, env
