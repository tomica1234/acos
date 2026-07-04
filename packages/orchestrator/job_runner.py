"""ACOS job orchestration engine."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.agents.runner import AgentRunner
from packages.llm.budget import estimate_tokens
from packages.llm.client import LLMClient
from packages.llm.registry import ModelRegistry
from packages.llm.routing import ModelRouter, RoutingContext
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.context_builder import ContextBuilder
from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.progress import summarize_job_progress
from packages.orchestrator.quality_gates import (
    QualityGateError,
    ensure_fixer_safe,
    ensure_reviews_pass,
)
from packages.orchestrator.scaffolds import build_scaffold
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
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    JobStatus,
    TaskComplexity,
    TestWriterStatus,
)
from packages.schemas.tasks import PlannedTask, TaskGraph


def _disable_mock_fallback_models(registry: ModelRegistry) -> None:
    for agent in registry.agents.values():
        agent.fallback_models = [
            model_key
            for model_key in agent.fallback_models
            if registry.get_provider(registry.get_model(model_key).provider).type.value != "mock"
        ]


class JobRunner:
    """Run ACOS jobs across explicit role phases."""

    CONTEXT_ONLY_ROLES = {"pm", "architect", "planner", "implementer", "test_writer", "fixer"}
    IMPLEMENTATION_TASK_ROLES = {"architect", "implementer"}
    TEST_TASK_ROLES = {"test_writer"}

    def __init__(
        self,
        registry: ModelRegistry,
        policy: PolicyEngine,
        router: MCPRouter,
        store: InMemoryJobStore | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.router = router
        self.store = store or InMemoryJobStore()
        self.audit = AuditRecorder()
        self.context_builder = ContextBuilder()
        self.model_router = ModelRouter(registry)
        self.llm_client = LLMClient(registry, self.model_router)
        self.agent_runner = AgentRunner(
            llm_client=self.llm_client,
            registry=registry,
            mcp_router=router,
            policy_engine=policy,
            audit_recorder=self.audit,
        )
        self.max_attempts_per_task = 3
        self.max_same_failure_repeats = 2
        self.max_steps_per_agent = 6
        self._active_record: JobRecord | None = None

    def submit(self, spec: JobSpec) -> JobRecord:
        return self.store.create(spec)

    def get(self, job_id: str) -> JobRecord:
        return self.store.get(job_id)

    def run_job(self, spec: JobSpec) -> JobRecord:
        record = self.store.create(spec)
        return self._run_record(record, resume=False)

    def plan_job(self, spec: JobSpec) -> JobRecord:
        record = self.store.create(spec)
        return self._plan_record(record, resume=False)

    def resume_job(self, job_id: str) -> JobRecord:
        record = self.store.get(job_id)
        return self._run_record(record, resume=True)

    def _plan_record(self, record: JobRecord, *, resume: bool) -> JobRecord:
        self._active_record = record
        try:
            if record.status == JobStatus.DONE:
                return record
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
            return self.store.update(record)
        except QualityGateError as exc:
            record.status = JobStatus.BLOCKED
            record.last_error = str(exc)
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            record.status = JobStatus.FAILED
            record.last_error = str(exc)
            return self.store.update(record)
        finally:
            self._active_record = None

    def _run_record(self, record: JobRecord, *, resume: bool) -> JobRecord:
        self._active_record = record
        try:
            if record.status == JobStatus.DONE:
                return record
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
            if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                return self.store.update(record)
            if not self._constraint_flag(record, "skip_review"):
                review, security_review = self._run_review_cycle(record, primary_task)
                if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                    return self.store.update(record)
                test_result = self._run_tests_with_fixes(record, primary_task)
            else:
                if record.status != JobStatus.TESTING:
                    apply_transition(record, JobStatus.REVIEWING)
            if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                return self.store.update(record)
            if not test_result.success:
                record.status = JobStatus.FAILED
                record.last_error = "tests_failed_after_retries"
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
            apply_transition(record, JobStatus.DONE)
            return self.store.update(record)
        except QualityGateError as exc:
            record.status = JobStatus.BLOCKED
            record.last_error = str(exc)
            return self.store.update(record)
        except Exception as exc:  # pragma: no cover - top-level safety net
            record.status = JobStatus.FAILED
            record.last_error = str(exc)
            return self.store.update(record)
        finally:
            self._active_record = None

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
        apply_transition(record, self._phase_for_role(role))
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
        output, selection, model_record = self.agent_runner.run(
            role=role,
            response_model=response_model,
            context_packet=packet,
            routing_context=routing_context,
            allowed_tools=self._allowed_tools_for_role(role),
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
        elif candidates:
            files["__repo_tree__.txt"] = "\n".join(candidates)
        return files

    def _allowed_tools_for_role(self, role: str) -> list[str]:
        agent_cfg = self.registry.get_agent(role)
        if not agent_cfg.allow_tools:
            return []
        if role in self.CONTEXT_ONLY_ROLES:
            return []
        return list(agent_cfg.allowed_tools)

    def _context_constraints(self, record: JobRecord) -> list[str]:
        constraints = [f"blocked_operation={item}" for item in self.policy.config.blocked_operations]
        job_constraints = self._constraints(record)
        for key in sorted(job_constraints):
            value = job_constraints[key]
            if isinstance(value, (str, int, float, bool)):
                constraints.append(f"job_constraint {key}={value}")
        return constraints

    @staticmethod
    def _clear_planning_repair_constraints(record: JobRecord) -> None:
        constraints = record.spec.metadata.get("constraints")
        if not isinstance(constraints, dict):
            return
        for key in list(constraints):
            if key.startswith("planning_repair_"):
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
        return guidance

    def _recovery_history_logs(self, record: JobRecord, role: str) -> list[str]:
        if role not in {
            "pm",
            "architect",
            "planner",
            "implementer",
            "test_writer",
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
        return logs

    def _pm_stall_guidance_logs(self, record: JobRecord, role: str) -> list[str]:
        if role not in {"pm", "planner", "architect", "implementer", "test_writer", "fixer"}:
            return []
        constraints = self._constraints(record)
        if constraints.get("pm_stall_recovery") is not True:
            return []
        strategy = constraints.get("pm_strategy", "unknown")
        focus_task_id = constraints.get("pm_focus_task_id", "unknown")
        reason = constraints.get("pm_reason", "same_progress_marker_repeated")
        logs = [
            (
                "pm_stall_recovery: "
                f"strategy={strategy}; focus_task_id={focus_task_id}; reason={reason}"
            )
        ]
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
        return None

    def _apply_patches(self, record: JobRecord, role: str, patches: list[Any]) -> None:
        max_patches = self._constraint_int(record, "max_patches_per_agent_output", 0)
        if max_patches and len(patches) > max_patches:
            raise QualityGateError(
                f"patch_limit_exceeded:{role}:{len(patches)}>{max_patches}"
            )
        for patch in patches:
            self._call_tool(
                role,
                "repo_server.apply_patch",
                path=patch.path,
                content=patch.content,
                operation=patch.operation,
            )
        self.store.update(record)

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
        return result

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
        return self._refine_prd_quality_for_autonomy(record, prd)

    def _refine_prd_quality_for_autonomy(self, record: JobRecord, prd: PRD) -> PRD | None:
        report = self._build_prd_quality_report(prd)
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
        for attempt in range(1, refinement_attempts + 1):
            current_prd = self._run_structured_role(
                record,
                "pm",
                PRD,
                (
                    "Refine the product requirements before implementation. "
                    "Fill every missing PRD quality field for autonomous large-scale execution: "
                    f"{', '.join(report['missing'])}."
                ),
                logs=[
                    "The previous PRD was not specific enough for autonomous execution.",
                    f"Missing fields: {', '.join(report['missing'])}",
                    f"Warnings: {', '.join(report['warnings'])}",
                ],
            )
            self._write_memory_item(record, "pm", "prd", current_prd.model_dump_json())
            report = self._build_prd_quality_report(current_prd)
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

        record.status = JobStatus.BLOCKED
        record.last_error = "prd_quality_gate_failed:" + ",".join(report["missing"])
        self.store.update(record)
        return None

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
    def _build_prd_quality_report(prd: PRD) -> dict[str, Any]:
        missing: list[str] = []
        warnings: list[str] = []
        if not prd.title.strip():
            missing.append("title")
        if not prd.problem_statement.strip():
            missing.append("problem_statement")
        if not JobRunner._non_empty_items(prd.smallest_working_core):
            missing.append("smallest_working_core")
        small_parts = JobRunner._non_empty_items(prd.small_parts)
        if not small_parts:
            missing.append("small_parts")
        elif len(small_parts) == 1:
            warnings.append("small_parts_has_single_item")
        if not JobRunner._non_empty_items(prd.incremental_milestones):
            missing.append("incremental_milestones")
        acceptance_tests = JobRunner._non_empty_items(prd.acceptance_tests)
        if not acceptance_tests:
            missing.append("acceptance_tests")
        elif small_parts and len(acceptance_tests) < len(small_parts):
            missing.append("acceptance_tests_cover_small_parts")
        if not JobRunner._non_empty_items(prd.definition_of_done):
            missing.append("definition_of_done")
        if prd.open_questions:
            warnings.append("open_questions_present")
        missing_acceptance_test_count = max(0, len(small_parts) - len(acceptance_tests))
        return {
            "passed": not missing,
            "missing": missing,
            "warnings": warnings,
            "small_part_count": len(small_parts),
            "acceptance_test_count": len(acceptance_tests),
            "acceptance_tests_cover_small_parts": missing_acceptance_test_count == 0,
            "missing_acceptance_test_count": missing_acceptance_test_count,
            "definition_of_done_count": len(JobRunner._non_empty_items(prd.definition_of_done)),
        }

    @staticmethod
    def _non_empty_items(items: list[str]) -> list[str]:
        return [item.strip() for item in items if item.strip()]

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
        for index, part in enumerate(small_parts, start=1):
            task_id = f"part-{index:02d}"
            criteria = (
                [acceptance_tests[index - 1]]
                if index <= len(acceptance_tests)
                else [f"{part} works and existing behavior remains covered by tests."]
            )
            task = PlannedTask(
                id=task_id,
                title=self._task_title_from_part(part),
                description=(
                    "Implement only this small part before moving on: "
                    f"{part}. Keep the change narrow enough to test immediately."
                ),
                role="implementer",
                complexity=TaskComplexity.MEDIUM,
                depends_on=[previous_id] if previous_id is not None else [],
                acceptance_criteria=criteria,
            )
            tasks.append(task)
            previous_id = task_id

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
        }
        self.store.update(record)
        return refined

    def _enrich_task_graph_acceptance_criteria(
        self,
        record: JobRecord,
        prd: PRD,
        task_graph: TaskGraph,
    ) -> TaskGraph:
        acceptance_tests = self._non_empty_items(prd.acceptance_tests)
        definition_of_done = self._non_empty_items(prd.definition_of_done)
        if not acceptance_tests and not definition_of_done:
            record.outputs["task_graph_acceptance_enrichment"] = {
                "applied": False,
                "reason": "no_prd_acceptance_sources",
                "updated_task_ids": [],
            }
            return task_graph

        updated_task_ids: list[str] = []
        implementation_index = 0
        tasks: list[PlannedTask] = []
        for task in task_graph.tasks:
            if task.role not in self.IMPLEMENTATION_TASK_ROLES:
                tasks.append(task)
                continue
            if self._non_empty_items(task.acceptance_criteria):
                tasks.append(task)
                implementation_index += 1
                continue
            criteria = self._criteria_for_task_from_prd(
                task,
                acceptance_tests,
                definition_of_done,
                implementation_index,
            )
            tasks.append(task.model_copy(update={"acceptance_criteria": criteria}))
            updated_task_ids.append(task.id)
            implementation_index += 1

        if not updated_task_ids:
            record.outputs["task_graph_acceptance_enrichment"] = {
                "applied": False,
                "reason": "all_implementation_tasks_already_have_criteria",
                "updated_task_ids": [],
            }
            return task_graph

        record.outputs["task_graph_acceptance_enrichment"] = {
            "applied": True,
            "reason": "filled_missing_task_acceptance_criteria_from_prd",
            "updated_task_ids": updated_task_ids,
        }
        return TaskGraph(
            goal=task_graph.goal,
            tasks=tasks,
            notes=[
                *task_graph.notes,
                "ACOS filled missing task acceptance_criteria from the PRD.",
            ],
        )

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
                    "testable acceptance_criteria on every implementer task, "
                    "and only autonomous-executable task roles."
                ),
                logs=[
                    "The previous task graph failed autonomy validation.",
                    f"Validation errors: {validation['errors']}",
                    f"PRD small_parts: {self._non_empty_items(prd.small_parts)}",
                ],
            )
            self._write_memory_item(record, "planner", "task_graph", task_graph.model_dump_json())
            task_graph = self._refine_task_graph_for_autonomy(record, prd, task_graph)
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

        record.status = JobStatus.BLOCKED
        record.last_error = "invalid_task_graph"
        self.store.update(record)
        return None

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
        attempts.append(
            {
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
            }
        )

    def _validate_task_graph_for_autonomy(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
    ) -> bool:
        validation = self._build_task_graph_validation(task_graph)
        record.outputs["task_graph_validation"] = validation
        if validation["valid"]:
            self.store.update(record)
            return True
        record.status = JobStatus.BLOCKED
        record.last_error = "invalid_task_graph"
        self.store.update(record)
        return False

    @staticmethod
    def _build_task_graph_validation(
        task_graph: TaskGraph,
        prd: PRD | None = None,
        require_acceptance_criteria: bool = False,
        require_executable_task_roles: bool = False,
    ) -> dict[str, Any]:
        ids = [task.id for task in task_graph.tasks]
        duplicate_ids = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
        id_set = set(ids)
        implementation_task_ids = [
            task.id for task in task_graph.tasks if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
        ]
        small_parts = JobRunner._non_empty_items(prd.small_parts) if prd is not None else []
        acceptance_tests = (
            JobRunner._non_empty_items(prd.acceptance_tests) if prd is not None else []
        )
        small_part_coverage = [
            {
                "small_part_index": index,
                "small_part": small_part,
                "task_id": (
                    implementation_task_ids[index - 1]
                    if index <= len(implementation_task_ids)
                    else None
                ),
                "covered": index <= len(implementation_task_ids),
            }
            for index, small_part in enumerate(small_parts, start=1)
        ]
        acceptance_test_coverage = [
            {
                "acceptance_test_index": index,
                "acceptance_test": acceptance_test,
                "task_id": (
                    implementation_task_ids[index - 1]
                    if index <= len(implementation_task_ids)
                    else None
                ),
                "covered": index <= len(implementation_task_ids),
            }
            for index, acceptance_test in enumerate(acceptance_tests, start=1)
        ]
        uncovered_small_parts = [
            item for item in small_part_coverage if not item["covered"]
        ]
        uncovered_acceptance_tests = [
            item for item in acceptance_test_coverage if not item["covered"]
        ]
        unknown_dependencies = [
            {"task_id": task.id, "dependency": dependency}
            for task in task_graph.tasks
            for dependency in task.depends_on
            if dependency not in id_set
        ]
        cycle = JobRunner._find_task_graph_cycle(task_graph)
        errors: list[dict[str, Any]] = []
        if not task_graph.tasks:
            errors.append({"type": "empty_task_graph"})
        elif not implementation_task_ids:
            errors.append({"type": "missing_implementation_tasks"})
        elif small_parts and len(implementation_task_ids) < len(small_parts):
            errors.append(
                {
                    "type": "undercovered_small_parts",
                    "small_part_count": len(small_parts),
                    "implementation_task_count": len(implementation_task_ids),
                    "uncovered_small_parts": uncovered_small_parts,
                }
            )
        if duplicate_ids:
            errors.append({"type": "duplicate_task_ids", "task_ids": duplicate_ids})
        if unknown_dependencies:
            errors.append({"type": "unknown_dependencies", "items": unknown_dependencies})
        if cycle:
            errors.append({"type": "dependency_cycle", "task_ids": cycle})
        tasks_missing_acceptance_criteria = [
            task.id
            for task in task_graph.tasks
            if task.role in JobRunner.IMPLEMENTATION_TASK_ROLES
            and not JobRunner._non_empty_items(task.acceptance_criteria)
        ]
        if require_acceptance_criteria and tasks_missing_acceptance_criteria:
            errors.append(
                {
                    "type": "missing_acceptance_criteria",
                    "task_ids": tasks_missing_acceptance_criteria,
                }
            )
        executable_roles = JobRunner.IMPLEMENTATION_TASK_ROLES | JobRunner.TEST_TASK_ROLES
        unsupported_task_roles = [
            {"task_id": task.id, "role": task.role}
            for task in task_graph.tasks
            if task.role not in executable_roles
        ]
        if require_executable_task_roles and unsupported_task_roles:
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
            "implementation_task_acceptance_criteria_count": (
                len(implementation_task_ids) - len(tasks_missing_acceptance_criteria)
            ),
            "require_acceptance_criteria": require_acceptance_criteria,
            "require_executable_task_roles": require_executable_task_roles,
            "unsupported_task_role_count": len(unsupported_task_roles),
            "small_part_count": len(small_parts),
            "small_part_coverage": small_part_coverage,
            "uncovered_small_parts": uncovered_small_parts,
            "acceptance_test_count": len(acceptance_tests),
            "acceptance_test_coverage": acceptance_test_coverage,
            "uncovered_acceptance_tests": uncovered_acceptance_tests,
            "errors": errors,
        }

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
        implementation_tasks = self._tasks_for_roles(task_graph, self.IMPLEMENTATION_TASK_ROLES)
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
        ready_task_ids: set[str] = set(completed_task_ids)
        recorded_implementation_task_ids = self._recorded_task_ids(
            record,
            "implementation_tasks",
            allowed_result_statuses={ImplementationStatus.IMPLEMENTED.value},
        )
        pending_test_tasks = [task for task in pending_test_tasks if task.id not in completed_task_ids]
        last_test_result = self._synthetic_test_result(success=True, output="No tests run yet.")

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
                stage_test_pairs = self._run_ready_test_tasks(
                    record=record,
                    pending_test_tasks=pending_test_tasks,
                    ready_task_ids=ready_task_ids,
                    implementation_results=implementation_results,
                    test_writer_results=test_writer_results,
                )
                if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
                    if record.status == JobStatus.STUCK or not last_test_result.success:
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
                stage_test_pairs: list[tuple[PlannedTask, TestWriterResult]] = []
                ready_task_ids.add(task.id)
                stage_test_pairs = self._run_ready_test_tasks(
                    record=record,
                    pending_test_tasks=pending_test_tasks,
                    ready_task_ids=ready_task_ids,
                    implementation_results=implementation_results,
                    test_writer_results=test_writer_results,
                )
                if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
                    if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
                if record.status == JobStatus.STUCK or not last_test_result.success:
                    return implementation_results, test_writer_results, last_test_result, stage_results
                stage_review = self._run_stage_review_gate(record, task)
                if stage_review is not None:
                    stage_result["stage_review"] = stage_review
                    self.store.update(record)
                    if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                        return implementation_results, test_writer_results, last_test_result, stage_results
                    last_test_result = self._run_tests_with_fixes(
                        record,
                        task,
                        logs=["stage review applied fixes"],
                    )
                    stage_result["post_review_test_run"] = last_test_result.model_dump()
                    self.store.update(record)
                    if (
                        record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}
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
            unmet_dependencies = [
                dependency for dependency in task.depends_on if dependency not in completed_task_ids
            ]
            if unmet_dependencies:
                record.status = JobStatus.STUCK
                record.last_error = f"unmet_task_dependencies:{','.join(unmet_dependencies)}"
                return implementation_results, test_writer_results, last_test_result, stage_results
            implementation = self._run_structured_role(
                record,
                "implementer",
                ImplementationResult,
                f"Implement the next autonomous stage task {task.id}: {task.title}",
                task=task,
                logs=implementation_summaries,
            )
            implementation_results.append(implementation)
            implementation_summaries.append(f"{task.id}: {implementation.summary}")
            self._record_task_output(record, "implementation_tasks", task, implementation)
            if not self._implementation_allows_progress(record, task, implementation):
                return implementation_results, test_writer_results, last_test_result, stage_results
            self._apply_patches(record, "implementer", implementation.patches)
            ready_task_ids.add(task.id)

            stage_test_pairs = self._run_ready_test_tasks(
                record=record,
                pending_test_tasks=pending_test_tasks,
                ready_task_ids=ready_task_ids,
                implementation_results=implementation_results,
                test_writer_results=test_writer_results,
            )
            if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
                if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
            if record.status == JobStatus.STUCK or not last_test_result.success:
                return implementation_results, test_writer_results, last_test_result, stage_results
            stage_review = self._run_stage_review_gate(record, task)
            if stage_review is not None:
                stage_result["stage_review"] = stage_review
                self.store.update(record)
                if record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}:
                    return implementation_results, test_writer_results, last_test_result, stage_results
                last_test_result = self._run_tests_with_fixes(
                    record,
                    task,
                    logs=["stage review applied fixes"],
                )
                stage_result["post_review_test_run"] = last_test_result.model_dump()
                self.store.update(record)
                if (
                    record.status in {JobStatus.STUCK, JobStatus.BLOCKED, JobStatus.FAILED}
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
            test_writer = self._run_test_writer_task(
                record,
                task,
                implementation_results,
                test_writer_results,
            )
            test_writer_results.append(test_writer)
            if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
            if record.status == JobStatus.STUCK or not last_test_result.success:
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
            if record.status == JobStatus.STUCK or not last_test_result.success:
                return implementation_results, test_writer_results, last_test_result, stage_results
            if primary_task is not None:
                self._mark_task_completed(record, primary_task.id)

        return implementation_results, test_writer_results, last_test_result, stage_results

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
            if record.status in {JobStatus.BLOCKED, JobStatus.FAILED}:
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
        while not test_result.success and attempts < self.max_attempts_per_task:
            fix = self._run_structured_role(
                record,
                "fixer",
                FixResult,
                "Fix only the current autonomous stage test failures",
                task=task,
                logs=[*(logs or []), test_result.output_excerpt],
            )
            attempts += 1
            record.failure_count += 1
            if test_result.failed_tests:
                same_failure_repeats += 1
                record.same_test_failure_count += 1
            else:
                same_failure_repeats = 0
                record.same_test_failure_count = 0
            self.store.update(record)
            if not self._fixer_allows_progress(record, task, fix):
                return test_result
            ensure_fixer_safe(fix.patches)
            self._apply_patches(record, "fixer", fix.patches)
            if same_failure_repeats >= self.max_same_failure_repeats:
                record.status = JobStatus.STUCK
                record.last_error = "same_failure_threshold_reached"
                return test_result
            test_result = self._run_tests(record)
        return test_result

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
        for task in self._tasks_for_roles(task_graph, self.IMPLEMENTATION_TASK_ROLES):
            objective = f"Implement planned task {task.id}: {task.title}"
            implementation = self._run_structured_role(
                record,
                "implementer",
                ImplementationResult,
                objective,
                task=task,
                logs=summaries,
            )
            results.append(implementation)
            summaries.append(f"{task.id}: {implementation.summary}")
            self._record_task_output(record, "implementation_tasks", task, implementation)
            if not self._implementation_allows_progress(record, task, implementation):
                return results
            self._apply_patches(record, "implementer", implementation.patches)
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
            objective = f"Add or update tests for planned task {task.id}: {task.title}"
            test_writer = self._run_structured_role(
                record,
                "test_writer",
                TestWriterResult,
                objective,
                task=task,
                logs=logs,
            )
            results.append(test_writer)
            logs.append(f"{task.id}: {test_writer.summary}")
            self._record_task_output(record, "test_writer_tasks", task, test_writer)
            if not self._test_writer_allows_progress(record, task, test_writer):
                return results
            self._apply_patches(record, "test_writer", test_writer.patches)
        return results

    def _implementation_allows_progress(
        self,
        record: JobRecord,
        task: PlannedTask | None,
        implementation: ImplementationResult,
    ) -> bool:
        if implementation.status == ImplementationStatus.IMPLEMENTED:
            return True
        task_id = task.id if task is not None else "unplanned"
        if implementation.status == ImplementationStatus.BLOCKED:
            record.status = JobStatus.BLOCKED
            record.last_error = f"implementation_blocked:{task_id}"
        else:
            record.status = JobStatus.FAILED
            record.last_error = f"implementation_failed:{task_id}"
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
            record.status = JobStatus.BLOCKED
            record.last_error = f"test_writer_blocked:{task_id}"
        else:
            record.status = JobStatus.FAILED
            record.last_error = f"test_writer_failed:{task_id}"
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
            record.status = JobStatus.STUCK
            record.last_error = f"fixer_stuck:{task_id}"
        else:
            record.status = JobStatus.FAILED
            record.last_error = f"fixer_failed:{task_id}"
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
        test_files = self._unique_paths(
            [
                path
                for result in test_writer_results
                for path in [
                    *result.changed_files,
                    *[patch.path for patch in result.patches],
                ]
            ]
        )
        test_patch_count = sum(len(result.patches) for result in test_writer_results)
        changed_files = self._unique_paths([*implementation_files, *test_files])
        return {
            "changed_files": changed_files,
            "implementation_files": implementation_files,
            "test_files": test_files,
            "implementation_patch_count": implementation_patch_count,
            "test_patch_count": test_patch_count,
            "patch_count": implementation_patch_count + test_patch_count,
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
        self.store.update(record)

    def _validate_completion_integrity(
        self,
        record: JobRecord,
        task_graph: TaskGraph,
        test_result: TestRunResult,
    ) -> bool:
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
        if report["passed"]:
            self.store.update(record)
            return True
        record.status = JobStatus.BLOCKED
        record.last_error = "completion_integrity_failed:" + ",".join(report["failure_reasons"])
        self.store.update(record)
        return False

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
        if missing_test_evidence:
            failure_reasons.append("missing_test_evidence")
        stages_missing_test_patches = JobRunner._stages_missing_test_patches(record)
        if require_stage_test_patches and stages_missing_test_patches:
            failure_reasons.append(
                "missing_stage_test_patches:"
                + "|".join(str(stage["stage"]) for stage in stages_missing_test_patches)
            )
        return {
            "passed": (
                (not require_completion_integrity or not missing_task_ids)
                and (not require_test_evidence or not missing_test_evidence)
                and (not require_stage_test_patches or not stages_missing_test_patches)
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
        }

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
    def _constraints(record: JobRecord) -> dict[str, Any]:
        constraints = record.spec.metadata.get("constraints", {})
        return constraints if isinstance(constraints, dict) else {}

    def _constraint_flag(self, record: JobRecord, key: str) -> bool:
        return bool(self._constraints(record).get(key, False))

    def _constraint_int(self, record: JobRecord, key: str, default: int) -> int:
        value = self._constraints(record).get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _autonomous_stage_limit_reached(
        self,
        record: JobRecord,
        stage_results: list[dict[str, Any]],
    ) -> bool:
        max_stages = self._constraint_int(record, "max_autonomous_stages", 0)
        if not max_stages or len(stage_results) < max_stages:
            return False
        record.status = JobStatus.BLOCKED
        record.last_error = "autonomous_stage_limit_reached"
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
                )
                if not self._fixer_allows_progress(record, primary_task, fix):
                    return review, security_review
                ensure_fixer_safe(fix.patches)
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
    memory_db = Path(memory_db_path or (workspace_path / ".acos_memory.sqlite3"))
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
    env = FakeMCPEnvironment(workspace_root=workspace_path, memory_db_path=memory_db)
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=env.build_router(),
        store=store,
    )
    return runner, env
