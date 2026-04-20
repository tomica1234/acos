"""ACOS job orchestration engine."""

from __future__ import annotations

import sys
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
from packages.orchestrator.quality_gates import (
    QualityGateError,
    ensure_fixer_safe,
    ensure_reviews_pass,
)
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
from packages.schemas.models import FixStatus, JobStatus
from packages.schemas.tasks import PlannedTask, TaskGraph


class JobRunner:
    """Run ACOS jobs across explicit role phases."""

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
        self._active_record = record
        try:
            self._prepare_branch(record)
            prd = self._run_structured_role(record, "pm", PRD, "Produce the product requirements")
            self._write_memory_item(record, "pm", "prd", prd.model_dump_json())
            architecture = self._run_structured_role(
                record,
                "architect",
                ArchitecturePlan,
                "Design the system architecture",
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
            )
            self._apply_patches(record, "implementer", implementation.patches)
            test_writer = self._run_structured_role(
                record,
                "test_writer",
                TestWriterResult,
                "Add tests for the implementation",
                task=primary_task,
            )
            self._apply_patches(record, "test_writer", test_writer.patches)
            review, security_review = self._run_review_cycle(record, primary_task)
            test_result = self._run_tests(record)
            while not test_result.success and record.failure_count < self.max_attempts_per_task:
                fix = self._run_structured_role(
                    record,
                    "fixer",
                    FixResult,
                    "Fix the deterministic test failures",
                    task=primary_task,
                    logs=[test_result.output_excerpt],
                )
                ensure_fixer_safe(fix.patches)
                self._apply_patches(record, "fixer", fix.patches)
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
            record.outputs["test_run"] = test_result.model_dump()
            record.outputs["summary"] = summary.model_dump()
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
        packet = self.context_builder.build(
            job_id=record.job_id,
            role=role,
            objective=objective,
            repo_path=record.spec.repo_path,
            request_text=record.spec.request_text,
            constraints=self.policy.config.blocked_operations,
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
        output, selection, model_record = self.agent_runner.run(
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

    def _apply_patches(self, record: JobRecord, role: str, patches: list[Any]) -> None:
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
        apply_transition(record, JobStatus.TESTING)
        payload = self._call_tool(
            "runner",
            "test_server.run_test",
            command_name="pytest",
            timeout_seconds=120,
        )
        result = TestRunResult.model_validate(payload)
        record.outputs["test_run"] = result.model_dump()
        return result

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
                ensure_fixer_safe(fix.patches)
                self._apply_patches(record, "fixer", fix.patches)

    @staticmethod
    def _choose_primary_task(task_graph: TaskGraph) -> PlannedTask | None:
        return task_graph.tasks[0] if task_graph.tasks else None


def build_default_runner(
    config_dir: str | Path = "configs",
    workspace_root: str | Path = ".",
    memory_db_path: str | Path | None = None,
) -> tuple[JobRunner, FakeMCPEnvironment]:
    """Build a JobRunner wired to the local config directory and fake MCP tools."""
    config_path = Path(config_dir)
    memory_db = Path(memory_db_path or (Path(workspace_root) / ".acos_memory.sqlite3"))
    registry = ModelRegistry.from_paths(
        provider_path=config_path / "model_providers.yaml",
        agents_path=config_path / "agents.yaml",
        routing_path=config_path / "model_routing.yaml",
    )
    policy = PolicyEngine.from_path(config_path / "policies.yaml")
    registry.validate_or_raise(policy=policy)
    env = FakeMCPEnvironment(workspace_root=workspace_root, memory_db_path=memory_db)
    runner = JobRunner(
        registry=registry,
        policy=policy,
        router=env.build_router(),
    )
    return runner, env
