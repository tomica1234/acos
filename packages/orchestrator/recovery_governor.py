"""Recover failed ACOS jobs without treating recoverable failures as terminal."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from packages.orchestrator.quality_gates import invalid_artifact_paths
from packages.orchestrator.statuses import (
    HARD_TERMINAL_STATUSES,
    RECOVERABLE_STATUSES,
    WAITING_STATUSES,
    is_hard_terminal_status,
    is_recoverable_status,
    is_waiting_status,
)
from packages.orchestrator.task_graph_validation import TASK_GRAPH_VALIDATION_CONTEXT_KEYS
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


POLICY_HARD_STOP_PREFIXES = (
    "policy_hard_stop",
    "policy_denied",
    "blocked_operation",
    "secret_access",
    "direct_main_write",
    "direct_master_write",
    "force_push",
    "production_deploy",
    "unsafe_shell",
    "workspace_escape",
    "sudo",
    "arbitrary_shell",
)


@dataclass
class RecoveryPlan:
    """A concrete autonomous recovery action for a recoverable failure."""

    trigger: str
    strategy: str
    next_status: JobStatus
    next_actor: str
    steps: list[str]
    reason: str
    checkpoint_policy: str = "preserve"
    constraints: dict[str, Any] = field(default_factory=dict)
    hard_stop: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "pending"
    current_step_index: int = 0
    executed_steps: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    failure_reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "trigger": self.trigger,
            "strategy": self.strategy,
            "current_step_index": self.current_step_index,
            "next_status": self.next_status.value,
            "next_actor": self.next_actor,
            "steps": list(self.steps),
            "executed_steps": list(self.executed_steps),
            "reason": self.reason,
            "checkpoint_policy": self.checkpoint_policy,
            "constraints": dict(self.constraints),
            "hard_stop": self.hard_stop,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "failure_reason": self.failure_reason,
        }


class RecoveryGovernor:
    """Convert BLOCKED/STUCK/FAILED into an explicit next recovery strategy."""

    def build_plan(
        self,
        record: JobRecord,
        *,
        error: str | None = None,
        runtime_state: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        last_error = str(error or record.last_error or "")
        lowered = last_error.lower()
        runtime = runtime_state or record.runtime_state

        if self.is_policy_hard_stop(last_error):
            return RecoveryPlan(
                trigger=self._trigger(last_error, "policy_hard_stop"),
                strategy="POLICY_HARD_STOP",
                next_status=JobStatus.POLICY_HARD_STOP,
                next_actor="human",
                steps=["STOP_FOR_POLICY"],
                reason=last_error or "policy hard stop",
                checkpoint_policy="preserve",
                constraints={"recovery_mode": "policy_hard_stop"},
                hard_stop=True,
            )

        if "provider" in lowered and any(token in lowered for token in ("unavailable", "unhealthy", "timeout", "timed out", "down")):
            return RecoveryPlan(
                trigger=self._trigger(last_error, "provider_unavailable"),
                strategy="WAIT_FOR_PROVIDER",
                next_status=JobStatus.WAITING_RUNTIME,
                next_actor="runtime",
                steps=["WAITING_RUNTIME", "AUTO_RESUME_WHEN_PROVIDER_HEALTHY"],
                reason=last_error or "provider unavailable",
                checkpoint_policy="preserve",
                constraints={"recovery_mode": "provider_wait_retry"},
            )

        if "approval rejected" in lowered or "approval_rejected" in lowered:
            if any(token in lowered for token in ("critical", "secret", "deploy", "main", "master", "force")):
                return RecoveryPlan(
                    trigger=self._trigger(last_error, "approval_rejected"),
                    strategy="POLICY_HARD_STOP",
                    next_status=JobStatus.POLICY_HARD_STOP,
                    next_actor="human",
                    steps=["STOP_FOR_REJECTED_CRITICAL_OPERATION"],
                    reason=last_error or "critical approval rejected",
                    checkpoint_policy="preserve",
                    constraints={"recovery_mode": "approval_policy_hard_stop"},
                    hard_stop=True,
                )
            return RecoveryPlan(
                trigger=self._trigger(last_error, "approval_rejected"),
                strategy="REPLAN_TO_AVOID_REJECTED_OPERATION",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["REPLAN_WITH_CONSTRAINTS", "AVOID_REJECTED_OPERATION"],
                reason=last_error or "approval rejected",
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "approval_replan"},
            )

        mapping = self._strategy_mapping(record, last_error, runtime)
        if mapping is not None:
            return mapping

        return RecoveryPlan(
            trigger=self._trigger(last_error, "recoverable_failure"),
            strategy="DIAGNOSE_FAILURE",
            next_status=JobStatus.DIAGNOSING,
            next_actor="diagnoser",
            steps=["DIAGNOSE_FAILURE", "REPLAN_TASK"],
            reason=last_error or "recoverable failure",
            checkpoint_policy="invalidate_failed_stage",
            constraints={"recovery_mode": "diagnose_and_replan"},
        )

    def recover(
        self,
        record: JobRecord,
        *,
        error: str | None = None,
        runtime_state: dict[str, Any] | None = None,
    ) -> RecoveryPlan:
        recoverable_error = str(error or record.last_error or "")
        plan = self.build_plan(record, error=error, runtime_state=runtime_state)
        now = datetime.now(timezone.utc).isoformat()
        plan.created_at = plan.created_at or now
        plan.updated_at = now
        plan_payload = plan.model_dump()
        record.runtime_state["recovery_plan"] = plan_payload
        recovery_event = {
            "id": plan.id,
            "at": now,
            "trigger": plan.trigger,
            "strategy": plan.strategy,
            "next_status": plan.next_status.value,
            "next_actor": plan.next_actor,
            "reason": plan.reason,
            "error": recoverable_error or plan.reason,
            "hard_stop": plan.hard_stop,
        }
        record.runtime_state["current_recovery_event"] = recovery_event
        history = record.outputs.setdefault("recovery_history", [])
        if not isinstance(history, list):
            history = []
            record.outputs["recovery_history"] = history
        history.append(plan_payload)
        events = record.outputs.setdefault("recovery_events", [])
        if not isinstance(events, list):
            events = []
            record.outputs["recovery_events"] = events
        events.append(recovery_event)
        constraints = record.spec.metadata.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            record.spec.metadata["constraints"] = constraints
        constraints.update(plan.constraints)
        constraints["recovery_strategy"] = plan.strategy
        constraints["recovery_next_actor"] = plan.next_actor
        constraints["recovery_step_count"] = len(history)
        self._invalidate_checkpoints(record, plan)
        self._invalidate_outputs(record, plan)
        record.status = plan.next_status
        record.history.append(plan.next_status)
        if plan.hard_stop:
            record.runtime_state.pop("last_recoverable_error", None)
            record.outputs.pop("last_recoverable_error", None)
            record.last_error = plan.reason
        else:
            recoverable_reason = recoverable_error or plan.reason
            record.runtime_state["last_recoverable_error"] = recoverable_reason
            record.outputs["last_recoverable_error"] = recoverable_reason
            record.last_error = None
        record.updated_at = datetime.now(timezone.utc)
        return plan

    def recover_if_needed(self, record: JobRecord) -> RecoveryPlan | None:
        if is_recoverable_status(record.status):
            return self.recover(record)
        return None

    @staticmethod
    def is_policy_hard_stop(last_error: str | None) -> bool:
        if not last_error:
            return False
        lowered = last_error.lower()
        return any(lowered.startswith(prefix) for prefix in POLICY_HARD_STOP_PREFIXES)

    @staticmethod
    def _trigger(last_error: str, fallback: str) -> str:
        if not last_error:
            return fallback
        return last_error.split(":", 1)[0]

    @staticmethod
    def _missing_patch_path(last_error: str, runtime_state: dict[str, Any]) -> str:
        candidate = runtime_state.get("failed_patch_path")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        marker = "update target does not exist:"
        if marker in last_error:
            return last_error.rsplit(marker, 1)[-1].strip()
        candidate = runtime_state.get("missing_target_file")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return ""

    @staticmethod
    def _looks_like_test_path(path: str) -> bool:
        normalized = path.replace("\\", "/")
        name = normalized.rsplit("/", 1)[-1]
        return (
            "/tests/" in f"/{normalized}"
            or "/test/" in f"/{normalized}"
            or name.startswith("test_")
            or ".test." in name
            or ".spec." in name
        )

    def _strategy_mapping(
        self,
        record: JobRecord,
        last_error: str,
        runtime_state: dict[str, Any],
    ) -> RecoveryPlan | None:
        lowered = last_error.lower()
        trigger = self._trigger(last_error, "recoverable_failure")
        max_step_role = self._agent_max_steps_role(last_error)
        if max_step_role is not None:
            return RecoveryPlan(
                trigger="agent_max_steps_exceeded",
                strategy="RETRY_AGENT_WITH_STRUCTURED_OUTPUT_GUARD",
                next_status=JobStatus.STRATEGY_CHANGE,
                next_actor=max_step_role,
                steps=[
                    "SUMMARIZE_TOOL_FINDINGS",
                    "RETRY_WITH_SMALLER_SCOPE",
                    "RETURN_VALID_STRUCTURED_OUTPUT",
                ],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={
                    "recovery_mode": "agent_max_steps_structured_output",
                    "max_steps_exceeded_role": max_step_role,
                    "avoid_tool_loop": True,
                    "force_structured_output": True,
                    "retry_small_scope": True,
                    "expand_context": True,
                },
            )
        if trigger in {"max_attempts_exceeded", "tests_failed_after_retries"}:
            return RecoveryPlan(
                trigger=trigger,
                strategy="REPLAN_TASK",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["DIAGNOSE_FAILURE", "REPLAN_TASK"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "max_attempts_replan"},
            )
        if trigger in {"same_failure_threshold_reached", "diagnosed_repeated_failure"}:
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETRY_WITH_DIFFERENT_STRATEGY",
                next_status=JobStatus.DIAGNOSING,
                next_actor="diagnoser",
                steps=["DIAGNOSE_FAILURE", "EXPAND_CONTEXT", "RETRY_WITH_DIFFERENT_STRATEGY"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={
                    "recovery_mode": "same_failure_strategy_change",
                    "avoid_same_fixer_loop": True,
                    "expand_context": True,
                },
            )
        if trigger == "design_review_max_attempts_exceeded":
            return RecoveryPlan(
                trigger=trigger,
                strategy="REVISE_PRD_AND_ARCHITECTURE",
                next_status=JobStatus.REPLANNING,
                next_actor="pm",
                steps=["REVISE_PRD", "REVISE_ARCHITECTURE", "REPLAN_TASK"],
                reason=last_error,
                checkpoint_policy="invalidate_planning",
                constraints={"recovery_mode": "design_review_revision"},
            )
        if trigger == "acceptance_review_max_attempts_exceeded":
            return RecoveryPlan(
                trigger=trigger,
                strategy="SPLIT_TASK_OR_REDEFINE_ACCEPTANCE",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["SPLIT_TASK", "REDEFINE_ACCEPTANCE", "CREATE_PENDING_FIX_REQUEST"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "acceptance_review_split"},
            )
        if trigger in {"required_artifacts_missing", "completion_integrity_failed"}:
            missing_artifacts = runtime_state.get("missing_artifacts")
            non_file_artifacts = runtime_state.get("non_file_artifacts")
            context_constraints = self._recovery_context_constraints(runtime_state)
            if runtime_state.get("force_project_setup_scaffold") is True and isinstance(
                missing_artifacts,
                list,
            ):
                non_file_artifacts = (
                    [str(item) for item in non_file_artifacts if str(item).strip()]
                    if isinstance(non_file_artifacts, list)
                    else []
                )
                return RecoveryPlan(
                    trigger=trigger,
                    strategy="RETURN_TO_IMPLEMENTER",
                    next_status=JobStatus.IMPLEMENTING,
                    next_actor="scaffold",
                    steps=["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"],
                    reason=last_error,
                    checkpoint_policy="invalidate_failed_stage",
                    constraints={
                        "recovery_mode": "project_setup_required_artifacts",
                        "required_artifacts": [
                            str(item) for item in missing_artifacts if str(item).strip()
                        ],
                        "target_files": [
                            str(item) for item in missing_artifacts if str(item).strip()
                        ],
                        "non_file_artifacts": non_file_artifacts,
                        "force_project_setup_scaffold": True,
                        **context_constraints,
                    },
                )
            required_artifacts = self._clean_string_list(
                runtime_state.get("required_artifacts")
            )
            target_files = self._clean_string_list(runtime_state.get("target_files"))
            missing_artifacts = self._clean_string_list(missing_artifacts)
            invalid_artifacts = self._clean_string_list(
                runtime_state.get("invalid_artifacts")
            )
            return RecoveryPlan(
                trigger=trigger,
                strategy="REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["REPLAN_TASK_WITH_REQUIRED_ARTIFACTS", "RETURN_TO_IMPLEMENTER"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={
                    "recovery_mode": "required_artifacts_replan",
                    **context_constraints,
                    **({"required_artifacts": required_artifacts} if required_artifacts else {}),
                    **({"target_files": target_files} if target_files else {}),
                    **({"missing_artifacts": missing_artifacts} if missing_artifacts else {}),
                    **({"invalid_artifacts": invalid_artifacts} if invalid_artifacts else {}),
                },
            )
        if trigger in {"invalid_task_graph", "unmet_task_dependencies"}:
            context_constraints = self._task_graph_context_constraints(runtime_state)
            return RecoveryPlan(
                trigger=trigger,
                strategy="REPLAN_TASK",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["REPLAN_TASK", "SPLIT_TASK"],
                reason=last_error,
                checkpoint_policy="invalidate_planning",
                constraints={
                    "recovery_mode": "task_graph_repair",
                    **context_constraints,
                },
            )
        if trigger == "prd_quality_gate_failed":
            context_constraints = self._prd_quality_context_constraints(runtime_state)
            return RecoveryPlan(
                trigger=trigger,
                strategy="REVISE_PRD_AND_ARCHITECTURE",
                next_status=JobStatus.ANALYZING,
                next_actor="pm",
                steps=["REVISE_PRD", "REVISE_ARCHITECTURE", "REPLAN_TASK"],
                reason=last_error,
                checkpoint_policy="invalidate_planning",
                constraints={
                    "recovery_mode": "prd_quality_revision",
                    **context_constraints,
                },
            )
        if trigger == "autonomous_stage_limit_reached":
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETRY_WITH_DIFFERENT_STRATEGY",
                next_status=JobStatus.STRATEGY_CHANGE,
                next_actor="planner",
                steps=["EXPAND_CONTEXT", "SPLIT_TASK", "RETRY_WITH_DIFFERENT_STRATEGY"],
                reason=last_error,
                checkpoint_policy="preserve",
                constraints={
                    "recovery_mode": "stage_limit_strategy_change",
                    "auto_bump_stage_limit": True,
                },
            )
        if (
            trigger in {
                "target_files_missing",
                "target_file_missing",
                "target_files_invalid",
                "target_file_invalid",
                "PATCH_OPERATION_MISMATCH",
            }
            or "target file" in lowered
            or "patch_operation_mismatch" in lowered
        ):
            missing_path = self._missing_patch_path(last_error, runtime_state)
            invalid_artifacts = runtime_state.get("invalid_artifacts")
            if not isinstance(invalid_artifacts, list):
                invalid_artifacts = invalid_artifact_paths([missing_path]) if missing_path else []
            if invalid_artifacts:
                return RecoveryPlan(
                    trigger=trigger,
                    strategy="REPLAN_TASK_WITH_REQUIRED_ARTIFACTS",
                    next_status=JobStatus.REPLANNING,
                    next_actor="planner",
                    steps=["REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"],
                    reason=last_error,
                    checkpoint_policy="invalidate_failed_stage",
                    constraints={
                        "recovery_mode": "invalid_artifacts_replan",
                        "invalid_artifacts": [
                            str(item) for item in invalid_artifacts if str(item).strip()
                        ],
                    },
                )
            failed_role = str(runtime_state.get("failed_patch_role") or "")
            if not failed_role:
                failed_role = "test_writer" if self._looks_like_test_path(missing_path) else "implementer"
            if failed_role == "test_writer" or self._looks_like_test_path(missing_path):
                next_status = JobStatus.WRITING_TESTS
                next_actor = "test_writer"
                strategy = "RETURN_TO_TEST_WRITER"
                steps = ["RETURN_TO_TEST_WRITER", "RECREATE_TARGET_FILES"]
            elif failed_role == "fixer":
                next_status = JobStatus.FIXING
                next_actor = "fixer"
                strategy = "RETURN_TO_FIXER"
                steps = ["RETURN_TO_FIXER", "RECREATE_TARGET_FILES"]
            else:
                next_status = JobStatus.IMPLEMENTING
                next_actor = "implementer"
                strategy = "RETURN_TO_IMPLEMENTER"
                steps = ["RETURN_TO_IMPLEMENTER", "RECREATE_TARGET_FILES"]
            constraints: dict[str, Any] = {
                "recovery_mode": "target_files_missing",
                "return_to_role": next_actor,
            }
            if missing_path:
                constraints["missing_target_file"] = missing_path
                constraints["required_artifacts"] = [missing_path]
                constraints["target_files"] = [missing_path]
            if (
                runtime_state.get("failed_patch_operation") == "update"
                or trigger == "PATCH_OPERATION_MISMATCH"
            ):
                constraints["patch_operation_hint"] = "create"
            return RecoveryPlan(
                trigger=trigger,
                strategy=strategy,
                next_status=next_status,
                next_actor=next_actor,
                steps=steps,
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints=constraints,
            )
        if trigger in {"test_patch_quality_failed", "fixer_attempted_to_weaken_tests"} or "weaken tests" in lowered:
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETURN_TO_TEST_WRITER",
                next_status=JobStatus.WRITING_TESTS,
                next_actor="test_writer",
                steps=["RETURN_TO_TEST_WRITER", "REWRITE_TEST_PATCH"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "test_patch_quality_rewrite"},
            )
        if "reviewer did not approve" in lowered or "security review did not approve" in lowered:
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETURN_TO_FIXER",
                next_status=JobStatus.FIXING,
                next_actor="fixer",
                steps=["RETURN_TO_FIXER", "ADDRESS_REVIEW_FINDINGS"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "review_repair"},
            )
        if trigger == "output_truncated":
            return RecoveryPlan(
                trigger=trigger,
                strategy="COMPACT_CONTEXT_AND_RETRY",
                next_status=JobStatus.RECOVERING,
                next_actor="orchestrator",
                steps=["COMPACT_CONTEXT", "RETRY_AGENT"],
                reason=last_error,
                checkpoint_policy="preserve",
                constraints={"recovery_mode": "compact_context_retry"},
            )
        if trigger == "context_budget_exceeded":
            return RecoveryPlan(
                trigger=trigger,
                strategy="EXPAND_COMPACT_RETRIEVAL_AND_RETRY",
                next_status=JobStatus.RECOVERING,
                next_actor="orchestrator",
                steps=["COMPACT_RETRIEVAL", "EXPAND_RELEVANT_FILES", "RETRY_AGENT"],
                reason=last_error,
                checkpoint_policy="preserve",
                constraints={"recovery_mode": "retrieval_retry", "expand_context": True},
            )
        if trigger == "implementation_failed":
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETURN_TO_IMPLEMENTER",
                next_status=JobStatus.IMPLEMENTING,
                next_actor="implementer",
                steps=["RETURN_TO_IMPLEMENTER", "RETRY_IMPLEMENTATION_WITH_RECOVERY_CONTEXT"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": "implementation_retry"},
            )
        if trigger in {"implementation_blocked", "fixer_failed", "fixer_stuck"}:
            return RecoveryPlan(
                trigger=trigger,
                strategy="REPLAN_TASK",
                next_status=JobStatus.REPLANNING,
                next_actor="planner",
                steps=["DIAGNOSE_FAILURE", "REPLAN_TASK"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": f"{trigger}_replan"},
            )
        if trigger in {"test_writer_blocked", "test_writer_failed"}:
            return RecoveryPlan(
                trigger=trigger,
                strategy="RETURN_TO_TEST_WRITER",
                next_status=JobStatus.WRITING_TESTS,
                next_actor="test_writer",
                steps=["REWRITE_TEST_STRATEGY", "RETRY_TEST_WRITER"],
                reason=last_error,
                checkpoint_policy="invalidate_failed_stage",
                constraints={"recovery_mode": f"{trigger}_rewrite"},
            )
        if runtime_state.get("provider_unavailable") is True:
            return self.build_plan(record, error="provider_unavailable", runtime_state={})
        return None

    @staticmethod
    def _clean_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @classmethod
    def _recovery_context_constraints(cls, runtime_state: dict[str, Any]) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        for key in (
            "failed_stage",
            "failed_task_id",
            "stage_failure_reason",
        ):
            value = runtime_state.get(key)
            if isinstance(value, str) and value.strip():
                constraints[key] = value.strip()
            elif isinstance(value, (int, float, bool)):
                constraints[key] = value
        for key in ("missing_artifacts", "invalid_artifacts", "non_file_artifacts"):
            value = cls._clean_string_list(runtime_state.get(key))
            if value:
                constraints[key] = value
        return constraints

    @classmethod
    def _prd_quality_context_constraints(cls, runtime_state: dict[str, Any]) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        for key in (
            "prd_quality_missing",
            "prd_quality_warnings",
            "prd_open_questions",
            "invalid_required_artifacts",
            "prd_required_artifacts",
            "source_required_artifacts",
            "test_required_artifacts",
        ):
            value = cls._clean_string_list(runtime_state.get(key))
            if value:
                constraints[key] = value
        uncovered = runtime_state.get("uncovered_acceptance_small_parts")
        if isinstance(uncovered, list) and uncovered:
            constraints["uncovered_acceptance_small_parts"] = uncovered
        return constraints

    @staticmethod
    def _non_empty_list(value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if item]

    @classmethod
    def _task_graph_context_constraints(cls, runtime_state: dict[str, Any]) -> dict[str, Any]:
        constraints: dict[str, Any] = {}
        errors = cls._clean_string_list(
            runtime_state.get("task_graph_validation_errors")
        )
        if errors:
            constraints["task_graph_validation_errors"] = errors
        for key in TASK_GRAPH_VALIDATION_CONTEXT_KEYS:
            value = cls._non_empty_list(runtime_state.get(key))
            if value:
                constraints[key] = value
        return constraints

    @staticmethod
    def _agent_max_steps_role(last_error: str) -> str | None:
        match = re.search(
            r"Agent\s+([A-Za-z0-9_-]+)\s+exceeded\s+max_steps=",
            last_error,
            flags=re.IGNORECASE,
        )
        if match is None or "without a valid structured response" not in last_error:
            return None
        return match.group(1)

    @staticmethod
    def _invalidate_checkpoints(record: JobRecord, plan: RecoveryPlan) -> None:
        if plan.checkpoint_policy == "preserve":
            return
        if plan.checkpoint_policy == "invalidate_planning":
            record.runtime_state["planning_invalidated"] = True
            return
        for checkpoint in record.checkpoints:
            if not isinstance(checkpoint, dict):
                continue
            if checkpoint.get("test_success") is False or checkpoint.get("stage") == record.outputs.get("failed_stage"):
                checkpoint["invalidated_by_recovery"] = True
                checkpoint["recovery_strategy"] = plan.strategy

    @staticmethod
    def _invalidate_outputs(record: JobRecord, plan: RecoveryPlan) -> None:
        keys_by_strategy = {
            "REVISE_PRD_AND_ARCHITECTURE": {
                "pm",
                "prd",
                "architect",
                "architecture",
                "planner",
                "task_graph",
            },
            "REPLAN_TASK": {
                "planner",
                "task_graph",
            },
            "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS": {
                "planner",
                "task_graph",
            },
            "SPLIT_TASK_OR_REDEFINE_ACCEPTANCE": {
                "planner",
                "task_graph",
            },
            "RETURN_TO_IMPLEMENTER": {
                "implementer",
                "implementation",
            },
            "RETURN_TO_TEST_WRITER": {
                "test_writer",
            },
        }
        invalidated = sorted(keys_by_strategy.get(plan.strategy, set()))
        for key in invalidated:
            record.outputs.pop(key, None)
        if invalidated:
            record.runtime_state["invalidated_outputs"] = invalidated


__all__ = [
    "HARD_TERMINAL_STATUSES",
    "WAITING_STATUSES",
    "RECOVERABLE_STATUSES",
    "RecoveryGovernor",
    "RecoveryPlan",
    "is_hard_terminal_status",
    "is_waiting_status",
    "is_recoverable_status",
]
