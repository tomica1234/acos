"""Execute durable ACOS recovery plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from packages.orchestrator.statuses import is_hard_terminal_status, is_waiting_status
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus


class RecoveryExecutor:
    """Consume RecoveryGovernor plans and make them actionable."""

    def __init__(self, store: Any | None = None) -> None:
        self.store = store

    def execute_until_ready(self, record: JobRecord) -> JobRecord:
        """Run all bookkeeping recovery steps until normal job flow can resume."""

        plan = record.runtime_state.get("recovery_plan")
        if not isinstance(plan, dict):
            return record
        if plan.get("status") == "completed":
            return record
        if is_hard_terminal_status(record.status):
            return record

        plan["status"] = "running"
        self._touch_plan(plan)
        steps = [str(step) for step in plan.get("steps", [])]
        current_index = int(plan.get("current_step_index") or 0)
        while current_index < len(steps):
            step = steps[current_index]
            self._checkpoint(record, plan, step)
            self._apply_step(record, plan, step)
            executed = plan.setdefault("executed_steps", [])
            if isinstance(executed, list):
                executed.append(step)
            current_index += 1
            plan["current_step_index"] = current_index
            self._touch_plan(plan)
            if is_hard_terminal_status(record.status) or is_waiting_status(record.status):
                break

        if current_index >= len(steps):
            plan["status"] = "completed"
            plan["completed_at"] = self._now()
            next_status = self._plan_next_status(plan)
            if next_status is not None and not is_hard_terminal_status(record.status):
                record.status = next_status
                if not record.history or record.history[-1] != next_status:
                    record.history.append(next_status)
        record.runtime_state["recovery_plan"] = plan
        record.updated_at = datetime.now(timezone.utc)
        self._persist(record)
        return record

    def _apply_step(self, record: JobRecord, plan: dict[str, Any], step: str) -> None:
        constraints = record.spec.metadata.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            record.spec.metadata["constraints"] = constraints
        runtime = record.runtime_state

        if step == "DIAGNOSE_FAILURE":
            runtime["diagnosis_requested"] = True
            record.status = JobStatus.DIAGNOSING
        elif step == "EXPAND_CONTEXT":
            constraints["expand_context"] = True
            runtime["expand_context"] = True
            runtime["context_expansion_count"] = int(runtime.get("context_expansion_count", 0)) + 1
        elif step == "COMPACT_CONTEXT":
            constraints["compact_context"] = True
            runtime["compact_context"] = True
        elif step in {"REPLAN_TASK", "REPLAN_TASK_WITH_REQUIRED_ARTIFACTS"}:
            runtime["planner_repair_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step == "SPLIT_TASK":
            constraints["split_task_on_retry"] = True
            record.status = JobStatus.REPLANNING
        elif step == "REVISE_PRD":
            runtime["prd_revision_requested"] = True
            record.status = JobStatus.ANALYZING
        elif step == "REVISE_ARCHITECTURE":
            runtime["architecture_revision_requested"] = True
            record.status = JobStatus.DESIGNING
        elif step == "REDEFINE_ACCEPTANCE":
            runtime["acceptance_revision_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step == "RETURN_TO_IMPLEMENTER":
            record.status = JobStatus.IMPLEMENTING
        elif step == "RETURN_TO_TEST_WRITER":
            record.status = JobStatus.WRITING_TESTS
        elif step == "RETURN_TO_FIXER":
            record.status = JobStatus.FIXING
        elif step == "ROLLBACK_LAST_PATCH":
            runtime["rollback_last_patch_requested"] = True
        elif step == "RETRY_WITH_DIFFERENT_STRATEGY":
            constraints["avoid_same_fixer_loop"] = True
            constraints["retry_with_different_strategy"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step == "RETRY_WITH_ESCALATED_MODEL":
            constraints["force_model_escalation"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step == "WAITING_RUNTIME":
            record.status = JobStatus.WAITING_RUNTIME
        elif step == "AVOID_REJECTED_OPERATION":
            constraints["avoid_rejected_operation"] = True
            record.status = JobStatus.REPLANNING
        elif step == "COMPLETION_AUDIT":
            runtime["completion_audit_requested"] = True
            record.status = JobStatus.REPLANNING
        elif step in {"SUMMARIZE_TOOL_FINDINGS", "RETRY_WITH_SMALLER_SCOPE", "RETURN_VALID_STRUCTURED_OUTPUT"}:
            constraints["force_structured_output"] = True
            constraints["retry_small_scope"] = True
            record.status = JobStatus.STRATEGY_CHANGE
        elif step.startswith("STOP_FOR_"):
            record.status = JobStatus.POLICY_HARD_STOP

        if record.history[-1:] != [record.status]:
            record.history.append(record.status)

    def _checkpoint(self, record: JobRecord, plan: dict[str, Any], step: str) -> None:
        payload = {
            "job_id": record.job_id,
            "plan_id": str(plan.get("id", "")),
            "strategy": str(plan.get("strategy", "")),
            "step": step,
            "created_at": self._now(),
        }
        record.checkpoints.append(
            {
                "checkpoint_key": f"recovery:{plan.get('id')}:{step}",
                "step_name": step,
                "status": "completed",
                "result_json": payload,
            }
        )
        if self.store is not None and hasattr(self.store, "save_checkpoint"):
            checkpoint = CheckpointRecord(
                job_id=record.job_id,
                checkpoint_key=f"recovery:{plan.get('id')}:{step}",
                step_name=step,
                idempotency_key=f"{record.job_id}:{plan.get('id')}:{step}",
                status="completed",
                result_json=payload,
            )
            self.store.save_checkpoint(checkpoint)

    def _persist(self, record: JobRecord) -> None:
        if self.store is not None and hasattr(self.store, "update"):
            self.store.update(record)

    @staticmethod
    def _plan_next_status(plan: dict[str, Any]) -> JobStatus | None:
        value = plan.get("next_status")
        if not isinstance(value, str):
            return None
        try:
            return JobStatus(value)
        except ValueError:
            return None

    @staticmethod
    def _touch_plan(plan: dict[str, Any]) -> None:
        plan["updated_at"] = RecoveryExecutor._now()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
