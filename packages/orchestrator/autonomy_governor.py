"""Autonomous recovery decisions for ACOS supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from packages.schemas.jobs import JobRecord


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
)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    strategy: str
    reason: str
    can_apply_automatically: bool
    constraints: dict[str, Any]
    summary: str
    next_actor: str | None = None

    def as_plan(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "strategy": self.strategy,
            "reason": self.reason,
            "can_apply_automatically": self.can_apply_automatically,
            "constraints": self.constraints,
            "summary": self.summary,
            "next_actor": self.next_actor,
        }


class AutonomyGovernor:
    """Choose recovery strategy without returning to a human by default."""

    def decide(self, record: JobRecord, summary: dict[str, Any]) -> RecoveryDecision:
        last_error = str(record.last_error or summary.get("last_error") or "")
        if self.is_policy_hard_stop(last_error):
            return RecoveryDecision(
                action="inspect",
                strategy="policy_hard_stop",
                reason=last_error or "policy_hard_stop",
                can_apply_automatically=False,
                constraints={"recovery_mode": "policy_hard_stop"},
                summary="Policy hard stop requires human inspection.",
            )

        failure = summary.get("failure_analysis")
        if isinstance(failure, dict):
            recovery = failure.get("recommended_recovery")
            if isinstance(recovery, dict):
                constraints = dict(recovery.get("constraints") or {})
                strategy = str(recovery.get("strategy") or constraints.get("recovery_strategy"))
                return RecoveryDecision(
                    action="continue",
                    strategy=strategy,
                    reason=str(recovery.get("reason") or last_error or "recoverable_failure"),
                    can_apply_automatically=True,
                    constraints=constraints,
                    summary="Recoverable failure; ACOS will change strategy and continue.",
                    next_actor=self._next_actor_for_strategy(strategy),
                )

        resume = summary.get("resume")
        if isinstance(resume, dict):
            action = str(resume.get("action") or "")
            if action == "raise_stage_limit_or_resume":
                next_limit = resume.get("suggested_max_autonomous_stages")
                constraints: dict[str, Any] = {
                    "recovery_mode": "stage_limit",
                    "recovery_strategy": "raise_stage_limit",
                    "pm_strategy_change": True,
                    "pm_strategy": "raise_stage_limit",
                }
                if isinstance(next_limit, int):
                    constraints["max_autonomous_stages"] = next_limit
                return RecoveryDecision(
                    action="continue",
                    strategy="raise_stage_limit",
                    reason=str(resume.get("reason") or "autonomous_stage_limit_reached"),
                    can_apply_automatically=True,
                    constraints=constraints,
                    summary="Stage limit reached; ACOS will bump the limit and continue.",
                    next_actor="orchestrator",
                )
            if action == "improve_planning_quality":
                return RecoveryDecision(
                    action="continue",
                    strategy="planning_repair_strategy_change",
                    reason=str(resume.get("reason") or "planning_quality_repair"),
                    can_apply_automatically=True,
                    constraints={
                        "recovery_mode": "planning_repair",
                        "recovery_strategy": "planning_repair_strategy_change",
                        "pm_strategy_change": True,
                        "pm_strategy": "planning_repair_strategy_change",
                        "require_prd_quality": True,
                        "require_task_acceptance_criteria": True,
                    },
                    summary="Planning quality is recoverable; ACOS will revise assumptions and continue.",
                    next_actor="pm",
                )

        return RecoveryDecision(
            action="continue",
            strategy="continue_next_action",
            reason=last_error or "continue",
            can_apply_automatically=True,
            constraints={"recovery_mode": "autonomous_continue"},
            summary="No policy hard stop detected; ACOS may continue autonomously.",
            next_actor=None,
        )

    @staticmethod
    def is_policy_hard_stop(last_error: str | None) -> bool:
        if not last_error:
            return False
        lowered = last_error.lower()
        return any(lowered.startswith(prefix) for prefix in POLICY_HARD_STOP_PREFIXES)

    @staticmethod
    def _next_actor_for_strategy(strategy: str) -> str | None:
        if strategy in {"completion_audit", "planning_repair_strategy_change"}:
            return "planner"
        if strategy in {"rewrite_tests", "split_or_clarify_tests"}:
            return "test_writer"
        if strategy in {"diagnosis_guided_retry", "escalated_retry"}:
            return "fixer"
        if strategy in {"replan_current_task", "split_or_clarify_task"}:
            return "planner"
        return None


def apply_recovery_plan(record: JobRecord, decision: RecoveryDecision) -> dict[str, Any]:
    """Persist a governor decision into job outputs and constraints."""
    plan = decision.as_plan()
    record.outputs["autonomous_recovery_plan"] = plan
    constraints = record.spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        record.spec.metadata["constraints"] = constraints
    for key, value in decision.constraints.items():
        if value is not None:
            constraints[key] = value
    interventions = record.outputs.setdefault("pm_interventions", [])
    if not isinstance(interventions, list):
        interventions = []
        record.outputs["pm_interventions"] = interventions
    if decision.strategy != "policy_hard_stop":
        intervention = {
            "action": "change_strategy",
            "reason": decision.reason,
            "strategy": decision.strategy,
            "summary": decision.summary,
            "can_apply_automatically": decision.can_apply_automatically,
            "applied": decision.can_apply_automatically,
            "next_actor": decision.next_actor,
            "intervention_index": len(interventions) + 1,
        }
        interventions.append(intervention)
        constraints["pm_intervention_count"] = intervention["intervention_index"]
    return plan
