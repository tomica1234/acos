"""Audit recording helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from packages.memory.redaction import redact_text, redact_value
from packages.schemas.approvals import ApprovalRequest, RiskDecision
from packages.schemas.audit import AuditEvent
from packages.schemas.models import ModelCallRecord, ModelSelection


def _hash_payload(payload: Any) -> str:
    redacted = redact_text(json.dumps(payload, sort_keys=True, default=str))
    return sha256(redacted.encode("utf-8")).hexdigest()

class AuditRecorder:
    """Create sanitized audit events."""

    def model_event(self, record: ModelCallRecord, selection: ModelSelection) -> AuditEvent:
        return AuditEvent(
            event_type="model_call",
            role=record.role,
            action=selection.model_key,
            status=record.status.value,
            input_hash=record.input_hash,
            output_hash=record.output_hash,
            metadata=redact_value(
                {
                    "model_key": selection.model_key,
                    "provider_key": selection.provider_key,
                    "routing_reason": selection.reason.value,
                    "routing_details": selection.details,
                    "provider": record.provider,
                    "error": record.error,
                    "finish_reason": record.finish_reason,
                    "configured_max_output_tokens": record.configured_max_output_tokens,
                    "estimated_input_tokens": record.estimated_input_tokens,
                    "resolved_max_output_tokens": record.resolved_max_output_tokens,
                    "model_max_context_tokens": record.model_max_context_tokens,
                    "safety_margin_tokens": record.safety_margin_tokens,
                    "context_budget_tokens": record.context_budget_tokens,
                    "output_truncated": record.output_truncated,
                    "prompt_tokens_estimate": record.prompt_tokens_estimate,
                    "completion_tokens_estimate": record.completion_tokens_estimate,
                    "total_tokens_estimate": record.total_tokens_estimate,
                }
            ),
        )

    def selection_event(self, role: str, selection: ModelSelection) -> AuditEvent:
        return AuditEvent(
            event_type="model_selection",
            role=role,
            action=selection.model_key,
            status="selected",
            metadata=redact_value(
                {
                    "model_key": selection.model_key,
                    "provider_key": selection.provider_key,
                    "routing_reason": selection.reason.value,
                    "routing_details": selection.details,
                }
            ),
        )

    def tool_event(
        self,
        role: str,
        tool_name: str,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None,
        status: str,
    ) -> AuditEvent:
        return AuditEvent(
            event_type="tool_call",
            role=role,
            action=tool_name,
            status=status,
            input_hash=_hash_payload(input_payload),
            output_hash=_hash_payload(output_payload or {}),
            metadata={"tool_name": tool_name},
        )

    def policy_event(
        self,
        *,
        role: str,
        job_id: str,
        task_id: str | None,
        decision: RiskDecision,
        approval_id: str | None = None,
    ) -> AuditEvent:
        return AuditEvent(
            event_type="policy_decision",
            role=role,
            action=decision.operation,
            status=decision.policy_action.value,
            metadata=redact_value(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "operation": decision.operation,
                    "risk_level": decision.risk_level.value,
                    "policy_action": decision.policy_action.value,
                    "reason": decision.reason,
                    "details": decision.details,
                    "approval_id": approval_id,
                }
            ),
        )

    def approval_event(
        self,
        *,
        role: str,
        action: str,
        approval: ApprovalRequest,
    ) -> AuditEvent:
        return AuditEvent(
            event_type="approval",
            role=role,
            action=action,
            status=approval.status,
            metadata=redact_value(
                {
                    "approval_id": approval.id,
                    "job_id": approval.job_id,
                    "task_id": approval.task_id,
                    "operation": approval.operation,
                    "risk_level": approval.risk_level.value,
                    "approver": approval.approver,
                    "resolution_reason": approval.resolution_reason,
                }
            ),
        )
