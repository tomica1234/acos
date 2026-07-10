"""Audit recording helpers."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

from packages.memory.redaction import redact_text
from packages.schemas.audit import AuditEvent
from packages.schemas.models import ModelCallRecord, ModelSelection


def _hash_payload(payload: Any) -> str:
    redacted = redact_text(json.dumps(payload, sort_keys=True, default=str))
    return sha256(redacted.encode("utf-8")).hexdigest()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


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
            metadata=_redact_value(
                {
                    "model_key": selection.model_key,
                    "provider_key": selection.provider_key,
                    "routing_reason": selection.reason.value,
                    "routing_details": selection.details,
                    "provider": record.provider,
                    "error": record.error,
                    "started_at": record.started_at.isoformat() if record.started_at else None,
                    "finished_at": (
                        record.finished_at.isoformat() if record.finished_at else None
                    ),
                    "duration_seconds": record.duration_seconds,
                    "usage_source": record.usage_source,
                    "prompt_tokens": record.prompt_tokens_estimate,
                    "completion_tokens": record.completion_tokens_estimate,
                    "total_tokens": record.total_tokens_estimate,
                    "completion_tokens_per_second": record.completion_tokens_per_second,
                    "total_tokens_per_second": record.total_tokens_per_second,
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
            metadata=_redact_value(
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
