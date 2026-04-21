"""Audit event schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditEvent(BaseModel):
    """A sanitized audit trail record."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utc_now)
    event_type: str
    role: str
    action: str
    status: str
    job_id: str | None = None
    task_id: str | None = None
    step: str | None = None
    provider_key: str | None = None
    model_key: str | None = None
    tool_name: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
