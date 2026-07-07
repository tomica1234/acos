"""Audit event schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    input_hash: str | None = None
    output_hash: str | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def mirror_legacy_fields_into_metadata(self) -> "AuditEvent":
        if self.tool_name is not None:
            self.metadata.setdefault("tool_name", self.tool_name)
        return self

