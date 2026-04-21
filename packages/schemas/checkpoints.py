"""Checkpoint schemas for durable job resumption."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CheckpointRecord(BaseModel):
    """Persisted checkpoint for idempotent step execution."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    job_id: str
    task_id: str | None = None
    checkpoint_key: str
    step_name: str
    idempotency_key: str
    status: str
    result_json: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
