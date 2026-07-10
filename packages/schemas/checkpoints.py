"""Checkpoint schemas for durable recovery execution."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.jobs import utc_now


class CheckpointRecord(BaseModel):
    """Persisted checkpoint for an idempotent recovery or job step."""

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
