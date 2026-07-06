"""Job submission and persistence schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from packages.schemas.audit import AuditEvent
from packages.schemas.models import JobStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


_ALLOWED_JOB_ID_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def validate_job_id_string(value: str) -> str:
    """Validate a job id before it is used as a persisted file name."""

    if not value:
        raise ValueError("job_id must not be empty")
    if len(value) > 128:
        raise ValueError("job_id must be 128 characters or fewer")
    if any(part in value for part in ("/", "\\", ":", "\x00")):
        raise ValueError("job_id must not contain path separators or ':'")
    if value in {".", ".."}:
        raise ValueError("job_id must not be a path segment")
    if any(character not in _ALLOWED_JOB_ID_CHARACTERS for character in value):
        raise ValueError("job_id may only contain letters, numbers, '.', '_', and '-'")
    return value


class JobSpec(BaseModel):
    """A user-submitted ACOS job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: uuid4().hex)
    request_text: str
    repo_path: str
    target_branch: str = "acos/default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_root: str | None = None
    title: str | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return validate_job_id_string(value)


class JobRecord(BaseModel):
    """The mutable record for a job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    spec: JobSpec
    status: JobStatus = JobStatus.SUBMITTED
    history: list[JobStatus] = Field(default_factory=lambda: [JobStatus.SUBMITTED])
    outputs: dict[str, Any] = Field(default_factory=dict)
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    failure_count: int = 0
    same_test_failure_count: int = 0
    last_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
