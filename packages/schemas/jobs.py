"""Job submission and persistence schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from packages.schemas.audit import AuditEvent
from packages.schemas.models import JobStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


class JobRecord(BaseModel):
    """The mutable record for a job."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str | None = None
    spec: JobSpec
    status: JobStatus = JobStatus.SUBMITTED
    history: list[JobStatus] = Field(default_factory=lambda: [JobStatus.SUBMITTED])
    outputs: dict[str, Any] = Field(default_factory=dict)
    audit_events: list[AuditEvent] = Field(default_factory=list)
    failure_count: int = 0
    same_test_failure_count: int = 0
    last_error: str | None = None
    runtime_error: str | None = None
    provider_status: str | None = None
    current_phase: str | None = None
    current_role: str | None = None
    current_task_id: str | None = None
    pending_approval_id: str | None = None
    pending_runtime_issue_id: str | None = None
    runtime_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None

    @computed_field
    @property
    def id(self) -> str:
        return self.job_id
