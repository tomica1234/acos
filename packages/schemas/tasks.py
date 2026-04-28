"""Task planning schemas."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.models import TaskComplexity, TaskStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PlannedTask(BaseModel):
    """A single planned unit of work."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    role: str
    status: TaskStatus = TaskStatus.TODO
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    depends_on: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 3
    approval_id: str | None = None
    pending_runtime_issue_id: str | None = None
    last_error: str | None = None
    checkpoint_key: str | None = None


class TaskRecord(BaseModel):
    """Persisted task state for durable execution."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    job_id: str
    status: TaskStatus = TaskStatus.QUEUED
    title: str
    description: str
    role: str
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    dependencies: list[str] = Field(default_factory=list)
    target_files: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 3
    pending_approval_id: str | None = None
    pending_runtime_issue_id: str | None = None
    last_error: str | None = None
    checkpoint_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_planned_task(cls, *, job_id: str, task: PlannedTask) -> "TaskRecord":
        return cls(
            task_id=task.id,
            job_id=job_id,
            status=task.status if task.status != TaskStatus.TODO else TaskStatus.QUEUED,
            title=task.title,
            description=task.description,
            role=task.role,
            complexity=task.complexity,
            dependencies=list(task.dependencies or task.depends_on),
            target_files=list(task.target_files),
            required_artifacts=list(task.required_artifacts),
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            pending_approval_id=task.approval_id,
            pending_runtime_issue_id=task.pending_runtime_issue_id,
            last_error=task.last_error,
            checkpoint_key=task.checkpoint_key,
        )


class TaskGraph(BaseModel):
    """A planned graph of ACOS tasks."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
