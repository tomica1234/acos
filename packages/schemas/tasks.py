"""Task planning schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.models import TaskComplexity, TaskStatus


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
    approval_id: str | None = None


class TaskGraph(BaseModel):
    """A planned graph of ACOS tasks."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
