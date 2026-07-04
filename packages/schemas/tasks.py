"""Task planning schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    acceptance_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if not normalized.get("title"):
            for key in ("name", "summary", "description", "instruction", "id"):
                candidate = normalized.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    normalized["title"] = candidate.strip().splitlines()[0][:120]
                    break
        if not normalized.get("description"):
            for key in ("instruction", "details", "summary", "title"):
                candidate = normalized.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    normalized["description"] = candidate.strip()
                    break
        if "depends_on" not in normalized and "dependencies" in normalized:
            normalized["depends_on"] = normalized.get("dependencies")
        if "acceptance_criteria" not in normalized:
            for key in ("acceptance_tests", "acceptance", "done_when"):
                if key in normalized:
                    normalized["acceptance_criteria"] = normalized.get(key)
                    break
        for alias in (
            "name",
            "summary",
            "instruction",
            "details",
            "dependencies",
            "acceptance_tests",
            "acceptance",
            "done_when",
        ):
            normalized.pop(alias, None)
        return normalized


class TaskGraph(BaseModel):
    """A planned graph of ACOS tasks."""

    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
