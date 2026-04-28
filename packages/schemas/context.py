"""Context packets passed into ACOS agents."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.tasks import PlannedTask


class ContextPacket(BaseModel):
    """Compact, role-scoped context for a single agent invocation."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    role: str
    objective: str
    repo_path: str
    request_text: str
    constraints: list[str] = Field(default_factory=list)
    relevant_files: dict[str, str] = Field(default_factory=dict)
    diff: str = ""
    memory_summaries: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    task: PlannedTask | None = None
    token_budget: int = 0
    model_context_budget: int = 0
    selected_model_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _INTERNAL_METADATA_KEYS = {
        "context_truncated",
        "context_truncation_notes",
        "estimated_input_tokens",
        "context_budget_tokens",
        "effective_context_budget_tokens",
        "safety_margin_tokens",
    }

    def render_text(self) -> str:
        """Render the packet into a readable prompt body."""
        lines = [
            f"job_id: {self.job_id}",
            f"role: {self.role}",
            f"objective: {self.objective}",
            f"repo_path: {self.repo_path}",
            f"model_context_budget: {self.model_context_budget}",
            f"selected_model_hint: {self.selected_model_hint or 'unset'}",
            "request:",
            self.request_text,
        ]
        visible_metadata = {
            key: value
            for key, value in self.metadata.items()
            if key not in self._INTERNAL_METADATA_KEYS
        }
        if visible_metadata:
            lines.append("metadata:")
            for key, value in visible_metadata.items():
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
                lines.append(f"- {key}: {rendered}")
        if self.constraints:
            lines.extend(["constraints:"] + [f"- {item}" for item in self.constraints])
        if self.memory_summaries:
            lines.extend(["memory:"] + [f"- {item}" for item in self.memory_summaries])
        if self.task is not None:
            lines.extend(
                [
                    "task:",
                    f"- id: {self.task.id}",
                    f"- title: {self.task.title}",
                    f"- description: {self.task.description}",
                    f"- complexity: {self.task.complexity.value}",
                ]
            )
        if self.relevant_files:
            lines.append("files:")
            for path, content in self.relevant_files.items():
                lines.append(f"--- {path}")
                lines.append(content)
        if self.diff:
            lines.extend(["diff:", self.diff])
        if self.logs:
            lines.extend(["logs:"] + self.logs)
        return "\n".join(lines)
