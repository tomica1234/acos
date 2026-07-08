"""Context packets passed into ACOS agents."""

from __future__ import annotations

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
                    f"- role: {self.task.role}",
                    f"- complexity: {self.task.complexity.value}",
                    f"- depends_on: {', '.join(self.task.depends_on) if self.task.depends_on else 'none'}",
                ]
            )
            if self.task.acceptance_criteria:
                lines.extend(
                    ["- acceptance_criteria:"]
                    + [f"  - {item}" for item in self.task.acceptance_criteria]
                )
            if self.task.target_files:
                lines.extend(
                    ["- target_files:"]
                    + [f"  - {item}" for item in self.task.target_files]
                )
            if self.task.required_artifacts:
                lines.extend(
                    ["- required_artifacts:"]
                    + [f"  - {item}" for item in self.task.required_artifacts]
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
