"""Structured outputs returned by ACOS roles."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    ReviewDecision,
    Severity,
)
from packages.schemas.runtime import RuntimeHttpCheck


class FilePatch(BaseModel):
    """A file write proposed by an agent."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    operation: Literal["create", "update"] = "update"
    rationale: str | None = None


class PRD(BaseModel):
    """Product requirements document."""

    model_config = ConfigDict(extra="forbid")

    title: str
    problem_statement: str
    users: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    framework_profile: str | None = None
    framework_entrypoint: str | None = None
    framework_project_name: str | None = None
    required_artifacts: list[str] = Field(default_factory=list)
    runtime: "RuntimePlan | None" = None
    acceptance_checks: list[RuntimeHttpCheck] = Field(default_factory=list)


class RuntimePlan(BaseModel):
    """Execution-time runtime contract hints emitted by the PM."""

    model_config = ConfigDict(extra="forbid")

    prepare_commands: list[list[str]] = Field(default_factory=list)
    start_command: list[str] | None = None
    http_probe_path: str | None = None
    http_checks: list[RuntimeHttpCheck] = Field(default_factory=list)
    prepare_timeout_seconds: int | None = None
    startup_timeout_seconds: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PMReviewResult(BaseModel):
    """Product-manager review output for plans and delivered work."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    summary: str
    findings: list["Finding"] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    required_verifications: list[str] = Field(default_factory=list)


class ArchitecturePlan(BaseModel):
    """System architecture description."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    components: list[str] = Field(default_factory=list)
    data_flows: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)


class ImplementationResult(BaseModel):
    """Implementation agent output."""

    model_config = ConfigDict(extra="forbid")

    status: ImplementationStatus
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    patches: list[FilePatch] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TestWriterResult(BaseModel):
    """Test authoring agent output."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    changed_files: list[str] = Field(default_factory=list)
    patches: list[FilePatch] = Field(default_factory=list)
    test_strategy: list[str] = Field(default_factory=list)


TestWriterResult.__test__ = False


class Finding(BaseModel):
    """A review finding."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    title: str
    description: str
    file_path: str | None = None
    suggestion: str | None = None


class ReviewResult(BaseModel):
    """Code review output."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    summary: str
    findings: list[Finding] = Field(default_factory=list)


class SecurityReviewResult(BaseModel):
    """Security review output."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecision
    summary: str
    findings: list[Finding] = Field(default_factory=list)


class TestRunResult(BaseModel):
    """Deterministic test runner output."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    command: list[str] = Field(default_factory=list)
    failed_tests: list[str] = Field(default_factory=list)
    output_excerpt: str = ""
    exit_code: int = 0


TestRunResult.__test__ = False


class FixResult(BaseModel):
    """Fixer output."""

    model_config = ConfigDict(extra="forbid")

    status: FixStatus
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    patches: list[FilePatch] = Field(default_factory=list)
    addressed_failures: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)


class ReleaseResult(BaseModel):
    """Release manager output."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    commit_message: str
    notify_message: str


class SummaryResult(BaseModel):
    """Summary and memory output."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    memory_entries: list[str] = Field(default_factory=list)
