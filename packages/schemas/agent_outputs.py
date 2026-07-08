"""Structured outputs returned by ACOS roles."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.schemas.models import (
    FailureClassification,
    FailureRetryMode,
    FixStatus,
    ImplementationStatus,
    ReviewDecision,
    Severity,
    TestWriterStatus,
)
from packages.schemas.runtime import RuntimeHttpCheck


class FilePatch(BaseModel):
    """A file write proposed by an agent."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str | None = None
    operation: Literal["create", "update", "delete", "rename"] = "update"
    rationale: str | None = None
    new_path: str | None = None
    unified_diff: str | None = None
    base_sha256: str | None = None
    expected_old_content: str | None = None
    executable: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_path_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "path" not in normalized:
            for alias in ("file", "filename"):
                if alias in normalized:
                    normalized["path"] = normalized.pop(alias)
                    break
        return normalized

    @model_validator(mode="after")
    def validate_operation_payload(self) -> "FilePatch":
        if self.operation in {"create", "update"} and self.content is None and self.unified_diff is None:
            raise ValueError("create/update patches require content or unified_diff")
        if self.operation == "rename" and not self.new_path:
            raise ValueError("rename patches require new_path")
        return self


class PRD(BaseModel):
    """Product requirements document."""

    model_config = ConfigDict(extra="forbid")

    title: str
    problem_statement: str
    users: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    non_goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    smallest_working_core: list[str] = Field(default_factory=list)
    small_parts: list[str] = Field(default_factory=list)
    incremental_milestones: list[str] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    definition_of_done: list[str] = Field(default_factory=list)
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

    @model_validator(mode="before")
    @classmethod
    def collect_unknown_runtime_hints(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        allowed = {
            "prepare_commands",
            "start_command",
            "http_probe_path",
            "http_checks",
            "prepare_timeout_seconds",
            "startup_timeout_seconds",
            "extra",
        }
        normalized = dict(value)
        extra = normalized.get("extra")
        if not isinstance(extra, dict):
            extra = {}
        for key in list(normalized):
            if key in allowed:
                continue
            extra[key] = normalized.pop(key)
        normalized["extra"] = extra
        return normalized


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

    status: TestWriterStatus = TestWriterStatus.TESTS_WRITTEN
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

    @model_validator(mode="before")
    @classmethod
    def normalize_recommendation_alias(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "suggestion" not in normalized and "recommendation" in normalized:
            normalized["suggestion"] = normalized.pop("recommendation")
        return normalized


ReviewFinding = Finding


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
    executed_test_count: int | None = None


TestRunResult.__test__ = False


class FailureDiagnosis(BaseModel):
    """Structured diagnosis for a deterministic test or build failure."""

    model_config = ConfigDict(extra="forbid")

    classification: FailureClassification = FailureClassification.UNKNOWN
    root_cause: str
    failed_files: list[str] = Field(default_factory=list)
    failed_tests: list[str] = Field(default_factory=list)
    recommended_fix_strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    should_retry: bool = True
    retry_mode: FailureRetryMode = FailureRetryMode.NORMAL_FIX
    failure_signature: str | None = None


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
