"""Schema definitions for provider, model, and routing configuration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

OutputTokenSetting = int | Literal["auto"]


class ProviderType(str, Enum):
    OPENAI_COMPATIBLE = "openai_compatible"
    MOCK = "mock"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


class ImplementationStatus(str, Enum):
    IMPLEMENTED = "implemented"
    BLOCKED = "blocked"
    FAILED = "failed"


class FixStatus(str, Enum):
    FIXED = "fixed"
    STUCK = "stuck"
    FAILED = "failed"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUBMITTED = "submitted"
    WAITING_RUNTIME = "waiting_runtime"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RETRYING_PROVIDER = "retrying_provider"
    WAITING_APPROVAL = "waiting_approval"
    PAUSED = "paused"
    RESUMING = "resuming"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CRASHED = "crashed"
    RECOVERING = "recovering"
    ANALYZING = "analyzing"
    DESIGNING = "designing"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    WRITING_TESTS = "writing_tests"
    REVIEWING = "reviewing"
    TESTING = "testing"
    FIXING = "fixing"
    FINALIZING = "finalizing"
    DONE = "done"
    BLOCKED = "blocked"
    STUCK = "stuck"
    FAILED = "failed"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    TODO = "todo"
    READY = "ready"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_RUNTIME = "waiting_runtime"
    PAUSED = "paused"
    RESUMING = "resuming"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTS_WRITTEN = "tests_written"
    UNDER_REVIEW = "under_review"
    CHANGES_REQUESTED = "changes_requested"
    TEST_RUNNING = "test_running"
    TEST_FAILED = "test_failed"
    DONE = "done"
    BLOCKED = "blocked"
    STUCK = "stuck"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ModelCallStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK_USED = "fallback_used"
    ESCALATED = "escalated"


class RoutingReason(str, Enum):
    ROLE_DEFAULT = "role_default"
    FALLBACK = "fallback"
    ESCALATION = "escalation"
    CAPABILITY_REQUIRED = "capability_required"
    CONTEXT_BUDGET = "context_budget"


class TaskComplexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelProviderConfig(BaseModel):
    """Provider-level configuration for model access."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: ProviderType
    base_url: str
    api_key_env: str
    allow_empty_api_key: bool = False
    default_api_key: str | None = None
    timeout_seconds: int = Field(default=60)
    default_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    max_context_tokens: int | None = None
    default_max_output_tokens: int | None = None

    @computed_field
    @property
    def timeout(self) -> int:
        return self.timeout_seconds


class ModelConfig(BaseModel):
    """Model-level configuration."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    provider: str
    model: str
    display_name: str
    max_context_tokens: int
    max_output_tokens: OutputTokenSetting
    supports_tool_calling: bool = False
    supports_structured_output: bool = False
    supports_json_repair: bool = True
    cost_hints: dict[str, float] | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output_limit(self) -> "ModelConfig":
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0 when provided as an integer")
        return self


class AgentModelConfig(BaseModel):
    """Role-to-model mapping."""

    model_config = ConfigDict(extra="forbid")

    role: str
    primary_model: str
    fallback_models: list[str] = Field(default_factory=list)
    temperature: float = 0.0
    top_p: float | None = None
    max_output_tokens: OutputTokenSetting
    context_budget_tokens: int
    allow_tools: bool = True
    allowed_tools: list[str] = Field(default_factory=list)
    require_json_schema: bool = True
    escalation_policy: dict[str, Any] = Field(default_factory=dict)
    output_schema: str

    @model_validator(mode="after")
    def validate_sampling(self) -> "AgentModelConfig":
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1 when provided")
        if self.context_budget_tokens <= 0:
            raise ValueError("context_budget_tokens must be > 0")
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be > 0 when provided as an integer")
        return self


class EscalationCondition(BaseModel):
    """Routing escalation thresholds."""

    model_config = ConfigDict(extra="forbid")

    repeated_failures_gte: int | None = None
    same_test_failure_gte: int | None = None
    changed_files_gte: int | None = None
    task_complexity_in: list[TaskComplexity] = Field(default_factory=list)
    security_sensitive: bool | None = None


class RoleEscalationConfig(BaseModel):
    """Per-role escalation rules."""

    model_config = ConfigDict(extra="forbid")

    escalate_when: EscalationCondition
    escalated_model: str


class FallbackConfig(BaseModel):
    """Fallback configuration."""

    model_config = ConfigDict(extra="forbid")

    on_errors: list[str] = Field(default_factory=list)


class CapabilityRequirements(BaseModel):
    """Roles requiring specific model capabilities."""

    model_config = ConfigDict(extra="forbid")

    roles_requiring_tools: list[str] = Field(default_factory=list)
    roles_requiring_strict_json: list[str] = Field(default_factory=list)


class ModelRoutingConfig(BaseModel):
    """Router configuration."""

    model_config = ConfigDict(extra="forbid")

    default_strategy: str = "role_primary"
    escalation: dict[str, RoleEscalationConfig] = Field(default_factory=dict)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    capability_requirements: CapabilityRequirements = Field(
        default_factory=CapabilityRequirements
    )


class ModelSelection(BaseModel):
    """A concrete routing decision for a role invocation."""

    model_config = ConfigDict(extra="forbid")

    role: str
    model_key: str
    provider_key: str
    reason: RoutingReason
    details: dict[str, Any] = Field(default_factory=dict)
    temperature: float
    top_p: float | None = None
    max_output_tokens: OutputTokenSetting

    @computed_field
    @property
    def model_id(self) -> str:
        return self.model_key

    @computed_field
    @property
    def provider(self) -> str:
        return self.provider_key


class ModelResult(BaseModel):
    """Normalized adapter output."""

    model_config = ConfigDict(extra="forbid")

    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] | None = None
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    output_truncated: bool = False


class ModelCallRecord(BaseModel):
    """Audit-ready model invocation record."""

    model_config = ConfigDict(extra="forbid")

    role: str
    model_key: str
    provider_key: str
    status: ModelCallStatus
    input_hash: str
    output_hash: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    total_tokens_estimate: int
    error: str | None = None
    finish_reason: str | None = None
    configured_max_output_tokens: OutputTokenSetting | None = None
    estimated_input_tokens: int | None = None
    resolved_max_output_tokens: int | None = None
    model_max_context_tokens: int | None = None
    safety_margin_tokens: int | None = None
    context_budget_tokens: int | None = None
    output_truncated: bool = False

    @computed_field
    @property
    def model_id(self) -> str:
        return self.model_key

    @computed_field
    @property
    def provider(self) -> str:
        return self.provider_key
