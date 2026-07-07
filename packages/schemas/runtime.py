"""Runtime health, lease, and durable worker schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.llm.budget import TokenBudgetPolicy


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ProviderHealthStatus(StrEnum):
    OK = "ok"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    AUTH_ERROR = "auth_error"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_RESPONSE = "invalid_response"
    INVALID_JSON_RESPONSE = "invalid_json_response"


class RuntimeIssueStatus(StrEnum):
    OPEN = "open"
    WAITING = "waiting"
    RESOLVED = "resolved"
    BLOCKED = "blocked"


class RuntimeIssueType(StrEnum):
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    AUTH_ERROR = "auth_error"
    MODEL_NOT_FOUND = "model_not_found"
    INVALID_RESPONSE = "invalid_response"
    INVALID_JSON_RESPONSE = "invalid_json_response"


class ProviderHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_key: str
    model_key: str | None = None
    status: ProviderHealthStatus
    message: str
    checked_at: datetime = Field(default_factory=utc_now)
    response_time_ms: int | None = None
    model_available: bool | None = None


class RuntimeIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    job_id: str
    provider_key: str
    model_key: str | None = None
    issue_type: RuntimeIssueType
    message: str
    status: RuntimeIssueStatus = RuntimeIssueStatus.OPEN
    retry_count: int = 0
    next_retry_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class WorkerHeartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: str
    status: str = "alive"
    heartbeat_at: datetime = Field(default_factory=utc_now)
    details: dict[str, str] = Field(default_factory=dict)


class JobLease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    worker_id: str
    acquired_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime


class ProviderHealthCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_backoff_seconds: int = 300
    test_chat_completion: bool = True
    test_json_response: bool = False


class ResumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_resume_after_provider_recovery: bool = True
    require_manual_resume_after_auth_error: bool = True


class RuntimeReactionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    notify: bool = True
    mark_job_status: str | None = None


class RuntimeHttpCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = None
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"] = "GET"
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    form: dict[str, Any] | None = None
    json_payload: dict[str, Any] | list[Any] | None = Field(default=None, alias="json", serialization_alias="json")
    body: str | None = None
    expect_status: int = 200
    body_contains: list[str] = Field(default_factory=list)
    body_not_contains: list[str] = Field(default_factory=list)
    follow_redirects: bool = True
    use_csrf_from_last_response: bool = True

    @model_validator(mode="after")
    def validate_payload_shape(self) -> "RuntimeHttpCheck":
        if not self.path.startswith("/"):
            raise ValueError("path must start with '/'")
        payload_count = sum(
            value is not None
            for value in (self.form, self.json_payload, self.body)
        )
        if payload_count > 1:
            raise ValueError("only one of form, json, or body may be provided")
        if self.expect_status < 100 or self.expect_status > 599:
            raise ValueError("expect_status must be a valid HTTP status code")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_health_check: ProviderHealthCheckConfig = Field(
        default_factory=ProviderHealthCheckConfig
    )
    token_budget: TokenBudgetPolicy = Field(default_factory=TokenBudgetPolicy)
    on_provider_unavailable: RuntimeReactionConfig = Field(
        default_factory=lambda: RuntimeReactionConfig(
            action="wait_and_retry",
            notify=True,
            mark_job_status="waiting_runtime",
        )
    )
    on_model_not_found: RuntimeReactionConfig = Field(
        default_factory=lambda: RuntimeReactionConfig(action="block", notify=True)
    )
    on_auth_error: RuntimeReactionConfig = Field(
        default_factory=lambda: RuntimeReactionConfig(action="block", notify=True)
    )
    resume: ResumeConfig = Field(default_factory=ResumeConfig)
