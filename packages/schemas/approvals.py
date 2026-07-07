"""Approval request schemas."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from packages.schemas.jobs import utc_now


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class RiskDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: PolicyAction
    risk_level: RiskLevel
    reason: str


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    job_id: str
    task_id: str | None = None
    role: str | None = None
    requested_by: str
    operation: str
    risk_level: RiskLevel
    reason: str
    proposed_action: dict
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approval_token_hash: str | None = None
    approver: str | None = None
    resolution_reason: str | None = None
    resolved_at: datetime | None = None


class ApprovalChallenge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: ApprovalRequest
    token: str | None = None
    approve_url: str | None = None
    reject_url: str | None = None
