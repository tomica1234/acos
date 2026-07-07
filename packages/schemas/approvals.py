"""Approval and policy-risk schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyAction(StrEnum):
    ALLOW = "allow"
    ALLOW_AND_AUDIT = "allow_and_audit"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RiskDecision(BaseModel):
    """Normalized risk classification for a requested operation."""

    model_config = ConfigDict(extra="forbid")

    operation: str
    policy_action: PolicyAction
    risk_level: RiskLevel
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    """Approval request persisted by the approval gateway."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid4().hex)
    job_id: str
    task_id: str | None = None
    role: str | None = None
    requested_by: str
    operation: str
    risk_level: RiskLevel
    reason: str
    proposed_action: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approval_token_hash: str | None = None
    approver: str | None = None
    resolution_reason: str | None = None
    resolved_at: datetime | None = None

    @computed_field
    @property
    def decided_at(self) -> datetime | None:
        return self.resolved_at

    @computed_field
    @property
    def decided_by(self) -> str | None:
        return self.approver

    @computed_field
    @property
    def reject_reason(self) -> str | None:
        return self.resolution_reason if self.status == ApprovalStatus.REJECTED else None


class ApprovalChallenge(BaseModel):
    """Ephemeral approval challenge including one-time token material."""

    model_config = ConfigDict(extra="forbid")

    request: ApprovalRequest
    token: str | None = None
    approve_url: str | None = None
    reject_url: str | None = None


class ApprovalActionPayload(BaseModel):
    """HTTP/API payload for approve/reject actions."""

    model_config = ConfigDict(extra="forbid")

    token: str | None = None
    approver: str | None = None
    reason: str | None = None
