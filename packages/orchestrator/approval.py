"""Approval gateway and persistent approval store."""

from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from packages.memory.redaction import redact_text, redact_value
from packages.schemas.approvals import (
    ApprovalChallenge,
    ApprovalRequest,
    ApprovalStatus,
    PolicyAction,
    RiskDecision,
    RiskLevel,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


class ApprovalError(RuntimeError):
    """Base class for approval-related failures."""


class ApprovalRequiredError(RuntimeError):
    """Raised when an operation requires human approval before continuing."""

    def __init__(
        self,
        *,
        requested_by: str,
        operation: str,
        decision: RiskDecision,
        proposed_action: dict,
        task_id: str | None = None,
    ) -> None:
        self.requested_by = requested_by
        self.operation = operation
        self.decision = decision
        self.proposed_action = redact_value(proposed_action)
        self.task_id = task_id
        super().__init__(decision.reason)


class SQLiteApprovalStore:
    """SQLite-backed store for approval requests."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    task_id TEXT,
                    role TEXT,
                    requested_by TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    proposed_action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    approval_token_hash TEXT,
                    approver TEXT,
                    resolution_reason TEXT,
                    resolved_at TEXT
                )
                """
            )
            conn.commit()

    def save(self, approval: ApprovalRequest) -> ApprovalRequest:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals(
                    id, job_id, task_id, role, requested_by, operation, risk_level, reason,
                    proposed_action, created_at, expires_at, status, approval_token_hash,
                    approver, resolution_reason, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    task_id=excluded.task_id,
                    role=excluded.role,
                    requested_by=excluded.requested_by,
                    operation=excluded.operation,
                    risk_level=excluded.risk_level,
                    reason=excluded.reason,
                    proposed_action=excluded.proposed_action,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at,
                    status=excluded.status,
                    approval_token_hash=excluded.approval_token_hash,
                    approver=excluded.approver,
                    resolution_reason=excluded.resolution_reason,
                    resolved_at=excluded.resolved_at
                """,
                (
                    approval.id,
                    approval.job_id,
                    approval.task_id,
                    approval.role,
                    approval.requested_by,
                    approval.operation,
                    approval.risk_level.value,
                    approval.reason,
                    json.dumps(approval.proposed_action, sort_keys=True),
                    approval.created_at.isoformat(),
                    approval.expires_at.isoformat() if approval.expires_at else None,
                    approval.status.value,
                    approval.approval_token_hash,
                    approval.approver,
                    approval.resolution_reason,
                    approval.resolved_at.isoformat() if approval.resolved_at else None,
                ),
            )
            conn.commit()
        return approval

    def get(self, approval_id: str) -> ApprovalRequest:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, job_id, task_id, role, requested_by, operation, risk_level, reason,
                       proposed_action, created_at, expires_at, status, approval_token_hash,
                       approver, resolution_reason, resolved_at
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return ApprovalRequest(
            id=row[0],
            job_id=row[1],
            task_id=row[2],
            role=row[3],
            requested_by=row[4],
            operation=row[5],
            risk_level=RiskLevel(row[6]),
            reason=row[7],
            proposed_action=json.loads(row[8]),
            created_at=datetime.fromisoformat(row[9]),
            expires_at=datetime.fromisoformat(row[10]) if row[10] else None,
            status=ApprovalStatus(row[11]),
            approval_token_hash=row[12],
            approver=row[13],
            resolution_reason=row[14],
            resolved_at=datetime.fromisoformat(row[15]) if row[15] else None,
        )

    def list(self, *, job_id: str | None = None, pending_only: bool = False) -> list[ApprovalRequest]:
        query = """
            SELECT id, job_id, task_id, role, requested_by, operation, risk_level, reason,
                   proposed_action, created_at, expires_at, status, approval_token_hash,
                   approver, resolution_reason, resolved_at
            FROM approvals
        """
        clauses: list[str] = []
        params: list[str] = []
        if job_id is not None:
            clauses.append("job_id = ?")
            params.append(job_id)
        if pending_only:
            clauses.append("status = 'pending'")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ApprovalRequest(
                id=row[0],
                job_id=row[1],
                task_id=row[2],
                role=row[3],
                requested_by=row[4],
                operation=row[5],
                risk_level=RiskLevel(row[6]),
                reason=row[7],
                proposed_action=json.loads(row[8]),
                created_at=datetime.fromisoformat(row[9]),
                expires_at=datetime.fromisoformat(row[10]) if row[10] else None,
                status=ApprovalStatus(row[11]),
                approval_token_hash=row[12],
                approver=row[13],
                resolution_reason=row[14],
                resolved_at=datetime.fromisoformat(row[15]) if row[15] else None,
            )
            for row in rows
        ]


class ApprovalGateway:
    """Create, inspect, and resolve approval requests."""

    def __init__(
        self,
        store: SQLiteApprovalStore,
        *,
        request_ttl_minutes: int = 1440,
        allow_cli_approval: bool = True,
        allow_http_approval: bool = True,
        allow_notification_links: bool = True,
        require_signed_tokens: bool = True,
        base_url: str = "http://127.0.0.1:8080",
    ) -> None:
        self.store = store
        self.request_ttl_minutes = request_ttl_minutes
        self.allow_cli_approval = allow_cli_approval
        self.allow_http_approval = allow_http_approval
        self.allow_notification_links = allow_notification_links
        self.require_signed_tokens = require_signed_tokens
        self.base_url = base_url.rstrip("/")

    def request_approval(
        self,
        job_id: str,
        task_id: str | None,
        role: str | None,
        requested_by: str,
        operation: str,
        risk_level: RiskLevel,
        reason: str,
        proposed_action: dict,
    ) -> ApprovalRequest:
        return self.create_challenge(
            job_id=job_id,
            task_id=task_id,
            role=role,
            requested_by=requested_by,
            operation=operation,
            risk_level=risk_level,
            reason=reason,
            proposed_action=proposed_action,
        ).request

    def create_challenge(
        self,
        *,
        job_id: str,
        task_id: str | None,
        role: str | None,
        requested_by: str,
        operation: str,
        risk_level: RiskLevel,
        reason: str,
        proposed_action: dict,
    ) -> ApprovalChallenge:
        token = secrets.token_urlsafe(24) if self.require_signed_tokens else None
        token_hash = _hash_token(token) if token is not None else None
        approval = ApprovalRequest(
            job_id=job_id,
            task_id=task_id,
            role=role,
            requested_by=requested_by,
            operation=operation,
            risk_level=risk_level,
            reason=redact_text(reason),
            proposed_action=redact_value(proposed_action),
            expires_at=utc_now() + timedelta(minutes=self.request_ttl_minutes),
            approval_token_hash=token_hash,
        )
        self.store.save(approval)
        approve_url = None
        reject_url = None
        if self.allow_notification_links and token is not None:
            approve_url = f"{self.base_url}/approvals/{approval.id}/approve?token={token}"
            reject_url = f"{self.base_url}/approvals/{approval.id}/reject?token={token}"
        return ApprovalChallenge(
            request=approval,
            token=token,
            approve_url=approve_url,
            reject_url=reject_url,
        )

    def approve(
        self,
        approval_id: str,
        token: str | None,
        approver: str | None,
    ) -> ApprovalRequest:
        approval = self._get_pending_or_raise(approval_id)
        self._validate_token_or_actor(approval, token=token, approver=approver)
        approval.status = ApprovalStatus.APPROVED
        approval.approver = approver or "token"
        approval.approval_token_hash = None
        approval.resolved_at = utc_now()
        self.store.save(approval)
        return approval

    def reject(
        self,
        approval_id: str,
        token: str | None,
        approver: str | None,
        reason: str | None,
    ) -> ApprovalRequest:
        approval = self._get_pending_or_raise(approval_id)
        self._validate_token_or_actor(approval, token=token, approver=approver)
        approval.status = ApprovalStatus.REJECTED
        approval.approver = approver or "token"
        approval.approval_token_hash = None
        approval.resolution_reason = redact_text(reason or "rejected")
        approval.resolved_at = utc_now()
        self.store.save(approval)
        return approval

    def get(self, approval_id: str) -> ApprovalRequest:
        approval = self.store.get(approval_id)
        if self._is_expired(approval) and approval.status == ApprovalStatus.PENDING:
            approval.status = ApprovalStatus.EXPIRED
            approval.resolved_at = utc_now()
            self.store.save(approval)
        return approval

    def get_pending(self, job_id: str | None = None) -> list[ApprovalRequest]:
        approvals = self.store.list(job_id=job_id, pending_only=True)
        pending: list[ApprovalRequest] = []
        for item in approvals:
            current = self.get(item.id)
            if current.status == ApprovalStatus.PENDING:
                pending.append(current)
        return pending

    def list_all(self, job_id: str | None = None) -> list[ApprovalRequest]:
        return [self.get(item.id) for item in self.store.list(job_id=job_id, pending_only=False)]

    def _get_pending_or_raise(self, approval_id: str) -> ApprovalRequest:
        approval = self.get(approval_id)
        if approval.status == ApprovalStatus.EXPIRED:
            raise ApprovalError("approval request has expired")
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"approval request is already {approval.status.value}")
        return approval

    @staticmethod
    def _is_expired(approval: ApprovalRequest) -> bool:
        return approval.expires_at is not None and approval.expires_at <= utc_now()

    def _validate_token_or_actor(
        self,
        approval: ApprovalRequest,
        *,
        token: str | None,
        approver: str | None,
    ) -> None:
        if token is not None:
            if approval.approval_token_hash != _hash_token(token):
                raise ApprovalError("invalid approval token")
            return
        if approver is not None and self.allow_cli_approval:
            return
        if self.require_signed_tokens:
            raise ApprovalError("approval token is required")
        if approver is None:
            raise ApprovalError("approver identity is required")


@dataclass(slots=True)
class PolicyOutcome:
    """Resolved policy outcome captured before tool execution."""

    decision: RiskDecision
    approval: ApprovalRequest | None = None
    challenge: ApprovalChallenge | None = None
