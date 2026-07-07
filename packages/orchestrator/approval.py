"""Approval gateway and SQLite approval persistence."""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

from packages.memory.redaction import redact_text, redact_value
from packages.schemas.approvals import (
    ApprovalChallenge,
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
)
from packages.schemas.jobs import utc_now


def _hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


class ApprovalError(RuntimeError):
    """Base class for approval failures."""


class SQLiteApprovalStore:
    """Small SQLite store for approval requests."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, approval: ApprovalRequest) -> ApprovalRequest:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals(id, job_id, status, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    job_id=excluded.job_id,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at
                """,
                (
                    approval.id,
                    approval.job_id,
                    approval.status.value,
                    approval.model_dump_json(),
                    approval.created_at.isoformat(),
                ),
            )
            conn.commit()
        return approval

    def get(self, approval_id: str) -> ApprovalRequest:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM approvals WHERE id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return ApprovalRequest.model_validate_json(row[0])

    def list(self, *, job_id: str | None = None, pending_only: bool = False) -> list[ApprovalRequest]:
        query = "SELECT payload_json FROM approvals"
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
        return [ApprovalRequest.model_validate_json(row[0]) for row in rows]


class ApprovalGateway:
    """Create and resolve human approvals for high-risk operations."""

    def __init__(
        self,
        store: SQLiteApprovalStore,
        *,
        request_ttl_minutes: int = 1440,
        require_signed_tokens: bool = True,
        base_url: str = "http://127.0.0.1:8080",
    ) -> None:
        self.store = store
        self.request_ttl_minutes = request_ttl_minutes
        self.require_signed_tokens = require_signed_tokens
        self.base_url = base_url.rstrip("/")

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
            approval_token_hash=_hash_token(token) if token is not None else None,
        )
        self.store.save(approval)
        return ApprovalChallenge(
            request=approval,
            token=token,
            approve_url=f"{self.base_url}/approvals/{approval.id}/approve?token={token}" if token else None,
            reject_url=f"{self.base_url}/approvals/{approval.id}/reject?token={token}" if token else None,
        )

    def approve(self, approval_id: str, token: str | None, approver: str | None) -> ApprovalRequest:
        approval = self._pending(approval_id)
        self._validate_token(approval, token)
        approval.status = ApprovalStatus.APPROVED
        approval.approver = approver or "token"
        approval.approval_token_hash = None
        approval.resolved_at = utc_now()
        return self.store.save(approval)

    def reject(
        self,
        approval_id: str,
        token: str | None,
        approver: str | None,
        reason: str | None,
    ) -> ApprovalRequest:
        approval = self._pending(approval_id)
        self._validate_token(approval, token)
        approval.status = ApprovalStatus.REJECTED
        approval.approver = approver or "token"
        approval.approval_token_hash = None
        approval.resolution_reason = redact_text(reason or "rejected")
        approval.resolved_at = utc_now()
        return self.store.save(approval)

    def get_pending(self, job_id: str | None = None) -> list[ApprovalRequest]:
        return [self._expire_if_needed(item) for item in self.store.list(job_id=job_id, pending_only=True)]

    def _pending(self, approval_id: str) -> ApprovalRequest:
        approval = self._expire_if_needed(self.store.get(approval_id))
        if approval.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"approval request is already {approval.status.value}")
        return approval

    def _expire_if_needed(self, approval: ApprovalRequest) -> ApprovalRequest:
        if approval.expires_at and approval.expires_at < utc_now() and approval.status == ApprovalStatus.PENDING:
            approval.status = ApprovalStatus.EXPIRED
            approval.resolved_at = utc_now()
            self.store.save(approval)
        return approval

    @staticmethod
    def _validate_token(approval: ApprovalRequest, token: str | None) -> None:
        if approval.approval_token_hash is None:
            return
        if token is None or _hash_token(token) != approval.approval_token_hash:
            raise ApprovalError("invalid approval token")
