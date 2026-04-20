from __future__ import annotations

from pathlib import Path

import pytest

from packages.orchestrator.approval import ApprovalError, ApprovalGateway, SQLiteApprovalStore
from packages.schemas.approvals import RiskLevel


def _build_gateway(tmp_path: Path, *, ttl_minutes: int = 1440) -> ApprovalGateway:
    return ApprovalGateway(
        SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        request_ttl_minutes=ttl_minutes,
        allow_cli_approval=True,
        allow_http_approval=True,
        allow_notification_links=True,
        require_signed_tokens=True,
    )


def test_gateway_creates_approval_request(tmp_path: Path) -> None:
    gateway = _build_gateway(tmp_path)

    challenge = gateway.create_challenge(
        job_id="job-1",
        task_id="task-1",
        role="implementer",
        requested_by="implementer",
        operation="large_patch",
        risk_level=RiskLevel.HIGH,
        reason="patch is large",
        proposed_action={"tool_name": "repo_server.apply_patch"},
    )

    approval = gateway.get(challenge.request.id)
    db_bytes = (tmp_path / "approvals.sqlite3").read_bytes()
    assert approval.status == "pending"
    assert challenge.token is not None
    assert approval.approval_token_hash != challenge.token
    assert challenge.token.encode("utf-8") not in db_bytes


def test_gateway_can_approve(tmp_path: Path) -> None:
    gateway = _build_gateway(tmp_path)
    challenge = gateway.create_challenge(
        job_id="job-1",
        task_id=None,
        role=None,
        requested_by="release_manager",
        operation="release_publish",
        risk_level=RiskLevel.HIGH,
        reason="release publish requested",
        proposed_action={"tool_name": "notify_server.send_notification"},
    )

    approval = gateway.approve(challenge.request.id, token=challenge.token, approver=None)

    assert approval.status == "approved"
    assert approval.approval_token_hash is None


def test_gateway_can_reject(tmp_path: Path) -> None:
    gateway = _build_gateway(tmp_path)
    challenge = gateway.create_challenge(
        job_id="job-1",
        task_id=None,
        role=None,
        requested_by="release_manager",
        operation="release_publish",
        risk_level=RiskLevel.HIGH,
        reason="release publish requested",
        proposed_action={"tool_name": "notify_server.send_notification"},
    )

    approval = gateway.reject(
        challenge.request.id,
        token=challenge.token,
        approver=None,
        reason="not now",
    )

    assert approval.status == "rejected"
    assert approval.resolution_reason == "not now"


def test_expired_approval_cannot_be_approved(tmp_path: Path) -> None:
    gateway = _build_gateway(tmp_path, ttl_minutes=-1)
    challenge = gateway.create_challenge(
        job_id="job-1",
        task_id=None,
        role=None,
        requested_by="release_manager",
        operation="release_publish",
        risk_level=RiskLevel.HIGH,
        reason="release publish requested",
        proposed_action={"tool_name": "notify_server.send_notification"},
    )

    with pytest.raises(ApprovalError):
        gateway.approve(challenge.request.id, token=challenge.token, approver=None)


def test_one_time_token_cannot_be_reused(tmp_path: Path) -> None:
    gateway = _build_gateway(tmp_path)
    challenge = gateway.create_challenge(
        job_id="job-1",
        task_id=None,
        role=None,
        requested_by="release_manager",
        operation="release_publish",
        risk_level=RiskLevel.HIGH,
        reason="release publish requested",
        proposed_action={"tool_name": "notify_server.send_notification"},
    )

    gateway.approve(challenge.request.id, token=challenge.token, approver=None)

    with pytest.raises(ApprovalError):
        gateway.approve(challenge.request.id, token=challenge.token, approver=None)
