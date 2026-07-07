from __future__ import annotations

from packages.schemas.models import JobStatus
from tests.fakes import build_approval_harness


def test_waiting_approval_resume_flow(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()

    assert record.status == JobStatus.WAITING_APPROVAL
    assert harness.environment.notify_server.approval_notifications

    harness.runner.approval_gateway.approve(record.pending_approval_id, token=None, approver="cli")
    resumed = harness.runner.resume_job(record.job_id)

    assert resumed.status == JobStatus.DONE


def test_waiting_approval_reject_flow(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()

    harness.runner.approval_gateway.reject(
        record.pending_approval_id,
        token=None,
        approver="cli",
        reason="blocked by reviewer",
    )
    blocked = harness.runner.resume_job(record.job_id)

    assert blocked.status == JobStatus.BLOCKED
