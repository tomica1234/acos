from __future__ import annotations

from packages.schemas.models import JobStatus
from tests.fakes import build_approval_harness


def test_waiting_approval_resume_flow(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()

    assert record.status == JobStatus.WAITING_APPROVAL
    assert harness.environment.notify_server.approval_notifications
    assert record.runtime_state["pending_approval_patch_role"] == "implementer"

    harness.runner.approval_gateway.approve(
        record.pending_approval_id,
        token=None,
        approver="cli",
    )
    resumed = harness.runner.resume_job(record.job_id)

    assert resumed.status == JobStatus.DONE
    assert "pending_approval_patch" not in resumed.runtime_state
    assert "pending_approval_patch_role" not in resumed.runtime_state


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
    assert "pending_approval_patch" not in blocked.runtime_state
    assert "pending_approval_patch_role" not in blocked.runtime_state


def test_approved_patch_is_revalidated_before_resume_apply(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    test_path = harness.workspace / "tests/test_feature.py"
    original = (
        "def test_feature_status() -> None:\n"
        "    result = build_feature()\n"
        "    assert result.status == 'ready'\n"
    )
    test_path.parent.mkdir(parents=True)
    test_path.write_text(original, encoding="utf-8")
    record.runtime_state["pending_approval_patch"] = {
        "path": "tests/test_feature.py",
        "content": (
            "def test_feature_status() -> None:\n"
            "    result = build_feature()\n"
            "    result.status\n"
        ),
        "operation": "update",
        "new_path": None,
        "unified_diff": None,
        "base_sha256": None,
        "expected_old_content": None,
        "executable": None,
    }
    record.runtime_state["pending_approval_patch_role"] = "fixer"
    harness.runner.store.update(record)

    harness.runner.approval_gateway.approve(
        record.pending_approval_id,
        token=None,
        approver="cli",
    )
    resumed = harness.runner.resume_job(record.job_id)

    assert test_path.read_text(encoding="utf-8") == original
    assert resumed.pending_approval_id is None
    assert resumed.status == JobStatus.WRITING_TESTS
    assert "pending_approval_patch" not in resumed.runtime_state
    assert "pending_approval_patch_role" not in resumed.runtime_state
    assert resumed.runtime_state["recovery_plan"]["reason"].startswith(
        "test_patch_quality_failed:"
    )
    assert resumed.runtime_state["recovery_plan"]["strategy"] == "RETURN_TO_TEST_WRITER"
