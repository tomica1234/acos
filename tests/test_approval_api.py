from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from apps.api.main import create_app
from packages.orchestrator.job_constraints import STRICT_JOB_CONSTRAINTS
from packages.schemas.models import JobStatus
from tests.fakes import build_approval_harness


def _assert_strict_constraints(payload: dict) -> None:
    constraints = payload["job"]["spec"]["metadata"]["constraints"]
    for key, value in STRICT_JOB_CONSTRAINTS.items():
        assert constraints[key] is value
    assert constraints["test_timeout_seconds"] == 1200


def _extract_token(url: str) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query)["token"][0]


def test_approval_api_list_and_show(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    app = create_app(job_runner=harness.runner, workspace_root=harness.workspace)
    client = TestClient(app)

    list_response = client.get("/approvals")
    show_response = client.get(f"/approvals/{record.pending_approval_id}")

    assert list_response.status_code == 200
    assert show_response.status_code == 200
    assert list_response.json()["approvals"][0]["id"] == record.pending_approval_id
    assert show_response.json()["id"] == record.pending_approval_id


def test_approval_api_approve_endpoint(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    app = create_app(job_runner=harness.runner, workspace_root=harness.workspace)
    client = TestClient(app)
    approve_url = harness.environment.notify_server.approval_notifications[0]["approve_url"]
    token = _extract_token(approve_url)

    invalid = client.get(
        f"/approvals/{record.pending_approval_id}/approve",
        params={"token": "invalid"},
    )
    assert invalid.status_code == 400

    response = client.get(
        f"/approvals/{record.pending_approval_id}/approve",
        params={"token": token},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval"]["status"] == "approved"
    assert payload["job"]["status"] == JobStatus.DONE.value
    _assert_strict_constraints(payload)


def test_approval_api_reject_endpoint(tmp_path) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    app = create_app(job_runner=harness.runner, workspace_root=harness.workspace)
    client = TestClient(app)
    reject_url = harness.environment.notify_server.approval_notifications[0]["reject_url"]
    token = _extract_token(reject_url)

    response = client.post(
        f"/approvals/{record.pending_approval_id}/reject",
        json={"token": token, "reason": "do not proceed"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval"]["status"] == "rejected"
    assert payload["job"]["status"] == JobStatus.BLOCKED.value
    _assert_strict_constraints(payload)
