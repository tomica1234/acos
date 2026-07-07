from __future__ import annotations

from pathlib import Path

import yaml

from apps import cli
from packages.schemas.models import JobStatus
from tests.fakes import build_approval_harness


def _patch_runner(monkeypatch, harness) -> None:
    monkeypatch.setattr(
        cli,
        "build_default_runner",
        lambda config_dir, workspace_root: (harness.runner, harness.environment),
    )


def test_cli_approvals_list_and_show(tmp_path: Path, monkeypatch, capsys) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    _patch_runner(monkeypatch, harness)

    exit_code = cli.main(
        [
            "approvals",
            "list",
            "--config-dir",
            "configs",
            "--workspace",
            str(harness.workspace),
        ]
    )
    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["approvals"][0]["id"] == record.pending_approval_id

    exit_code = cli.main(
        [
            "approvals",
            "show",
            record.pending_approval_id,
            "--config-dir",
            "configs",
            "--workspace",
            str(harness.workspace),
        ]
    )
    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["approval"]["id"] == record.pending_approval_id


def test_cli_approvals_approve_and_resume(tmp_path: Path, monkeypatch, capsys) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    _patch_runner(monkeypatch, harness)

    exit_code = cli.main(
        [
            "approvals",
            "approve",
            record.pending_approval_id,
            "--config-dir",
            "configs",
            "--workspace",
            str(harness.workspace),
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["approval"]["status"] == "approved"
    assert payload["job"]["status"] == JobStatus.DONE.value


def test_cli_approvals_reject_blocks_job(tmp_path: Path, monkeypatch, capsys) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    _patch_runner(monkeypatch, harness)

    exit_code = cli.main(
        [
            "approvals",
            "reject",
            record.pending_approval_id,
            "--config-dir",
            "configs",
            "--workspace",
            str(harness.workspace),
            "--reason",
            "blocked by reviewer",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["approval"]["status"] == "rejected"
    assert payload["job"]["status"] == JobStatus.BLOCKED.value


def test_cli_jobs_resume(tmp_path: Path, monkeypatch, capsys) -> None:
    harness = build_approval_harness(tmp_path)
    record = harness.run_job()
    _patch_runner(monkeypatch, harness)
    harness.runner.approval_gateway.approve(
        record.pending_approval_id,
        token=None,
        approver="cli",
    )

    exit_code = cli.main(
        [
            "jobs",
            "resume",
            record.job_id,
            "--config-dir",
            "configs",
            "--workspace",
            str(harness.workspace),
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["job"]["status"] == JobStatus.DONE.value
