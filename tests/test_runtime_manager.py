from __future__ import annotations

from datetime import timedelta

from packages.orchestrator.job_store import InMemoryJobStore, utc_now
from packages.orchestrator.runtime import RuntimeManager
from packages.schemas.jobs import JobSpec
from packages.schemas.models import JobStatus
from packages.schemas.runtime import (
    ProviderHealth,
    ProviderHealthStatus,
    ResumeConfig,
    RuntimeConfig,
    RuntimeIssueType,
)


class _HealthChecker:
    def __init__(self, status: ProviderHealthStatus) -> None:
        self.status = status

    def check_model(self, model_key: str) -> ProviderHealth:
        return ProviderHealth(
            provider_key="local_qwen",
            model_key=model_key,
            status=self.status,
            message=self.status.value,
        )

    def check_provider(self, provider_key: str) -> ProviderHealth:
        return ProviderHealth(
            provider_key=provider_key,
            status=self.status,
            message=self.status.value,
        )


def _make_record(store: InMemoryJobStore, tmp_path) -> tuple[object, object]:
    record = store.create(
        JobSpec(
            request_text="durable runtime",
            repo_path=str(tmp_path),
            workspace_root=str(tmp_path),
        ),
        status=JobStatus.RUNNING,
    )
    return record, store


def test_runtime_manager_marks_waiting_runtime_for_provider_timeout(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record, _ = _make_record(store, tmp_path)
    manager = RuntimeManager(store=store, health_checker=_HealthChecker(ProviderHealthStatus.CONNECTION_ERROR))

    manager.handle_provider_error(
        record=record,
        provider_key="local_qwen",
        model_key="qwen_35b",
        issue_type=RuntimeIssueType.TIMEOUT,
        message="provider timed out",
    )

    assert store.get(record.job_id).status == JobStatus.WAITING_RUNTIME


def test_runtime_manager_blocks_on_auth_error(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record, _ = _make_record(store, tmp_path)
    manager = RuntimeManager(store=store, health_checker=_HealthChecker(ProviderHealthStatus.AUTH_ERROR))

    manager.handle_provider_error(
        record=record,
        provider_key="local_qwen",
        model_key="qwen_35b",
        issue_type=RuntimeIssueType.AUTH_ERROR,
        message="unauthorized",
    )

    assert store.get(record.job_id).status == JobStatus.BLOCKED


def test_runtime_manager_resumes_when_provider_recovers(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record, _ = _make_record(store, tmp_path)
    manager = RuntimeManager(store=store, health_checker=_HealthChecker(ProviderHealthStatus.OK))
    issue = manager.handle_provider_error(
        record=record,
        provider_key="local_qwen",
        model_key="qwen_35b",
        issue_type=RuntimeIssueType.CONNECTION_ERROR,
        message="connection error",
    )
    issue.next_retry_at = utc_now() - timedelta(seconds=1)
    store.save_runtime_issue(issue)

    resumed = manager.maybe_resume_waiting_jobs()

    assert resumed[0].status == JobStatus.RESUMING
    assert store.get(record.job_id).pending_runtime_issue_id is None


def test_runtime_manager_honors_manual_resume_after_recovery(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record, _ = _make_record(store, tmp_path)
    manager = RuntimeManager(
        store=store,
        health_checker=_HealthChecker(ProviderHealthStatus.OK),
        config=RuntimeConfig(
            resume=ResumeConfig(auto_resume_after_provider_recovery=False),
        ),
    )
    issue = manager.handle_provider_error(
        record=record,
        provider_key="local_qwen",
        model_key="qwen_35b",
        issue_type=RuntimeIssueType.CONNECTION_ERROR,
        message="connection error",
    )
    issue.next_retry_at = utc_now() - timedelta(seconds=1)
    store.save_runtime_issue(issue)

    manager.maybe_resume_waiting_jobs()

    assert store.get(record.job_id).status == JobStatus.PAUSED


def test_runtime_manager_applies_backoff(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record, _ = _make_record(store, tmp_path)
    manager = RuntimeManager(store=store, health_checker=_HealthChecker(ProviderHealthStatus.CONNECTION_ERROR))
    issue = manager.handle_provider_error(
        record=record,
        provider_key="local_qwen",
        model_key="qwen_35b",
        issue_type=RuntimeIssueType.CONNECTION_ERROR,
        message="connection error",
    )
    issue.next_retry_at = utc_now() - timedelta(seconds=1)
    store.save_runtime_issue(issue)

    manager.maybe_resume_waiting_jobs()
    updated = store.get_runtime_issue(issue.id)

    assert updated.retry_count == 1
    assert updated.next_retry_at is not None
