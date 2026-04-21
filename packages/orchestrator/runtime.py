"""Runtime wait/retry management for provider outages."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import yaml

from packages.orchestrator.job_store import JobStore, utc_now
from packages.orchestrator.provider_health import ProviderHealthChecker
from packages.schemas.jobs import JobRecord
from packages.schemas.models import JobStatus
from packages.schemas.runtime import (
    ProviderHealthStatus,
    RuntimeConfig,
    RuntimeIssue,
    RuntimeIssueStatus,
    RuntimeIssueType,
)


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider outage should pause runtime instead of failing."""

    def __init__(self, *, provider_key: str, model_key: str | None, issue_type: RuntimeIssueType, message: str) -> None:
        self.provider_key = provider_key
        self.model_key = model_key
        self.issue_type = issue_type
        super().__init__(message)


class RuntimeWaitRequiredError(RuntimeError):
    """Raised when job execution is paused by the runtime manager."""


class RuntimeManager:
    """Pause and resume jobs around provider health issues."""

    def __init__(
        self,
        store: JobStore,
        health_checker: ProviderHealthChecker,
        *,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.store = store
        self.health_checker = health_checker
        self.config = config or RuntimeConfig()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        store: JobStore,
        health_checker: ProviderHealthChecker,
    ) -> "RuntimeManager":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(store=store, health_checker=health_checker, config=RuntimeConfig(**payload["runtime"]))

    def handle_provider_error(
        self,
        *,
        record: JobRecord,
        provider_key: str,
        model_key: str | None,
        issue_type: RuntimeIssueType,
        message: str,
    ) -> RuntimeIssue:
        reaction = self.config.on_provider_unavailable
        if issue_type == RuntimeIssueType.AUTH_ERROR:
            reaction = self.config.on_auth_error
        elif issue_type == RuntimeIssueType.MODEL_NOT_FOUND:
            reaction = self.config.on_model_not_found
        issue = RuntimeIssue(
            id=uuid4().hex,
            job_id=record.job_id,
            provider_key=provider_key,
            model_key=model_key,
            issue_type=issue_type,
            message=message,
            status=RuntimeIssueStatus.WAITING if reaction.action == "wait_and_retry" else RuntimeIssueStatus.BLOCKED,
            retry_count=0,
            next_retry_at=utc_now() + timedelta(seconds=self.config.provider_health_check.check_interval_seconds),
        )
        self.store.save_runtime_issue(issue)
        record.pending_runtime_issue_id = issue.id
        record.runtime_error = message
        record.provider_status = issue_type.value
        if reaction.action == "wait_and_retry":
            record.status = JobStatus(
                reaction.mark_job_status or JobStatus.WAITING_RUNTIME.value
            )
        else:
            record.status = JobStatus.BLOCKED
        self.store.update(record)
        return issue

    def maybe_resume_waiting_jobs(self) -> list[JobRecord]:
        resumed: list[JobRecord] = []
        waiting_statuses = [
            JobStatus.WAITING_RUNTIME,
            JobStatus.PROVIDER_UNAVAILABLE,
            JobStatus.RETRYING_PROVIDER,
        ]
        for record in self.store.list_jobs(statuses=waiting_statuses):
            if not record.pending_runtime_issue_id:
                continue
            issue = self.store.get_runtime_issue(record.pending_runtime_issue_id)
            if issue.status == RuntimeIssueStatus.RESOLVED:
                continue
            if issue.next_retry_at is not None and issue.next_retry_at > utc_now():
                continue
            health = (
                self.health_checker.check_model(issue.model_key)
                if issue.model_key
                else self.health_checker.check_provider(issue.provider_key)
            )
            if health.status == ProviderHealthStatus.OK:
                issue.status = RuntimeIssueStatus.RESOLVED
                issue.resolved_at = utc_now()
                issue.updated_at = utc_now()
                self.store.save_runtime_issue(issue)
                record.pending_runtime_issue_id = None
                record.runtime_error = None
                record.provider_status = ProviderHealthStatus.OK.value
                if self.config.resume.auto_resume_after_provider_recovery:
                    record.status = JobStatus.RESUMING
                else:
                    record.status = JobStatus.PAUSED
                    record.runtime_error = "provider recovered; waiting for manual resume"
                self.store.update(record)
                resumed.append(record)
                continue
            issue.retry_count += 1
            backoff = min(
                self.config.provider_health_check.check_interval_seconds * max(issue.retry_count, 1),
                self.config.provider_health_check.max_backoff_seconds,
            )
            issue.next_retry_at = utc_now() + timedelta(seconds=backoff)
            issue.updated_at = utc_now()
            self.store.save_runtime_issue(issue)
        return resumed
