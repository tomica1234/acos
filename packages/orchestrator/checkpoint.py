"""Checkpoint helpers for durable step execution."""

from __future__ import annotations

from packages.orchestrator.job_store import JobStore
from packages.schemas.checkpoints import CheckpointRecord


class CheckpointStore:
    """Thin helper around the JobStore checkpoint tables."""

    def __init__(self, store: JobStore) -> None:
        self.store = store

    def has_completed(
        self,
        *,
        job_id: str,
        checkpoint_key: str,
        task_id: str | None = None,
    ) -> bool:
        return any(
            checkpoint.status == "completed"
            for checkpoint in self.store.list_checkpoints(job_id=job_id, task_id=task_id)
            if checkpoint.checkpoint_key == checkpoint_key
        )

    def mark_started(
        self,
        *,
        job_id: str,
        checkpoint_key: str,
        step_name: str,
        idempotency_key: str,
        task_id: str | None = None,
        result_json: dict[str, object] | None = None,
    ) -> CheckpointRecord:
        return self.store.save_checkpoint(
            CheckpointRecord(
                job_id=job_id,
                task_id=task_id,
                checkpoint_key=checkpoint_key,
                step_name=step_name,
                idempotency_key=idempotency_key,
                status="started",
                result_json=result_json or {},
            )
        )

    def mark_completed(
        self,
        *,
        job_id: str,
        checkpoint_key: str,
        step_name: str,
        idempotency_key: str,
        task_id: str | None = None,
        result_json: dict[str, object] | None = None,
    ) -> CheckpointRecord:
        return self.store.save_checkpoint(
            CheckpointRecord(
                job_id=job_id,
                task_id=task_id,
                checkpoint_key=checkpoint_key,
                step_name=step_name,
                idempotency_key=idempotency_key,
                status="completed",
                result_json=result_json or {},
            )
        )
