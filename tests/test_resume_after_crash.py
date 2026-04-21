from __future__ import annotations

from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.worker_daemon import WorkerConfig, WorkerDaemon
from packages.schemas.checkpoints import CheckpointRecord
from packages.schemas.jobs import JobSpec
from packages.schemas.models import JobStatus


class _RecoveringRunner:
    def __init__(self, store: InMemoryJobStore) -> None:
        self.store = store
        self.calls = 0

    def run_next_step(self, job_id: str):
        self.calls += 1
        assert self.store.get_checkpoint(job_id=job_id, checkpoint_key="branch_prepared") is not None
        record = self.store.get(job_id)
        record.status = JobStatus.DONE
        return self.store.update(record)


class _StubRuntimeManager:
    def maybe_resume_waiting_jobs(self):
        return []


def test_worker_recovers_stale_job_from_checkpoint(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(request_text="recover work", repo_path=str(tmp_path)),
        status=JobStatus.RUNNING,
    )
    record.heartbeat_at = record.created_at.replace(year=record.created_at.year - 1)
    store.update(record)
    store.save_checkpoint(
        CheckpointRecord(
            job_id=record.job_id,
            checkpoint_key="branch_prepared",
            step_name="prepare_branch",
            idempotency_key="branch_prepared",
            status="completed",
        )
    )
    runner = _RecoveringRunner(store)
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1", recover_stale_jobs_after_seconds=1),
    )

    daemon.recover_stale_jobs()
    daemon.run_once()

    assert store.get(record.job_id).status == JobStatus.DONE
    assert runner.calls == 1
