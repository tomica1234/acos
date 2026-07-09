from __future__ import annotations

from packages.orchestrator.job_store import InMemoryJobStore
from packages.orchestrator.worker_daemon import WorkerConfig, WorkerDaemon
from packages.schemas.jobs import JobSpec
from packages.schemas.models import JobStatus


class _StubRunner:
    def __init__(self, store: InMemoryJobStore) -> None:
        self.store = store
        self.calls = 0

    def run_next_step(self, job_id: str):
        self.calls += 1
        record = self.store.get(job_id)
        record.status = JobStatus.DONE
        return self.store.update(record)


class _StubRuntimeManager:
    def maybe_resume_waiting_jobs(self):
        return []


class _TwoStepRunner:
    def __init__(self, store: InMemoryJobStore) -> None:
        self.store = store
        self.calls = 0

    def run_next_step(self, job_id: str):
        self.calls += 1
        record = self.store.get(job_id)
        if self.calls == 1:
            record.status = JobStatus.TESTING
        else:
            record.status = JobStatus.DONE
        return self.store.update(record)


class _ConstraintCapturingRunner:
    def __init__(self, store: InMemoryJobStore) -> None:
        self.store = store
        self.constraints: dict[str, object] = {}

    def run_next_step(self, job_id: str):
        record = self.store.get(job_id)
        self.constraints = dict(record.spec.metadata["constraints"])
        record.status = JobStatus.DONE
        return self.store.update(record)


def test_worker_daemon_processes_queued_job_and_updates_heartbeat(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(request_text="do work", repo_path=str(tmp_path)),
        status=JobStatus.QUEUED,
    )
    runner = _StubRunner(store)
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1"),
    )

    processed = daemon.run_once()

    assert processed[0].job_id == record.job_id
    assert store.get(record.job_id).status == JobStatus.DONE
    assert store.list_worker_heartbeats()[0].worker_id == "worker-1"
    assert store.get_job_lease(record.job_id) is None


def test_worker_daemon_applies_strict_constraints_before_runner(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(
            request_text="do work",
            repo_path=str(tmp_path),
            metadata={
                "constraints": {
                    "require_prd_quality": False,
                    "require_task_acceptance_criteria": False,
                    "require_task_artifacts": False,
                    "require_completion_integrity": False,
                    "require_test_evidence": False,
                    "require_stage_test_patches": False,
                    "stage_review": False,
                }
            },
        ),
        status=JobStatus.QUEUED,
    )
    runner = _ConstraintCapturingRunner(store)
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1"),
    )

    processed = daemon.run_once()

    assert processed[0].job_id == record.job_id
    assert runner.constraints == {
        "require_prd_quality": True,
        "require_task_acceptance_criteria": True,
        "require_task_artifacts": True,
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "stage_review": True,
        "test_timeout_seconds": 1200,
    }


def test_worker_daemon_does_not_double_execute_completed_job(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(request_text="do work", repo_path=str(tmp_path)),
        status=JobStatus.QUEUED,
    )
    runner = _StubRunner(store)
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1"),
    )

    daemon.run_once()
    processed = daemon.run_once()

    assert runner.calls == 1
    assert processed == []


def test_worker_daemon_run_until_job_settled_drives_queue_to_completion(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(request_text="do work", repo_path=str(tmp_path)),
        status=JobStatus.SUBMITTED,
    )
    runner = _TwoStepRunner(store)
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1", poll_interval_seconds=0),
    )

    settled = daemon.run_until_job_settled(record.job_id, max_iterations=5)

    assert settled.status == JobStatus.DONE
    assert runner.calls == 2


def test_worker_daemon_recovers_stale_jobs(tmp_path) -> None:
    store = InMemoryJobStore(tmp_path / ".jobs.json")
    record = store.create(
        JobSpec(request_text="resume work", repo_path=str(tmp_path)),
        status=JobStatus.RUNNING,
    )
    record.heartbeat_at = record.created_at.replace(year=record.created_at.year - 1)
    store.update(record)
    daemon = WorkerDaemon(
        runner=_StubRunner(store),
        store=store,
        runtime_manager=_StubRuntimeManager(),
        config=WorkerConfig(id="worker-1", recover_stale_jobs_after_seconds=1),
    )

    recovered = daemon.recover_stale_jobs()

    assert recovered[0].job_id == record.job_id
    assert store.get(record.job_id).status == JobStatus.RECOVERING
