"""Worker process for ACOS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from packages.orchestrator.job_constraints import apply_strict_job_constraints
from packages.orchestrator.job_runner import build_default_runner
from packages.orchestrator.job_store import SQLiteJobStore
from packages.orchestrator.worker_daemon import WorkerConfig, WorkerDaemon
from packages.schemas.jobs import JobSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acos-worker")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--request")
    parser.add_argument("--branch", default="acos/default")
    parser.add_argument("--sqlite-path", default=".acos/acos.sqlite3")
    parser.add_argument("--job-id")
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--worker-id", default="local-worker")
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = SQLiteJobStore(args.sqlite_path)
    runner, _ = build_default_runner(
        config_dir=args.config_dir,
        workspace_root=Path(args.repo),
        store=store,
    )
    daemon = WorkerDaemon(
        runner=runner,
        store=store,
        config=WorkerConfig(
            id=args.worker_id,
            poll_interval_seconds=args.poll_interval_seconds,
        ),
    )
    if args.forever:
        daemon.run_forever()
        return 0
    if args.job_id:
        record = daemon.run_once(args.job_id)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value in {"done", "waiting_runtime", "waiting_approval"} else 1
    if not args.request:
        raise SystemExit("--request is required unless --job-id or --forever is used")
    spec = JobSpec(
        request_text=args.request,
        repo_path=str(Path(args.repo).resolve()),
        target_branch=args.branch,
    )
    apply_strict_job_constraints(spec)
    record = store.create(spec)
    record = daemon.run_once(record.job_id)
    print(record.model_dump_json(indent=2))
    return 0 if record.status.value == "done" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

