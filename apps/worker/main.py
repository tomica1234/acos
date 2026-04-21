"""Worker process for ACOS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from packages.orchestrator.job_runner import build_default_runner
from packages.orchestrator.worker_daemon import WorkerDaemon
from packages.schemas.jobs import JobSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acos-worker")
    parser.add_argument("action", nargs="?", choices=["run", "recover"], default="run")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--request")
    parser.add_argument("--branch", default="acos/default")
    parser.add_argument("--file")
    parser.add_argument("--forever", action="store_true")
    return parser


def _build_daemon(config_dir: str | Path, repo: str | Path) -> WorkerDaemon:
    runner, _ = build_default_runner(config_dir=config_dir, workspace_root=Path(repo))
    if runner.runtime_manager is None:
        raise RuntimeError("runtime manager is not configured")
    return WorkerDaemon.from_path(
        Path(config_dir) / "worker.yaml",
        runner=runner,
        store=runner.store,
        runtime_manager=runner.runtime_manager,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner, _ = build_default_runner(config_dir=args.config_dir, workspace_root=Path(args.repo))

    if args.file:
        payload = yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}
        spec = JobSpec.model_validate(
            {
                "request_text": payload.get("request_text") or payload.get("requester_input"),
                "repo_path": str(Path(payload.get("repo_path", args.repo)).resolve()),
                "workspace_root": str(Path(payload.get("workspace_root", payload.get("repo_path", args.repo))).resolve()),
                "target_branch": payload.get("target_branch", args.branch),
                "metadata": payload.get("metadata", {}),
            }
        )
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1

    if args.request:
        spec = JobSpec(
            request_text=args.request,
            repo_path=str(Path(args.repo).resolve()),
            target_branch=args.branch,
        )
        record = runner.run_job(spec)
        print(record.model_dump_json(indent=2))
        return 0 if record.status.value == "done" else 1

    daemon = _build_daemon(args.config_dir, args.repo)
    if args.action == "recover":
        recovered = daemon.recover_stale_jobs()
        print(
            yaml.safe_dump(
                {
                    "recovered_jobs": [item.model_dump(mode="json") for item in recovered],
                },
                sort_keys=False,
                allow_unicode=True,
            )
        )
        return 0

    if args.forever:
        daemon.run_forever()
        return 0

    processed = daemon.run_once()
    print(
        yaml.safe_dump(
            {
                "processed_jobs": [item.model_dump(mode="json") for item in processed],
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
