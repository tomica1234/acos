"""Worker process for ACOS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from packages.orchestrator.job_runner import build_default_runner
from packages.schemas.jobs import JobSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acos-worker")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--request", required=True)
    parser.add_argument("--branch", default="acos/default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner, _ = build_default_runner(config_dir=args.config_dir, workspace_root=Path(args.repo))
    spec = JobSpec(
        request_text=args.request,
        repo_path=str(Path(args.repo).resolve()),
        target_branch=args.branch,
    )
    record = runner.run_job(spec)
    print(record.model_dump_json(indent=2))
    return 0 if record.status.value == "done" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

