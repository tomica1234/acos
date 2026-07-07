"""Quality gate enforcement."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from packages.schemas.agent_outputs import FilePatch, ReviewResult, SecurityReviewResult, TestRunResult
from packages.schemas.models import ReviewDecision
from packages.schemas.tasks import PlannedTask


class QualityGateError(Exception):
    """Raised when a quality gate fails."""


def ensure_reviews_pass(
    review: ReviewResult, security_review: SecurityReviewResult
) -> None:
    if review.decision != ReviewDecision.APPROVE:
        raise QualityGateError("Reviewer did not approve the change set")
    if security_review.decision != ReviewDecision.APPROVE:
        raise QualityGateError("Security review did not approve the change set")


def ensure_fixer_safe(patches: list[FilePatch]) -> None:
    ensure_test_patch_quality(patches, role="fixer")


def ensure_test_patch_quality(patches: list[FilePatch], *, role: str) -> None:
    suspicious_tokens = (
        "xfail",
        "skip(",
        "skipif(",
        "mark.skip",
        "mark.xfail",
        "assert True",
    )
    for patch in patches:
        if patch.path.startswith("tests/") and any(token in patch.content for token in suspicious_tokens):
            raise QualityGateError(f"{role} attempted to weaken tests")


def ensure_tests_passed(result: TestRunResult) -> None:
    if not result.success:
        raise QualityGateError("Tests did not pass")


def ensure_task_target_files_exist(
    task: PlannedTask | None,
    *,
    workspace_root: str | Path,
) -> None:
    if task is None or not task.target_files:
        return
    ensure_required_artifacts_exist(
        task.target_files,
        workspace_root=workspace_root,
        label=f"task {task.id} target_files",
    )


def ensure_task_required_artifacts_exist(
    task: PlannedTask | None,
    *,
    workspace_root: str | Path,
) -> None:
    if task is None or not task.required_artifacts:
        return
    ensure_required_artifacts_exist(
        task.required_artifacts,
        workspace_root=workspace_root,
        label=f"task {task.id} required_artifacts",
    )


def ensure_required_artifacts_assigned_to_tasks(
    tasks: Sequence[PlannedTask],
    required_artifacts: Iterable[str],
    *,
    label: str = "required_artifacts",
) -> None:
    normalized_required, invalid_required = _normalize_artifact_paths(required_artifacts)
    if invalid_required:
        raise QualityGateError(
            f"{label} are incomplete; invalid required_artifacts: "
            + ", ".join(sorted(invalid_required))
        )
    if not normalized_required:
        return
    assigned: set[str] = set()
    invalid_task_artifacts: list[str] = []
    for task in tasks:
        declared, invalid = _normalize_artifact_paths(
            [*task.required_artifacts, *task.target_files]
        )
        assigned.update(declared)
        invalid_task_artifacts.extend(invalid)
    problems: list[str] = []
    if invalid_task_artifacts:
        problems.append(
            "invalid task artifact declarations: "
            + ", ".join(sorted(set(invalid_task_artifacts)))
        )
    missing = sorted(normalized_required - assigned)
    if missing:
        problems.append("missing task assignments: " + ", ".join(missing))
    if problems:
        raise QualityGateError(f"{label} are not fully assigned; " + "; ".join(problems))


def ensure_required_artifacts_exist(
    required_artifacts: Iterable[str],
    *,
    workspace_root: str | Path,
    label: str = "required_artifacts",
) -> None:
    workspace = Path(workspace_root).resolve()
    normalized_required, invalid = _normalize_artifact_paths(required_artifacts)
    missing: list[str] = []
    non_files: list[str] = []
    for artifact_path in normalized_required:
        normalized = PurePosixPath(artifact_path)
        target = (workspace / Path(*normalized.parts)).resolve()
        if workspace not in [target, *target.parents]:
            invalid.append(artifact_path)
            continue
        if not target.exists():
            missing.append(normalized.as_posix())
            continue
        if not target.is_file():
            non_files.append(normalized.as_posix())
    problems: list[str] = []
    if invalid:
        problems.append(f"invalid target_files: {', '.join(sorted(invalid))}")
    if missing:
        problems.append(f"missing target_files: {', '.join(sorted(missing))}")
    if non_files:
        problems.append(f"non-file entries: {', '.join(sorted(non_files))}")
    if problems:
        raise QualityGateError(f"{label} are incomplete; " + "; ".join(problems))


def _normalize_artifact_paths(paths: Iterable[str]) -> tuple[set[str], list[str]]:
    normalized_paths: set[str] = set()
    invalid_paths: list[str] = []
    for raw_path in paths:
        normalized = PurePosixPath(str(raw_path).replace("\\", "/").strip())
        if (
            normalized.is_absolute()
            or not normalized.parts
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            invalid_paths.append(str(raw_path))
            continue
        normalized_paths.add(normalized.as_posix())
    return normalized_paths, invalid_paths
