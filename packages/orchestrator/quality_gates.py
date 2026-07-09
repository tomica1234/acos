"""Quality gate enforcement."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import re
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
    for patch in patches:
        if _looks_like_test_path(patch.path) and _test_patch_is_suspicious(patch):
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


def artifact_path_exists(
    relative_path: str,
    *,
    workspace_root: str | Path,
) -> bool:
    target = _resolve_valid_artifact_path(relative_path, workspace_root=workspace_root)
    return target is not None and target.is_file()


def _resolve_valid_artifact_path(
    relative_path: str,
    *,
    workspace_root: str | Path,
) -> Path | None:
    normalized_paths, invalid = _normalize_artifact_paths([relative_path])
    if invalid or not normalized_paths:
        return None
    normalized = PurePosixPath(next(iter(normalized_paths)))
    workspace = Path(workspace_root).resolve()
    target = (workspace / Path(*normalized.parts)).resolve()
    if workspace not in [target, *target.parents]:
        return None
    return target


def _normalize_artifact_paths(paths: Iterable[str]) -> tuple[set[str], list[str]]:
    normalized_paths: set[str] = set()
    invalid_paths: list[str] = []
    for raw_path in paths:
        value = str(raw_path).replace("\\", "/").strip()
        normalized = PurePosixPath(value)
        if (
            not value
            or value.endswith("/")
            or (len(value) >= 3 and value[1:3] == ":/" and value[0].isalpha())
            or normalized.is_absolute()
            or not normalized.parts
            or any(part in {"", ".", ".."} for part in normalized.parts)
        ):
            invalid_paths.append(str(raw_path))
            continue
        normalized_paths.add(normalized.as_posix())
    return normalized_paths, invalid_paths


def valid_artifact_paths(paths: Iterable[str]) -> set[str]:
    normalized, _invalid = _normalize_artifact_paths(paths)
    return normalized


def invalid_artifact_paths(paths: Iterable[str]) -> list[str]:
    _normalized, invalid = _normalize_artifact_paths(paths)
    return invalid


def _looks_like_test_path(path: str) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{normalized}"
        or "/test/" in f"/{normalized}"
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
    )


def _test_patch_is_suspicious(patch: FilePatch) -> bool:
    if _test_patch_removes_test_coverage(patch):
        return True
    if _test_patch_removes_assertions_without_replacement(patch):
        return True
    payload = _test_patch_payload(patch)
    if patch.operation in {"create", "update"} and not payload.strip():
        return True
    if not payload:
        return False
    compact = re.sub(r"\s+", "", payload).lower()
    vacuous_assertions = (
        "expect(true).tobe(true)",
        "expect(true).tobetruthy()",
        "expect(false).tobe(false)",
        "expect(false).tobefalsy()",
        "expect(1).tobe(1)",
        "expect(0).tobe(0)",
    )
    if any(snippet in compact for snippet in vacuous_assertions):
        return True
    if _python_payload_has_vacuous_assertion(payload):
        return True
    if _python_test_has_empty_body(payload):
        return True
    if _javascript_test_has_empty_body(compact):
        return True

    suspicious_patterns = (
        r"\bassert\s+True\b",
        r"\bpytest\s*\.\s*(?:skip|xfail|skipif)\s*\(",
        r"\bskipif\s*\(",
        r"\bxfail\s*\(",
        r"\bmark\s*\.\s*(?:skip|xfail)\b",
        r"\b(?:describe|it|test)\s*\.\s*(?:skip|only)\s*\(",
        r"\.\s*(?:skip|only)\s*\(",
    )
    return any(re.search(pattern, payload) for pattern in suspicious_patterns)


def _test_patch_removes_test_coverage(patch: FilePatch) -> bool:
    operation = patch.operation
    if operation == "delete":
        return _looks_like_test_path(patch.path)
    if operation != "rename":
        return False
    source_is_test = _looks_like_test_path(patch.path)
    target_is_test = bool(
        patch.new_path and _looks_like_active_test_location(patch.new_path)
    )
    return source_is_test and not target_is_test


def _test_patch_removes_assertions_without_replacement(patch: FilePatch) -> bool:
    if patch.operation != "update" or patch.unified_diff is None:
        return False
    removed = _test_patch_diff_lines(patch.unified_diff, "-")
    if not any(_line_has_test_assertion(line) for line in removed):
        return False
    added = _test_patch_diff_lines(patch.unified_diff, "+")
    return not any(_line_has_test_assertion(line) for line in added)


def _test_patch_diff_lines(unified_diff: str, prefix: str) -> list[str]:
    lines: list[str] = []
    header_prefix = "---" if prefix == "-" else "+++"
    for line in unified_diff.splitlines():
        if line.startswith(header_prefix) or not line.startswith(prefix):
            continue
        lines.append(line[1:])
    return lines


def _line_has_test_assertion(line: str) -> bool:
    return any(
        re.search(pattern, line)
        for pattern in (
            r"\bassert\b",
            r"\bexpect\s*\(",
            r"\bpytest\s*\.\s*raises\s*\(",
            r"\.\s*assert[A-Za-z_]*\s*\(",
        )
    )


def _python_payload_has_vacuous_assertion(payload: str) -> bool:
    try:
        tree = ast.parse(payload)
    except SyntaxError:
        return any(
            _python_assert_line_is_vacuous(line)
            for line in payload.splitlines()
        )
    return any(
        isinstance(node, ast.Assert) and _python_assert_expr_is_vacuous(node.test)
        for node in ast.walk(tree)
    )


def _python_assert_line_is_vacuous(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("assert "):
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    statements = tree.body
    if len(statements) != 1 or not isinstance(statements[0], ast.Assert):
        return False
    return _python_assert_expr_is_vacuous(statements[0].test)


def _python_assert_expr_is_vacuous(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Constant):
        return bool(expression.value) is True
    if not isinstance(expression, ast.Compare) or len(expression.ops) != 1:
        return False
    left = _literal_value(expression.left)
    right = _literal_value(expression.comparators[0])
    if left is _MISSING_LITERAL or right is _MISSING_LITERAL:
        return False
    op = expression.ops[0]
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Is):
        return left == right
    if isinstance(op, ast.IsNot):
        return left != right
    try:
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
    except TypeError:
        return False
    return False


_MISSING_LITERAL = object()


def _literal_value(expression: ast.expr) -> object:
    try:
        return ast.literal_eval(expression)
    except (ValueError, TypeError):
        return _MISSING_LITERAL


def _looks_like_active_test_location(path: str) -> bool:
    normalized = str(path).replace("\\", "/").lower().lstrip("./")
    name = normalized.rsplit("/", 1)[-1]
    if "/tests/" in f"/{normalized}" or "/test/" in f"/{normalized}":
        return True
    if "/" not in normalized and name.startswith("test_") and name.endswith(".py"):
        return True
    if normalized.startswith(("docs/", "doc/")):
        return False
    return ".test." in name or ".spec." in name


def _python_test_has_empty_body(payload: str) -> bool:
    try:
        tree = ast.parse(payload)
    except SyntaxError:
        return bool(
            re.search(
                r"(?ms)^\s*def\s+test_[A-Za-z0-9_]+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*pass\s*$",
                payload,
            )
        )
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        body = [
            statement
            for statement in node.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        if len(body) == 1 and _is_empty_test_statement(body[0]):
            return True
    return False


def _is_empty_test_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        )
    )


def _javascript_test_has_empty_body(compact_payload: str) -> bool:
    empty_test_patterns = (
        r"\b(?:describe|it|test)\([^;{}]*(?:async)?\(\)=>\{\}\)",
        r"\b(?:describe|it|test)\([^;{}]*function\(\)\{\}\)",
    )
    return any(re.search(pattern, compact_payload) for pattern in empty_test_patterns)


def _test_patch_payload(patch: FilePatch) -> str:
    if patch.content is not None:
        return patch.content
    if patch.unified_diff is None:
        return ""
    added_lines: list[str] = []
    for line in patch.unified_diff.splitlines():
        if line.startswith("+++") or not line.startswith("+"):
            continue
        added_lines.append(line[1:])
    return "\n".join(added_lines)
