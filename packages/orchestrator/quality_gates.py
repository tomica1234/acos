"""Quality gate enforcement."""

from __future__ import annotations

import ast
import difflib
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


def ensure_fixer_safe(
    patches: list[FilePatch],
    *,
    workspace_root: str | Path | None = None,
) -> None:
    ensure_test_patch_quality(patches, role="fixer", workspace_root=workspace_root)


def ensure_test_patch_quality(
    patches: list[FilePatch],
    *,
    role: str,
    workspace_root: str | Path | None = None,
) -> None:
    for patch in patches:
        if workspace_root is not None:
            patch = _patch_with_workspace_diff(patch, workspace_root=workspace_root)
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
    if _test_patch_introduces_test_case_without_assertion(payload):
        return True
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
    if _javascript_payload_has_vacuous_expectation(payload):
        return True
    if _python_payload_has_vacuous_assertion(payload):
        return True
    if _python_test_has_empty_body(payload):
        return True
    if _javascript_test_has_empty_body(compact):
        return True

    suspicious_patterns = (
        r"\bassert\s+True\b",
        r"\bpytest\s*\.\s*(?:skip|xfail|skipif|importorskip)\s*\(",
        r"\bunittest\s*\.\s*skip(?:If|Unless)?\s*\(",
        r"\bself\s*\.\s*skipTest\s*\(",
        r"\bskipif\s*\(",
        r"\bxfail\s*\(",
        r"\bmark\s*\.\s*(?:skip|xfail)\b",
        r"\b(?:describe|it|test)(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*\.\s*(?:skip|only|todo)\s*\(",
        r"\.\s*(?:skip|only|todo)\s*\.",
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


def _patch_with_workspace_diff(
    patch: FilePatch,
    *,
    workspace_root: str | Path,
) -> FilePatch:
    if (
        patch.operation != "update"
        or patch.unified_diff is not None
        or patch.content is None
    ):
        return patch
    target = _resolve_valid_artifact_path(patch.path, workspace_root=workspace_root)
    if target is None or not target.is_file():
        return patch
    try:
        original = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return patch
    unified_diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            patch.content.splitlines(),
            fromfile=patch.path,
            tofile=patch.path,
            lineterm="",
        )
    )
    return patch.model_copy(update={"unified_diff": unified_diff})


def _test_patch_removes_assertions_without_replacement(patch: FilePatch) -> bool:
    if patch.operation != "update" or patch.unified_diff is None:
        return False
    for hunk in _test_patch_diff_hunks(patch.unified_diff):
        if _hunk_removes_assertions_without_replacement(hunk):
            return True
    return False


def _test_patch_diff_hunks(unified_diff: str) -> list[list[tuple[str, str]]]:
    hunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] | None = None
    for line in unified_diff.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = []
            continue
        if current is None:
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith(("-", "+", " ")):
            current.append((line[:1], line[1:]))
    if current is not None:
        hunks.append(current)
    return hunks


def _hunk_removes_assertions_without_replacement(
    hunk: list[tuple[str, str]],
) -> bool:
    removed = [line for marker, line in hunk if marker == "-"]
    if not any(_line_has_test_assertion(line) for line in removed):
        return False
    added = [line for marker, line in hunk if marker == "+"]
    if not any(_line_has_test_assertion(line) for line in added):
        return True

    removed_scopes: list[str | None] = []
    added_scopes: set[str | None] = set()
    unscoped_removed_blocks: set[int] = set()
    unscoped_added_blocks: set[int] = set()
    current_removed_scope: str | None = None
    current_added_scope: str | None = None
    current_change_block = -1
    in_change_block = False
    for marker, line in hunk:
        if marker == " ":
            in_change_block = False
        elif not in_change_block:
            current_change_block += 1
            in_change_block = True
        scope = _test_case_scope_key(line)
        if scope is not None and marker != "+":
            current_removed_scope = scope
        if scope is not None and marker != "-":
            current_added_scope = scope
        if marker == "-" and _line_has_test_assertion(line):
            if current_removed_scope is None:
                unscoped_removed_blocks.add(current_change_block)
            else:
                removed_scopes.append(current_removed_scope)
        elif marker == "+" and _line_has_test_assertion(line):
            if current_added_scope is None:
                unscoped_added_blocks.add(current_change_block)
            else:
                added_scopes.add(current_added_scope)

    if any(scope not in added_scopes for scope in removed_scopes):
        return True
    return any(block not in unscoped_added_blocks for block in unscoped_removed_blocks)


def _test_case_scope_key(line: str) -> str | None:
    python_match = re.match(
        r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)\s*\(",
        line,
    )
    if python_match:
        return f"python:{python_match.group(1)}"
    javascript_match = re.match(
        r"^\s*(?:describe|it|test)(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*"
        r"\(\s*(['\"`])(?P<name>.*?)\1",
        line,
    )
    if javascript_match:
        return f"javascript:{javascript_match.group('name')}"
    return None


def _test_patch_introduces_test_case_without_assertion(payload: str) -> bool:
    python_result = _python_payload_has_test_case_without_assertion(payload)
    if python_result is not None:
        return python_result
    if _javascript_payload_has_test_case_without_assertion(payload):
        return True
    if not _payload_introduces_test_case(payload):
        return False
    return not any(_line_has_test_assertion(line) for line in payload.splitlines())


def _python_payload_has_test_case_without_assertion(payload: str) -> bool | None:
    try:
        tree = ast.parse(payload)
    except SyntaxError:
        return None
    test_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if not test_functions:
        return None
    return any(not _python_test_function_has_assertion(node) for node in test_functions)


def _python_test_function_has_assertion(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for child in ast.walk(node):
        if child is node:
            continue
        if isinstance(child, ast.Assert):
            return True
        if _python_call_is_assertion(child):
            return True
    return False


def _python_call_is_assertion(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Attribute):
        if function.attr.startswith("assert"):
            return True
        return (
            function.attr == "raises"
            and isinstance(function.value, ast.Name)
            and function.value.id == "pytest"
        )
    return False


def _javascript_payload_has_test_case_without_assertion(payload: str) -> bool:
    test_blocks = (
        r"\b(?:it|test)(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*"
        r"\([^,{]*,\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>\s*\{"
        r"(?P<body>.*?)"
        r"^\s*\}\s*\)",
        r"\b(?:it|test)(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*"
        r"\([^,{]*,\s*(?:async\s*)?function(?:\s+[A-Za-z_$][\w$]*)?\s*\([^)]*\)\s*\{"
        r"(?P<body>.*?)"
        r"^\s*\}\s*\)",
    )
    for pattern in test_blocks:
        test_block = re.compile(pattern, re.MULTILINE | re.DOTALL)
        for match in test_block.finditer(payload):
            body = match.group("body")
            if not any(_line_has_test_assertion(line) for line in body.splitlines()):
                return True
    return False


def _payload_introduces_test_case(payload: str) -> bool:
    test_case_patterns = (
        r"(?m)^\s*(?:async\s+)?def\s+test_[A-Za-z0-9_]+\s*\(",
        r"\b(?:describe|it|test)(?:\s*\.\s*[A-Za-z_$][\w$]*)*\s*\(",
    )
    return any(re.search(pattern, payload) for pattern in test_case_patterns)


def _line_has_test_assertion(line: str) -> bool:
    stripped = _mask_quoted_segments(line.strip())
    stripped = _line_code_before_comment(stripped)
    if not stripped:
        return False
    return any(
        re.search(pattern, stripped)
        for pattern in (
            r"\bassert\b",
            r"\bexpect\s*\(",
            r"\bpytest\s*\.\s*raises\s*\(",
            r"\.\s*assert[A-Za-z_]*\s*\(",
        )
    )


def _line_code_before_comment(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith(("#", "//", "/*", "*")):
        return ""
    comment_start: int | None = None
    for marker in ("#", "//", "/*"):
        search_from = 0
        while True:
            index = stripped.find(marker, search_from)
            if index == -1:
                break
            if index == 0 or stripped[index - 1].isspace():
                comment_start = (
                    index if comment_start is None else min(comment_start, index)
                )
                break
            search_from = index + len(marker)
    if comment_start is None:
        return stripped
    return stripped[:comment_start].rstrip()


def _mask_quoted_segments(line: str) -> str:
    output: list[str] = []
    quote: str | None = None
    escaped = False
    for character in line:
        if quote is not None:
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if character == quote:
                quote = None
                output.append('""')
            continue
        if character in {"'", '"', "`"}:
            quote = character
            continue
        output.append(character)
    return "".join(output)


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
    if not isinstance(expression, ast.Compare):
        return False
    values = [_literal_value(expression.left)]
    values.extend(_literal_value(comparator) for comparator in expression.comparators)
    if any(value is _MISSING_LITERAL for value in values):
        return False
    return _python_literal_compare_chain_is_true(values, expression.ops)


def _python_literal_compare_chain_is_true(
    values: list[object],
    operators: list[ast.cmpop],
) -> bool:
    try:
        return all(
            _python_literal_comparison_is_true(left, operator, right)
            for left, operator, right in zip(values, operators, values[1:])
        )
    except TypeError:
        return False


def _python_literal_comparison_is_true(
    left: object,
    op: ast.cmpop,
    right: object,
) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Is):
        return left == right
    if isinstance(op, ast.IsNot):
        return left != right
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
    return False


_MISSING_LITERAL = object()
_JS_UNDEFINED = object()


def _literal_value(expression: ast.expr) -> object:
    try:
        return ast.literal_eval(expression)
    except (ValueError, TypeError):
        return _MISSING_LITERAL


_JS_LITERAL = (
    r"(?:true|false|null|undefined|-?\d+(?:\.\d+)?|"
    r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)"
)


def _javascript_payload_has_vacuous_expectation(payload: str) -> bool:
    expectation_with_argument = re.compile(
        rf"\bexpect\s*\(\s*(?P<actual>{_JS_LITERAL})\s*\)"
        rf"\s*\.\s*(?:(?P<negated>not)\s*\.\s*)?"
        rf"(?P<matcher>"
        rf"toBeGreaterThanOrEqual|toBeLessThanOrEqual|toBeGreaterThan|"
        rf"toBeLessThan|toBe|toEqual|toStrictEqual|toContain"
        rf")\s*"
        rf"\(\s*(?P<expected>{_JS_LITERAL})\s*\)",
        re.MULTILINE,
    )
    expectation_without_argument = re.compile(
        rf"\bexpect\s*\(\s*(?P<actual>{_JS_LITERAL})\s*\)"
        rf"\s*\.\s*(?:(?P<negated>not)\s*\.\s*)?"
        rf"(?P<matcher>"
        rf"toBeTruthy|toBeFalsy|toBeNull|toBeUndefined|toBeDefined"
        rf")\s*\(\s*\)",
        re.MULTILINE,
    )
    for match in expectation_with_argument.finditer(payload):
        actual = _javascript_literal_value(match.group("actual"))
        expected = _javascript_literal_value(match.group("expected"))
        if actual is _MISSING_LITERAL or expected is _MISSING_LITERAL:
            continue
        passes = _javascript_matcher_passes(
            actual,
            match.group("matcher"),
            expected,
        )
        if passes is None:
            continue
        if bool(match.group("negated")):
            passes = not passes
        if passes:
            return True
    for match in expectation_without_argument.finditer(payload):
        actual = _javascript_literal_value(match.group("actual"))
        if actual is _MISSING_LITERAL:
            continue
        passes = _javascript_matcher_passes(actual, match.group("matcher"))
        if passes is None:
            continue
        if bool(match.group("negated")):
            passes = not passes
        if passes:
            return True
    return False


def _javascript_matcher_passes(
    actual: object,
    matcher: str,
    expected: object = _MISSING_LITERAL,
) -> bool | None:
    if matcher in {"toBe", "toEqual", "toStrictEqual"}:
        return actual == expected
    if matcher == "toContain":
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        return None
    if matcher == "toBeTruthy":
        return _javascript_truthy(actual)
    if matcher == "toBeFalsy":
        return not _javascript_truthy(actual)
    if matcher == "toBeNull":
        return actual is None
    if matcher == "toBeUndefined":
        return actual is _JS_UNDEFINED
    if matcher == "toBeDefined":
        return actual is not _JS_UNDEFINED
    try:
        if matcher == "toBeGreaterThan":
            return actual > expected
        if matcher == "toBeGreaterThanOrEqual":
            return actual >= expected
        if matcher == "toBeLessThan":
            return actual < expected
        if matcher == "toBeLessThanOrEqual":
            return actual <= expected
    except TypeError:
        return None
    return None


def _javascript_truthy(value: object) -> bool:
    if value is None or value is _JS_UNDEFINED:
        return False
    if value is False:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def _javascript_literal_value(raw_literal: str) -> object:
    raw = raw_literal.strip()
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if lowered == "undefined":
        return _JS_UNDEFINED
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return float(raw) if "." in raw else int(raw)
    if len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return _MISSING_LITERAL
    if len(raw) >= 2 and raw.startswith("`") and raw.endswith("`"):
        body = raw[1:-1]
        if "${" in body:
            return _MISSING_LITERAL
        return body
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
