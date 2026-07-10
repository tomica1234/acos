"""Shared task graph validation helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

TASK_GRAPH_VALIDATION_DETAIL_KEYS = (
    "unassigned_required_artifacts",
    "invalid_prd_required_artifacts",
    "unowned_required_artifacts",
    "uncovered_test_writer_acceptance_tests",
    "role_mismatched_target_files",
    "role_mismatched_required_artifacts",
    "required_artifacts_missing_target_files",
    "target_files_missing_required_artifacts",
    "unordered_target_file_owner_conflicts",
    "duplicate_task_ids",
    "unknown_dependencies",
    "dependency_cycle_task_ids",
    "prd_test_required_artifacts",
    "missing_test_writer_task_requirements",
    "executable_tasks_missing_required_artifacts",
    "test_writer_tasks_missing_acceptance_criteria",
    "duplicate_task_acceptance_criteria",
    "generic_task_acceptance_criteria",
    "implementation_tasks_missing_target_files",
    "test_writer_tasks_missing_target_files",
    "test_writer_missing_implementation_dependencies",
    "test_writer_dependency_semantic_mismatches",
    "test_writer_acceptance_dependency_mismatches",
    "executor_order_dependency_violations",
    "invalid_task_titles",
    "invalid_task_descriptions",
    "invalid_task_ids",
    "invalid_task_artifacts",
    "ignored_project_setup_artifacts",
    "unsupported_task_roles",
)

TASK_GRAPH_VALIDATION_CONTEXT_KEYS = (
    "uncovered_small_parts",
    "uncovered_acceptance_tests",
    *TASK_GRAPH_VALIDATION_DETAIL_KEYS,
)

TASK_GRAPH_VALIDATION_CONTRACT_KEYS = (
    "task_count",
    "implementation_task_count",
    "test_writer_task_count",
    "executable_task_count",
    "task_ids",
    "implementation_task_ids",
    "test_writer_task_ids",
    "executable_task_ids",
    "task_graph_fingerprint",
    "prd_fingerprint",
    "require_acceptance_criteria",
    "require_task_artifacts",
    "require_executable_task_roles",
)

_TASK_GRAPH_FINGERPRINT_FIELDS = (
    "id",
    "title",
    "description",
    "role",
    "depends_on",
    "acceptance_criteria",
    "target_files",
    "required_artifacts",
)
_PRD_VALIDATION_FINGERPRINT_FIELDS = (
    "small_parts",
    "acceptance_tests",
    "required_artifacts",
)
_PRD_QUALITY_FINGERPRINT_FIELDS = (
    "title",
    "problem_statement",
    "smallest_working_core",
    "small_parts",
    "incremental_milestones",
    "acceptance_tests",
    "definition_of_done",
    "required_artifacts",
    "open_questions",
)
_MISSING = object()


def task_graph_validation_fingerprint(
    tasks: Iterable[Mapping[str, Any]],
) -> str:
    """Return a stable fingerprint for task fields covered by validation."""
    canonical_tasks = [
        {
            field: _fingerprint_value(task.get(field))
            for field in _TASK_GRAPH_FINGERPRINT_FIELDS
        }
        for task in tasks
    ]
    payload = json.dumps(
        canonical_tasks,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prd_validation_fingerprint(prd: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for PRD fields covered by task validation."""
    payload = json.dumps(
        {
            field: _fingerprint_value(prd.get(field))
            for field in _PRD_VALIDATION_FINGERPRINT_FIELDS
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prd_quality_fingerprint(prd: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for PRD fields covered by PRD quality gates."""
    payload = json.dumps(
        {
            field: _fingerprint_value(prd.get(field))
            for field in _PRD_QUALITY_FINGERPRINT_FIELDS
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, tuple):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING and (
        isinstance(enum_value, (str, int, float, bool)) or enum_value is None
    ):
        return enum_value
    return str(value)


__all__ = [
    "TASK_GRAPH_VALIDATION_CONTRACT_KEYS",
    "TASK_GRAPH_VALIDATION_CONTEXT_KEYS",
    "TASK_GRAPH_VALIDATION_DETAIL_KEYS",
    "prd_quality_fingerprint",
    "prd_validation_fingerprint",
    "task_graph_validation_fingerprint",
]
