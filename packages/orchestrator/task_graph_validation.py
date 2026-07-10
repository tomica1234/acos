"""Shared task graph validation context keys."""

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

__all__ = [
    "TASK_GRAPH_VALIDATION_CONTEXT_KEYS",
    "TASK_GRAPH_VALIDATION_DETAIL_KEYS",
]
