"""Progress summaries for long-running ACOS jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from packages.orchestrator.job_constraints import STRICT_JOB_CONSTRAINTS
from packages.orchestrator.quality_gates import (
    invalid_planning_artifact_paths,
    looks_like_placeholder_planning_item,
    valid_planning_artifact_paths,
)
from packages.orchestrator.task_graph_validation import (
    TASK_GRAPH_VALIDATION_CONTRACT_KEYS,
    prd_quality_fingerprint,
    prd_validation_fingerprint,
    task_graph_validation_fingerprint,
)
from packages.schemas.jobs import JobRecord

_PROJECT_SETUP_REQUIRED_ARTIFACTS = {
    "backend/main.py",
    "backend/requirements.txt",
    "backend/tests/test_project_setup.py",
    "frontend/package.json",
    "frontend/vite.config.js",
    "frontend/src/main.tsx",
    "frontend/src/App.tsx",
    "shared/.gitkeep",
    ".gitignore",
    "package.json",
    "README.md",
    ".env.example",
}


def summarize_job_progress(record: JobRecord) -> dict[str, Any]:
    """Return a compact, machine-readable progress summary for a job."""
    done = _is_done(record)
    planned_tasks = _planned_tasks(record)
    completed_task_ids = list(record.completed_task_ids)
    planned_ids = [task["id"] for task in planned_tasks if isinstance(task.get("id"), str)]
    planned_id_set = set(planned_ids)
    pending_ids = [task_id for task_id in planned_ids if task_id not in completed_task_ids]
    last_stage = _last_stage(record)
    raw_failed_stage = _last_failed_stage(record)
    stage_statuses = _stage_statuses(record)
    recovery_history = _recovery_history(stage_statuses)
    change_summary = _change_summary(record)
    planning_quality = _planning_quality(record)
    autonomy_readiness = _autonomy_readiness(record, planned_tasks, planning_quality)
    planning_summary = _planning_summary(
        record,
        planned_tasks,
        planning_quality,
        autonomy_readiness,
    )
    execution_limits = _execution_limits(record)
    completion_integrity = _completion_integrity(record)
    active_recovery_context = _has_active_recovery_context(record)
    failed_stage = _active_failed_stage(
        record,
        raw_failed_stage,
        active_recovery_context=active_recovery_context,
    )
    failure_analysis = _failure_analysis(record, failed_stage, recovery_history)
    failure_diagnosis = (
        _failure_diagnosis(record) if active_recovery_context else None
    )
    model_metrics = _model_metrics(record)
    active_model_call = _active_model_call(record)
    recovery_plan = _active_recovery_plan(record)
    current_recovery_event = (
        record.runtime_state.get("current_recovery_event")
        if active_recovery_context
        else None
    )
    if not isinstance(current_recovery_event, dict):
        current_recovery_event = None
    last_recoverable_error = _last_recoverable_error(record)
    total_tasks = len(planned_ids)
    completed_count = len([task_id for task_id in completed_task_ids if task_id in planned_id_set])
    progress_ratio = completed_count / total_tasks if total_tasks else 0.0
    next_task = next((task for task in planned_tasks if task.get("id") in pending_ids), None)
    resume = _resume_recommendation(
        record,
        pending_ids,
        failed_stage,
        execution_limits,
        failure_analysis,
        autonomy_readiness,
    )
    payload = {
        "job_id": record.job_id,
        "status": record.status.value,
        "total_tasks": total_tasks,
        "completed_task_count": completed_count,
        "pending_task_count": len(pending_ids),
        "progress_ratio": round(progress_ratio, 4),
        "completed_task_ids": completed_task_ids,
        "pending_task_ids": pending_ids,
        "next_task": next_task,
        "checkpoint_count": len(record.checkpoints),
        "last_stage": last_stage,
        "failed_stage": failed_stage,
        "stage_statuses": stage_statuses,
        "successful_stage_task_ids": _stage_task_ids(stage_statuses, "passed"),
        "failed_stage_task_ids": (
            _stage_task_ids(stage_statuses, "failed")
            if failed_stage is not None
            else []
        ),
        "recovered_stage_task_ids": _stage_task_ids(stage_statuses, "superseded"),
        "recovery_history": recovery_history,
        "resume": resume,
        "planning_summary": planning_summary,
        "planning_quality": planning_quality,
        "autonomy_readiness": autonomy_readiness,
        "completion_integrity": completion_integrity,
        "model_metrics": model_metrics,
        "execution_limits": execution_limits,
        "failure_analysis": failure_analysis,
        "active_model_call": active_model_call,
        "recovery_plan": recovery_plan,
        "current_recovery_event": current_recovery_event,
        "last_recoverable_error": last_recoverable_error,
        "change_summary": change_summary,
        "last_error": record.last_error,
        "updated_at": record.updated_at.isoformat(),
    }
    if failure_diagnosis is not None:
        payload["failure_diagnosis"] = failure_diagnosis
    return payload


def _active_model_call(record: JobRecord) -> dict[str, Any] | None:
    role = record.runtime_state.get("active_role")
    if not isinstance(role, str) or not role:
        return None
    started_at = _datetime_metric(record.runtime_state.get("active_started_at"))
    elapsed_seconds: float | None = None
    if started_at is not None:
        elapsed_seconds = max(
            (datetime.now(timezone.utc) - started_at).total_seconds(),
            0.0,
        )
    timeout_seconds = _float_metric(record.runtime_state.get("active_model_timeout_seconds"))
    timeout_ratio: float | None = None
    if (
        elapsed_seconds is not None
        and timeout_seconds is not None
        and timeout_seconds > 0
    ):
        timeout_ratio = elapsed_seconds / timeout_seconds
    return {
        "role": role,
        "objective": record.runtime_state.get("active_objective"),
        "task_id": record.runtime_state.get("active_task_id"),
        "model": record.runtime_state.get("active_model"),
        "started_at": started_at.isoformat() if started_at is not None else None,
        "elapsed_seconds": (
            round(elapsed_seconds, 1) if elapsed_seconds is not None else None
        ),
        "timeout_seconds": timeout_seconds,
        "timeout_ratio": (
            round(timeout_ratio, 4) if timeout_ratio is not None else None
        ),
        "long_running": (
            elapsed_seconds is not None
            and (
                elapsed_seconds >= 300
                or (timeout_ratio is not None and timeout_ratio >= 0.5)
            )
        ),
    }


def _model_metrics(record: JobRecord) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    weighted_completion_tokens = 0
    weighted_duration_seconds = 0.0
    by_role: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

    for event in record.audit_events:
        payload = _audit_event_payload(event)
        if payload.get("event_type") != "model_call":
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        role = str(payload.get("role") or "unknown")
        model_key = str(metadata.get("model_key") or payload.get("action") or "unknown")
        prompt_tokens = _int_metric(
            metadata.get("prompt_tokens"),
            metadata.get("prompt_tokens_estimate"),
        )
        completion_tokens = _int_metric(
            metadata.get("completion_tokens"),
            metadata.get("completion_tokens_estimate"),
        )
        call_total_tokens = _int_metric(
            metadata.get("total_tokens"),
            metadata.get("total_tokens_estimate"),
        )
        if call_total_tokens == 0:
            call_total_tokens = prompt_tokens + completion_tokens
        duration_seconds = _float_metric(metadata.get("duration_seconds"))
        completion_tps = _float_metric(metadata.get("completion_tokens_per_second"))
        total_tps = _float_metric(metadata.get("total_tokens_per_second"))
        call = {
            "timestamp": payload.get("timestamp"),
            "role": role,
            "model_key": model_key,
            "provider_key": metadata.get("provider_key"),
            "status": payload.get("status"),
            "usage_source": metadata.get("usage_source") or "estimate",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": call_total_tokens,
            "duration_seconds": duration_seconds,
            "completion_tokens_per_second": completion_tps,
            "total_tokens_per_second": total_tps,
        }
        calls.append(call)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += call_total_tokens
        if duration_seconds is not None and duration_seconds > 0 and completion_tokens > 0:
            weighted_completion_tokens += completion_tokens
            weighted_duration_seconds += duration_seconds
        _add_model_metric_bucket(by_role, role, call)
        _add_model_metric_bucket(by_model, model_key, call)

    average_completion_tps = (
        weighted_completion_tokens / weighted_duration_seconds
        if weighted_duration_seconds > 0
        else None
    )
    latest_call = calls[-1] if calls else None
    return {
        "model_call_count": len(calls),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "latest_call": latest_call,
        "latest_completion_tps": (
            latest_call.get("completion_tokens_per_second") if latest_call else None
        ),
        "average_completion_tps": (
            round(average_completion_tps, 4) if average_completion_tps is not None else None
        ),
        "by_role": by_role,
        "by_model": by_model,
    }


def _audit_event_payload(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return event.model_dump(mode="json")
    if isinstance(event, dict):
        return event
    return {}


def _int_metric(*values: Any) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                continue
    return 0


def _float_metric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _datetime_metric(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_model_metric_bucket(
    buckets: dict[str, dict[str, Any]],
    key: str,
    call: dict[str, Any],
) -> None:
    bucket = buckets.setdefault(
        key,
        {
            "model_call_count": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "duration_seconds": 0.0,
            "average_completion_tps": None,
        },
    )
    bucket["model_call_count"] += 1
    bucket["total_prompt_tokens"] += call["prompt_tokens"]
    bucket["total_completion_tokens"] += call["completion_tokens"]
    bucket["total_tokens"] += call["total_tokens"]
    duration = call.get("duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        bucket["duration_seconds"] = round(bucket["duration_seconds"] + duration, 4)
        if bucket["total_completion_tokens"] > 0:
            bucket["average_completion_tps"] = round(
                bucket["total_completion_tokens"] / bucket["duration_seconds"],
                4,
            )


def _failure_diagnosis(record: JobRecord) -> dict[str, Any] | None:
    diagnosis = record.outputs.get("failure_diagnosis")
    if isinstance(diagnosis, dict):
        return diagnosis
    diagnoses = record.outputs.get("failure_diagnoses")
    if isinstance(diagnoses, list):
        for item in reversed(diagnoses):
            if isinstance(item, dict):
                return item
    return None


def _last_recoverable_error(record: JobRecord) -> str | None:
    if not _has_active_recovery_context(record):
        return None
    value = record.runtime_state.get("last_recoverable_error")
    if isinstance(value, str) and value:
        return value
    value = record.outputs.get("last_recoverable_error")
    if isinstance(value, str) and value:
        return value
    event = record.runtime_state.get("current_recovery_event")
    if isinstance(event, dict):
        for key in ("error", "reason"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _effective_failure_error(record: JobRecord) -> str | None:
    if _is_done(record):
        return None
    return record.last_error or _last_recoverable_error(record)


def _is_done(record: JobRecord) -> bool:
    return record.status.value == "done"


def _active_recovery_plan(record: JobRecord) -> dict[str, Any] | None:
    if _is_done(record):
        return None
    plan = record.runtime_state.get("recovery_plan")
    if not isinstance(plan, dict):
        return None
    if plan.get("status") == "completed" and plan.get("consumed_by_runner") is True:
        return None
    return plan


def _has_active_recovery_context(record: JobRecord) -> bool:
    if _is_done(record):
        return False
    if isinstance(record.last_error, str) and record.last_error:
        return True
    if _active_recovery_plan(record) is not None:
        return True
    return record.status.value in {"blocked", "failed", "stuck"}


def _active_failed_stage(
    record: JobRecord,
    failed_stage: dict[str, Any] | None,
    *,
    active_recovery_context: bool,
) -> dict[str, Any] | None:
    if failed_stage is None:
        return None
    if active_recovery_context:
        return failed_stage
    if record.status.value in {"testing", "fixing"}:
        return failed_stage
    return None


def _planned_tasks(record: JobRecord) -> list[dict[str, Any]]:
    task_graph = record.outputs.get("task_graph")
    if not isinstance(task_graph, dict):
        return []
    tasks = task_graph.get("tasks", [])
    return [task for task in tasks if isinstance(task, dict)]


def _last_stage(record: JobRecord) -> dict[str, Any] | None:
    stages = record.outputs.get("autonomous_stages", [])
    if not isinstance(stages, list) or not stages:
        return None
    stage = stages[-1]
    return stage if isinstance(stage, dict) else None


def _last_failed_stage(record: JobRecord) -> dict[str, Any] | None:
    stages = record.outputs.get("autonomous_stages", [])
    if not isinstance(stages, list):
        return None
    later_passed_task_ids: set[str] = set()
    for stage in reversed(stages):
        if not isinstance(stage, dict):
            continue
        task = stage.get("task")
        task_id = task.get("id") if isinstance(task, dict) else None
        test_run = stage.get("test_run")
        post_review_test_run = stage.get("post_review_test_run")
        test_success = test_run.get("success") if isinstance(test_run, dict) else None
        post_review_success = (
            post_review_test_run.get("success") if isinstance(post_review_test_run, dict) else None
        )
        status = _resolved_stage_status(stage, test_success, post_review_success)
        if status == "failed" and task_id not in later_passed_task_ids:
            return stage
        if status == "passed" and isinstance(task_id, str):
            later_passed_task_ids.add(task_id)
    return None


def _change_summary(record: JobRecord) -> dict[str, Any]:
    stage_summaries: list[dict[str, Any]] = []
    recovery_created_files = _recovery_created_files(record)
    changed_files: list[str] = list(recovery_created_files)
    patch_count = 0
    stages = record.outputs.get("autonomous_stages", [])
    if not isinstance(stages, list):
        return {
            "changed_files": changed_files,
            "patch_count": 0,
            "stages": [],
            "recovery_created_files": recovery_created_files,
        }
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        summary = stage.get("change_summary")
        if not isinstance(summary, dict):
            continue
        files = [path for path in summary.get("changed_files", []) if isinstance(path, str)]
        stage_patch_count = summary.get("patch_count", 0)
        if not isinstance(stage_patch_count, int):
            stage_patch_count = 0
        changed_files = _unique_paths([*changed_files, *files])
        patch_count += stage_patch_count
        task = stage.get("task")
        stage_summaries.append(
            {
                "stage": stage.get("stage"),
                "task_id": task.get("id") if isinstance(task, dict) else None,
                "changed_files": files,
                "patch_count": stage_patch_count,
            }
        )
    return {
        "changed_files": changed_files,
        "patch_count": patch_count,
        "stages": stage_summaries,
        "recovery_created_files": recovery_created_files,
    }


def _recovery_created_files(record: JobRecord) -> list[str]:
    paths: list[str] = []
    recovery_plan = record.runtime_state.get("recovery_plan")
    if isinstance(recovery_plan, dict):
        constraints = recovery_plan.get("constraints")
        if isinstance(constraints, dict):
            paths.extend(_string_list(constraints.get("deterministically_created_files")))
    metadata_constraints = record.spec.metadata.get("constraints")
    if isinstance(metadata_constraints, dict):
        paths.extend(
            _string_list(metadata_constraints.get("deterministically_created_files"))
        )
    deterministic_test_scaffolds = record.outputs.get("deterministic_test_scaffolds")
    if isinstance(deterministic_test_scaffolds, list):
        for item in deterministic_test_scaffolds:
            if isinstance(item, dict):
                path = item.get("path")
                if isinstance(path, str):
                    paths.append(path)
            elif isinstance(item, str):
                paths.append(item)
    return _unique_paths(paths)


def _stage_statuses(record: JobRecord) -> list[dict[str, Any]]:
    stages = record.outputs.get("autonomous_stages", [])
    if not isinstance(stages, list):
        return []
    later_passed_task_ids: set[str] = set()
    resolved_status_by_stage: dict[int, str] = {}
    for index, stage in reversed(list(enumerate(stages))):
        if not isinstance(stage, dict):
            continue
        task = stage.get("task")
        task_id = task.get("id") if isinstance(task, dict) else None
        test_run = stage.get("test_run")
        test_success = test_run.get("success") if isinstance(test_run, dict) else None
        post_review_test_run = stage.get("post_review_test_run")
        post_review_success = (
            post_review_test_run.get("success") if isinstance(post_review_test_run, dict) else None
        )
        status = _resolved_stage_status(stage, test_success, post_review_success)
        if status == "failed" and task_id in later_passed_task_ids:
            status = "superseded"
        resolved_status_by_stage[index] = status
        if status == "passed" and isinstance(task_id, str):
            later_passed_task_ids.add(task_id)
    statuses: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        task = stage.get("task")
        task_id = task.get("id") if isinstance(task, dict) else None
        test_run = stage.get("test_run")
        test_success = test_run.get("success") if isinstance(test_run, dict) else None
        post_review_test_run = stage.get("post_review_test_run")
        post_review_success = (
            post_review_test_run.get("success") if isinstance(post_review_test_run, dict) else None
        )
        summary = stage.get("change_summary")
        changed_files = summary.get("changed_files") if isinstance(summary, dict) else []
        patch_count = summary.get("patch_count") if isinstance(summary, dict) else 0
        status = resolved_status_by_stage.get(
            index,
            _resolved_stage_status(stage, test_success, post_review_success),
        )
        statuses.append(
            {
                "stage": stage.get("stage"),
                "task_id": task_id,
                "status": status,
                "test_success": test_success,
                "post_review_success": post_review_success,
                "changed_files": changed_files if isinstance(changed_files, list) else [],
                "patch_count": patch_count if isinstance(patch_count, int) else 0,
            }
        )
    return statuses


def _resolved_stage_status(
    stage: dict[str, Any],
    test_success: Any,
    post_review_success: Any,
) -> str:
    explicit_status = _explicit_stage_status(stage)
    if explicit_status is not None:
        return explicit_status
    return _stage_status(test_success, post_review_success)


def _explicit_stage_status(stage: dict[str, Any]) -> str | None:
    raw_status = stage.get("status")
    if not isinstance(raw_status, str):
        return None
    status = raw_status.strip().lower()
    if status == "failed_for_recovery":
        return "failed"
    if status in {"failed", "passed", "superseded"}:
        return status
    return None


def _stage_status(test_success: Any, post_review_success: Any) -> str:
    if test_success is False or post_review_success is False:
        return "failed"
    if test_success is True and post_review_success is not False:
        return "passed"
    return "unknown"


def _stage_task_ids(stage_statuses: list[dict[str, Any]], status: str) -> list[str]:
    task_ids: list[str] = []
    for stage in stage_statuses:
        task_id = stage.get("task_id")
        if stage.get("status") == status and isinstance(task_id, str):
            task_ids.append(task_id)
    return _unique_paths(task_ids)


def _recovery_history(stage_statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    passed_by_task: dict[str, dict[str, Any]] = {}
    for stage in reversed(stage_statuses):
        task_id = stage.get("task_id")
        if not isinstance(task_id, str):
            continue
        status = stage.get("status")
        if status == "passed":
            passed_by_task.setdefault(task_id, stage)
            continue
        if status != "superseded":
            continue
        resolved_by = passed_by_task.get(task_id)
        recovered.append(
            {
                "task_id": task_id,
                "failed_stage": stage.get("stage"),
                "resolved_by_stage": resolved_by.get("stage") if resolved_by else None,
                "failed_changed_files": stage.get("changed_files", []),
                "failed_patch_count": stage.get("patch_count", 0),
                "resolved_changed_files": (
                    resolved_by.get("changed_files", []) if resolved_by else []
                ),
                "resolved_patch_count": resolved_by.get("patch_count", 0) if resolved_by else 0,
            }
        )
    return list(reversed(recovered))


def _resume_recommendation(
    record: JobRecord,
    pending_ids: list[str],
    failed_stage: dict[str, Any] | None,
    execution_limits: dict[str, Any],
    failure_analysis: dict[str, Any],
    autonomy_readiness: dict[str, Any],
) -> dict[str, Any]:
    if _is_done(record):
        return {
            "action": "none",
            "task_id": None,
            "stage": None,
            "reason": None,
            "can_auto_continue": False,
            "suggested_cli_args": [],
            "suggested_continue_cli_args": [],
        }
    failure_error = _effective_failure_error(record)
    if _is_policy_hard_stop(record.last_error):
        return {
            "action": "inspect_policy_hard_stop",
            "task_id": None,
            "stage": None,
            "reason": record.last_error,
            "can_auto_continue": False,
            "suggested_cli_args": [],
            "suggested_continue_cli_args": [],
        }
    recovery_plan = _active_recovery_plan(record)
    if isinstance(recovery_plan, dict) and not recovery_plan.get("hard_stop"):
        return {
            "action": str(recovery_plan.get("strategy") or "run_recovery_plan").lower(),
            "task_id": recovery_plan.get("task_id"),
            "stage": recovery_plan.get("stage"),
            "reason": recovery_plan.get("reason") or record.last_error,
            "can_auto_continue": True,
            "recovery_plan": recovery_plan,
            "suggested_cli_args": _resume_cli_args(record.job_id),
            "suggested_continue_cli_args": _continue_cli_args(record.job_id),
        }
    if failure_analysis.get("auto_continue_blocked"):
        classification = failure_analysis.get("classification")
        action = (
            "split_or_clarify_task"
            if classification == "recurring_stage_failure"
            else "recover_repeated_failure"
        )
        task_id = failure_analysis.get("failed_task_id")
        stage = failure_analysis.get("failed_stage")
        suggested_cli_args: list[str] = []
        suggested_continue_cli_args: list[str] = []
        extra: dict[str, Any] = {}
        if classification == "diagnosed_repeated_failure":
            action = "diagnosis_guided_recovery"
        if classification == "completion_integrity_failed":
            action = "completion_audit_recovery"
        if classification == "autonomous_stage_limit_reached":
            action = "raise_stage_limit_or_resume"
            stage_limit = execution_limits.get("autonomous_stage_limit")
            if isinstance(stage_limit, dict):
                suggested_next_limit = stage_limit.get("suggested_next_max_autonomous_stages")
                task_id = pending_ids[0] if pending_ids else task_id
                stage = stage_limit.get("completed_stage_count")
                extra["limit"] = stage_limit
                extra["suggested_max_autonomous_stages"] = suggested_next_limit
                suggested_cli_args = _resume_cli_args(
                    record.job_id,
                    suggested_max_autonomous_stages=suggested_next_limit,
                )
                suggested_continue_cli_args = _continue_cli_args(record.job_id)
        if classification in {
            "prd_quality_gate_failed",
            "invalid_task_graph",
        }:
            action = "improve_planning_quality"
            blocking_items = autonomy_readiness.get("blocking_items", [])
            extra["blocking_items"] = blocking_items if isinstance(blocking_items, list) else []
            suggested_cli_args = _resume_cli_args(record.job_id)
            suggested_continue_cli_args = _continue_cli_args(record.job_id)
        return {
            "action": action,
            "task_id": task_id,
            "stage": stage,
            "reason": failure_error,
            "can_auto_continue": True,
            "suggested_cli_args": suggested_cli_args,
            "suggested_continue_cli_args": suggested_continue_cli_args,
            **extra,
        }
    stage_limit = execution_limits.get("autonomous_stage_limit")
    if failure_error == "autonomous_stage_limit_reached" and isinstance(stage_limit, dict):
        suggested_next_limit = stage_limit.get("suggested_next_max_autonomous_stages")
        return {
            "action": "raise_stage_limit_or_resume",
            "task_id": pending_ids[0] if pending_ids else None,
            "stage": stage_limit.get("completed_stage_count"),
            "reason": failure_error,
            "can_auto_continue": True,
            "limit": stage_limit,
            "suggested_max_autonomous_stages": suggested_next_limit,
            "suggested_cli_args": _resume_cli_args(
                record.job_id,
                suggested_max_autonomous_stages=suggested_next_limit,
            ),
            "suggested_continue_cli_args": _continue_cli_args(record.job_id),
        }
    if record.status.value != "done" and not autonomy_readiness.get("ready", True):
        blocking_items = autonomy_readiness.get("blocking_items", [])
        return {
            "action": "improve_planning_quality",
            "task_id": None,
            "stage": None,
            "reason": failure_error,
            "can_auto_continue": True,
            "blocking_items": blocking_items if isinstance(blocking_items, list) else [],
            "suggested_cli_args": _resume_cli_args(record.job_id),
            "suggested_continue_cli_args": _continue_cli_args(record.job_id),
        }
    if failed_stage is not None:
        task = failed_stage.get("task")
        task_id = task.get("id") if isinstance(task, dict) else None
        return {
            "action": "retry_failed_stage",
            "task_id": task_id,
            "stage": failed_stage.get("stage"),
            "reason": failure_error,
            "can_auto_continue": True,
            "suggested_cli_args": _resume_cli_args(record.job_id),
            "suggested_continue_cli_args": _continue_cli_args(record.job_id),
        }
    if pending_ids:
        return {
            "action": "continue_next_task",
            "task_id": pending_ids[0],
            "stage": None,
            "reason": failure_error,
            "can_auto_continue": True,
            "suggested_cli_args": _resume_cli_args(record.job_id),
            "suggested_continue_cli_args": _continue_cli_args(record.job_id),
        }
    return {
        "action": "none",
        "task_id": None,
        "stage": None,
        "reason": failure_error,
        "can_auto_continue": False,
        "suggested_cli_args": [],
        "suggested_continue_cli_args": [],
    }


def _failure_analysis(
    record: JobRecord,
    failed_stage: dict[str, Any] | None,
    recovery_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if _is_done(record):
        return {
            "classification": None,
            "last_error": None,
            "failure_count": record.failure_count,
            "same_test_failure_count": record.same_test_failure_count,
            "failed_task_id": None,
            "failed_stage": None,
            "auto_continue_blocked": False,
            "manual_intervention_recommended": False,
            "recommended_recovery": None,
        }
    task = failed_stage.get("task") if isinstance(failed_stage, dict) else None
    failed_task_id = task.get("id") if isinstance(task, dict) else None
    failure_error = _effective_failure_error(record)
    failed_task_id = failed_task_id or _task_id_from_error(failure_error)
    failed_stage_number = failed_stage.get("stage") if isinstance(failed_stage, dict) else None
    classification = _failure_classification(failure_error)
    prior_recoveries = [
        item
        for item in recovery_history
        if isinstance(item, dict) and item.get("task_id") == failed_task_id
    ]
    if failed_stage is not None and prior_recoveries:
        classification = "recurring_stage_failure"
    auto_continue_blocked = classification in {
        "repeated_test_failure",
        "diagnosed_repeated_failure",
        "recurring_stage_failure",
        "completion_integrity_failed",
        "prd_quality_gate_failed",
        "invalid_task_graph",
        "autonomous_stage_limit_reached",
    }
    recommended_recovery = _recommended_recovery(
        classification=classification,
        failed_task_id=failed_task_id,
        failed_stage=failed_stage_number,
    )
    analysis = {
        "classification": classification,
        "last_error": failure_error,
        "failure_count": record.failure_count,
        "same_test_failure_count": record.same_test_failure_count,
        "failed_task_id": failed_task_id,
        "failed_stage": failed_stage_number,
        "auto_continue_blocked": auto_continue_blocked,
        "manual_intervention_recommended": False,
        "recommended_recovery": recommended_recovery,
    }
    if prior_recoveries:
        analysis["prior_recovery_count"] = len(prior_recoveries)
        analysis["prior_recovered_stages"] = [
            item.get("resolved_by_stage") for item in prior_recoveries
        ]
    return analysis


def _failure_classification(last_error: str | None) -> str | None:
    if last_error is None:
        return None
    if _is_policy_hard_stop(last_error):
        return "policy_hard_stop"
    if last_error == "same_failure_threshold_reached":
        return "repeated_test_failure"
    if last_error.startswith("diagnosed_repeated_failure:"):
        return "diagnosed_repeated_failure"
    if last_error.startswith("fixer_failed:"):
        return "fixer_failed"
    if last_error.startswith("fixer_stuck:"):
        return "fixer_stuck"
    if last_error.startswith("implementation_failed:"):
        return "implementation_failed"
    if last_error.startswith("implementation_blocked:"):
        return "implementation_blocked"
    if last_error.startswith("test_writer_failed:"):
        return "test_writer_failed"
    if last_error.startswith("test_writer_blocked:"):
        return "test_writer_blocked"
    if last_error.startswith("completion_integrity_failed:"):
        return "completion_integrity_failed"
    if last_error.startswith("prd_quality_gate_failed:"):
        return "prd_quality_gate_failed"
    if last_error == "invalid_task_graph":
        return "invalid_task_graph"
    if last_error == "autonomous_stage_limit_reached":
        return "autonomous_stage_limit_reached"
    return "other"


def _is_policy_hard_stop(last_error: str | None) -> bool:
    if not last_error:
        return False
    lowered = last_error.lower()
    return lowered.startswith(
        (
            "policy_hard_stop",
            "policy_denied",
            "blocked_operation",
            "secret_access",
            "direct_main_write",
            "direct_master_write",
            "force_push",
            "production_deploy",
            "unsafe_shell",
        )
    )


def _task_id_from_error(last_error: str | None) -> str | None:
    if last_error is None or ":" not in last_error:
        return None
    prefix, suffix = last_error.split(":", 1)
    if prefix in {
        "fixer_failed",
        "fixer_stuck",
        "implementation_failed",
        "implementation_blocked",
        "test_writer_failed",
        "test_writer_blocked",
    }:
        return suffix or None
    return None


def _recommended_recovery(
    *,
    classification: str | None,
    failed_task_id: str | None,
    failed_stage: Any,
) -> dict[str, Any] | None:
    strict_planning_constraints = dict(STRICT_JOB_CONSTRAINTS)
    recovery_by_classification: dict[str, dict[str, Any]] = {
        "recurring_stage_failure": {
            "strategy": "split_or_clarify_task",
            "reason": (
                "the same task failed again after a previous autonomous recovery"
            ),
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                **strict_planning_constraints,
                "recovery_mode": "recurring_failure",
                "recovery_strategy": "split_or_clarify_task",
                "stage_review": True,
            },
        },
        "repeated_test_failure": {
            "strategy": "escalated_retry",
            "reason": (
                "same test failure repeated until the autonomous fixer threshold was reached"
            ),
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "repeated_failure",
                "recovery_strategy": "escalated_retry",
            },
        },
        "diagnosed_repeated_failure": {
            "strategy": "diagnosis_guided_retry",
            "reason": (
                "the same deterministic failure repeated, and a structured diagnosis is "
                "available to guide the next fixer attempt"
            ),
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "diagnosed_repeated_failure",
                "recovery_strategy": "diagnosis_guided_retry",
                "stage_review": True,
            },
        },
        "fixer_failed": {
            "strategy": "escalated_retry",
            "reason": "the fixer explicitly failed to safely repair the current task",
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "fixer_failure",
                "recovery_strategy": "escalated_retry",
                "stage_review": True,
            },
        },
        "fixer_stuck": {
            "strategy": "escalated_retry",
            "reason": "the fixer reported it is stuck on the current task",
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "fixer_stuck",
                "recovery_strategy": "escalated_retry",
                "stage_review": True,
            },
        },
        "implementation_failed": {
            "strategy": "replan_current_task",
            "reason": "the implementer failed before producing a safe completed change",
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                **strict_planning_constraints,
                "recovery_mode": "implementation_failure",
                "recovery_strategy": "replan_current_task",
                "stage_review": True,
            },
        },
        "implementation_blocked": {
            "strategy": "split_or_clarify_task",
            "reason": "the implementer reported the task is blocked",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                **strict_planning_constraints,
                "recovery_mode": "implementation_blocked",
                "recovery_strategy": "split_or_clarify_task",
            },
        },
        "test_writer_failed": {
            "strategy": "rewrite_tests",
            "reason": "the test writer failed to produce usable tests",
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "test_generation_failure",
                "recovery_strategy": "rewrite_tests",
                "require_test_evidence": True,
            },
        },
        "test_writer_blocked": {
            "strategy": "split_or_clarify_tests",
            "reason": "the test writer reported the test task is blocked",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                "require_task_artifacts": True,
                "require_completion_integrity": True,
                "recovery_mode": "test_generation_blocked",
                "recovery_strategy": "split_or_clarify_tests",
                "require_task_acceptance_criteria": True,
                "require_test_evidence": True,
            },
        },
        "completion_integrity_failed": {
            "strategy": "completion_audit",
            "reason": "the completion integrity gate found missing work or missing evidence",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                "recovery_mode": "completion_integrity",
                "recovery_strategy": "completion_audit",
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "require_stage_test_patches": True,
            },
        },
        "prd_quality_gate_failed": {
            "strategy": "planning_repair_strategy_change",
            "reason": "PRD quality gate failed and needs autonomous PM refinement",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                **strict_planning_constraints,
                "recovery_mode": "prd_quality_repair",
                "recovery_strategy": "planning_repair_strategy_change",
            },
            "next_actor": "pm",
        },
        "invalid_task_graph": {
            "strategy": "task_graph_replanning",
            "reason": "task graph validation failed and needs autonomous replanning",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                **strict_planning_constraints,
                "recovery_mode": "task_graph_replanning",
                "recovery_strategy": "task_graph_replanning",
            },
            "next_actor": "planner",
        },
        "autonomous_stage_limit_reached": {
            "strategy": "raise_stage_limit",
            "reason": "autonomous stage limit was reached and can be bumped",
            "preserve_failure_counts_for_model_escalation": False,
            "constraints": {
                "recovery_mode": "stage_limit",
                "recovery_strategy": "raise_stage_limit",
            },
        },
    }
    if classification not in recovery_by_classification:
        return None
    recovery = dict(recovery_by_classification[classification])
    recovery["failed_task_id"] = failed_task_id
    recovery["failed_stage"] = failed_stage
    return recovery


def _resume_cli_args(
    job_id: str,
    *,
    suggested_max_autonomous_stages: int | None = None,
) -> list[str]:
    args = ["resume-job", "--job-id", job_id]
    if suggested_max_autonomous_stages is not None:
        args.extend(["--max-autonomous-stages", str(suggested_max_autonomous_stages)])
    return args


def _continue_cli_args(job_id: str) -> list[str]:
    return ["continue-job", "--job-id", job_id]


def _planning_quality(record: JobRecord) -> dict[str, Any]:
    prd_quality = _dict_output(record, "prd_quality")
    task_graph_validation = _dict_output(record, "task_graph_validation")
    prd_attempts = _list_output(record, "prd_quality_attempts")
    current_prd_attempts = _current_prd_quality_attempts(
        prd_attempts,
        record,
        prd_quality,
    )
    task_graph_attempts = _list_output(record, "task_graph_validation_attempts")
    current_task_graph_attempts = _current_task_graph_validation_attempts(
        task_graph_attempts,
        task_graph_validation,
    )
    return {
        "prd_quality": prd_quality,
        "prd_quality_attempt_count": len(prd_attempts),
        "last_prd_quality_attempt": (
            current_prd_attempts[-1] if current_prd_attempts else None
        ),
        "task_graph_validation": task_graph_validation,
        "task_graph_validation_attempt_count": len(task_graph_attempts),
        "last_task_graph_validation_attempt": (
            current_task_graph_attempts[-1]
            if current_task_graph_attempts
            else None
        ),
        "planning_repair": _planning_repair_summary(
            current_prd_attempts,
            current_task_graph_attempts,
        ),
    }


def _planning_summary(
    record: JobRecord,
    planned_tasks: list[dict[str, Any]],
    planning_quality: dict[str, Any],
    autonomy_readiness: dict[str, Any],
) -> dict[str, Any]:
    planning_only = _dict_output(record, "planning_only")
    if not isinstance(planning_only, dict):
        planning_only = {}
    prd_quality = planning_quality.get("prd_quality")
    if not isinstance(prd_quality, dict):
        prd_quality = {}
    task_graph_validation = planning_quality.get("task_graph_validation")
    if not isinstance(task_graph_validation, dict):
        task_graph_validation = {}
    blocking_items = autonomy_readiness.get("blocking_items", [])
    if not isinstance(blocking_items, list):
        blocking_items = []
    small_part_coverage = task_graph_validation.get("small_part_coverage", [])
    if not isinstance(small_part_coverage, list):
        small_part_coverage = []
    uncovered_small_parts = task_graph_validation.get("uncovered_small_parts", [])
    if not isinstance(uncovered_small_parts, list):
        uncovered_small_parts = []
    acceptance_test_coverage = task_graph_validation.get("acceptance_test_coverage", [])
    if not isinstance(acceptance_test_coverage, list):
        acceptance_test_coverage = []
    uncovered_acceptance_tests = task_graph_validation.get(
        "uncovered_acceptance_tests",
        [],
    )
    if not isinstance(uncovered_acceptance_tests, list):
        uncovered_acceptance_tests = []
    planning_complete = bool(planning_only.get("complete"))
    declared_ready = bool(planning_only.get("ready_for_implementation"))
    autonomy_ready = bool(autonomy_readiness.get("ready"))
    return {
        "complete": planning_complete,
        "declared_ready_for_implementation": declared_ready,
        "ready_for_implementation": planning_complete and declared_ready and autonomy_ready,
        "prd_quality_passed": _optional_bool(prd_quality, "passed"),
        "task_graph_valid": _optional_bool(task_graph_validation, "valid"),
        "task_count": len(planned_tasks),
        "implementation_task_count": task_graph_validation.get(
            "implementation_task_count"
        ),
        "small_part_count": task_graph_validation.get("small_part_count"),
        "small_part_coverage": small_part_coverage,
        "uncovered_small_parts": uncovered_small_parts,
        "acceptance_test_count": task_graph_validation.get("acceptance_test_count"),
        "acceptance_test_coverage": acceptance_test_coverage,
        "uncovered_acceptance_tests": uncovered_acceptance_tests,
        "blocking_items": blocking_items,
    }


def _planning_repair_summary(
    prd_attempts: list[dict[str, Any]],
    task_graph_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    consecutive_prd_failure_count = _consecutive_attempt_count(
        prd_attempts,
        success_key="passed",
        success_value=False,
    )
    consecutive_task_graph_failure_count = _consecutive_attempt_count(
        task_graph_attempts,
        success_key="valid",
        success_value=False,
    )
    last_failed_prd = _last_attempt_with_value(
        prd_attempts,
        key="passed",
        value=False,
    )
    last_failed_task_graph = _last_attempt_with_value(
        task_graph_attempts,
        key="valid",
        value=False,
    )
    repeated_prd_missing = _repeated_recent_items(
        [
            item
            for attempt in prd_attempts
            if attempt.get("passed") is False
            for item in _strings(attempt.get("missing"))
        ]
    )
    repeated_task_graph_error_types = _repeated_recent_items(
        [
            error_type
            for attempt in task_graph_attempts
            if attempt.get("valid") is False
            for error_type in _error_types(attempt.get("errors"))
        ]
    )
    return {
        "consecutive_prd_failure_count": consecutive_prd_failure_count,
        "consecutive_task_graph_failure_count": consecutive_task_graph_failure_count,
        "last_prd_missing": (
            _strings(last_failed_prd.get("missing")) if last_failed_prd else []
        ),
        "last_task_graph_error_types": (
            _error_types(last_failed_task_graph.get("errors"))
            if last_failed_task_graph
            else []
        ),
        "repeated_prd_missing": repeated_prd_missing,
        "repeated_task_graph_error_types": repeated_task_graph_error_types,
        "strategy_change_recommended": (
            consecutive_prd_failure_count >= 3
            or consecutive_task_graph_failure_count >= 3
        ),
    }


def _current_prd_quality_attempts(
    attempts: list[dict[str, Any]],
    record: JobRecord,
    current_prd_quality: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    current_contract = _current_prd_quality_contract(record, current_prd_quality)
    if not current_contract:
        return attempts
    return [
        attempt
        for attempt in attempts
        if _prd_quality_attempt_matches_current_contract(attempt, current_contract)
    ]


def _current_prd_quality_contract(
    record: JobRecord,
    current_prd_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    prd = record.outputs.get("prd")
    if not isinstance(prd, dict):
        prd = record.outputs.get("pm")
    if isinstance(prd, dict):
        contract["prd_quality_fingerprint"] = prd_quality_fingerprint(prd)
    min_small_parts = _constraint_int(record, "min_prd_small_parts", 0)
    contract["required_small_part_count"] = min_small_parts
    if (
        min_small_parts <= 0
        and isinstance(current_prd_quality, dict)
    ):
        required_small_part_count = current_prd_quality.get("required_small_part_count")
        if isinstance(required_small_part_count, int):
            contract["required_small_part_count"] = required_small_part_count
    return contract


def _constraint_int(record: JobRecord, key: str, default: int) -> int:
    constraints = record.spec.metadata.get("constraints", {})
    if not isinstance(constraints, dict):
        return default
    value = constraints.get(key, default)
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _prd_quality_attempt_matches_current_contract(
    attempt: dict[str, Any],
    current_contract: dict[str, Any],
) -> bool:
    for key, current_value in current_contract.items():
        attempt_value = attempt.get(key)
        if key == "prd_quality_fingerprint":
            if not isinstance(attempt_value, str) or not attempt_value:
                continue
        elif key == "required_small_part_count":
            if isinstance(attempt_value, bool) or not isinstance(attempt_value, int):
                continue
        else:
            continue
        if attempt_value != current_value:
            return False
    return True


def _prd_quality_contract_mismatches(
    record: JobRecord,
    current_prd_quality: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    current_contract = _current_prd_quality_contract(record, current_prd_quality)
    stored_contract = _dict_output(record, "prd_quality_contract")
    if not isinstance(stored_contract, dict):
        if (
            _optional_bool(current_prd_quality, "passed") is True
            and "prd_quality_fingerprint" in current_contract
        ):
            return [
                {
                    "field": "prd_quality_contract",
                    "validation_value": None,
                    "current_value": current_contract,
                }
            ]
        return []
    mismatches: list[dict[str, Any]] = []
    for field, current_value in current_contract.items():
        validation_value = stored_contract.get(field)
        if field == "prd_quality_fingerprint":
            if not isinstance(validation_value, str) or not validation_value:
                mismatches.append(
                    {
                        "field": field,
                        "validation_value": validation_value,
                        "current_value": current_value,
                    }
                )
                continue
        elif field == "required_small_part_count":
            if isinstance(validation_value, bool) or not isinstance(
                validation_value,
                int,
            ):
                mismatches.append(
                    {
                        "field": field,
                        "validation_value": validation_value,
                        "current_value": current_value,
                    }
                )
                continue
        else:
            continue
        if validation_value != current_value:
            mismatches.append(
                {
                    "field": field,
                    "validation_value": validation_value,
                    "current_value": current_value,
                }
            )
    return mismatches


def _current_task_graph_validation_attempts(
    attempts: list[dict[str, Any]],
    current_validation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if current_validation is None:
        return attempts
    comparable_contract = {
        key: current_validation.get(key)
        for key in TASK_GRAPH_VALIDATION_CONTRACT_KEYS
        if _is_comparable_task_graph_validation_contract_value(
            current_validation.get(key)
        )
    }
    if not comparable_contract:
        return attempts
    return [
        attempt
        for attempt in attempts
        if _task_graph_attempt_matches_current_validation(
            attempt,
            comparable_contract,
        )
    ]


def _task_graph_attempt_matches_current_validation(
    attempt: dict[str, Any],
    current_contract: dict[str, Any],
) -> bool:
    for key, current_value in current_contract.items():
        attempt_value = attempt.get(key)
        if not _is_comparable_task_graph_validation_contract_value(attempt_value):
            continue
        if attempt_value != current_value:
            return False
    return True


def _is_comparable_task_graph_validation_contract_value(value: Any) -> bool:
    if isinstance(value, (bool, int, str)):
        return not (isinstance(value, str) and not value)
    if isinstance(value, list):
        return all(isinstance(item, str) for item in value)
    return False


def _consecutive_attempt_count(
    attempts: list[dict[str, Any]],
    *,
    success_key: str,
    success_value: bool,
) -> int:
    count = 0
    for attempt in reversed(attempts):
        if attempt.get(success_key) is success_value:
            count += 1
            continue
        break
    return count


def _last_attempt_with_value(
    attempts: list[dict[str, Any]],
    *,
    key: str,
    value: bool,
) -> dict[str, Any] | None:
    for attempt in reversed(attempts):
        if attempt.get(key) is value:
            return attempt
    return None


def _repeated_recent_items(items: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    ordered: list[str] = []
    for item in items:
        if item not in counts:
            ordered.append(item)
        counts[item] = counts.get(item, 0) + 1
    return [item for item in ordered if counts[item] >= 2]


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _error_types(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    error_types: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        error_type = item.get("type")
        if isinstance(error_type, str):
            error_types.append(error_type)
    return error_types


def _autonomy_readiness(
    record: JobRecord,
    planned_tasks: list[dict[str, Any]],
    planning_quality: dict[str, Any],
) -> dict[str, Any]:
    constraints = record.spec.metadata.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    prd_quality = planning_quality.get("prd_quality")
    if not isinstance(prd_quality, dict):
        prd_quality = None
    task_graph_validation = planning_quality.get("task_graph_validation")
    if not isinstance(task_graph_validation, dict):
        task_graph_validation = None

    require_prd_quality = bool(constraints.get("require_prd_quality"))
    require_acceptance_criteria = bool(
        constraints.get("require_task_acceptance_criteria")
    )
    require_task_artifacts = bool(constraints.get("require_task_artifacts"))
    require_completion_integrity = bool(constraints.get("require_completion_integrity"))
    require_test_evidence = bool(constraints.get("require_test_evidence"))
    require_stage_test_patches = bool(constraints.get("require_stage_test_patches"))
    require_executable_task_roles = bool(
        constraints.get("require_executable_task_roles")
    )
    stage_review = bool(constraints.get("stage_review"))
    strict_controls_enabled = any(
        [
            require_prd_quality,
            require_acceptance_criteria,
            require_task_artifacts,
            require_completion_integrity,
            require_test_evidence,
            require_stage_test_patches,
            require_executable_task_roles,
            stage_review,
        ]
    )

    prd_quality_passed = _optional_bool(prd_quality, "passed")
    prd_quality_contract_mismatches = _prd_quality_contract_mismatches(
        record,
        prd_quality,
    )
    task_graph_valid = _optional_bool(task_graph_validation, "valid")
    implementation_roles = {"implementer", "scaffold"}
    executable_roles = {*implementation_roles, "test_writer"}
    implementation_tasks = [
        task for task in planned_tasks if task.get("role") in implementation_roles
    ]
    test_writer_tasks = [
        task for task in planned_tasks if task.get("role") == "test_writer"
    ]
    executable_tasks = [
        task for task in planned_tasks if task.get("role") in executable_roles
    ]
    task_graph_validation_count_mismatches = _task_graph_validation_count_mismatches(
        task_graph_validation,
        {
            "task_count": len(planned_tasks),
            "implementation_task_count": len(implementation_tasks),
            "test_writer_task_count": len(test_writer_tasks),
            "executable_task_count": len(executable_tasks),
        },
    )
    task_graph_validation_identity_mismatches = (
        _task_graph_validation_identity_mismatches(
            task_graph_validation,
            {
                "task_ids": [
                    task["id"]
                    for task in planned_tasks
                    if isinstance(task.get("id"), str)
                ],
                "implementation_task_ids": [
                    task["id"]
                    for task in implementation_tasks
                    if isinstance(task.get("id"), str)
                ],
                "test_writer_task_ids": [
                    task["id"]
                    for task in test_writer_tasks
                    if isinstance(task.get("id"), str)
                ],
                "executable_task_ids": [
                    task["id"]
                    for task in executable_tasks
                    if isinstance(task.get("id"), str)
                ],
            },
        )
    )
    task_graph_validation_fingerprint_mismatches = (
        _task_graph_validation_fingerprint_mismatches(
            task_graph_validation,
            task_graph_validation_fingerprint(planned_tasks),
        )
    )
    task_graph_validation_prd_fingerprint_mismatches = (
        _task_graph_validation_prd_fingerprint_mismatches(
            task_graph_validation,
            _current_prd_validation_fingerprint(record),
        )
    )
    task_graph_validation_constraint_mismatches = (
        _task_graph_validation_constraint_mismatches(
            task_graph_validation,
            {
                "require_acceptance_criteria": require_acceptance_criteria,
                "require_task_artifacts": require_task_artifacts,
                "require_executable_task_roles": (
                    require_completion_integrity or require_executable_task_roles
                ),
            },
        )
    )
    task_graph_validation_stale_mismatches = [
        *task_graph_validation_count_mismatches,
        *task_graph_validation_identity_mismatches,
        *task_graph_validation_fingerprint_mismatches,
        *task_graph_validation_prd_fingerprint_mismatches,
        *task_graph_validation_constraint_mismatches,
    ]
    unsupported_task_roles = [
        {"task_id": task.get("id"), "role": task.get("role")}
        for task in planned_tasks
        if isinstance(task.get("id"), str)
        and isinstance(task.get("role"), str)
        and task.get("role") not in executable_roles
    ]
    task_ids = [
        task["id"] for task in planned_tasks if isinstance(task.get("id"), str)
    ]
    duplicate_task_ids = _duplicate_strings(task_ids)
    unknown_dependencies = _unknown_dependencies(planned_tasks)
    dependency_cycle_task_ids = _dependency_cycle_task_ids(planned_tasks)
    invalid_task_ids = []
    for task in planned_tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str):
            continue
        reason = _invalid_task_id_reason(task_id)
        if reason:
            invalid_task_ids.append(
                {
                    "task_id": task_id,
                    "role": (
                        task.get("role") if isinstance(task.get("role"), str) else ""
                    ),
                    "reason": reason,
                }
            )
    invalid_task_titles = [
        {
            "task_id": task["id"],
            "role": task["role"],
            "title": task["title"],
        }
        for task in executable_tasks
        if isinstance(task.get("id"), str)
        and isinstance(task.get("role"), str)
        and isinstance(task.get("title"), str)
        and _looks_like_placeholder_planning_item(task["title"])
    ]
    invalid_task_descriptions = [
        {
            "task_id": task["id"],
            "role": task["role"],
            "description": task["description"],
        }
        for task in executable_tasks
        if isinstance(task.get("id"), str)
        and isinstance(task.get("role"), str)
        and isinstance(task.get("description"), str)
        and _looks_like_placeholder_planning_item(task["description"])
    ]
    generic_task_acceptance_criteria = []
    for task in executable_tasks:
        task_id = task.get("id")
        role = task.get("role")
        if not isinstance(task_id, str) or not isinstance(role, str):
            continue
        for criterion in _non_empty_strings(task.get("acceptance_criteria")):
            if _looks_like_placeholder_planning_item(criterion):
                continue
            if _looks_like_generic_task_acceptance_criterion(criterion):
                generic_task_acceptance_criteria.append(
                    {
                        "task_id": task_id,
                        "role": role,
                        "acceptance_criteria": criterion,
                    }
                )
    missing_acceptance_task_ids = [
        task["id"]
        for task in implementation_tasks
        if isinstance(task.get("id"), str)
        and not _meaningful_task_acceptance_criteria(task.get("acceptance_criteria"))
    ]
    implementation_tasks_have_acceptance_criteria = (
        None if not implementation_tasks else not missing_acceptance_task_ids
    )
    missing_test_writer_acceptance_task_ids = [
        task["id"]
        for task in test_writer_tasks
        if isinstance(task.get("id"), str)
        and not _meaningful_task_acceptance_criteria(task.get("acceptance_criteria"))
    ]
    test_writer_tasks_have_acceptance_criteria = (
        None
        if not test_writer_tasks
        else not missing_test_writer_acceptance_task_ids
    )
    missing_implementation_artifact_task_ids = [
        task["id"]
        for task in implementation_tasks
        if isinstance(task.get("id"), str)
        and not _valid_artifact_paths(_task_artifact_paths(task))
    ]
    implementation_tasks_have_artifacts = (
        None
        if not implementation_tasks
        else not missing_implementation_artifact_task_ids
    )
    missing_artifact_task_ids = [
        task["id"]
        for task in executable_tasks
        if isinstance(task.get("id"), str)
        and not _valid_artifact_paths(_task_artifact_paths(task))
    ]
    executable_tasks_have_artifacts = (
        None if not executable_tasks else not missing_artifact_task_ids
    )
    invalid_task_artifacts = []
    for task in executable_tasks:
        if not isinstance(task.get("id"), str):
            continue
        invalid_paths = _invalid_artifact_paths(_task_artifact_paths(task))
        if invalid_paths:
            invalid_task_artifacts.append(
                {"task_id": task["id"], "paths": invalid_paths}
            )
    required_artifacts_missing_target_files = []
    target_files_missing_required_artifacts = []
    role_mismatched_target_files = []
    role_mismatched_required_artifacts = []
    for task in executable_tasks:
        task_id = task.get("id")
        role = task.get("role")
        if not isinstance(task_id, str) or not isinstance(role, str):
            continue
        target_files = set(_valid_artifact_paths(_non_empty_strings(task.get("target_files"))))
        required_artifacts = set(
            _valid_artifact_paths(_non_empty_strings(task.get("required_artifacts")))
        )
        missing_target_files = sorted(required_artifacts - target_files)
        if missing_target_files:
            required_artifacts_missing_target_files.append(
                {
                    "task_id": task_id,
                    "role": role,
                    "paths": missing_target_files,
                }
            )
        missing_required_artifacts = sorted(target_files - required_artifacts)
        if missing_required_artifacts:
            target_files_missing_required_artifacts.append(
                {
                    "task_id": task_id,
                    "role": role,
                    "paths": missing_required_artifacts,
                }
            )
        for path in sorted(target_files):
            expected_roles = _artifact_owner_roles(path)
            if role not in expected_roles:
                role_mismatched_target_files.append(
                    {
                        "task_id": task_id,
                        "role": role,
                        "path": path,
                        "expected_roles": sorted(expected_roles),
                    }
                )
        for path in sorted(required_artifacts):
            expected_roles = _artifact_owner_roles(path)
            if role not in expected_roles:
                role_mismatched_required_artifacts.append(
                    {
                        "task_id": task_id,
                        "role": role,
                        "path": path,
                        "expected_roles": sorted(expected_roles),
                    }
                )
    implementation_task_id_set = {
        task["id"] for task in implementation_tasks if isinstance(task.get("id"), str)
    }
    test_writer_missing_implementation_dependencies = [
        {
            "task_id": task["id"],
            "depends_on": _task_dependencies(task),
            "required_dependency_roles": sorted(implementation_roles),
        }
        for task in test_writer_tasks
        if isinstance(task.get("id"), str)
        and implementation_task_id_set
        and not any(
            dependency in implementation_task_id_set
            for dependency in _task_dependencies(task)
        )
    ]
    executor_order_dependency_violations = (
        []
        if dependency_cycle_task_ids
        else _executor_order_dependency_violations(planned_tasks)
    )

    blocking_items: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not planned_tasks:
        blocking_items.append({"type": "task_graph_missing"})
    elif not implementation_tasks:
        blocking_items.append({"type": "missing_implementation_tasks"})
    if require_prd_quality and prd_quality_passed is not True:
        blocking_items.append(
            {
                "type": "prd_quality_not_passed",
                "missing": prd_quality.get("missing", []) if prd_quality else [],
            }
        )
    elif require_prd_quality and prd_quality_contract_mismatches:
        blocking_items.append(
            {
                "type": "prd_quality_stale",
                "mismatches": prd_quality_contract_mismatches,
            }
        )
    elif prd_quality_passed is False:
        warnings.append(
            {
                "type": "prd_quality_not_passed",
                "missing": prd_quality.get("missing", []) if prd_quality else [],
            }
        )
    elif prd_quality_contract_mismatches:
        warnings.append(
            {
                "type": "prd_quality_stale",
                "mismatches": prd_quality_contract_mismatches,
            }
        )
    if planned_tasks and task_graph_valid is False:
        blocking_items.append(
            {
                "type": "task_graph_not_valid",
                "errors": (
                    task_graph_validation.get("errors", []) if task_graph_validation else []
                ),
            }
        )
    elif planned_tasks and task_graph_valid is None and strict_controls_enabled:
        blocking_items.append({"type": "task_graph_validation_missing"})
    elif planned_tasks and task_graph_valid is None:
        warnings.append({"type": "task_graph_validation_missing"})
    if planned_tasks and task_graph_valid is True and task_graph_validation_stale_mismatches:
        blocking_items.append(
            {
                "type": "task_graph_validation_stale",
                "mismatches": task_graph_validation_stale_mismatches,
            }
        )
    if require_acceptance_criteria and missing_acceptance_task_ids:
        blocking_items.append(
            {
                "type": "missing_acceptance_criteria",
                "task_ids": missing_acceptance_task_ids,
            }
        )
    if require_acceptance_criteria and missing_test_writer_acceptance_task_ids:
        blocking_items.append(
            {
                "type": "missing_test_writer_acceptance_criteria",
                "task_ids": missing_test_writer_acceptance_task_ids,
            }
        )
    if require_acceptance_criteria and generic_task_acceptance_criteria:
        blocking_items.append(
            {
                "type": "generic_task_acceptance_criteria",
                "items": generic_task_acceptance_criteria,
            }
        )
    if require_task_artifacts and missing_artifact_task_ids:
        blocking_items.append(
            {
                "type": "missing_task_artifacts",
                "task_ids": missing_artifact_task_ids,
            }
        )
    if strict_controls_enabled and invalid_task_artifacts:
        blocking_items.append(
            {
                "type": "invalid_task_artifacts",
                "items": invalid_task_artifacts,
            }
        )
    if require_task_artifacts and role_mismatched_target_files:
        blocking_items.append(
            {
                "type": "role_mismatched_target_files",
                "items": role_mismatched_target_files,
            }
        )
    if require_task_artifacts and role_mismatched_required_artifacts:
        blocking_items.append(
            {
                "type": "role_mismatched_required_artifacts",
                "items": role_mismatched_required_artifacts,
            }
        )
    if require_task_artifacts and required_artifacts_missing_target_files:
        blocking_items.append(
            {
                "type": "required_artifacts_missing_target_files",
                "items": required_artifacts_missing_target_files,
            }
        )
    if require_task_artifacts and target_files_missing_required_artifacts:
        blocking_items.append(
            {
                "type": "target_files_missing_required_artifacts",
                "items": target_files_missing_required_artifacts,
            }
        )
    if require_task_artifacts and test_writer_missing_implementation_dependencies:
        blocking_items.append(
            {
                "type": "test_writer_missing_implementation_dependency",
                "items": test_writer_missing_implementation_dependencies,
            }
        )
    if require_task_artifacts and executor_order_dependency_violations:
        blocking_items.append(
            {
                "type": "executor_order_dependency_violations",
                "items": executor_order_dependency_violations,
            }
        )
    if strict_controls_enabled and unsupported_task_roles:
        blocking_items.append(
            {
                "type": "unsupported_autonomous_task_roles",
                "items": unsupported_task_roles,
                "allowed_roles": sorted(executable_roles),
            }
        )
    if strict_controls_enabled and invalid_task_ids:
        blocking_items.append(
            {
                "type": "invalid_task_ids",
                "items": invalid_task_ids,
            }
        )
    if strict_controls_enabled and invalid_task_titles:
        blocking_items.append(
            {
                "type": "invalid_task_titles",
                "items": invalid_task_titles,
            }
        )
    if strict_controls_enabled and invalid_task_descriptions:
        blocking_items.append(
            {
                "type": "invalid_task_descriptions",
                "items": invalid_task_descriptions,
            }
        )
    if duplicate_task_ids:
        blocking_items.append(
            {
                "type": "duplicate_task_ids",
                "task_ids": duplicate_task_ids,
            }
        )
    if unknown_dependencies:
        blocking_items.append(
            {
                "type": "unknown_dependencies",
                "items": unknown_dependencies,
            }
        )
    if dependency_cycle_task_ids:
        blocking_items.append(
            {
                "type": "dependency_cycle",
                "task_ids": dependency_cycle_task_ids,
            }
        )

    return {
        "ready": not blocking_items,
        "strict_controls_enabled": strict_controls_enabled,
        "blocking_items": blocking_items,
        "warnings": warnings,
        "checks": {
            "prd_quality_passed": prd_quality_passed,
            "prd_quality_stale_count": len(prd_quality_contract_mismatches),
            "task_graph_valid": task_graph_valid,
            "implementation_task_count": len(implementation_tasks),
            "task_graph_validation_stale_count": len(
                task_graph_validation_stale_mismatches
            ),
            "implementation_tasks_have_acceptance_criteria": (
                implementation_tasks_have_acceptance_criteria
            ),
            "test_writer_tasks_have_acceptance_criteria": (
                test_writer_tasks_have_acceptance_criteria
            ),
            "implementation_tasks_have_artifacts": implementation_tasks_have_artifacts,
            "executable_tasks_have_artifacts": executable_tasks_have_artifacts,
            "invalid_task_artifact_count": len(invalid_task_artifacts),
            "role_mismatched_target_file_count": len(role_mismatched_target_files),
            "role_mismatched_required_artifact_count": len(
                role_mismatched_required_artifacts
            ),
            "required_artifacts_missing_target_file_count": len(
                required_artifacts_missing_target_files
            ),
            "target_files_missing_required_artifact_count": len(
                target_files_missing_required_artifacts
            ),
            "test_writer_missing_implementation_dependency_count": len(
                test_writer_missing_implementation_dependencies
            ),
            "executor_order_dependency_violation_count": len(
                executor_order_dependency_violations
            ),
            "unsupported_task_role_count": len(unsupported_task_roles),
            "invalid_task_id_count": len(invalid_task_ids),
            "invalid_task_title_count": len(invalid_task_titles),
            "invalid_task_description_count": len(invalid_task_descriptions),
            "duplicate_task_id_count": len(duplicate_task_ids),
            "unknown_dependency_count": len(unknown_dependencies),
            "dependency_cycle_task_count": len(dependency_cycle_task_ids),
            "generic_task_acceptance_criteria_count": len(
                generic_task_acceptance_criteria
            ),
            "require_prd_quality": require_prd_quality,
            "require_task_acceptance_criteria": require_acceptance_criteria,
            "require_task_artifacts": require_task_artifacts,
            "require_completion_integrity": require_completion_integrity,
            "require_test_evidence": require_test_evidence,
            "require_stage_test_patches": require_stage_test_patches,
            "require_executable_task_roles": require_executable_task_roles,
            "stage_review": stage_review,
        },
    }


def _optional_bool(payload: dict[str, Any] | None, key: str) -> bool | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _task_graph_validation_count_mismatches(
    validation: dict[str, Any] | None,
    current_counts: dict[str, int],
) -> list[dict[str, Any]]:
    if validation is None:
        return []
    mismatches: list[dict[str, Any]] = []
    for field, current_value in current_counts.items():
        validation_value = validation.get(field)
        if isinstance(validation_value, bool) or not isinstance(validation_value, int):
            continue
        if validation_value != current_value:
            mismatches.append(
                {
                    "field": field,
                    "validation_value": validation_value,
                    "current_value": current_value,
                }
            )
    return mismatches


def _task_graph_validation_identity_mismatches(
    validation: dict[str, Any] | None,
    current_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if validation is None:
        return []
    mismatches: list[dict[str, Any]] = []
    for field, current_value in current_ids.items():
        validation_value = validation.get(field)
        if not isinstance(validation_value, list):
            continue
        validation_ids = [
            item for item in validation_value if isinstance(item, str)
        ]
        if len(validation_ids) != len(validation_value):
            mismatches.append(
                {
                    "field": field,
                    "validation_value": validation_value,
                    "current_value": current_value,
                }
            )
            continue
        if validation_ids != current_value:
            mismatches.append(
                {
                    "field": field,
                    "validation_value": validation_ids,
                    "current_value": current_value,
                }
            )
    return mismatches


def _task_graph_validation_fingerprint_mismatches(
    validation: dict[str, Any] | None,
    current_fingerprint: str,
) -> list[dict[str, Any]]:
    if validation is None:
        return []
    validation_fingerprint = validation.get("task_graph_fingerprint")
    if not isinstance(validation_fingerprint, str) or not validation_fingerprint:
        return []
    if validation_fingerprint == current_fingerprint:
        return []
    return [
        {
            "field": "task_graph_fingerprint",
            "validation_value": validation_fingerprint,
            "current_value": current_fingerprint,
        }
    ]


def _current_prd_validation_fingerprint(record: JobRecord) -> str | None:
    prd = record.outputs.get("prd")
    if not isinstance(prd, dict):
        return None
    return prd_validation_fingerprint(prd)


def _task_graph_validation_prd_fingerprint_mismatches(
    validation: dict[str, Any] | None,
    current_fingerprint: str | None,
) -> list[dict[str, Any]]:
    if validation is None or current_fingerprint is None:
        return []
    validation_fingerprint = validation.get("prd_fingerprint")
    if not isinstance(validation_fingerprint, str) or not validation_fingerprint:
        return []
    if validation_fingerprint == current_fingerprint:
        return []
    return [
        {
            "field": "prd_fingerprint",
            "validation_value": validation_fingerprint,
            "current_value": current_fingerprint,
        }
    ]


def _task_graph_validation_constraint_mismatches(
    validation: dict[str, Any] | None,
    current_constraints: dict[str, bool],
) -> list[dict[str, Any]]:
    if validation is None:
        return []
    mismatches: list[dict[str, Any]] = []
    for field, current_value in current_constraints.items():
        validation_value = validation.get(field)
        if not isinstance(validation_value, bool):
            continue
        if current_value and not validation_value:
            mismatches.append(
                {
                    "field": field,
                    "validation_value": validation_value,
                    "current_value": current_value,
                }
            )
    return mismatches


def _non_empty_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _meaningful_task_acceptance_criteria(value: object) -> list[str]:
    return [
        item
        for item in _non_empty_strings(value)
        if not _looks_like_placeholder_planning_item(item)
        and not _looks_like_generic_task_acceptance_criterion(item)
    ]


def _looks_like_placeholder_planning_item(item: str) -> bool:
    return looks_like_placeholder_planning_item(item)


def _invalid_task_id_reason(task_id: str) -> str | None:
    raw = str(task_id)
    value = raw.strip()
    if _looks_like_placeholder_planning_item(value):
        return "placeholder"
    if raw != value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        return "unsafe_task_id_format"
    return None


def _duplicate_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def _task_dependencies(task: dict[str, Any]) -> list[str]:
    depends_on = task.get("depends_on")
    if not isinstance(depends_on, list):
        return []
    return [
        dependency
        for dependency in depends_on
        if isinstance(dependency, str) and dependency.strip()
    ]


def _unknown_dependencies(planned_tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    task_ids = {
        task["id"] for task in planned_tasks if isinstance(task.get("id"), str)
    }
    return [
        {"task_id": task["id"], "dependency": dependency}
        for task in planned_tasks
        if isinstance(task.get("id"), str)
        for dependency in _task_dependencies(task)
        if dependency not in task_ids
    ]


def _dependency_cycle_task_ids(planned_tasks: list[dict[str, Any]]) -> list[str]:
    dependencies = {
        task["id"]: _task_dependencies(task)
        for task in planned_tasks
        if isinstance(task.get("id"), str)
    }
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def visit(task_id: str) -> list[str] | None:
        if task_id in visiting:
            cycle_start = stack.index(task_id) if task_id in stack else 0
            return [*stack[cycle_start:], task_id]
        if task_id in visited:
            return None
        visiting.add(task_id)
        stack.append(task_id)
        for dependency in dependencies.get(task_id, []):
            if dependency not in dependencies:
                continue
            cycle = visit(dependency)
            if cycle is not None:
                return cycle
        visiting.remove(task_id)
        visited.add(task_id)
        stack.pop()
        return None

    for task_id in dependencies:
        cycle = visit(task_id)
        if cycle is not None:
            return cycle
    return []


def _looks_like_generic_task_acceptance_criterion(criterion: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9_]+", criterion.lower()))
    if not tokens:
        return True
    generic_tokens = {
        "a",
        "all",
        "an",
        "app",
        "application",
        "as",
        "acceptance",
        "automated",
        "be",
        "behavior",
        "behaviour",
        "check",
        "checks",
        "code",
        "complete",
        "completed",
        "correctly",
        "criteria",
        "done",
        "expected",
        "feature",
        "functionality",
        "generated",
        "implementation",
        "is",
        "it",
        "module",
        "pass",
        "passes",
        "passing",
        "properly",
        "should",
        "screen",
        "service",
        "system",
        "task",
        "test",
        "tests",
        "the",
        "work",
        "working",
        "works",
    }
    return tokens <= generic_tokens


def _task_artifact_paths(task: dict[str, Any]) -> list[str]:
    return [
        *_non_empty_strings(task.get("target_files")),
        *_non_empty_strings(task.get("required_artifacts")),
    ]


def _valid_artifact_paths(paths: list[str]) -> list[str]:
    return list(valid_planning_artifact_paths(paths))


def _invalid_artifact_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in invalid_planning_artifact_paths(paths):
        if path and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


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


def _artifact_owner_roles(path: str) -> set[str]:
    if _looks_like_test_path(path):
        if path in _PROJECT_SETUP_REQUIRED_ARTIFACTS:
            return {"scaffold", "test_writer"}
        return {"test_writer"}
    if path in _PROJECT_SETUP_REQUIRED_ARTIFACTS:
        return {"implementer", "scaffold"}
    return {"implementer"}


def _executor_order_dependency_violations(
    planned_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_by_id = {
        task["id"]: task for task in planned_tasks if isinstance(task.get("id"), str)
    }
    implementation_tasks = _order_tasks_by_dependencies(
        [
            task
            for task in planned_tasks
            if task.get("role") in {"implementer", "scaffold"}
            and isinstance(task.get("id"), str)
        ]
    )
    implementation_tasks = [
        task
        for _index, task in sorted(
            enumerate(implementation_tasks),
            key=lambda item: (
                0 if _is_project_setup_task(item[1]) else 1,
                item[0],
            ),
        )
    ]
    pending_test_tasks = _order_tasks_by_dependencies(
        [
            task
            for task in planned_tasks
            if task.get("role") == "test_writer" and isinstance(task.get("id"), str)
        ]
    )
    completed_task_ids: set[str] = set()
    violations: list[dict[str, Any]] = []

    def known_unmet_dependencies(task: dict[str, Any]) -> list[str]:
        return [
            dependency
            for dependency in _task_dependencies(task)
            if dependency in task_by_id and dependency not in completed_task_ids
        ]

    def append_violation(
        task: dict[str, Any],
        phase: str,
        unmet: list[str],
    ) -> None:
        violations.append(
            {
                "task_id": task["id"],
                "role": task.get("role"),
                "executor_phase": phase,
                "unmet_dependencies": unmet,
                "dependency_roles": [
                    {
                        "task_id": dependency,
                        "role": task_by_id[dependency].get("role"),
                    }
                    for dependency in unmet
                ],
            }
        )

    def complete_ready_tests() -> None:
        while True:
            ready_tasks: list[dict[str, Any]] = []
            for task in list(pending_test_tasks):
                local_dependencies = [
                    dependency
                    for dependency in _task_dependencies(task)
                    if any(
                        dependency == pending_task["id"]
                        for pending_task in pending_test_tasks
                    )
                    or dependency not in completed_task_ids
                ]
                if not local_dependencies or all(
                    dependency in completed_task_ids
                    for dependency in _task_dependencies(task)
                ):
                    ready_tasks.append(task)
                    pending_test_tasks.remove(task)
            if not ready_tasks:
                return
            completed_task_ids.update(task["id"] for task in ready_tasks)

    for task in implementation_tasks:
        unmet = known_unmet_dependencies(task)
        if unmet:
            append_violation(task, "implementation", unmet)
            continue
        completed_task_ids.add(task["id"])
        complete_ready_tests()

    for task in pending_test_tasks:
        unmet = known_unmet_dependencies(task)
        if unmet:
            append_violation(task, "test_writer", unmet)

    return violations


def _order_tasks_by_dependencies(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(tasks) < 2:
        return tasks
    remaining = list(tasks)
    remaining_ids = {task["id"] for task in remaining}
    completed: set[str] = set()
    ordered: list[dict[str, Any]] = []
    while remaining:
        progressed = False
        for task in list(remaining):
            local_dependencies = [
                dependency
                for dependency in _task_dependencies(task)
                if dependency in remaining_ids
            ]
            if all(dependency in completed for dependency in local_dependencies):
                ordered.append(task)
                completed.add(task["id"])
                remaining.remove(task)
                progressed = True
        if not progressed:
            ordered.extend(remaining)
            break
    return ordered


def _is_project_setup_task(task: dict[str, Any]) -> bool:
    if task.get("role") == "test_writer":
        return False
    artifacts = _valid_artifact_paths(_task_artifact_paths(task))
    artifact_text = " ".join(artifacts).lower()
    identity = " ".join(
        [
            str(task.get("id") or ""),
            str(task.get("title") or ""),
        ]
    ).lower()
    description = str(task.get("description") or "").lower()
    haystack = " ".join([identity, description, artifact_text])
    declares_project_setup_artifacts = any(
        artifact in _PROJECT_SETUP_REQUIRED_ARTIFACTS for artifact in artifacts
    )
    has_no_declared_artifacts = not artifacts
    strong_identity = any(
        keyword in identity
        for keyword in (
            "project-scaffold",
            "project scaffold",
            "project-setup",
            "project setup",
            "verify-project-setup",
        )
    )
    structural_setup = (
        "monorepo" in haystack
        or "backend/frontend/shared" in haystack
        or all(token in haystack for token in ("backend", "frontend", "shared"))
    )
    if strong_identity and (
        has_no_declared_artifacts
        or declares_project_setup_artifacts
        or structural_setup
    ):
        return True
    return structural_setup and (
        has_no_declared_artifacts or declares_project_setup_artifacts
    )


def _execution_limits(record: JobRecord) -> dict[str, Any]:
    stage_limit = _dict_output(record, "autonomous_stage_limit")
    if stage_limit is not None:
        stage_limit = dict(stage_limit)
        suggested_next_limit = _suggested_next_stage_limit(stage_limit)
        if suggested_next_limit is not None:
            stage_limit["suggested_next_max_autonomous_stages"] = suggested_next_limit
    return {
        "autonomous_stage_limit": stage_limit,
    }


def _completion_integrity(record: JobRecord) -> dict[str, Any] | None:
    report = _dict_output(record, "completion_integrity")
    return dict(report) if report is not None else None


def _suggested_next_stage_limit(stage_limit: dict[str, Any]) -> int | None:
    current = stage_limit.get("max_autonomous_stages")
    completed = stage_limit.get("completed_stage_count")
    if not isinstance(current, int) or not isinstance(completed, int):
        return None
    return max(current + 1, completed + 1)


def _dict_output(record: JobRecord, key: str) -> dict[str, Any] | None:
    value = record.outputs.get(key)
    return value if isinstance(value, dict) else None


def _list_output(record: JobRecord, key: str) -> list[dict[str, Any]]:
    value = record.outputs.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique
