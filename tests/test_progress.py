from datetime import datetime, timedelta, timezone

from packages.orchestrator.job_constraints import STRICT_JOB_CONSTRAINTS
from packages.orchestrator.progress import summarize_job_progress
from packages.orchestrator.task_graph_validation import task_graph_validation_fingerprint
from packages.schemas.audit import AuditEvent
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus, TaskComplexity, TaskStatus
from packages.schemas.tasks import PlannedTask, TaskGraph


def test_summarize_job_progress_reports_pending_and_failed_stage(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Test core",
                role="test_writer",
                depends_on=["core"],
            ),
            PlannedTask(
                id="extra",
                title="Extra",
                description="Build extra",
                role="implementer",
                depends_on=["core-tests"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="progress-job",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.STUCK)
    record.completed_task_ids = ["core", "core-tests"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["failure_diagnosis"] = {
        "classification": "import_error",
        "root_cause": "backend/main.py imports Base from database.py",
        "recommended_fix_strategy": "Import Base from models.py",
        "retry_mode": "targeted_fix",
    }
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py", "tests/test_feature.py"],
                "patch_count": 2,
            },
            "test_run": {"success": True},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[2].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py"],
                "patch_count": 1,
            },
            "test_run": {"success": False},
        },
    ]
    record.checkpoints = [{"stage": 1}, {"stage": 2}]
    record.last_error = "same_failure_threshold_reached"

    payload = summarize_job_progress(record)

    assert payload["status"] == "stuck"
    assert payload["total_tasks"] == 3
    assert payload["completed_task_count"] == 2
    assert payload["pending_task_ids"] == ["extra"]
    assert payload["next_task"]["id"] == "extra"
    assert payload["failed_stage"]["stage"] == 2
    assert payload["stage_statuses"][0]["status"] == "passed"
    assert payload["stage_statuses"][1]["status"] == "failed"
    assert payload["successful_stage_task_ids"] == ["core"]
    assert payload["failed_stage_task_ids"] == ["extra"]
    assert payload["failure_analysis"] == {
        "classification": "repeated_test_failure",
        "last_error": "same_failure_threshold_reached",
        "failure_count": 0,
        "same_test_failure_count": 0,
        "failed_task_id": "extra",
        "failed_stage": 2,
        "auto_continue_blocked": True,
            "manual_intervention_recommended": False,
        "recommended_recovery": {
            "strategy": "escalated_retry",
            "reason": (
                "same test failure repeated until the autonomous fixer threshold was reached"
            ),
            "failed_task_id": "extra",
            "failed_stage": 2,
            "preserve_failure_counts_for_model_escalation": True,
            "constraints": {
                "recovery_mode": "repeated_failure",
                "recovery_strategy": "escalated_retry",
            },
        },
    }
    assert payload["failure_diagnosis"] == {
        "classification": "import_error",
        "root_cause": "backend/main.py imports Base from database.py",
        "recommended_fix_strategy": "Import Base from models.py",
        "retry_mode": "targeted_fix",
    }
    assert payload["resume"]["action"] == "recover_repeated_failure"
    assert payload["resume"]["task_id"] == "extra"
    assert payload["resume"]["can_auto_continue"] is True
    assert payload["resume"]["suggested_continue_cli_args"] == []
    assert payload["progress_ratio"] == 0.6667
    assert payload["change_summary"]["changed_files"] == ["feature.py", "tests/test_feature.py"]
    assert payload["change_summary"]["patch_count"] == 3
    assert payload["change_summary"]["stages"][1]["task_id"] == "extra"


def test_summarize_job_progress_reports_recovery_created_files(tmp_path) -> None:
    spec = JobSpec(
        job_id="recovery-created-progress",
        request_text="Recover missing files",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "deterministically_created_files": ["tests/test_app.py"],
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.WRITING_TESTS)
    record.runtime_state["recovery_plan"] = {
        "status": "completed",
        "constraints": {
            "deterministically_created_files": ["tests/test_app.py"],
        },
    }

    payload = summarize_job_progress(record)

    assert payload["change_summary"] == {
        "changed_files": ["tests/test_app.py"],
        "patch_count": 0,
        "stages": [],
        "recovery_created_files": ["tests/test_app.py"],
    }


def test_summarize_job_progress_merges_deterministic_test_scaffolds(
    tmp_path,
) -> None:
    spec = JobSpec(
        job_id="deterministic-scaffold-progress",
        request_text="Recover missing test",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.WRITING_TESTS)
    record.outputs["deterministic_test_scaffolds"] = [
        {
            "path": "frontend/test/project_scaffold.test.tsx",
            "reason": "repeated_missing_target_file",
        }
    ]
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "change_summary": {
                "changed_files": ["backend/main.py"],
                "patch_count": 1,
            },
        }
    ]

    payload = summarize_job_progress(record)

    assert payload["change_summary"]["changed_files"] == [
        "frontend/test/project_scaffold.test.tsx",
        "backend/main.py",
    ]
    assert payload["change_summary"]["recovery_created_files"] == [
        "frontend/test/project_scaffold.test.tsx"
    ]
    assert payload["change_summary"]["patch_count"] == 1


def test_done_progress_does_not_surface_stale_recovery_state(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Recover and finish",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
            ),
        ],
    )
    spec = JobSpec(
        job_id="done-with-stale-recovery",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": True},
        },
    ]
    record.runtime_state["last_recoverable_error"] = "invalid_task_graph"
    record.runtime_state["current_recovery_event"] = {
        "error": "invalid_task_graph",
        "reason": "invalid_task_graph",
    }
    record.runtime_state["recovery_plan"] = {
        "strategy": "task_graph_replanning",
        "reason": "invalid_task_graph",
    }
    record.outputs["last_recoverable_error"] = "invalid_task_graph"
    record.outputs["failure_diagnosis"] = {
        "classification": "invalid_task_graph",
        "root_cause": "Task graph was repaired before completion.",
    }

    payload = summarize_job_progress(record)

    assert payload["status"] == "done"
    assert payload["last_recoverable_error"] is None
    assert payload["current_recovery_event"] is None
    assert payload["recovery_plan"] is None
    assert "failure_diagnosis" not in payload
    assert payload["failure_analysis"] == {
        "classification": None,
        "last_error": None,
        "failure_count": 0,
        "same_test_failure_count": 0,
        "failed_task_id": None,
        "failed_stage": None,
        "auto_continue_blocked": False,
        "manual_intervention_recommended": False,
        "recommended_recovery": None,
    }
    assert payload["resume"] == {
        "action": "none",
        "task_id": None,
        "stage": None,
        "reason": None,
        "can_auto_continue": False,
        "suggested_cli_args": [],
        "suggested_continue_cli_args": [],
    }
    assert payload["recovered_stage_task_ids"] == ["core"]
    assert payload["recovery_history"][0]["task_id"] == "core"


def test_summarize_job_progress_reports_active_model_call(tmp_path) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(seconds=420)
    spec = JobSpec(
        job_id="active-model-progress",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.ANALYZING)
    record.runtime_state.update(
        {
            "active_role": "pm",
            "active_objective": "Produce requirements",
            "active_model": "ornith_35b_q4",
            "active_started_at": started_at.isoformat(),
            "active_model_timeout_seconds": 600.0,
        }
    )

    payload = summarize_job_progress(record)

    assert payload["active_model_call"]["role"] == "pm"
    assert payload["active_model_call"]["objective"] == "Produce requirements"
    assert payload["active_model_call"]["model"] == "ornith_35b_q4"
    assert payload["active_model_call"]["timeout_seconds"] == 600.0
    assert payload["active_model_call"]["elapsed_seconds"] >= 420
    assert payload["active_model_call"]["timeout_ratio"] >= 0.7
    assert payload["active_model_call"]["long_running"] is True


def test_summarize_job_progress_reports_diagnosis_guided_recovery(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Recover with diagnosis",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="diagnosis-guided-progress",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.STUCK)
    record.last_error = "diagnosed_repeated_failure:missing_dependency"
    record.failure_count = 2
    record.same_test_failure_count = 2
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["failure_diagnosis"] = {
        "classification": "missing_dependency",
        "root_cause": "pydantic-settings and pydantic versions are incompatible",
        "recommended_fix_strategy": "Align the dependency versions",
        "retry_mode": "targeted_fix",
    }
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["backend/requirements.txt"], "patch_count": 1},
            "test_run": {"success": False},
        },
    ]

    payload = summarize_job_progress(record)

    assert payload["last_error"] == "diagnosed_repeated_failure:missing_dependency"
    assert payload["failure_analysis"]["classification"] == "diagnosed_repeated_failure"
    assert payload["failure_analysis"]["recommended_recovery"]["strategy"] == (
        "diagnosis_guided_retry"
    )
    assert payload["resume"]["action"] == "diagnosis_guided_recovery"
    assert payload["resume"]["can_auto_continue"] is True
    assert payload["failure_diagnosis"]["root_cause"] == (
        "pydantic-settings and pydantic versions are incompatible"
    )


def test_summarize_job_progress_reports_planning_quality_attempts(tmp_path) -> None:
    spec = JobSpec(
        job_id="quality-progress-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.outputs["prd_quality"] = {
        "passed": False,
        "missing": ["acceptance_tests"],
        "warnings": [],
    }
    record.outputs["prd_quality_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "passed": False,
            "missing": ["small_parts", "acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 1,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
    ]
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "errors": [],
    }
    record.outputs["task_graph_validation_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
        {
            "attempt": 1,
            "action": "repair",
            "valid": True,
            "errors": [],
        },
    ]

    payload = summarize_job_progress(record)

    assert payload["planning_quality"]["prd_quality"]["missing"] == ["acceptance_tests"]
    assert payload["planning_quality"]["prd_quality_attempt_count"] == 2
    assert payload["planning_quality"]["last_prd_quality_attempt"]["action"] == "refine"
    assert payload["planning_quality"]["task_graph_validation"]["valid"] is True
    assert payload["planning_quality"]["task_graph_validation_attempt_count"] == 2
    assert payload["planning_quality"]["last_task_graph_validation_attempt"]["valid"] is True
    assert payload["planning_quality"]["planning_repair"] == {
        "consecutive_prd_failure_count": 2,
        "consecutive_task_graph_failure_count": 0,
        "last_prd_missing": ["acceptance_tests"],
        "last_task_graph_error_types": ["unknown_dependencies"],
        "repeated_prd_missing": ["acceptance_tests"],
        "repeated_task_graph_error_types": [],
        "strategy_change_recommended": False,
    }
    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["autonomy_readiness"]["blocking_items"] == [
        {"type": "task_graph_missing"}
    ]
    assert payload["autonomy_readiness"]["warnings"] == [
        {"type": "prd_quality_not_passed", "missing": ["acceptance_tests"]}
    ]


def test_summarize_job_progress_reports_planning_summary_for_plan_only_job(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["core works"],
            )
        ],
    )
    spec = JobSpec(
        job_id="planned-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.PLANNING)
    record.outputs["planning_only"] = {
        "complete": True,
        "ready_for_implementation": True,
    }
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["prd_quality"] = {
        "passed": True,
        "missing": [],
        "warnings": [],
    }
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "small_part_count": 1,
        "small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Build core",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_small_parts": [],
        "acceptance_test_count": 1,
        "acceptance_test_coverage": [
            {
                "acceptance_test_index": 1,
                "acceptance_test": "core works",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_acceptance_tests": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["planning_summary"] == {
        "complete": True,
        "declared_ready_for_implementation": True,
        "ready_for_implementation": True,
        "prd_quality_passed": True,
        "task_graph_valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "small_part_count": 1,
        "small_part_coverage": [
            {
                "small_part_index": 1,
                "small_part": "Build core",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_small_parts": [],
        "acceptance_test_count": 1,
        "acceptance_test_coverage": [
            {
                "acceptance_test_index": 1,
                "acceptance_test": "core works",
                "task_id": "core",
                "covered": True,
            }
        ],
        "uncovered_acceptance_tests": [],
        "blocking_items": [],
    }
    assert payload["resume"]["action"] == "continue_next_task"
    assert payload["next_task"]["id"] == "core"


def test_consumed_recovery_plan_is_not_reported_as_active_progress(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build after recovery",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["core works"],
            )
        ],
    )
    spec = JobSpec(
        job_id="consumed-recovery-plan",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.PLANNING)
    record.outputs["planning_only"] = {
        "complete": True,
        "ready_for_implementation": True,
    }
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": False},
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
        }
    ]
    record.outputs["prd_quality"] = {"passed": True, "missing": [], "warnings": []}
    record.outputs["task_graph_validation"] = {"valid": True, "errors": []}
    record.outputs["failure_diagnosis"] = {
        "classification": "invalid_task_graph",
        "root_cause": "A previous task graph was invalid.",
    }
    record.outputs["last_recoverable_error"] = "invalid_task_graph"
    record.runtime_state["last_recoverable_error"] = "invalid_task_graph"
    record.runtime_state["current_recovery_event"] = {
        "error": "invalid_task_graph",
        "reason": "invalid_task_graph",
    }
    record.runtime_state["recovery_plan"] = {
        "status": "completed",
        "consumed_by_runner": True,
        "strategy": "REPLAN_TASK",
        "reason": "invalid_task_graph",
        "next_status": "planning",
    }

    payload = summarize_job_progress(record)

    assert payload["recovery_plan"] is None
    assert payload["current_recovery_event"] is None
    assert payload["last_recoverable_error"] is None
    assert payload["failed_stage"] is None
    assert payload["failed_stage_task_ids"] == []
    assert "failure_diagnosis" not in payload
    assert payload["failure_analysis"]["classification"] is None
    assert payload["failure_analysis"]["last_error"] is None
    assert payload["failure_analysis"]["auto_continue_blocked"] is False
    assert payload["resume"]["action"] == "continue_next_task"
    assert payload["resume"]["reason"] is None
    assert payload["next_task"]["id"] == "core"


def test_summarize_job_progress_recommends_strategy_change_after_repeated_planning_failures(
    tmp_path,
) -> None:
    spec = JobSpec(
        job_id="repeated-planning-failure-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.outputs["prd_quality_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "passed": False,
            "missing": ["small_parts", "acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 1,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
        {
            "attempt": 2,
            "action": "refine",
            "passed": False,
            "missing": ["acceptance_tests"],
            "warnings": [],
        },
    ]
    record.outputs["task_graph_validation_attempts"] = [
        {
            "attempt": 0,
            "action": "initial",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
        {
            "attempt": 1,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "unknown_dependencies"}],
        },
        {
            "attempt": 2,
            "action": "repair",
            "valid": False,
            "errors": [{"type": "dependency_cycle"}],
        },
    ]

    payload = summarize_job_progress(record)

    assert payload["planning_quality"]["planning_repair"] == {
        "consecutive_prd_failure_count": 3,
        "consecutive_task_graph_failure_count": 3,
        "last_prd_missing": ["acceptance_tests"],
        "last_task_graph_error_types": ["dependency_cycle"],
        "repeated_prd_missing": ["acceptance_tests"],
        "repeated_task_graph_error_types": ["unknown_dependencies"],
        "strategy_change_recommended": True,
    }


def test_summarize_job_progress_reports_ready_for_large_autonomy(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Regression tests",
                description="Test core",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["VALUE is covered by a regression test"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-ready-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
                "require_task_artifacts": True,
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "require_stage_test_patches": True,
                "stage_review": True,
            }
        },
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["prd_quality"] = {
        "passed": True,
        "missing": [],
        "warnings": [],
    }
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"] == {
        "ready": True,
        "strict_controls_enabled": True,
        "blocking_items": [],
        "warnings": [],
        "checks": {
            "prd_quality_passed": True,
            "task_graph_valid": True,
            "implementation_task_count": 1,
            "task_graph_validation_stale_count": 0,
            "implementation_tasks_have_acceptance_criteria": True,
            "test_writer_tasks_have_acceptance_criteria": True,
            "implementation_tasks_have_artifacts": True,
            "executable_tasks_have_artifacts": True,
            "invalid_task_artifact_count": 0,
            "role_mismatched_target_file_count": 0,
            "role_mismatched_required_artifact_count": 0,
            "required_artifacts_missing_target_file_count": 0,
            "target_files_missing_required_artifact_count": 0,
            "test_writer_missing_implementation_dependency_count": 0,
            "executor_order_dependency_violation_count": 0,
            "unsupported_task_role_count": 0,
            "invalid_task_id_count": 0,
            "invalid_task_title_count": 0,
            "invalid_task_description_count": 0,
            "duplicate_task_id_count": 0,
            "unknown_dependency_count": 0,
            "dependency_cycle_task_count": 0,
            "generic_task_acceptance_criteria_count": 0,
            "require_prd_quality": True,
            "require_task_acceptance_criteria": True,
            "require_task_artifacts": True,
            "require_completion_integrity": True,
            "require_test_evidence": True,
            "require_stage_test_patches": True,
            "require_executable_task_roles": False,
            "stage_review": True,
        },
    }


def test_summarize_job_progress_blocks_autonomy_without_task_acceptance(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-missing-acceptance-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 0,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["autonomy_readiness"]["blocking_items"] == [
        {"type": "missing_acceptance_criteria", "task_ids": ["core"]}
    ]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_acceptance_criteria"
    ] is False


def test_summarize_job_progress_blocks_autonomy_without_test_writer_acceptance(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
            ),
            PlannedTask(
                id="tests",
                title="Regression tests",
                description="Test core",
                role="test_writer",
                depends_on=["core"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-missing-test-acceptance-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "test_writer_task_acceptance_criteria_count": 0,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["autonomy_readiness"]["blocking_items"] == [
        {
            "type": "missing_test_writer_acceptance_criteria",
            "task_ids": ["tests"],
        }
    ]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_acceptance_criteria"
    ] is True
    assert payload["autonomy_readiness"]["checks"][
        "test_writer_tasks_have_acceptance_criteria"
    ] is False


def test_summarize_job_progress_blocks_stale_valid_graph_with_generic_acceptance(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["All tests pass"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Regression tests",
                description="Test core",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Tests pass"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-generic-acceptance-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "test_writer_task_acceptance_criteria_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "generic_task_acceptance_criteria",
        "items": [
            {
                "task_id": "core",
                "role": "implementer",
                "acceptance_criteria": "All tests pass",
            },
            {
                "task_id": "tests",
                "role": "test_writer",
                "acceptance_criteria": "Tests pass",
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_acceptance_criteria"
    ] is False
    assert payload["autonomy_readiness"]["checks"][
        "test_writer_tasks_have_acceptance_criteria"
    ] is False


def test_summarize_job_progress_blocks_autonomy_without_task_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-missing-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_artifact_count": 0,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["autonomy_readiness"]["blocking_items"] == [
        {"type": "missing_task_artifacts", "task_ids": ["core"]}
    ]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_artifacts"
    ] is False
    assert payload["autonomy_readiness"]["checks"][
        "executable_tasks_have_artifacts"
    ] is False


def test_summarize_job_progress_blocks_autonomy_without_test_writer_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="tests",
                title="Regression tests",
                description="Test core",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["VALUE is covered by a regression test"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-missing-test-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "implementation_task_artifact_count": 1,
        "executable_task_artifact_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["autonomy_readiness"]["blocking_items"] == [
        {"type": "missing_task_artifacts", "task_ids": ["tests"]}
    ]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_artifacts"
    ] is True
    assert payload["autonomy_readiness"]["checks"][
        "executable_tasks_have_artifacts"
    ] is False


def test_summarize_job_progress_blocks_autonomy_with_invalid_task_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["../feature.py", "C:\\feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-invalid-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": False,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_artifact_count": 0,
        "executable_task_artifact_count": 0,
        "errors": [
            {
                "type": "invalid_task_artifacts",
                "items": [
                    {
                        "task_id": "core",
                        "paths": ["../feature.py", "C:\\feature.py"],
                    }
                ],
            }
        ],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_artifacts",
        "items": [
            {
                "task_id": "core",
                "paths": ["../feature.py", "C:\\feature.py"],
            }
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "invalid_task_artifact_count"
    ] == 1
    assert payload["autonomy_readiness"]["checks"][
        "executable_tasks_have_artifacts"
    ] is False


def test_summarize_job_progress_blocks_invalid_artifacts_under_any_strict_gate(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["../feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-invalid-artifacts-any-strict-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_artifacts",
        "items": [
            {"task_id": "core", "paths": ["../feature.py"]},
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "invalid_task_artifact_count"
    ] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_strict_invalid_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["Core source is implemented"],
                target_files=["frontend/src", ".github"],
                required_artifacts=["frontend/src", ".github"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-strict-invalid-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_artifact_count": 1,
        "executable_task_artifact_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_artifacts",
        "items": [
            {"task_id": "core", "paths": ["frontend/src", ".github"]},
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "invalid_task_artifact_count"
    ] == 1
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_artifacts"
    ] is False
    assert payload["autonomy_readiness"]["checks"][
        "executable_tasks_have_artifacts"
    ] is False


def test_summarize_job_progress_blocks_stale_valid_graph_with_placeholder_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["Core source is implemented"],
                target_files=["placeholder.ts"],
                required_artifacts=["placeholder.ts"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-placeholder-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_artifact_count": 1,
        "executable_task_artifact_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_artifacts",
        "items": [
            {"task_id": "core", "paths": ["placeholder.ts"]},
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "invalid_task_artifact_count"
    ] == 1
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_artifacts"
    ] is False


def test_summarize_job_progress_blocks_stale_valid_graph_with_placeholder_acceptance(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["No open questions"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-placeholder-acceptance-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "implementation_task_acceptance_criteria_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "missing_acceptance_criteria",
        "task_ids": ["core"],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "implementation_tasks_have_acceptance_criteria"
    ] is False


def test_summarize_job_progress_blocks_stale_valid_graph_with_unsupported_role(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="release-notes",
                title="Release notes",
                description="Write release notes after implementation.",
                role="release_manager",
                acceptance_criteria=["Release notes summarize VALUE behavior"],
                target_files=["CHANGELOG.md"],
                required_artifacts=["CHANGELOG.md"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-unsupported-role-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_executable_task_roles": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "unsupported_task_role_count": 0,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "unsupported_autonomous_task_roles",
        "items": [{"task_id": "release-notes", "role": "release_manager"}],
        "allowed_roles": ["implementer", "scaffold", "test_writer"],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "unsupported_task_role_count"
    ] == 1
    assert payload["autonomy_readiness"]["checks"][
        "require_executable_task_roles"
    ] is True


def test_summarize_job_progress_blocks_stale_valid_graph_with_invalid_task_id(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="Task 1",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-invalid-task-id-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "invalid_task_ids": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_ids",
        "items": [
            {
                "task_id": "Task 1",
                "role": "implementer",
                "reason": "unsafe_task_id_format",
            }
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["invalid_task_id_count"] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_placeholder_task_metadata(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="TBD",
                description="TODO",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-placeholder-task-metadata-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_acceptance_criteria": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "invalid_task_titles": [],
        "invalid_task_descriptions": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "invalid_task_titles",
        "items": [
            {"task_id": "core", "role": "implementer", "title": "TBD"},
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert {
        "type": "invalid_task_descriptions",
        "items": [
            {"task_id": "core", "role": "implementer", "description": "TODO"},
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["invalid_task_title_count"] == 1
    assert (
        payload["autonomy_readiness"]["checks"]["invalid_task_description_count"]
        == 1
    )


def test_summarize_job_progress_blocks_stale_valid_graph_without_implementation_tasks(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="tests",
                title="Regression tests",
                description="Test core behavior.",
                role="test_writer",
                acceptance_criteria=["VALUE is covered by a regression test"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-missing-implementation-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 0,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "missing_implementation_tasks",
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["implementation_task_count"] == 0


def test_summarize_job_progress_blocks_stale_valid_graph_when_validation_counts_do_not_match(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Cover core behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-validation-counts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "implementation_task_count": 1,
        "test_writer_task_count": 0,
        "executable_task_count": 1,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "task_graph_validation_stale",
        "mismatches": [
            {
                "field": "task_count",
                "validation_value": 1,
                "current_value": 2,
            },
            {
                "field": "test_writer_task_count",
                "validation_value": 0,
                "current_value": 1,
            },
            {
                "field": "executable_task_count",
                "validation_value": 1,
                "current_value": 2,
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "task_graph_validation_stale_count"
    ] == 3


def test_summarize_job_progress_blocks_stale_valid_graph_when_validation_task_ids_do_not_match(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Cover core behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-validation-task-ids-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "test_writer_task_count": 1,
        "executable_task_count": 2,
        "task_ids": ["old-core", "old-tests"],
        "implementation_task_ids": ["old-core"],
        "test_writer_task_ids": ["old-tests"],
        "executable_task_ids": ["old-core", "old-tests"],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "task_graph_validation_stale",
        "mismatches": [
            {
                "field": "task_ids",
                "validation_value": ["old-core", "old-tests"],
                "current_value": ["core", "core-tests"],
            },
            {
                "field": "implementation_task_ids",
                "validation_value": ["old-core"],
                "current_value": ["core"],
            },
            {
                "field": "test_writer_task_ids",
                "validation_value": ["old-tests"],
                "current_value": ["core-tests"],
            },
            {
                "field": "executable_task_ids",
                "validation_value": ["old-core", "old-tests"],
                "current_value": ["core", "core-tests"],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "task_graph_validation_stale_count"
    ] == 4


def test_summarize_job_progress_blocks_stale_valid_graph_when_fingerprint_does_not_match(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Cover core behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    stale_tasks = [
        {
            **task.model_dump(mode="json"),
            "target_files": ["old_feature.py"],
            "required_artifacts": ["old_feature.py"],
        }
        if task.id == "core"
        else task.model_dump(mode="json")
        for task in task_graph.tasks
    ]
    stale_fingerprint = task_graph_validation_fingerprint(stale_tasks)
    current_fingerprint = task_graph_validation_fingerprint(
        [task.model_dump(mode="json") for task in task_graph.tasks]
    )
    spec = JobSpec(
        job_id="autonomy-stale-validation-fingerprint-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "test_writer_task_count": 1,
        "executable_task_count": 2,
        "task_ids": ["core", "core-tests"],
        "implementation_task_ids": ["core"],
        "test_writer_task_ids": ["core-tests"],
        "executable_task_ids": ["core", "core-tests"],
        "task_graph_fingerprint": stale_fingerprint,
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "task_graph_validation_stale",
        "mismatches": [
            {
                "field": "task_graph_fingerprint",
                "validation_value": stale_fingerprint,
                "current_value": current_fingerprint,
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "task_graph_validation_stale_count"
    ] == 1


def test_summarize_job_progress_does_not_treat_status_or_complexity_only_fingerprint_change_as_stale(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                status=TaskStatus.READY,
                complexity=TaskComplexity.HIGH,
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Cover core behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    validation_tasks = [
        {
            **task.model_dump(mode="json"),
            "status": TaskStatus.TODO.value,
            "complexity": TaskComplexity.MEDIUM.value,
        }
        if task.id == "core"
        else task.model_dump(mode="json")
        for task in task_graph.tasks
    ]
    spec = JobSpec(
        job_id="autonomy-status-complexity-fingerprint-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "test_writer_task_count": 1,
        "executable_task_count": 2,
        "task_ids": ["core", "core-tests"],
        "implementation_task_ids": ["core"],
        "test_writer_task_ids": ["core-tests"],
        "executable_task_ids": ["core", "core-tests"],
        "task_graph_fingerprint": task_graph_validation_fingerprint(validation_tasks),
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is True
    assert payload["autonomy_readiness"]["checks"][
        "task_graph_validation_stale_count"
    ] == 0


def test_summarize_job_progress_blocks_stale_valid_graph_with_duplicate_task_ids(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core",
                role="implementer",
                acceptance_criteria=["VALUE equals 1"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="core",
                title="Extra core",
                description="Build another core task.",
                role="implementer",
                acceptance_criteria=["VALUE equals 2"],
                target_files=["extra.py"],
                required_artifacts=["extra.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-duplicate-task-id-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_completion_integrity": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "duplicate_task_ids": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "duplicate_task_ids",
        "task_ids": ["core"],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["duplicate_task_id_count"] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_unknown_dependency(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="views",
                title="Views",
                description="Build views after models.",
                role="implementer",
                depends_on=["models"],
                acceptance_criteria=["Views render model data"],
                target_files=["views.py"],
                required_artifacts=["views.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-unknown-dependency-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 1,
        "unknown_dependencies": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "unknown_dependencies",
        "items": [{"task_id": "views", "dependency": "models"}],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["unknown_dependency_count"] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_role_mismatched_artifacts(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["frontend/test/core.test.tsx"],
                required_artifacts=["frontend/test/core.test.tsx"],
            ),
            PlannedTask(
                id="ui-tests",
                title="UI tests",
                description="Cover UI behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["UI behavior is covered by tests"],
                target_files=["frontend/src/App.tsx"],
                required_artifacts=["frontend/src/App.tsx"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-role-mismatched-artifacts-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "role_mismatched_target_files": [],
        "role_mismatched_required_artifacts": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "role_mismatched_target_files",
        "items": [
            {
                "task_id": "core",
                "role": "implementer",
                "path": "frontend/test/core.test.tsx",
                "expected_roles": ["test_writer"],
            },
            {
                "task_id": "ui-tests",
                "role": "test_writer",
                "path": "frontend/src/App.tsx",
                "expected_roles": ["implementer", "scaffold"],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "role_mismatched_target_file_count"
    ] == 2


def test_summarize_job_progress_blocks_stale_valid_graph_with_artifact_shape_mismatch(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="core",
                title="Core",
                description="Build core behavior.",
                role="implementer",
                acceptance_criteria=["Core behavior returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py", "config.py"],
            ),
            PlannedTask(
                id="core-tests",
                title="Core tests",
                description="Cover core behavior.",
                role="test_writer",
                depends_on=["core"],
                acceptance_criteria=["Core behavior has regression tests"],
                target_files=["tests/test_feature.py", "tests/test_extra.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-artifact-shape-mismatch-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "required_artifacts_missing_target_files": [],
        "target_files_missing_required_artifacts": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "required_artifacts_missing_target_files",
        "items": [
            {
                "task_id": "core",
                "role": "implementer",
                "paths": ["config.py"],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert {
        "type": "target_files_missing_required_artifacts",
        "items": [
            {
                "task_id": "core-tests",
                "role": "test_writer",
                "paths": ["tests/test_extra.py"],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "required_artifacts_missing_target_file_count"
    ] == 1
    assert payload["autonomy_readiness"]["checks"][
        "target_files_missing_required_artifact_count"
    ] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_executor_order_violation(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="feature",
                title="Feature",
                description="Build feature after test preparation.",
                role="implementer",
                depends_on=["feature-tests"],
                acceptance_criteria=["Feature returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="feature-tests",
                title="Feature tests",
                description="Prepare feature tests.",
                role="test_writer",
                acceptance_criteria=["Feature has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-executor-order-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "executor_order_dependency_violations": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "executor_order_dependency_violations",
        "items": [
            {
                "task_id": "feature",
                "role": "implementer",
                "executor_phase": "implementation",
                "unmet_dependencies": ["feature-tests"],
                "dependency_roles": [
                    {"task_id": "feature-tests", "role": "test_writer"},
                ],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "executor_order_dependency_violation_count"
    ] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_test_writer_without_implementation_dependency(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="feature",
                title="Feature",
                description="Build feature behavior.",
                role="implementer",
                acceptance_criteria=["Feature returns VALUE"],
                target_files=["feature.py"],
                required_artifacts=["feature.py"],
            ),
            PlannedTask(
                id="feature-tests",
                title="Feature tests",
                description="Cover feature behavior.",
                role="test_writer",
                acceptance_criteria=["Feature behavior has regression tests"],
                target_files=["tests/test_feature.py"],
                required_artifacts=["tests/test_feature.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-test-writer-dependency-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_task_artifacts": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "implementation_task_count": 1,
        "test_writer_missing_implementation_dependencies": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "test_writer_missing_implementation_dependency",
        "items": [
            {
                "task_id": "feature-tests",
                "depends_on": [],
                "required_dependency_roles": ["implementer", "scaffold"],
            },
        ],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"][
        "test_writer_missing_implementation_dependency_count"
    ] == 1


def test_summarize_job_progress_blocks_stale_valid_graph_with_dependency_cycle(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(
                id="a",
                title="A",
                description="Build A after B.",
                role="implementer",
                depends_on=["b"],
                acceptance_criteria=["A exists"],
                target_files=["a.py"],
                required_artifacts=["a.py"],
            ),
            PlannedTask(
                id="b",
                title="B",
                description="Build B after A.",
                role="implementer",
                depends_on=["a"],
                acceptance_criteria=["B exists"],
                target_files=["b.py"],
                required_artifacts=["b.py"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="autonomy-stale-dependency-cycle-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": True,
        "task_count": 2,
        "dependency_cycle_task_ids": [],
        "errors": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert {
        "type": "dependency_cycle",
        "task_ids": ["a", "b", "a"],
    } in payload["autonomy_readiness"]["blocking_items"]
    assert payload["autonomy_readiness"]["checks"]["dependency_cycle_task_count"] == 3


def test_summarize_job_progress_recommends_planning_repair_for_prd_quality_gate(
    tmp_path,
) -> None:
    spec = JobSpec(
        job_id="prd-quality-repair-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_prd_quality": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.last_error = "prd_quality_gate_failed:acceptance_tests"
    record.outputs["prd_quality"] = {
        "passed": False,
        "missing": ["acceptance_tests"],
        "warnings": [],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["resume"] == {
        "action": "improve_planning_quality",
        "task_id": None,
        "stage": None,
        "reason": "prd_quality_gate_failed:acceptance_tests",
        "can_auto_continue": True,
        "blocking_items": [
            {"type": "task_graph_missing"},
            {"type": "prd_quality_not_passed", "missing": ["acceptance_tests"]},
        ],
        "suggested_cli_args": ["resume-job", "--job-id", "prd-quality-repair-job"],
        "suggested_continue_cli_args": [
            "continue-job",
            "--job-id",
            "prd-quality-repair-job",
        ],
    }


def test_summarize_job_progress_recommends_planning_repair_for_invalid_task_graph(
    tmp_path,
) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="invalid-task-graph-repair-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={"constraints": {"require_completion_integrity": True}},
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.last_error = "invalid_task_graph"
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["task_graph_validation"] = {
        "valid": False,
        "task_count": 1,
        "implementation_task_count": 1,
        "errors": [{"type": "missing_small_part_coverage"}],
    }

    payload = summarize_job_progress(record)

    assert payload["autonomy_readiness"]["ready"] is False
    assert payload["resume"]["action"] == "improve_planning_quality"
    assert payload["resume"]["can_auto_continue"] is True
    assert payload["resume"]["blocking_items"] == [
        {
            "type": "task_graph_not_valid",
            "errors": [{"type": "missing_small_part_coverage"}],
        }
    ]
    assert payload["resume"]["suggested_continue_cli_args"] == [
        "continue-job",
        "--job-id",
        "invalid-task-graph-repair-job",
    ]


def test_summarize_job_progress_marks_recovered_failed_stage_as_superseded(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Recover incrementally",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="recovered-stage-progress",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.DONE)
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py"],
                "patch_count": 1,
            },
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py", "tests/test_feature.py"],
                "patch_count": 2,
            },
            "test_run": {"success": True},
        },
    ]

    payload = summarize_job_progress(record)

    assert payload["failed_stage"] is None
    assert payload["failed_stage_task_ids"] == []
    assert payload["recovered_stage_task_ids"] == ["core"]
    assert payload["successful_stage_task_ids"] == ["core"]
    assert payload["stage_statuses"][0]["status"] == "superseded"
    assert payload["stage_statuses"][1]["status"] == "passed"
    assert payload["recovery_history"] == [
        {
            "task_id": "core",
            "failed_stage": 1,
            "resolved_by_stage": 2,
            "failed_changed_files": ["feature.py"],
            "failed_patch_count": 1,
            "resolved_changed_files": ["feature.py", "tests/test_feature.py"],
            "resolved_patch_count": 2,
        }
    ]
    assert payload["resume"]["action"] == "none"
    assert payload["failure_analysis"]["classification"] is None


def test_summarize_job_progress_detects_recurring_failure_after_recovery(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Recover but catch recurrence",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
        ],
    )
    spec = JobSpec(
        job_id="recurring-stage-progress",
        request_text="Build it",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.STUCK)
    record.last_error = "same_failure_threshold_reached"
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": False},
        },
        {
            "stage": 2,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {
                "changed_files": ["feature.py", "tests/test_feature.py"],
                "patch_count": 2,
            },
            "test_run": {"success": True},
        },
        {
            "stage": 3,
            "task": task_graph.tasks[0].model_dump(),
            "change_summary": {"changed_files": ["feature.py"], "patch_count": 1},
            "test_run": {"success": False},
        },
    ]

    payload = summarize_job_progress(record)

    assert payload["failed_stage"]["stage"] == 3
    assert payload["stage_statuses"][0]["status"] == "superseded"
    assert payload["stage_statuses"][1]["status"] == "passed"
    assert payload["stage_statuses"][2]["status"] == "failed"
    assert payload["failure_analysis"]["classification"] == "recurring_stage_failure"
    assert payload["failure_analysis"]["failed_task_id"] == "core"
    assert payload["failure_analysis"]["prior_recovery_count"] == 1
    assert payload["failure_analysis"]["prior_recovered_stages"] == [2]
    assert payload["failure_analysis"]["recommended_recovery"]["strategy"] == (
        "split_or_clarify_task"
    )
    assert payload["failure_analysis"]["recommended_recovery"]["constraints"] == {
        **STRICT_JOB_CONSTRAINTS,
        "recovery_mode": "recurring_failure",
        "recovery_strategy": "split_or_clarify_task",
    }
    assert payload["resume"]["action"] == "split_or_clarify_task"
    assert payload["resume"]["can_auto_continue"] is True


def test_summarize_job_progress_recommends_recovery_by_failure_type(tmp_path) -> None:
    cases = [
        (
            "implementation_failed:core",
            "implementation_failed",
            "core",
            "replan_current_task",
            {
                **STRICT_JOB_CONSTRAINTS,
                "recovery_mode": "implementation_failure",
                "recovery_strategy": "replan_current_task",
            },
        ),
        (
            "test_writer_failed:core-tests",
            "test_writer_failed",
            "core-tests",
            "rewrite_tests",
            {
                "recovery_mode": "test_generation_failure",
                "recovery_strategy": "rewrite_tests",
                "require_test_evidence": True,
            },
        ),
        (
            "completion_integrity_failed:missing_test_evidence",
            "completion_integrity_failed",
            None,
            "completion_audit",
            {
                "recovery_mode": "completion_integrity",
                "recovery_strategy": "completion_audit",
                "require_completion_integrity": True,
                "require_test_evidence": True,
                "require_stage_test_patches": True,
            },
        ),
        (
            "fixer_failed:core",
            "fixer_failed",
            "core",
            "escalated_retry",
            {
                "recovery_mode": "fixer_failure",
                "recovery_strategy": "escalated_retry",
                "stage_review": True,
            },
        ),
    ]
    for last_error, classification, task_id, strategy, constraints in cases:
        spec = JobSpec(
            job_id=f"recovery-{classification}",
            request_text="Build it carefully",
            repo_path=str(tmp_path),
        )
        record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.FAILED)
        record.last_error = last_error

        payload = summarize_job_progress(record)

        recovery = payload["failure_analysis"]["recommended_recovery"]
        assert payload["failure_analysis"]["classification"] == classification
        assert payload["failure_analysis"]["failed_task_id"] == task_id
        assert recovery["strategy"] == strategy
        assert recovery["failed_task_id"] == task_id
        assert recovery["constraints"] == constraints


def test_summarize_job_progress_includes_completion_integrity_report(tmp_path) -> None:
    spec = JobSpec(
        job_id="completion-integrity-progress",
        request_text="Build it with evidence",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.last_error = "completion_integrity_failed:missing_test_evidence"
    record.outputs["completion_integrity"] = {
        "passed": False,
        "failure_reasons": ["missing_test_evidence"],
        "require_completion_integrity": True,
        "require_test_evidence": True,
        "require_stage_test_patches": True,
        "planned_task_count": 1,
        "completed_task_count": 1,
        "planned_task_ids": ["core"],
        "completed_task_ids": ["core"],
        "missing_task_ids": [],
        "test_success": True,
        "executed_test_count": 0,
        "stages_missing_test_patches": [],
    }

    payload = summarize_job_progress(record)

    assert payload["completion_integrity"] == record.outputs["completion_integrity"]
    assert payload["failure_analysis"]["classification"] == "completion_integrity_failed"
    assert payload["failure_analysis"]["recommended_recovery"]["strategy"] == (
        "completion_audit"
    )
    assert payload["resume"] == {
        "action": "completion_audit_recovery",
        "task_id": None,
        "stage": None,
        "reason": "completion_integrity_failed:missing_test_evidence",
        "can_auto_continue": True,
        "suggested_cli_args": [],
        "suggested_continue_cli_args": [],
    }


def test_summarize_job_progress_reports_stage_limit_resume_guidance(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Build incrementally",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
            PlannedTask(
                id="extra",
                title="Extra",
                description="Build extra",
                role="implementer",
                depends_on=["core"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="stage-limit-progress-job",
        request_text="Build it in stages",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.BLOCKED)
    record.completed_task_ids = ["core"]
    record.last_error = "autonomous_stage_limit_reached"
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stage_limit"] = {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
    }

    payload = summarize_job_progress(record)

    assert payload["execution_limits"]["autonomous_stage_limit"] == {
        "max_autonomous_stages": 1,
        "completed_stage_count": 1,
        "suggested_next_max_autonomous_stages": 2,
    }
    assert payload["resume"]["action"] == "raise_stage_limit_or_resume"
    assert payload["resume"]["task_id"] == "extra"
    assert payload["resume"]["stage"] == 1
    assert payload["resume"]["limit"]["max_autonomous_stages"] == 1
    assert payload["resume"]["suggested_max_autonomous_stages"] == 2
    assert payload["resume"]["suggested_cli_args"] == [
        "resume-job",
        "--job-id",
        "stage-limit-progress-job",
        "--max-autonomous-stages",
        "2",
    ]
    assert payload["resume"]["suggested_continue_cli_args"] == [
        "continue-job",
        "--job-id",
        "stage-limit-progress-job",
    ]


def test_summarize_job_progress_marks_post_review_failures_as_failed(tmp_path) -> None:
    task_graph = TaskGraph(
        goal="Build with review",
        tasks=[
            PlannedTask(id="core", title="Core", description="Build core", role="implementer"),
            PlannedTask(
                id="extra",
                title="Extra",
                description="Build extra",
                role="implementer",
                depends_on=["core"],
            ),
        ],
    )
    spec = JobSpec(
        job_id="review-progress-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.TESTING)
    record.completed_task_ids = ["core"]
    record.outputs["task_graph"] = task_graph.model_dump()
    record.outputs["autonomous_stages"] = [
        {
            "stage": 1,
            "task": task_graph.tasks[0].model_dump(),
            "test_run": {"success": True},
            "post_review_test_run": {"success": False},
            "change_summary": {
                "changed_files": ["feature.py"],
                "patch_count": 2,
            },
        }
    ]

    payload = summarize_job_progress(record)

    assert payload["stage_statuses"] == [
        {
            "stage": 1,
            "task_id": "core",
            "status": "failed",
            "test_success": True,
            "post_review_success": False,
            "changed_files": ["feature.py"],
            "patch_count": 2,
        }
    ]
    assert payload["failed_stage_task_ids"] == ["core"]
    assert payload["failed_stage"]["stage"] == 1
    assert payload["resume"]["action"] == "retry_failed_stage"
    assert payload["resume"]["task_id"] == "core"


def test_summarize_job_progress_reports_model_token_metrics(tmp_path) -> None:
    spec = JobSpec(
        job_id="model-metrics-progress",
        request_text="Track model metrics",
        repo_path=str(tmp_path),
    )
    record = JobRecord(job_id=spec.job_id, spec=spec, status=JobStatus.RUNNING)
    record.audit_events.extend(
        [
            AuditEvent(
                timestamp="2026-07-04T00:00:00Z",
                event_type="model_call",
                role="pm",
                action="ornith_35b_q4",
                status="success",
                metadata={
                    "model_key": "ornith_35b_q4",
                    "provider_key": "local_ornith",
                    "usage_source": "provider",
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                    "duration_seconds": 2.0,
                    "completion_tokens_per_second": 20.0,
                    "total_tokens_per_second": 70.0,
                },
            ),
            AuditEvent(
                timestamp="2026-07-04T00:00:03Z",
                event_type="model_call",
                role="fixer",
                action="ncmoe40_q4",
                status="success",
                metadata={
                    "model_key": "ncmoe40_q4",
                    "provider_key": "local_ornith",
                    "usage_source": "estimate",
                    "prompt_tokens": 60,
                    "completion_tokens": 20,
                    "total_tokens": 80,
                    "duration_seconds": 1.0,
                    "completion_tokens_per_second": 20.0,
                    "total_tokens_per_second": 80.0,
                },
            ),
        ]
    )

    payload = summarize_job_progress(record)

    metrics = payload["model_metrics"]
    assert metrics["model_call_count"] == 2
    assert metrics["total_prompt_tokens"] == 160
    assert metrics["total_completion_tokens"] == 60
    assert metrics["total_tokens"] == 220
    assert metrics["latest_call"]["role"] == "fixer"
    assert metrics["latest_call"]["usage_source"] == "estimate"
    assert metrics["latest_completion_tps"] == 20.0
    assert metrics["average_completion_tps"] == 20.0
    assert metrics["by_role"]["pm"]["total_tokens"] == 140
    assert metrics["by_model"]["ncmoe40_q4"]["model_call_count"] == 1
