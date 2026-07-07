from packages.orchestrator.progress import summarize_job_progress
from packages.schemas.audit import AuditEvent
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import JobStatus
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
        job_id="autonomy-ready-job",
        request_text="Build it carefully",
        repo_path=str(tmp_path),
        metadata={
            "constraints": {
                "require_prd_quality": True,
                "require_task_acceptance_criteria": True,
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
            "implementation_tasks_have_acceptance_criteria": True,
            "require_prd_quality": True,
            "require_task_acceptance_criteria": True,
            "require_completion_integrity": True,
            "require_test_evidence": True,
            "require_stage_test_patches": True,
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
        "recovery_mode": "recurring_failure",
        "recovery_strategy": "split_or_clarify_task",
        "require_task_acceptance_criteria": True,
        "stage_review": True,
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
                "recovery_mode": "implementation_failure",
                "recovery_strategy": "replan_current_task",
                "require_task_acceptance_criteria": True,
                "stage_review": True,
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
