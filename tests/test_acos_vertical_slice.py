"""Readable vertical-slice tests that document the ACOS MVP control loop."""

from __future__ import annotations

from pathlib import Path

from packages.schemas.agent_outputs import FixResult, ReleaseResult, ReviewResult, SummaryResult, TestRunResult
from packages.schemas.models import FixStatus, ReviewDecision, Severity

from tests.fakes import (
    approval_review,
    approval_security_review,
    base_vertical_slice_scenario,
    build_vertical_slice_harness,
    implemented_result,
    make_test_writer_result,
)


def _failing_test_result(label: str = "test_add") -> TestRunResult:
    return TestRunResult(
        success=False,
        command=["pytest", "-q"],
        failed_tests=[label],
        output_excerpt=f"{label} FAILED",
        exit_code=1,
    )


def _passing_test_result() -> TestRunResult:
    return TestRunResult(
        success=True,
        command=["pytest", "-q"],
        failed_tests=[],
        output_excerpt="1 passed",
        exit_code=0,
    )


def test_vertical_slice_happy_path_documents_the_expected_mvp_loop(tmp_path: Path) -> None:
    """Happy path:
    Job -> PM -> Architect -> Planner -> Implementer -> Test Writer -> Reviews -> Tests -> Release -> done
    """

    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review("Proceed to tests"),
        security_reviewer=approval_security_review("No security risks in this slice"),
        summarizer=SummaryResult(
            summary="Feature implemented and validated.",
            memory_entries=["add helper implemented", "tests passing"],
        ).model_dump(),
        release_manager=ReleaseResult(
            summary="Ready for release",
            commit_message="acos: ship add helper",
            notify_message="ACOS happy path completed",
        ).model_dump(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[_passing_test_result()],
    )

    record = harness.run_job(target_branch="acos/happy-path")

    assert record.status.value == "done"
    assert record.outputs["prd"]["title"] == "Add Helper"
    assert record.outputs["architecture"]["summary"] == "Single module and pytest test."
    assert record.outputs["task_graph"]["tasks"][0]["id"] == "task-1"
    assert record.outputs["test_run"]["success"] is True
    assert harness.environment.notify_server.notifications == ["ACOS happy path completed"]
    assert len(harness.environment.git_server.commits) == 1
    assert harness.role_invocations("fixer") == []


def test_vertical_slice_test_failure_then_fix_records_history_and_reroutes_models(tmp_path: Path) -> None:
    """Failure loop:
    initial test fail -> memory save -> fixer -> fail -> fixer -> fail -> escalated fixer -> pass -> done
    """

    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a buggy implementation",
            "def add(a: int, b: int) -> int:\n    return a - b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
        fixer=[
            FixResult(
                status=FixStatus.FIXED,
                summary="Attempt a small fix",
                changed_files=["feature.py"],
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                        "operation": "update",
                    }
                ],
                addressed_failures=["test_add"],
            ).model_dump(),
            FixResult(
                status=FixStatus.FIXED,
                summary="Try another small fix",
                changed_files=["feature.py"],
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a\n",
                        "operation": "update",
                    }
                ],
                addressed_failures=["test_add"],
            ).model_dump(),
            FixResult(
                status=FixStatus.FIXED,
                summary="Apply the final correct fix",
                changed_files=["feature.py"],
                patches=[
                    {
                        "path": "feature.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                        "operation": "update",
                    }
                ],
                addressed_failures=["test_add"],
            ).model_dump(),
        ],
        summarizer=SummaryResult(
            summary="Recovered after repeated failures.",
            memory_entries=["failure loop resolved"],
        ).model_dump(),
        release_manager=ReleaseResult(
            summary="Ready after retries",
            commit_message="acos: finalize after retries",
            notify_message="ACOS retry path completed",
        ).model_dump(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _failing_test_result(),
            _failing_test_result(),
            _failing_test_result(),
            _passing_test_result(),
        ],
        max_same_failure_repeats=10,
    )

    record = harness.run_job(target_branch="acos/test-failure-then-fix")
    memory_keys = {entry["key"] for entry in harness.memory_entries(limit=20)}
    fixer_models = [item["model_key"] for item in harness.role_invocations("fixer")]

    assert record.status.value == "done"
    assert {"test_failure_1", "test_failure_2", "test_failure_3"} <= memory_keys
    assert fixer_models == ["qwen_small", "qwen_small", "qwen_35b"]
    assert record.outputs["fixer_model_selection"]["reason"] == "escalation"
    assert record.outputs["test_run"]["success"] is True


def test_vertical_slice_review_changes_then_fix_reenters_review_until_approved(tmp_path: Path) -> None:
    """Review loop:
    reviewer requests changes -> fixer runs -> reviewer approves -> tests pass -> done
    """

    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create an implementation that needs review cleanup",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=[
            ReviewResult(
                decision=ReviewDecision.REQUEST_CHANGES,
                summary="Please rename helper and add a clearer comment",
                findings=[
                    {
                        "severity": Severity.MEDIUM,
                        "title": "Style",
                        "description": "Please refactor for readability",
                    }
                ],
            ).model_dump(),
            approval_review("Looks good now"),
        ],
        security_reviewer=[
            approval_security_review("Safe before fixes"),
            approval_security_review("Safe after fixes"),
        ],
        fixer=FixResult(
            status=FixStatus.FIXED,
            summary="Addressed review feedback",
            changed_files=["feature.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    # Readable helper used by the vertical slice test.\n    return a + b\n",
                    "operation": "update",
                }
            ],
        ).model_dump(),
        release_manager=ReleaseResult(
            summary="Ready after review cycle",
            commit_message="acos: finalize review cycle",
            notify_message="ACOS review path completed",
        ).model_dump(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[_passing_test_result()],
    )

    record = harness.run_job(target_branch="acos/review-then-fix")

    assert record.status.value == "done"
    assert len(harness.role_invocations("fixer")) == 1
    assert len(harness.role_invocations("reviewer")) == 2
    assert record.outputs["reviewer"]["decision"] == "approve"
    assert record.outputs["test_run"]["success"] is True


def test_vertical_slice_marks_job_stuck_when_max_attempts_are_exceeded(tmp_path: Path) -> None:
    """Stuck loop:
    tests keep failing -> fixer cannot converge before max_attempts_per_task -> job becomes stuck
    """

    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a persistently broken implementation",
            "def add(a: int, b: int) -> int:\n    return 0\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
        fixer=[
            FixResult(
                status=FixStatus.FIXED,
                summary="No effective change",
                patches=[],
            ).model_dump(),
            FixResult(
                status=FixStatus.FIXED,
                summary="Still no effective change",
                patches=[],
            ).model_dump(),
        ],
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _failing_test_result(),
            _failing_test_result(),
            _failing_test_result(),
        ],
        max_attempts_per_task=2,
        max_same_failure_repeats=10,
    )

    record = harness.run_job(target_branch="acos/max-attempts-exceeded")
    memory_keys = {entry["key"] for entry in harness.memory_entries(limit=20)}

    assert record.status.value == "stuck"
    assert record.last_error == "max_attempts_exceeded"
    assert len(harness.role_invocations("fixer")) == 2
    assert {"test_failure_1", "test_failure_2", "test_failure_3"} <= memory_keys
