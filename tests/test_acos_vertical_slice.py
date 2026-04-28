"""Readable vertical-slice tests that document the ACOS MVP control loop."""

from __future__ import annotations

from pathlib import Path

from packages.schemas.agent_outputs import (
    FixResult,
    ImplementationResult,
    PMReviewResult,
    ReleaseResult,
    ReviewResult,
    SummaryResult,
    TestRunResult,
)
from packages.schemas.models import FixStatus, ImplementationStatus, ReviewDecision, Severity

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


def _passing_runtime_prepare_result() -> TestRunResult:
    return TestRunResult(
        success=True,
        command=["prepare-runtime-auto"],
        failed_tests=[],
        output_excerpt="runtime prepare ok",
        exit_code=0,
    )


def _passing_runtime_smoke_result() -> TestRunResult:
    return TestRunResult(
        success=True,
        command=["runtime-smoke-auto"],
        failed_tests=[],
        output_excerpt="runtime smoke ok",
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
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
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


def test_vertical_slice_respects_task_target_files_when_artifacts_exist(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    scenario["planner"]["tasks"][0]["target_files"] = [
        "feature.py",
        "tests/test_feature.py",
    ]
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(target_branch="acos/target-files-happy")

    assert record.status.value == "done"
    assert record.outputs["task_graph"]["tasks"][0]["status"] == "done"


def test_vertical_slice_blocks_when_required_task_target_files_are_missing(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation without Django scaffold",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    scenario["planner"]["tasks"][0]["target_files"] = [
        "feature.py",
        "tests/test_feature.py",
        "manage.py",
    ]
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(target_branch="acos/target-files-missing")

    assert record.status.value == "blocked"
    assert record.outputs["task_graph"]["tasks"][0]["status"] == "blocked"
    assert record.last_error is not None
    assert "manage.py" in record.last_error


def test_vertical_slice_blocks_at_planning_when_runtime_bootstrap_artifact_is_unassigned(
    tmp_path: Path,
) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation without bootstrap contract",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(
        target_branch="acos/runtime-artifact-unassigned",
        metadata={
            "runtime": {
                "prepare_commands": [["python", "manage.py", "migrate", "--noinput"]],
                "start_command": [
                    "python",
                    "manage.py",
                    "runserver",
                    "{host}:{port}",
                    "--noreload",
                ],
            }
        },
    )

    assert record.status.value == "blocked"
    assert record.last_error is not None
    assert "manage.py" in record.last_error
    assert harness.role_invocations("implementer") == []


def test_vertical_slice_blocks_at_planning_when_framework_profile_artifact_is_unassigned(
    tmp_path: Path,
) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation without bootstrap contract",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(
        target_branch="acos/framework-profile-artifact-unassigned",
        metadata={"framework_profile": "django-web"},
    )

    assert record.status.value == "blocked"
    assert record.last_error is not None
    assert "manage.py" in record.last_error
    assert harness.role_invocations("implementer") == []


def test_vertical_slice_framework_profile_scaffold_supplies_bootstrap_before_implementer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_name = "workspace"
    scaffold_artifacts = [
        "manage.py",
        f"{project_name}/__init__.py",
        f"{project_name}/settings.py",
        f"{project_name}/urls.py",
        f"{project_name}/wsgi.py",
    ]
    required_artifacts = [*scaffold_artifacts, "feature.py", "tests/test_feature.py"]
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create feature implementation without Django bootstrap files",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    scenario["planner"]["tasks"][0]["required_artifacts"] = required_artifacts
    scenario["pm"][1]["required_artifacts"] = required_artifacts
    scenario["pm"][2]["required_artifacts"] = required_artifacts
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )
    original_call_tool = harness.runner._call_tool

    def fake_call_tool(role: str, tool_name: str, **kwargs):
        if tool_name == "test_server.run_command":
            return {
                "success": True,
                "command": kwargs["argv"],
                "failed_tests": [],
                "output_excerpt": "ok",
                "exit_code": 0,
            }
        return original_call_tool(role, tool_name, **kwargs)

    monkeypatch.setattr(harness.runner, "_call_tool", fake_call_tool)

    record = harness.run_job(
        target_branch="acos/framework-profile-scaffold-happy",
        metadata={"framework_profile": "django-web"},
    )

    assert record.status.value == "done"
    assert (harness.workspace / "manage.py").exists()
    assert (harness.workspace / project_name / "wsgi.py").exists()
    assert harness.role_invocations("fixer") == []


def test_vertical_slice_blocks_after_implementer_when_runtime_bootstrap_artifact_is_missing(
    tmp_path: Path,
) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create implementation but omit bootstrap file",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    scenario["planner"]["tasks"][0]["required_artifacts"] = [
        "manage.py",
        "feature.py",
        "tests/test_feature.py",
    ]
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(
        target_branch="acos/runtime-artifact-missing-after-implementer",
        metadata={
            "runtime": {
                "prepare_commands": [["python", "manage.py", "migrate", "--noinput"]],
                "start_command": [
                    "python",
                    "manage.py",
                    "runserver",
                    "{host}:{port}",
                    "--noreload",
                ],
            }
        },
    )

    assert record.status.value == "blocked"
    assert record.last_error is not None
    assert "manage.py" in record.last_error
    assert harness.role_invocations("test_writer") == []


def test_vertical_slice_runtime_smoke_failure_then_fix_recovers_to_done(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Create runtime scaffold without wsgi module",
            changed_files=["feature.py", "manage.py", "mytodo/settings.py", "mytodo/__init__.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                    "operation": "create",
                },
                {
                    "path": "manage.py",
                    "content": (
                        "import os\n\n"
                        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mytodo.settings')\n"
                    ),
                    "operation": "create",
                },
                {
                    "path": "mytodo/__init__.py",
                    "content": "",
                    "operation": "create",
                },
                {
                    "path": "mytodo/settings.py",
                    "content": "WSGI_APPLICATION = 'mytodo.wsgi.application'\n",
                    "operation": "create",
                },
            ],
        ).model_dump(),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
        fixer=FixResult(
            status=FixStatus.FIXED,
            summary="Add the missing wsgi module",
            changed_files=["mytodo/wsgi.py"],
            patches=[
                {
                    "path": "mytodo/wsgi.py",
                    "content": (
                        "import os\n\n"
                        "from django.core.wsgi import get_wsgi_application\n\n"
                        "os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mytodo.settings')\n"
                        "application = get_wsgi_application()\n"
                    ),
                    "operation": "create",
                }
            ],
        ).model_dump(),
    )
    scenario["planner"]["tasks"][0]["target_files"] = [
        "feature.py",
        "tests/test_feature.py",
        "manage.py",
        "mytodo/wsgi.py",
    ]
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            TestRunResult(
                success=False,
                command=["runtime-smoke-auto"],
                failed_tests=[],
                output_excerpt="ModuleNotFoundError: No module named 'mytodo.wsgi'",
                exit_code=1,
            ),
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(target_branch="acos/runtime-smoke-fix")

    assert record.status.value == "done"
    assert len(harness.role_invocations("fixer")) == 1
    assert record.outputs["runtime_smoke"]["success"] is True
    assert (harness.workspace / "mytodo/wsgi.py").exists()


def test_vertical_slice_acceptance_checks_failure_then_fix_recovers_to_done(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        implementer=implemented_result(
            "Create a correct implementation",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
        fixer=FixResult(
            status=FixStatus.FIXED,
            summary="Adjust implementation after acceptance check failure",
            changed_files=["feature.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                    "operation": "update",
                }
            ],
        ).model_dump(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
            TestRunResult(
                success=False,
                command=["runtime-smoke-auto"],
                failed_tests=[],
                output_excerpt="acceptance check failed: missing body text: milk",
                exit_code=1,
            ),
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
            TestRunResult(
                success=True,
                command=["runtime-smoke-auto"],
                failed_tests=[],
                output_excerpt="acceptance checks ok",
                exit_code=0,
            ),
        ],
    )

    record = harness.run_job(
        target_branch="acos/acceptance-checks-fix",
        metadata={
            "acceptance_checks": [
                {
                    "name": "create",
                    "method": "POST",
                    "path": "/create/",
                    "form": {"title": "milk"},
                    "expect_status": 200,
                    "body_contains": ["milk"],
                }
            ]
        },
    )

    assert record.status.value == "done"
    assert len(harness.role_invocations("fixer")) == 1
    assert record.outputs["acceptance_checks"]["success"] is True


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
    assert fixer_models == ["qwen_35b", "qwen_35b", "qwen_35b"]
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


def test_vertical_slice_pm_design_review_can_trigger_architect_and_planner_rework(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        pm=[
            {
                "title": "Add Helper",
                "problem_statement": "Need a correct add helper",
                "goals": ["Pass tests"],
            },
            PMReviewResult(
                decision=ReviewDecision.REQUEST_CHANGES,
                summary="The plan is missing the bootstrap entrypoint artifact.",
                findings=[
                    {
                        "severity": Severity.HIGH,
                        "title": "Bootstrap gap",
                        "description": "Add the required setup artifact before implementation starts.",
                    }
                ],
                required_artifacts=["manage.py", "feature.py", "tests/test_feature.py"],
            ).model_dump(),
            PMReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="The revised design now covers the required bootstrap artifact.",
                required_artifacts=["manage.py", "feature.py", "tests/test_feature.py"],
            ).model_dump(),
            PMReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="The delivered result matches the revised design.",
                required_artifacts=["manage.py", "feature.py", "tests/test_feature.py"],
            ).model_dump(),
        ],
        implementer=ImplementationResult(
            status=ImplementationStatus.IMPLEMENTED,
            summary="Create the helper module and bootstrap file",
            changed_files=["manage.py", "feature.py"],
            patches=[
                {
                    "path": "manage.py",
                    "content": "print('bootstrap')\n",
                    "operation": "create",
                },
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                    "operation": "create",
                },
            ],
        ).model_dump(),
        test_writer=make_test_writer_result(),
        reviewer=approval_review(),
        security_reviewer=approval_security_review(),
    )
    scenario["architect"] = [
        {
            "summary": "Single module and pytest test.",
            "components": ["feature.py", "tests/test_feature.py"],
            "data_flows": [],
            "risks": [],
            "decisions": [],
        },
        {
            "summary": "Bootstrap file, module, and pytest test.",
            "components": ["manage.py", "feature.py", "tests/test_feature.py"],
            "data_flows": [],
            "risks": [],
            "decisions": [],
        },
    ]
    scenario["planner"] = [
        scenario["planner"],
        {
            "goal": "Implement and validate add helper",
            "tasks": [
                {
                    "id": "task-1",
                    "title": "Implement helper",
                    "description": "Create add function, bootstrap file, and tests",
                    "role": "implementer",
                    "target_files": ["manage.py", "feature.py", "tests/test_feature.py"],
                }
            ],
            "notes": ["bootstrap artifact required before runtime verification"],
        },
    ]
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[
            _passing_test_result(),
            _passing_runtime_prepare_result(),
            _passing_runtime_smoke_result(),
        ],
    )

    record = harness.run_job(target_branch="acos/design-review-rework")

    assert record.status.value == "done"
    assert len(harness.role_invocations("pm")) == 4
    assert len(harness.role_invocations("architect")) == 2
    assert len(harness.role_invocations("planner")) == 2
    assert record.outputs["pm_design_review"]["decision"] == "approve"


def test_vertical_slice_pm_acceptance_review_can_trigger_fix_and_revalidation(tmp_path: Path) -> None:
    scenario = base_vertical_slice_scenario(
        pm=[
            {
                "title": "Add Helper",
                "problem_statement": "Need a correct add helper",
                "goals": ["Pass tests"],
            },
            PMReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="The design covers the feature.",
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            PMReviewResult(
                decision=ReviewDecision.REQUEST_CHANGES,
                summary="The delivered helper still needs a user-facing clarification.",
                findings=[
                    {
                        "severity": Severity.MEDIUM,
                        "title": "Acceptance gap",
                        "description": "Clarify the implementation so the delivered result matches the intended behavior.",
                    }
                ],
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
            PMReviewResult(
                decision=ReviewDecision.APPROVE,
                summary="The delivered result now matches the request.",
                required_artifacts=["feature.py", "tests/test_feature.py"],
            ).model_dump(),
        ],
        implementer=implemented_result(
            "Create a correct implementation",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        ),
        test_writer=make_test_writer_result(),
        reviewer=[
            approval_review("Code is ready for acceptance review"),
            approval_review("Code is still good after the acceptance fix"),
        ],
        security_reviewer=[
            approval_security_review("Safe before acceptance fix"),
            approval_security_review("Safe after acceptance fix"),
        ],
        fixer=FixResult(
            status=FixStatus.FIXED,
            summary="Addressed PM acceptance feedback",
            changed_files=["feature.py"],
            patches=[
                {
                    "path": "feature.py",
                    "content": "def add(a: int, b: int) -> int:\n    # Final version accepted by PM review.\n    return a + b\n",
                    "operation": "update",
                }
            ],
        ).model_dump(),
    )
    harness = build_vertical_slice_harness(
        tmp_path,
        scenario=scenario,
        scripted_test_results=[_passing_test_result(), _passing_test_result()],
    )

    record = harness.run_job(target_branch="acos/acceptance-review-rework")

    assert record.status.value == "done"
    assert len(harness.role_invocations("pm")) == 4
    assert len(harness.role_invocations("fixer")) == 1
    assert len(harness.role_invocations("reviewer")) == 2
    assert record.outputs["pm_acceptance_review"]["decision"] == "approve"


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
