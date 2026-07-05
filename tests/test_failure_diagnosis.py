from pathlib import Path

from packages.orchestrator.job_runner import JobRunner
from packages.schemas.agent_outputs import FailureDiagnosis, FixResult, TestRunResult
from packages.schemas.jobs import JobRecord, JobSpec
from packages.schemas.models import FailureClassification, FixStatus, JobStatus
from packages.schemas.tasks import PlannedTask


class StoreSpy:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self, record: JobRecord) -> JobRecord:
        self.update_count += 1
        return record


def _runner_for_static_helpers() -> JobRunner:
    return JobRunner.__new__(JobRunner)


def test_deterministic_diagnosis_classifies_import_error() -> None:
    runner = _runner_for_static_helpers()
    result = TestRunResult(
        success=False,
        failed_tests=[
            "FAILED tests/test_project_structure.py::TestProjectStructure::test_main_app_created"
        ],
        output_excerpt=(
            "from main import app\n"
            "backend\\main.py:4: ImportError\n"
            "E   ImportError: cannot import name 'Base' from 'database' "
            "(C:\\Users\\jalan\\wip\\acos-runs\\app\\backend\\database.py)\n"
        ),
        exit_code=1,
    )

    diagnosis = runner._deterministic_failure_diagnosis(result)

    assert diagnosis.classification == FailureClassification.IMPORT_ERROR
    assert "Base is imported from database" in diagnosis.root_cause
    assert diagnosis.retry_mode.value == "targeted_fix"
    assert diagnosis.failure_signature is not None
    assert diagnosis.failure_signature.startswith(
        "ImportError: cannot import name Base from database"
    )


def test_deterministic_diagnosis_classifies_syntax_error() -> None:
    runner = _runner_for_static_helpers()
    result = TestRunResult(
        success=False,
        output_excerpt='E     File "backend/main.py", line 1\nE   SyntaxError: invalid syntax\n',
        exit_code=1,
    )

    diagnosis = runner._deterministic_failure_diagnosis(result)

    assert diagnosis.classification == FailureClassification.SYNTAX_ERROR
    assert diagnosis.retry_mode.value == "targeted_fix"


def test_deterministic_diagnosis_classifies_assertion_mismatch() -> None:
    runner = _runner_for_static_helpers()
    result = TestRunResult(
        success=False,
        failed_tests=["FAILED tests/test_feature.py::test_value"],
        output_excerpt=">       assert VALUE == 1\nE       AssertionError: assert 0 == 1\n",
        exit_code=1,
    )

    diagnosis = runner._deterministic_failure_diagnosis(result)

    assert diagnosis.classification == FailureClassification.TEST_EXPECTATION_MISMATCH
    assert diagnosis.retry_mode.value == "normal_fix"


def test_deterministic_diagnosis_classifies_no_tests_ran_as_discovery_mismatch() -> None:
    runner = _runner_for_static_helpers()
    result = TestRunResult(
        success=False,
        output_excerpt="no tests ran in 0.04s",
        exit_code=5,
        executed_test_count=0,
    )

    diagnosis = runner._deterministic_failure_diagnosis(result)

    assert diagnosis.classification == FailureClassification.TEST_EXPECTATION_MISMATCH
    assert "Pytest did not discover any tests" in diagnosis.root_cause
    assert "pytest.ini" in diagnosis.failed_files
    assert diagnosis.retry_mode.value == "inspect_files_first"


def test_run_tests_with_fixes_passes_diagnosis_to_fixer_and_stores_threshold(
    tmp_path: Path,
) -> None:
    runner = _runner_for_static_helpers()
    runner.max_attempts_per_task = 3
    runner.max_same_failure_repeats = 2
    runner.store = StoreSpy()
    runner._apply_patches = lambda record, role, patches: None
    runner._fixer_allows_progress = lambda record, task, fix: True

    failing_result = TestRunResult(
        success=False,
        failed_tests=["FAILED tests/test_feature.py::test_value"],
        output_excerpt=">       assert VALUE == 1\nE       AssertionError: assert 0 == 1\n",
        exit_code=1,
    )
    runner._run_tests = lambda record: failing_result
    fixer_logs: list[list[str]] = []

    def run_role(record, role, response_model, objective, task=None, logs=None, **kwargs):
        if role == "diagnoser":
            return FailureDiagnosis(
                classification="test_expectation_mismatch",
                root_cause="VALUE remains 0 while the test expects 1",
                failed_tests=failing_result.failed_tests,
                recommended_fix_strategy="Set VALUE to 1 in feature.py",
                confidence=0.9,
                retry_mode="targeted_fix",
                failure_signature="AssertionError: assert 0 == 1",
            )
        if role == "fixer":
            fixer_logs.append(list(logs or []))
            return FixResult(status=FixStatus.FIXED, summary="no-op", patches=[])
        raise AssertionError(f"unexpected role {role}")

    runner._run_structured_role = run_role
    spec = JobSpec(
        request_text="Build it",
        repo_path=str(tmp_path),
        target_branch="acos/diagnosis-test",
    )
    record = JobRecord(job_id=spec.job_id, spec=spec)
    task = PlannedTask(
        id="core",
        title="Core",
        description="Build core",
        role="implementer",
    )

    result = runner._run_tests_with_fixes(record, task)

    assert result is failing_result
    assert record.status == JobStatus.STUCK
    assert record.last_error == "diagnosed_repeated_failure:test_expectation_mismatch"
    assert record.outputs["failure_diagnosis"]["root_cause"] == (
        "VALUE remains 0 while the test expects 1"
    )
    assert record.outputs["failure_diagnosis"]["failure_signature"] == (
        "AssertionError: assert 0 == 1"
    )
    assert len(record.outputs["failure_diagnoses"]) >= 2
    assert record.outputs["recovery_ready"]["reason"] == "same_failure_threshold_reached"
    assert record.outputs["recovery_ready"]["retry_mode"] == "targeted_fix"
    assert record.outputs["recovery_ready"]["root_cause"] == (
        "VALUE remains 0 while the test expects 1"
    )
    assert any("failure_diagnosis:" in item for item in fixer_logs[0])
    assert any("repeated_failure_instruction:" in item for item in fixer_logs[1])
