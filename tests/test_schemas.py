import pytest
from pydantic import ValidationError

from packages.schemas.agent_outputs import (
    ArchitecturePlan,
    FailureDiagnosis,
    FilePatch,
    Finding,
    FixResult,
    ImplementationResult,
    PRD,
)
from packages.schemas.context import ContextPacket
from packages.schemas.jobs import JobSpec
from packages.schemas.models import (
    AgentModelConfig,
    FixStatus,
    ImplementationStatus,
    ModelConfig,
    ModelProviderConfig,
    ModelRoutingConfig,
    ProviderType,
)
from packages.schemas.tasks import PlannedTask, TaskGraph


def test_schema_instantiation() -> None:
    provider = ModelProviderConfig(
        name="local",
        type=ProviderType.OPENAI_COMPATIBLE,
        base_url="http://localhost:8000/v1",
        api_key_env="TEST_KEY",
    )
    model = ModelConfig(
        model_id="test-model",
        provider="local",
        model="test/model",
        display_name="Test",
        max_context_tokens=8192,
        max_output_tokens=2048,
    )
    agent = AgentModelConfig(
        role="implementer",
        primary_model="test-model",
        max_output_tokens=1024,
        context_budget_tokens=4096,
        max_tool_steps=24,
        output_schema="ImplementationResult",
    )
    assert agent.max_tool_steps == 24
    routing = ModelRoutingConfig()
    task = PlannedTask(
        id="t1",
        title="Implement feature",
        description="Do the work",
        role="implementer",
    )
    graph = TaskGraph(goal="Ship it", tasks=[task])
    patch = FilePatch(path="feature.py", content="print('ok')", operation="create")
    implementation = ImplementationResult(
        status=ImplementationStatus.IMPLEMENTED,
        summary="done",
        patches=[patch],
    )
    fix = FixResult(status=FixStatus.FIXED, summary="fixed", patches=[patch])
    prd = PRD(title="ACOS", problem_statement="Automate coding")
    architecture = ArchitecturePlan(summary="modular")
    diagnosis = FailureDiagnosis(
        classification="import_error",
        root_cause="main imports Base from the wrong module",
        recommended_fix_strategy="Import Base from models",
        confidence=0.9,
        retry_mode="targeted_fix",
    )
    spec = JobSpec(request_text="build something", repo_path=".")
    packet = ContextPacket(
        job_id=spec.job_id,
        role="implementer",
        objective="Implement task",
        repo_path=".",
        request_text="build something",
        token_budget=4096,
    )

    assert provider.type == ProviderType.OPENAI_COMPATIBLE
    assert model.model_id == "test-model"
    assert agent.role == "implementer"
    assert routing.default_strategy == "role_primary"
    assert graph.tasks[0].id == "t1"
    assert implementation.patches[0].path == "feature.py"
    assert fix.status == FixStatus.FIXED
    assert prd.title == "ACOS"
    assert architecture.summary == "modular"
    assert diagnosis.classification.value == "import_error"
    assert packet.role == "implementer"


def test_invalid_schema_values_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        Finding(
            severity="urgent",
            title="bad",
            description="bad",
        )

    with pytest.raises(ValidationError):
        ModelProviderConfig(
            name="broken",
            type="unknown",
            base_url="http://localhost",
            api_key_env="KEY",
        )

    with pytest.raises(ValidationError):
        FailureDiagnosis(
            classification="network_mystery",
            root_cause="bad",
            recommended_fix_strategy="bad",
            confidence=0.5,
        )


def test_file_patch_accepts_common_model_path_aliases() -> None:
    fix = FixResult.model_validate(
        {
            "status": "fixed",
            "summary": "fixed import",
            "patches": [
                {
                    "file": "backend/main.py",
                    "content": "print('ok')\n",
                    "operation": "update",
                }
            ],
        }
    )

    assert fix.patches[0].path == "backend/main.py"

    patch = FilePatch.model_validate(
        {
            "filename": "backend/models.py",
            "content": "class User: pass\n",
        }
    )

    assert patch.path == "backend/models.py"


def test_file_patch_still_rejects_unknown_extra_keys() -> None:
    with pytest.raises(ValidationError):
        FilePatch.model_validate(
            {
                "path": "feature.py",
                "content": "VALUE = 1\n",
                "unexpected": True,
            }
        )


def test_file_patch_supports_delete_rename_and_integrity_metadata() -> None:
    delete_patch = FilePatch.model_validate(
        {
            "path": "old.py",
            "operation": "delete",
            "base_sha256": "abc123",
        }
    )
    rename_patch = FilePatch.model_validate(
        {
            "path": "old.py",
            "operation": "rename",
            "new_path": "new.py",
            "expected_old_content": "VALUE = 1\n",
        }
    )

    assert delete_patch.content is None
    assert rename_patch.new_path == "new.py"

    with pytest.raises(ValidationError):
        FilePatch.model_validate({"path": "old.py", "operation": "rename"})

    with pytest.raises(ValidationError):
        FilePatch.model_validate({"path": "old.py", "operation": "update"})


def test_prd_captures_strict_incremental_requirements() -> None:
    prd = PRD(
        title="Notes",
        problem_statement="Need a small notes app.",
        smallest_working_core=["Create a note and list notes"],
        small_parts=[
            "Note model",
            "In-memory store",
            "Create/list UI",
            "Toggle completion",
        ],
        incremental_milestones=[
            "Core model passes tests",
            "Create/list workflow passes tests",
            "Polished README exists",
        ],
        acceptance_tests=[
            "Creating a note makes it appear in the list",
            "Toggling a note changes completion state",
        ],
        definition_of_done=["All tests pass", "README explains setup"],
    )

    assert prd.smallest_working_core == ["Create a note and list notes"]
    assert prd.small_parts[0] == "Note model"
    assert "All tests pass" in prd.definition_of_done


def test_planned_task_can_fill_missing_title_from_description() -> None:
    task = PlannedTask.model_validate(
        {
            "id": "setup-django",
            "description": "Create the Django project skeleton.\nAdd settings.",
            "role": "implementer",
        }
    )

    assert task.title == "Create the Django project skeleton."


def test_planned_task_normalizes_common_llm_aliases() -> None:
    task = PlannedTask.model_validate(
        {
            "id": "views",
            "title": "Implement views",
            "instruction": "Add list, create, toggle, and delete views.",
            "role": "implementer",
            "dependencies": ["models"],
            "acceptance_tests": ["Creating an item shows it in the list"],
        }
    )

    assert task.description == "Add list, create, toggle, and delete views."
    assert task.depends_on == ["models"]
    assert task.acceptance_criteria == ["Creating an item shows it in the list"]


def test_context_packet_renders_task_acceptance_criteria() -> None:
    task = PlannedTask(
        id="core",
        title="Create core",
        description="Create the smallest working core.",
        role="implementer",
        acceptance_criteria=["The core behavior can be exercised by one test"],
    )
    packet = ContextPacket(
        job_id="job-1",
        role="test_writer",
        objective="Add focused tests",
        repo_path=".",
        request_text="Build it",
        task=task,
    )

    rendered = packet.render_text()

    assert "acceptance_criteria" in rendered
    assert "The core behavior can be exercised by one test" in rendered
