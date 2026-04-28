import pytest
from pydantic import ValidationError

from packages.schemas.agent_outputs import (
    ArchitecturePlan,
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
        max_output_tokens="auto",
    )
    agent = AgentModelConfig(
        role="implementer",
        primary_model="test-model",
        max_output_tokens="auto",
        context_budget_tokens=4096,
        output_schema="ImplementationResult",
    )
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
    assert model.max_output_tokens == "auto"
    assert agent.role == "implementer"
    assert agent.max_output_tokens == "auto"
    assert routing.default_strategy == "role_primary"
    assert graph.tasks[0].id == "t1"
    assert implementation.patches[0].path == "feature.py"
    assert fix.status == FixStatus.FIXED
    assert prd.title == "ACOS"
    assert architecture.summary == "modular"
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
