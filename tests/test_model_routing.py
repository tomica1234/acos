import pytest

from packages.llm.errors import ContextBudgetExceededError, RoutingError
from packages.llm.routing import ModelRouter, RoutingContext
from packages.schemas.models import RoutingReason, TaskComplexity

from tests.conftest import load_registry


def test_model_router_default_fallback_escalation_and_budget() -> None:
    registry = load_registry()
    router = ModelRouter(registry)

    default_selection = router.select_model(RoutingContext(role="pm"))
    assert default_selection.model_id == "qwen_35b"
    assert default_selection.reason == RoutingReason.ROLE_DEFAULT

    summarizer_selection = router.select_model(RoutingContext(role="summarizer"))
    assert summarizer_selection.model_id == "qwen_35b"
    assert summarizer_selection.reason == RoutingReason.ROLE_DEFAULT

    implementer_escalation = router.select_model(
        RoutingContext(role="implementer", failure_count=2)
    )
    assert implementer_escalation.model_id == "qwen_35b"
    assert implementer_escalation.reason == RoutingReason.ESCALATION
    assert implementer_escalation.details["repeated_failures"] == 2

    escalation_selection = router.select_model(
        RoutingContext(role="fixer", failure_count=2)
    )
    assert escalation_selection.model_id == "qwen_35b"
    assert escalation_selection.reason == RoutingReason.ESCALATION
    assert escalation_selection.details["repeated_failures"] == 2

    with pytest.raises(
        RoutingError,
        match=r"Fallbacks exhausted for role fixer after timeout; attempted models: none",
    ):
        router.select_model(RoutingContext(role="fixer", last_error="timeout"))

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(
            RoutingContext(role="fixer", context_tokens=400000, task_complexity=TaskComplexity.HIGH)
        )

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(RoutingContext(role="summarizer", context_tokens=400000))


def test_model_router_reports_exhausted_fallback_cause() -> None:
    registry = load_registry()
    router = ModelRouter(registry)

    with pytest.raises(
        RoutingError,
        match=(
            r"Fallbacks exhausted for role pm after timeout; "
            r"attempted models: qwen_35b"
        ),
    ):
        router.select_model(
            RoutingContext(
                role="pm",
                last_error="timeout",
                attempted_model_keys=["qwen_35b"],
            )
        )
