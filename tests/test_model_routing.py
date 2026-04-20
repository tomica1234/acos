import pytest

from packages.llm.errors import ContextBudgetExceededError
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
    assert summarizer_selection.model_id == "qwen_small"
    assert summarizer_selection.reason == RoutingReason.ROLE_DEFAULT

    fallback_selection = router.select_model(
        RoutingContext(role="fixer", last_error="timeout")
    )
    assert fallback_selection.model_id == "qwen_35b"
    assert fallback_selection.reason == RoutingReason.FALLBACK

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

    fallback_second_selection = router.select_model(
        RoutingContext(role="fixer", last_error="invalid_json", fallback_index=1)
    )
    assert fallback_second_selection.model_id == "qwen_small"
    assert fallback_second_selection.reason == RoutingReason.FALLBACK

    capability_selection = router.select_model(
        RoutingContext(role="fixer", last_error="invalid_json")
    )
    assert capability_selection.model_id == "qwen_35b"
    assert capability_selection.reason == RoutingReason.FALLBACK

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(
            RoutingContext(role="fixer", context_tokens=100000, task_complexity=TaskComplexity.HIGH)
        )

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(RoutingContext(role="summarizer", context_tokens=400000))
