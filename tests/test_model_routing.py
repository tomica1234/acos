import pytest

from packages.llm.errors import ContextBudgetExceededError, RoutingError
from packages.llm.routing import ModelRouter, RoutingContext
from packages.schemas.models import RoutingReason, TaskComplexity

from tests.conftest import load_registry


def test_model_router_default_fallback_escalation_and_budget() -> None:
    registry = load_registry()
    router = ModelRouter(registry)

    default_selection = router.select_model(RoutingContext(role="pm"))
    assert default_selection.model_id == "ornith_35b_q4"
    assert default_selection.reason == RoutingReason.ROLE_DEFAULT

    summarizer_selection = router.select_model(RoutingContext(role="summarizer"))
    assert summarizer_selection.model_id == "ornith_35b_q4"
    assert summarizer_selection.reason == RoutingReason.ROLE_DEFAULT

    fallback_selection = router.select_model(
        RoutingContext(role="fixer", last_error="timeout")
    )
    assert fallback_selection.model_id == "mock_structured"
    assert fallback_selection.reason == RoutingReason.FALLBACK

    implementer_escalation = router.select_model(
        RoutingContext(role="implementer", failure_count=2)
    )
    assert implementer_escalation.model_id == "ncmoe40_q4"
    assert implementer_escalation.reason == RoutingReason.ESCALATION
    assert implementer_escalation.details["repeated_failures"] == 2

    escalation_selection = router.select_model(
        RoutingContext(role="fixer", failure_count=2)
    )
    assert escalation_selection.model_id == "ncmoe40_q4"
    assert escalation_selection.reason == RoutingReason.ESCALATION
    assert escalation_selection.details["repeated_failures"] == 2

    with pytest.raises(RoutingError):
        router.select_model(
            RoutingContext(role="fixer", last_error="invalid_json", fallback_index=1)
        )

    capability_selection = router.select_model(
        RoutingContext(role="fixer", last_error="invalid_json")
    )
    assert capability_selection.model_id == "mock_structured"
    assert capability_selection.reason == RoutingReason.FALLBACK

    security_fallback = router.select_model(
        RoutingContext(
            role="security_reviewer",
            security_sensitive=True,
            last_error="timeout",
            attempted_model_keys=["ornith_35b_q4"],
        )
    )
    assert security_fallback.model_id == "mock_structured"
    assert security_fallback.reason == RoutingReason.FALLBACK

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(
            RoutingContext(role="fixer", context_tokens=140000, task_complexity=TaskComplexity.HIGH)
        )

    with pytest.raises(ContextBudgetExceededError):
        router.select_model(RoutingContext(role="summarizer", context_tokens=400000))
