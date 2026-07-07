import pytest

from packages.llm.budget import compute_max_output_tokens
from packages.llm.errors import ContextBudgetExceededError


def test_compute_max_output_tokens_uses_remaining_context_for_auto() -> None:
    resolved = compute_max_output_tokens(
        model_max_context_tokens=10000,
        estimated_input_tokens=1000,
        configured_max_output_tokens="auto",
        safety_margin_tokens=500,
        minimum_output_tokens=256,
    )

    assert resolved == 8500


def test_compute_max_output_tokens_uses_min_of_fixed_value_and_remaining_context() -> None:
    resolved = compute_max_output_tokens(
        model_max_context_tokens=10000,
        estimated_input_tokens=3000,
        configured_max_output_tokens=8000,
        safety_margin_tokens=500,
        minimum_output_tokens=256,
    )

    assert resolved == 6500


def test_compute_max_output_tokens_respects_hard_max() -> None:
    resolved = compute_max_output_tokens(
        model_max_context_tokens=10000,
        estimated_input_tokens=1000,
        configured_max_output_tokens="auto",
        safety_margin_tokens=500,
        minimum_output_tokens=256,
        hard_max_output_tokens=2048,
    )

    assert resolved == 2048


def test_compute_max_output_tokens_raises_when_input_is_too_large() -> None:
    with pytest.raises(ContextBudgetExceededError):
        compute_max_output_tokens(
            model_max_context_tokens=4096,
            estimated_input_tokens=3500,
            configured_max_output_tokens="auto",
            safety_margin_tokens=512,
            minimum_output_tokens=256,
        )


def test_compute_max_output_tokens_never_overflows_model_context() -> None:
    estimated_input_tokens = 12000
    safety_margin_tokens = 4096
    resolved = compute_max_output_tokens(
        model_max_context_tokens=262144,
        estimated_input_tokens=estimated_input_tokens,
        configured_max_output_tokens="auto",
        safety_margin_tokens=safety_margin_tokens,
        minimum_output_tokens=1024,
    )

    assert estimated_input_tokens + resolved + safety_margin_tokens <= 262144
