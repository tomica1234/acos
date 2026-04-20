"""Errors raised by the ACOS model layer."""


class LLMError(Exception):
    """Base class for LLM layer errors."""


class UnknownProviderError(LLMError):
    """Raised when a provider name is not configured."""


class UnknownModelError(LLMError):
    """Raised when a model id is not configured."""


class UnknownRoleError(LLMError):
    """Raised when an agent role is not configured."""


class ConfigValidationError(LLMError):
    """Raised when model, routing, or policy configuration is inconsistent."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


class AdapterError(LLMError):
    """Raised when an adapter call fails."""

    def __init__(self, message: str, code: str = "adapter_error") -> None:
        super().__init__(message)
        self.code = code


class StructuredOutputError(LLMError):
    """Raised when structured output cannot be parsed."""


class RoutingError(LLMError):
    """Raised when routing cannot satisfy constraints."""


class ContextBudgetExceededError(RoutingError):
    """Raised when no model can fit the requested context."""

    def __init__(
        self,
        message: str,
        *,
        required_tokens: int | None = None,
        candidate_model_keys: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.required_tokens = required_tokens
        self.candidate_model_keys = candidate_model_keys or []
