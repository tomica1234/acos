"""Provider and model health checks for runtime wait/retry."""

from __future__ import annotations

import os
import time
from pathlib import PurePath

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
)

from packages.llm.registry import ModelRegistry
from packages.schemas.models import ModelConfig, ModelProviderConfig
from packages.schemas.runtime import (
    ProviderHealth,
    ProviderHealthCheckConfig,
    ProviderHealthStatus,
)


class ProviderHealthChecker:
    """Check OpenAI-compatible provider reachability."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        config: ProviderHealthCheckConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or ProviderHealthCheckConfig()

    def check_provider(self, provider_key: str) -> ProviderHealth:
        provider = self.registry.get_provider(provider_key)
        started = time.monotonic()
        try:
            client = self._build_client(provider)
            models = client.models.list()
            return ProviderHealth(
                provider_key=provider_key,
                status=ProviderHealthStatus.OK,
                message="provider is reachable",
                response_time_ms=int((time.monotonic() - started) * 1000),
                model_available=bool(getattr(models, "data", [])),
            )
        except Exception as exc:
            return ProviderHealth(
                provider_key=provider_key,
                status=self._classify_exception(exc),
                message=str(exc),
                response_time_ms=int((time.monotonic() - started) * 1000),
                model_available=False,
            )

    def check_model(self, model_key: str) -> ProviderHealth:
        model = self.registry.get_model(model_key)
        provider = self.registry.get_provider(model.provider)
        started = time.monotonic()
        try:
            client = self._build_client(provider)
            models = client.models.list()
            available_ids = {item.id for item in getattr(models, "data", [])}
            acceptable_model_ids = self._acceptable_model_ids(model)
            if available_ids and not (acceptable_model_ids & available_ids):
                return ProviderHealth(
                    provider_key=provider.name,
                    model_key=model_key,
                    status=ProviderHealthStatus.MODEL_NOT_FOUND,
                    message=f"model {model.model} is not reported by provider",
                    response_time_ms=int((time.monotonic() - started) * 1000),
                    model_available=False,
                )
            if self.config.test_chat_completion:
                self._test_chat_completion(client, model)
            return ProviderHealth(
                provider_key=provider.name,
                model_key=model_key,
                status=ProviderHealthStatus.OK,
                message="model is reachable",
                response_time_ms=int((time.monotonic() - started) * 1000),
                model_available=True,
            )
        except Exception as exc:
            return ProviderHealth(
                provider_key=provider.name,
                model_key=model_key,
                status=self._classify_exception(exc),
                message=str(exc),
                response_time_ms=int((time.monotonic() - started) * 1000),
                model_available=False,
            )

    @staticmethod
    def _acceptable_model_ids(model: ModelConfig) -> set[str]:
        model_name = model.model.strip()
        basename = model_name.replace("\\", "/").rsplit("/", 1)[-1]
        stem = PurePath(basename).stem
        return {
            item
            for item in {
                model.model_id,
                model_name,
                basename,
                stem,
            }
            if item
        }

    def _test_chat_completion(self, client: OpenAI, model: ModelConfig) -> None:
        client.chat.completions.create(
            model=model.model,
            messages=[{"role": "user", "content": "Return only OK"}],
            temperature=0,
            max_tokens=8,
        )

    def _build_client(self, provider: ModelProviderConfig) -> OpenAI:
        return OpenAI(
            api_key=self._resolve_api_key(provider),
            base_url=provider.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=0,
            default_headers=provider.default_headers,
        )

    @staticmethod
    def _resolve_api_key(provider: ModelProviderConfig) -> str:
        env_name = provider.api_key_env.strip()
        if env_name:
            value = os.environ.get(env_name)
            if value:
                return value
        if provider.default_api_key is not None:
            return provider.default_api_key
        if provider.allow_empty_api_key:
            return "EMPTY"
        return "EMPTY"

    @staticmethod
    def _classify_exception(exc: Exception) -> ProviderHealthStatus:
        if isinstance(exc, AuthenticationError):
            return ProviderHealthStatus.AUTH_ERROR
        if isinstance(exc, (APITimeoutError, TimeoutError)):
            return ProviderHealthStatus.TIMEOUT
        if isinstance(exc, (APIConnectionError, ConnectionError)):
            return ProviderHealthStatus.CONNECTION_ERROR
        if isinstance(exc, NotFoundError):
            return ProviderHealthStatus.MODEL_NOT_FOUND
        if isinstance(exc, BadRequestError):
            return ProviderHealthStatus.INVALID_RESPONSE
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                return ProviderHealthStatus.AUTH_ERROR
            if status == 404:
                return ProviderHealthStatus.MODEL_NOT_FOUND
            return ProviderHealthStatus.INVALID_RESPONSE
        return ProviderHealthStatus.INVALID_RESPONSE
