"""Role-aware model routing with fallback and escalation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.llm.errors import ContextBudgetExceededError, RoutingError
from packages.llm.budget import (
    TokenBudgetManager,
    TokenBudgetPolicy,
    estimate_tokens,
    resolve_configured_max_output_tokens,
)
from packages.llm.registry import ModelRegistry
from packages.schemas.context import ContextPacket
from packages.schemas.models import (
    AgentModelConfig,
    ModelConfig,
    ModelSelection,
    RoutingReason,
    TaskComplexity,
)


@dataclass(slots=True)
class RoutingContext:
    role: str
    task_complexity: TaskComplexity = TaskComplexity.MEDIUM
    failure_count: int = 0
    same_test_failure_count: int = 0
    changed_files_count: int = 0
    security_sensitive: bool = False
    context_tokens: int = 0
    last_error: str | None = None
    fallback_index: int = 0
    forced_model_key: str | None = None
    attempted_model_keys: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskState:
    changed_files_count: int = 0
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FailureHistory:
    repeated_failures: int = 0
    same_test_failure_repeats: int = 0
    fallback_attempts: int = 0
    last_error: str | None = None
    attempted_model_keys: list[str] = field(default_factory=list)


class ModelRouter:
    """Select models for agent invocations."""

    def __init__(
        self,
        registry: ModelRegistry,
        token_budget_policy: TokenBudgetPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.token_budget_policy = token_budget_policy or TokenBudgetPolicy()
        self.budget_manager = TokenBudgetManager(self.token_budget_policy)

    def select_model(
        self,
        role: str | RoutingContext,
        context_packet: ContextPacket | None = None,
        task_state: TaskState | None = None,
        failure_history: FailureHistory | None = None,
    ) -> ModelSelection:
        routing_context = self._coerce_routing_context(
            role=role,
            context_packet=context_packet,
            task_state=task_state,
            failure_history=failure_history,
        )
        agent = self.registry.get_agent(routing_context.role)
        candidate_keys, reason, details = self._ordered_candidates(agent, routing_context)
        valid_models = self._filter_capabilities(
            role=routing_context.role,
            agent=agent,
            candidate_keys=candidate_keys,
        )
        if not valid_models:
            raise RoutingError(f"No valid model candidates remain for role {routing_context.role}")
        selected = self._pick_by_context_budget(
            agent=agent,
            routing_context=routing_context,
            valid_models=valid_models,
            reason=reason,
            details=details,
        )
        return selected

    def explain_routing(
        self,
        role: str | RoutingContext,
        context_packet: ContextPacket | None = None,
        task_state: TaskState | None = None,
        failure_history: FailureHistory | None = None,
    ) -> dict[str, Any]:
        routing_context = self._coerce_routing_context(
            role=role,
            context_packet=context_packet,
            task_state=task_state,
            failure_history=failure_history,
        )
        agent = self.registry.get_agent(routing_context.role)
        selection = self.select_model(
            role=routing_context,
            context_packet=context_packet,
            task_state=task_state,
            failure_history=failure_history,
        )
        escalation = self.registry.routing.escalation.get(routing_context.role)
        escalation_config: dict[str, Any] | None = None
        if escalation is not None:
            escalation_config = {
                "escalated_model": escalation.escalated_model,
                "conditions": escalation.escalate_when.model_dump(mode="json"),
            }
        return {
            "role": routing_context.role,
            "selection": selection.model_dump(mode="json"),
            "primary_model": agent.primary_model,
            "fallback_models": list(agent.fallback_models),
            "fallback_errors": list(self.registry.routing.fallback.on_errors),
            "escalation": escalation_config,
            "capability_requirements": {
                "requires_tools": routing_context.role
                in self.registry.routing.capability_requirements.roles_requiring_tools,
                "requires_strict_json": routing_context.role
                in self.registry.routing.capability_requirements.roles_requiring_strict_json,
            },
            "routing_context": {
                "task_complexity": routing_context.task_complexity.value,
                "repeated_failures": routing_context.failure_count,
                "same_test_failure_repeats": routing_context.same_test_failure_count,
                "changed_files_count": routing_context.changed_files_count,
                "security_sensitive": routing_context.security_sensitive,
                "context_tokens": routing_context.context_tokens,
                "last_error": routing_context.last_error,
                "fallback_index": routing_context.fallback_index,
                "attempted_model_keys": list(routing_context.attempted_model_keys),
            },
        }

    def _maybe_escalate(self, context: RoutingContext) -> str | None:
        config = self.registry.routing.escalation.get(context.role)
        if config is None:
            return None
        rules = config.escalate_when
        if rules.repeated_failures_gte is not None and context.failure_count >= rules.repeated_failures_gte:
            return config.escalated_model
        if (
            rules.same_test_failure_gte is not None
            and context.same_test_failure_count >= rules.same_test_failure_gte
        ):
            return config.escalated_model
        if (
            rules.changed_files_gte is not None
            and context.changed_files_count >= rules.changed_files_gte
        ):
            return config.escalated_model
        if rules.task_complexity_in and context.task_complexity in rules.task_complexity_in:
            return config.escalated_model
        if rules.security_sensitive and context.security_sensitive:
            return config.escalated_model
        return None

    def _ordered_candidates(
        self, agent: AgentModelConfig, context: RoutingContext
    ) -> tuple[list[str], RoutingReason, dict[str, Any]]:
        if context.forced_model_key is not None:
            return (
                [context.forced_model_key],
                RoutingReason.ROLE_DEFAULT,
                {"forced_model_key": context.forced_model_key},
            )
        escalated_model = self._maybe_escalate(context)
        if escalated_model is not None:
            return (
                [escalated_model, *agent.fallback_models],
                RoutingReason.ESCALATION,
                {
                    "repeated_failures": context.failure_count,
                    "same_test_failure_repeats": context.same_test_failure_count,
                    "changed_files_count": context.changed_files_count,
                    "task_complexity": context.task_complexity.value,
                    "security_sensitive": context.security_sensitive,
                },
            )
        if self._needs_fallback(context):
            fallback_candidates = [
                key for key in agent.fallback_models if key not in context.attempted_model_keys
            ]
            if context.fallback_index >= len(fallback_candidates):
                attempted_models = list(dict.fromkeys(context.attempted_model_keys))
                attempted_label = ", ".join(attempted_models) if attempted_models else "none"
                error_label = f" after {context.last_error}" if context.last_error else ""
                raise RoutingError(
                    f"Fallbacks exhausted for role {context.role}{error_label}; "
                    f"attempted models: {attempted_label}"
                )
            fallback_model = fallback_candidates[context.fallback_index]
            return (
                [fallback_model, *fallback_candidates[context.fallback_index + 1 :]],
                RoutingReason.FALLBACK,
                {
                    "last_error": context.last_error,
                    "fallback_index": context.fallback_index,
                    "attempted_model_keys": context.attempted_model_keys,
                },
            )
        return (
            [agent.primary_model, *agent.fallback_models],
            RoutingReason.ROLE_DEFAULT,
            {"primary_model": agent.primary_model},
        )

    def _filter_capabilities(
        self,
        *,
        role: str,
        agent: AgentModelConfig,
        candidate_keys: list[str],
    ) -> list[ModelConfig]:
        valid: list[ModelConfig] = []
        for model_key in candidate_keys:
            model = self.registry.get_model(model_key)
            provider = self.registry.get_provider(model.provider)
            if self._requires_tool_support(role, agent) and not (
                model.supports_tool_calling and provider.supports_tools
            ):
                continue
            if self._requires_strict_json(role) and not (
                model.supports_structured_output or model.supports_json_repair
            ):
                continue
            valid.append(model)
        return valid

    def _pick_by_context_budget(
        self,
        *,
        agent: AgentModelConfig,
        routing_context: RoutingContext,
        valid_models: list[ModelConfig],
        reason: RoutingReason,
        details: dict[str, Any],
    ) -> ModelSelection:
        for model in valid_models:
            configured_max_output_tokens = resolve_configured_max_output_tokens(
                agent.max_output_tokens,
                model.max_output_tokens,
                self.token_budget_policy.default_output_tokens,
            )
            try:
                self.budget_manager.assert_context_fits(
                    context_tokens=routing_context.context_tokens,
                    requested_budget=agent.context_budget_tokens,
                    model_max_context_tokens=model.max_context_tokens,
                    configured_max_output_tokens=configured_max_output_tokens,
                )
            except ContextBudgetExceededError:
                continue
            actual_reason = reason
            actual_details = dict(details)
            if model.model_id != valid_models[0].model_id and reason == RoutingReason.ROLE_DEFAULT:
                actual_reason = RoutingReason.CAPABILITY_REQUIRED
                actual_details["switched_to_model"] = model.model_id
            return ModelSelection(
                role=routing_context.role,
                model_key=model.model_id,
                provider_key=model.provider,
                reason=actual_reason,
                details=actual_details,
                temperature=agent.temperature,
                top_p=agent.top_p,
                max_output_tokens=configured_max_output_tokens
                if configured_max_output_tokens is not None
                else self.token_budget_policy.default_output_tokens,
            )
        raise ContextBudgetExceededError(
            f"Context requires compaction for role {routing_context.role}",
            required_tokens=routing_context.context_tokens,
            candidate_model_keys=[model.model_id for model in valid_models],
        )

    def _requires_tool_support(self, role: str, agent: AgentModelConfig) -> bool:
        return agent.allow_tools and (
            role in self.registry.routing.capability_requirements.roles_requiring_tools
        )

    def _requires_strict_json(self, role: str) -> bool:
        return role in self.registry.routing.capability_requirements.roles_requiring_strict_json

    def _needs_fallback(self, context: RoutingContext) -> bool:
        return (
            context.last_error is not None
            and context.last_error in self.registry.routing.fallback.on_errors
        )

    def _coerce_routing_context(
        self,
        *,
        role: str | RoutingContext,
        context_packet: ContextPacket | None,
        task_state: TaskState | None,
        failure_history: FailureHistory | None,
    ) -> RoutingContext:
        if isinstance(role, RoutingContext):
            return role
        return RoutingContext(
            role=role,
            task_complexity=(
                task_state.complexity if task_state is not None else TaskComplexity.MEDIUM
            ),
            failure_count=(
                failure_history.repeated_failures if failure_history is not None else 0
            ),
            same_test_failure_count=(
                failure_history.same_test_failure_repeats if failure_history is not None else 0
            ),
            changed_files_count=(
                task_state.changed_files_count if task_state is not None else 0
            ),
            security_sensitive=bool(
                task_state.metadata.get("security_sensitive", False)
                if task_state is not None
                else False
            ),
            context_tokens=(
                estimate_tokens(context_packet.render_text()) if context_packet is not None else 0
            ),
            last_error=failure_history.last_error if failure_history is not None else None,
            fallback_index=failure_history.fallback_attempts if failure_history is not None else 0,
            attempted_model_keys=(
                list(failure_history.attempted_model_keys)
                if failure_history is not None
                else []
            ),
        )
