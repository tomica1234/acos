"""Generic structured agent runner."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from packages.agents.config import get_role_prompt
from packages.llm.budget import estimate_tokens
from packages.llm.client import LLMClient
from packages.llm.errors import AdapterError, StructuredOutputError
from packages.llm.messages import build_messages
from packages.llm.registry import ModelRegistry
from packages.llm.routing import FailureHistory, ModelRouter, RoutingContext, TaskState
from packages.llm.tool_schema import build_response_schema, build_tool_manifest
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.audit import AuditEvent
from packages.schemas.context import ContextPacket
from packages.schemas.models import (
    ModelCallRecord,
    ModelCallStatus,
    ModelResult,
    ModelSelection,
)

T = TypeVar("T", bound=BaseModel)


def _extract_json(content: str) -> dict[str, Any]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise StructuredOutputError("No JSON object found in model response")
    return json.loads(content[start : end + 1])


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return sha256(text.encode("utf-8")).hexdigest()


class AgentRunner:
    """Run a role and validate the response model."""

    def __init__(
        self,
        registry: ModelRegistry,
        model_router: ModelRouter | None = None,
        mcp_router: MCPRouter | None = None,
        policy_engine: PolicyEngine | None = None,
        audit_recorder: AuditRecorder | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.model_router = model_router or (llm_client.router if llm_client is not None else None)
        if self.model_router is None:
            self.model_router = ModelRouter(registry)
        self.mcp_router = mcp_router
        self.policy_engine = policy_engine
        self.audit_recorder = audit_recorder or AuditRecorder()
        self._adapter_cache: dict[str, Any] = {}

    def run(
        self,
        role: str,
        response_model: type[T],
        context_packet: ContextPacket,
        routing_context: RoutingContext | None = None,
        task_state: TaskState | None = None,
        failure_history: FailureHistory | None = None,
        allowed_tools: list[str] | None = None,
        require_json_schema: bool = True,
        max_steps: int = 6,
        audit_events: list[AuditEvent] | None = None,
    ) -> tuple[T, ModelSelection, ModelCallRecord]:
        agent_config = self.registry.get_agent(role)
        configured_tools = (
            allowed_tools
            if allowed_tools is not None
            else (agent_config.allowed_tools if agent_config.allow_tools else [])
        )
        self._assert_tools_allowed(role, configured_tools)
        tool_manifest = build_tool_manifest(configured_tools) if configured_tools else None
        messages = build_messages(get_role_prompt(role), context_packet)
        response_schema = build_response_schema(response_model) if require_json_schema else None
        base_context = self._initial_routing_context(
            role=role,
            routing_context=routing_context,
            task_state=task_state,
            failure_history=failure_history,
            context_packet=context_packet,
        )

        last_error: str | None = base_context.last_error
        fallback_index = base_context.fallback_index
        attempted_model_keys: list[str] = list(base_context.attempted_model_keys)
        repair_attempted = False
        last_selection: ModelSelection | None = None
        last_record: ModelCallRecord | None = None

        for _ in range(max_steps):
            step_context = RoutingContext(
                role=role,
                task_complexity=base_context.task_complexity,
                failure_count=base_context.failure_count,
                same_test_failure_count=base_context.same_test_failure_count,
                changed_files_count=base_context.changed_files_count,
                security_sensitive=base_context.security_sensitive,
                context_tokens=base_context.context_tokens,
                last_error=last_error,
                fallback_index=fallback_index,
                forced_model_key=(
                    last_selection.model_key
                    if repair_attempted and last_selection is not None
                    else None
                ),
                attempted_model_keys=list(attempted_model_keys),
            )
            selection = self.model_router.select_model(step_context)
            if audit_events is not None:
                audit_events.append(self.audit_recorder.selection_event(role, selection))
            adapter = self._get_adapter(selection.model_key)
            try:
                result = adapter.generate(
                    messages=messages,
                    tools=tool_manifest,
                    temperature=selection.temperature,
                    top_p=selection.top_p,
                    max_tokens=selection.max_output_tokens,
                    response_schema=response_schema,
                    metadata={
                        "role": role,
                        "model_key": selection.model_key,
                        "model_name": self.registry.get_model(selection.model_key).model,
                        "provider_name": selection.provider_key,
                    },
                )
            except AdapterError as exc:
                record = self._build_model_record(
                    role=role,
                    selection=selection,
                    messages=messages,
                    result=None,
                    error=exc.code,
                )
                if audit_events is not None:
                    audit_events.append(self.audit_recorder.model_event(record, selection))
                last_selection = selection
                last_record = record
                if selection.model_key not in attempted_model_keys:
                    attempted_model_keys.append(selection.model_key)
                if exc.code == "invalid_json" and require_json_schema:
                    if not repair_attempted:
                        repair_attempted = True
                        messages.append(self._repair_message())
                        continue
                    last_error = "invalid_json"
                    fallback_index = self._next_fallback_index(selection, fallback_index)
                    repair_attempted = False
                    continue
                if exc.code in self.registry.routing.fallback.on_errors:
                    last_error = exc.code
                    fallback_index = self._next_fallback_index(selection, fallback_index)
                    repair_attempted = False
                    continue
                raise

            record = self._build_model_record(
                role=role,
                selection=selection,
                messages=messages,
                result=result,
                error=None,
            )
            last_selection = selection
            last_record = record
            if selection.model_key not in attempted_model_keys:
                attempted_model_keys.append(selection.model_key)
            if audit_events is not None:
                audit_events.append(self.audit_recorder.model_event(record, selection))

            if result.tool_calls:
                self._handle_tool_calls(
                    role=role,
                    messages=messages,
                    tool_calls=result.tool_calls,
                    audit_events=audit_events,
                )
                repair_attempted = False
                last_error = None
                continue

            try:
                parsed = self._parse_response(result.content, response_model)
            except (ValidationError, json.JSONDecodeError, StructuredOutputError):
                if not require_json_schema:
                    raise
                if not repair_attempted:
                    repair_attempted = True
                    messages.append(self._repair_message())
                    continue
                last_error = "invalid_json"
                fallback_index = self._next_fallback_index(selection, fallback_index)
                repair_attempted = False
                continue
            return parsed, selection, record

        last_model = last_selection.model_key if last_selection is not None else "unknown"
        raise StructuredOutputError(
            f"Agent {role} exceeded max_steps={max_steps} without a valid structured response; "
            f"last_model={last_model}; last_status={last_record.status.value if last_record else 'none'}"
        )

    @staticmethod
    def _repair_message() -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "The previous response was not valid JSON for the required schema. "
                "Return only repaired JSON that conforms exactly to the schema."
            ),
        }

    def _initial_routing_context(
        self,
        *,
        role: str,
        routing_context: RoutingContext | None,
        task_state: TaskState | None,
        failure_history: FailureHistory | None,
        context_packet: ContextPacket,
    ) -> RoutingContext:
        if routing_context is not None:
            return routing_context
        return RoutingContext(
            role=role,
            task_complexity=(
                task_state.complexity
                if task_state is not None
                else RoutingContext(role=role).task_complexity
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
            context_tokens=estimate_tokens(context_packet.render_text()),
            last_error=failure_history.last_error if failure_history is not None else None,
            fallback_index=(
                failure_history.fallback_attempts if failure_history is not None else 0
            ),
            attempted_model_keys=(
                list(failure_history.attempted_model_keys)
                if failure_history is not None
                else []
            ),
        )

    def _build_model_record(
        self,
        *,
        role: str,
        selection: ModelSelection,
        messages: list[dict[str, Any]],
        result: ModelResult | None,
        error: str | None,
    ) -> ModelCallRecord:
        status = ModelCallStatus.SUCCESS
        if error is not None:
            status = ModelCallStatus.FAILED
        elif selection.reason.value == "fallback":
            status = ModelCallStatus.FALLBACK_USED
        elif selection.reason.value == "escalation":
            status = ModelCallStatus.ESCALATED
        prompt_tokens_estimate = sum(len(str(item)) for item in messages) // 4
        completion_tokens_estimate = (
            result.usage.get("completion_tokens", len(result.content) // 4)
            if result is not None and result.usage is not None
            else (len(result.content) // 4 if result is not None else 0)
        )
        total_tokens_estimate = (
            result.usage.get("total_tokens", prompt_tokens_estimate + completion_tokens_estimate)
            if result is not None and result.usage is not None
            else prompt_tokens_estimate + completion_tokens_estimate
        )
        output_payload: dict[str, Any] = (
            result.model_dump() if result is not None else {"error": error or "adapter_error"}
        )
        return ModelCallRecord(
            role=role,
            model_key=selection.model_key,
            provider_key=selection.provider_key,
            status=status,
            input_hash=_hash_payload(messages),
            output_hash=_hash_payload(output_payload),
            prompt_tokens_estimate=prompt_tokens_estimate,
            completion_tokens_estimate=completion_tokens_estimate,
            total_tokens_estimate=total_tokens_estimate,
            error=error,
        )

    def _get_adapter(self, model_key: str) -> Any:
        if model_key not in self._adapter_cache:
            self._adapter_cache[model_key] = self.registry.build_adapter(model_key)
        return self._adapter_cache[model_key]

    @staticmethod
    def _next_fallback_index(selection: ModelSelection, current_index: int) -> int:
        if selection.reason.value == "fallback":
            return current_index + 1
        return 0

    def _handle_tool_calls(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        audit_events: list[AuditEvent] | None,
    ) -> None:
        if self.mcp_router is None:
            raise RuntimeError("MCP router is not configured for tool calls")
        messages.append({"role": "assistant", "tool_calls": tool_calls, "content": ""})
        for tool_call in tool_calls:
            tool_name = str(tool_call["name"])
            self._assert_tools_allowed(role, [tool_name])
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            result = self.mcp_router.call(tool_name, **arguments)
            event = self.audit_recorder.tool_event(
                role=role,
                tool_name=tool_name,
                input_payload=arguments,
                output_payload=result.data,
                status="success" if result.ok else "failed",
            )
            if audit_events is not None:
                audit_events.append(event)
            if not result.ok:
                raise RuntimeError(result.error or f"tool call failed: {tool_name}")
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result.data, sort_keys=True, default=str),
                }
            )

    def _assert_tools_allowed(self, role: str, tool_names: list[str]) -> None:
        if self.policy_engine is None:
            return
        for tool_name in tool_names:
            self.policy_engine.assert_tool_allowed(role, tool_name)

    def _parse_response(self, content: str, response_model: type[T]) -> T:
        return response_model.model_validate(_extract_json(content))
