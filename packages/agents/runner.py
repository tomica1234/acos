"""Generic structured agent runner."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from packages.agents.config import get_role_prompt
from packages.llm.budget import (
    TokenBudgetPolicy,
    compute_max_output_tokens,
    estimate_tokens,
    estimate_tokens_from_messages,
    resolve_configured_max_output_tokens,
)
from packages.llm.client import LLMClient
from packages.llm.errors import AdapterError, ContextBudgetExceededError, StructuredOutputError
from packages.llm.messages import build_messages
from packages.llm.registry import ModelRegistry
from packages.llm.routing import FailureHistory, ModelRouter, RoutingContext, TaskState
from packages.llm.tool_schema import build_response_schema, build_tool_manifest
from packages.mcp_client.router import MCPRouter
from packages.orchestrator.approval import ApprovalRequiredError
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.policy import PolicyEngine
from packages.orchestrator.quality_gates import ensure_test_patch_quality
from packages.schemas.agent_outputs import FilePatch
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
        token_budget_policy: TokenBudgetPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.model_router = model_router or (llm_client.router if llm_client is not None else None)
        if self.model_router is None:
            self.model_router = ModelRouter(
                registry,
                token_budget_policy=token_budget_policy,
            )
        self.mcp_router = mcp_router
        self.policy_engine = policy_engine
        self.audit_recorder = audit_recorder or AuditRecorder()
        self.token_budget_policy = token_budget_policy or TokenBudgetPolicy()
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
        self._assert_requested_tools_subset(role, allowed_tools, agent_config.allowed_tools)
        configured_tools = (
            allowed_tools
            if allowed_tools is not None
            else (agent_config.allowed_tools if agent_config.allow_tools else [])
        )
        self._assert_tools_allowed(role, configured_tools)
        tool_manifest = build_tool_manifest(configured_tools) if configured_tools else None
        response_schema = build_response_schema(response_model) if require_json_schema else None
        messages = build_messages(get_role_prompt(role), context_packet)
        if response_schema is not None:
            messages[0]["content"] = (
                f"{messages[0]['content']}\n\n"
                f"{self._schema_instruction_text(response_schema)}"
            )
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
            budget_details = self._resolve_budget(
                agent_role=role,
                selection=selection,
                messages=messages,
                context_budget_tokens=agent_config.context_budget_tokens,
            )
            try:
                result = adapter.generate(
                    messages=messages,
                    tools=tool_manifest,
                    temperature=selection.temperature,
                    top_p=selection.top_p,
                    max_tokens=budget_details["resolved_max_output_tokens"],
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
                    finish_reason=None,
                    budget_details=budget_details,
                )
                if audit_events is not None:
                    audit_events.append(self.audit_recorder.model_event(record, selection))
                last_selection = selection
                last_record = record
                if selection.model_key not in attempted_model_keys:
                    attempted_model_keys.append(selection.model_key)
                if exc.code == "output_truncated":
                    last_error = "output_truncated before valid JSON"
                    raise StructuredOutputError(last_error) from exc
                if exc.code == "invalid_json" and require_json_schema:
                    if not repair_attempted:
                        repair_attempted = True
                        messages.append(
                            self._repair_message(
                                response_schema=response_schema,
                                error_detail=str(exc),
                            )
                        )
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

            last_selection = selection
            if selection.model_key not in attempted_model_keys:
                attempted_model_keys.append(selection.model_key)

            if result.tool_calls:
                record = self._build_model_record(
                    role=role,
                    selection=selection,
                    messages=messages,
                    result=result,
                    error=None,
                    finish_reason=result.finish_reason,
                    budget_details=budget_details,
                )
                last_record = record
                if audit_events is not None:
                    audit_events.append(self.audit_recorder.model_event(record, selection))
                self._handle_tool_calls(
                    role=role,
                    context_packet=context_packet,
                    messages=messages,
                    tool_calls=result.tool_calls,
                    audit_events=audit_events,
                )
                repair_attempted = False
                last_error = None
                continue

            try:
                parsed = self._parse_response(result.content, response_model)
            except (ValidationError, json.JSONDecodeError, StructuredOutputError) as exc:
                error_code = "invalid_json"
                error_message = "invalid_json"
                if result.output_truncated or result.finish_reason == "length":
                    error_code = "output_truncated"
                    error_message = "output_truncated before valid JSON"
                record = self._build_model_record(
                    role=role,
                    selection=selection,
                    messages=messages,
                    result=result,
                    error=error_code,
                    finish_reason=result.finish_reason,
                    budget_details=budget_details,
                )
                last_record = record
                if audit_events is not None:
                    audit_events.append(self.audit_recorder.model_event(record, selection))
                if not require_json_schema:
                    raise
                if error_code == "output_truncated":
                    last_error = error_message
                    raise StructuredOutputError(error_message) from exc
                if not repair_attempted:
                    repair_attempted = True
                    messages.append(
                        self._repair_message(
                            response_schema=response_schema,
                            error_detail=str(exc),
                        )
                    )
                    continue
                last_error = error_message
                fallback_index = self._next_fallback_index(selection, fallback_index)
                repair_attempted = False
                continue
            record = self._build_model_record(
                role=role,
                selection=selection,
                messages=messages,
                result=result,
                error=None,
                finish_reason=result.finish_reason,
                budget_details=budget_details,
            )
            last_record = record
            if audit_events is not None:
                audit_events.append(self.audit_recorder.model_event(record, selection))
            return parsed, selection, record

        last_model = last_selection.model_key if last_selection is not None else "unknown"
        raise StructuredOutputError(
            f"Agent {role} exceeded max_steps={max_steps} without a valid structured response; "
            f"last_model={last_model}; last_status={last_record.status.value if last_record else 'none'}"
        )

    @staticmethod
    def _schema_instruction_text(response_schema: dict[str, Any]) -> str:
        schema_json = json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
        patch_path_note = ""
        properties = response_schema.get("properties", {})
        if isinstance(properties, dict) and "patches" in properties:
            patch_path_note = (
                " For every item in `patches`, `path` must be a workspace-relative POSIX path "
                "such as `todos/models.py`, never an absolute path."
            )
        return (
            "Return exactly one JSON object that conforms to this schema. "
            "Do not wrap the payload under extra keys. "
            f"Do not include prose, markdown fences, or thinking tags.{patch_path_note}\n"
            f"Schema:\n```json\n{schema_json}\n```"
        )

    @staticmethod
    def _repair_message(
        *,
        response_schema: dict[str, Any] | None,
        error_detail: str,
    ) -> dict[str, str]:
        content = [
            "The previous response was not valid for the required JSON schema.",
            f"Validation/parsing error: {error_detail}",
            "Return only corrected JSON with no prose, markdown fences, or thinking tags.",
        ]
        if response_schema is not None:
            schema_json = json.dumps(response_schema, ensure_ascii=False, sort_keys=True)
            content.append(f"Schema:\n```json\n{schema_json}\n```")
        return {
            "role": "user",
            "content": "\n".join(content),
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
        finish_reason: str | None,
        budget_details: dict[str, Any],
    ) -> ModelCallRecord:
        status = ModelCallStatus.SUCCESS
        if error is not None:
            status = ModelCallStatus.FAILED
        elif selection.reason.value == "fallback":
            status = ModelCallStatus.FALLBACK_USED
        elif selection.reason.value == "escalation":
            status = ModelCallStatus.ESCALATED
        prompt_tokens_estimate = budget_details["estimated_input_tokens"]
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
            finish_reason=finish_reason,
            configured_max_output_tokens=budget_details["configured_max_output_tokens"],
            estimated_input_tokens=budget_details["estimated_input_tokens"],
            resolved_max_output_tokens=budget_details["resolved_max_output_tokens"],
            model_max_context_tokens=budget_details["model_max_context_tokens"],
            safety_margin_tokens=budget_details["safety_margin_tokens"],
            context_budget_tokens=budget_details["context_budget_tokens"],
            output_truncated=(
                bool(result.output_truncated) or finish_reason == "length"
                if result is not None
                else error == "output_truncated"
            ),
        )

    def _resolve_budget(
        self,
        *,
        agent_role: str,
        selection: ModelSelection,
        messages: list[dict[str, Any]],
        context_budget_tokens: int,
    ) -> dict[str, Any]:
        selected_model = self.registry.get_model(selection.model_key)
        configured_max_output_tokens = resolve_configured_max_output_tokens(
            selection.max_output_tokens,
            selected_model.max_output_tokens,
            self.token_budget_policy.default_output_tokens,
        )
        estimated_input_tokens = estimate_tokens_from_messages(messages)
        resolved_max_output_tokens = compute_max_output_tokens(
            model_max_context_tokens=selected_model.max_context_tokens,
            estimated_input_tokens=estimated_input_tokens,
            configured_max_output_tokens=configured_max_output_tokens,
            safety_margin_tokens=self.token_budget_policy.safety_margin_tokens,
            minimum_output_tokens=self.token_budget_policy.minimum_output_tokens,
            hard_max_output_tokens=self.token_budget_policy.hard_max_output_tokens,
        )
        if estimated_input_tokens > context_budget_tokens:
            raise ContextBudgetExceededError(
                f"Context exceeds role budget for {agent_role}",
                required_tokens=estimated_input_tokens,
                candidate_model_keys=[selection.model_key],
            )
        return {
            "configured_max_output_tokens": configured_max_output_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "resolved_max_output_tokens": resolved_max_output_tokens,
            "model_max_context_tokens": selected_model.max_context_tokens,
            "safety_margin_tokens": self.token_budget_policy.safety_margin_tokens,
            "context_budget_tokens": context_budget_tokens,
        }

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
        context_packet: ContextPacket,
        messages: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        audit_events: list[AuditEvent] | None,
    ) -> None:
        if self.mcp_router is None:
            raise RuntimeError("MCP router is not configured for tool calls")
        serialized_tool_calls: list[dict[str, Any]] = []
        resolved_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
        for index, tool_call in enumerate(tool_calls):
            tool_name = str(tool_call["name"])
            tool_call_id = str(tool_call.get("id") or f"tool_call_{index}")
            self._assert_tools_allowed(role, [tool_name])
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            resolved_tool_calls.append((tool_call_id, tool_name, arguments))
            serialized_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, sort_keys=True, default=str),
                    },
                }
            )
        messages.append(
            {"role": "assistant", "tool_calls": serialized_tool_calls, "content": None}
        )
        for tool_call_id, tool_name, arguments in resolved_tool_calls:
            if self.policy_engine is not None:
                decision = self.policy_engine.classify_tool_call(
                    role=role,
                    tool_name=tool_name,
                    arguments=arguments,
                    workspace_root=context_packet.repo_path,
                )
                if audit_events is not None:
                    audit_events.append(
                        self.audit_recorder.policy_event(
                            role=role,
                            job_id=context_packet.job_id,
                            task_id=context_packet.task.id if context_packet.task else None,
                            decision=decision,
                        )
                    )
                if decision.policy_action.value == "deny":
                    raise PermissionError(decision.reason)
                if decision.policy_action.value == "require_approval":
                    raise ApprovalRequiredError(
                        requested_by=role,
                        operation=decision.operation,
                        decision=decision,
                        proposed_action={
                            "tool_name": tool_name,
                            "arguments": arguments,
                        },
                        task_id=context_packet.task.id if context_packet.task else None,
                    )
            if tool_name == "repo_server.apply_patch" and self.policy_engine is not None:
                path = arguments.get("path")
                if not isinstance(path, str):
                    raise PermissionError("repo_server.apply_patch requires a string path")
                self.policy_engine.assert_patch_target_allowed(role, path)
                if role == "test_writer":
                    ensure_test_patch_quality(
                        [
                            FilePatch(
                                path=path,
                                content=str(arguments.get("content", "")),
                                operation=str(arguments.get("operation", "update")),
                            )
                        ],
                        role=role,
                    )
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
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result.data, sort_keys=True, default=str),
                }
            )

    def _assert_tools_allowed(self, role: str, tool_names: list[str]) -> None:
        if self.policy_engine is None:
            return
        for tool_name in tool_names:
            self.policy_engine.assert_tool_allowed(role, tool_name)

    @staticmethod
    def _assert_requested_tools_subset(
        role: str,
        requested_tools: list[str] | None,
        configured_tools: list[str],
    ) -> None:
        if requested_tools is None:
            return
        unexpected = sorted(set(requested_tools) - set(configured_tools))
        if unexpected:
            raise PermissionError(
                f"Role {role} requested tools outside agent config: {', '.join(unexpected)}"
            )

    def _parse_response(self, content: str, response_model: type[T]) -> T:
        return response_model.model_validate(_extract_json(content))
