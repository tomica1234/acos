import json

import pytest

from packages.agents.runner import AgentRunner
from packages.llm.errors import RoutingError, StructuredOutputError
from packages.llm.routing import FailureHistory, ModelRouter, RoutingContext, TaskState
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import FixResult, ImplementationResult, PRD
from packages.schemas.context import ContextPacket
from packages.schemas.models import (
    FixStatus,
    ImplementationStatus,
    ModelResult,
    ProviderType,
    TaskComplexity,
)

from tests.conftest import attach_mock_adapter, config_dir, load_registry


def _packet(role: str, repo_path: str = ".") -> ContextPacket:
    return ContextPacket(
        job_id=f"job-{role}",
        role=role,
        objective=f"Run {role}",
        repo_path=repo_path,
        request_text="Build ACOS with api_key=sk-this-should-not-appear",
        token_budget=4096,
        model_context_budget=4096,
        selected_model_hint="auto",
    )


def _runner(registry, workspace_root: str = ".", memory_db_path=":memory:"):
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")
    environment = FakeMCPEnvironment(
        workspace_root=workspace_root,
        memory_db_path=memory_db_path,
    )
    runner = AgentRunner(
        registry=registry,
        model_router=ModelRouter(registry),
        mcp_router=environment.build_router(),
        policy_engine=policy,
        audit_recorder=AuditRecorder(),
    )
    return runner, environment


def test_agent_runner_uses_role_primary_model_and_records_audit() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "pm": PRD(
                title="ACOS",
                problem_statement="Automate product delivery",
                goals=["Generate structured outputs"],
            ).model_dump()
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, selection, record = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        allowed_tools=["memory_server.write_memory"],
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.title == "ACOS"
    assert selection.model_key == "qwen_35b"
    assert record.role == "pm"

    selection_event = next(event for event in audit_events if event.event_type == "model_selection")
    model_event = next(event for event in audit_events if event.event_type == "model_call")
    assert selection_event.metadata["model_key"] == "qwen_35b"
    assert model_event.metadata["model_key"] == "qwen_35b"
    assert model_event.metadata["provider_key"] == "local_qwen"
    assert model_event.metadata["routing_reason"] == "role_default"
    assert model_event.metadata["configured_max_output_tokens"] == "auto"
    assert isinstance(model_event.metadata["resolved_max_output_tokens"], int)
    assert model_event.metadata["resolved_max_output_tokens"] > 0
    assert isinstance(model_event.metadata["estimated_input_tokens"], int)

    serialized_audit = json.dumps([event.model_dump(mode="json") for event in audit_events])
    assert "sk-this-should-not-appear" not in serialized_audit


def test_agent_runner_raises_when_timeout_requires_missing_fallback() -> None:
    registry = load_registry()
    runner, _ = _runner(registry)
    with pytest.raises(
        RoutingError,
        match=r"Fallbacks exhausted for role fixer after timeout; attempted models: none",
    ):
        runner.run(
            role="fixer",
            response_model=FixResult,
            context_packet=_packet("fixer"),
            routing_context=RoutingContext(role="fixer", last_error="timeout"),
            require_json_schema=True,
        )


def test_agent_runner_resolves_auto_max_output_tokens_before_adapter_call() -> None:
    registry = load_registry()

    class RecordingAdapter:
        def __init__(self) -> None:
            self.max_tokens: list[object] = []

        def generate(self, **kwargs):
            self.max_tokens.append(kwargs["max_tokens"])
            return ModelResult(
                content=json.dumps(
                    {
                        "title": "ACOS",
                        "problem_statement": "Automate product delivery",
                        "goals": ["Generate structured outputs"],
                    }
                ),
                tool_calls=[],
                raw={"mock": True},
                model="qwen_35b",
                provider="local_qwen",
                finish_reason="stop",
                usage={"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160},
            )

    adapter = RecordingAdapter()
    registry.register_adapter_factory(
        ProviderType.OPENAI_COMPATIBLE,
        lambda provider, model: adapter,
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, _, _ = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        allowed_tools=["memory_server.write_memory"],
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.title == "ACOS"
    assert adapter.max_tokens
    assert isinstance(adapter.max_tokens[0], int)
    assert adapter.max_tokens[0] > 0
    assert "auto" not in adapter.max_tokens
    model_event = next(event for event in audit_events if event.event_type == "model_call")
    assert model_event.metadata["resolved_max_output_tokens"] == adapter.max_tokens[0]


def test_agent_runner_repairs_invalid_json_once_on_same_model() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "implementer": [
                "{not valid json",
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Implemented after repair",
                    patches=[],
                ).model_dump(),
            ]
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, selection, _ = runner.run(
        role="implementer",
        response_model=ImplementationResult,
        context_packet=_packet("implementer"),
        routing_context=RoutingContext(role="implementer"),
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Implemented after repair"
    assert selection.model_key == "qwen_35b"
    model_calls = [event for event in audit_events if event.event_type == "model_call"]
    assert len(model_calls) == 2
    assert [event.metadata["model_key"] for event in model_calls] == ["qwen_35b", "qwen_35b"]


def test_agent_runner_includes_response_schema_in_messages() -> None:
    registry = load_registry()

    class RecordingAdapter:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] = []

        def generate(self, **kwargs):
            self.messages = kwargs["messages"]
            return ModelResult(
                content=json.dumps(
                    {
                        "title": "ACOS",
                        "problem_statement": "Automate product delivery",
                        "goals": ["Generate structured outputs"],
                    }
                ),
                tool_calls=[],
                raw={"mock": True},
                model="qwen_35b",
                provider="local_qwen",
                finish_reason="stop",
                usage={"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160},
            )

    adapter = RecordingAdapter()
    registry.register_adapter_factory(
        ProviderType.OPENAI_COMPATIBLE,
        lambda provider, model: adapter,
    )
    runner, _ = _runner(registry)

    result, _, _ = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        require_json_schema=True,
    )

    assert result.title == "ACOS"
    assert len(adapter.messages) == 2
    assert adapter.messages[0]["role"] == "system"
    assert '"title"' in str(adapter.messages[0]["content"])
    assert '"problem_statement"' in str(adapter.messages[0]["content"])


def test_agent_runner_raises_when_repair_fails_without_fallbacks() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "fixer": [
                "{bad json",
                "{still bad json",
                FixResult(
                    status=FixStatus.FIXED,
                    summary="Fixed after fallback",
                ).model_dump(),
            ]
        },
    )
    runner, _ = _runner(registry)
    with pytest.raises(
        RoutingError,
        match=r"Fallbacks exhausted for role fixer after invalid_json; attempted models: qwen_35b",
    ):
        runner.run(
            role="fixer",
            response_model=FixResult,
            context_packet=_packet("fixer"),
            routing_context=RoutingContext(role="fixer"),
            require_json_schema=True,
        )


def test_agent_runner_uses_escalated_model_from_failure_history() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Fixed on escalated model",
            ).model_dump()
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, selection, _ = runner.run(
        role="fixer",
        response_model=FixResult,
        context_packet=_packet("fixer"),
        task_state=TaskState(changed_files_count=1, complexity=TaskComplexity.MEDIUM),
        failure_history=FailureHistory(repeated_failures=2),
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Fixed on escalated model"
    assert selection.model_key == "qwen_35b"
    assert selection.reason.value == "escalation"
    assert any(
        event.event_type == "model_call"
        and event.metadata["routing_reason"] == "escalation"
        for event in audit_events
    )


def test_agent_runner_marks_length_finish_reason_as_output_truncated() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "pm": {
                "content": "{\"title\": \"ACOS\"",
                "tool_calls": [],
                "raw": {"mock": True},
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 1024,
                    "total_tokens": 1144,
                },
            }
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    with pytest.raises(
        StructuredOutputError,
        match="output_truncated before valid JSON",
    ):
        runner.run(
            role="pm",
            response_model=PRD,
            context_packet=_packet("pm"),
            routing_context=RoutingContext(role="pm"),
            allowed_tools=["memory_server.write_memory"],
            require_json_schema=True,
            audit_events=audit_events,
        )

    model_event = next(event for event in audit_events if event.event_type == "model_call")
    assert model_event.metadata["error"] == "output_truncated"
    assert model_event.metadata["output_truncated"] is True
    assert model_event.metadata["finish_reason"] == "length"
    assert isinstance(model_event.metadata["resolved_max_output_tokens"], int)
    assert model_event.metadata["completion_tokens_estimate"] == 1024


def test_agent_runner_executes_allowed_tool_calls_only(tmp_path) -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "implementer": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "repo_server.search_text",
                            "arguments": {"query": "needle"},
                        }
                    ],
                },
                ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Implemented with tool assistance",
                    patches=[],
                ).model_dump(),
            ]
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "example.py").write_text("needle = 1\n", encoding="utf-8")
    runner, _ = _runner(
        registry,
        workspace_root=str(workspace),
        memory_db_path=workspace / ".memory.sqlite3",
    )
    audit_events = []

    result, _, _ = runner.run(
        role="implementer",
        response_model=ImplementationResult,
        context_packet=_packet("implementer", repo_path=str(workspace)),
        routing_context=RoutingContext(role="implementer"),
        allowed_tools=["repo_server.search_text"],
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Implemented with tool assistance"
    assert any(
        event.event_type == "tool_call" and event.metadata["tool_name"] == "repo_server.search_text"
        for event in audit_events
    )


def test_agent_runner_rejects_forbidden_tool_calls(tmp_path) -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "implementer": {
                "content": "",
                "tool_calls": [
                    {
                        "name": "git_server.commit",
                        "arguments": {"message": "acos: bad", "branch": "acos/test"},
                    }
                ],
            }
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner, _ = _runner(
        registry,
        workspace_root=str(workspace),
        memory_db_path=workspace / ".memory.sqlite3",
    )

    with pytest.raises(PermissionError):
        runner.run(
            role="implementer",
            response_model=ImplementationResult,
            context_packet=_packet("implementer", repo_path=str(workspace)),
            routing_context=RoutingContext(role="implementer"),
            allowed_tools=["repo_server.search_text"],
            require_json_schema=True,
        )


def test_agent_runner_serializes_tool_messages_for_openai_compat(tmp_path) -> None:
    registry = load_registry()

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []

        def generate(self, **kwargs):
            messages = kwargs["messages"]
            self.calls.append(messages)
            if len(self.calls) == 1:
                return ModelResult(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "repo_server.search_text",
                            "arguments": {"query": "needle"},
                        }
                    ],
                    raw={"mock": True},
                    model="qwen_35b",
                    provider="local_qwen",
                    finish_reason="tool_calls",
                )
            return ModelResult(
                content=json.dumps(
                    {
                        "status": "implemented",
                        "summary": "Implemented with tool assistance",
                        "patches": [],
                    }
                ),
                tool_calls=[],
                raw={"mock": True},
                model="qwen_35b",
                provider="local_qwen",
                finish_reason="stop",
            )

    adapter = RecordingAdapter()
    registry.register_adapter_factory(
        ProviderType.OPENAI_COMPATIBLE,
        lambda provider, model: adapter,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "example.py").write_text("needle = 1\n", encoding="utf-8")
    runner, _ = _runner(
        registry,
        workspace_root=str(workspace),
        memory_db_path=workspace / ".memory.sqlite3",
    )

    result, _, _ = runner.run(
        role="implementer",
        response_model=ImplementationResult,
        context_packet=_packet("implementer", repo_path=str(workspace)),
        routing_context=RoutingContext(role="implementer"),
        allowed_tools=["repo_server.search_text"],
        require_json_schema=True,
    )

    assert result.summary == "Implemented with tool assistance"
    second_messages = adapter.calls[1]
    assistant_message = second_messages[-2]
    tool_message = second_messages[-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "repo_server.search_text",
                "arguments": "{\"query\": \"needle\"}",
            },
        }
    ]
    assert assistant_message["content"] is None
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
