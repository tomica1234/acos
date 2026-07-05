import json

import pytest

from packages.agents.runner import AgentRunner
from packages.llm.errors import AdapterError, StructuredOutputError
from packages.llm.routing import FailureHistory, ModelRouter, RoutingContext, TaskState
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import FixResult, ImplementationResult, PRD
from packages.schemas.context import ContextPacket
from packages.schemas.models import FixStatus, ImplementationStatus, ModelResult, TaskComplexity

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


def test_memory_server_defaults_missing_scope_to_global(tmp_path) -> None:
    registry = load_registry()
    _, environment = _runner(registry, memory_db_path=tmp_path / "memory.sqlite3")

    result = environment.memory_server.write_memory(content="remember this")
    entries = environment.memory_server.read_memory(scope="global")["entries"]

    assert result == {"scope": "global", "key": "default"}
    assert entries[0]["scope"] == "global"
    assert entries[0]["key"] == "default"


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
    assert selection.model_key == "ornith_35b_q4"
    assert record.role == "pm"

    selection_event = next(event for event in audit_events if event.event_type == "model_selection")
    model_event = next(event for event in audit_events if event.event_type == "model_call")
    assert selection_event.metadata["model_key"] == "ornith_35b_q4"
    assert model_event.metadata["model_key"] == "ornith_35b_q4"
    assert model_event.metadata["provider_key"] == "local_ornith"
    assert model_event.metadata["routing_reason"] == "role_default"

    serialized_audit = json.dumps([event.model_dump(mode="json") for event in audit_events])
    assert "sk-this-should-not-appear" not in serialized_audit


def test_agent_runner_retries_transient_timeout_without_fallback() -> None:
    registry = load_registry()
    registry.get_agent("pm").fallback_models = []

    class FlakyAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise AdapterError("temporary timeout", code="timeout")
            return ModelResult(
                content=PRD(
                    title="ACOS",
                    problem_statement="Recovered after retry",
                    goals=["retry transient provider failures"],
                ).model_dump_json(),
                tool_calls=[],
                raw={"mock": True},
                model="ornith-1.0-35b-Q4_K_M.gguf",
                provider="local_ornith",
                finish_reason="stop",
            )

    adapter = FlakyAdapter()
    registry.register_adapter_factory(
        registry.get_provider("local_ornith").type,
        lambda provider, model: adapter,
    )
    runner, _ = _runner(registry)

    result, selection, _ = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        require_json_schema=True,
    )

    assert result.problem_statement == "Recovered after retry"
    assert selection.model_key == "ornith_35b_q4"


def test_agent_runner_reports_timeout_clearly_when_no_fallback_exists() -> None:
    registry = load_registry()
    registry.get_agent("pm").fallback_models = []

    class AlwaysTimeoutAdapter:
        def generate(self, **kwargs):
            raise AdapterError("temporary timeout", code="timeout")

    adapter = AlwaysTimeoutAdapter()
    registry.register_adapter_factory(
        registry.get_provider("local_ornith").type,
        lambda provider, model: adapter,
    )
    runner, _ = _runner(registry)

    with pytest.raises(AdapterError) as exc:
        runner.run(
            role="pm",
            response_model=PRD,
            context_packet=_packet("pm"),
            routing_context=RoutingContext(role="pm"),
            require_json_schema=True,
        )

    assert exc.value.code == "timeout"
    assert "no fallback model available" in str(exc.value)


def test_agent_runner_uses_fallback_model_when_requested_by_routing_state() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Recovered on fallback",
            ).model_dump()
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, selection, _ = runner.run(
        role="fixer",
        response_model=FixResult,
        context_packet=_packet("fixer"),
        routing_context=RoutingContext(role="fixer", last_error="timeout"),
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Recovered on fallback"
    assert selection.model_key == "mock_structured"
    assert selection.reason.value == "fallback"
    assert any(
        event.event_type == "model_call"
        and event.metadata["model_key"] == "mock_structured"
        and event.metadata["routing_reason"] == "fallback"
        for event in audit_events
    )


def test_agent_runner_uses_fallback_model_from_failure_history() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "fixer": FixResult(
                status=FixStatus.FIXED,
                summary="Recovered from failure history fallback",
            ).model_dump()
        },
    )
    runner, _ = _runner(registry)

    result, selection, _ = runner.run(
        role="fixer",
        response_model=FixResult,
        context_packet=_packet("fixer"),
        failure_history=FailureHistory(last_error="timeout"),
        require_json_schema=True,
    )

    assert result.summary == "Recovered from failure history fallback"
    assert selection.model_key == "mock_structured"
    assert selection.reason.value == "fallback"


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
    assert selection.model_key == "ornith_35b_q4"
    model_calls = [event for event in audit_events if event.event_type == "model_call"]
    assert len(model_calls) == 2
    assert [event.metadata["model_key"] for event in model_calls] == ["ornith_35b_q4", "ornith_35b_q4"]


def test_agent_runner_falls_back_after_repair_failure() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "fixer": [
                "{bad json",
                "{still bad json",
                "{third bad json",
                "{\"status\":\"fixed\"}",
                FixResult(
                    status=FixStatus.FIXED,
                    summary="Fixed after fallback",
                ).model_dump(),
            ]
        },
    )
    runner, _ = _runner(registry)
    audit_events = []

    result, selection, _ = runner.run(
        role="fixer",
        response_model=FixResult,
        context_packet=_packet("fixer"),
        routing_context=RoutingContext(role="fixer"),
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Fixed after fallback"
    assert selection.model_key == "mock_structured"
    model_calls = [event for event in audit_events if event.event_type == "model_call"]
    assert [event.metadata["model_key"] for event in model_calls] == [
        "ornith_35b_q4",
        "ornith_35b_q4",
        "ornith_35b_q4",
        "ornith_35b_q4",
        "mock_structured",
    ]
    assert model_calls[-1].metadata["routing_reason"] == "fallback"


def test_agent_runner_allows_multiple_structured_repairs_before_success() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "pm": [
                "{bad json",
                "{\"title\": 1}",
                PRD(
                    title="ACOS",
                    problem_statement="Need a precise PRD",
                    goals=["ship the feature"],
                ).model_dump(),
            ]
        },
    )
    runner, _ = _runner(registry)

    result, selection, _ = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        require_json_schema=True,
        allowed_tools=[],
    )

    assert result.title == "ACOS"
    assert selection.model_key == "ornith_35b_q4"


def test_agent_runner_reports_structured_failure_clearly_without_fallback() -> None:
    registry = load_registry()
    registry.get_agent("pm").fallback_models = []
    attach_mock_adapter(
        registry,
        {
            "pm": [
                "{bad json",
                "{\"title\": 1}",
                "{\"title\": \"still incomplete\"}",
                "{\"title\": \"still incomplete\"}",
            ]
        },
    )
    runner, _ = _runner(registry)

    with pytest.raises(StructuredOutputError) as exc:
        runner.run(
            role="pm",
            response_model=PRD,
            context_packet=_packet("pm"),
            routing_context=RoutingContext(role="pm"),
            require_json_schema=True,
            allowed_tools=[],
        )

    assert "invalid structured output" in str(exc.value)


def test_agent_runner_accepts_first_json_object_when_trailing_text_exists() -> None:
    registry = load_registry()
    attach_mock_adapter(
        registry,
        {
            "pm": (
                PRD(
                    title="ACOS",
                    problem_statement="Need a precise PRD",
                    goals=["ship the feature"],
                ).model_dump_json()
                + "\n\nAdditional explanation that should be ignored."
            )
        },
    )
    runner, _ = _runner(registry)

    result, selection, _ = runner.run(
        role="pm",
        response_model=PRD,
        context_packet=_packet("pm"),
        routing_context=RoutingContext(role="pm"),
        require_json_schema=True,
        allowed_tools=[],
    )

    assert result.title == "ACOS"
    assert selection.model_key == "ornith_35b_q4"


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
    assert selection.model_key == "ncmoe40_q4"
    assert selection.reason.value == "escalation"
    assert any(
        event.event_type == "model_call"
        and event.metadata["routing_reason"] == "escalation"
        for event in audit_events
    )


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


def test_agent_runner_formats_tool_replay_messages_for_openai_compatible_api(tmp_path) -> None:
    registry = load_registry()
    captured_messages: list[list[dict[str, object]]] = []

    from packages.schemas.models import ModelResult

    class ScenarioAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            captured_messages.append(list(kwargs["messages"]))
            self.calls += 1
            if self.calls == 1:
                return ModelResult(
                    content="",
                    tool_calls=[
                        {
                            "name": "repo_server.search_text",
                            "arguments": {"query": "needle"},
                        }
                    ],
                    raw={"mock": True},
                    model="mock/model",
                    provider="mock_provider",
                    finish_reason="tool_calls",
                )
            return ModelResult(
                content=ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Implemented with compatible tool replay",
                    patches=[],
                ).model_dump_json(),
                tool_calls=[],
                raw={"mock": True},
                model="mock/model",
                provider="mock_provider",
                finish_reason="stop",
            )

    adapter = ScenarioAdapter()
    registry.register_adapter_factory(
        registry.get_provider("local_ornith").type,
        lambda provider, model: adapter,
    )
    registry.register_adapter_factory(
        registry.get_provider("mock_provider").type,
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

    assert result.summary == "Implemented with compatible tool replay"
    assert len(captured_messages) == 2
    replay_messages = captured_messages[1]
    assistant_message = replay_messages[-2]
    tool_message = replay_messages[-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"][0]["type"] == "function"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "repo_server.search_text"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == assistant_message["tool_calls"][0]["id"]


def test_agent_runner_replays_allowed_tool_errors_to_model(tmp_path) -> None:
    registry = load_registry()
    captured_messages: list[list[dict[str, object]]] = []

    class ScenarioAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **kwargs):
            captured_messages.append(list(kwargs["messages"]))
            self.calls += 1
            if self.calls == 1:
                return ModelResult(
                    content="",
                    tool_calls=[
                        {
                            "name": "repo_server.read_file",
                            "arguments": {},
                        }
                    ],
                    raw={"mock": True},
                    model="mock/model",
                    provider="mock_provider",
                    finish_reason="tool_calls",
                )
            return ModelResult(
                content=ImplementationResult(
                    status=ImplementationStatus.IMPLEMENTED,
                    summary="Recovered from tool error",
                    patches=[],
                ).model_dump_json(),
                tool_calls=[],
                raw={"mock": True},
                model="mock/model",
                provider="mock_provider",
                finish_reason="stop",
            )

    adapter = ScenarioAdapter()
    registry.register_adapter_factory(
        registry.get_provider("local_ornith").type,
        lambda provider, model: adapter,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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
        allowed_tools=["repo_server.read_file"],
        require_json_schema=True,
        audit_events=audit_events,
    )

    assert result.summary == "Recovered from tool error"
    tool_message = captured_messages[1][-1]
    assert tool_message["role"] == "tool"
    assert "missing 1 required positional argument" in str(tool_message["content"])
    assert any(event.event_type == "tool_call" and event.status == "failed" for event in audit_events)


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
