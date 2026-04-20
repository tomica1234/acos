import json

import pytest

from packages.agents.runner import AgentRunner
from packages.llm.routing import FailureHistory, ModelRouter, RoutingContext, TaskState
from packages.mcp_client.fake import FakeMCPEnvironment
from packages.orchestrator.audit import AuditRecorder
from packages.orchestrator.policy import PolicyEngine
from packages.schemas.agent_outputs import FixResult, ImplementationResult, PRD
from packages.schemas.context import ContextPacket
from packages.schemas.models import FixStatus, ImplementationStatus, TaskComplexity

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

    serialized_audit = json.dumps([event.model_dump(mode="json") for event in audit_events])
    assert "sk-this-should-not-appear" not in serialized_audit


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
    assert selection.model_key == "qwen_35b"
    assert selection.reason.value == "fallback"
    assert any(
        event.event_type == "model_call"
        and event.metadata["model_key"] == "qwen_35b"
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
    assert selection.model_key == "qwen_35b"
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
    assert selection.model_key == "qwen_35b"
    model_calls = [event for event in audit_events if event.event_type == "model_call"]
    assert len(model_calls) == 2
    assert [event.metadata["model_key"] for event in model_calls] == ["qwen_35b", "qwen_35b"]


def test_agent_runner_falls_back_after_repair_failure() -> None:
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
    assert selection.model_key == "qwen_35b"
    model_calls = [event for event in audit_events if event.event_type == "model_call"]
    assert [event.metadata["model_key"] for event in model_calls] == [
        "qwen_small",
        "qwen_small",
        "qwen_35b",
    ]
    assert model_calls[-1].metadata["routing_reason"] == "fallback"


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
