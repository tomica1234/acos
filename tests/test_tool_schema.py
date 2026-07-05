from packages.llm.tool_schema import build_response_schema, build_tool_manifest
from packages.schemas.agent_outputs import ImplementationResult


def test_tool_schema_helpers() -> None:
    response_schema = build_response_schema(ImplementationResult)
    tool_manifest = build_tool_manifest(["repo_server.apply_patch"])

    assert "properties" in response_schema
    assert response_schema["title"] == "ImplementationResult"
    assert tool_manifest[0]["function"]["name"] == "repo_server.apply_patch"

