import pytest

from packages.orchestrator.policy import PolicyEngine

from tests.conftest import config_dir


def test_policy_enforcement() -> None:
    policy = PolicyEngine.from_path(config_dir() / "policies.yaml")

    assert policy.is_tool_allowed("implementer", "repo_server.apply_patch")
    assert not policy.is_tool_allowed("implementer", "test_server.run_test")
    assert policy.is_tool_allowed("release_manager", "git_server.commit")

    with pytest.raises(PermissionError):
        policy.assert_release_commit_allowed("implementer")

    with pytest.raises(PermissionError):
        policy.assert_branch_allowed("main")

