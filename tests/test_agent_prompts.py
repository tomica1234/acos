from packages.agents import planner, pm


def test_planner_prompt_requires_task_artifact_paths() -> None:
    prompt = planner.SYSTEM_PROMPT

    assert "target_files" in prompt
    assert "required_artifacts" in prompt
    assert '"role": "scaffold|implementer|test_writer"' in prompt
    assert "implementer, scaffold, and test_writer" in prompt
    assert "do not schedule pm, architect, reviewer" in prompt
    assert "exact test file path" in prompt
    assert "do not assign app/source target_files to test_writer" in prompt
    assert "ordinary test target_files to implementer" in prompt
    assert "PM required_artifact" in prompt
    assert "repo-relative file path" in prompt


def test_pm_prompt_describes_runtime_and_required_artifacts_schema() -> None:
    prompt = pm.SYSTEM_PROMPT

    assert "required_artifacts" in prompt
    assert "runtime: optional object" in prompt
    assert "prepare_commands" in prompt
    assert "http_probe_path" in prompt
    assert "runtime.extra" in prompt
