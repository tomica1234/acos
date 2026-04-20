from __future__ import annotations

from pathlib import Path

import yaml

from apps.cli import main

from tests.conftest import config_dir


def _copy_configs(tmp_path: Path) -> Path:
    target = tmp_path / "configs"
    target.mkdir()
    for name in [
        "model_providers.yaml",
        "agents.yaml",
        "model_routing.yaml",
        "policies.yaml",
    ]:
        (target / name).write_text((config_dir() / name).read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_validate_config_succeeds(capsys) -> None:
    exit_code = main(["validate-config", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_validate_config_fails_for_broken_config(tmp_path, capsys) -> None:
    configs = _copy_configs(tmp_path)
    providers = yaml.safe_load((configs / "model_providers.yaml").read_text(encoding="utf-8"))
    providers["models"]["qwen_35b"]["provider"] = "missing_provider"
    (configs / "model_providers.yaml").write_text(yaml.safe_dump(providers), encoding="utf-8")

    exit_code = main(["validate-config", "--config-dir", str(configs)])

    assert exit_code == 1
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any("missing_provider" in error for error in payload["errors"])


def test_list_models_returns_expected_fields(capsys) -> None:
    exit_code = main(["list-models", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    model_keys = {item["model_key"] for item in payload["models"]}
    assert {"qwen_35b", "qwen_small"} <= model_keys
    qwen = next(item for item in payload["models"] if item["model_key"] == "qwen_35b")
    assert qwen["provider"] == "local_qwen"
    assert qwen["supports_tool_calling"] is True
    assert "agentic" in qwen["tags"]


def test_list_agents_returns_role_model_mapping(capsys) -> None:
    exit_code = main(["list-agents", "--config-dir", str(config_dir())])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    implementer = next(item for item in payload["agents"] if item["role"] == "implementer")
    assert implementer["primary_model"] == "qwen_35b"
    assert "mock_structured" in implementer["fallback_models"]


def test_resolve_model_for_implementer_returns_qwen_35b(capsys) -> None:
    exit_code = main(["resolve-model", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["model_key"] == "qwen_35b"
    assert payload["reason"] == "role_default"


def test_resolve_model_with_repeated_failures_returns_escalation(capsys) -> None:
    exit_code = main(
        [
            "resolve-model",
            "--config-dir",
            str(config_dir()),
            "--role",
            "implementer",
            "--repeated-failures",
            "2",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["model_key"] == "qwen_35b"
    assert payload["reason"] == "escalation"
    assert payload["details"]["repeated_failures"] == 2


def test_explain_routing_includes_selection_and_conditions(capsys) -> None:
    exit_code = main(["explain-routing", "--config-dir", str(config_dir()), "--role", "implementer"])

    assert exit_code == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["role"] == "implementer"
    assert payload["selection"]["model_key"] == "qwen_35b"
    assert payload["primary_model"] == "qwen_35b"
    assert "mock_structured" in payload["fallback_models"]
    assert "timeout" in payload["fallback_errors"]
