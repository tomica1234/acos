"""Synthesize executable job contracts from PM requirements output."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Mapping

from packages.orchestrator.framework_profiles import resolve_framework_profile
from packages.orchestrator.framework_scaffolds import resolve_framework_scaffold
from packages.orchestrator.quality_gates import valid_artifact_paths
from packages.schemas.agent_outputs import PRD
from packages.schemas.runtime import RuntimeHttpCheck

_PYTHON_IDENTIFIER_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def synthesize_job_metadata_from_prd(
    prd: PRD,
    metadata: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    merged = deepcopy(dict(metadata))
    inferred_profile = _resolve_framework_profile_from_prd(prd, merged)
    if inferred_profile is not None and not merged.get("framework_profile"):
        merged["framework_profile"] = inferred_profile

    if merged.get("framework_profile") == "django-web" and not merged.get("framework_project_name"):
        merged["framework_project_name"] = _resolve_django_project_name(
            prd,
            workspace_root=workspace_root,
        )

    if (
        merged.get("framework_profile") in {"fastapi-api", "flask-web"}
        and not merged.get("framework_entrypoint")
    ):
        merged["framework_entrypoint"] = prd.framework_entrypoint or "app.main:app"

    merged["runtime"] = _merge_runtime_contract(prd, merged)
    if not merged["runtime"]:
        merged.pop("runtime", None)

    acceptance_checks = _merge_acceptance_checks(prd, merged)
    if acceptance_checks:
        merged["acceptance_checks"] = acceptance_checks

    required_artifacts = _merge_required_artifacts(prd, merged, workspace_root=workspace_root)
    if required_artifacts:
        merged["required_artifacts"] = required_artifacts

    return merged


def _resolve_framework_profile_from_prd(prd: PRD, metadata: Mapping[str, Any]) -> str | None:
    explicit = metadata.get("framework_profile") or metadata.get("framework")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if prd.framework_profile:
        return prd.framework_profile
    combined = " ".join(
        [
            prd.title,
            prd.problem_statement,
            *prd.goals,
            *prd.constraints,
            *prd.success_criteria,
        ]
    ).lower()
    if "django" in combined:
        return "django-web"
    if "fastapi" in combined:
        return "fastapi-api"
    if "flask" in combined:
        return "flask-web"
    if any(token in combined for token in ("node", "express", "npm", "package.json")):
        return "node-web"
    return None


def _resolve_django_project_name(prd: PRD, *, workspace_root: str | Path) -> str:
    if prd.framework_project_name:
        return prd.framework_project_name
    prd_implies_django = _resolve_framework_profile_from_prd(prd, {}) == "django-web"
    candidate = (
        prd.title.strip()
        if prd_implies_django and prd.title.strip()
        else Path(workspace_root).resolve().name
    )
    normalized = _PYTHON_IDENTIFIER_PATTERN.sub("_", candidate).strip("_")
    if not normalized:
        return "acos_project"
    if normalized[0].isdigit():
        normalized = f"app_{normalized}"
    return normalized


def _merge_runtime_contract(prd: PRD, metadata: Mapping[str, Any]) -> dict[str, Any]:
    runtime_payload = metadata.get("runtime", {})
    merged = dict(runtime_payload) if isinstance(runtime_payload, dict) else {}
    if prd.runtime is not None:
        runtime_hint = prd.runtime.model_dump(exclude_none=True)
        for key, value in runtime_hint.items():
            if key == "extra":
                continue
            if key not in merged and value not in (None, [], {}):
                merged[key] = value
        for key, value in runtime_hint.get("extra", {}).items():
            if key not in merged:
                merged[key] = value
    profile = resolve_framework_profile(metadata if merged is runtime_payload else {**metadata, **{"runtime": merged}})
    if profile is not None:
        if profile.runtime_prepare_commands is not None and "prepare_commands" not in merged:
            merged["prepare_commands"] = [list(command) for command in profile.runtime_prepare_commands]
        if profile.runtime_start_command is not None and "start_command" not in merged:
            merged["start_command"] = list(profile.runtime_start_command)
        if "http_probe_path" not in merged:
            merged["http_probe_path"] = profile.runtime_http_probe_path
    return merged


def _merge_acceptance_checks(prd: PRD, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = metadata.get("acceptance_checks")
    if isinstance(current, list) and current:
        return [
            RuntimeHttpCheck.model_validate(item).model_dump(exclude_none=True)
            for item in current
        ]
    if prd.acceptance_checks:
        return [item.model_dump(exclude_none=True) for item in prd.acceptance_checks]
    profile_name = metadata.get("framework_profile") or metadata.get("framework")
    if profile_name in {"django-web", "fastapi-api", "flask-web"}:
        return [
            RuntimeHttpCheck(
                name="home",
                method="GET",
                path="/",
                expect_status=200,
            ).model_dump(exclude_none=True)
        ]
    return []


def _merge_required_artifacts(
    prd: PRD,
    metadata: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> list[str]:
    artifacts: set[str] = set()
    current = metadata.get("required_artifacts")
    if isinstance(current, list):
        artifacts.update(_normalized_artifacts(current))
    artifacts.update(_normalized_artifacts(prd.required_artifacts))
    if _mentions_readme(prd):
        artifacts.add("README.md")
    scaffold = resolve_framework_scaffold(metadata, workspace_root=workspace_root)
    if scaffold is not None:
        artifacts.update(scaffold.required_artifacts)
    return sorted(artifacts)


def _normalized_artifacts(values: list[str]) -> set[str]:
    return valid_artifact_paths(item for item in values if isinstance(item, str))


def _mentions_readme(prd: PRD) -> bool:
    haystack = " ".join([*prd.goals, *prd.success_criteria, *prd.constraints]).lower()
    return "readme" in haystack
