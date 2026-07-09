"""Shared constraint helpers for ACOS job entrypoints."""

from __future__ import annotations

from packages.schemas.jobs import JobRecord, JobSpec


STRICT_JOB_CONSTRAINTS = {
    "require_prd_quality": True,
    "require_task_acceptance_criteria": True,
    "require_task_artifacts": True,
    "require_completion_integrity": True,
    "require_test_evidence": True,
    "require_stage_test_patches": True,
    "stage_review": True,
}


def apply_strict_job_constraints(spec_or_record: JobSpec | JobRecord) -> None:
    """Require the quality gates expected for normal autonomous job entrypoints."""

    spec = getattr(spec_or_record, "spec", spec_or_record)
    constraints = spec.metadata.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
        spec.metadata["constraints"] = constraints
    constraints.update(STRICT_JOB_CONSTRAINTS)
    constraints.setdefault("test_timeout_seconds", 1200)
