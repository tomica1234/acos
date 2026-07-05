"""Quality gate enforcement."""

from __future__ import annotations

from packages.schemas.agent_outputs import FilePatch, ReviewResult, SecurityReviewResult, TestRunResult
from packages.schemas.models import ReviewDecision


class QualityGateError(Exception):
    """Raised when a quality gate fails."""


def ensure_reviews_pass(
    review: ReviewResult, security_review: SecurityReviewResult
) -> None:
    if review.decision != ReviewDecision.APPROVE:
        raise QualityGateError("Reviewer did not approve the change set")
    if security_review.decision != ReviewDecision.APPROVE:
        raise QualityGateError("Security review did not approve the change set")


def ensure_fixer_safe(patches: list[FilePatch]) -> None:
    suspicious_tokens = ("xfail", "skip(", "skipif(", "assert True")
    for patch in patches:
        if patch.path.startswith("tests/") and any(token in patch.content for token in suspicious_tokens):
            raise QualityGateError("Fixer attempted to weaken tests")


def ensure_tests_passed(result: TestRunResult) -> None:
    if not result.success:
        raise QualityGateError("Tests did not pass")

