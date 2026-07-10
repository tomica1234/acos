"""Self-evaluation scoring for ACOS completion readiness."""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.orchestrator.completion_verifier import DefinitionOfDoneVerifier
from packages.schemas.jobs import JobRecord


@dataclass
class SelfEvaluation:
    score: float
    confidence: float
    missing_evidence: list[str] = field(default_factory=list)
    unresolved_risks: list[str] = field(default_factory=list)


class SelfEvaluator:
    """Score PRD/task/test/runtime/review/security evidence."""

    def __init__(self, verifier: DefinitionOfDoneVerifier | None = None) -> None:
        self.verifier = verifier or DefinitionOfDoneVerifier()

    def evaluate(self, record: JobRecord) -> SelfEvaluation:
        verification = self.verifier.verify(record)
        missing_count = len(verification.missing_evidence) + len(verification.unresolved_findings)
        total = max(1, missing_count + 10)
        score = 1.0 if verification.passed else max(0.0, 1.0 - (missing_count / total))
        return SelfEvaluation(
            score=score,
            confidence=score,
            missing_evidence=list(verification.missing_evidence),
            unresolved_risks=list(verification.unresolved_findings),
        )
