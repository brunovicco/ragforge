"""Post-generation citation and support audit contract (ADR-0016).

Deterministic Citation Accuracy (metrics/citation.py) is an evaluation
metric against a golden set - it says nothing about a live answer whose
citation points to a real, existing structural ID that was simply never
retrieved for this question, or whose claim merely looks supported by its
cited text. This module defines the audit's schema and the two LLM-backed
ports it needs (semantic support verification, bounded rewrite) so
citation_audit.py's orchestration and its deterministic checks never depend
on a concrete LLM SDK (ADR-0009's adapter boundary) - only on these
Protocols.

``ModelIdentity`` is reused from judge_ports.py rather than duplicated: it
is already exactly "provider + model + reasoning_effort + schema version",
which is everything ADR-0016 asks the audit's manifest entry to carry.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ragforge.evaluation.judge_ports import ModelIdentity


class TemporalStatus(StrEnum):
    """Whether a cited device was in force at the relevant date (ADR-0016)."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class SupportVerdict(StrEnum):
    """The semantic verifier's judgment of whether cited text supports a claim."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    INDETERMINATE = "indeterminate"


class AuditOutcome(StrEnum):
    """The audit's final delivery decision for one answer."""

    ACCEPTED = "accepted"
    ACCEPTED_WITH_CAVEAT = "accepted_with_caveat"
    ABSTAIN = "abstain"
    AUDIT_FAILED = "audit_failed"


@dataclass(frozen=True, slots=True)
class AnswerClaim:
    """One segmented unit of a generated answer, with the structural IDs it cites."""

    claim_id: str
    text: str
    cited_structural_ids: tuple[str, ...]
    sentence_index: int
    material: bool


@dataclass(frozen=True, slots=True)
class DeterministicCitationCheck:
    """Non-LLM checks for one cited structural ID (ADR-0016) - never invokes an LLM."""

    structural_id: str
    well_formed: bool
    exists_in_corpus: bool
    belongs_to_selected_document_version: bool
    present_in_retrieved_context: bool
    source_text_hash_matches: bool


@dataclass(frozen=True, slots=True)
class SemanticSupportResult:
    """The semantic verifier's structured judgment for one claim."""

    verdict: SupportVerdict
    rationale: str
    supported_citation_ids: tuple[str, ...]
    unsupported_citation_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimAudit:
    """The complete audit trail for one claim.

    ``semantic_support`` is ``None`` when any of the claim's citations
    failed a deterministic check - the LLM verifier is never called for a
    claim deterministic checks have already condemned (ADR-0016: "prefer
    deterministic checks before LLM verification").
    """

    claim: AnswerClaim
    citation_checks: tuple[DeterministicCitationCheck, ...]
    semantic_support: SemanticSupportResult | None


@dataclass(frozen=True, slots=True)
class AuditResult:
    """The complete audit outcome for one generated answer (ADR-0016)."""

    outcome: AuditOutcome
    claims: tuple[ClaimAudit, ...]
    original_answer: str
    final_answer: str
    rewritten: bool
    temporal_status: TemporalStatus
    verifier_identity: ModelIdentity | None


@runtime_checkable
class SemanticSupportVerifier(Protocol):
    """Judges whether cited authoritative text supports one claim - no outside retrieval."""

    @property
    def identity(self) -> ModelIdentity:
        """Exact verifier configuration used by verify() - recorded in the run manifest."""
        ...

    def verify(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        """Return the structured support judgment for ``claim_text`` given only the cited text.

        ``cited_citations`` is ``(structural_id, source_text)`` pairs, not
        bare text - the verifier needs the ID alongside its text to
        attribute ``supported_citation_ids``/``unsupported_citation_ids``
        to specific citations rather than judging the claim as an
        undifferentiated whole.
        """
        ...


@runtime_checkable
class AnswerRewriter(Protocol):
    """Produces at most one corrected answer from an audit's findings (ADR-0016)."""

    def rewrite(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings: tuple[ClaimAudit, ...],
    ) -> str:
        """Return a rewritten answer removing unsupported material, introducing no new citations."""
        ...
