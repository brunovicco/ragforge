"""Tests for the post-generation citation audit pipeline (ADR-0016), using fakes."""

from ragforge.domain.models import Chunk, RetrievalResult
from ragforge.evaluation.audit_ports import (
    AuditOutcome,
    ClaimAudit,
    SemanticSupportResult,
    SupportVerdict,
)
from ragforge.evaluation.citation_audit import (
    audit_answer,
    run_deterministic_checks,
    segment_claims,
)
from ragforge.evaluation.judge_ports import ModelIdentity

NORM = "LC-105/2001"
ART_1 = f"{NORM}::art-1"
ART_99 = f"{NORM}::art-99"
_IDENTITY = ModelIdentity(
    provider="fake", model="fake", reasoning_effort=None, output_schema_version=1
)


def _chunk(structural_id: str, source_text: str = "texto autoritativo") -> Chunk:
    return Chunk(
        chunk_id=structural_id,
        source_text=source_text,
        retrieval_text=source_text,
        structural_ids=(structural_id,),
    )


def _retrieved(*structural_ids: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk=_chunk(sid), score=1.0, strategy="fake") for sid in structural_ids
    ]


class _FakeVerifier:
    identity = _IDENTITY

    def __init__(self, verdict: SupportVerdict = SupportVerdict.SUPPORTED) -> None:
        self._verdict = verdict
        self.calls: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []

    def verify(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        self.calls.append((question, claim_text, cited_citations))
        supported = self._verdict == SupportVerdict.SUPPORTED
        return SemanticSupportResult(
            verdict=self._verdict,
            rationale="fake rationale",
            supported_citation_ids=tuple(c[0] for c in cited_citations) if supported else (),
            unsupported_citation_ids=() if supported else tuple(c[0] for c in cited_citations),
            missing_evidence=(),
        )


class _FakeRewriter:
    def __init__(self, rewritten_text: str = "rewritten") -> None:
        self._rewritten_text = rewritten_text
        self.calls: list[tuple[str, str, tuple[str, ...], tuple[ClaimAudit, ...]]] = []

    def rewrite(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings: tuple[ClaimAudit, ...],
    ) -> str:
        self.calls.append((question, original_answer, valid_source_texts, findings))
        return self._rewritten_text


class _FailingVerifier:
    identity = _IDENTITY

    def verify(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        raise RuntimeError("verifier boom")


class _FailingRewriter:
    def rewrite(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings: tuple[ClaimAudit, ...],
    ) -> str:
        raise RuntimeError("rewriter boom")


# --- segment_claims ---------------------------------------------------------


def test_segment_claims_splits_on_real_sentence_boundaries() -> None:
    """A genuine sentence boundary (period + space + uppercase) splits into two claims."""
    text = f"Devem adotar controles [{ART_1}]. Isso é obrigatório [{ART_1}]."

    claims = segment_claims(text)

    assert len(claims) == 2
    assert claims[0].sentence_index == 0
    assert claims[1].sentence_index == 1


def test_segment_claims_does_not_split_on_legal_abbreviations() -> None:
    """ "Art. 1º" is never split - a digit after the period doesn't match the split lookahead."""
    text = f"Art. 1º estabelece que devem adotar controles [{ART_1}]."

    claims = segment_claims(text)

    assert len(claims) == 1
    assert claims[0].text == text


def test_segment_claims_extracts_citations_per_sentence() -> None:
    """Each sentence's own citations are extracted independently, not pooled."""
    text = f"Primeira frase [{ART_1}]. Segunda frase [{ART_99}]."

    claims = segment_claims(text)

    assert claims[0].cited_structural_ids == (ART_1,)
    assert claims[1].cited_structural_ids == (ART_99,)


def test_segment_claims_marks_abstention_sentences_as_non_material() -> None:
    """A sentence expressing insufficient evidence needs no citation and isn't material."""
    text = "Não há evidência suficiente para responder a esta pergunta."

    [claim] = segment_claims(text)

    assert claim.material is False
    assert claim.cited_structural_ids == ()


def test_segment_claims_marks_ordinary_sentences_as_material() -> None:
    """A normal legal assertion is material by default."""
    text = f"Devem adotar controles de segurança [{ART_1}]."

    [claim] = segment_claims(text)

    assert claim.material is True


# --- run_deterministic_checks ------------------------------------------------


def test_run_deterministic_checks_passes_a_real_retrieved_citation() -> None:
    """A well-formed, retrieved, corpus-existing ID belonging to the manifest passes every check."""
    [claim] = segment_claims(f"Texto [{ART_1}].")
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}

    [check] = run_deterministic_checks(claim, retrieved, corpus_ids, document_versions)

    assert check.well_formed is True
    assert check.exists_in_corpus is True
    assert check.present_in_retrieved_context is True
    assert check.belongs_to_selected_document_version is True
    assert check.source_text_hash_matches is True


def test_run_deterministic_checks_flags_a_malformed_id() -> None:
    """An ID that fails StructuralRef.parse fails well_formed and everything else."""
    [claim] = segment_claims("Texto [not-a-valid-id].")
    retrieved: list[RetrievalResult] = []

    [check] = run_deterministic_checks(claim, retrieved, {}, {})

    assert check.well_formed is False
    assert check.exists_in_corpus is False
    assert check.present_in_retrieved_context is False


def test_run_deterministic_checks_flags_a_nonexistent_citation() -> None:
    """A well-formed ID absent from the whole corpus fails exists_in_corpus."""
    [claim] = segment_claims(f"Texto [{ART_99}].")
    corpus_ids = {NORM: {ART_1}}

    [check] = run_deterministic_checks(claim, [], corpus_ids, {NORM: "sha-abc"})

    assert check.well_formed is True
    assert check.exists_in_corpus is False


def test_run_deterministic_checks_flags_a_citation_not_in_retrieved_context() -> None:
    """A real, existing ID this question's retrieval never surfaced fails the context check.

    This is the ADR-0016 motivating scenario.
    """
    [claim] = segment_claims(f"Texto [{ART_1}].")
    corpus_ids = {NORM: {ART_1}}

    [check] = run_deterministic_checks(claim, [], corpus_ids, {NORM: "sha-abc"})

    assert check.well_formed is True
    assert check.exists_in_corpus is True
    assert check.present_in_retrieved_context is False


def test_run_deterministic_checks_flags_a_document_outside_the_manifest() -> None:
    """A norm not in document_versions fails belongs_to_selected_document_version."""
    [claim] = segment_claims(f"Texto [{ART_1}].")
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}

    [check] = run_deterministic_checks(claim, retrieved, corpus_ids, {})

    assert check.belongs_to_selected_document_version is False


# --- audit_answer: policy decisions ------------------------------------------


def test_audit_answer_accepts_a_fully_supported_answer_without_rewriting() -> None:
    """Every material claim's citations pass deterministic checks and are semantically supported."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}

    result = audit_answer(
        "pergunta",
        f"Devem adotar controles [{ART_1}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter(),
    )

    assert result.outcome == AuditOutcome.ACCEPTED
    assert result.rewritten is False
    assert result.final_answer == result.original_answer


def test_audit_answer_abstains_when_rewrite_cannot_salvage_anything() -> None:
    """An invented citation triggers a rewrite; if the rewrite has nothing citable, abstain."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}

    result = audit_answer(
        "pergunta",
        f"Devem adotar controles [{ART_99}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter("resposta reescrita sem nenhuma citação"),
    )

    assert result.outcome == AuditOutcome.ABSTAIN
    assert result.rewritten is True
    assert "evidência" in result.final_answer.lower()


def test_audit_answer_accepts_after_a_rewrite_that_fully_resolves_the_problem() -> None:
    """A rewrite that produces a fully-supported answer is ACCEPTED, marked as rewritten."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    fixed_text = f"Devem adotar controles [{ART_1}]."

    result = audit_answer(
        "pergunta",
        f"Devem adotar controles [{ART_99}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter(fixed_text),
    )

    assert result.outcome == AuditOutcome.ACCEPTED
    assert result.rewritten is True
    assert result.final_answer == fixed_text


def test_audit_answer_caveats_when_a_supported_subset_survives_the_rewrite() -> None:
    """A rewrite leaving one supported and one still-unsupported material claim is a caveat."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    partial_text = f"Primeira frase [{ART_1}]. Segunda frase [{ART_99}]."

    result = audit_answer(
        "pergunta",
        f"Texto original [{ART_99}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter(partial_text),
    )

    assert result.outcome == AuditOutcome.ACCEPTED_WITH_CAVEAT
    assert result.rewritten is True


def test_audit_answer_never_attempts_a_second_rewrite() -> None:
    """At most one rewrite call is made, even when the rewrite itself doesn't resolve things."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    rewriter = _FakeRewriter("ainda sem citação nenhuma")

    audit_answer(
        "pergunta",
        f"Texto [{ART_99}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        rewriter,
    )

    assert len(rewriter.calls) == 1


def test_audit_answer_reports_audit_failed_when_the_verifier_raises() -> None:
    """A verifier infrastructure failure fails the audit closed, not silently accepted."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}

    result = audit_answer(
        "pergunta",
        f"Texto [{ART_1}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FailingVerifier(),
        _FakeRewriter(),
    )

    assert result.outcome == AuditOutcome.AUDIT_FAILED


def test_audit_answer_reports_audit_failed_when_the_rewriter_raises() -> None:
    """A rewriter infrastructure failure fails the audit closed."""
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}

    result = audit_answer(
        "pergunta",
        f"Texto [{ART_99}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FailingRewriter(),
    )

    assert result.outcome == AuditOutcome.AUDIT_FAILED


def test_audit_answer_temporal_status_is_always_unknown() -> None:
    """No trustworthy temporal corpus exists yet (ADR-0019) - always UNKNOWN, never inferred."""
    from ragforge.evaluation.audit_ports import TemporalStatus

    retrieved = _retrieved(ART_1)
    result = audit_answer(
        "pergunta",
        f"Texto [{ART_1}].",
        retrieved,
        {NORM: {ART_1}},
        {NORM: "sha-abc"},
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter(),
    )

    assert result.temporal_status == TemporalStatus.UNKNOWN


def test_audit_answer_triggers_rewrite_for_an_unsupported_semantic_verdict() -> None:
    """A citation passing every deterministic check but semantically unsupported still rewrites.

    Deterministic checks alone are not enough - the semantic verdict matters too.
    """
    retrieved = _retrieved(ART_1)
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    rewriter = _FakeRewriter(f"Texto reescrito [{ART_1}].")

    result = audit_answer(
        "pergunta",
        f"Texto [{ART_1}].",
        retrieved,
        corpus_ids,
        document_versions,
        _FakeVerifier(SupportVerdict.UNSUPPORTED),
        rewriter,
    )

    assert len(rewriter.calls) == 1
    assert result.rewritten is True
