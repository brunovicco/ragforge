"""Tests for the AuditingAnswerGenerator decorator (ADR-0016), using fakes."""

from ragforge.domain.models import Answer, Chunk, Query, RetrievalResult
from ragforge.evaluation.audit_ports import (
    ClaimAudit,
    SemanticSupportResult,
    SupportVerdict,
)
from ragforge.evaluation.judge_ports import ModelIdentity
from ragforge.generation.auditing_answer_generator import AuditingAnswerGenerator

_IDENTITY = ModelIdentity(
    provider="fake", model="fake", reasoning_effort=None, output_schema_version=1
)

NORM = "LC-105/2001"
ART_1 = f"{NORM}::art-1"
ART_99 = f"{NORM}::art-99"


def _chunk(structural_id: str) -> Chunk:
    return Chunk(
        chunk_id=structural_id,
        source_text="texto autoritativo",
        retrieval_text="texto autoritativo",
        structural_ids=(structural_id,),
    )


def _retrieved(*structural_ids: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk=_chunk(sid), score=1.0, strategy="fake") for sid in structural_ids
    ]


class _FakeGenerator:
    name = "fake-generator"

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, query: Query, results: list[RetrievalResult]) -> Answer:
        return Answer(text=self._text, citations=())


class _FakeVerifier:
    identity = _IDENTITY

    def __init__(self, verdict: SupportVerdict = SupportVerdict.SUPPORTED) -> None:
        self._verdict = verdict

    def verify(
        self, question: str, claim_text: str, cited_citations: tuple[tuple[str, str], ...]
    ) -> SemanticSupportResult:
        return SemanticSupportResult(
            verdict=self._verdict,
            rationale="fake",
            supported_citation_ids=(),
            unsupported_citation_ids=(),
            missing_evidence=(),
        )


class _FakeRewriter:
    def __init__(self, rewritten_text: str) -> None:
        self._rewritten_text = rewritten_text

    def rewrite(
        self,
        question: str,
        original_answer: str,
        valid_source_texts: tuple[str, ...],
        findings: tuple[ClaimAudit, ...],
    ) -> str:
        return self._rewritten_text


def test_name_reflects_the_wrapped_generator_plus_audit_suffix() -> None:
    """.name is derived from the wrapped generator's name, marked as audited."""
    generator = AuditingAnswerGenerator(
        _FakeGenerator("texto"), _FakeVerifier(), _FakeRewriter("x"), {}, {}
    )

    assert generator.name == "fake-generator+audit"


def test_generate_returns_the_original_answer_when_fully_supported() -> None:
    """A fully-supported answer passes through unchanged - no rewrite needed."""
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    generator = AuditingAnswerGenerator(
        _FakeGenerator(f"Devem adotar controles [{ART_1}]."),
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter("nunca deveria ser usado"),
        corpus_ids,
        document_versions,
    )

    answer = generator.generate(Query(text="pergunta"), _retrieved(ART_1))

    assert answer.text == f"Devem adotar controles [{ART_1}]."
    assert answer.citations == (ART_1,)


def test_generate_returns_the_rewritten_answer_when_the_original_is_invented() -> None:
    """An invented citation triggers a rewrite - the final Answer reflects the rewritten text."""
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    rewritten_text = f"Resposta corrigida [{ART_1}]."
    generator = AuditingAnswerGenerator(
        _FakeGenerator(f"Texto [{ART_99}]."),
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter(rewritten_text),
        corpus_ids,
        document_versions,
    )

    answer = generator.generate(Query(text="pergunta"), _retrieved(ART_1))

    assert answer.text == rewritten_text
    assert answer.citations == (ART_1,), "citations are re-extracted from the final text"


def test_drain_audit_results_returns_and_clears_the_buffer() -> None:
    """Every generate() call's AuditResult accumulates; draining returns and resets the buffer."""
    corpus_ids = {NORM: {ART_1}}
    document_versions = {NORM: "sha-abc"}
    generator = AuditingAnswerGenerator(
        _FakeGenerator(f"Texto [{ART_1}]."),
        _FakeVerifier(SupportVerdict.SUPPORTED),
        _FakeRewriter("x"),
        corpus_ids,
        document_versions,
    )

    generator.generate(Query(text="q1"), _retrieved(ART_1))
    generator.generate(Query(text="q2"), _retrieved(ART_1))

    first_drain = generator.drain_audit_results()
    second_drain = generator.drain_audit_results()

    assert len(first_drain) == 2
    assert second_drain == []
