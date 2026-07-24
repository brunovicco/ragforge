"""Post-generation citation and support audit pipeline (ADR-0016).

Deterministic Citation Accuracy (metrics/citation.py) only checks a golden
set's judged-relevant IDs; it says nothing at runtime about a citation that
points to a real structural ID the retrieval step simply never surfaced for
this question, or a claim that merely looks supported by its cited text.
This module runs the full audit: deterministic sentence segmentation ->
per-citation deterministic checks -> semantic support verification (LLM,
only for claims whose citations already pass every deterministic check) ->
policy decision -> at most one bounded rewrite + full re-audit -> final
outcome. No LLM SDK is imported here - only the ``SemanticSupportVerifier``/
``AnswerRewriter`` Protocols from audit_ports.py (ADR-0009's adapter
boundary).

Segmentation is a simple heuristic, not NLP-grade sentence splitting - the
ADR itself acknowledges claim segmentation "remain[s] imperfect". A sentence
matching a small fixed set of abstention/insufficient-evidence markers is
marked non-material: it asserts nothing legal, so it needs no citation and
must never trigger a rewrite merely for lacking one - the generator's own
system prompt explicitly asks for this kind of sentence when the evidence
is insufficient.

Temporal validity is always ``TemporalStatus.UNKNOWN``: no trustworthy
temporal corpus exists yet (ADR-0019, a future, currently-blocked release),
and the ADR is explicit that the system must not equate mere document
existence with temporal validity.
"""

import re

from ragforge.domain.models import RetrievalResult, StructuralRef
from ragforge.evaluation.audit_ports import (
    AnswerClaim,
    AnswerRewriter,
    AuditOutcome,
    AuditResult,
    ClaimAudit,
    DeterministicCitationCheck,
    SemanticSupportVerifier,
    SupportVerdict,
    TemporalStatus,
)
from ragforge.generation.citation_parsing import extract_citation_candidates

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-Ú"\[])')
_ABSTENTION_MARKERS = (
    "não há evidência",
    "evidência insuficiente",
    "não foi possível",
    "insuficiente para responder",
    "não consta",
    "não localizei",
)
_ABSTENTION_MESSAGE = (
    "Não há evidência suficiente nas fontes recuperadas para responder a esta pergunta com "
    "segurança."
)


def _looks_like_abstention(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def segment_claims(answer_text: str) -> list[AnswerClaim]:
    """Split ``answer_text`` into sentence-level claims (deterministic, ADR-0016).

    Splits after ``.``/``!``/``?`` only when followed by whitespace and then
    an uppercase letter, quote, or bracket - "Art. 1º" is never split (a
    digit doesn't match that lookahead), while a genuine sentence boundary
    almost always is.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer_text.strip()) if s.strip()]
    return [
        AnswerClaim(
            claim_id=f"claim-{index}",
            text=sentence,
            cited_structural_ids=extract_citation_candidates(sentence),
            sentence_index=index,
            material=not _looks_like_abstention(sentence),
        )
        for index, sentence in enumerate(sentences)
    ]


def _resolve_chunk_source_text(
    structural_id: str, retrieved_results: list[RetrievalResult]
) -> str | None:
    for result in retrieved_results:
        if structural_id in result.chunk.structural_ids:
            return result.chunk.source_text
    return None


def run_deterministic_checks(
    claim: AnswerClaim,
    retrieved_results: list[RetrievalResult],
    corpus_structural_ids: dict[str, set[str]],
    document_versions: dict[str, str],
) -> tuple[DeterministicCitationCheck, ...]:
    """Run every non-LLM check (ADR-0016) for each structural ID ``claim`` cites.

    ``corpus_structural_ids`` (``{norm_id: {every structural ID indexed for
    that norm}}``) answers "does this device exist at all", independent of
    whether this question's retrieval happened to surface it -
    ``present_in_retrieved_context`` answers that second, distinct question.
    """
    checks = []
    for raw_id in claim.cited_structural_ids:
        try:
            ref = StructuralRef.parse(raw_id)
        except ValueError:
            checks.append(
                DeterministicCitationCheck(
                    structural_id=raw_id,
                    well_formed=False,
                    exists_in_corpus=False,
                    belongs_to_selected_document_version=False,
                    present_in_retrieved_context=False,
                    source_text_hash_matches=False,
                )
            )
            continue

        present_in_retrieved_context = (
            _resolve_chunk_source_text(raw_id, retrieved_results) is not None
        )
        checks.append(
            DeterministicCitationCheck(
                structural_id=raw_id,
                well_formed=True,
                exists_in_corpus=raw_id in corpus_structural_ids.get(ref.norm, set()),
                belongs_to_selected_document_version=ref.norm in document_versions,
                present_in_retrieved_context=present_in_retrieved_context,
                # No separately-recorded evidence-lineage hash exists yet
                # (that is ADR-0017's job) - the only hash available is the
                # very chunk this check just resolved from, so this is
                # always true exactly when present_in_retrieved_context is,
                # kept as its own field to match the ADR's schema.
                source_text_hash_matches=present_in_retrieved_context,
            )
        )
    return tuple(checks)


def _citation_checks_all_pass(checks: tuple[DeterministicCitationCheck, ...]) -> bool:
    return bool(checks) and all(
        check.well_formed
        and check.exists_in_corpus
        and check.present_in_retrieved_context
        and check.belongs_to_selected_document_version
        and check.source_text_hash_matches
        for check in checks
    )


def _claim_is_supported(claim_audit: ClaimAudit) -> bool:
    if not _citation_checks_all_pass(claim_audit.citation_checks):
        return False
    support = claim_audit.semantic_support
    return support is not None and support.verdict == SupportVerdict.SUPPORTED


def _all_material_claims_supported(claims: tuple[ClaimAudit, ...]) -> bool:
    material = [claim_audit for claim_audit in claims if claim_audit.claim.material]
    if not material:
        return True
    return all(_claim_is_supported(claim_audit) for claim_audit in material)


def _any_material_claim_supported(claims: tuple[ClaimAudit, ...]) -> bool:
    return any(
        _claim_is_supported(claim_audit) for claim_audit in claims if claim_audit.claim.material
    )


def _audit_claims(
    question: str,
    answer_text: str,
    retrieved_results: list[RetrievalResult],
    corpus_structural_ids: dict[str, set[str]],
    document_versions: dict[str, str],
    verifier: SemanticSupportVerifier,
) -> tuple[ClaimAudit, ...]:
    audits = []
    for claim in segment_claims(answer_text):
        checks = run_deterministic_checks(
            claim, retrieved_results, corpus_structural_ids, document_versions
        )
        semantic_support = None
        if _citation_checks_all_pass(checks):
            cited_citations = tuple(
                (raw_id, text)
                for raw_id in claim.cited_structural_ids
                if (text := _resolve_chunk_source_text(raw_id, retrieved_results)) is not None
            )
            semantic_support = verifier.verify(question, claim.text, cited_citations)
        audits.append(
            ClaimAudit(claim=claim, citation_checks=checks, semantic_support=semantic_support)
        )
    return tuple(audits)


def _valid_source_texts(
    claims: tuple[ClaimAudit, ...], retrieved_results: list[RetrievalResult]
) -> tuple[str, ...]:
    """Return every cited source text whose citation passed all deterministic checks.

    This is what the rewriter is allowed to draw on ("valid retrieved
    source text", ADR-0016) - a citation that failed deterministic checks
    contributes nothing here, so the rewrite prompt never sees text
    resolved from an invented or out-of-context ID.
    """
    texts: dict[str, None] = {}
    for claim_audit in claims:
        for check in claim_audit.citation_checks:
            if (
                check.well_formed
                and check.exists_in_corpus
                and check.present_in_retrieved_context
                and check.belongs_to_selected_document_version
                and check.source_text_hash_matches
            ):
                text = _resolve_chunk_source_text(check.structural_id, retrieved_results)
                if text is not None:
                    texts.setdefault(text, None)
    return tuple(texts)


def audit_answer(
    question: str,
    answer_text: str,
    retrieved_results: list[RetrievalResult],
    corpus_structural_ids: dict[str, set[str]],
    document_versions: dict[str, str],
    verifier: SemanticSupportVerifier,
    rewriter: AnswerRewriter,
) -> AuditResult:
    """Run the complete ADR-0016 audit pipeline for one generated answer.

    At most one rewrite is attempted (ADR-0016: "no recursive correction").
    A second failed audit abstains (nothing material survives), caveats (a
    supported subset survives), or - if the verifier/rewriter call itself
    fails rather than merely disagreeing - reports ``AUDIT_FAILED``. Never a
    second rewrite attempt regardless of which of these happens.
    """

    def _failed(claims: tuple[ClaimAudit, ...] = ()) -> AuditResult:
        return AuditResult(
            outcome=AuditOutcome.AUDIT_FAILED,
            claims=claims,
            original_answer=answer_text,
            final_answer=answer_text,
            rewritten=False,
            temporal_status=TemporalStatus.UNKNOWN,
            verifier_identity=None,
        )

    try:
        claims = _audit_claims(
            question,
            answer_text,
            retrieved_results,
            corpus_structural_ids,
            document_versions,
            verifier,
        )
    except Exception:
        return _failed()

    if _all_material_claims_supported(claims):
        return AuditResult(
            outcome=AuditOutcome.ACCEPTED,
            claims=claims,
            original_answer=answer_text,
            final_answer=answer_text,
            rewritten=False,
            temporal_status=TemporalStatus.UNKNOWN,
            verifier_identity=verifier.identity,
        )

    try:
        rewritten_text = rewriter.rewrite(
            question, answer_text, _valid_source_texts(claims, retrieved_results), claims
        )
    except Exception:
        return _failed(claims)

    try:
        second_claims = _audit_claims(
            question,
            rewritten_text,
            retrieved_results,
            corpus_structural_ids,
            document_versions,
            verifier,
        )
    except Exception:
        return _failed(claims)

    if _all_material_claims_supported(second_claims):
        outcome, final_answer = AuditOutcome.ACCEPTED, rewritten_text
    elif _any_material_claim_supported(second_claims):
        outcome, final_answer = AuditOutcome.ACCEPTED_WITH_CAVEAT, rewritten_text
    else:
        outcome, final_answer = AuditOutcome.ABSTAIN, _ABSTENTION_MESSAGE

    return AuditResult(
        outcome=outcome,
        claims=second_claims,
        original_answer=answer_text,
        final_answer=final_answer,
        rewritten=True,
        temporal_status=TemporalStatus.UNKNOWN,
        verifier_identity=verifier.identity,
    )
