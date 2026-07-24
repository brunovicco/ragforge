"""Tests for deterministic aggregate audit rates (ADR-0016)."""

from ragforge.evaluation.audit_metrics import (
    abstention_rate,
    citation_not_in_context_rate,
    compute_audit_report,
    malformed_citation_rate,
    nonexistent_citation_rate,
    rewrite_rate,
    rewrite_success_rate,
    uncited_material_claim_rate,
    unsupported_claim_rate,
)
from ragforge.evaluation.audit_ports import (
    AnswerClaim,
    AuditOutcome,
    AuditResult,
    ClaimAudit,
    DeterministicCitationCheck,
    SemanticSupportResult,
    SupportVerdict,
    TemporalStatus,
)

ART_1 = "LC-105/2001::art-1"


def _check(
    well_formed: bool = True,
    exists_in_corpus: bool = True,
    belongs_to_selected_document_version: bool = True,
    present_in_retrieved_context: bool = True,
    source_text_hash_matches: bool = True,
) -> DeterministicCitationCheck:
    return DeterministicCitationCheck(
        structural_id=ART_1,
        well_formed=well_formed,
        exists_in_corpus=exists_in_corpus,
        belongs_to_selected_document_version=belongs_to_selected_document_version,
        present_in_retrieved_context=present_in_retrieved_context,
        source_text_hash_matches=source_text_hash_matches,
    )


def _claim(cited: tuple[str, ...] = (ART_1,), material: bool = True) -> AnswerClaim:
    return AnswerClaim(
        claim_id="claim-0",
        text="texto",
        cited_structural_ids=cited,
        sentence_index=0,
        material=material,
    )


def _support(verdict: SupportVerdict) -> SemanticSupportResult:
    return SemanticSupportResult(
        verdict=verdict,
        rationale="r",
        supported_citation_ids=(),
        unsupported_citation_ids=(),
        missing_evidence=(),
    )


def _result(
    claims: tuple[ClaimAudit, ...],
    outcome: AuditOutcome = AuditOutcome.ACCEPTED,
    rewritten: bool = False,
) -> AuditResult:
    return AuditResult(
        outcome=outcome,
        claims=claims,
        original_answer="original",
        final_answer="final",
        rewritten=rewritten,
        temporal_status=TemporalStatus.UNKNOWN,
        verifier_identity=None,
    )


def test_malformed_citation_rate_counts_not_well_formed_checks() -> None:
    """One malformed check out of two gives 0.5."""
    claim = ClaimAudit(
        claim=_claim(), citation_checks=(_check(well_formed=False), _check()), semantic_support=None
    )
    assert malformed_citation_rate([_result((claim,))]) == 0.5


def test_malformed_citation_rate_is_zero_with_no_citations() -> None:
    """No citations at all means nothing to be malformed."""
    claim = ClaimAudit(claim=_claim(cited=()), citation_checks=(), semantic_support=None)
    assert malformed_citation_rate([_result((claim,))]) == 0.0


def test_nonexistent_citation_rate_only_considers_well_formed_checks() -> None:
    """A malformed check is excluded from this rate's denominator entirely."""
    claim = ClaimAudit(
        claim=_claim(),
        citation_checks=(_check(well_formed=False), _check(exists_in_corpus=False)),
        semantic_support=None,
    )
    assert nonexistent_citation_rate([_result((claim,))]) == 1.0


def test_citation_not_in_context_rate_flags_the_adr_motivating_case() -> None:
    """A real, existing citation never surfaced by retrieval is the core ADR-0016 scenario."""
    claim = ClaimAudit(
        claim=_claim(),
        citation_checks=(_check(present_in_retrieved_context=False),),
        semantic_support=None,
    )
    assert citation_not_in_context_rate([_result((claim,))]) == 1.0


def test_uncited_material_claim_rate_counts_material_claims_with_no_citations() -> None:
    """One uncited material claim out of two gives 0.5."""
    cited = ClaimAudit(
        claim=_claim(cited=(ART_1,)), citation_checks=(_check(),), semantic_support=None
    )
    uncited = ClaimAudit(claim=_claim(cited=()), citation_checks=(), semantic_support=None)
    assert uncited_material_claim_rate([_result((cited, uncited))]) == 0.5


def test_uncited_material_claim_rate_ignores_non_material_claims() -> None:
    """A non-material claim (e.g. an abstention sentence) never counts, cited or not."""
    non_material = ClaimAudit(
        claim=_claim(cited=(), material=False), citation_checks=(), semantic_support=None
    )
    assert uncited_material_claim_rate([_result((non_material,))]) == 0.0


def test_unsupported_claim_rate_only_considers_semantically_verified_claims() -> None:
    """A claim that failed deterministic checks (never verified) is excluded from this rate."""
    unverified = ClaimAudit(
        claim=_claim(), citation_checks=(_check(exists_in_corpus=False),), semantic_support=None
    )
    verified_unsupported = ClaimAudit(
        claim=_claim(),
        citation_checks=(_check(),),
        semantic_support=_support(SupportVerdict.UNSUPPORTED),
    )
    assert unsupported_claim_rate([_result((unverified, verified_unsupported))]) == 1.0


def test_unsupported_claim_rate_is_zero_when_all_verified_claims_are_supported() -> None:
    """Every verified claim being SUPPORTED gives a 0.0 unsupported rate."""
    claim = ClaimAudit(
        claim=_claim(),
        citation_checks=(_check(),),
        semantic_support=_support(SupportVerdict.SUPPORTED),
    )
    assert unsupported_claim_rate([_result((claim,))]) == 0.0


def test_rewrite_rate_counts_results_marked_rewritten() -> None:
    """One rewritten result out of two gives 0.5."""
    claim = ClaimAudit(claim=_claim(), citation_checks=(_check(),), semantic_support=None)
    results = [_result((claim,), rewritten=True), _result((claim,), rewritten=False)]
    assert rewrite_rate(results) == 0.5


def test_rewrite_success_rate_counts_deliverable_outcomes_among_rewritten_results() -> None:
    """Only rewritten results count; ACCEPTED/ACCEPTED_WITH_CAVEAT are success, ABSTAIN isn't."""
    claim = ClaimAudit(claim=_claim(), citation_checks=(_check(),), semantic_support=None)
    results = [
        _result((claim,), outcome=AuditOutcome.ACCEPTED, rewritten=True),
        _result((claim,), outcome=AuditOutcome.ABSTAIN, rewritten=True),
        _result((claim,), outcome=AuditOutcome.ACCEPTED, rewritten=False),
    ]
    assert rewrite_success_rate(results) == 0.5


def test_rewrite_success_rate_is_zero_when_nothing_was_rewritten() -> None:
    """No rewrite attempts at all means nothing to measure success over."""
    claim = ClaimAudit(claim=_claim(), citation_checks=(_check(),), semantic_support=None)
    assert rewrite_success_rate([_result((claim,), rewritten=False)]) == 0.0


def test_abstention_rate_counts_abstain_outcomes() -> None:
    """One abstained result out of two gives 0.5."""
    claim = ClaimAudit(claim=_claim(), citation_checks=(_check(),), semantic_support=None)
    results = [
        _result((claim,), outcome=AuditOutcome.ABSTAIN),
        _result((claim,), outcome=AuditOutcome.ACCEPTED),
    ]
    assert abstention_rate(results) == 0.5


def test_all_rates_are_zero_for_an_empty_result_list() -> None:
    """No results at all means every rate is 0.0, never a division error."""
    assert malformed_citation_rate([]) == 0.0
    assert nonexistent_citation_rate([]) == 0.0
    assert citation_not_in_context_rate([]) == 0.0
    assert uncited_material_claim_rate([]) == 0.0
    assert unsupported_claim_rate([]) == 0.0
    assert rewrite_rate([]) == 0.0
    assert rewrite_success_rate([]) == 0.0
    assert abstention_rate([]) == 0.0


def test_compute_audit_report_aggregates_every_rate() -> None:
    """The combined report includes every rate this module computes, plus n."""
    claim = ClaimAudit(
        claim=_claim(),
        citation_checks=(_check(),),
        semantic_support=_support(SupportVerdict.SUPPORTED),
    )
    report = compute_audit_report([_result((claim,))])

    assert set(report) == {
        "malformed_citation_rate",
        "nonexistent_citation_rate",
        "citation_not_in_context_rate",
        "uncited_material_claim_rate",
        "unsupported_claim_rate",
        "rewrite_rate",
        "rewrite_success_rate",
        "abstention_rate",
        "n",
    }
    assert report["n"] == 1.0
