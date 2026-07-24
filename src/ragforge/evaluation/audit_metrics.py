"""Deterministic aggregate rates over post-generation citation audits (ADR-0016).

Every rate here is computed purely from AuditResult/ClaimAudit data already
produced by citation_audit.py - no LLM call, no human label. ``false
abstention rate`` from the ADR's own Report list is not implemented: it
needs a human judgment of what SHOULD have been abstained on, the same
human-curation constraint every calibration effort in this project runs
into (see judge_calibration.py). Latency and token usage are the concern of
the adapters that make the calls (openai_semantic_verifier.py,
openai_answer_rewriter.py), not this module.

Lives directly under evaluation/, not evaluation/metrics/: that package's
architecture boundary (ADR-0009) restricts it to domain-only imports, but
this module needs AuditResult/ClaimAudit from audit_ports.py - the same
reason judge_calibration.py (which needs its own CalibrationSample) also
sits at this level rather than under metrics/.
"""

from ragforge.evaluation.audit_ports import AuditOutcome, AuditResult, ClaimAudit, SupportVerdict


def malformed_citation_rate(results: list[AuditResult]) -> float:
    """Fraction of all cited structural IDs across every claim that are not well-formed.

    0.0 when there are no citations at all - nothing to be malformed.
    """
    checks = [
        check for result in results for claim in result.claims for check in claim.citation_checks
    ]
    if not checks:
        return 0.0
    return sum(1 for check in checks if not check.well_formed) / len(checks)


def nonexistent_citation_rate(results: list[AuditResult]) -> float:
    """Fraction of well-formed citations that don't exist anywhere in the indexed corpus.

    0.0 when there are no well-formed citations to check.
    """
    well_formed = [
        check
        for result in results
        for claim in result.claims
        for check in claim.citation_checks
        if check.well_formed
    ]
    if not well_formed:
        return 0.0
    return sum(1 for check in well_formed if not check.exists_in_corpus) / len(well_formed)


def citation_not_in_context_rate(results: list[AuditResult]) -> float:
    """Fraction of real, existing citations that were never surfaced by this question's retrieval.

    This is the ADR-0016 motivating case: a valid, existing structural ID
    cited by the generator without ever having been retrieved. 0.0 when
    there are no existing citations to check.
    """
    existing = [
        check
        for result in results
        for claim in result.claims
        for check in claim.citation_checks
        if check.well_formed and check.exists_in_corpus
    ]
    if not existing:
        return 0.0
    return sum(1 for check in existing if not check.present_in_retrieved_context) / len(existing)


def uncited_material_claim_rate(results: list[AuditResult]) -> float:
    """Fraction of material claims that cite no structural ID at all.

    0.0 when there are no material claims.
    """
    material_claims = [
        claim_audit.claim
        for result in results
        for claim_audit in result.claims
        if claim_audit.claim.material
    ]
    if not material_claims:
        return 0.0
    return sum(1 for claim in material_claims if not claim.cited_structural_ids) / len(
        material_claims
    )


def unsupported_claim_rate(results: list[AuditResult]) -> float:
    """Fraction of semantically-verified material claims the verifier did not call fully supported.

    Scoped to claims that actually reached the semantic verifier (citations
    already passed every deterministic check) - a claim that failed a
    deterministic check is already counted by the citation-level rates
    above, not here. 0.0 when no material claim was ever verified.
    """
    verified = [
        claim_audit
        for result in results
        for claim_audit in result.claims
        if claim_audit.claim.material and claim_audit.semantic_support is not None
    ]
    if not verified:
        return 0.0
    return sum(1 for claim_audit in verified if _is_unsupported(claim_audit)) / len(verified)


def _is_unsupported(claim_audit: ClaimAudit) -> bool:
    support = claim_audit.semantic_support
    if support is None:
        return False
    return support.verdict != SupportVerdict.SUPPORTED


def rewrite_rate(results: list[AuditResult]) -> float:
    """Fraction of audited answers that triggered a rewrite. 0.0 for an empty result list."""
    if not results:
        return 0.0
    return sum(1 for result in results if result.rewritten) / len(results)


def rewrite_success_rate(results: list[AuditResult]) -> float:
    """Fraction of rewritten answers that ended up deliverable (accepted, with or without caveat).

    0.0 when no rewrite was ever attempted.
    """
    rewritten = [result for result in results if result.rewritten]
    if not rewritten:
        return 0.0
    deliverable = {AuditOutcome.ACCEPTED, AuditOutcome.ACCEPTED_WITH_CAVEAT}
    return sum(1 for result in rewritten if result.outcome in deliverable) / len(rewritten)


def abstention_rate(results: list[AuditResult]) -> float:
    """Fraction of audited answers whose final outcome is abstention. 0.0 for an empty list."""
    if not results:
        return 0.0
    return sum(1 for result in results if result.outcome == AuditOutcome.ABSTAIN) / len(results)


def compute_audit_report(results: list[AuditResult]) -> dict[str, float]:
    """Aggregate every ADR-0016 audit rate this module can compute, over one strategy's results."""
    return {
        "malformed_citation_rate": malformed_citation_rate(results),
        "nonexistent_citation_rate": nonexistent_citation_rate(results),
        "citation_not_in_context_rate": citation_not_in_context_rate(results),
        "uncited_material_claim_rate": uncited_material_claim_rate(results),
        "unsupported_claim_rate": unsupported_claim_rate(results),
        "rewrite_rate": rewrite_rate(results),
        "rewrite_success_rate": rewrite_success_rate(results),
        "abstention_rate": abstention_rate(results),
        "n": float(len(results)),
    }
