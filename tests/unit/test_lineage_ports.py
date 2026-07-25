"""Tests for pure lineage builders build_audit_lineage/build_judge_lineage (ADR-0017)."""

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
from ragforge.evaluation.judge_ports import (
    AbstentionJudgment,
    JudgeResult,
    MetricScore,
    ModelIdentity,
)
from ragforge.evaluation.lineage_ports import build_audit_lineage, build_judge_lineage


def _citation_check(
    structural_id: str = "id-1", *, well_formed: bool = True
) -> DeterministicCitationCheck:
    return DeterministicCitationCheck(
        structural_id=structural_id,
        well_formed=well_formed,
        exists_in_corpus=True,
        belongs_to_selected_document_version=True,
        present_in_retrieved_context=True,
        source_text_hash_matches=True,
    )


def _claim_audit(
    claim_id: str = "claim-0",
    *,
    citation_checks: tuple[DeterministicCitationCheck, ...] | None = None,
    semantic_support: SemanticSupportResult | None = None,
) -> ClaimAudit:
    claim = AnswerClaim(
        claim_id=claim_id,
        text="some claim text",
        cited_structural_ids=("id-1",),
        sentence_index=0,
        material=True,
    )
    return ClaimAudit(
        claim=claim,
        citation_checks=citation_checks if citation_checks is not None else (_citation_check(),),
        semantic_support=semantic_support,
    )


def test_build_audit_lineage_derives_answer_hashes_and_rewrite_count() -> None:
    """rewrite_count is 1 when rewritten, hashes reflect original vs. final answer text."""
    audit_result = AuditResult(
        outcome=AuditOutcome.ACCEPTED_WITH_CAVEAT,
        claims=(
            _claim_audit(
                semantic_support=SemanticSupportResult(
                    verdict=SupportVerdict.SUPPORTED,
                    rationale="ok",
                    supported_citation_ids=("id-1",),
                    unsupported_citation_ids=(),
                    missing_evidence=(),
                )
            ),
        ),
        original_answer="original text",
        final_answer="final text",
        rewritten=True,
        temporal_status=TemporalStatus.VALID,
        verifier_identity=ModelIdentity(
            provider="openai", model="gpt-test", reasoning_effort="low", output_schema_version=1
        ),
    )

    lineage = build_audit_lineage(audit_result)

    assert lineage.outcome == "accepted_with_caveat"
    assert lineage.rewrite_count == 1
    assert lineage.claim_checks_summary == ("claim-0:supported",)
    assert lineage.verifier_identity_hash is not None


def test_build_audit_lineage_reports_zero_rewrite_count_when_not_rewritten() -> None:
    """rewrite_count is 0 for an accepted answer that required no rewrite."""
    audit_result = AuditResult(
        outcome=AuditOutcome.ACCEPTED,
        claims=(),
        original_answer="same text",
        final_answer="same text",
        rewritten=False,
        temporal_status=TemporalStatus.NOT_APPLICABLE,
        verifier_identity=None,
    )

    lineage = build_audit_lineage(audit_result)

    assert lineage.rewrite_count == 0
    assert lineage.verifier_identity_hash is None
    assert lineage.original_answer_hash == lineage.final_answer_hash


def test_build_audit_lineage_flags_malformed_citation_before_semantic_support() -> None:
    """A malformed citation short-circuits the claim summary, even with no semantic_support run."""
    audit_result = AuditResult(
        outcome=AuditOutcome.ABSTAIN,
        claims=(
            _claim_audit(
                citation_checks=(_citation_check(well_formed=False),),
                semantic_support=None,
            ),
        ),
        original_answer="text",
        final_answer="text",
        rewritten=False,
        temporal_status=TemporalStatus.UNKNOWN,
        verifier_identity=None,
    )

    lineage = build_audit_lineage(audit_result)

    assert lineage.claim_checks_summary == ("claim-0:malformed_citation",)


def test_build_judge_lineage_derives_provider_model_and_final_metrics() -> None:
    """build_judge_lineage carries identity fields and the caller-supplied final metrics through."""
    judge_result = JudgeResult(
        schema_version=1,
        faithfulness=MetricScore(score=0.9),
        answer_relevancy=MetricScore(score=0.8),
        abstention=AbstentionJudgment(appropriate=True, rationale="no evidence"),
    )
    identity = ModelIdentity(
        provider="openai", model="gpt-test", reasoning_effort="low", output_schema_version=1
    )

    lineage = build_judge_lineage(
        judge_result,
        identity,
        prompt_hash="prompt-hash-v1",
        final_metrics={"faithfulness": 0.9, "answer_relevancy": 0.8},
    )

    assert lineage.provider == "openai"
    assert lineage.model == "gpt-test"
    assert lineage.prompt_hash == "prompt-hash-v1"
    assert lineage.metric_implementation_version == 1
    assert lineage.final_metrics == {"faithfulness": 0.9, "answer_relevancy": 0.8}


def test_build_judge_lineage_hashes_raw_structured_output_deterministically() -> None:
    """Two JudgeResults with identical content hash to the same raw_structured_output_hash."""
    identity = ModelIdentity(
        provider="openai", model="gpt-test", reasoning_effort=None, output_schema_version=1
    )

    def make_result() -> JudgeResult:
        return JudgeResult(
            schema_version=1,
            faithfulness=MetricScore(score=0.5),
            answer_relevancy=MetricScore(score=0.5),
            abstention=AbstentionJudgment(appropriate=False, rationale="rationale text"),
        )

    lineage_a = build_judge_lineage(make_result(), identity, prompt_hash="p", final_metrics={})
    lineage_b = build_judge_lineage(make_result(), identity, prompt_hash="p", final_metrics={})

    assert lineage_a.raw_structured_output_hash == lineage_b.raw_structured_output_hash
