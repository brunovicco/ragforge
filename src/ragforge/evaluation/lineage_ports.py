"""Evidence lineage schemas and pure builders (ADR-0017).

A reviewer should be able to trace every published score back to exact
inputs and model identities. This module defines the immutable records that
make that possible - retrieval candidate lineage, generation lineage, audit/
judge lineage, the hash-chained event envelope, and the run manifest - plus
two pure functions (``build_audit_lineage``, ``build_judge_lineage``) that
derive audit/judge lineage entirely from data the ADR-0016/ADR-0018
pipelines (citation_audit.py, ragas_judge.py) already compute, without
touching either module.

No LLM SDK or framework import here (ADR-0009's adapter boundary) - only
dataclasses and the existing audit_ports.py/judge_ports.py schemas.
"""

from dataclasses import dataclass

from ragforge.evaluation.audit_ports import AuditResult, ClaimAudit
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.judge_ports import JudgeResult, ModelIdentity


@dataclass(frozen=True, slots=True)
class RetrievalCandidateLineage:
    """One retrieved candidate's rank, score, and embedding identity (ADR-0017)."""

    query_id: str
    strategy: str
    embedding_identity_hash: str
    candidate_rank: int
    chunk_id: str
    structural_ids: tuple[str, ...]
    raw_score: float


@dataclass(frozen=True, slots=True)
class GenerationLineage:
    """One answer-generation call's full provenance (ADR-0017): the answer generator only.

    Token usage/latency/cache_hit are scoped to the answer generator by the
    ADR's own field list - audit and judge lineage do not carry these
    fields (see AuditLineage/JudgeLineage below).
    """

    provider: str
    model: str
    prompt_hash: str
    input_chunk_ids: tuple[str, ...]
    input_source_hashes: tuple[str, ...]
    answer_hash: str
    parsed_citations: tuple[str, ...]
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class AuditLineage:
    """One post-generation citation audit's provenance (ADR-0016/ADR-0017)."""

    outcome: str
    rewrite_count: int
    original_answer_hash: str
    final_answer_hash: str
    verifier_identity_hash: str | None
    claim_checks_summary: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgeLineage:
    """One answer-quality judge call's provenance (ADR-0018/ADR-0017)."""

    provider: str
    model: str
    prompt_hash: str
    metric_implementation_version: int
    raw_structured_output_hash: str
    final_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """One hash-chained event in a run's events.jsonl (ADR-0017).

    ``event_hash`` covers the canonical serialization of every other field
    in this envelope; ``previous_event_hash`` links to the prior event,
    forming a local tamper-evident chain (event_log.py builds this, never
    hand-constructed elsewhere).
    """

    schema_version: int
    sequence: int
    event_id: str
    run_id: str
    correlation_id: str
    stage: str
    event_type: str
    occurred_at: str
    payload_hash: str
    previous_event_hash: str | None
    event_hash: str


@dataclass(frozen=True, slots=True)
class RunManifest:
    """The top-level, versioned manifest for one run's evidence directory (ADR-0017)."""

    schema_version: int
    run_id: str
    status: str
    git_sha: str
    started_at: str
    completed_at: str | None
    corpus_hash: str
    dataset_hash: str
    split_hash: str
    configuration_hash: str
    models: dict[str, str]
    strategies: tuple[str, ...]
    execution: dict[str, object]
    artifact_root_hash: str | None


def _claim_status(claim_audit: ClaimAudit) -> str:
    """Return a short human-readable status for one claim, for AuditLineage's summary."""
    if any(not check.well_formed for check in claim_audit.citation_checks):
        return "malformed_citation"
    if any(
        not (check.exists_in_corpus and check.present_in_retrieved_context)
        for check in claim_audit.citation_checks
    ):
        return "citation_not_verifiable"
    if claim_audit.semantic_support is None:
        return "not_semantically_verified"
    return claim_audit.semantic_support.verdict.value


def build_audit_lineage(audit_result: AuditResult) -> AuditLineage:
    """Derive an AuditLineage entirely from an already-computed AuditResult (ADR-0016/0017)."""
    verifier_identity_hash = (
        canonical_json_hash(
            {
                "provider": audit_result.verifier_identity.provider,
                "model": audit_result.verifier_identity.model,
                "reasoning_effort": audit_result.verifier_identity.reasoning_effort,
                "output_schema_version": audit_result.verifier_identity.output_schema_version,
            }
        )
        if audit_result.verifier_identity is not None
        else None
    )
    return AuditLineage(
        outcome=audit_result.outcome.value,
        rewrite_count=1 if audit_result.rewritten else 0,
        original_answer_hash=canonical_json_hash(audit_result.original_answer),
        final_answer_hash=canonical_json_hash(audit_result.final_answer),
        verifier_identity_hash=verifier_identity_hash,
        claim_checks_summary=tuple(
            f"{claim_audit.claim.claim_id}:{_claim_status(claim_audit)}"
            for claim_audit in audit_result.claims
        ),
    )


def build_judge_lineage(
    judge_result: JudgeResult,
    identity: ModelIdentity,
    prompt_hash: str,
    final_metrics: dict[str, float],
) -> JudgeLineage:
    """Derive a JudgeLineage entirely from an already-computed JudgeResult (ADR-0018/0017).

    ``prompt_hash`` is supplied by the caller: JudgeResult itself carries no
    prompt text or hash (ragas_judge.py's cache key already computes one
    internally but doesn't expose it) - callers pass the same prompt-version
    constant already used for cache keying (e.g. ABSTENTION_PROMPT_VERSION).
    """
    return JudgeLineage(
        provider=identity.provider,
        model=identity.model,
        prompt_hash=prompt_hash,
        metric_implementation_version=judge_result.schema_version,
        raw_structured_output_hash=canonical_json_hash(
            {
                "faithfulness": judge_result.faithfulness.score,
                "answer_relevancy": judge_result.answer_relevancy.score,
                "abstention_appropriate": judge_result.abstention.appropriate,
                "abstention_rationale": judge_result.abstention.rationale,
            }
        ),
        final_metrics=final_metrics,
    )
