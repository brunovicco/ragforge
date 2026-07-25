"""Pure reporting/record-building functions for run.py (ADR-0004/ADR-0007/ADR-0018).

Split out of run.py (which grew past 1200 lines): fixed-width table
rendering and assembling the JSON-serializable run record. No I/O, no
external adapter - every function here takes already-computed data and
returns a string or a dict.
"""

from collections import defaultdict
from statistics import mean

from ragforge.domain.models import Judgment
from ragforge.embeddings.identity import EmbeddingIdentity
from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.evaluation.ragas_judge import ABSTENTION_PROMPT_VERSION
from ragforge.evaluation.records import QuestionRecord

# No reranker model has been chosen via a dedicated comparison (unlike the
# embedding model, ADR-0005); a placeholder, not a data-driven winner -
# already used and verified in test_cross_encoder_reranker.py.
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CONTEXTUALIZER_MODEL = "gemini-3.1-flash-lite"
_SUMMARIZER_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_LLM_MODEL = "gemini-3.1-flash-lite"
_GRAPHRAG_MODE = "local"


def _aggregate_record_group(records: list[QuestionRecord]) -> dict[str, object]:
    """Aggregate every available numeric metric with explicit coverage."""
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for metric, value in record.metrics.items():
            values_by_metric[metric].append(value)
    return {
        "selected": len(records),
        "retrieval_succeeded": sum(record.retrieval_status == "succeeded" for record in records),
        "answer_succeeded": sum(record.generation_status == "succeeded" for record in records),
        "metrics": {
            metric: {"mean": mean(values), "n": len(values)}
            for metric, values in sorted(values_by_metric.items())
        },
    }


def build_metric_breakdowns(
    records: list[QuestionRecord],
    judgments: list[Judgment],
) -> dict[str, object]:
    """Aggregate per-strategy metrics by query class and relevant document."""
    judgment_by_id = {judgment.question_id: judgment for judgment in judgments}
    by_class: dict[tuple[str, str], list[QuestionRecord]] = defaultdict(list)
    by_document: dict[tuple[str, str], list[QuestionRecord]] = defaultdict(list)
    for record in records:
        by_class[(record.strategy, record.query_class or "unknown")].append(record)
        judgment = judgment_by_id[record.question_id]
        documents = (
            {judged.ref.norm for judged in judgment.relevant_refs}
            if judgment.relevant_refs
            else {"__unanswerable__"}
        )
        for document in documents:
            by_document[(record.strategy, document)].append(record)

    def _render(
        grouped: dict[tuple[str, str], list[QuestionRecord]],
    ) -> dict[str, dict[str, object]]:
        rendered: dict[str, dict[str, object]] = {}
        for (strategy, dimension), grouped_records in sorted(grouped.items()):
            rendered.setdefault(strategy, {})[dimension] = _aggregate_record_group(grouped_records)
        return rendered

    return {
        "by_query_class": _render(by_class),
        "by_document": _render(by_document),
    }


def summarize_generation_usage(
    lineage_by_strategy: dict[str, list[GenerationLineage]],
    *,
    input_price_per_million_usd: float | None,
    output_price_per_million_usd: float | None,
) -> dict[str, dict[str, float | int | None]]:
    """Summarize generation-only latency, tokens, cache hits, and optional cost."""
    summaries: dict[str, dict[str, float | int | None]] = {}
    for strategy, entries in sorted(lineage_by_strategy.items()):
        prompt_tokens = sum(entry.prompt_tokens or 0 for entry in entries)
        completion_tokens = sum(entry.completion_tokens or 0 for entry in entries)
        estimated_cost: float | None = None
        if input_price_per_million_usd is not None and output_price_per_million_usd is not None:
            estimated_cost = (
                prompt_tokens * input_price_per_million_usd
                + completion_tokens * output_price_per_million_usd
            ) / 1_000_000
        summaries[strategy] = {
            "calls": len(entries),
            "latency_seconds": sum(entry.latency_seconds for entry in entries),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": sum(entry.total_tokens or 0 for entry in entries),
            "cache_hits": sum(entry.cache_hit for entry in entries),
            "estimated_cost_usd": estimated_cost,
        }
    return summaries


def format_results_table(
    strategy_labels: list[str], run_metrics: dict[str, dict[str, float]]
) -> str:
    """Render a fixed-width recall/precision/nDCG/MRR/DRM@k table for the given strategy order."""
    header = (
        f"{'strategy':<14} {'recall@k':>9} {'precision@k':>12} {'ndcg@k':>8} {'mrr':>6} "
        f"{'drm@k':>7} {'n':>4} {'errors':>6}"
    )
    lines = [header]
    for label in strategy_labels:
        metrics = run_metrics.get(label)
        if metrics is None:
            lines.append(f"{label:<14} (not run)")
            continue
        lines.append(
            f"{label:<14} {metrics['recall_at_k']:>9.3f} {metrics['precision_at_k']:>12.3f} "
            f"{metrics['ndcg_at_k']:>8.3f} {metrics['mrr']:>6.3f} {metrics['drm_at_k']:>7.3f} "
            f"{metrics['n']:>4.0f} {metrics['errors']:>6.0f}"
        )
    return "\n".join(lines)


def format_answer_quality_table(
    strategy_labels: list[str], run_metrics: dict[str, dict[str, float]]
) -> str:
    """Render a Citation Accuracy/Faithfulness/Relevancy/abstention table (ADR-0007/ADR-0018)."""
    header = (
        f"{'strategy':<14} {'citation_acc':>12} {'faithfulness':>12} {'relevancy':>10} "
        f"{'abstention':>10} {'n':>4} {'errors':>6}"
    )
    lines = [header]
    for label in strategy_labels:
        metrics = run_metrics.get(label)
        if metrics is None or "citation_accuracy" not in metrics:
            lines.append(f"{label:<14} (not run)")
            continue
        lines.append(
            f"{label:<14} {metrics['citation_accuracy']:>12.3f} {metrics['faithfulness']:>12.3f} "
            f"{metrics['answer_relevancy']:>10.3f} {metrics['abstention_appropriate']:>10.3f} "
            f"{metrics['answer_n']:>4.0f} {metrics['answer_errors']:>6.0f}"
        )
    return "\n".join(lines)


def build_run_record(
    *,
    run_id: str,
    mode: str,
    config_path: str,
    embedding_identity: EmbeddingIdentity,
    index_namespace: str,
    generation_model: str,
    judge_provider: str,
    judge_model: str,
    judge_reasoning_effort: str | None,
    audit_enabled: bool,
    audit_provider: str | None,
    audit_model: str | None,
    corpus_version: str,
    split_dataset_version: str,
    n_chunks: int,
    top_k: int,
    run_metrics: dict[str, dict[str, float]],
    metric_breakdowns: dict[str, object] | None = None,
    stage_durations_seconds: dict[str, float] | None = None,
    generation_usage: dict[str, dict[str, float | int | None]] | None = None,
) -> dict[str, object]:
    """Assemble the JSON-serializable run record written to experiments/<run_id>/results.json.

    ``judge_label`` is "exploratory_same_provider_judge" whenever
    ``judge_provider == "gemini"`` (ADR-0018): the answer generator is always
    Gemini-based (GeminiAnswerGenerator), so a Gemini judge is never
    independent from it. ``None`` for the canonical "openai" judge.

    ``audit_enabled``/``audit_provider``/``audit_model`` are always present
    (ADR-0016: audit calls must be identified) - ``provider``/``model`` are
    ``None`` when auditing is off, never silently omitted.
    """
    judge_label = "exploratory_same_provider_judge" if judge_provider == "gemini" else None
    return {
        "run_id": run_id,
        "mode": mode,
        "config_path": config_path,
        "embedding": {
            "provider": embedding_identity.provider,
            "model": embedding_identity.model,
            "revision": embedding_identity.revision,
            "dimensions": embedding_identity.dimensions,
            "normalize": embedding_identity.normalize,
            "runtime": embedding_identity.runtime,
        },
        "index_namespace": index_namespace,
        "reranker_model": _RERANKER_MODEL,
        "contextualizer_model": _CONTEXTUALIZER_MODEL,
        "summarizer_model": _SUMMARIZER_MODEL,
        "graphrag_llm_model": _GRAPHRAG_LLM_MODEL,
        "graphrag_mode": _GRAPHRAG_MODE,
        "generation_model": generation_model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_reasoning_effort": judge_reasoning_effort,
        "judge_prompt_version": ABSTENTION_PROMPT_VERSION,
        "judge_label": judge_label,
        "audit_enabled": audit_enabled,
        "audit_provider": audit_provider,
        "audit_model": audit_model,
        "corpus_version": corpus_version,
        "split_dataset_version": split_dataset_version,
        "k": top_k,
        "n_chunks": n_chunks,
        "metrics": run_metrics,
        "metric_breakdowns": metric_breakdowns or {},
        "stage_durations_seconds": stage_durations_seconds or {},
        "generation_usage": generation_usage or {},
        "records_path": "records.jsonl",
    }
