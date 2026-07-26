"""Streamlit analytical dashboard for published RAGForge benchmark runs."""

from pathlib import Path

import streamlit as st

from ragforge.adapters.published_benchmarks import JsonPublishedBenchmarkRepository
from ragforge.application.benchmark_dashboard import strategy_metric_rows
from ragforge.entrypoints.dashboard_content import (
    Language,
    dashboard_copy,
    metric_explanations,
    technique_explanations,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Render the latest explicitly published benchmark without provider calls."""
    st.set_page_config(page_title="RAGForge Benchmark", page_icon="⚒️", layout="wide")
    repository = JsonPublishedBenchmarkRepository(_REPOSITORY_ROOT)
    runs = repository.list_runs()
    run_by_label = {f"{run.run_id} — {run.title}": run for run in runs}

    language_label = st.sidebar.selectbox(
        "Idioma / Language",
        options=("Português", "English"),
    )
    language: Language = "pt" if language_label == "Português" else "en"
    copy = dashboard_copy(language)
    techniques = technique_explanations(language)
    metrics = metric_explanations(language)

    st.title(copy.title)
    selected_label = st.selectbox(copy.published_run, options=list(run_by_label))
    run = run_by_label[selected_label]

    st.caption(
        copy.sample_caption.format(
            question_count=run.sample.question_count,
            source_count=run.sample.source_split_question_count,
            seed=run.sample.sampling_seed,
            k=run.k,
        )
    )
    st.success(
        f"{copy.recommendation}: **{run.recommendation.strategy.upper()}** — "
        f"{copy.recommendation_rationale.format(strategy=techniques[run.recommendation.strategy].title)}"
    )
    st.info(
        copy.recommendation_scope.format(
            question_count=run.sample.question_count,
        )
    )

    embedding, generation, judge = st.columns(3)
    embedding.metric(copy.retrieval_embedding, run.embedding_model)
    generation.metric(copy.answer_generation, run.generation_model)
    judge.metric(copy.independent_judge, run.judge_model)

    rows = strategy_metric_rows(run)
    overview_tab, techniques_tab, metrics_tab, evidence_tab = st.tabs(
        (copy.overview, copy.techniques, copy.indicators, copy.evidence)
    )

    with overview_tab:
        table = [
            {
                copy.strategy: techniques[row.strategy].title,
                metrics["recall_at_5"].label: row.recall_at_5,
                metrics["precision_at_5"].label: row.precision_at_5,
                metrics["ndcg_at_5"].label: row.ndcg_at_5,
                metrics["mrr"].label: row.mrr,
                metrics["document_mismatch_at_5"].label: row.document_mismatch_at_5,
                metrics["citation_accuracy"].label: row.citation_accuracy,
                metrics["faithfulness"].label: row.faithfulness,
                metrics["answer_relevancy"].label: row.answer_relevancy,
                metrics["abstention"].label: row.abstention,
            }
            for row in rows
        ]
        st.subheader(copy.strategy_comparison)
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                metric.label: st.column_config.NumberColumn(format="%.3f")
                for metric in metrics.values()
            },
        )

        chart_data = {
            techniques[row.strategy].title: {
                metrics["ndcg_at_5"].label: row.ndcg_at_5,
                metrics["citation_accuracy"].label: row.citation_accuracy,
                metrics["faithfulness"].label: row.faithfulness,
                metrics["answer_relevancy"].label: row.answer_relevancy,
            }
            for row in rows
        }
        st.subheader(copy.quality_profile)
        st.bar_chart(chart_data, horizontal=True)

    with techniques_tab:
        for technique in techniques.values():
            with st.expander(
                f"{technique.title} — {technique.summary}",
                expanded=technique.title.startswith("SAC ("),
            ):
                st.markdown(f"**{copy.how_it_works}:** {technique.how_it_works}")
                st.markdown(f"**{copy.strengths}:** {technique.strengths}")
                st.markdown(f"**{copy.limitations}:** {technique.limitations}")

    with metrics_tab:
        st.info(copy.scale_note)
        for metric in metrics.values():
            with st.expander(metric.label):
                st.write(metric.definition)
                st.markdown(f"**{copy.interpretation}:** {metric.interpretation}")

    with evidence_tab:
        st.subheader(copy.verify_evidence)
        st.code(
            f"uv run python scripts/verify_run.py {run.run_id}",
            language="bash",
        )
        st.write(f"{copy.evidence_path}: `{run.evidence_path}`")
        st.write(f"{copy.results_path}: `{run.results_path}`")
        st.write(f"{copy.git_revision}: `{run.git_sha}`")
