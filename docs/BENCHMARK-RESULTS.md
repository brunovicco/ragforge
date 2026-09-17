# RAGForge v0.1 benchmark results

## Published run

| Field | Value |
| --- | --- |
| Run ID | `20260726T185553Z` |
| Git revision | `81029c6079df98dd6f6a771bcf5bf710c802efa0` |
| Corpus | 5 documents, 735 chunks |
| Dataset split | RegRAG-BR 0.2 test |
| Evaluation sample | 60 of 194 test questions |
| Sampling | Deterministic and stratified by query class |
| Seed | `regrag-br-benchmark-sample-v1` |
| Retrieval embedding | `gemini-embedding-001`, 1,536 dimensions |
| Answer generation | `gemini-3.1-flash-lite` |
| Independent judge | `gpt-5.4-mini-2026-03-17` |
| Semantic audit | Disabled |

This cost-controlled run is the final sampled benchmark for RAGForge v0.1. It
does not estimate results for the complete 194-question test split. The exact
question IDs are versioned in
`artifacts/runs/20260726T185553Z/question-selection.snapshot.json`.

Retrieval metrics use 57 answerable questions. The other three selected
questions are intentionally unanswerable and participate in answer-quality and
abstention evaluation, so all answer metrics use 60 questions.

## Scorecard

| Strategy | Recall@5 | Precision@5 | nDCG@5 | MRR | DRM@5 | Citation accuracy | Faithfulness | Answer relevancy | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.972 | 0.270 | 0.957 | 0.974 | 0.014 | 0.674 | 0.928 | **0.843** | **1.000** |
| Sparse BM25 | 0.918 | 0.239 | 0.857 | 0.868 | 0.074 | 0.631 | 0.943 | 0.790 | 0.950 |
| Hybrid + RRF | 0.965 | 0.260 | 0.932 | 0.949 | 0.035 | 0.674 | 0.951 | 0.834 | 0.967 |
| Reranked | 0.898 | 0.214 | 0.804 | 0.801 | 0.070 | 0.667 | 0.903 | 0.786 | 0.883 |
| Contextual | 0.968 | 0.267 | 0.956 | 0.982 | 0.032 | 0.678 | **0.968** | 0.840 | 0.983 |
| Parent-child | 0.972 | 0.301 | 0.957 | 0.974 | 0.015 | 0.657 | 0.952 | 0.833 | 0.967 |
| **SAC** | 0.975 | 0.274 | **0.963** | **0.991** | **0.000** | **0.689** | 0.956 | 0.840 | 0.967 |
| SAC + Contextual | 0.967 | 0.277 | 0.953 | 0.978 | **0.000** | 0.676 | 0.952 | 0.808 | 0.983 |
| RAPTOR | **1.000** | **0.611** | 0.945 | 0.988 | 0.004 | 0.657 | 0.953 | 0.826 | **1.000** |
| GraphRAG | 0.565 | 0.158 | 0.543 | 0.564 | 0.109 | 0.399 | 0.817 | 0.524 | 0.633 |

All ten strategies completed without retrieval or answer-evaluation errors.

## Interpretation

SAC had the strongest balanced profile in this v0.1 sample rather than winning
every individual metric:

- highest nDCG@5, MRR, and Citation Accuracy;
- zero Document-Level Retrieval Mismatch;
- Recall@5 above the Dense baseline;
- strong Faithfulness and Answer Relevancy without RAPTOR's synthetic-node
  evidence trade-off.

RAPTOR produced the strongest raw structural recall in this sample, but its
recursive summaries can enter answer-generation context. Dense remains a
strong low-complexity baseline. Contextual produced the highest Faithfulness
but adds one enrichment call per chunk. The current cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, an English-trained MS MARCO model - a
likely cause of Reranked's weak PT-BR results) and the GraphRAG configuration
need further experimentation before drawing broader conclusions; a multilingual
reranker (for example mMARCO or bge-reranker-v2-m3) is an obvious next test.

These interpretations are intentionally scoped to this deterministic sample. A
future full-split run, calibrated judge, materially different corpus, or
confidence-interval analysis may change the conclusions.

## Evidence and verification

The versioned evidence bundle is under
`artifacts/runs/20260726T185553Z/`. Aggregate results, per-question records, and
captured write-through LLM calls are under `experiments/20260726T185553Z/`.
Those captured calls support traceability and future replay work; the
deterministic zero-provider replay executor described by ADR-0004/ADR-0020 is
not implemented yet.

Verify checksums, the event hash chain, and manifest references locally:

```bash
uv run python scripts/verify_run.py 20260726T185553Z
```

Expected result:

```text
OK: 20260726T185553Z - checksums, event chain, and manifest all verify cleanly.
```

## API and dashboard

Run the read-only API:

```bash
make api
```

Endpoints:

- `GET /health`
- `GET /api/v1/benchmark-runs`
- `GET /api/v1/benchmark-runs/20260726T185553Z`

Run the offline analytical dashboard:

```bash
make dashboard
```

The dashboard defaults to Portuguese and can be switched to English from the
sidebar. Besides the aggregate comparison, it includes bilingual explanations
of all ten retrieval techniques and all nine reported indicators, including
their interpretation and whether higher or lower values are preferable.

Both surfaces load only runs listed in `experiments/published-runs.json`.
Uncataloged experiment directories are not exposed. These endpoints are
intentionally unauthenticated because they expose only immutable public
benchmark aggregates; they are read-only and do not expose prompts,
per-question model responses, or provider credentials.

## Limitations

- The 60-question sample is deterministic and query-class-stratified, but it is
  not the complete 194-question test split.
- OpenAI judge scores remain qualified until the ADR-0007 human calibration
  exercise reaches its agreement gate.
- Provider pricing was not populated in the experiment configuration, so the
  report records token usage but no authoritative cost estimate.
- The semantic citation audit was disabled to control cost.
- LightRAG emitted warnings because no additional internal reranker was
  configured; GraphRAG proceeded without that optional reranking step.
- Confidence intervals are not yet reported, so small metric differences should
  not be read as statistically established wins.
- The dashboard implements the ADR-0008 analytical view. The live Arena depends
  on the still-planned router and corrective workflow.
