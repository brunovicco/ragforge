# RAGForge

*Leia isto em [Português](README.pt-BR.md).*

**Adaptive RAG benchmarking platform for Brazilian financial and regulatory documents.**

RAGForge is being built to benchmark sparse, dense, hybrid, contextual, hierarchical (RAPTOR), graph (GraphRAG) and corrective strategies - measuring answer quality, retrieval precision, latency and cost on **RegRAG-BR**, a 230-question golden dataset over CMN/BCB and CVM norms.

> 🚧 v0.1 in progress. See [Status](#status) for what is implemented today versus planned.

## Why this exists

Most RAG comparisons are anecdotal. RAGForge treats the question "*which RAG strategy should I use?*" as an experiment: 10 strategy configurations × 7 query classes, with an adaptive router meant to be evaluated against an **empirical oracle** and every published number reproducible bit-for-bit from a versioned LLM call cache. See [Status](#status) for what is built so far.

## Benchmarked strategies

| # | Strategy | Approach | Status |
| --- | ---------- | ---------- | -------- |
| 1 | Dense (baseline) | pgvector, fixed top-k | Implemented |
| 2 | Sparse BM25 | OpenSearch, `brazilian` analyzer | Implemented |
| 3 | Hybrid + RRF | BM25 + dense + Reciprocal Rank Fusion | Implemented |
| 4 | Reranked | Hybrid top-50 → cross-encoder → top-5 | Implemented |
| 5 | Contextual Retrieval | Per-chunk LLM context + prompt caching | Implemented |
| 6 | Parent-child / multi-vector | Search small chunks, deliver the section | Implemented |
| 7 | Summary-Augmented Chunking (SAC) | Document summary + authoritative chunk text | Implemented |
| 8 | SAC + Contextual | Document summary + per-chunk context + authoritative text | Implemented |
| 9 | RAPTOR | Recursive summary tree (minimal impl.) | Implemented |
| 10 | GraphRAG | LightRAG adapter (local + global) | Implemented |

Cross-cutting: **Adaptive Router** (rules + few-shot, planned), **Corrective workflow** (evidence evaluator with retry / reformulation / insufficient-evidence declaration, LangGraph, planned), **governance** (answer → chunk → article citation tracing via Citation Accuracy, plus a post-generation semantic-support audit with bounded rewrite and a tamper-evident evidence trail per run, implemented), **observability** (Langfuse metadata-only tracing implemented; OpenTelemetry planned).

## Status

RAGForge is under active development (v0.1, see the checkpoint note above). This section tracks what
is actually running today versus what the design targets - see the [PR history](../../pulls?q=is%3Apr) for how each row landed.

| Component | Status |
|---|---|
| Legal structural chunker (ADR-0006) | Implemented |
| Ingestion pipeline (extraction, snapshot hashing) | Implemented |
| All 10 benchmarked retrieval configurations (Dense through GraphRAG) | Implemented |
| Evaluation harness + structural-coverage judgments (ADR-0002) | Implemented |
| Observability (Langfuse, metadata-only) | Implemented |
| Answer generation + Citation Accuracy | Implemented |
| Independent LLM judge - Faithfulness/Answer Relevancy + abstention (ADR-0018) | Implemented - uncalibrated (ADR-0007 kappa exercise pending) |
| Post-generation citation/semantic-support audit + bounded rewrite (ADR-0016) | Implemented - off by default (`audit.enabled: false`) |
| Auditable, tamper-evident run evidence directory (ADR-0017) | Implemented - `artifacts/runs/<run_id>/`, verified via `scripts/verify_run.py` |
| Main benchmark runner (`make bench-live`, all 10 strategies + answer quality) | Implemented - live mode only |
| Adaptive Router, Corrective workflow | Planned |
| RegRAG-BR golden set | 230 questions published: 36 validation/dev + 194 test |
| API / dashboard apps | Published-results API and analytical dashboard implemented; live Arena planned |
| `make bench` (cached, bit-for-bit replay, ADR-0004) | Planned - needs a versioned LLM call cache, not built yet |

## v0.1 benchmark result

Run [`20260726T185553Z`](experiments/20260726T185553Z/results.json) evaluates a
**deterministic, query-class-stratified sample of 60 questions** from the
194-question test split. The seed is
`regrag-br-benchmark-sample-v1`; this is a cost-controlled v0.1 result, not a
claim about the complete test split.

**SAC is the recommended v0.1 strategy** for its balanced profile: the highest
nDCG@5 (`0.963`), MRR (`0.991`), and Citation Accuracy (`0.689`) in this run,
with zero Document-Level Retrieval Mismatch. RAPTOR achieved the highest
Recall@5 (`1.000`) and Precision@5 (`0.611`), but its generated summary nodes
carry a different evidence-quality trade-off.

The full scorecard, methodology, limitations, and verification instructions
are in [Benchmark results](docs/BENCHMARK-RESULTS.md).

## Quick start

```bash
uv sync --all-groups
make infra-up                                      # Postgres+pgvector, OpenSearch
GEMINI_API_KEY=... OPENAI_API_KEY=... make bench-live
make bench-live-local                              # same matrix, local Qwen embeddings
make api                                           # read-only published-results API
make dashboard                                     # offline analytical benchmark dashboard
```

`make bench-live` calls real providers (embeddings, contextualization, RAPTOR summarization, GraphRAG entity extraction - see the strategy table above). `make bench` (deterministic, zero-cost replay from a versioned LLM cache) is the target design per [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md), but that cache layer doesn't exist yet - only live mode is implemented.

The canonical publishable matrix uses `gemini-embedding-001`, selected by the isolated PT-BR
embedding comparison (ADR-0005). `make bench-live-local` uses
`Qwen/Qwen3-Embedding-0.6B` as ADR-0013's credential-free embedding alternative; it is a
separately identified run, not a silent fallback or a relabeling of the quality winner.

Both live commands persist per-text embeddings and completed indexes under the ignored
`.ragforge/cache/` directory. A deterministic index fingerprint includes corpus-derived text,
embedding identity, and synthetic-text producer identity; partial indexes are never marked
reusable. `--resume <run-id>` skips strategies already checkpointed. A repository lock rejects
concurrent benchmark processes before they can mutate shared indexes.

## Key design decisions

All non-obvious choices are recorded as [ADRs](docs/adr/README.md). The load-bearing ones:

- [ADR-0002](docs/adr/0002-article-level-relevance-judgments.md) - relevance judgments at **norm-article level**, so retrieval metrics stay comparable across strategies that chunk differently (or don't return chunks at all).
- [ADR-0003](docs/adr/0003-empirical-router-oracle.md) - the router is scored against an **empirical per-question oracle** (best strategy measured, not assumed), with a dev/test split preventing few-shot leakage.
- [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md) - `make bench` replays a versioned LLM cache: bit-for-bit reproduction, zero API cost.
- [ADR-0006](docs/adr/0006-legal-structural-chunker.md) - domain-aware chunking by legal hierarchy (Art./§/inciso) with stable structural IDs.
- [ADR-0007](docs/adr/0007-llm-judge-calibration-ptbr.md) - the LLM judge is calibrated against human evaluation in PT-BR and the agreement is published.
- [ADR-0011](docs/adr/0011-structural-id-collision-in-amended-norms.md) - structural IDs that collide across amendment history/appended annexes are excluded from golden-set citations, not fixed at the chunker level.
- [ADR-0016](docs/adr/0016-post-generation-citation-audit.md) - a semantic-support verifier and at most one bounded rewrite catch unsupported claims a citation-existence check alone would miss.
- [ADR-0017](docs/adr/0017-auditable-evidence-lineage.md) - every published score traces back to a hash-chained, tamper-evident evidence directory per run - exact inputs, model identities, and retrieval candidates, not just the aggregate metric.

## Repository layout

```
apps/            # api/ (FastAPI) and dashboard/ (Streamlit: benchmark + Arena)
src/ragforge/    # domain/ (framework-free core) · ingestion/ chunking/ embeddings/
                 # retrieval/ reranking/ routing/ generation/ evaluation/ governance/
datasets/        # corpus/ (versioned snapshot) + regrag-br/ (golden set, CC-BY-4.0)
experiments/     # versioned results + LLM cache per run-id
configs/         # declarative experiment configs - every README number is born here
docs/adr/        # architecture decision records
```

The core is framework-free: `RetrievalStrategy` is a Protocol; LLM SDKs are banned from core packages by a CI architecture guard (`scripts/validate_architecture.py`, boundaries in `pyproject.toml`).

## Dataset - RegRAG-BR

230 questions (7 query classes) over selected CMN/BCB resolutions (4,893, risk management, Open Finance, AML) and CVM/CMN norms, with article-level relevance judgments and reference answers, published under CC-BY-4.0 with a datasheet. The deterministic stratified split reserves 36 questions for router development/validation and 194 for official test metrics. Norms are official acts (art. 8, I, Law 9,610/98 - not copyright-protected).

Published (`datasets/regrag-br/judgments.json`): 230 hand-curated questions, each with a reference answer, verified against the real parsed text of 5 corpus documents (LC-105/2001, RES-CMN-4893/2021, RES-CMN-5274/2025, LEI-13709/2018-LGPD, ICVM-607/2019). A 6th corpus document, LEI-6385/1976, is not yet curated. Structural IDs known to be ambiguous in the real source text - amendment-history and appended-annex artifacts in 3 of the 5 documents - are excluded from citation; see [ADR-0011](docs/adr/0011-structural-id-collision-in-amended-norms.md).

## Development

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/quality_gate.py   # ruff, mypy, pytest (≥80% core), bandit, pip-audit, architecture guard
```

Scaffolded with [claude-python-engineering-harness](https://github.com/brunovicco/claude-python-engineering-harness) ([ADR-0009](docs/adr/0009-scaffold-via-engineering-harness.md)).

## License

Code: MIT · Dataset (RegRAG-BR): CC-BY-4.0
