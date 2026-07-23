# RAGForge

**Adaptive RAG benchmarking platform for Brazilian financial and regulatory documents.**

RAGForge is being built to benchmark sparse, dense, hybrid, contextual, hierarchical (RAPTOR), graph (GraphRAG) and corrective strategies - measuring answer quality, retrieval precision, latency and cost on **RegRAG-BR**, a golden dataset targeting 210 questions over CMN/BCB and CVM norms.

> 🚧 v0.1 in progress. See [Status](#status) for what is implemented today versus planned.

## Why this exists

Most RAG comparisons are anecdotal. RAGForge treats the question "*which RAG strategy should I use?*" as an experiment: 8 targeted strategies × 7 query classes, with an adaptive router meant to be evaluated against an **empirical oracle** and every published number reproducible bit-for-bit from a versioned LLM call cache. See [Status](#status) for what is built so far.

## Benchmarked strategies

| # | Strategy | Approach | Status |
|---|----------|----------|--------|
| 1 | Dense (baseline) | pgvector, fixed top-k | Implemented |
| 2 | Sparse BM25 | OpenSearch, `brazilian` analyzer | Implemented |
| 3 | Hybrid + RRF | BM25 + dense + Reciprocal Rank Fusion | Implemented |
| 4 | Reranked | Hybrid top-50 → cross-encoder → top-5 | Implemented |
| 5 | Contextual Retrieval | Per-chunk LLM context + prompt caching | Implemented |
| 6 | Parent-child / multi-vector | Search small chunks, deliver the section | Implemented |
| 7 | RAPTOR | Recursive summary tree (minimal impl.) | Implemented |
| 8 | GraphRAG | LightRAG adapter (local + global) | Implemented |

Cross-cutting: **Adaptive Router** (rules + few-shot, planned), **Corrective workflow** (evidence evaluator with retry / reformulation / insufficient-evidence declaration, LangGraph, planned), **governance** (answer → chunk → article citation tracing, planned), **observability** (Langfuse metadata-only tracing implemented; OpenTelemetry planned).

## Status

RAGForge is under active development (v0.1, see the checkpoint note above). This section tracks what
is actually running today versus what the design targets - see the [PR history](../../pulls?q=is%3Apr) for how each row landed.

| Component | Status |
|---|---|
| Legal structural chunker (ADR-0006) | Implemented |
| Ingestion pipeline (extraction, snapshot hashing) | Implemented |
| All 8 benchmarked retrieval strategies (Dense through GraphRAG) | Implemented |
| Evaluation harness + structural-coverage judgments (ADR-0002) | Implemented |
| Observability (Langfuse, metadata-only) | Implemented |
| Main benchmark runner (`make bench-live`, all 8 strategies) | Implemented - live mode only |
| Adaptive Router, Corrective workflow, Governance | Planned |
| RegRAG-BR golden set | In progress - 20 questions published, 210 targeted |
| API / dashboard apps | Planned (scaffolding only) |
| `make bench` (cached, bit-for-bit replay, ADR-0004) | Planned - needs a versioned LLM call cache, not built yet |

## Quick start

```bash
uv sync --all-groups
make infra-up                  # Postgres+pgvector, OpenSearch (docker compose profiles)
GEMINI_API_KEY=... make bench-live   # real run, all 8 strategies, real API cost
make dashboard                       # benchmark view + side-by-side strategy Arena
```

`make bench-live` calls real providers (embeddings, contextualization, RAPTOR summarization, GraphRAG entity extraction - see the strategy table above). `make bench` (deterministic, zero-cost replay from a versioned LLM cache) is the target design per [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md), but that cache layer doesn't exist yet - only live mode is implemented.

## Key design decisions

All non-obvious choices are recorded as [ADRs](docs/adr/README.md). The load-bearing ones:

- [ADR-0002](docs/adr/0002-article-level-relevance-judgments.md) - relevance judgments at **norm-article level**, so retrieval metrics stay comparable across strategies that chunk differently (or don't return chunks at all).
- [ADR-0003](docs/adr/0003-empirical-router-oracle.md) - the router is scored against an **empirical per-question oracle** (best strategy measured, not assumed), with a dev/test split preventing few-shot leakage.
- [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md) - `make bench` replays a versioned LLM cache: bit-for-bit reproduction, zero API cost.
- [ADR-0006](docs/adr/0006-legal-structural-chunker.md) - domain-aware chunking by legal hierarchy (Art./§/inciso) with stable structural IDs.
- [ADR-0007](docs/adr/0007-llm-judge-calibration-ptbr.md) - the LLM judge is calibrated against human evaluation in PT-BR and the agreement is published.

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

Targeting 210 questions (7 classes × 30) over selected CMN/BCB resolutions (4,893, risk management, Open Finance, AML) and CVM rules, with article-level relevance judgments and reference answers, published under CC-BY-4.0 with a datasheet. Norms are official acts (art. 8, I, Law 9,610/98 - not copyright-protected).

Today, 20 questions are published (`datasets/regrag-br/judgments.json`) - a hand-curated starter set verified against the real parsed text of 4 corpus documents (LC-105/2001, RES-CMN-4893/2021, RES-CMN-5274/2025, LEI-13709/2018), demonstrating the judgment format end-to-end. The remaining 190 are in progress.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/quality_gate.py   # ruff, mypy, pytest (≥80% core), bandit, pip-audit, architecture guard
```

Scaffolded with [claude-python-engineering-harness](https://github.com/brunovicco/claude-python-engineering-harness) ([ADR-0009](docs/adr/0009-scaffold-via-engineering-harness.md)).

## License

Code: MIT · Dataset (RegRAG-BR): CC-BY-4.0
