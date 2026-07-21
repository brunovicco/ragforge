# RAGForge

**Adaptive RAG benchmarking platform for Brazilian financial and regulatory documents.**

Routes queries across sparse, dense, hybrid, contextual, hierarchical (RAPTOR), graph (GraphRAG) and corrective strategies - measuring answer quality, retrieval precision, latency and cost on **RegRAG-BR**, a curated golden dataset of 210 questions over CMN/BCB and CVM norms.

> 🚧 v0.1 in progress - 4-week sprint. Weekly publishable checkpoints; results table lands here as runs complete.

## Why this exists

Most RAG comparisons are anecdotal. RAGForge treats the question "*which RAG strategy should I use?*" as an experiment: 8 strategies × 7 query classes, an adaptive router evaluated against an **empirical oracle** and every published number reproducible bit-for-bit from a versioned LLM call cache.

## Benchmarked strategies

| # | Strategy | Approach |
|---|----------|----------|
| 1 | Dense (baseline) | pgvector, fixed top-k |
| 2 | Sparse BM25 | OpenSearch, `brazilian` analyzer |
| 3 | Hybrid + RRF | BM25 + dense + Reciprocal Rank Fusion |
| 4 | Reranked | Hybrid top-50 → cross-encoder → top-5 |
| 5 | Contextual Retrieval | Per-chunk LLM context + prompt caching |
| 6 | Parent-child / multi-vector | Search small chunks, deliver the section |
| 7 | RAPTOR | Recursive summary tree (minimal impl.) |
| 8 | GraphRAG | LightRAG adapter (local + global) |

Cross-cutting: **Adaptive Router** (rules + few-shot), **Corrective workflow** (evidence evaluator with retry / reformulation / insufficient-evidence declaration, LangGraph), **governance** (answer → chunk → article citation tracing), **observability** (Langfuse + OpenTelemetry).

## Quick start

```bash
uv sync --all-groups
make infra-up                  # Postgres+pgvector, OpenSearch (docker compose profiles)
make bench                     # deterministic replay from the versioned LLM cache - no API key needed
make dashboard                 # benchmark view + side-by-side strategy Arena
```

`make bench-live` re-runs against providers (see [ADR-0004](docs/adr/0004-benchmark-reproducibility-policy.md) for the reproducibility policy).

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

210 questions (7 classes × 30) over selected CMN/BCB resolutions (4,893, risk management, Open Finance, AML) and CVM rules, with article-level relevance judgments and reference answers. Published under CC-BY-4.0 with a datasheet. Norms are official acts (art. 8, I, Law 9,610/98 - not copyright-protected).

## Development

```bash
uv sync --all-groups
uv run pytest
uv run python scripts/quality_gate.py   # ruff, mypy, pytest (≥80% core), bandit, pip-audit, architecture guard
```

Scaffolded with [claude-python-engineering-harness](https://github.com/brunovicco/claude-python-engineering-harness) ([ADR-0009](docs/adr/0009-scaffold-via-engineering-harness.md)).

## License

Code: MIT · Dataset (RegRAG-BR): CC-BY-4.0
