# ADR-0005: Embedding comparison restricted to Dense and Hybrid

- Status: Accepted
- Date: 2026-07-21

## Context

The plan compares 2 embedding models (an open multilingual one — BGE-M3 or Qwen3-Embedding — vs a proprietary one) "in the benchmark". Taken literally this doubles the matrix: 2 embeddings × 8 strategies × 210 questions, with double indexing of every structure (pgvector, multi-vector, RAPTOR tree, LightRAG graph). In a 4-week sprint this is scope and indexing-cost explosion.

## Decision

The embedding comparison is an **isolated experiment**, not a dimension of the main matrix:

1. Compare both models only on **Dense** and **Hybrid+RRF** (the strategies most sensitive to the embedding), over the full test set, with a dedicated section on PT-BR regulatory performance.
2. The winner is **frozen** for all remaining strategies and the main benchmark matrix. The choice and its supporting numbers are recorded in `configs/experiments/embeddings-ptbr.yaml`.
3. The main matrix reports a single embedding configuration, declared in the README.

## Consequences

- Controlled scope: one full indexing pass instead of two; RAPTOR and GraphRAG (highest-effort items) index once.
- The comparison remains a publishable result — and the backlog already plans a standalone article extending it.
- The embedding choice becomes data, not opinion.
- Possible embedding × strategy interaction goes unmeasured (the Dense loser could win in RAPTOR). Limitation declared in README and datasheet.

## Alternatives considered

- **Full 2 × 8 matrix** — rejected: doubles indexing and evaluation cost; infeasible in the sprint.
- **Single embedding, no comparison** — rejected: the PT-BR regulatory section is a differentiator with little equivalent public material.
- **Compare on Dense only** — rejected: Hybrid is cheap to include and shows the embedding effect combined with BM25, the more realistic scenario.
