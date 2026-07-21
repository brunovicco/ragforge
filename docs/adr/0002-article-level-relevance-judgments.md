# ADR-0002: Relevance judgments at norm-article level, not chunk level

- Status: Accepted
- Date: 2026-07-21

## Context

RegRAG-BR needs relevance judgments for Recall@k, Precision@k, MRR and nDCG. The initial plan annotated at chunk level, but chunks are not a stable unit across strategies: parent-child indexes small chunks yet returns sections; RAPTOR returns synthetic summaries that do not exist in the corpus; GraphRAG (global mode) returns community answers, not chunks; and any chunking parameter change would invalidate all judgments. Judgments annotated against the baseline chunking would make retrieval metrics for hierarchical and graph strategies incomparable - undermining the benchmark's central result.

## Decision

Annotate relevance at the **stable structural unit of the norm**: `norm → article → paragraph/item` (e.g. `RES-CMN-4893/2021::art-3::par-1`). Each golden-set question references a set of relevant structural IDs (graded: relevant / partially relevant).

At evaluation time, each strategy projects its results onto structural IDs via a `chunk → structural spans` mapping maintained by the ingestion pipeline (see ADR-0006). A retrieved chunk counts as a hit if it covers at least one relevant ID. For synthetic results (RAPTOR, GraphRAG global), projection uses the source-node IDs that produced the summary.

## Consequences

- Retrieval metrics are comparable across all 8 strategies, including those that do not return literal chunks.
- Judgments survive chunking changes - the published dataset (CC-BY-4.0) remains reusable by third parties with any strategy.
- The chunk → article mapping is the same artifact required by the governance layer (traceable citations); one implementation serves both.
- Ingestion must preserve structural hierarchy (dependency on ADR-0006).
- Structural-coverage Recall@k is slightly more permissive than exact-chunk matching; the datasheet must document the metric semantics.

## Alternatives considered

- **Chunk-level judgments** - rejected: breaks cross-strategy comparison; fragile to re-chunking.
- **Document-level judgments** - rejected: insufficient granularity for long norms; Precision@k loses meaning.
- **Dual judgments (chunk + article)** - rejected: doubles curation cost, which is already the critical path.
