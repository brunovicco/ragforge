# ADR-0010: GraphRAG (LightRAG) evaluation scope and provenance recovery

- Status: Accepted
- Date: 2026-07-23

## Context

GraphRAG (README strategy #8) integrates LightRAG (`lightrag-hku`, already a declared
dependency). LightRAG runs its own end-to-end pipeline internally - chunking, entity/relation
extraction, knowledge-graph construction, community detection for "global" mode - and its public
`aquery()` returns a synthesized narrative answer (`str`), not a ranked list of source chunks.

Every other benchmarked strategy (Dense, Sparse, Hybrid, Reranked, Parent-child, Contextual,
RAPTOR) satisfies ADR-0002 - relevance judgments and recall/precision/nDCG/MRR@k computed at the
`norm::article::fragment` structural level - because they all return `list[RetrievalResult]` with
real `Chunk` objects carrying `structural_ids`. GraphRAG's default output has no such mapping.

Two things make recovering that mapping possible without depending on LightRAG's undocumented
internal storage schema, verified empirically (not assumed) against the real library:

1. **`chunking_func`** is a public, documented extension point. Supplying our own returns our
   exact ADR-0006 structural chunks (content, order) instead of LightRAG's token-window split.
   Verified: inserting two chunks with a custom `chunking_func` produces exactly two internal
   chunks, in our order, not LightRAG's own splitting.
2. **`aquery_data()`** (and its sync wrapper `query_data()`) returns structured data - entities,
   relationships, and a `chunks` array with literal `content` per item - instead of only a
   synthesized answer. Verified: querying an index built from two custom chunks, in "global" mode,
   returned both chunks' exact original content strings, recoverable by exact string matching
   against a pre-built `content -> Chunk` lookup.

## Decision

1. `GraphRagRetrieval` indexes each norm with a custom `chunking_func` that returns our own
   ADR-0006 chunks unchanged, and retrieves via `query_data()`, mapping each returned chunk's
   `content` back to our `Chunk` (and its `structural_ids`) through an exact-match
   `content -> Chunk` lookup built at indexing time.
2. Metrics ARE computed for GraphRAG the same way as every other strategy (recall, precision,
   nDCG, MRR @k against ADR-0002 judgments) - not skipped - with two disclosed limitations:
   - **Coverage, not correctness, can silently drop**: if `query_data()` returns a chunk whose
     content isn't an exact match in the lookup (unexpected reformatting, or a genuine collision
     across documents with identical text - not observed in the current corpus but theoretically
     possible), that item is dropped rather than raised. `retrieve()` can then return fewer than
     `top_k` results even when LightRAG found relevant evidence.
   - **No native per-chunk relevance score**: LightRAG doesn't expose one via `query_data()`, so
     `RetrievalResult.score` is a rank-based proxy (`1 / rank` over the order LightRAG already
     returned), not a model-computed score. Recall/precision/MRR@k are rank-based and unaffected;
     nDCG's ideal-DCG normalization assumes LightRAG's own internal ranking is already
     relevance-ordered, which is not independently verified here.
3. What stays genuinely unmeasured: the knowledge-graph reasoning itself (which entities and
   relationships led to a chunk being selected, and at what confidence) isn't captured by our
   metrics - only the resulting chunk set's overlap with judged-relevant structural units is.
   This is the real, disclosed gap this ADR accepts rather than hides.
4. Both README-required modes ("local", "global") are supported via `QueryParam(mode=...)`;
   "hybrid", "mix", and "naive" are exposed as the same constructor parameter but not required.

## Consequences

- GraphRAG participates in the same recall/precision/nDCG/MRR@k comparison table as every other
  strategy - no separate, weaker evaluation track.
- The adapter is coupled to two specific (public, documented) LightRAG extension points; a future
  LightRAG upgrade that changes `chunking_func`'s calling convention or `query_data()`'s response
  shape would need re-verification, not just a version bump.
- Indexing cost is materially higher than the other strategies: entity/relation extraction runs
  one or more LLM passes per chunk (verified against the real library, not estimated), on top of
  the single embedding or single contextualization/summarization call the other strategies need.

## Alternatives considered

- **Use `aquery()`'s synthesized answer only, skip chunk-level metrics** - rejected: this is what
  the user explicitly declined in favor of measuring GraphRAG on the same footing as every other
  strategy; an end-to-end-only evaluation would make GraphRAG incomparable in the main results
  table.
- **Read LightRAG's internal storage directly** (`text_chunks`, `chunks_vdb`) to recover exact
  chunk IDs instead of content-matching - rejected: couples this adapter to an undocumented,
  version-fragile internal schema instead of the two public extension points already sufficient
  for the mapping.
