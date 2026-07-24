# ADR-0015: Evaluate Summary-Augmented Chunking as a separate retrieval strategy

- SStatus: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0002, ADR-0004, ADR-0005, ADR-0006, ADR-0013, ADR-0014

## Context

Legal and regulatory documents contain locally similar clauses. A chunk can describe obligations, controls, deadlines, exceptions, or governance requirements without repeating the identity and scope of the source norm. Dense retrieval may therefore return a semantically plausible clause from the wrong document.

This failure is commonly described as Document-Level Retrieval Mismatch.

RAGForge already includes Contextual Retrieval, which generates a short context specifically for each chunk using the complete source document. Summary-Augmented Chunking (SAC) is related but distinct: SAC creates one global summary per document version and prefixes the same summary to each retrievable unit from that document.

SAC must be tested rather than assumed superior. A summary may improve document discrimination, but can also dominate the embedding, reproduce one summarization error across all chunks, or reduce the ability to distinguish nearby provisions from the same norm.

## Decision drivers

- Measure document-level mismatch directly.
- Preserve authoritative source text.
- Prevent generated summaries from becoming cited legal evidence.
- Compare SAC independently from Contextual Retrieval.
- Cache every LLM-derived enrichment.
- Preserve structural judgments across transformations.
- Report quality, cost, latency, and index-size trade-offs.

## Decision

RAGForge SHALL implement SAC as an experimental retrieval-text transformation. Authoritative source text SHALL remain separate from synthetic retrieval text.

### Source and retrieval representations

The chunk contract SHALL distinguish:

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    source_text: str
    retrieval_text: str
    structural_ids: tuple[str, ...]
    parent_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
```

Rules:

- `source_text` is extracted from the authoritative source and is immutable;
- `retrieval_text` may include synthetic text used only for indexing and retrieval;
- the answer generator receives authoritative `source_text`;
- citations resolve only to structural IDs associated with authoritative text;
- golden judgments remain structural and independent from enrichment.

### Document summary contract

Generate one summary per immutable document version.

The prompt SHALL request:

- document purpose;
- scope;
- issuing authority;
- regulated subject;
- main categories of provisions;
- concise Brazilian Portuguese;
- no external interpretation;
- no invented obligations;
- no unsupported claims.

Use a structured result:

```json
{
  "schema_version": 1,
  "document_id": "RES-CMN-4893/2021",
  "document_version": "source-sha256",
  "summary": "...",
  "topics": ["segurança cibernética"],
  "generation_model": "...",
  "prompt_version": "sac-summary-ptbr-v1"
}
```

Generated topics are retrieval metadata, not an authoritative legal taxonomy.

### Cache identity

The summary cache key SHALL include:

```text
source hash
+ extraction version
+ prompt hash
+ provider
+ generation model snapshot
+ generation parameters
+ structured-output schema version
```

Cache-only reproducible mode SHALL fail when the required summary is absent.

### Retrieval-text variants

The experiment SHALL implement:

```text
baseline:
    retrieval_text = source_text

sac:
    retrieval_text = document_summary + source_text

contextual:
    retrieval_text = chunk_context + source_text

sac_contextual:
    retrieval_text = document_summary + chunk_context + source_text
```

Prefix labels and separators SHALL be versioned.

Example:

```text
[DOCUMENT SUMMARY]
...

[CHUNK CONTEXT]
...

[AUTHORITATIVE SOURCE TEXT]
...
```

The generator SHALL only receive the authoritative section unless the experiment explicitly tests generated metadata as generation context.

### Strategy identity

Variants SHALL have distinct names:

```text
dense
contextual_dense
sac_dense
sac_contextual_dense
```

SAC results SHALL never be reported as ordinary Dense results.

### Parallel execution

Document summaries are independent and MAY be generated concurrently under ADR-0014.

Only one task SHALL exist per unique summary cache key. Duplicate requests SHALL be coalesced.

Deterministic local prefix assembly may execute after summaries are complete.

### Metrics

In addition to the standard retrieval metrics, report:

- document precision@k;
- document recall@k;
- Document-Level Retrieval Mismatch rate;
- structural-unit precision@k;
- structural-unit recall@k;
- index size;
- summary generation tokens;
- summary generation cost;
- indexing duration;
- query latency.

The exact DRM formula SHALL be documented. At minimum, it SHALL identify retrieved results whose source document is outside the judged relevant-document set.

### Ablation protocol

Compare:

1. Dense baseline.
2. Contextual Retrieval.
3. SAC.
4. SAC plus Contextual Retrieval.

Keep constant:

- corpus;
- split;
- authoritative chunks;
- embedding model;
- candidate depth;
- top-k;
- storage;
- judgments;
- metric implementation.

Run with the local default embedding and at least one independent embedding family.

### Promotion rule

SAC remains experimental until it demonstrates:

- lower DRM on the full test split;
- no material regression in structural recall;
- confidence intervals supporting the difference;
- stable direction across at least two embedding families;
- acceptable indexing cost and latency.

A single improved RAGAS Context Precision value is insufficient for promotion.

## Failure behavior

- summary generation fails: fail SAC index construction;
- summary absent in cache-only mode: fail closed;
- invalid structured output: retry within policy, then fail;
- empty summary: fail the affected document;
- no fallback to baseline under an SAC strategy name.

## Security and privacy

Document summaries may expose source content to a provider. Public benchmark sources may use hosted generation. Private documents require explicit external-processing authorization or a local summarizer.

Summary content, prompts, provider identity, and hashes SHALL be included in lineage.

## Consequences

### Positive

- Directly tests an important legal-RAG failure mode.
- Preserves legal evidence separately from synthetic retrieval text.
- Creates a clean comparison with Contextual Retrieval.
- Pays summary cost once per document version.

### Negative

- Increases index size.
- Propagates summary errors across all chunks in a document.
- May reduce within-document discrimination.
- Adds generation, cache, and lineage complexity.

## Alternatives considered

- **Replace Contextual Retrieval:** rejected because the two techniques encode different levels of context.
- **Put summaries into source text:** rejected because generated text is not authoritative evidence.
- **Generate one summary per chunk:** rejected as SAC; that is chunk-level contextualization.
- **Assume improved RAGAS metrics:** rejected; RegRAG-BR must measure the effect.

## Acceptance criteria

- [ ] Source and retrieval text are separate.
- [ ] The generator does not cite synthetic summaries as legal evidence.
- [ ] One cached summary exists per document version.
- [ ] Prompt and model identity are recorded.
- [ ] Four ablation variants run on the same split.
- [ ] DRM is implemented and documented.
- [ ] Failure cannot silently become baseline behavior.
- [ ] Cost, latency, and index size are reported.
- [ ] Tests verify structural IDs and source text remain unchanged.

## Rollout

1. Introduce source/retrieval text separation.
2. Update stores and strategies to index retrieval text.
3. Preserve generation from source text.
4. Implement summary port and provider adapter.
5. Add immutable summary cache.
6. Add SAC variants.
7. Add DRM metrics.
8. Run and publish the ablation.
