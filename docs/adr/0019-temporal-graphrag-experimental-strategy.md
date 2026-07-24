# ADR-0019: Add Temporal GraphRAG only as a future experimental strategy

- Status: Accepted
- Date: 2026-07-24
- Target: Future release
- Related: ADR-0002, ADR-0003, ADR-0006, ADR-0010, ADR-0011, ADR-0012, ADR-0016, ADR-0017

## Context

Legal questions may depend on when a rule was published, became effective, was amended, suspended, or revoked. Flat retrieval can return a semantically relevant but temporally invalid version.

RAGForge already models normative hierarchy, but does not yet maintain a fully versioned temporal legal corpus. Adding graph traversal before trustworthy version boundaries exist would create the appearance of temporal reasoning without reliable temporal evidence.

Temporal GraphRAG is therefore a future comparison strategy. It is not a migration away from Dense, Hybrid, Parent-child, Contextual Retrieval, SAC, RAPTOR, or the current GraphRAG implementation.

## Decision drivers

- Reduce stale and anachronistic evidence.
- Preserve comparison with current strategies.
- Represent hierarchy and cross-references.
- Support explicit “as of” queries.
- Avoid temporal claims before the corpus supports them.
- Preserve exact version provenance.

## Decision

Temporal GraphRAG SHALL be implemented only after the temporal-corpus and temporal-golden-set prerequisites are complete.

It SHALL combine:

- temporal filtering;
- lexical retrieval;
- vector retrieval;
- bounded graph expansion;
- reranking;
- version-aware generation;
- temporal citation audit.

It SHALL remain a separate strategy in the benchmark matrix.

## Prerequisites

### Versioned normative corpus

Every legal text version SHALL have immutable identity and source evidence.

Required fields:

```text
published_at
effective_from
effective_until
revoked_at
recorded_at
source_version_id
```

`recorded_at` enables bitemporal reasoning:

- valid time: when the rule applies;
- transaction time: when RAGForge recorded the fact.

### Version-qualified structural IDs

A device whose wording changes SHALL not retain an ambiguous evidence ID.

Example:

```text
RES-BCB-538/2025::version-2025-09-01::art-2
RES-BCB-538/2025::version-2026-03-15::art-2
```

A logical ID may group historical versions, but a citation SHALL identify one exact version.

This future work is expected to supersede the current collision treatment in ADR-0011.

### Temporal golden set

Create human-curated questions such as:

- What requirement was effective on a specified date?
- Was the obligation already effective on the transaction date?
- Which wording replaced the prior version?
- Which act revoked a provision?
- What changed between two dates?

Judgments SHALL reference exact versioned structural units.

### Temporal extraction validation

Publication, effectiveness, amendment, and revocation facts SHALL be backed by authoritative source evidence.

LLM-extracted graph edges SHALL remain candidates until validated. Model output alone is not authoritative metadata.

## Graph model

### Nodes

```text
Authority
Norm
NormVersion
StructuralUnit
Topic
Definition
Obligation
Exception
```

### Edges

```text
CONTAINS
VERSION_OF
REFERS_TO
AMENDS
REVOKES
SUPERSEDES
EXCEPTION_TO
DEFINES
REGULATES
VALID_DURING
```

Each edge SHALL include:

- source structural ID;
- source hash;
- extraction method;
- validation status;
- temporal interval where applicable.

## Query semantics

A temporal query SHALL resolve an explicit `as_of` date.

When the user provides no date:

- the strategy MAY use the latest known effective version;
- the resolved policy and date SHALL be returned in metadata;
- the response SHALL not imply a historical interpretation.

Ambiguous relative dates SHALL be normalized and recorded.

## Retrieval pipeline

```text
query analysis
→ resolve as_of
→ select valid document versions
→ lexical and dense candidate retrieval
→ bounded graph expansion
→ temporal and semantic reranking
→ authoritative source assembly
→ generation
→ temporal citation audit
```

Temporal filtering SHALL occur before graph expansion. An invalid version SHALL not re-enter through a neighbor.

## Hybrid behavior

Temporal GraphRAG SHALL not abandon text retrieval.

- lexical retrieval handles exact terminology and identifiers;
- dense retrieval handles semantic paraphrases;
- graph traversal handles hierarchy and remissions;
- temporal filters enforce version validity;
- reranking combines relevance and graph evidence.

When validity facts are authoritative, temporally invalid nodes are excluded rather than merely penalized.

## Graph expansion constraints

Expansion SHALL be bounded by:

- maximum depth;
- maximum nodes;
- allowed edge types;
- document/version validity;
- provenance availability;
- query intent.

Every expanded candidate SHALL retain the graph path that introduced it.

## Scoring

A documented scoring model MAY combine:

```text
semantic score
lexical score
graph distance
source authority
temporal validity
```

The weights SHALL be versioned and evaluated. No opaque “graph relevance” score may be published without a definition.

## Metrics

Report:

- temporal evidence recall@k;
- temporal precision@k;
- exact-version correctness;
- stale-evidence rate;
- anachronistic-citation rate;
- temporal abstention accuracy;
- graph expansion size;
- latency;
- indexing cost;
- standard retrieval metrics.

Compare Temporal GraphRAG against applicable non-temporal baselines on the same temporal split.

## Parallel execution

Independent temporal queries MAY run concurrently under ADR-0014.

Graph construction MAY process independent documents concurrently, but shared graph writes SHALL follow the storage adapter's safety contract.

CPU-heavy graph clustering and community detection SHALL not use naive Python threads. A process pool or graph-engine-native parallelism requires a later implementation decision.

## Audit integration

ADR-0016 SHALL evolve to validate:

- exact cited version;
- validity at `as_of`;
- evidence for amendment and revocation relationships;
- absence of mixed incompatible versions;
- presence of the cited version in the retrieved context.

A publishable temporal answer SHOULD abstain when required validity facts are unknown.

## Lineage integration

ADR-0017 SHALL record:

- original temporal expression;
- normalized `as_of`;
- version-selection policy;
- included and excluded versions;
- graph paths;
- edge provenance;
- temporal filter decisions;
- exact cited versions.

## Consequences

### Positive

- Creates a differentiated advanced legal-RAG experiment.
- Targets anachronistic evidence directly.
- Preserves lexical and dense baselines.
- Makes temporal claims auditable.

### Negative

- Requires substantial curation.
- Legal effectiveness can be more complex than one interval.
- Graph extraction errors can create false relations.
- Versioned indexes increase storage.
- Latency will likely exceed simple retrieval.

## Alternatives considered

- **Replace all retrieval with a graph:** rejected because exact and semantic retrieval remain necessary.
- **Temporal fields without version-qualified IDs:** rejected because evidence must identify exact wording.
- **Infer all relations with an LLM:** rejected because generated metadata is not authoritative.
- **Implement before a temporal golden set:** rejected because benefits would not be measurable.
- **Treat publication as effectiveness:** rejected because they are distinct legal events.

## Acceptance criteria

### Prerequisites

- [ ] Temporal documents have immutable versions.
- [ ] Citations resolve to exact versions.
- [ ] Temporal metadata has source evidence.
- [ ] Curated temporal test split exists.
- [ ] Temporal audit behavior exists.

### Strategy

- [ ] Temporal GraphRAG is separately configurable.
- [ ] Current baselines remain.
- [ ] Temporal filtering precedes expansion.
- [ ] Invalid versions cannot re-enter.
- [ ] Paths and edge evidence are recorded.
- [ ] Temporal and standard metrics are published.
- [ ] Documentation claims measured reduction, not elimination of hallucination.

## Rollout

1. Curate versioned sources.
2. Introduce version-qualified IDs.
3. Add temporal domain models.
4. Build temporal golden questions.
5. Implement deterministic version filtering.
6. Add validated graph relationships.
7. Implement hybrid temporal retrieval.
8. Integrate temporal audit.
9. Compare against current strategies.
10. Promote only if measured benefits justify the complexity.
