# ADR-0012: Enforce benchmark corpus integrity and end-to-end evaluation

- Status: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0002, ADR-0003, ADR-0004, ADR-0006, ADR-0007, ADR-0011

## Context

RAGForge compares retrieval and generation strategies over the RegRAG-BR legal-regulatory corpus. A benchmark is valid only when the corpus, structural references, split, indexed documents, generated answers, and reported metrics describe the same immutable experiment.

The current architecture has integrity risks:

1. document discovery may be hard-coded in the runner rather than loaded from a canonical corpus manifest;
2. the golden set covers five documents while a run may index a smaller subset;
3. judgments may reference structural units absent from the indexed corpus;
4. a configured split may not be enforced;
5. retrieval-only and end-to-end generation results do not yet share one run contract;
6. unanswerable questions may be excluded from aggregate reporting;
7. RAPTOR-like structures may be built across document boundaries;
8. failed queries may disappear from the denominator.

These conditions can produce precise-looking but invalid strategy rankings.

## Decision drivers

- Scientific validity.
- Fail-closed execution.
- Reproducibility across machines and providers.
- Stable per-question artifacts.
- Explicit abstention evaluation.
- Compatibility with embeddings, SAC, auditing, lineage, and future routing.

## Decision

RAGForge SHALL introduce a canonical benchmark manifest and a preflight integrity gate. Retrieval evaluation SHALL not start until every selected document, question, and structural reference has been validated against the exact indexed corpus.

### Canonical corpus manifest

Create a versioned manifest:

```yaml
schema_version: 1
corpus_id: regrag-br
corpus_version: "0.2"
documents:
  - norm_id: RES-CMN-4893/2021
    source_path: datasets/regrag-br/sources/res-cmn-4893-2021.pdf
    expected_article_count: 27
    source_sha256: "<sha256>"
    enabled: true
```

The manifest SHALL be the only source for benchmark document discovery. Runtime constants containing document paths SHALL be removed.

Each entry SHALL include:

- canonical norm ID;
- immutable source locator;
- expected structural counts;
- source hash;
- extraction configuration;
- enabled state;
- optional publication and temporal metadata.

### Versioned split

A split artifact is mandatory:

```json
{
  "schema_version": 1,
  "dataset_version": "0.2",
  "train": [],
  "validation": [],
  "test": ["q-0001", "q-0002"]
}
```

The runner SHALL:

- require an explicit split;
- reject unknown IDs;
- reject duplicate IDs across splits;
- record the exact split hash;
- never interpret a missing split as “all questions”.

### Preflight integrity gate

Before indexing or evaluation, validate:

1. each enabled source exists and matches its hash;
2. extraction and article-count validation succeed;
3. each selected question exists exactly once;
4. every relevant structural reference resolves exactly once;
5. every relevant unit belongs to an enabled document;
6. no unresolved structural-ID collision exists in scope;
7. all requested strategies can be constructed;
8. the output directory is new or explicitly resumable.

Any failure SHALL stop the run with a typed integrity error. The runner SHALL not continue with a reduced corpus.

### Per-document hierarchical structures

Document-derived structures, including RAPTOR trees, summaries, and graph projections, SHALL be built inside a document boundary unless the strategy explicitly declares cross-document behavior.

For the current release:

```text
for each document:
    chunks = chunks_by_document[document.id]
    tree = build_raptor_tree(chunks)
    register(tree, document.id)
```

Cross-document RAPTOR clustering is prohibited because a generated node may otherwise combine unrelated norms without one authoritative source.

### End-to-end run contract

A run SHALL support these stages:

```text
manifest validation
→ extraction
→ structural parsing
→ chunking
→ indexing
→ retrieval
→ answer generation
→ post-generation audit
→ LLM-judge evaluation
→ deterministic metrics
→ aggregation
```

Each stage SHALL expose:

```text
pending | running | succeeded | failed | skipped
```

A stage may be skipped only when the configuration explicitly requests a retrieval-only experiment.

### Per-question records

Every selected question SHALL produce one immutable record, including failures:

```json
{
  "question_id": "q-0001",
  "query_class": "exact_factual",
  "strategy": "hybrid",
  "retrieval_status": "succeeded",
  "generation_status": "succeeded",
  "audit_status": "succeeded",
  "judge_status": "succeeded",
  "results": [],
  "answer": {},
  "metrics": {},
  "errors": []
}
```

Failed samples SHALL not disappear. Aggregation SHALL report selected, successful, failed, skipped, and metric-covered counts.

### Unanswerable questions

Unanswerable questions SHALL remain in the test split.

Report:

- abstention precision;
- abstention recall;
- abstention F1;
- false-answer rate;
- evidence-insufficient accuracy.

Retrieval metrics requiring positive references SHALL be `not_applicable`, rather than silently removing the sample.

### Aggregation

Metrics SHALL be reported:

- globally;
- by strategy;
- by query class;
- by document;
- by embedding model;
- for answerable and unanswerable subsets;
- with explicit coverage and denominators.

### Run immutability

A completed run SHALL not be overwritten. A rerun creates a new run ID. Cache reuse is allowed, but cache hits and the full cache identity SHALL be recorded.

## Failure behavior

- corpus mismatch: fail before indexing;
- unresolved judgment reference: fail before evaluation;
- terminal query failure: retain the question record and fail the publishable run;
- exploratory continue-on-error mode: permitted but labeled non-publishable;
- insufficient coverage: no strategy leaderboard is published.

## Security and privacy

Public benchmark content may be retained locally. Future private queries SHALL not be sent to external telemetry by default. IDs, hashes, durations, and bounded metadata may be exported.

## Consequences

### Positive

- Prevents corpus/golden-set drift.
- Makes partial failures visible.
- Establishes reliable inputs for future routing.
- Evaluates abstention explicitly.
- Unifies retrieval-only and end-to-end experiments.

### Negative

- Existing result artifacts become legacy.
- Preflight validation adds startup time.
- Some formerly “successful” exploratory runs will fail.
- Result schemas become more extensive.

## Alternatives considered

- **Keep hard-coded discovery:** rejected because code and dataset can silently diverge.
- **Warn and continue:** rejected because a reduced corpus changes the benchmark.
- **Exclude unanswerable questions:** rejected because abstention is a core legal-RAG behavior.
- **Build one global RAPTOR tree:** rejected for the current release due to provenance loss.

## Acceptance criteria

- [ ] All benchmark documents come from one manifest.
- [ ] All five current documents are indexed.
- [ ] All 230 selected questions produce records.
- [ ] Every relevant structural reference resolves exactly once.
- [ ] The split is enforced and hashed.
- [ ] RAPTOR is built per document.
- [ ] Unanswerable metrics are published.
- [ ] Failed samples remain in coverage.
- [ ] Retrieval-only and end-to-end modes share one schema.
- [ ] Missing-document and unknown-reference tests fail closed.

## Rollout

1. Add corpus and split schemas.
2. Replace hard-coded document discovery.
3. Implement preflight validation.
4. Fix per-document RAPTOR construction.
5. Introduce per-question result records.
6. Connect generation, audit, and judge stages.
7. Run the full split.
8. Mark earlier partial artifacts as legacy.
