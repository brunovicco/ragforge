# Architecture

## Purpose and scope

RAGForge is an **experimental RAG evaluation platform**, not an end-user RAG
service. Its job is to compare retrieval architectures over a frozen,
versioned regulatory corpus and produce evidence that makes the comparison
inspectable and reproducible.

The current system:

1. ingests official Brazilian legal/regulatory sources;
2. preserves source provenance and snapshot hashes;
3. performs legal-structure-aware chunking;
4. builds strategy-specific retrieval indexes;
5. runs multiple retrieval configurations over the frozen RegRAG-BR split;
6. generates cited answers from authoritative source text;
7. evaluates retrieval quality, answer quality, citation behavior, and
   abstention;
8. writes per-run evidence, manifests, hashes, and aggregate results;
9. exposes only explicitly published aggregate benchmark runs through a
   read-only API/dashboard.

Adaptive routing, the corrective workflow, deterministic replay execution, and
the live Arena are future surfaces and are not part of the current runtime.

## System flow

```text
official source documents
        |
        v
source snapshot + SHA-256 provenance
        |
        v
extraction + legal structural parsing
        |
        v
authoritative chunks
        |
        +--------------------------+
        |                          |
        v                          v
strategy-specific enrichment   authoritative source_text
        |                          |
        v                          |
indexes / retrieval_text           |
        |                          |
        v                          |
10 retrieval configurations -------+
        |
        v
retrieved authoritative evidence
        |
        v
cited answer generation
        |
        +--> deterministic citation checks
        +--> independent LLM judge
        +--> optional semantic support audit
        |
        v
per-question evidence + aggregate metrics
        |
        v
manifest / checksums / event hash chain
        |
        v
explicit publication catalog
        |
        +--> read-only FastAPI
        +--> offline analytical dashboard
```

The important boundary is between **retrieval text** and **authoritative source
text**. A strategy may enrich text to improve retrieval (for example contextual
retrieval or SAC), but generated answers are grounded in the authoritative
source text rather than synthetic enrichment.

## Code boundaries

### Domain

`src/ragforge/domain/`

Framework-free contracts and value objects used across the benchmark, such as
queries, chunks, answers, judgments, retrieval results, and strategy
protocols. The architecture guard keeps LLM/framework SDK imports out of the
core.

### Benchmark orchestration and evaluation

`src/ragforge/evaluation/`

Owns benchmark execution, metrics, answer-quality evaluation, calibration,
cache/index identity, lineage, integrity checks, artifacts, and publication
evidence. This is the composition center for the experiment rather than a web
application service layer.

### Retrieval implementations

`src/ragforge/retrieval/`

Contains dense, sparse/BM25, hybrid RRF, reranked, contextual, hierarchical,
SAC, RAPTOR, and GraphRAG-related implementations. Strategies implement common
retrieval contracts so the evaluation harness can compare them under the same
question set and metric semantics.

### Ingestion and chunking

`src/ragforge/ingestion/` and `src/ragforge/chunking/`

Extraction, source snapshot hashing, normalization, and Brazilian legal
structure parsing. Stable structural identifiers are part of the evaluation
contract because relevance judgments are projected at legal-structure level,
not tied to one physical chunking strategy.

### Generation and external model adapters

`src/ragforge/generation/`, `src/ragforge/embeddings/`, and
`src/ragforge/adapters/`

Provider-facing code lives at the edges. Hosted model calls use explicit model
identities, bounded concurrency/retry behavior, optional write-through call
caching, and lineage capture. Retrieval evidence is treated as untrusted data;
provider prompts must not allow instructions embedded in retrieved documents
to redefine system behavior.

### Published-results surfaces

`apps/api/` and `apps/dashboard/`

These surfaces do not run retrieval or call model providers. They consume only
runs explicitly listed in `experiments/published-runs.json`. The catalog is a
publication boundary: incomplete or arbitrary experiment directories are not
automatically exposed.

## External dependencies and trust boundaries

```text
Git-versioned corpus / golden set
        |
        | trusted benchmark inputs after integrity gates
        v
RAGForge benchmark runtime
   |          |          |
   |          |          +--> local LightRAG storage
   |          +-------------> loopback Postgres + pgvector / OpenSearch
   +------------------------> hosted Gemini / OpenAI APIs during live runs

Optional metadata-only tracing --> configured Langfuse backend
```

The current corpus is public official material. That substantially reduces,
but does not eliminate, RAG-specific security concerns such as indirect prompt
injection, poisoned retrieval text, unsupported citations, or provider data
leakage. Those boundaries and controls are documented in
[`THREAT-MODEL.md`](THREAT-MODEL.md) and [`PRIVACY.md`](PRIVACY.md).

## Reproducibility model

RAGForge distinguishes three related concepts:

- **write-through call cache:** live provider calls can be captured under a run
  for traceability and future reuse;
- **versioned benchmark evidence:** manifests, per-question records, model
  identities, hashes, and aggregate results are published with a run;
- **deterministic replay engine:** the planned `make bench` execution mode that
  must reproduce a run from captured calls without provider credentials.

The first two exist today. The deterministic replay engine and its CI gate are
still planned (ADR-0004/ADR-0020); a cache artifact being present must not be
read as evidence that replay is already implemented.

## Quality and security gates

The default quality workflow runs repository-owned checks for lock-file
consistency, linting/formatting, architecture boundaries, typing, unit tests,
Bandit, dependency auditing, and governance-related validators. A separate CI
integration job exercises the real pgvector and OpenSearch adapters without
requiring hosted-model credentials.

Live-provider tests remain opt-in because they are non-deterministic,
credentialed, and cost-bearing.
