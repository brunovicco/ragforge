# Architecture

## Context

RAGForge is an experimental evaluation platform, not an end-user RAG service.
It compares retrieval strategies over the versioned RegRAG-BR regulatory
corpus, generates grounded answers, scores retrieval and answer quality, and
produces tamper-evident evidence for every publishable run.

External dependencies are official corpus snapshots, hosted Gemini/OpenAI
model APIs, Postgres+pgvector, and OpenSearch. LightRAG uses repository-local
storage. The v0.1 read-only API and analytical dashboard consume explicitly
published benchmark results. The adaptive router, corrective workflow, and
live comparison Arena remain future product surfaces.

## Layers

```text
src/ragforge/
├── domain/
├── application/
├── adapters/
└── entrypoints/
```

### Domain

Pure business concepts, invariants, Value Objects, domain services, events, and domain errors.

### Application

Use cases, commands, queries, ports, authorization decisions, and transaction coordination.

### Adapters

Implementations of application ports for databases, messaging, HTTP, cache, storage, identity, and external SDKs.

### Entrypoints

HTTP, CLI, jobs, events, and serverless handlers. Entrypoints validate and translate transport data but do not own business rules.

## Dependency rule

```text
entrypoints -> application -> domain
adapters    -> application/domain
domain      -> no outer layer
```

## Cross-cutting decisions

- Configuration: environment variables validated at startup.
- Logging: structured events to stdout/stderr.
- Tracing: W3C trace context propagated across boundaries.
- Errors: infrastructure errors translated at adapters; external errors mapped at entrypoints.
- Time: UTC internally with timezone-aware values.
- Money: `Decimal` wrapped in a domain Value Object.
- Idempotency: required for externally visible side effects.
- Packaging: containerized via the repo `Dockerfile` (multi-stage, uv-based); the runtime `CMD` is defined per project.

## Diagrams

```text
corpus manifest + split + judgments
  -> integrity gate
  -> extraction and legal structural chunking
  -> embedding and strategy-specific indexing
  -> retrieval over the frozen test split
  -> cited answer generation
  -> independent judge and optional semantic audit
  -> aggregate metrics + per-question evidence + checksums
```

The benchmark composition root is `ragforge.evaluation.run`. `domain/` owns
framework-free query, chunk, judgment, and retrieval contracts. Retrieval,
generation, embedding, and storage packages implement boundary behavior.
`evaluation/` coordinates the benchmark and owns its evidence/reporting
contracts.

Published results use a separate read-only path:

```text
experiments/published-runs.json
  -> JsonPublishedBenchmarkRepository (adapter)
  -> PublishedBenchmarkRepository (application contract)
  -> FastAPI v1 responses / Streamlit analytical view
```

The explicit catalog is the publication boundary: arbitrary, incomplete, or
unverified experiment directories are not discovered automatically. The API
does not expose prompts or per-question model responses. The dashboard reads
the same application model directly from versioned artifacts and performs no
retrieval or provider calls.
