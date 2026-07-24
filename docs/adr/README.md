# Architecture Decision Records - RAGForge

Format: [MADR-style](https://adr.github.io/). ADRs are immutable once accepted; changes supersede via a new ADR.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-clean-architecture.md) | Adopt Clean Architecture dependency boundaries | Accepted |
| [0002](0002-article-level-relevance-judgments.md) | Relevance judgments at norm-article level, not chunk level | Accepted |
| [0003](0003-empirical-router-oracle.md) | Empirical per-question oracle for router evaluation | Accepted |
| [0004](0004-benchmark-reproducibility-policy.md) | Benchmark reproducibility via versioned LLM call cache | Accepted |
| [0005](0005-embedding-comparison-scope.md) | Embedding comparison restricted to Dense and Hybrid | Accepted |
| [0006](0006-legal-structural-chunker.md) | Structural chunking by normative hierarchy (Art./§/item) | Accepted |
| [0007](0007-llm-judge-calibration-ptbr.md) | LLM judge (RAGAS) calibration against human evaluation in PT-BR | Accepted |
| [0008](0008-streamlit-comparison-frontend.md) | Comparison frontend in Streamlit (dashboard + arena) | Accepted |
| [0009](0009-scaffold-via-engineering-harness.md) | Scaffold and quality gates via claude-python-engineering-harness | Accepted |
| [0010](0010-graphrag-evaluation-scope.md) | GraphRAG evaluation scope and provenance recovery | Accepted |
| [0011](0011-structural-id-collision-in-amended-norms.md) | Structural-ID collisions in amended norms | Accepted |
| [0012](0012-benchmark-integrity-and-end-to-end-evaluation.md) | Enforce benchmark corpus integrity and end-to-end evaluation | Proposed |
| [0013](0013-provider-neutral-embedding-backends.md) | Adopt provider-neutral embedding backends with a local default | Proposed |
| [0014](0014-bounded-parallel-benchmark-execution.md) | Use bounded, deterministic parallel execution for benchmark stages | Proposed |
| [0015](0015-summary-augmented-chunking.md) | Evaluate Summary-Augmented Chunking as a separate retrieval strategy | Proposed |
| [0016](0016-post-generation-citation-audit.md) | Add bounded post-generation citation and support auditing | Proposed |
| [0017](0017-auditable-evidence-lineage.md) | Produce auditable and tamper-evident evidence lineage for every run | Proposed |
| [0018](0018-independent-llm-judge-provider.md) | Use an independent and calibrated OpenAI LLM judge | Proposed |
| [0019](0019-temporal-graphrag-experimental-strategy.md) | Add Temporal GraphRAG only as a future experimental strategy | Proposed |
