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
| [0010](0010-graphrag-evaluation-scope.md) | GraphRAG (LightRAG) evaluation scope and provenance recovery | Accepted |
| [0011](0011-structural-id-collision-in-amended-norms.md) | Structural-ID collisions in amended norms excluded from golden-set citations, not fixed at the source | Accepted |
