---
paths:
  - "src/**/*.py"
---

# Clean Architecture rules

- Preserve dependency direction: entrypoints -> application -> domain; adapters -> application/domain.
- Domain must not import Pydantic, web frameworks, ORMs, messaging clients, cloud SDKs, or concrete adapters.
- Define protocols on the consumer side, near the use case that needs them.
- Application services coordinate work; domain objects enforce business invariants.
- Translate Pydantic, ORM, SDK, and transport objects at boundaries.
- Translate infrastructure exceptions before they leave an adapter.
- Keep the composition root in or near the entrypoint.
- Add abstractions for demonstrated variation or isolation, not ritualistically.
- Do not move simple CRUD through unnecessary layers when no domain behavior exists.

# RAGForge-specific boundaries (ADR-0009)

- LLM/framework SDKs (openai, anthropic, langchain*, langgraph, lightrag, ragas) are allowed only
  at the edges: `adapters/`, `generation/`, routing adapters, and `apps/`. Never in `domain/`,
  `chunking/`, `evaluation/metrics/`, or `governance/` (enforced by `scripts/validate_architecture.py`).
- Every retrieval strategy implements the `RetrievalStrategy` Protocol from `ragforge.domain.protocols`.
- Chunks must always carry `structural_ids` (ADR-0006); never emit a chunk without provenance.
- Do not modify `datasets/regrag-br/` or `experiments/` without explicit user confirmation:
  curated data and official run results are treated as immutable evidence (ADR-0003/0004).
