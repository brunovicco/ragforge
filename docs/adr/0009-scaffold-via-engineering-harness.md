# ADR-0009: Scaffold and quality gates via claude-python-engineering-harness

- Status: Accepted
- Date: 2026-07-21

## Context

RAGForge is developed with Claude Code, and the author maintains [claude-python-engineering-harness](https://github.com/brunovicco/claude-python-engineering-harness): a scaffold with CLAUDE.md, rules, hooks, agents, skills and CI gates (Ruff, Mypy, Pytest, Bandit, pip-audit), plus `scripts/validate_architecture.py` for configurable import boundaries. The RAGForge plan already requires CI with lint, tests (core ≥ 80%) and a guard preventing LLM SDK imports in the core — requirements the harness covers natively. Using the author's own harness also demonstrates the tool on a real project, reinforcing both repositories.

## Decision

1. **Repository bootstrapped by the harness:** `bootstrap.py --name ragforge --package ragforge --python 3.12 --profile service --git-init`. Divergences between the scaffold and the planned structure (`apps/`, `datasets/`, `experiments/`, `benchmarks/`, `configs/`) are added on top; on conflict, the RAGForge plan structure prevails.
2. **Python 3.12** (RAG ecosystem compatibility — LightRAG, RAGAS, Docling); divergence from the harness 3.13 default recorded here.
3. **Architecture guard:** the "LLM import CI guard" is implemented via the harness boundary config in `pyproject.toml` (`[tool.engineering-harness.architecture.boundaries]`): default-deny boundaries on `chunking/`, `evaluation/metrics/` and `governance/`; LLM/framework SDKs (`openai`, `anthropic`, `langchain*`, `langgraph`, `lightrag`, `ragas`) allowed only at the edges (`adapters/`, `generation/`, `routing` adapters, `apps/`). `RetrievalStrategy` remains a Protocol in `domain/`.
4. **Harness layers adapted to the domain:** `.claude/rules/architecture.md` extended with RAGForge boundaries; protection hooks extended to `datasets/regrag-br/` and `experiments/` (curated data and official results must not be altered by an agent without explicit confirmation — consistent with the immutability required by ADR-0003/0004); the harness human-authorship principle stands (the agent does not commit/publish unrequested).
5. **Gates inherited without reduction:** Ruff, Mypy, Pytest (core ≥ 80%), Bandit, pip-audit and architecture validation run in the harness `quality.yml`, plus a `make bench` job in cache mode (ADR-0004).

## Consequences

- D1–2 of the schedule shrinks — the harness delivers scaffold, CI and guards ready; the slack absorbs the legal chunker (ADR-0006) and the cache layer (ADR-0004).
- Uniform, demonstrable quality standard; RAGForge becomes a real use case of the harness (double portfolio value).
- Dataset/experiment protection is fail-closed (hooks), not convention.
- Coupling between two repositories of the same author: harness convention changes do not back-propagate (the scaffold is copied, not linked) — intentional snapshot.
- Python 3.12 pin should be revisited when the RAG ecosystem stabilizes on 3.13.

## Alternatives considered

- **Manual scaffold** — rejected: rework of something already solved and tested by the author; loses the cross-demonstration.
- **Harness as plugin only (no bootstrap)** — rejected: the plugin covers agents/skills/hooks but not pyproject, CI and `src/` structure.
- **A new bespoke import-guard script** — rejected: the harness `validate_architecture.py` already does exactly this via configuration.
