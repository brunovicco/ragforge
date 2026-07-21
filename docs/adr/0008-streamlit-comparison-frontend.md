# ADR-0008: Comparison frontend in Streamlit (dashboard + arena)

- Status: Accepted
- Date: 2026-07-21

## Context

The project is a public showcase of RAG expertise on GitHub. Beyond the planned analytical dashboard, an **interactive comparison** mode is required: submit a query and see strategies side by side. The question is the stack: extend Streamlit or build a dedicated React/Next.js app consuming the FastAPI.

## Decision

**Extended Streamlit**, in `apps/dashboard/`, with two views:

1. **Benchmark (analytical):** strategy × class matrix, quality × cost pareto, drill-down by class and by question, router vs best-fixed vs oracle chart (ADR-0003). Reads exclusively from versioned results in `experiments/` - works offline, no API key, straight from a clone.
2. **Arena (live comparison):** free query → execution across 2–4 selected strategies via the FastAPI → side-by-side panels with answer, citations traceable to the article (ADR-0006), retrieved chunks, router decision with rationale, evidence-evaluator verdict, latency and cost per panel. The Arena produces the README GIF.

Rationale: the target audience (technical evaluators of RAG knowledge) judges the benchmark and the architecture, not the frontend. Streamlit keeps the repository 100% Python - covered by the same harness quality gates (ADR-0009) - and fits the D16–17 budget. A React app would cost ~3–4 extra days, introduce a second stack outside the harness, and not improve the portfolio's central argument.

The architectural separation (frontend consumes only the FastAPI and `experiments/` artifacts; zero retrieval logic in the dashboard) makes a post-v0.1 swap to React/Next.js a cheap, reversible decision.

## Consequences

- Fits the sprint; no additional stack, JS build, or extra CI.
- The Arena visually demonstrates the router, corrective workflow and citations - the project's three differentiators - in a single GIF.
- Offline benchmark mode makes the repo demonstrable by anyone at zero API cost.
- Aesthetics and interactivity limited vs React; 4-strategy side-by-side layout demands discipline with `st.columns`.
- Live Arena consumes paid API - mitigated with cached demo queries (reusing the ADR-0004 layer) and a cap on simultaneous strategies.

## Alternatives considered

- **Dedicated React/Next.js** - rejected in v0.1: 3–4 days and a second stack for aesthetic gain that does not move the portfolio goal. Natural post-v0.1 candidate.
- **Analytical dashboard only, no Arena** - rejected: interactive comparison is a requirement and the most persuasive demo artifact.
- **Gradio** - rejected: little gain over Streamlit for multi-panel layout and less flexible for the analytical view.
