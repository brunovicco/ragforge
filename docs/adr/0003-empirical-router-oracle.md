# ADR-0003: Empirical per-question oracle for router evaluation

- Status: Accepted
- Date: 2026-07-21

## Context

The original plan defined the "oracle" as the a-priori mapping `query class → expected strategy` (e.g. multi-hop → GraphRAG). Evaluating the Adaptive Router against it is circular: it measures agreement with the author's intuition, not routing quality. There is also leakage risk if the router's few-shot examples come from the golden set itself.

## Decision

1. **Empirical per-question oracle.** Since the benchmark already runs all 8 strategies over all questions, the oracle is defined *ex post*: for each question, the strategy with the best composite score (RAGAS quality primary; cost as tie-breaker). The a-priori class mapping is demoted to a **testable hypothesis** — and the "where intuition was wrong" table becomes a report section.
2. **Router metrics:** routing accuracy vs the empirical oracle; **regret** (per-question quality delta between chosen strategy and oracle); comparison of three policies: adaptive router vs best-fixed-strategy vs oracle (ceiling).
3. **Anti-leakage split.** Before curation, a stratified-by-class split: `dev` (~15%, sole source of few-shot examples and rule tuning) and `test` (~85%, frozen; all README numbers come from it). Versioned in `datasets/regrag-br/splits.json`.

## Consequences

- The router-vs-fixed-vs-oracle chart gains methodological validity; regret quantifies the real cost of misrouting.
- Divergence between a-priori hypothesis and empirical oracle is a publishable finding, not an embarrassment.
- The oracle only exists after the full benchmark run — router evaluation (D14–15) depends on the 8-strategy run being complete.
- A ~180-question test set (~26/class) implies wide per-class confidence intervals; report bootstrap CIs and avoid categorical ranking claims where CIs overlap.

## Alternatives considered

- **A-priori class oracle** — rejected: circular; assumes what the benchmark should demonstrate.
- **Human-annotated ideal strategy** — rejected: expensive, subjective, equally hypothetical.
- **No oracle (router vs fixed only)** — rejected: loses the ceiling that shows how much headroom routing still has.
