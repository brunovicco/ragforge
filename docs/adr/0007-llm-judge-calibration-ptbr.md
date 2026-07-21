# ADR-0007: LLM judge (RAGAS) calibration against human evaluation in PT-BR

- Status: Accepted
- Date: 2026-07-21

## Context

Generation metrics (Faithfulness, Answer Relevancy, Factual Correctness) rely on LLM-as-judge. Published RAGAS validation is mostly general-domain English; Brazilian legal-regulatory Portuguese is barely validated territory — normative negations ("é vedado", "salvo disposto"), cross-references and formulaic language can confuse the judge. Publishing a strategy ranking on an unvalidated judge is the easiest attack point on the benchmark. Additionally, Citation Accuracy is not a native RAGAS metric and requires its own implementation.

## Decision

1. **Calibration sample:** ~30 (question, context, answer) triples stratified by class, manually scored for Faithfulness and Correctness during the already-planned daily curation.
2. **Measure agreement** judge × human (weighted Cohen's kappa for ordinal scales; Spearman for continuous scores) and **publish the number** in README/datasheet, with disagreement examples.
3. **Acceptance criterion:** kappa ≥ 0.6 validates the judge for the main report. Below that, iterate the judge prompt (PT-BR instructions, legal few-shot) and re-measure; if still unmet, generation metrics are reported with the agreement as an explicit caveat — never omitted.
4. **Citation Accuracy** is implemented in `src/ragforge/evaluation/`: the proportion of citations in the answer whose structural IDs (ADR-0006) belong to the judgment's relevant set (ADR-0002). Deterministic, no LLM judge.
5. Judge calls go through the ADR-0004 cache layer — README scores are reproducible.

## Consequences

- The benchmark declares the reliability of its own measuring instrument — rare rigor and a valuable article section.
- Deterministic Citation Accuracy anchors the report in at least one judge-free quality metric.
- ~2–3h of extra human evaluation in week 3; low marginal cost by reusing the curation flow.
- Real risk of low kappa — discovering that before publication is the point of this decision.

## Alternatives considered

- **Trust RAGAS uncalibrated** — rejected: central methodological weakness on a PT-BR legal corpus.
- **Multi-judge ensemble** — rejected in v0.1: multiplies cost; backlog candidate if single-judge agreement is insufficient.
- **Fully human evaluation** — rejected: does not scale to 8 strategies × 210 questions.
