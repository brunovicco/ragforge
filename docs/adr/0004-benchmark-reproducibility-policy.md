# ADR-0004: Benchmark reproducibility via versioned LLM call cache

- Status: Accepted
- Date: 2026-07-21

## Context

The Definition of Done requires `make bench` to reproduce "all README numbers on a clean machine". Taken literally this is impossible: generation and LLM judging (RAGAS) are non-deterministic even at `temperature=0`, providers silently update models, and a full run (8 strategies × 210 questions × generation + multiple judge calls per answer) exceeds 10k calls - rerunning from scratch would cost tens of dollars per CI execution.

## Decision

Two execution modes, declared in the README:

1. **`make bench` (default, deterministic):** every LLM call goes through a cache layer keyed by hash of `(model, prompt, parameters)`. The official run's cache is versioned (Git LFS or release asset) under `experiments/<run-id>/llm-cache/`. With a warm cache the pipeline reproduces README numbers **bit-for-bit**, with no API key and no cost. This is the mode the DoD references.
2. **`make bench-live`:** re-executes against providers. The README declares expected tolerance (±2pp on aggregate metrics) and the estimated run cost, published as a metric.

Supporting rules: `temperature=0` and fixed `seed` where supported; exact model versions pinned in `configs/experiments/*.yaml`; embeddings and derived indexes likewise (corpus hash already planned); every published number carries a traceable `run-id` under `experiments/`.

## Consequences

- Real, auditable reproducibility - rare among RAG benchmark repos and a strong portfolio argument.
- CI can run the full bench in cache mode with zero API cost and no flakiness.
- The versioned cache documents the exact responses behind every number - audit evidence aligned with the governance layer.
- The cache layer must exist from day 1 and intercept **all** LLM call paths, including those internal to RAGAS and LightRAG - the main technical risk; validate interception on both during week 1.
- A 10k+ response cache weighs tens of MB - requires Git LFS or release-asset distribution.

## Alternatives considered

- **`temperature=0` and hope** - rejected: does not survive provider model updates; the DoD would be a false promise.
- **Statistical tolerance without cache** - rejected as default: expensive, slow, flaky CI; kept only as `bench-live`.
- **Local models for everything** - rejected in v0.1: changes the quality profile and scope of the benchmark.
