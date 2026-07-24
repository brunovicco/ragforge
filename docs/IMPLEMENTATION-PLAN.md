# RAGForge implementation plan

## Current release

1. ADR-0012 — benchmark integrity and end-to-end evaluation.
2. ADR-0013 — provider-neutral embeddings with local default.
3. ADR-0014 — bounded deterministic parallel execution.
4. ADR-0015 — Summary-Augmented Chunking experiment.
5. ADR-0018 — independent calibrated LLM judge.
6. ADR-0016 — post-generation citation audit.
7. ADR-0017 — auditable evidence lineage.

## Future release

8. ADR-0019 — Temporal GraphRAG experimental strategy.

Temporal GraphRAG remains blocked by a versioned temporal corpus, exact version-qualified structural IDs, and a temporal golden set.

## Dependency order

```text
ADR-0012 benchmark integrity
├── ADR-0013 embeddings
├── ADR-0014 parallel scheduler
└── ADR-0017 evidence schema foundation

ADR-0013 + ADR-0014
└── ADR-0015 SAC

ADR-0018 calibrated judge
└── ADR-0016 semantic audit and bounded rewrite

ADR-0012 + ADR-0016 + ADR-0017
└── ADR-0019 Temporal GraphRAG
```

## Increment 1 — Integrity foundation

- Add corpus manifest and split schema.
- Remove hard-coded document discovery.
- Index all five documents.
- Validate all structural references.
- Build RAPTOR per document.
- Include all 230 questions.
- Keep unanswerable questions.
- Emit one record for each question and strategy.

Exit gate: integrity failures stop before evaluation and all selected questions have explicit outcomes.

## Increment 2 — Provider-free embeddings

- Add provider-neutral port.
- Wrap Gemini.
- Add local Sentence Transformers adapter.
- Use Qwen3-Embedding-0.6B as the operational local default.
- Add BGE-M3 and multilingual E5.
- Isolate indexes by complete identity.

Exit gate: retrieval benchmark runs without an API key.

## Increment 3 — Parallel runner

- Implement serial reference scheduler.
- Add one stage-aware `ThreadPoolExecutor`.
- Add per-provider semaphores and rate limits.
- Preserve canonical result order.
- Add atomic cache publication.
- Add request coalescing.
- Add resume and cancellation.
- Measure serial versus parallel wall-clock duration.

Use threads for I/O-bound stages:

- hosted embeddings;
- storage retrieval;
- generation;
- semantic auditing;
- LLM judging.

Use model-native batching for local embeddings. Do not create one thread per local chunk.

Exit gate: `workers=1` and `workers>1` produce equivalent metrics and deterministic artifacts.

## Increment 4 — SAC

- Separate source text from retrieval text.
- Generate and cache one summary per document version.
- Add SAC and SAC+Contextual variants.
- Implement Document-Level Retrieval Mismatch.
- Publish ablation results across two embedding families.

Exit gate: SAC is promoted only after measured full-split improvement.

## Increment 5 — Canonical judge

Use:

```yaml
provider: openai
model: gpt-5.4-mini-2026-03-17
reasoning_effort: medium
cache_mode: required
```

- Add judge port.
- Add strict schemas.
- Integrate RAGAS behind the port.
- Calibrate at least 30 PT-BR legal samples.
- Publish agreement.
- Keep Gemini as exploratory fallback.
- Add a local open-weight judge only as an experiment.

Exit gate: weighted Cohen's kappa is at least 0.60, or the limitation is explicitly published and the leaderboard remains qualified.

## Increment 6 — Citation auditor

- Add claim segmentation.
- Add deterministic syntax, existence, and context checks.
- Add semantic support verification.
- Permit one rewrite.
- Abstain after failed re-audit.
- Score original and audited answers.

Exit gate: unsupported material cannot pass unchanged.

## Increment 7 — Evidence lineage

- Add immutable run manifest.
- Add atomic per-question artifacts.
- Add serialized event writer and hash chain.
- Record source, embedding, retrieval, generation, audit, and judge identities.
- Add `ragforge verify-run`.
- Keep external telemetry metadata-only by default.

Exit gate: every published score can be traced to exact inputs and configurations.

## Future increment — Temporal GraphRAG

- Curate versioned legal sources.
- Replace ambiguous structural IDs with version-qualified evidence IDs.
- Build temporal questions.
- Apply validity filtering before graph expansion.
- Compare as one new strategy.
- Audit exact temporal citations.

Exit gate: reduced stale-evidence and anachronistic-citation rates on a curated temporal split.
