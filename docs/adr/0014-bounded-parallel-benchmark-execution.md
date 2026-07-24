# ADR-0014: Use bounded, deterministic parallel execution for benchmark stages

- Status: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0004, ADR-0012, ADR-0013, ADR-0016, ADR-0017, ADR-0018

## Context

A complete benchmark can issue thousands of independent operations:

- source reads;
- hosted embedding batches;
- retrieval queries;
- answer generation calls;
- semantic audit calls;
- LLM-judge calls.

Serial I/O wastes wall-clock time. Unbounded concurrency can exceed rate limits, corrupt caches, overload stores, create nondeterministic ordering, and obscure failures.

Python threads are appropriate for I/O-bound work. They are not the default solution for local tensor inference or CPU-heavy graph algorithms, where model batching, native runtimes, or later process-based execution are preferable.

## Decision drivers

- Reduce wall-clock time.
- Preserve deterministic artifacts.
- Respect provider quotas.
- Avoid nested concurrency.
- Keep caches and evidence writes safe.
- Make limits observable and reproducible.
- Support fail-closed resume.

## Decision

RAGForge SHALL use one stage-aware bounded scheduler. I/O-bound work MAY execute with `ThreadPoolExecutor`. Local tensor inference SHALL prefer model-native batching.

### Configuration

```yaml
execution:
  mode: threaded
  fail_fast: true
  preserve_input_order: true
  global_max_in_flight: 8

  stages:
    extraction:
      workers: 4
    hosted_embedding:
      workers: 4
      max_in_flight_per_provider: 4
    retrieval:
      workers: 8
    generation:
      workers: 4
      max_in_flight_per_provider: 4
    audit:
      workers: 4
    judge:
      workers: 4
      max_in_flight_per_provider: 4
```

Defaults SHALL be conservative and every resolved value SHALL appear in the run manifest.

### Threaded stages

Threads MAY be used for:

- independent document I/O;
- remote embedding requests;
- thread-safe retrieval storage calls;
- remote generation;
- remote semantic audit;
- remote LLM judging;
- independent artifact reads.

### Non-threaded stages

Do not create naive per-item threads for:

- local embedding inference;
- large matrix operations;
- CPU-heavy graph clustering;
- CPU-heavy parsing that does not release the GIL;
- writes to one non-thread-safe index;
- aggregation over shared mutable state.

Local embeddings SHALL use batches. CPU-bound process pools require a later explicit decision.

### One scheduler, no nested executors

The runner owns the scheduler. Strategies and adapters SHALL not create private executors.

Libraries with internal thread pools SHALL be configured to avoid oversubscription where possible.

### Deterministic order

Each task receives a stable ordinal:

```text
(document_order, strategy_order, question_order, stage_order)
```

Workers may finish in any order. Persisted arrays and JSONL outputs SHALL be restored to the stable order. Metrics SHALL never depend on completion order.

### Immutable inputs

Worker tasks SHALL receive immutable query, judgment, chunk, and configuration objects. Shared mutable collections are prohibited.

### Provider limiter

Every hosted provider SHALL have a limiter covering:

- concurrent requests;
- requests per minute;
- tokens per minute when available;
- `Retry-After`;
- bounded exponential backoff with jitter.

Retries SHALL preserve idempotency and cache identity.

### Client lifecycle

Adapters SHALL declare:

```python
class ConcurrencyCapability(StrEnum):
    THREAD_SAFE = "thread_safe"
    CLIENT_PER_THREAD = "client_per_thread"
    SERIAL_ONLY = "serial_only"
```

Thread-local clients SHALL be used when required. SDK thread safety SHALL not be assumed without documentation or tests.

### Cache safety

Concurrent calls sharing one cache key SHOULD coalesce into one provider call.

Cache publication SHALL be atomic:

1. write temporary artifact;
2. flush and validate;
3. atomic rename or transaction;
4. never expose partial payloads.

### Storage writes

Workers MAY write unique per-question artifacts. Shared reports and event logs SHALL use a single writer.

Vector index writes SHALL follow the storage adapter's thread-safety declaration. Unknown safety means single-writer mode.

### Failure semantics

Publishable mode is fail-closed:

- after terminal failure, dependent work is not scheduled;
- running tasks may finish;
- completed artifacts remain resumable;
- run status becomes failed;
- no ranking is published.

Exploratory collection mode is allowed but SHALL be labeled non-publishable.

### Cancellation and resume

Workers SHALL honor cancellation between operations. A resumed run SHALL verify the original configuration, corpus, split, and model identities before skipping completed tasks.

Changing model, prompt, split, or strategy creates a new run.

### Measurement

Record:

- wall-clock duration;
- configured and observed concurrency;
- queue wait time;
- provider latency;
- retries;
- rate-limit events;
- cache-hit ratio;
- throughput by stage.

Performance comparisons SHALL use the same workload and cache state.

## Consequences

### Positive

- Faster hosted stages.
- Explicit quota control.
- Deterministic output ordering.
- Resumable failed runs.
- Useful capacity metrics.

### Negative

- More runner complexity.
- Thread-safety requires dedicated tests.
- Parallel calls can consume quotas faster.
- Local inference may not benefit from threads.

## Alternatives considered

- **Serial only:** rejected for I/O-bound workloads.
- **Async rewrite of the entire project:** deferred because current adapters are synchronous.
- **Executor per strategy:** rejected due to nested and unpredictable concurrency.
- **Thread per chunk:** rejected due to oversubscription.
- **Publish successful subset:** rejected because missing samples bias rankings.

## Acceptance criteria

- [ ] Bounded pools exist for I/O-bound stages.
- [ ] Local embeddings use batching.
- [ ] `workers=1` and `workers>1` produce identical ordering and equivalent metrics.
- [ ] A fake slow provider demonstrates speedup.
- [ ] Provider concurrency never exceeds limits.
- [ ] Duplicate calls cannot corrupt the cache.
- [ ] `Retry-After` is honored.
- [ ] Failed tasks remain resumable.
- [ ] No strategy creates a nested executor.
- [ ] The manifest records concurrency settings.

## Rollout

1. Add scheduler port and configuration.
2. Implement serial reference scheduler.
3. Implement bounded thread scheduler.
4. Parallelize retrieval with deterministic tests.
5. Parallelize hosted embeddings.
6. Parallelize generation, audit, and judge calls.
7. Add request coalescing.
8. Add resume and cancellation.
9. Compare serial and threaded wall-clock performance.
