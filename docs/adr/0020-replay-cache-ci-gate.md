# ADR-0020: Deliver the deterministic replay cache together with its CI gate

- Status: Proposed
- Date: 2026-07-28
- Target: Future release
- Related: ADR-0004, ADR-0009, ADR-0012, ADR-0014, ADR-0017

## Context

ADR-0004 defines two benchmark modes. `make bench-live` re-executes against real providers and is implemented. `make bench` is meant to replay a versioned LLM call cache keyed by hash of `(model, prompt, parameters)`, reproducing published numbers bit-for-bit with no API key and no cost.

The replay mode does not exist. `ragforge.evaluation.run` fails closed on `--mode cache` with an explicit message rather than silently degrading to a live run, and `make bench` therefore always exits non-zero. What exists today under `experiments/<run_id>/llm-cache/` is a **write-through call cache** populated during live runs (ADR-0004's cache primitive, ADR-0014's bounded execution) - it deduplicates repeated calls within and across runs, but there is no recording format, no cache-hit-required enforcement, and no published cache artifact.

ADR-0009 decision item 5 nevertheless lists "a `make bench` job in cache mode (ADR-0004)" among the inherited CI gates. `quality.yml` has no such job. This is the correct outcome - a CI job invoking `make bench` today would fail on every pull request, and adding one that asserts the failure would gate on a placeholder rather than on the property the ADR cares about - but it leaves a documented gate that does not exist, which is precisely the kind of gap this project's evidence discipline (ADR-0017) exists to prevent.

Two options were available. Adding the job now as a fail-closed smoke test would keep ADR-0009 literally satisfied, but it would assert the absence of a feature, and would have to be rewritten the day replay lands - a test whose passing tells a reviewer nothing about reproducibility. Recording the obligation against the work that unblocks it keeps CI honest about what it actually verifies.

## Decision drivers

- CI gates must verify a real property, never the absence of one.
- A documented gate that does not exist is worse than an acknowledged gap.
- Reproducibility claims are the benchmark's main credibility asset (ADR-0004, ADR-0017).
- The obligation must survive in a place someone implementing replay will read.

## Decision

1. **The CI bench job is scoped to this ADR, not to ADR-0009.** ADR-0009's decision item 5 is superseded on this single point: `quality.yml` inherits Ruff, Mypy, Pytest (core ≥ 80%), Bandit, pip-audit and architecture validation, and does **not** carry a `make bench` job until deterministic replay exists. Every other ADR-0009 gate stands unchanged.

2. **No placeholder job is added in the meantime.** `make bench` keeps failing closed with its current explanatory message. `quality.yml` carries a comment recording why the job is absent and pointing here, so the gap is visible to anyone reading the workflow rather than only to anyone reading the ADRs.

3. **The replay implementation lands with its CI job in the same change.** Whoever implements `--mode cache` ships, in one pull request: the recording/replay layer; a published cache artifact for the run being reproduced (Git LFS or release asset, per ADR-0004); a `bench-replay` job in `quality.yml` that runs `make bench` against that artifact **with no provider credentials available in the job environment**, so a cache miss cannot silently fall through to a live call; and an assertion that the replayed aggregate metrics match the published `results.json` exactly, not within a tolerance - the ±2pp tolerance in ADR-0004 belongs to `bench-live` only.

4. **Replay must fail closed on a miss.** A cache miss in `--mode cache` is an error, never a live call and never a skipped question. The job's absence of credentials is a second line of defense, not the mechanism.

5. **Scope of the replayed run.** The CI job replays the canonical published run identified in `experiments/published-runs.json`. If that run's cache artifact is unavailable, the job fails rather than selecting a different run.

## Consequences

- CI states truthfully what it verifies; no green check implies a reproducibility guarantee the codebase cannot make.
- The reproducibility gap stays legible in three places a reader actually visits: this ADR, the ADR index, and `quality.yml` itself.
- Publishable results remain reproducible only by re-running live within ADR-0004's ±2pp tolerance until replay lands; `docs/BENCHMARK-RESULTS.md` should keep saying so.
- Implementing replay becomes a slightly larger unit of work, since the CI job and the published cache artifact are part of its definition of done rather than a follow-up.
- The cache artifact adds repository weight (Git LFS or a release asset); that cost is deferred until replay exists, rather than paid now for an unused artifact.

## Alternatives considered

- **Add a `make bench` job now that expects the fail-closed exit** - rejected: it gates on a placeholder. It would pass for exactly as long as the feature is missing, invert its meaning the day replay lands, and give a reviewer a green check that means "the feature is still absent".
- **Silently drop the ADR-0009 commitment** - rejected: an undocumented missing gate is the failure mode this project's evidence discipline exists to prevent.
- **Edit ADR-0009 in place** - rejected: ADRs are immutable once accepted (see `docs/adr/README.md`); a superseding decision belongs in its own record. ADR-0009 carries only a dated pointer here.
- **Run `make bench-live` in CI instead** - rejected by ADR-0004 already: expensive, slow, provider-dependent and flaky, and it would require production credentials in CI.
