# ADR-0017: Produce auditable and tamper-evident evidence lineage for every run

- Status: Accepted
- Date: 2026-07-24
- Target: Current release, with future hardening
- Related: ADR-0004, ADR-0006, ADR-0012, ADR-0014, ADR-0016, ADR-0018, ADR-0019

## Context

RAGForge already carries structural IDs and source hashes, but institutional evaluation requires a complete chain from source acquisition to the final score.

A reviewer should be able to determine:

- which source bytes were used;
- how content was extracted and chunked;
- which synthetic retrieval text was created;
- which embedding and index produced each candidate;
- which chunks reached the generator;
- which claims and citations passed or failed;
- which judge produced each metric;
- whether any artifact changed after completion.

External observability is not evidence lineage. Telemetry can be sampled, content-minimized, or retained under another provider's policy.

The goal is an auditable, tamper-evident chain. It is not “perfect forensic traceability” and does not eliminate legal responsibility.

## Decision drivers

- Reproduce published scores.
- Investigate per-question failures.
- Preserve authoritative source provenance.
- Detect post-run modification.
- Support parallel execution safely.
- Minimize external telemetry content.
- Prepare for temporal versions and signed packages.

## Decision

Every run SHALL produce a versioned local evidence directory containing a manifest, immutable per-question records, stage events, checksums, and reports.

### Artifact layout

```text
artifacts/runs/<run_id>/
├── manifest.json
├── configuration.resolved.yaml
├── corpus-manifest.snapshot.yaml
├── split.snapshot.json
├── events.jsonl
├── questions/
│   └── <question-id>/<strategy-id>.json
├── summaries/
│   ├── retrieval.json
│   ├── generation.json
│   └── audit.json
├── report.json
├── report.md
└── checksums.sha256
```

A completed directory SHALL not be overwritten by the application.

### Run manifest

```json
{
  "schema_version": 1,
  "run_id": "20260724T010000Z-<random>",
  "status": "running",
  "git_sha": "<sha>",
  "started_at": "...",
  "completed_at": null,
  "corpus_hash": "...",
  "dataset_hash": "...",
  "split_hash": "...",
  "configuration_hash": "...",
  "models": {},
  "strategies": [],
  "execution": {},
  "artifact_root_hash": null
}
```

Environment data SHALL be allowlisted and SHALL not contain secrets.

### Source lineage

For each document record:

```text
canonical document ID
official authority
source URL or source locator
retrieval timestamp
publication date when known
source SHA-256
media type
extractor name and version
extracted-text SHA-256
article-count validation result
```

PDF sources SHOULD include page references when supported. HTML and text sources SHALL use offsets, selectors, headings, or structural paths instead of invented page numbers.

### Chunk lineage

Record:

```text
chunk ID
document ID and version
structural IDs
parent ID
source location
source-text hash
retrieval-text hash
chunking configuration hash
summary/context enrichment identities
```

Generated enrichment SHALL have distinct hashes and model provenance.

### Retrieval lineage

For each candidate:

```text
query ID and query hash
strategy
index identity and hash
embedding identity
candidate rank
final rank
raw score
normalized score when applicable
reranker identity and score
retrieval timestamp
```

Candidate removals SHOULD include a machine-readable reason.

### Generation lineage

Record:

```text
provider
model snapshot
prompt-template hash
generation parameters
input chunk IDs
input source hashes
answer hash
parsed citations
token usage
latency
cache hit
```

Prompts and answers remain local under the configured retention mode.

### Audit and judge lineage

Record independently:

```text
deterministic validation results
semantic auditor identity
audit prompt hash
rewrite count
original answer hash
final answer hash
judge provider and snapshot
judge prompt hash
metric implementation version
raw structured output
final metric
```

Audit provider and judge provider SHALL not be conflated.

### Event envelope

```json
{
  "schema_version": 1,
  "sequence": 42,
  "event_id": "...",
  "run_id": "...",
  "correlation_id": "...",
  "stage": "retrieval",
  "event_type": "completed",
  "occurred_at": "...",
  "payload_hash": "...",
  "previous_event_hash": "...",
  "event_hash": "..."
}
```

`event_hash` covers canonical serialization excluding itself. `previous_event_hash` forms a local tamper-evident chain.

### Parallel event handling

Parallel workers SHALL NOT write directly to the shared event stream.

Workers submit events to a single serialized writer that assigns monotonic sequence numbers and updates the hash chain.

Per-question artifacts may use unique independent paths.

### Atomic writes

Artifacts SHALL use:

1. temporary write;
2. validation;
3. flush;
4. atomic rename or transaction;
5. checksum generation after all files close.

Manifest status changes to `completed` only after checksum verification.

A failed run remains inspectable.

### External telemetry

External observability SHALL receive metadata only by default:

```text
run ID
correlation ID
stage
operation
provider
model
latency
token counts
outcome
retry count
cache hit
```

Question text, source text, prompts, and answers require explicit opt-in.

The local evidence directory is the source of truth.

### Retention modes

```text
full_local
hashes_and_metrics
public_benchmark
```

Private institutional runs default to hashes and metrics plus separately controlled encrypted content storage.

### Verification

Provide:

```bash
ragforge verify-run <run-id>
```

The command SHALL:

- validate checksums;
- validate canonical manifests;
- validate the event hash chain;
- verify referenced artifacts exist;
- report modifications and missing evidence;
- never repair artifacts silently.

### Future hardening

Future work may add:

- digital signatures;
- trusted timestamps;
- append-only object storage;
- WORM/object lock;
- external transparency logs;
- key rotation;
- legal hold and retention policy.

Local hashes detect alteration but do not prove who altered data.

## Consequences

### Positive

- Each score traces to exact inputs and models.
- Post-run changes become detectable.
- Parallel execution remains auditable.
- External telemetry stays content-minimized.
- Temporal and institutional controls gain a stable base.

### Negative

- Artifact volume increases.
- Schemas require migrations.
- Atomicity and canonical serialization add work.
- Sensitive content needs retention controls.
- Hash chains alone do not establish identity or intent.

## Alternatives considered

- **Langfuse only:** rejected because traces may be sampled or externally retained.
- **Final aggregate metrics only:** rejected because failures cannot be investigated.
- **All content in external telemetry:** rejected for private future corpora.
- **Claim perfect forensic lineage:** rejected as technically and legally indefensible.

## Acceptance criteria

- [ ] Every run creates the defined structure.
- [ ] Manifest includes corpus, split, config, Git, model, and concurrency identities.
- [ ] Per-question retrieval, generation, audit, and judge records are linked.
- [ ] Final artifacts appear in `checksums.sha256`.
- [ ] Event chain verifies.
- [ ] Parallel workers use a serialized event writer.
- [ ] Completed artifacts are not overwritten.
- [ ] External telemetry is metadata-only by default.
- [ ] Verification detects a modified artifact.
- [ ] Failed runs preserve completed evidence.

## Rollout

1. Define canonical schemas.
2. Add run and correlation IDs.
3. Implement atomic artifact writer.
4. Implement serialized event writer and hash chain.
5. Add stage lineage.
6. Integrate audit and judge records.
7. Add verification command.
8. Add report generation.
9. Evaluate signatures and WORM storage later.
