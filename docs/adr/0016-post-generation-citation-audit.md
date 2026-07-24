# ADR-0016: Add bounded post-generation citation and support auditing

- Status: Accepted
- Date: 2026-07-24
- Target: Current release
- Related: ADR-0002, ADR-0006, ADR-0007, ADR-0011, ADR-0014, ADR-0017, ADR-0018

## Context

The answer generator requests structural-ID citations and parses well-formed IDs. Syntax does not establish that:

- the cited device exists;
- it belongs to the selected corpus version;
- it appeared in the retrieved context;
- it was effective at the relevant date;
- its text supports the associated claim;
- all material claims are cited.

Deterministic Citation Accuracy against a golden set is an evaluation metric, not a runtime safety control. A generated answer can cite a valid relevant ID that was never retrieved, or an existing provision that does not support the specific statement.

A post-generation auditor is required. It must be bounded, explicit, and able to abstain. It SHALL NOT start an unlimited recursive correction loop.

## Decision drivers

- Detect invented and unavailable citations.
- Separate syntax, existence, context availability, temporal validity, and semantic support.
- Prefer deterministic checks before LLM verification.
- Bound cost and latency.
- Preserve original and corrected answers.
- Make abstention explicit.
- Keep runtime audit separate from benchmark judging.

## Decision

Every generated answer in end-to-end mode SHALL pass through a structured post-generation audit.

The audit SHALL execute:

```text
claim segmentation
→ deterministic citation validation
→ semantic support validation
→ policy decision
→ optional single rewrite
→ complete re-audit
→ deliver, limit, or abstain
```

### Claim representation

```python
@dataclass(frozen=True, slots=True)
class AnswerClaim:
    claim_id: str
    text: str
    cited_structural_ids: tuple[str, ...]
    sentence_index: int
    material: bool
```

Formatting, greetings, or connective prose may be marked non-material. Legal assertions, conditions, exceptions, dates, duties, prohibitions, and numerical claims are material.

Deterministic sentence segmentation SHOULD be used first. LLM-assisted refinement, when needed, SHALL use structured output and cache.

### Deterministic checks

For each citation:

```text
well_formed
exists_in_corpus
belongs_to_selected_document_version
present_in_retrieved_context
source_text_hash_matches
```

For each material claim:

```text
has_citation
citation_count
```

These checks SHALL not invoke an LLM.

### Temporal status

The schema SHALL support:

```text
valid
invalid
unknown
not_applicable
```

Until a trustworthy temporal corpus exists, the result SHALL be `unknown`. The system SHALL not equate document existence with temporal validity.

### Semantic support verifier

Claims passing deterministic checks MAY be sent to a verifier with only:

- question;
- claim text;
- cited authoritative source text;
- relevant temporal metadata;
- no outside retrieval tools.

The output SHALL use a closed schema:

```json
{
  "support": "supported",
  "rationale": "bounded explanation",
  "supported_citation_ids": [],
  "unsupported_citation_ids": [],
  "missing_evidence": []
}
```

Allowed support states:

```text
supported
partially_supported
unsupported
indeterminate
```

The verifier SHALL be instructed to use only supplied evidence.

### Audit result

```python
@dataclass(frozen=True, slots=True)
class AuditResult:
    outcome: AuditOutcome
    claims: tuple[ClaimAudit, ...]
    rewrite_required: bool
    abstention_required: bool
    temporal_status: TemporalStatus
    verifier_identity: ModelIdentity | None
```

Allowed outcomes:

```text
accepted
accepted_with_caveat
rewrite_required
abstain
audit_failed
```

### Bounded rewrite

The current release SHALL allow at most one rewrite.

The rewrite prompt SHALL include:

- original question;
- original answer;
- valid retrieved source text;
- structured audit findings;
- instruction to remove unsupported statements;
- instruction not to introduce new citation IDs;
- instruction to state insufficient evidence when necessary.

After rewrite, the full audit SHALL run again.

If the second audit does not pass:

- unsupported material claim: abstain;
- supported subset available: return only the supported subset with a limitation;
- audit infrastructure failure: fail closed in publishable mode.

### No recursive correction

The default and maximum for the current release is one rewrite attempt. A higher limit requires a later measured decision.

### Separation of roles

- **Generator:** produces the candidate answer.
- **Auditor:** controls whether the answer is deliverable.
- **Benchmark judge:** scores answer quality for research.

The judge SHALL not replace deterministic audit controls. For research, both original and final answers SHOULD be scored to measure audit impact.

### Metrics

Report:

- malformed citation rate;
- nonexistent citation rate;
- citation-not-in-context rate;
- uncited material-claim rate;
- unsupported-claim rate;
- rewrite rate;
- rewrite success rate;
- abstention rate;
- false abstention rate;
- audit latency;
- audit token usage;
- original-versus-final quality.

### Concurrency

Different questions MAY be audited concurrently under ADR-0014.

Within one question, these dependencies are sequential:

```text
deterministic checks
→ semantic verification
→ rewrite
→ second audit
```

### Cache

Cache semantic verification and rewrite calls by:

```text
question hash
+ answer hash
+ cited source hashes
+ prompt hash
+ provider
+ model snapshot
+ output schema version
```

A changed answer invalidates its audit cache.

## Failure behavior

- invalid citation: do not treat it as semantic evidence;
- verifier timeout: retry within policy, then mark audit failed;
- rewrite failure: abstain or fail according to run mode;
- artifact write failure: fail the question;
- unaudited answer: never considered deliverable in enforced mode.

## Consequences

### Positive

- Distinguishes multiple citation failure classes.
- Prevents syntactically valid but unsupported answers from passing.
- Bounds correction cost.
- Makes abstention a first-class behavior.
- Supports before/after measurement.

### Negative

- Adds latency and provider cost.
- Claim segmentation and entailment remain imperfect.
- A strict verifier can increase false abstention.
- Calibration against humans is required.

## Alternatives considered

- **Existence check only:** rejected because existence does not prove support.
- **Generator self-critique only:** rejected due to correlated errors.
- **Recursive rewrite until accepted:** rejected because convergence is not guaranteed.
- **Use judge as runtime control:** rejected because scoring and delivery policy have distinct contracts.

## Acceptance criteria

- [ ] Every generated answer has a structured audit result.
- [ ] Syntax, existence, and context presence are deterministic.
- [ ] Temporal validity can be unknown.
- [ ] Unsupported material cannot pass unchanged.
- [ ] At most one rewrite occurs.
- [ ] Failed second audit causes limitation or abstention.
- [ ] Original and final answers are retained.
- [ ] Audit calls are cached and identified.
- [ ] Tests cover invented IDs, missing-context IDs, uncited claims, partial support, rewrite failure, and abstention.
- [ ] Audit impact is published for the full split.

## Rollout

1. Add claim and audit domain models.
2. Implement deterministic checks.
3. Add semantic verifier port.
4. Implement bounded rewrite.
5. Add abstention policy.
6. Integrate with runner.
7. Calibrate decisions against humans.
8. Publish original-versus-audited results.
