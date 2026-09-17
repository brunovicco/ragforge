# RAG threat model

## Scope

This threat model covers RAGForge's current benchmark runtime: ingestion of
versioned public regulatory documents, strategy-specific indexing/retrieval,
hosted embedding/generation/judging calls, local benchmark infrastructure, and
publication of benchmark evidence.

RAGForge is not an end-user production RAG service. It has no user accounts,
no tool-calling agents, no write actions against business systems, and no
private enterprise corpus in the current scope. Those absences materially
reduce the impact of several common GenAI threats, but they do not remove
RAG-specific risks.

The model is organized around practical RAG attack surfaces also covered by
industry guidance such as the OWASP GenAI/LLM risk taxonomy: prompt injection,
poisoning, sensitive-information disclosure, excessive consumption, supply
chain risk, and weaknesses in vector/embedding-based retrieval.

## Assets to protect

- integrity of the versioned corpus and RegRAG-BR judgments;
- integrity of benchmark metrics and published recommendations;
- provenance from source document to chunk, retrieval result, answer, and run;
- provider credentials and local infrastructure credentials;
- confidentiality of prompts/completions if the project scope later includes
  non-public data;
- availability/cost of hosted model calls;
- separation between synthetic retrieval enrichment and authoritative source
  evidence;
- integrity of CI, dependencies, containers, and GitHub publication controls.

## Trust boundaries

```text
official source material
        |
        | ingestion + snapshot/integrity boundary
        v
versioned corpus
        |
        v
chunking / enrichment / indexing
        |
        | retrieval boundary
        v
retrieved evidence -----------------------+
        |                                  |
        | prompt boundary                  |
        v                                  |
hosted generation model                   |
        |                                  |
        +--> answer + citations            |
        |                                  |
        +--> independent judge/audit ------+
        |
        v
run evidence + published aggregates
```

External trust boundaries also exist around Gemini/OpenAI APIs, optional
Langfuse tracing, Python dependencies/model packages, container images,
Postgres/pgvector, OpenSearch, and GitHub Actions.

## Threats and controls

### 1. Indirect prompt injection from retrieved documents

**Scenario.** A retrieved document contains text such as "ignore previous
instructions", role changes, fake policies, or instructions intended to alter
the model's behavior.

**Current controls.**

- The answer-generation system prompt explicitly classifies retrieved evidence
  as **untrusted data** and forbids following instructions found inside it.
- Each context block is marked as untrusted retrieved evidence.
- The answer generator receives authoritative `source_text`; synthetic
  `retrieval_text` enrichment is not passed as answer evidence.
- Generated answers are expected to cite structural IDs and can be subjected to
  deterministic citation checks and the optional semantic-support audit.

**Residual risk.** Prompt-only defenses are not proof of resistance. A model
can still follow adversarial retrieved content. A live red-team evaluation with
malicious documents and measured attack success rate remains backlog work.

### 2. Corpus or golden-set poisoning

**Scenario.** A source document, parsed snapshot, question, or relevance
judgment is modified so the benchmark silently rewards a desired strategy.

**Current controls.**

- raw source files are content-addressed with SHA-256 snapshot hashes;
- corpus/golden-set changes are versioned in Git;
- published runs record exact run identity and evidence;
- publication is explicit through `published-runs.json` rather than directory
  auto-discovery;
- benchmark evidence includes manifests/checksums and a tamper-evident event
  chain.

**Residual risk.** The maintainer currently authors both the dataset and the
system under evaluation. Test-set adaptation and curator bias are methodological
risks even when files are cryptographically intact. A sealed holdout or
independent second reviewer would reduce this risk.

### 3. Synthetic enrichment contaminates answer evidence

**Scenario.** Contextual Retrieval, SAC, RAPTOR, or another synthetic artifact
introduces claims not present in the authoritative source and those claims are
later presented as source-backed facts.

**Current controls.**

- sparse retrieval stores both `retrieval_text` and `source_text`;
- answer generation is grounded on authoritative `source_text` for ordinary
  chunk-based strategies;
- benchmark documentation calls out strategies such as RAPTOR where generated
  summary nodes create a different evidence trade-off;
- citation/semantic-support auditing can detect unsupported answer claims.

**Residual risk.** Strategies whose retrieved nodes are themselves synthetic
must remain explicitly identified in reports and should not be compared as if
all evidence paths had identical provenance.

### 4. Vector/embedding retrieval manipulation

**Scenario.** Crafted text changes embedding/BM25 behavior, creates retrieval
collisions, or causes irrelevant/attacker-controlled passages to dominate
ranking.

**Current controls.**

- retrieval is benchmarked against structural relevance judgments rather than
  accepted as trustworthy by default;
- dense, sparse, hybrid, reranked, hierarchical, contextual, and graph-based
  approaches are compared under the same evaluation contract;
- exact index fingerprints and chunk identities are used to prevent silent
  reuse of stale indexes;
- hybrid ranking uses rank fusion rather than treating incompatible score
  scales as directly comparable.

**Residual risk.** There is no dedicated adversarial retrieval benchmark yet.
Future work should include keyword stuffing, embedding-collision-like passages,
near-duplicate chunks, misleading headings, and poisoned synthetic summaries.

### 5. Citation laundering / unsupported claims

**Scenario.** A generated answer includes a real-looking structural citation
that does not support the adjacent claim.

**Current controls.**

- deterministic Citation Accuracy is measured against the golden judgments;
- citation parsing is separate from model judging;
- an optional semantic-support verifier can inspect whether cited evidence
  actually supports generated claims and allows at most a bounded rewrite;
- published runs preserve per-question evidence for audit.

**Residual risk.** The semantic audit is disabled in the published v0.1 run for
cost reasons, so citation presence must not be interpreted as semantic proof.

### 6. Judge manipulation and measurement error

**Scenario.** Generated text exploits the evaluator, or the LLM judge is simply
poorly calibrated for Brazilian legal Portuguese.

**Current controls.**

- canonical generation and judging use different providers/models;
- judge identity is pinned in experiment configuration;
- deterministic retrieval/citation metrics provide judge-free anchors;
- ADR-0007 requires human calibration before judge metrics are considered
  validated.

**Residual risk.** Human calibration is still pending. Judge-derived metrics
remain qualified until the acceptance gate is met.

### 7. Sensitive-information disclosure

**Scenario.** Corpus text, prompts, completions, provider secrets, or trace
content are disclosed through logs, observability, API responses, or published
artifacts.

**Current controls.**

- the current corpus is public official material;
- API keys are environment-injected and not part of benchmark artifacts;
- Langfuse tracing is opt-in and metadata-only by default;
- tracing metadata is allowlisted and bounded;
- the public API/dashboard expose aggregate cataloged benchmark data rather
  than prompts or per-question provider responses.

**Residual risk.** These assumptions must be re-evaluated before any non-public
corpus is introduced. `docs/PRIVACY.md` is the source of truth for that scope
change.

### 8. Cost/availability abuse and provider instability

**Scenario.** Retries, high concurrency, unexpectedly large context, or provider
errors create runaway cost or unstable benchmark results.

**Current controls.**

- provider concurrency is bounded;
- retry behavior is centralized;
- benchmark sample size is configurable and the published v0.1 run is
  cost-controlled;
- model identities and token usage are captured in run evidence;
- live provider tests are not part of the default CI path.

**Residual risk.** Authoritative provider pricing is not yet populated in the
v0.1 benchmark configuration, so cost budgets are not currently enforced as a
benchmark gate.

### 9. Local infrastructure exposure

**Scenario.** Development Postgres/OpenSearch services become reachable from
outside the developer machine while using development credentials or disabled
search security.

**Current controls.**

- Docker Compose publishes both services only on `127.0.0.1`;
- the compose file is labeled development-only;
- OpenSearch security being disabled is explicitly treated as unacceptable for
  non-local deployment.

**Residual risk.** A future hosted deployment requires a separate hardened
configuration with authentication, TLS, network policy, secret management, and
backup/retention decisions.

### 10. Software/model supply chain compromise

**Scenario.** A dependency, GitHub Action, container image, downloaded model, or
transitive package is compromised or introduces a known vulnerability.

**Current controls.**

- Python dependencies are locked and checked with `uv lock --check`;
- `pip-audit` and Bandit run in the project quality gate;
- the runtime image and `uv` image are pinned by digest;
- GitHub Actions used by the quality workflow are pinned to immutable commit
  SHAs;
- dependency advisories with no upstream fix are explicitly enumerated rather
  than silently suppressed.

**Residual risk.** Local sentence-transformer/model downloads and transitive
model artifacts are not yet represented by a formal model-SBOM or checksum
allowlist.

## Security verification strategy

The repository separates deterministic security invariants from live model
behavior:

- **unit tests** verify that answer generation uses authoritative source text
  rather than retrieval enrichment and that the retrieved-evidence trust
  boundary is present;
- **integration CI** verifies real pgvector and OpenSearch behavior without
  hosted-provider credentials;
- **quality CI** runs static analysis, dependency audit, typing, unit tests,
  architecture checks, and repository governance checks;
- **future adversarial RAG evaluation** should measure attack success rate using
  malicious retrieved documents rather than claiming prompt wording alone
  solves indirect injection.

## Out-of-scope changes that require a new threat assessment

Re-run this threat model before introducing any of the following:

- private/customer documents or personal data;
- document upload by untrusted users;
- multi-tenant indexes;
- tool calling or agent actions;
- write access to external systems;
- authenticated user-facing APIs;
- hosted Postgres/OpenSearch deployments;
- observability with prompt/completion capture;
- automatic web ingestion or untrusted crawled sources.
