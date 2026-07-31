# Privacy and data handling

Scope: RAGForge processes **public official Brazilian legal and regulatory
texts** and synthetic benchmark artifacts. It is not an end-user service, has
no user accounts, and does not process personal data as a product function.
This document records that assessment and the controls that keep it true.

## Data inventory

| Data category | Source | Purpose | Legal/contractual basis | Destination | Retention | Deletion method |
| --- | --- | --- | --- | --- | --- | --- |
| Corpus text (CMN/BCB, CVM norms, federal laws) | Official public sources, versioned snapshot under `datasets/corpus/` | Indexing, retrieval, benchmark evaluation | Official acts; not copyright-protected (art. 8, I, Law 9,610/1998); no personal data intended | Local stores (Postgres+pgvector, OpenSearch, LightRAG local storage); hosted LLM/embedding APIs (Google Gemini, OpenAI) during live runs | Indefinite (versioned snapshot is the benchmark's ground truth) | Git history rewrite + store re-index (not expected) |
| RegRAG-BR golden set (questions, judgments, reference answers) | Hand-authored by the maintainer | Ground truth for evaluation | Original work, published CC BY 4.0 | Same as corpus; question text is sent to hosted LLM APIs in live runs | Indefinite (published dataset) | New dataset version |
| LLM request/response payloads | Live benchmark runs | Generation, enrichment, judging; write-through call cache | Provider terms (Google, OpenAI); content is corpus/golden-set text only | `experiments/<run-id>/llm-cache/`, provider-side per their retention policies | Versioned with the run | Delete run directory + new run identity |
| Run evidence (manifests, events, per-question records, exact local judge contexts) | Benchmark runner (ADR-0017) | Reproducibility, human judge calibration, and audit | Legitimate interest in scientific integrity; contains no personal data in the current public corpus | `artifacts/runs/`, `experiments/` (public repo) | Indefinite (published evidence) | Remove run from `published-runs.json` + delete directories |
| Observability traces | Optional Langfuse extra | Latency/cost telemetry | Opt-in; **metadata-only** (no prompt/completion content) per `docs/LLM_OBSERVABILITY.md` | Configured Langfuse backend | Backend policy | Backend deletion API |
| Personal data | — | — | — | — | — | None processed by design |

## Controls

- **Data minimization:** only corpus and golden-set text reach external
  providers; observability is metadata-only by default; the published API and
  dashboard expose only immutable aggregates from `published-runs.json` — no
  prompts, no per-question provider responses, no credentials.
- **Access control:** no service-side user data; provider API keys supplied
  via environment variables (`.env`, git-ignored), never committed; published
  read-only endpoints are intentionally unauthenticated because they serve
  only public benchmark aggregates.
- **Encryption in transit:** all provider and infrastructure traffic over
  TLS (HTTPS to Gemini/OpenAI/Langfuse; local Docker services are
  loopback-bound in the development compose profile).
- **Encryption at rest:** local development stores are unencrypted volumes on
  the operator's machine; acceptable because contents are public texts and
  synthetic artifacts. Re-assess before any deployment holding non-public
  data.
- **Masking/tokenization:** not applicable — no personal or confidential data
  in scope.
- **Non-production data strategy:** there is no production; all data is the
  public corpus and its derivatives. Any future ingestion of non-public
  documents requires updating this inventory **first** (fail-closed rule
  below).
- **Logging and tracing restrictions:** structured logs contain identifiers
  and metrics, not document content; the Langfuse backend is opt-in
  (`uv sync --extra tracing`) with content capture disabled — see
  `docs/LLM_OBSERVABILITY.md` before enabling any backend.

## LGPD assessment

RAGForge processes no personal data (Lei 13.709/2018, art. 5º, I) as designed:
the corpus consists of normative acts, and the golden set is authored
synthetic material. Incidental names appearing inside official acts (e.g.
signatories of resolutions) are processed as part of public official texts.
Consequently no RIPD/DPIA is required for the current scope.

**Fail-closed rule:** adding any corpus source that is not a public official
act — or enabling any telemetry backend that captures content — requires
updating the inventory above and re-running this assessment before the data
is ingested. Repository hooks protecting `datasets/` (ADR-0009) are the
enforcement point.

## Provider data-sharing summary

Live runs (`make bench-live`) transmit corpus and question text to Google
(Gemini embeddings/generation) and OpenAI (independent judge, ADR-0018).
Credential-free execution with no third-party transmission of corpus text is
for the embedding stage via the local path (`make bench-live-local`,
ADR-0013); the generation, enrichment and judging stages remain hosted in
both live configurations.
