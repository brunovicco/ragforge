# Development guide

## Setup

```bash
uv sync --frozen --all-groups
```

## Run checks

```bash
uv run python scripts/quality_gate.py
```

The default pytest configuration excludes tests marked `integration`; the
quality gate is intended to stay deterministic and credential-free.

To exercise the real local retrieval stores:

```bash
docker compose --profile core --profile search up -d --wait
docker compose exec -T postgres \
  psql -U ragforge -d ragforge -c 'CREATE EXTENSION IF NOT EXISTS vector;'
uv run pytest -o addopts="--strict-config --strict-markers" \
  -m integration \
  tests/integration/test_dense_chunk_store.py \
  tests/integration/test_sparse_chunk_store.py
```

Those two infrastructure integration tests are also executed by GitHub Actions.
Provider-backed integration tests remain opt-in because they require
credentials, can incur cost, and are not deterministic enough for the default
PR gate.

## Container

```bash
docker build -t ragforge .
docker run --rm -p 8000:8000 ragforge
```

`Dockerfile` is a multi-stage, uv-based build. The builder installs the locked
dependencies and project; the runtime stage contains the resulting environment
and source only, runs as a non-root user, and starts the published-results
FastAPI app with Uvicorn on port 8000.

## Local configuration

Copy `.env.example` only when the application supports local dotenv loading.
Never commit `.env` or real credentials.

`gemini-embedding-001` is the canonical embedding configuration for the
published v0.1 benchmark (ADR-0005, `configs/experiments/benchmark-v01.yaml`).
`GeminiContextualizer` generates per-chunk context for Contextual Retrieval;
`GeminiSummarizer` generates summaries used by hierarchical strategies; the
LightRAG GraphRAG adapter uses Gemini for embedding and graph extraction.
Therefore `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is required for live runs that
exercise those adapters. The canonical independent answer judge additionally
requires `OPENAI_API_KEY`.

Live runs can write provider calls to the per-run LLM cache for traceability.
That is distinct from the planned deterministic `make bench` replay engine:
the cache artifact exists, but zero-provider replay and its CI gate are not yet
implemented (ADR-0004/ADR-0020).

## Local infrastructure (retrieval indexing)

`docker-compose.yml` provides the two stores the retrieval strategies index into
(ADR-0005): Postgres + pgvector for dense retrieval, and OpenSearch for
BM25/hybrid. Both are opt-in via Compose profiles so a plain `docker compose up`
starts nothing.

```bash
docker compose --profile core --profile search up -d --wait
docker compose ps
docker compose down
docker compose down -v   # also drop development data volumes
```

Both exposed ports are bound to `127.0.0.1` only. Credentials are development
values baked into `docker-compose.yml`, not secrets. OpenSearch runs with its
security plugin disabled for this local-only profile. Do not reuse the compose
file as a hosted or production deployment configuration; a non-local deployment
must add authentication, TLS, network restrictions, and secret-managed
credentials.

Bring up only what the current task needs: `--profile core` alone for
pgvector/dense work, or `--profile search` alone for OpenSearch/BM25 work.

## Benchmark execution

```bash
GEMINI_API_KEY=... OPENAI_API_KEY=... make bench-live
make bench-live-local
```

The live commands persist cache/index material under `.ragforge/cache/` and
write run artifacts under `experiments/<run-id>/` and `artifacts/runs/<run-id>/`.
Use the run manifest and verification tooling when publishing results; do not
manually copy aggregate numbers into documentation without the corresponding
run evidence.

## Security and privacy

Before changing corpus scope, hosted providers, or tracing behavior, review:

- `docs/THREAT-MODEL.md`
- `docs/PRIVACY.md`
- `docs/LLM_OBSERVABILITY.md`

The current assessment assumes public official source material. Adding private
or user-supplied documents is a scope change, not a routine configuration
change.

## Claude Code

- Run `/memory` to confirm loaded instructions.
- Run `/hooks` to inspect configured hooks.
- Run `claude doctor` from the shell for a read-only installation and
  configuration check.
- Use `/plan-change` before complex work.
- Use `/quality-gate` before completion.
- Use `/prepare-pr` to produce a reviewable PR description.

### Isolating riskier changes in a worktree

For a larger or harder-to-reverse change, add `isolation: worktree` to
`.claude/agents/python-implementer.md`'s frontmatter before delegating the
change. The subagent then works from a temporary git worktree branched off the
default branch instead of editing the working tree directly. Remove the
setting again when the isolated change is complete rather than leaving it as a
blanket default for routine work.
