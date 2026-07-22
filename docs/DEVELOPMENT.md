# Development guide

## Setup

```bash
uv sync --frozen
```

## Run checks

```bash
uv run python scripts/quality_gate.py
```

## Container

```bash
docker build -t ragforge .
docker run --rm ragforge
```

`Dockerfile` is a multi-stage, uv-based build: a `builder` stage installs the locked
dependencies and builds the package, then only the resulting virtualenv and source are copied
into a slim, non-root runtime image. The shipped `CMD` is a placeholder - this harness is
framework-agnostic and does not assume an ASGI app, CLI, or worker loop. Replace it with the
project's real entrypoint. Adjust `.dockerignore` if new top-level files or directories need to
be excluded from the build context.

## Local configuration

Copy `.env.example` only when the application supports local dotenv loading. Never commit `.env` or real credentials.

`gemini-embedding-001` is the frozen embedding model (ADR-0005, `configs/experiments/benchmark-v01.yaml`); `GeminiContextualizer` generates per-chunk context for Contextual Retrieval; `GeminiSummarizer` generates per-group summaries for the RAPTOR tree; the LightRAG GraphRAG adapter (`retrieval.graph.lightrag_gemini`, ADR-0010) uses Gemini for both embeddings and entity/relation extraction - so `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is required for `make bench-live` and any other live run through any of these adapters, not for `make bench`, which replays the versioned LLM cache (ADR-0004).

## Local infrastructure (retrieval indexing)

`docker-compose.yml` provides the two stores the retrieval strategies index into (ADR-0005):
Postgres + pgvector for dense retrieval, and OpenSearch for BM25/hybrid. Both are opt-in via
Compose profiles so a plain `docker compose up` starts nothing.

```bash
docker compose --profile core --profile search up -d   # postgres+pgvector, opensearch
docker compose ps                                       # wait for both healthy
docker compose down                                     # stop; add -v to also drop data volumes
```

Credentials are dev-only defaults baked into `docker-compose.yml` (not secrets); the matching
`DATABASE_URL`/`OPENSEARCH_URL` are in `.env.example`. OpenSearch runs with its security plugin
disabled (`plugins.security.disabled: "true"`) for local dev only - plain HTTP, no auth. Never
run it this way outside a local machine.

Bring up only what the current task needs: `--profile core` alone for pgvector/dense work,
`--profile search` alone for OpenSearch/BM25 work.

## Claude Code

- Run `/memory` to confirm loaded instructions.
- Run `/hooks` to inspect configured hooks.
- Run `claude doctor` from the shell for a read-only installation and configuration check. Reserve
  interactive `/doctor` for cases that may need guided repair, and review its requested commands.
- Use `/plan-change` before complex work.
- Use `/quality-gate` before completion.
- Use `/prepare-pr` to produce a reviewable PR description.

### Isolating riskier changes in a worktree

For a larger or harder-to-reverse change, add `isolation: worktree` to
`.claude/agents/python-implementer.md`'s frontmatter before delegating the change. The subagent
then works from a temporary git worktree branched off the default branch instead of editing the
working tree directly; the worktree is cleaned up automatically if it makes no changes. This is
not the harness default because it changes where edits land - add it deliberately for a specific
change you want to inspect before merging into your working tree, then remove it again, rather
than leaving it on for routine, well-scoped work.
