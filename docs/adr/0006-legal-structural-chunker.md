# ADR-0006: Structural chunking by normative hierarchy (Art./§/item)

- Status: Accepted
- Date: 2026-07-21

## Context

CMN/BCB/CVM norms follow a predictable formal structure: Chapter → Section → Article → Paragraph (§) → Item (inciso) → Sub-item (alínea). Docling extracts generic layout (headings, tables) but does not recognize this legal hierarchy. Fixed-size chunking would cut articles in half and make two central requirements impossible: structural-unit judgments (ADR-0002) and citations traceable to the article (governance).

## Decision

Implement a **legal structural parser** as its own stage in `src/ragforge/chunking/`, applied on top of Docling output (PyMuPDF fallback):

1. Regex/rule-based segmentation over canonical markers (`Art. N`, `§ N`, incisos `I–...`, alíneas `a)`), producing a structural tree per norm.
2. Every node gets a **canonical, stable structural ID**: `{norm}::{art}::{par|inc|ali}` (e.g. `RES-CMN-4893/2021::art-3::par-1`). This ID is the key for judgments (ADR-0002) and citations (governance).
3. Chunks derive from the tree: target unit = article; long articles subdivide into §/incisos carrying parent metadata (which also gives the parent-child strategy the norm's real hierarchy instead of arbitrary windows).
4. Every chunk carries `structural_ids: list[str]` - the chunk → article mapping required by evaluation and the API.
5. Validation: an automated test compares, per ingested norm, the extracted article count against a curated expected count; failure blocks indexing.

Budget: ~1 day within the ingestion window (D3–4). Domain-aware chunking vs naive splitting is precisely the kind of decision this repository intends to demonstrate.

## Consequences

- Enables ADR-0002 and citation traceability with a single artifact.
- Parent-child becomes semantically correct (real section/article as parent).
- Tables (limits, risk weights in annexes) are preserved as nodes by Docling and linked to the referencing article - input for the numeric/tabular query class.
- Legal regex has a long tail (older norms, inconsistent formatting); the per-norm validation test contains the risk, and problematic norms can be swapped during corpus selection (~15–25 norms gives slack).
- Couples the pipeline to the Brazilian domain - acceptable: it is the project's stated purpose.

## Alternatives considered

- **Fixed-size chunking + overlap** - rejected: breaks articles, kills structural judgments and precise citation.
- **LLM-based structure extraction** - rejected as the main path: cost and non-determinism for a task regex solves; kept as a targeted fallback.
- **Relying on Docling's hierarchy alone** - rejected: generic headings, not Art./§/inciso semantics.
