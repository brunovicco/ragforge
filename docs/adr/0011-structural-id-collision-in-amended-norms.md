# ADR-0011: Structural-ID collisions in amended norms are excluded from golden-set citations, not fixed at the source

- Status: Accepted
- Date: 2026-07-23

## Context

While authoring the 230-question RegRAG-BR golden set (`datasets/regrag-br/judgments.json`) across
5 corpus documents, validation tooling (`chunk_norm`/`parse_norm` run against every candidate
structural ID) found that the structural chunker (ADR-0006) does not deduplicate amendment history
below the article level. When a norm's real source text embeds a superseded wording alongside its
current wording - via footnoted amendment history (e.g. "Redação dada pela Medida Provisória nº
869, de 2018") or an appended annex that restarts numbering - `chunk_norm` attaches the **same**
canonical structural ID to **multiple chunks holding different real text**. The parser only merges
duplicate top-level article numbers; it does not detect or resolve collisions at the
paragraph/inciso level.

Three collision patterns were found across the 5 curated documents:

1. **RES-CMN-5274/2025** (amending resolution): its Art. 1º quotes the full amended text of
   several articles of RES-CMN-4893/2021, each restarting its own §/inciso numbering.
   1 colliding ID (`art-1::par-1`).
2. **LEI-13709/2018 (LGPD)**: 65 colliding single-id chunks, driven by a multi-year amendment
   history (MP 869/2018, Lei 13.853/2019, and later amendments). Entire articles 55-C, 55-D,
   55-J, 58-A, 58-B are unusable at fragment level.
3. **ICVM-607/2019**: 9 colliding IDs, all `art-113::inc-*` - the extracted text appends the
   Instrução's penalty-group annex tables (Grupos I/II/III, each restarting `inc-i..inc-v`)
   directly after art-113's revocation clause with no distinct article marker, so the parser
   merges annex-table rows into art-113's node under reused inciso labels.

A `find_all_collisions.py` script (comparing every single-ID chunk's text pairwise) is the only
way to detect this; nothing in the shipped pipeline flags it today.

## Decision

For this golden set, treat collision-detection as a **curation-time gate**, not a parser fix:
every candidate structural ID is checked against real `chunk_norm` output before being cited, and
any ID known to collide (different real text under the same ID) is excluded from
`relevant_refs`. The 75 colliding IDs found above were avoided entirely when authoring questions.
No golden-set question cites `LEI-13709/2018::art-55-c/55-d/55-j/58-a/58-b`, any of the 65 listed
LGPD paragraph/inciso IDs, `RES-CMN-5274/2025::art-1::par-1`, or any
`ICVM-607/2019::art-113::inc-*`.

This is a deliberate scope cut, not a fix: the underlying ambiguity in `chunk_norm`/`parse_norm`
remains in the shipped pipeline. Any retrieval strategy under evaluation could still retrieve one
of the ambiguous chunks for an unrelated query and have its citation land on the wrong text body
under a correct-looking ID - this ADR does not change that risk, it only keeps the golden set
itself free of it.

## Consequences

- The 230-question golden set has no known citation ambiguity: every judged ID maps to exactly
  one real chunk text.
- The pipeline's real behavior on amended norms is unchanged and remains a genuine data-quality
  gap: `chunk_norm` can silently attach a stable-looking ID to superseded text. This is a latent
  correctness risk for governance/citation features (ADR-0007) on any norm with amendment history
  or appended annexes, beyond the 5 documents curated here.
- Expanding the golden set to additional norms (e.g. `LEI-6385/1976`, currently out of scope,
  or new amended norms) requires re-running `find_all_collisions.py`-equivalent detection before
  citing any new structural ID.
- Follow-up tracked as [issue #19](https://github.com/brunovicco/ragforge/issues/19) rather than
  fixed here, since resolving it requires a design decision on `chunk_norm`'s merge semantics
  (ADR-0006 owner), which is out of scope for a golden-set curation change.

## Alternatives considered

- **Fix `chunk_norm` to deduplicate below article level now** - rejected for this change: requires
  deciding which wording is authoritative (current vs. superseded) per amendment, a legal-parsing
  design change that belongs with ADR-0006, not with golden-set curation.
- **Keep colliding IDs in the golden set, grade them lower** - rejected: a "partially_relevant"
  grade does not communicate *ambiguous provenance*; a strategy citing the wrong text body under a
  technically-matching ID would still score as correct, undermining Citation Accuracy (ADR-0007).
- **Drop the 3 affected documents from the golden set** - rejected: LEI-13709/2018 (LGPD) is the
  single most important document in the corpus; excluding it would gut coverage far more than
  excluding ~75 specific sub-article IDs out of hundreds available.
