# RegRAG-BR Datasheet

Format: [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et
al.), abridged to the questions that apply. Dataset version: **0.2**. This
datasheet is referenced by the repository README and by ADR-0002, ADR-0007 and
ADR-0011.

## Motivation

**Why was the dataset created?** To benchmark retrieval-augmented generation
(RAG) strategies over Brazilian financial-regulatory text, where public golden
sets are scarce and English general-domain benchmarks do not transfer.
RegRAG-BR provides the ground truth for the RAGForge benchmark
(github.com/brunovicco/ragforge).

**Who created it and who funded it?** Hand-curated by the repository author.
No external funding.

## Composition

**What do the instances represent?** Each of the **230 instances** is a
Brazilian-Portuguese question over the corpus, with:

- `question_id`, `text`
- `query_class` — one of 7 classes
- `relevant_refs` — graded, article-level structural IDs
  (`{norm}::{art}::{fragment}`, per ADR-0002/ADR-0006), empty for
  unanswerable questions
- `reference_answer`

**Class distribution (v0.2):**

| Class | Count |
| --- | ---: |
| exact_factual | 127 |
| numeric_tabular | 39 |
| semantic | 17 |
| multi_hop | 13 |
| section_comparative | 13 |
| unanswerable | 11 |
| global | 10 |

The distribution is intentionally realistic rather than balanced; it is
**heavily skewed toward exact_factual (55%)**. Per-class metrics for the small
classes (multi_hop, global, section_comparative) carry wide confidence
intervals and must not be read as categorical rankings — see "Uses".

**Corpus coverage.** Judgments are verified against the real parsed text of 5
documents: LC-105/2001, RES-CMN-4893/2021, RES-CMN-5274/2025, LEI-13709/2018
(LGPD), ICVM-607/2019. A sixth corpus document (LEI-6385/1976) is present in
the corpus snapshot but **not yet curated** — no question cites it.

**Known exclusions.** 75 structural IDs that collide across amendment history
or appended annexes (different real text under the same canonical ID) are
excluded from `relevant_refs` by a curation-time gate rather than fixed in the
parser — 1 ID in RES-CMN-5274/2025, 65 in LEI-13709/2018, 9 in ICVM-607/2019.
Rationale and full list: ADR-0011.

**Splits.** Deterministic, stratified by query class
(`split.json`, schema_version 1): `validation` = 36 (router
development/few-shot source), `test` = 194 (frozen; all published metrics),
`train` = empty until a learned router exists (ADR-0003).

## Collection and labeling process

Questions and judgments were authored manually during corpus curation. Every
candidate structural ID was validated against real `chunk_norm`/`parse_norm`
output before citation (ADR-0011). Relevance is graded (relevant / partially
relevant); grades map to nDCG gains 1.0 / 0.5. No crowd workers, no personal
data subjects.

**Validation status of automated judging.** LLM-as-judge metrics computed over
this dataset (Faithfulness, Answer Relevancy) are **not yet calibrated against
human evaluation**; the ADR-0007 weighted-kappa exercise is pending. Judge
scores published before that exercise are qualified accordingly.

## Uses

**Intended.** Evaluating retrieval strategies via structural-coverage
Recall/Precision/nDCG/MRR (ADR-0002 semantics — coverage, not exact-chunk
match, which is slightly more permissive), deterministic Citation Accuracy,
and answer-quality evaluation including explicit abstention on the 11
unanswerable questions.

**Cautions.**

- Coverage-based projection favors results that span many structural IDs
  (e.g. hierarchical summary nodes); compare Precision@k across strategies
  with that bias in mind.
- Do not draw per-class conclusions for classes with n ≤ 13 without
  bootstrap confidence intervals.
- Reference answers reflect the norms **as captured in the versioned corpus
  snapshot** at curation time; they are not legal advice and may not reflect
  later amendments.

## Distribution and licensing

Published in this repository under **CC BY 4.0**
(`datasets/regrag-br/LICENSE`). The underlying norms are official acts, not
copyright-protected (art. 8, I, Law 9,610/1998).

## Maintenance

Maintained in-repo; the golden set and split are protected by fail-closed
tooling hooks (ADR-0009) and are immutable per published benchmark run —
changes create a new dataset version and a new run identity.
