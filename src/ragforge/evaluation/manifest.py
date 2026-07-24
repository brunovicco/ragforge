"""Loads the canonical corpus manifest (ADR-0012) from datasets/regrag-br/.

The manifest is the only source for benchmark document discovery - runtime
constants containing document paths are removed from the runner in favor of
this file.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One manifest entry: a norm's immutable source locator and extraction config."""

    norm_id: str
    source_path: str
    extractor: str
    expected_article_count: int | None
    source_sha256: str | None
    enabled: bool


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """The canonical corpus manifest (ADR-0012)."""

    schema_version: int
    corpus_id: str
    corpus_version: str
    documents: tuple[CorpusDocument, ...]

    @property
    def enabled_documents(self) -> tuple[CorpusDocument, ...]:
        """Return only the documents enabled for indexing."""
        return tuple(doc for doc in self.documents if doc.enabled)

    @property
    def content_hash(self) -> str:
        """Deterministic hash of every enabled document's identity (ADR-0013).

        Sorted by norm_id so the result depends only on which documents are
        enabled and their pinned source hashes, never on YAML entry order.
        """
        pairs = sorted(f"{doc.norm_id}:{doc.source_sha256}" for doc in self.enabled_documents)
        return hashlib.sha256("|".join(pairs).encode("utf-8")).hexdigest()


def load_corpus_manifest(path: Path) -> CorpusManifest:
    """Parse the corpus manifest YAML file into a CorpusManifest.

    Raises:
        ValueError: If an enabled document is missing its expected article
            count or source hash - both are mandatory for anything that will
            actually be indexed and integrity-checked.
        KeyError: If a required field is missing from an entry.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    documents = []
    for entry in data["documents"]:
        doc = CorpusDocument(
            norm_id=entry["norm_id"],
            source_path=entry["source_path"],
            extractor=entry["extractor"],
            expected_article_count=entry.get("expected_article_count"),
            source_sha256=entry.get("source_sha256"),
            enabled=entry["enabled"],
        )
        if doc.enabled and (doc.expected_article_count is None or doc.source_sha256 is None):
            msg = (
                f"{doc.norm_id}: enabled documents require expected_article_count and source_sha256"
            )
            raise ValueError(msg)
        documents.append(doc)
    return CorpusManifest(
        schema_version=data["schema_version"],
        corpus_id=data["corpus_id"],
        corpus_version=data["corpus_version"],
        documents=tuple(documents),
    )
