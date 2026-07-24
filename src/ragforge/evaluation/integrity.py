"""Preflight integrity gate (ADR-0012): fail closed before indexing or evaluation.

Three independent checks, each collecting every problem it finds rather than
stopping at the first one, so a run that fails closed reports the complete
list in one shot instead of a slow one-problem-per-rerun loop:

- ``verify_source_integrity`` - every enabled manifest document exists and
  matches its pinned hash. Cheap; runs before extraction.
- ``verify_split_integrity`` - the split and the golden set agree on exactly
  which questions are selected, with no duplicate or orphaned ID. Cheap;
  runs before extraction.
- ``verify_structural_references`` - every judged structural ID actually
  resolves against the real indexed corpus. Requires the parsed/chunked
  documents, so it runs after extraction but before indexing.
"""

from collections import Counter
from pathlib import Path

from ragforge.domain.models import Chunk, Judgment
from ragforge.evaluation.manifest import CorpusManifest
from ragforge.evaluation.split import Split
from ragforge.ingestion.snapshot import snapshot_hash


class IntegrityError(Exception):
    """Raised when a preflight integrity check fails. The run must not continue."""


def _fail_if_any(problems: list[str]) -> None:
    if problems:
        raise IntegrityError("\n".join(f"- {problem}" for problem in problems))


def verify_source_integrity(manifest: CorpusManifest, root: Path) -> None:
    """Fail if any enabled document's source file is missing or hash-mismatched.

    Raises:
        IntegrityError: If one or more enabled documents fail this check.
    """
    problems: list[str] = []
    for doc in manifest.enabled_documents:
        source_path = root / doc.source_path
        if not source_path.exists():
            problems.append(f"{doc.norm_id}: source file not found: {source_path}")
            continue
        actual_hash = snapshot_hash(source_path)
        if actual_hash != doc.source_sha256:
            problems.append(
                f"{doc.norm_id}: source hash mismatch for {source_path} "
                f"(manifest expects {doc.source_sha256}, actual {actual_hash})"
            )
    _fail_if_any(problems)


def verify_split_integrity(split: Split, judgments: list[Judgment]) -> None:
    """Fail unless the split and the golden set select exactly the same questions once each.

    Raises:
        IntegrityError: If any ID is duplicated across partitions, any split
            ID is unknown to the golden set, or any golden-set question is
            not covered by the split.
    """
    problems: list[str] = []

    duplicates = [id_ for id_, count in Counter(split.all_ids).items() if count > 1]
    if duplicates:
        problems.append(f"question IDs duplicated across split partitions: {sorted(duplicates)}")

    judgment_ids = {judgment.question_id for judgment in judgments}
    split_ids = set(split.all_ids)

    unknown_in_split = split_ids - judgment_ids
    if unknown_in_split:
        problems.append(f"split references unknown question IDs: {sorted(unknown_in_split)}")

    missing_from_split = judgment_ids - split_ids
    if missing_from_split:
        problems.append(
            f"golden-set questions are not selected by any split partition: "
            f"{sorted(missing_from_split)}"
        )

    _fail_if_any(problems)


def verify_structural_references(
    judgments: list[Judgment], documents: dict[str, tuple[str, list[Chunk]]]
) -> None:
    """Fail unless every judged structural ID resolves against the real indexed corpus.

    ``documents`` is the ``{norm_id: (full_text, chunks)}`` mapping of the
    documents actually extracted and chunked this run - a reference to a
    disabled or unknown norm fails here because it is simply absent from
    this mapping.

    Raises:
        IntegrityError: If one or more judged structural IDs do not resolve.
    """
    real_ids_by_norm = {
        norm_id: {ref for chunk in chunks for ref in chunk.structural_ids}
        for norm_id, (_, chunks) in documents.items()
    }

    problems: list[str] = []
    for judgment in judgments:
        for judged in judgment.relevant_refs:
            norm_id = judged.ref.norm
            canonical = judged.ref.canonical
            if norm_id not in real_ids_by_norm:
                problems.append(
                    f"{judgment.question_id}: {canonical} references a document not indexed "
                    f"(disabled or unknown norm: {norm_id})"
                )
            elif canonical not in real_ids_by_norm[norm_id]:
                problems.append(
                    f"{judgment.question_id}: {canonical} does not resolve to any "
                    "structural ID produced by the indexed corpus"
                )
    _fail_if_any(problems)
