#!/usr/bin/env python3
"""Build a blind human-labeling worksheet for judge calibration.

Usage:
    uv run python scripts/build_calibration_worksheet.py 20260726T185553Z
    uv run python scripts/build_calibration_worksheet.py 20260726T185553Z --size 40

The run must contain ``judge_contexts`` persisted by the answer harness.
Citation text cannot reconstruct the evidence that RAGAS Faithfulness saw,
so runs created before evidence schema v1 fail closed and must be rerun.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragforge.chunking.legal_parser import parse_norm  # noqa: E402
from ragforge.evaluation.artifact_writer import write_atomic  # noqa: E402
from ragforge.evaluation.calibration_workflow import (  # noqa: E402
    build_human_labels,
    build_sealed_calibration,
    calibration_feature_coverage,
    load_human_labels,
    load_sealed_calibration,
    merge_existing_human_labels,
    render_calibration_worksheet,
    select_calibration_records,
)
from ragforge.evaluation.judgments import load_judgments  # noqa: E402
from ragforge.evaluation.manifest import load_corpus_manifest  # noqa: E402
from ragforge.evaluation.records import read_records_jsonl  # noqa: E402
from ragforge.ingestion.html_extractor import HtmlTextExtractor  # noqa: E402
from ragforge.ingestion.pymupdf_extractor import PyMuPdfExtractor  # noqa: E402

OUT_DIR = ROOT / "datasets" / "regrag-br" / "calibration"


class _TextExtractor(Protocol):
    def extract(self, path: Path) -> str:
        """Extract normalized text from ``path``."""


_EXTRACTOR_FACTORIES: dict[str, Callable[[], _TextExtractor]] = {
    "html": HtmlTextExtractor,
    "pymupdf": PyMuPdfExtractor,
}


def load_corpus_text() -> dict[str, dict[str, str]]:
    """Resolve canonical structural IDs for citation-audit display only."""
    manifest = load_corpus_manifest(ROOT / "datasets/regrag-br/corpus_manifest.yaml")
    resolved: dict[str, dict[str, str]] = {}
    for document in manifest.enabled_documents:
        try:
            extractor_factory = _EXTRACTOR_FACTORIES[document.extractor]
            text = extractor_factory().extract(ROOT / document.source_path)
            tree = parse_norm(document.norm_id, text)
        except Exception as exc:
            print(
                f"  ! {document.norm_id}: {type(exc).__name__} - citation audit will show IDs only"
            )
            resolved[document.norm_id] = {}
            continue

        by_id: dict[str, str] = {}
        for article in tree.articles:
            by_id[tree.structural_id(article)] = article.full_text
            stack = [(child, (child,)) for child in article.children]
            while stack:
                node, path = stack.pop()
                by_id[tree.structural_id(article, path)] = node.full_text
                stack.extend((child, (*path, child)) for child in node.children)
        resolved[document.norm_id] = by_id
        print(f"  {document.norm_id}: {len(by_id)} structural IDs resolved")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--size", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    """Select records and atomically publish a blind calibration workspace."""
    args = _parse_args()
    records_path = ROOT / "experiments" / args.run_id / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"No such run: {records_path}")

    records = read_records_jsonl(records_path)
    judgments = {
        judgment.question_id: judgment
        for judgment in load_judgments(ROOT / "datasets/regrag-br/judgments.json")
    }
    try:
        sample = select_calibration_records(records, judgments, args.size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("Resolving citation-audit text (judge contexts already come from the run)...")
    corpus = load_corpus_text()
    worksheet = render_calibration_worksheet(sample, judgments, corpus, args.run_id)
    expected_labels = build_human_labels(sample)
    sealed = build_sealed_calibration(sample, args.run_id)

    labels_path = OUT_DIR / "labels.json"
    sealed_path = OUT_DIR / ".judge-sealed.json"
    if labels_path.exists() != sealed_path.exists():
        raise SystemExit(
            "calibration workspace is incomplete; labels and sealed scores must coexist"
        )
    labels = expected_labels
    if labels_path.exists():
        try:
            existing_labels = load_human_labels(labels_path)
            existing_sealed = load_sealed_calibration(sealed_path)
            if existing_sealed != sealed:
                raise ValueError(
                    "existing sealed scores belong to a different run or sample; archive the "
                    "calibration directory before rebuilding"
                )
            labels = merge_existing_human_labels(expected_labels, existing_labels)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    write_atomic(OUT_DIR / "worksheet.md", worksheet)
    write_atomic(
        labels_path,
        json.dumps(
            [label.model_dump() for label in labels],
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
    )
    write_atomic(
        sealed_path,
        json.dumps(sealed.model_dump(), indent=2, ensure_ascii=False, allow_nan=False),
    )

    coverage = calibration_feature_coverage(sample, judgments)
    print(f"Selected {len(sample)} samples; validated {len(coverage)} coverage features.")
    print(f"Worksheet: {OUT_DIR / 'worksheet.md'}")
    print(f"Fill in:   {labels_path} ({len(labels)} scores; existing scores preserved)")
    print(f"Sealed:    {sealed_path} (do not open until labeling is complete)")


if __name__ == "__main__":
    main()
