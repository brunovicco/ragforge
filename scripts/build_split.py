"""Rebuild the versioned RegRAG-BR validation/test split deterministically."""

import json
from pathlib import Path

from ragforge.evaluation.judgments import load_judgments
from ragforge.evaluation.split_builder import build_stratified_split

ROOT = Path(__file__).resolve().parents[1]
JUDGMENTS_PATH = ROOT / "datasets" / "regrag-br" / "judgments.json"
SPLIT_PATH = ROOT / "datasets" / "regrag-br" / "split.json"
VALIDATION_RATIO = 0.15
SEED = "regrag-br-v1"


def main() -> None:
    """Write the deterministic stratified split to its canonical path."""
    judgments = load_judgments(JUDGMENTS_PATH)
    dataset_version = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))["version"]
    split = build_stratified_split(
        judgments,
        dataset_version=dataset_version,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED,
    )
    payload = {
        "schema_version": split.schema_version,
        "dataset_version": split.dataset_version,
        "train": list(split.train),
        "validation": list(split.validation),
        "test": list(split.test),
    }
    SPLIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
