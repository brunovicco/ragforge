"""Tests for the versioned split loader (ADR-0012)."""

import json
from pathlib import Path

from ragforge.evaluation.split import load_split

ROOT = Path(__file__).resolve().parents[2]
REAL_SPLIT_PATH = ROOT / "datasets/regrag-br/split.json"


def _write_split(tmp_path: Path, **overrides: object) -> Path:
    data = {
        "schema_version": 1,
        "dataset_version": "0.2",
        "train": ["q1"],
        "validation": ["q2"],
        "test": ["q3", "q4"],
        **overrides,
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_split_parses_every_partition(tmp_path: Path) -> None:
    """Every partition round-trips as a tuple of question IDs, in file order."""
    path = _write_split(tmp_path)

    split = load_split(path)

    assert split.schema_version == 1
    assert split.dataset_version == "0.2"
    assert split.train == ("q1",)
    assert split.validation == ("q2",)
    assert split.test == ("q3", "q4")


def test_all_ids_concatenates_every_partition_in_order(tmp_path: Path) -> None:
    """all_ids is train + validation + test, in that order."""
    path = _write_split(tmp_path)

    split = load_split(path)

    assert split.all_ids == ("q1", "q2", "q3", "q4")


def test_load_real_split_selects_all_230_questions_as_test() -> None:
    """The real split puts every currently-curated question in the test partition."""
    split = load_split(REAL_SPLIT_PATH)

    assert split.train == ()
    assert split.validation == ()
    assert len(split.test) == 230
    assert len(set(split.test)) == 230
