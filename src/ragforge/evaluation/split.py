"""Loads the versioned question-ID split (ADR-0012) from datasets/regrag-br/."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Split:
    """A versioned partition of golden-set question IDs into train/validation/test."""

    schema_version: int
    dataset_version: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        """Every selected question ID across all partitions, in file order."""
        return self.train + self.validation + self.test


def load_split(path: Path) -> Split:
    """Parse the split JSON file into a Split.

    Raises:
        KeyError: If a required field is missing from the file.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return Split(
        schema_version=data["schema_version"],
        dataset_version=data["dataset_version"],
        train=tuple(data["train"]),
        validation=tuple(data["validation"]),
        test=tuple(data["test"]),
    )
