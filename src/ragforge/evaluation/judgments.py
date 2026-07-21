"""Loads the golden-set relevance judgments (ADR-0002) from datasets/regrag-br/."""

import json
from pathlib import Path

from ragforge.domain.models import (
    JudgedRef,
    Judgment,
    Query,
    QueryClass,
    RelevanceGrade,
    StructuralRef,
)


def load_judgments(path: Path) -> list[Judgment]:
    """Parse the judgments JSON file into a list of Judgment objects.

    Raises:
        ValueError: If a structural ref, query_class, or grade value is malformed.
        KeyError: If a required field is missing from an entry.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    judgments: list[Judgment] = []
    for entry in data["judgments"]:
        query = Query(
            text=entry["text"],
            query_class=QueryClass(entry["query_class"]) if entry.get("query_class") else None,
        )
        relevant_refs = tuple(
            JudgedRef(ref=StructuralRef.parse(ref["ref"]), grade=RelevanceGrade(ref["grade"]))
            for ref in entry["relevant_refs"]
        )
        judgments.append(
            Judgment(question_id=entry["question_id"], query=query, relevant_refs=relevant_refs)
        )
    return judgments
