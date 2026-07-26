"""Tests for resumable benchmark summary persistence."""

import dataclasses
import json
from pathlib import Path

import pytest

from ragforge.evaluation.artifact_writer import write_atomic
from ragforge.evaluation.canonical_hash import canonical_json_hash
from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.evaluation.run_evidence import finalize_evidence_directory, write_summaries
from ragforge.evaluation.run_manifest import build_initial_manifest
from ragforge.ingestion.snapshot import snapshot_hash


def _lineage(model: str) -> GenerationLineage:
    return GenerationLineage(
        provider="gemini",
        model=model,
        prompt_hash="prompt",
        input_chunk_ids=(),
        input_source_hashes=(),
        answer_hash="answer",
        parsed_citations=(),
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        latency_seconds=0.5,
        cache_hit=False,
    )


def test_write_summaries_merges_prior_strategies_on_resume(tmp_path: Path) -> None:
    """A resumed run preserves summaries checkpointed by earlier strategies."""
    write_summaries(
        tmp_path,
        {"dense": {"n": 1.0}},
        {"dense": [_lineage("model")]},
        {},
        generation_usage={"dense": {"calls": 1}},
    )

    merged_usage = write_summaries(
        tmp_path,
        {"dense": {"n": 1.0}, "sparse": {"n": 1.0}},
        {"sparse": [_lineage("model")]},
        {},
        generation_usage={"sparse": {"calls": 1}},
    )

    assert set(merged_usage) == {"dense", "sparse"}
    generation = (tmp_path / "summaries" / "generation.json").read_text(encoding="utf-8")
    assert '"dense"' in generation
    assert '"sparse"' in generation


def test_finalize_evidence_covers_the_final_manifest_without_a_circular_root(
    tmp_path: Path,
) -> None:
    """The completed manifest verifies while its root excludes only self-reference."""
    manifest = build_initial_manifest(
        run_id="run-1",
        git_sha="abc123",
        corpus_hash="corpus",
        dataset_hash="dataset",
        split_hash="split",
        configuration_hash="config",
        models={},
        strategies=("dense",),
        execution={},
    )
    write_atomic(
        tmp_path / "manifest.json",
        json.dumps(dataclasses.asdict(manifest), ensure_ascii=False, indent=2),
    )
    write_atomic(tmp_path / "report.json", "{}")

    final_manifest = finalize_evidence_directory(tmp_path, manifest)

    recorded = {
        relative: digest
        for digest, relative in (
            line.split("  ", 1)
            for line in (tmp_path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        )
    }
    assert final_manifest.status == "completed"
    assert recorded["manifest.json"] == snapshot_hash(tmp_path / "manifest.json")
    assert final_manifest.artifact_root_hash == canonical_json_hash(
        {relative: digest for relative, digest in recorded.items() if relative != "manifest.json"}
    )


def test_finalize_evidence_rejects_an_already_completed_manifest(tmp_path: Path) -> None:
    """Completed evidence cannot be silently checksummed and overwritten again."""
    manifest = build_initial_manifest(
        run_id="run-1",
        git_sha="abc123",
        corpus_hash="corpus",
        dataset_hash="dataset",
        split_hash="split",
        configuration_hash="config",
        models={},
        strategies=("dense",),
        execution={},
    )
    completed = finalize_evidence_directory(tmp_path, manifest)

    with pytest.raises(ValueError, match="running manifest"):
        finalize_evidence_directory(tmp_path, completed)
