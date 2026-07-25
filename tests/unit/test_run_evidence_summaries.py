"""Tests for resumable benchmark summary persistence."""

from pathlib import Path

from ragforge.evaluation.lineage_ports import GenerationLineage
from ragforge.evaluation.run_evidence import write_summaries


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
