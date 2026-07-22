"""Tests for the cross-encoder reranking adapter, using a fake CrossEncoder (no model load)."""

from collections.abc import Callable

import pytest

from ragforge.domain.models import Chunk, Query
from ragforge.reranking import cross_encoder_reranker as module
from ragforge.reranking.cross_encoder_reranker import CrossEncoderReranker
from ragforge.reranking.errors import RerankingError


class _FakeScores:
    """Stands in for the numpy array sentence-transformers' CrossEncoder.predict returns."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeCrossEncoder:
    def __init__(self, predict: Callable[[list[tuple[str, str]]], _FakeScores]) -> None:
        self._predict = predict
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> _FakeScores:
        self.predict_calls.append(pairs)
        return self._predict(pairs)


def _install_fake_cross_encoder(
    monkeypatch: pytest.MonkeyPatch, predict: Callable[[list[tuple[str, str]]], _FakeScores]
) -> _FakeCrossEncoder:
    fake_model = _FakeCrossEncoder(predict)
    monkeypatch.setattr(module, "CrossEncoder", lambda model_name, device=None: fake_model)
    return fake_model


def test_score_returns_one_value_per_chunk_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful predict() call maps to one score per chunk, same order."""
    fake_model = _install_fake_cross_encoder(monkeypatch, lambda pairs: _FakeScores([0.9, 0.1]))
    reranker = CrossEncoderReranker("test-model")
    chunks = [
        Chunk(chunk_id="c1", text="texto um", structural_ids=("c1",)),
        Chunk(chunk_id="c2", text="texto dois", structural_ids=("c2",)),
    ]

    scores = reranker.score(Query(text="pergunta"), chunks)

    assert scores == [0.9, 0.1]
    assert fake_model.predict_calls == [[("pergunta", "texto um"), ("pergunta", "texto dois")]]


def test_score_returns_empty_list_for_no_chunks_without_calling_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No chunks means no model call - an empty pool has nothing to score."""
    fake_model = _install_fake_cross_encoder(monkeypatch, lambda pairs: _FakeScores([]))
    reranker = CrossEncoderReranker("test-model")

    scores = reranker.score(Query(text="pergunta"), [])

    assert scores == []
    assert fake_model.predict_calls == []


def test_score_raises_when_predict_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A predict() failure is translated to RerankingError."""

    def predict(pairs: list[tuple[str, str]]) -> _FakeScores:
        raise RuntimeError("boom")

    _install_fake_cross_encoder(monkeypatch, predict)
    reranker = CrossEncoderReranker("test-model")
    chunk = Chunk(chunk_id="c1", text="texto", structural_ids=("c1",))

    with pytest.raises(RerankingError, match="failed to score 1 pair"):
        reranker.score(Query(text="pergunta"), [chunk])


def test_raises_when_model_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CrossEncoder construction failure is translated to RerankingError."""

    def raise_error(model_name: str, device: str | None = None) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "CrossEncoder", raise_error)

    with pytest.raises(RerankingError, match="failed to load model"):
        CrossEncoderReranker("test-model")
