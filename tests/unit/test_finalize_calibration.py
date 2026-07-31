"""Behavior tests for the canonical calibration finalizer entrypoint."""

import json
import sys
from pathlib import Path

import pytest
import scripts.finalize_calibration as finalize_calibration

_ANSWER_SAMPLE_COUNT = 30


def _write_workspace(
    calibration_dir: Path,
    *,
    incomplete: bool = False,
    invert_faithfulness: bool = False,
) -> None:
    labels: list[dict[str, object]] = []
    judge_scores: dict[str, float] = {}
    for index in range(_ANSWER_SAMPLE_COUNT):
        judge_score = float(index % 2)
        for dimension in ("faithfulness", "answer_relevancy"):
            sample_id = f"q{index}-dense-{dimension}"
            human_score = judge_score
            if dimension == "faithfulness" and invert_faithfulness:
                human_score = 1.0 - judge_score
            labels.append(
                {
                    "sample_id": sample_id,
                    "dimension": dimension,
                    "human_score": (
                        None
                        if incomplete and index == 0 and dimension == "faithfulness"
                        else human_score
                    ),
                }
            )
            judge_scores[sample_id] = judge_score

    calibration_dir.mkdir()
    (calibration_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False),
        encoding="utf-8",
    )
    (calibration_dir / ".judge-sealed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_schema_version": 1,
                "run_id": "run-test",
                "judge_scores": judge_scores,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _invoke_finalizer(monkeypatch: pytest.MonkeyPatch, calibration_dir: Path) -> None:
    monkeypatch.setattr(finalize_calibration, "CAL_DIR", calibration_dir)
    monkeypatch.setattr(sys, "argv", ["finalize_calibration.py"])
    finalize_calibration.main()


def _read_report(calibration_dir: Path) -> dict[str, object]:
    payload: object = json.loads((calibration_dir / "report.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_finalizer_publishes_report_when_workspace_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration_dir = tmp_path / "calibration"
    _write_workspace(calibration_dir)

    _invoke_finalizer(monkeypatch, calibration_dir)

    report = _read_report(calibration_dir)
    gate = report["gate"]
    assert isinstance(gate, dict)
    assert gate["passed"] is True
    assert report["run_id"] == "run-test"
    assert "Calibration passed" in capsys.readouterr().out


def test_finalizer_rejects_incomplete_human_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_dir = tmp_path / "calibration"
    _write_workspace(calibration_dir, incomplete=True)

    with pytest.raises(SystemExit, match="labels are still empty"):
        _invoke_finalizer(monkeypatch, calibration_dir)

    assert not (calibration_dir / "report.json").exists()


def test_finalizer_returns_failure_when_one_required_dimension_misses_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calibration_dir = tmp_path / "calibration"
    _write_workspace(calibration_dir, invert_faithfulness=True)

    with pytest.raises(SystemExit) as exc_info:
        _invoke_finalizer(monkeypatch, calibration_dir)

    assert exc_info.value.code == 1
    report = _read_report(calibration_dir)
    gate = report["gate"]
    assert isinstance(gate, dict)
    assert gate["passed"] is False
    failed_results = gate["failed_results"]
    assert isinstance(failed_results, list)
    assert "faithfulness" in failed_results
    assert "Calibration gate failed" in capsys.readouterr().out
