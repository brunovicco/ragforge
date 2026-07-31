#!/usr/bin/env python3
"""Validate human labels, publish a calibration report, and enforce its gate.

Usage:
    uv run python scripts/finalize_calibration.py
    uv run python scripts/finalize_calibration.py --show-disagreements
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragforge.evaluation.artifact_writer import write_atomic  # noqa: E402
from ragforge.evaluation.calibration_workflow import (  # noqa: E402
    KAPPA_TARGET,
    build_calibration_report,
    build_calibration_samples,
    calibration_gate_passed,
    load_human_labels,
    load_sealed_calibration,
    scores_have_ordinal_disagreement,
)

CAL_DIR = ROOT / "datasets" / "regrag-br" / "calibration"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-disagreements",
        action="store_true",
        help="list samples where judge and human land in different ordinal bins",
    )
    return parser.parse_args()


def main() -> None:
    """Compute the report and return failure when any required kappa misses the floor."""
    args = _parse_args()
    try:
        labels = load_human_labels(CAL_DIR / "labels.json")
        sealed = load_sealed_calibration(CAL_DIR / ".judge-sealed.json")
        samples = build_calibration_samples(labels, sealed)
        report = build_calibration_report(samples, sealed.run_id)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid calibration workspace: {exc}") from exc

    write_atomic(
        CAL_DIR / "report.json",
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))

    if args.show_disagreements:
        print("\nDisagreements (different ordinal bins):\n")
        for sample in samples:
            if scores_have_ordinal_disagreement(sample):
                print(
                    f"  {sample.sample_id:44s} judge={sample.judge_score:.2f} "
                    f"human={sample.human_score:.2f}"
                )

    if not calibration_gate_passed(report):
        print("\nCalibration gate failed; judge metrics are not validated for publication.")
        raise SystemExit(1)

    overall = report["overall"]
    if isinstance(overall, dict) and overall.get("weighted_kappa", 0.0) >= KAPPA_TARGET:
        print(f"\nCalibration passed and meets the {KAPPA_TARGET:.2f} publication target.")
    else:
        print("\nCalibration passed the floor but remains below the publication target.")
    print(f"Report written to {CAL_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
