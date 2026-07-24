#!/usr/bin/env python3
"""Print a judge calibration report from a human-labeled sample file (ADR-0007/ADR-0018).

Usage: uv run python scripts/judge_calibration_report.py <calibration_file.json>

No calibration file ships with this repository: the samples it must contain
(~30+, stratified by query class, answerable/unanswerable, strong/weak
strategies, negation, exceptions, cross-references, numerical claims) are
genuine human curation work - see
src/ragforge/evaluation/judge_calibration.py for the exact expected schema
and why this script cannot generate that file itself.
"""

import argparse
import sys
from pathlib import Path

from ragforge.evaluation.judge_calibration import (
    compute_calibration_report,
    load_calibration_samples,
)


def parse_args() -> argparse.Namespace:
    """Parse this script's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_file", type=Path)
    return parser.parse_args()


def main() -> None:
    """Load the calibration file and print its aggregate agreement report."""
    args = parse_args()
    if not args.calibration_file.exists():
        raise SystemExit(f"calibration file not found: {args.calibration_file}")

    samples = load_calibration_samples(args.calibration_file)
    report = compute_calibration_report(samples)

    print(f"Calibration report for {args.calibration_file} ({int(report['n'])} samples):\n")
    for key, value in sorted(report.items()):
        if key == "n":
            continue
        print(f"  {key}: {value:.3f}")

    kappa = report["weighted_kappa"]
    if kappa < 0.60:
        print(
            f"\nweighted_kappa={kappa:.3f} is below the ADR-0007/ADR-0018 acceptance "
            "threshold (0.60) - judge scores must be reported with this limitation, "
            "not as validated ground truth."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
