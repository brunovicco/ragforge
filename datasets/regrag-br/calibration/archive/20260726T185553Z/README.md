# Legacy calibration workspace

This directory preserves the attempted calibration workspace derived from run
`20260726T185553Z` for audit history only.

The run predates evidence schema v1 and does not contain the exact ordered
`judge_contexts` used by the Faithfulness judge. The worksheet reconstructs
cited source excerpts instead, so neither it nor the sealed scores are eligible
for judge calibration. Do not fill these labels or use this workspace to report
agreement.

Original artifact SHA-256 values at archival time:

- `worksheet.md`: `f7ff210d12f7265049cc4cd741d9b0a9f22e520eaefe04fcfb9fc0fbc234ffaa`
- `labels.json`: `b11eae0437820ac3578c8389a2ef63613b6fa17b45f1eddccb24abd00e5133ca`
- `.judge-sealed.json`: `af5ed04f740bf3799b0f55ebde36e68a54e3afa2dcb04a1ff134b1d0f621aaf9`

A future eligible run must generate a fresh workspace directly under
`datasets/regrag-br/calibration/` with `build_calibration_worksheet.py`.
