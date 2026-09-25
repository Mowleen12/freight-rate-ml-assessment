# Task List — Freight Rate ML Assessment

- [x] Task 1: Scratch experiment to verify model ranking and metric ranges
      Verification: results table printed; winner identified (Ridge)
- [x] Task 2: Build notebook (18 required sections) via nbformat builder
      Verification: .ipynb valid, all 18 sections present (59 cells)
- [x] Task 3: Execute notebook top-to-bottom; fix any cell failures
      Verification: nbconvert --execute completes with exit code 0, zero error outputs (4 full runs)
- [x] Task 4: Verify outputs: validation_predictions.csv (12,000), December file (31), scorer chart
      Verification: score.py exit 0 standalone + in-notebook; 16/16 output checks pass;
      scorer_results/candidate_december.png generated (smooth December curve)
- [x] Task 5: Final checklist + report-ready summary
      Verification: notebook prints FINAL SUBMISSION CHECKLIST (16/16);
      report_summary.md written with live-computed statistics

## Checkpoint: Foundation (after Task 1)
- [x] Model ranking known; notebook narrative matches actual results

## Fixes made during execution
- [x] Table join bug (NaN calendar deltas) — model-name suffix stripped before join
- [x] Hardcoded unseen-city counts corrected to computed union (1,447 rows / 12.06%)
- [x] ISO week-of-year wrap (52->1) caused a false ~$190 jump at 2025-12-29 in the
      December chart -> continuous week index (days_since_start // 7 + 1); holdout MAE unchanged
- [x] Report column count corrected to 14 source columns (2 EDA helper columns excluded)
