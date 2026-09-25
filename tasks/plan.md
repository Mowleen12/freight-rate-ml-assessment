# Implementation Plan: Freight Rate Prediction Assessment

## Overview
Build a complete, reproducible end-to-end ML solution in a single Jupyter notebook
(`freight_rate_assessment.ipynb`) covering EDA, data quality, feature engineering,
time-based validation, model comparison, final retraining, submission CSV generation,
official scorer execution, and a report-ready summary — supporting the GitHub repo,
PDF/DOCX report, and Loom deliverables.

## Key Data Facts (verified read-only)
- `train-test.csv`: 48,000 rows, 2025-01-01 → 2025-10-31, target `posted_rate`
- `validation.csv`: 12,000 rows, 2025-11-01 → 2025-12-31 (future, no target)
- Missing: `weight` (300), `market_index` (374) in train; same pattern in validation
- `weight` has 292 negative values (sign errors; |weight| within valid 5,000–47,500)
- 8 pickup/delivery cities unseen in validation (725 rows ≈ 6%)
- Coordinates are fixed per city; haversine ≈ distance (corr 0.9995) → redundancy
- Environment: Python 3.10, sklearn 1.7.1, xgboost 3.0.5 (no lightgbm/catboost)
- No duplicate rows/ids; `market_index` mean shifts 1.08 → 0.93 in validation

## Architecture Decisions
- Single notebook, relative paths, fixed seed 42, pipelines (preprocessing fit on train only)
- Chronological split: train ≤ 2025-09-30, holdout October 2025; `validation.csv` never touched
- Unseen categories: `OneHotEncoder(handle_unknown='ignore')` + raw coordinates retained
- weight negatives → `abs()` (physically impossible, recovered magnitude plausible)
- Candidates: Ridge (+log variant), RandomForest, ExtraTrees, HistGradientBoosting, XGBoost

## Task List
- [ ] Task 1: Scratch experiment to verify model ranking and metric ranges
- [ ] Task 2: Build notebook (18 required sections) via nbformat builder
- [ ] Task 3: Execute notebook top-to-bottom; fix any cell failures
- [ ] Task 4: Verify outputs: validation_predictions.csv (12,000), December file (31), scorer chart
- [ ] Task 5: Final checklist + report-ready summary

## Checkpoint: Complete
- Notebook executes cleanly with no hidden state
- `score.py` passes; `scorer_results/candidate_december.png` exists
- All numbers in the report summary computed live from the files
