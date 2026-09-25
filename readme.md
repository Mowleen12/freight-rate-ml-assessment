# Freight Rate Prediction Challenge — Solution

End-to-end machine learning solution for the Freight Rate Prediction assessment: predict the `posted_rate`
for every load in the final validation set (12,000 rows) and for the fixed December 2025 chart (31 rows),
with a leakage-safe pipeline, chronological validation, an evidence-based model comparison, and verification
by the official `score.py` scorer. Everything runs from a single reproducible Jupyter notebook at the repo
root — no hidden state, fixed random seed, relative paths, all metrics computed live from the supplied files.

**Headline result:** regularized linear regression (**Ridge**, α=10, time-series-tuned) selected on an
October-2025 holdout with **MAE $127.31, RMSE $647.63, R² 0.8205** — 49.8% lower MAE than a strong
rate-per-mile baseline and 89.2% lower than a mean baseline.

---

## Quick Start

```bash
git clone https://github.com/Mowleen12/freight-rate-ml-assessment.git
cd freight-rate-ml-assessment
python -m pip install -r requirements.txt

# run the full solution (EDA -> models -> CSVs -> official scorer)
python -m jupyter nbconvert --to notebook --execute freight_rate_assessment.ipynb --inplace

# verify with the official scorer (unmodified)
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

Expected scorer output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
```

| Command | Purpose |
|---------|---------|
| `python -m pip install -r requirements.txt` | Install dependencies (scorer + notebook) |
| `python -m jupyter nbconvert --to notebook --execute freight_rate_assessment.ipynb --inplace` | Execute the notebook top-to-bottom, regenerating all outputs |
| `python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv` | Official validation + December chart generation |
| `jupyter lab freight_rate_assessment.ipynb` | Open the notebook interactively (any state works — it is self-contained) |

Requires Python ≥ 3.10. Tested with pandas 2.3, scikit-learn 1.7, XGBoost 3.0.

---

## Repository Layout

| Path | Description |
|------|-------------|
| `freight_rate_assessment.ipynb` | **The solution** — 18 documented sections, executed and committed with outputs |
| `requirements.txt` | Dependencies (scorer: matplotlib/numpy/pandas; notebook: scikit-learn, xgboost, jupyter) |
| `score.py` | Official scorer (supplied, **unmodified**) |
| `train-test.csv` | Labeled development data — 48,000 rows, 2025-01-01 → 2025-10-31 |
| `validation.csv` | Final validation set — 12,000 rows, 2025-11-01 → 2025-12-31, **no target** |
| `validation-predictions-template.csv` | Submission template (load_id order) |
| `december-chart-inputs.csv` | Supplied December inputs — 31 rows, blank `predicted_rate` (left untouched) |
| `validation_predictions.csv` | **Output:** 12,000 predictions, `load_id,predicted_rate` |
| `december_chart_inputs.csv` | **Output:** 31 December rows with `predicted_rate` filled (7-column structure preserved) |
| `scorer_results/candidate_december.png` | **Output:** official December chart (created by `score.py`) |
| `report_summary.md` | **Output:** report-ready summary (stats, methodology, model table, metrics) for the PDF/DOCX report |
| `freight-rate-ml-assessment.pdf` | Original assessment instructions |
| `tasks/` | Implementation plan and task checklist used to build the solution |
| `readme.md` | This file |

---

## The Problem

Predict freight `posted_rate` (USD) from load characteristics. Labeled data covers **January–October 2025**;
the rows to predict are **November–December 2025** — a forecasting problem, not an i.i.d. one. Features:
pickup/delivery city + coordinates, distance, equipment, weight, `market_index`, `quote_signal`, date.
`load_id` is an identifier only (never a feature, preserved for submission).

---

## Data Findings (all computed in the notebook)

| Check | Result |
|-------|--------|
| Shapes | train 48,000 × 14 · validation 12,000 × 13 · December 31 × 7 · template 12,000 × 2 |
| Date ranges | train 2025-01-01→10-31 · validation 2025-11-01→12-31 (strictly future) |
| Missing values | `weight` 300 (0.62%), `market_index` 374 (0.78%) in train; 165 / 249 in validation |
| Impossible values | 292 negative weights (sign errors — abs value within valid 5,000–47,500 lb range) |
| Duplicates | None (rows or IDs), template order matches validation order exactly |
| Unseen categories | 8 pickup + 8 delivery cities only in validation → **1,447 rows (12.06%)** |
| Target | mean $2,374 · median $2,031 · min $57 · max $25,533 · skew 1.9 (right tail is real, not corrupt) |
| Dominant driver | distance: corr 0.909 with rate (0.967 log-log); rate ≈ distance × price-per-mile |
| Geography | one fixed coordinate per city; haversine ≈ distance (corr 0.9995) → redundant |
| Distribution shift | `market_index` mean 1.083 (train) → 0.927 (validation) |
| Leakage review | target excluded; `load_id` excluded; market/quote corr with target only +0.03/−0.04 and are supplied at prediction time; no target-based encodings; validation rows never fit anything |

---

## Methodology

### Validation strategy (the core decision)

- **Train partition:** 2025-01-01 → 2025-09-30 (43,147 rows) — fits every preprocessing step and model.
- **Holdout:** all of **October 2025** (4,853 rows) — the most recent labeled month, the closest proxy for
  "the future".
- **`validation.csv` (Nov–Dec):** never used for training, imputation, scaling, encoding, tuning or
  selection. It has no target, so it cannot be scored; its features are also kept out of preprocessing.

A random split would put the same weeks on both sides of the fold and flatter every model, while the actual
task requires extrapolating two months past the training range. The chronological split measures exactly
that. All preprocessing (median imputation, scaling, one-hot) lives inside scikit-learn `Pipeline`s, so it
is fit only on training rows of each split.

### Feature engineering

- **Calendar:** month, day, day-of-week, day-of-year, `is_weekend`, `days_since_start` (continuous, origin
  2025-01-01), and `weekofyear` as a **continuous week index** (`days_since_start // 7 + 1`). ISO week
  numbers wrap 52→1 when 2025-12-29 opens ISO week 1 of 2026 — that wrap was observed as a false ~$190
  jump in the first generated December chart and eliminated; the index keeps weekly granularity without
  the discontinuity. `year` dropped (constant).
- **Geography:** `delta_lat` / `delta_lon` kept (tree splits cannot derive them from raw columns);
  great-circle distance **excluded** (redundant with given `distance`); raw coordinates kept only as a
  geographic fallback for the 12.06% unseen-city rows.
- **Domain:** `distance × equipment` interactions (price-per-mile slope differs by equipment:
  Reefer > Flatbed > Dry Van).
- **Cleanup:** `weight = abs(weight)` for the 292 sign-flipped rows; missing numerics imputed in-pipeline.
- **Categoricals:** `OneHotEncoder(handle_unknown="ignore")` — unseen cities become an all-zero block,
  leakage-free by construction (target/frequency encoding rejected: needs cross-fitting to be safe and
  still assigns unseen cities a meaningless global average).

### Model experiments (same split, same pipeline, same features)

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| **Ridge (α grid, TimeSeriesSplit)** | **127.31** | **647.63** | **0.8205** |
| ExtraTrees | 131.47 | 651.67 | 0.8182 |
| RandomForest | 137.87 | 652.59 | 0.8177 |
| HistGradientBoosting | 152.46 | 657.53 | 0.8150 |
| Baseline: median $/mile × distance | 253.85 | 695.05 | 0.7932 |
| XGBoost | 335.17 | 829.99 | 0.7052 |
| Baseline: train mean | 1,179.80 | 1,528.57 | ~0 |

(Pre-registered rule: lowest October-holdout MAE among production-eligible models, with RMSE/R² guards.)
LightGBM/CatBoost are not installed in the environment and not in the supplied requirements, so they were
excluded rather than silently skipped.

A no-calendar diagnostic shows calendar features **help** Ridge (+12.5 MAE) but **hurt** every tree family
(ExtraTrees −5.1, HGB −21.9, XGBoost −148.3) — trees fit in-sample calendar patterns that do not survive
into the later month and cannot extrapolate a time trend beyond their training range. The production model
must carry calendar features (the December chart needs date sensitivity), which further favors Ridge.

### Selected model

**Ridge** — best holdout MAE (margin 4.15 over ExtraTrees), best RMSE/R², handles unseen cities via the
one-hot fallback, tolerates the December file's absent columns through the same imputer, extrapolates
calendar features into November/December, fast and interpretable, and produces a smoothly date-varying
December curve. Residuals on the holdout are centred (bias +$2.4); largest absolute errors sit on the most
expensive loads (RMSE > MAE everywhere) — no single collapsed segment.

Final pipeline retrained on **all 48,000 labeled rows** before generating submissions.

---

## Outputs & Verification

| File | Status |
|------|--------|
| `validation_predictions.csv` | 12,000 rows · exact column order `load_id,predicted_rate` · IDs match the template in set **and** order · no duplicates/missing · all finite and strictly positive |
| `december_chart_inputs.csv` | 31 rows · original 7-column structure preserved · dates 2025-12-01→12-31 complete · fixed inputs intact (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) · predictions $968.7–$1,030.8 |
| `scorer_results/candidate_december.png` | Created by the unmodified official `score.py` (exit 0); also embedded in the notebook |

The notebook's Section 17 re-reads every written file from disk and prints a **16/16 submission checklist**
(the same checks the grader runs). The supplied `december-chart-inputs.csv` (hyphenated) is intentionally
left blank; the completed file uses the scorer's expected name `december_chart_inputs.csv`.

---

## Notebook Structure (18 sections)

1. Assessment Objective · 2. Environment and Imports · 3. Load Data · 4. Dataset Overview ·
5. Exploratory Data Analysis · 6. Data Quality Analysis · 7. Feature Engineering ·
8. Validation Strategy · 9. Baseline Model · 10. Model Experiments and Comparison ·
11. Error/Residual Analysis · 12. Final Model Selection · 13. Final Training on All Labeled Data ·
14. Generate Validation Predictions · 15. Generate December Predictions · 16. Run Official Scorer ·
17. Final Output Validation · 18. Conclusions and Limitations

Every modeling decision is explained in Markdown immediately before/after the relevant code. The notebook
is executable top-to-bottom with a fresh kernel (`nbconvert --execute`), uses seed 42 everywhere, creates
output directories automatically, and hard-codes no results.

---

## Assessment Deliverables

| Deliverable | Where |
|-------------|-------|
| GitHub repo with code, dependencies, run instructions | This repository |
| `validation_predictions.csv` | Repository root |
| PDF/DOCX report (validation/split approach + `candidate_december.png`) | Build from `report_summary.md` + `scorer_results/candidate_december.png`; full narrative in the notebook |
| 2–3 minute Loom (data findings, quality issues, model reasoning, split approach, code walkthrough) | Walk the notebook's Sections 5 → 6 → 8 → 10 → 12 → 16 |

---

## Limitations

- Accuracy on `validation.csv` cannot be measured locally — the target does not exist; only the spotter's
  post-submission metrics will confirm it. All reported metrics are October-2025 holdout metrics.
- One chronological split (October); `market_index` already shifts between train and validation, so
  holdout error is a proxy, not a guarantee.
- December predictions extrapolate two months past the training range (linear calendar extrapolation).
- For the ~12% unseen-city rows the model falls back to generic behavior (all-zero city block +
  coordinates); per-city precision is impossible there by construction.
- Residual RMSE (~$648) is much larger than MAE (~$127): substantial rate noise remains unexplained by
  any available feature.

---

## Original Assessment Instructions

See `freight-rate-ml-assessment.pdf` for the full assessment instructions.

### What to do

1. Train and validate your model using `train-test.csv`.
2. Predict every load in `validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `validation-predictions-template.csv` and save it as
   `validation_predictions.csv`.
4. Predict every row in `december-chart-inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

### Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
