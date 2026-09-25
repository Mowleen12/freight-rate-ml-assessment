# Report-Ready Summary — Freight Rate Prediction Assessment

## Dataset statistics
- train-test.csv: 48,000 rows x 14 source columns, 2025-01-01 to 2025-10-31, target `posted_rate`.
- validation.csv: 12,000 rows, 2025-11-01 to 2025-12-31, no target (future window).
- Target: mean $2,374, median $2,031, min $57, max $25,533, skew 1.9.
- Distance: mean 1,136 mi; corr(distance, rate) = 0.909 (log-log 0.967).
- Unique pickup/delivery cities: 64/64; equipment levels: ['Dry Van', 'Flatbed', 'Reefer'].

## Data-quality findings
- Missing: weight 300 (0.62%), market_index 374 (0.78%) in train; validation similar (165 / 249). Imputed inside the pipeline (fit on training rows only).
- 292 negative weights (impossible) recovered via abs(); |weight| then spans 5,000-47,500 lb.
- No duplicate rows or IDs; template order matches validation order.
- Unseen categories: 8 pickup / 8 delivery cities only in validation (1,447 rows, 12.06%) -> OneHotEncoder(handle_unknown='ignore').
- Distribution shift: market_index mean 1.083 (train) -> 0.927 (validation).
- Leakage checks: target excluded from features; load_id excluded; market_index/quote_signal corr with target only +0.03/-0.04 and are provided at prediction time; no target-based encodings; validation.csv never used for fitting.

## Validation methodology
- Chronological split: train 2025-01-01 to 2025-09-30 (43,147 rows) -> validate October 2025 (4,853 rows); final validation.csv (Nov-Dec) untouched. Mirrors production: train past, predict future; a random split would leak the mild monthly drift.
- Ridge alpha chosen by TimeSeriesSplit(3) inside the training partition; all preprocessing in a Pipeline fit on training rows only.

## Model comparison (October 2025 holdout)

| model                                      |      MAE |     RMSE |      R2 |   fit_seconds | detail        |
|:-------------------------------------------|---------:|---------:|--------:|--------------:|:--------------|
| Ridge (alpha grid, TimeSeriesSplit)        |  127.312 |  647.629 |  0.8205 |          1.88 | best alpha=10 |
| ExtraTrees                                 |  131.466 |  651.665 |  0.8182 |         13.21 |               |
| RandomForest                               |  137.874 |  652.591 |  0.8177 |         12.59 |               |
| HistGradientBoosting                       |  152.46  |  657.529 |  0.815  |          4.36 |               |
| Baseline: median $/mile (2.143) x distance |  253.848 |  695.047 |  0.7932 |          0    | baseline      |
| XGBoost                                    |  335.167 |  829.987 |  0.7052 |          1.67 |               |
| Baseline: train mean                       | 1179.8   | 1528.57  | -0      |          0    | baseline      |

No-calendar diagnostics (Section 10): Ridge (alpha grid, TimeSeriesSplit) [no calendar]: MAE 139.8; ExtraTrees [no calendar]: MAE 126.4; HistGradientBoosting [no calendar]: MAE 130.5; XGBoost [no calendar]: MAE 186.9.

## Selected model and justification
- **Ridge (alpha grid, TimeSeriesSplit)** — best holdout MAE among production-eligible models (MAE 127.31, RMSE 647.63, R2 0.8205); margin over runner-up 4.15 MAE; 89.2% lower MAE than the mean baseline and 49.8% lower than the strong $/mile baseline. Chosen under a pre-registered rule (lowest MAE, RMSE/R2 guards).
- Advantages: handles unseen cities safely, extrapolates calendar features into Nov/Dec (trees cannot extrapolate beyond their training range), tolerant of the December file's absent columns via the same imputer, fast and interpretable; December output varies smoothly with date as the chart requires.

## Validation metrics (October 2025 holdout, selected model)
- MAE 127.31 $ | RMSE 647.63 $ | R2 0.8205 | residuals centred (no bias); largest errors on the most expensive loads (RMSE > MAE).

## Feature-engineering decisions
- Calendar: month, day, dayofweek, dayofyear, is_weekend, days_since_start (continuous, origin 2025-01-01); weekofyear as a continuous week index (ISO weeks wrap 52->1 at the year boundary and would inject a false step into the December extrapolation); year dropped (constant).
- weight = abs() for sign errors; missing numerics imputed in-pipeline.
- delta_lat / delta_lon coordinate differences kept; great-circle distance EXCLUDED (corr 0.9995 with given distance = redundancy); load_id EXCLUDED from features.
- distance x equipment interactions (price-per-mile slope differs by equipment).
- Categoricals: one-hot with handle_unknown='ignore' (unseen-city safe, leakage-free).

## Outputs
- validation_predictions.csv — 12,000 rows, columns load_id,predicted_rate (validated).
- december_chart_inputs.csv — 31 rows, predicted range 968.7-1030.8 $.
- December chart: scorer_results/candidate_december.png (official score.py, unmodified).