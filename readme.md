# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Solution (this repository)

- `freight_rate_assessment.ipynb` — complete end-to-end solution: data inspection, EDA, data-quality analysis, feature engineering, chronological validation (train Jan–Sep 2025, validate October 2025), baseline + 5-model comparison, residual analysis, evidence-based model selection, final retraining, submission generation, and the official scorer run. Executes top-to-bottom from the project root with a fixed random seed; all results are computed live (no hard-coded metrics).
- Outputs produced by the notebook: `validation_predictions.csv` (12,000 rows), `december_chart_inputs.csv` (31 rows, filled), `scorer_results/candidate_december.png`, `report_summary.md` (report-ready summary).

### Run the solution

```bash
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute freight_rate_assessment.ipynb --inplace
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
```

Note: the supplied `december-chart-inputs.csv` (hyphenated) is left untouched; the notebook writes the completed file to `december_chart_inputs.csv`.
