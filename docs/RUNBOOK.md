# Runbook

## Environment

- Python: `3.13.x` tested locally; CI uses `3.12`.
- Install: `pip install -r requirements.txt`
- Deterministic test seeds are enforced in `tests/conftest.py`.

## Data snapshot (bundled sample)

- File: `data/loans.csv`
- Source: public Lending Club sample mirror (documented in commit history)
- Raw rows: `6,305`
- Rows after filtering to binary loan_status: `5,942`
- Class balance: ~81.1% Fully Paid, 18.9% Charged Off
- Example issue date in sample: `Dec-2014`

If you use a different CSV, set `LENDING_CLUB_DATA_PATH` before running the notebook.

## Pipeline summary

```
data/loans.csv
  └─ preprocessing (encode, scale, impute)
        └─ stratified 80/20 split on original data
              ├─ X_train → SMOTE → X_train_balanced (for model fitting)
              │     └─ CalibratedClassifierCV (cv=3, isotonic) → calibrated_model
              └─ X_test (original distribution — evaluation only)
                    └─ decisioning layer (tiers, thresholds, EL, UL)
```

SMOTE is applied **after** the split. The test set is never augmented.

## Standard execution path

1. Fast checks (decisioning unit tests, smoke, regression, notebook schema):
   ```
   pytest
   ```
2. Full notebook execution check (several minutes):
   ```
   pytest --run-notebook
   ```
3. Interactive notebook:
   - Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb`
   - **Kernel → Restart & Run All** (must run from clean kernel; cells depend on execution order)

## Policy-driven decisioning outside Jupyter

Use the CLI with a score file that contains `p_good`:

```
python scripts/run_decisioning.py --scores-csv <scores.csv> --policy config/policy.default.yaml
```

Optional: include `y_true` in `scores.csv` to compute default-rate, break-even comparison, lifetime EL, and portfolio UL in CLI output.

## Known performance numbers (honest features, post-leakage-fix)

For reference when comparing runs. These are from the corrected pipeline (SMOTE + calibration, application-time features only) on the bundled `data/loans.csv`:

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC (various models) | 0.60–0.89 | Varies by model; XGBoost ~0.63-0.70 range |
| Gini coefficient | 0.20–0.78 | 2×AUC - 1; credit-specific discrimination measure |
| KS statistic | 0.15–0.45 | Max separation between cumulative good/bad distributions |
| Default rate (test set, p≥0.70) | ~18% | Realistic for 2014 near-prime cohort |
| Score distribution (p≥0.70) | ~88% of test set | Most loans score as approvable under policy threshold |
| Brier score (calibrated) | 0.15–0.16 | Consistent with moderate discrimination power |

> These reflect honest model performance on application-time features (no post-outcome leakage). See `docs/MODEL_CARD.md` for full metrics and `LEAKAGE_FIX_SUMMARY.md` for before/after comparison.
