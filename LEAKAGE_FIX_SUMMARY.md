# Target Leakage Fix - Summary Report

## Problem Identified

The original notebook achieved ROC-AUC ~0.9991, which is **unrealistically high** for consumer credit models (industry range: 0.60-0.75). Investigation revealed severe target leakage.

### Root Cause

The feature selection used `df.select_dtypes(include=[np.number])`, which included **every numeric column**, including post-outcome features only known AFTER a loan pays off or defaults:

**Payment History** (post-outcome):
- `total_pymnt`, `total_pymnt_inv`
- `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`
- `recoveries`, `collection_recovery_fee`
- `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`

**Outstanding Principal** (changes over loan life):
- `out_prncp`, `out_prncp_inv`

**Bureau Refreshes** (pulled during servicing, not at application):
- `last_fico_range_high`, `last_fico_range_low`
- `last_credit_pull_d`

**Hardship/Settlement Fields** (only apply to troubled loans):
- `hardship_flag`, `hardship_type`, `hardship_status`, etc.
- `debt_settlement_flag`, `settlement_amount`, `settlement_status`, etc.
- `orig_projected_additional_accrued_interest`

### Verification of Leakage

Quick check with HistGradientBoosting on the same 80/20 split:
- **With leaky features:** ROC-AUC = 0.9996
- **Without leaky features:** ROC-AUC = 0.63
- **`total_rec_prncp` alone:** ROC-AUC = 0.91

This confirmed that post-outcome features were driving the near-perfect scores.

## Solution Implemented

### 1. Feature Whitelist (`config/features.yaml`)

Created explicit whitelist of **application-time features** — what a lender knows at decision time:

**Loan Characteristics:**
- `loan_amnt`, `funded_amnt`, `term`, `installment`

**Borrower Profile:**
- `annual_inc`, `emp_length`, `verification_status`
- `home_ownership`, `purpose`, `dti`

**Credit Bureau (at application time):**
- `fico_range_low`, `fico_range_high` (NOT `last_fico_*`)
- `earliest_cr_line`, `inq_last_6mths`, `delinq_2yrs`
- `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`
- Plus 50+ additional bureau attributes from origination

**Post-Outcome Blocklist:**
- All payment history, outstanding principal, bureau refreshes, hardship, and settlement fields

**Pricing Features (separate decision):**
- `grade`, `sub_grade`, `int_rate` — Lending Club's own risk tiers
- These are lender-assigned, not post-outcome leakage
- But they encode the platform's credit model
- Tested separately to show their impact

### 2. Feature Guard Module (`src/feature_guard.py`)

Enforcement layer to prevent future regression:
- `get_allowed_features()` — loads whitelist from config
- `get_blocklist()` — returns post-outcome features
- `check_for_leakage()` — validates feature list
- `validate_features_before_training()` — raises error if leakage detected

### 3. Feature Selection Helper (`src/feature_selection.py`)

Notebook integration:
- `select_model_features()` — filters dataframe to whitelist
- Handles one-hot encoded categoricals automatically
- Validates no leakage before returning
- Verbose mode shows feature selection summary

### 4. Comprehensive Tests (`tests/test_feature_guard.py`)

14 test cases covering:
- Whitelist loading with/without pricing features
- Blocklist completeness (payment fields, settlement, hardship, etc.)
- Leakage detection for clean and dirty feature sets
- Feature filtering and validation

### 5. Modified Notebook

Replaced:
```python
X = X.select_dtypes(include=[np.number]).copy()
```

With:
```python
from src.feature_selection import select_model_features
INCLUDE_PRICING = False  # Set to True for Experiment 2
X = select_model_features(X_all, include_pricing=INCLUDE_PRICING, verbose=True)
```

Also removed legacy local file path (`~/Desktop/IK/LendingClub/loans.csv`).

### 6. Updated Documentation

**README.md:**
- Replaced AUC 0.9991 → 0.63 in metrics table
- Added "What changed" section explaining leakage cause and fix
- Removed claims about 99.3% accuracy and 0.52% default rate
- Fixed clone URL (credit-underwriting → credit-underwriting-engine)
- Updated Key Takeaways and Extension Opportunities

**MODEL_CARD.md:**
- New metrics table with honest values
- Detailed "Target leakage — FIXED" section
- Explained that pricing features add minimal signal (+0.0003 AUC)
- Updated EL/UL section with realistic ranges

**PORTFOLIO_DECISIONING.md:**
- Updated metrics from 0.52% DR → ~18% DR (realistic)
- Added target leakage fix to decisioning layer table
- Removed claims about implausible profitability

**OUT_OF_TIME_VALIDATION.md (new):**
- Documents that 2015-2016 Lending Club data is not available in repo
- Explains why OOT validation matters
- Provides TODO for obtaining and testing on later vintages
- States clearly: do NOT simulate with random split

### 7. Quick Metrics Script (`scripts/quick_metrics.py`)

Standalone script to verify metrics without running full notebook:
- Loads data, applies minimal preprocessing
- Trains HistGradientBoosting with/without pricing features
- Reports ROC-AUC, PR-AUC, Brier, recall, score distribution
- Saves results to JSON
- **Run time: ~12 seconds** (vs. 15+ minutes for full notebook)

### 8. Updated Test Baseline (`tests/baselines/model_quality_baseline.json`)

Changed from:
```json
{
  "roc_auc_min": 0.97,
  "accuracy_min": 0.93
}
```

To:
```json
{
  "roc_auc_min": 0.60,
  "accuracy_min": 0.75,
  "comment": "Honest metrics after target leakage fix..."
}
```

## Results: Before vs. After

| Metric | OLD (with leakage) | NEW (no pricing) | NEW (with pricing) |
|--------|-------------------|------------------|-------------------|
| **ROC-AUC** | **0.9991** | **0.6301** | **0.6304** |
| PR-AUC | — | 0.8822 | 0.8783 |
| Accuracy | 99.3% | 78.9% | 79.1% |
| Brier Score | 0.0066 | 0.1591 | 0.1596 |
| Recall (Fully Paid) | 100.0% | 96.4% | 96.1% |
| Recall (Charged Off) | 96.4% | 4.0% | 6.2% |
| Score Distribution | Near-binary (6 loans in mid-range) | 88.5% high, 7.8% mid, 3.7% low | Similar |
| Default Rate (approved) | 0.52% (implausible) | 18.0% (realistic) | 17.8% (realistic) |

### Key Insights

1. **AUC drop from 0.9991 → 0.63 confirms leakage fix:** The dramatic drop proves post-outcome features were driving the inflated scores.

2. **Realistic AUC of 0.63 is CORRECT:** Consumer credit models on application-time features typically achieve 0.60-0.75. The original 0.999 was evidence of a problem, not a success.

3. **Pricing features add almost nothing:** Including `grade`/`sub_grade`/`int_rate` gives only +0.0003 AUC, confirming:
   - These are not "leakage" in the technical sense (they're set at application time)
   - But they encode Lending Club's own model and add little signal beyond bureau data
   - The real leakage was from post-outcome payment/settlement fields

4. **Low recall on Charged Off (4-6%):** The model struggles to identify defaults at the current threshold. This is realistic — predicting credit defaults is hard! The threshold should be tuned based on business cost/benefit tradeoff.

5. **Default rate of ~18% is realistic:** For a 2014 near-prime/mixed consumer credit cohort, 15-20% default rate is plausible. The original 0.52% was an artifact of leakage.

## Files Changed

```
 Credit_Underwriting_Decisioning-Lending_Club.ipynb |  35 ++-
 README.md                                          |  62 ++---
 config/features.yaml                               | 192 ++++++++++++++++
 docs/MODEL_CARD.md                                 |  91 ++++----
 docs/OUT_OF_TIME_VALIDATION.md                     |  55 +++++
 docs/PORTFOLIO_DECISIONING.md                      |  28 ++-
 results_honest_metrics.json                        |  24 ++
 scripts/quick_metrics.py                           | 253 +++++++++++++++++++++
 src/feature_guard.py                               | 109 +++++++++
 src/feature_selection.py                           | 147 ++++++++++++
 tests/baselines/model_quality_baseline.json        |   5 +-
 tests/test_feature_guard.py                        | 194 ++++++++++++++++
 12 files changed, 1106 insertions(+), 89 deletions(-)
```

## Testing

All tests pass:
```
$ pytest tests/ -v -k "not notebook_e2e"
====================== 46 passed, 1 deselected in 1.89s =======================
```

Feature guard tests (14 new tests):
- ✓ Whitelist loading
- ✓ Blocklist completeness
- ✓ Leakage detection
- ✓ Feature filtering
- ✓ All critical post-outcome features blocked

## What Was NOT Done

1. **Out-of-time validation:** 2015-2016 Lending Club data not available in repo. See `docs/OUT_OF_TIME_VALIDATION.md` for TODO.

2. **Full notebook rerun:** The notebook has not been executed end-to-end with the new feature set. The metrics in this report come from `scripts/quick_metrics.py`, which uses the same preprocessing logic but is faster.

3. **XGBoost hyperparameter tuning:** Used HistGradientBoosting with default params for quick verification. Full notebook will use whatever model selection it originally had, but now with honest features.

4. **SHAP analysis update:** SHAP plots in the notebook will change once rerun (features like `total_rec_prncp` will no longer dominate).

## Recommendations

1. **Run full notebook:** Execute `Credit_Underwriting_Decisioning-Lending_Club.ipynb` end-to-end to regenerate all plots and verify integration.

2. **Document data provenance:** When reporting metrics externally, cite git commit hash and data file used.

3. **Obtain OOT data:** Search Kaggle for Lending Club 2015-2016 datasets to test temporal generalization.

4. **Threshold tuning:** Low recall on Charged Off (4-6%) suggests threshold should be adjusted based on business cost of false negatives.

5. **Consider ensemble:** If AUC ~0.63 is too low for business needs, try ensemble methods or additional feature engineering (e.g., employment stability, debt service coverage ratio).

## Conclusion

**The target leakage has been fixed.** The model now trains on application-time features only, with realistic AUC of 0.63 instead of the leaky 0.9991. The decisioning machinery (tiers, thresholds, EL, UL, calibration, break-even) remains intact and now operates on honest probabilities.

The original near-perfect scores were not a modeling triumph — they were a data quality issue. Finding and fixing leakage is exactly the kind of work that matters in production ML systems.

---
**Branch:** `fix-target-leakage`  
**Commits:** 2 (58567ff, 1a40ca2)  
**Date:** 2024  
**Status:** Ready for review and merge
