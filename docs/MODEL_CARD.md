# Model Card (Prototype)

## Purpose

Demonstrate **credit risk decisioning** on top of classifier outputs: tiers, approve/review/decline, threshold tradeoffs, calibration, SHAP views, and a directional capital lens. **Not** a production underwriting or capital system.

## Scope

| In scope | Out of scope |
|----------|----------------|
| Notebook-based training/eval on a fixed Lending Club–style sample | Live scoring API, model registry, or governance sign-off |
| Policy logic in `src/decisioning.py` + `config/policy.default.yaml` | Calibrated PD as regulatory default probability |
| Reproducibility via pinned `requirements.txt` and tests | Out-of-time validation on current vintages |
| Break-even DR, lifetime EL, and UL helpers (illustrative) | CECL/IFRS 9 compliant ACL computation |

## Data

- **Default artifact:** `data/loans.csv` (public Lending Club sample mirror; **6,305** raw rows, **5,942** after filtering to binary loan_status; **2014-era** issue dates).
- **Override:** set `LENDING_CLUB_DATA_PATH` before the notebook load cell.
- **Label:** binary `loan_status` mapped to Fully Paid (1) vs Charged Off (0) after filtering.
- **Class balance:** ~81.1% Fully Paid / 18.9% Charged Off in the original data.

## Pipeline methodology

### SMOTE and train/test split

SMOTE is applied **after** the stratified train/test split, exclusively to the training fold:

```
X, y  (original, imbalanced — 81.1% good)
  └─ train_test_split (stratified, 80/20, random_state=42)
        ├─ X_train, y_train  →  SMOTE  →  X_train_smote (balanced, used to fit models)
        └─ X_test,  y_test   (original distribution — used ONLY for evaluation)
```

**Previous state (corrected):** The original notebook applied SMOTE to the full dataset before splitting. This caused synthetic minority-class samples—generated from the neighborhood of test-set observations—to leak into training, inflating apparent performance. The corrected order means `X_test` reflects real loan prevalence.

### Probability calibration

`CalibratedClassifierCV(best_model, cv=3, method='isotonic')` is fit on the SMOTE-balanced training data. The result (`calibrated_model`) is used in the Portfolio Decisioning Layer. Reliability diagrams (before and after) are rendered in the notebook's Calibration Assessment section.

## Measured performance (honest metrics, post-leakage-fix)

All metrics are on the **honest, unaugmented test set** (81.1% good loans) using HistGradientBoosting with **application-time features only**.

### Credit-Specific Metrics (Primary Evaluation)

| Metric | No pricing features | With pricing (grade/sub_grade/int_rate) | Industry Benchmark |
|--------|---------------------|----------------------------------------|-------------------|
| **ROC-AUC** | **0.6301** | **0.6304** | 0.60-0.75 (consumer credit) |
| **Gini** | **0.2602** | **0.2608** | 0.40+ moderate, 0.60+ strong |
| **KS Statistic** | **0.2235** | **0.2168** | 0.30+ moderate, 0.40+ strong |
| **PR-AUC (Charged Off)** | **0.2452** | **0.2551** | Baseline 0.189 (minority class %) |
| Brier score | 0.1591 | 0.1596 | Lower is better |
| Default rate (p≥0.70) | 18.0% | 17.8% | Realistic for 2014 near-prime |

**Interpretation:** Weak but honest discrimination (Gini 0.26). Top decile has 4× lower default rate than worst decile. Pricing features add almost no lift (+0.0003 AUC), confirming bureau data captures the credit signal.

### Decile Analysis (No Pricing)

| Decile | Count | Score Range | Default Rate | Lift |
|--------|-------|-------------|--------------|------|
| 1 (best) | 118 | 0.93-1.00 | 4.2% | 0.22× |
| 2-4 | 354 | 0.88-0.93 | 11-14% | 0.6-0.7× |
| 5-9 | 590 | 0.68-0.88 | 22-31% | 1.2-1.6× |
| 10 (worst) | 127 | 0.06-0.68 | 25.2% | 1.33× |

**Key finding:** ~4-7× separation between best and worst deciles. Model provides some risk ranking, but lift is weak (typical for AUC 0.63 on small sample).

### Classification Metrics (Secondary, for Reference)

| Metric | No pricing | With pricing | Note |
|--------|------------|--------------|------|
| Accuracy (0.5 cutoff) | 78.9% | 79.1% | **Misleading:** below 81.1% baseline (approve-all) |
| Precision (Fully Paid) | 81.1% | 81.4% | Slightly above baseline |
| Recall (Fully Paid) | 96.4% | 96.1% | Model approves most good loans |
| Recall (Charged Off) | 4.0% | 6.2% | **Irrelevant:** 0.5 cutoff not used in practice |

**Why accuracy/recall are misleading:** See `docs/CREDIT_EVALUATION.md` for full explanation. Short version: accuracy penalizes declining bad loans, and 0.5-cutoff recall is arbitrary for 19% default rate. Credit models use Gini/KS/decile lift instead.

**Score distribution (no pricing):**

| Score band | Count (n=1,189) | % of test set |
|---|---|---|
| ≥ 0.70 (prime) | 1,052 | 88.5% |
| 0.50–0.70 (near-prime) | 93 | 7.8% |
| < 0.50 (subprime) | 44 | 3.7% |

**What changed:** The original notebook trained on ALL numeric columns, including post-outcome features like `total_pymnt`, `recoveries`, `out_prncp`, `last_fico_range_*`, and hardship/settlement flags. These are only known AFTER the loan outcome, causing severe target leakage (original AUC 0.9991). The fix: explicit whitelist of application-time features in `config/features.yaml`, validated by `src/feature_guard.py` and tested in `tests/test_feature_guard.py`.

See `docs/CREDIT_EVALUATION.md` for comprehensive credit-focused analysis.

## Expected loss and capital helpers

| Helper | Function | What it computes |
|--------|----------|-----------------|
| Single-period EL | `simple_expected_loss_per_approved` | DR × LGD × avg_loan — annual snapshot |
| Lifetime EL | `lifetime_expected_loss` | Σ conditional_PD_t × LGD × EAD_t over loan tenor |
| Unexpected Loss | `unexpected_loss` | Portfolio loss std-dev via Vasicek single-factor (ρ=0.12) |
| Break-even DR | `break_even_default_rate` | Revenue / (LGD × avg_loan); threshold above which approvals destroy value |

**Illustrative values** (will vary depending on threshold; these are directional examples, not from a full decisioning run):

| Metric | Typical range for consumer credit |
|---|---|
| Default rate among approved | 15–20% (realistic for near-prime/mixed cohort) |
| Break-even DR | ~67% (revenue $3k / LGD 45% × $10k) |
| Lifetime EL/loan (36 months) | ~$500–$900 (depends on DR and tenor) |
| RAROC | 1.2–2.5× (typical for consumer lending) |

The **realistic default rates** (15–20% in this 2014 sample) are consistent with consumer credit. The original "0.52% DR" was an artifact of target leakage, not genuine model performance.

All values are **illustrative** (assumed avg_loan, LGD=0.45 Basel II floor, tenor=36mo). They are **not** CECL/IFRS 9 compliant. See `docs/CECL_MAPPING.md`.

## LGD assumption

`lgd = 0.45` is the **Basel II/III supervisory floor** for senior unsecured consumer credit. Actual LGD varies:

| Loan type | Typical LGD range |
|-----------|------------------|
| Secured (auto, mortgage) | 10–30% |
| Unsecured personal (with collections) | 40–65% |
| Charged-off, no collections | 70–90% |

Replace with portfolio-specific empirical recovery data for any production use.

## Known limitations

### Target leakage — FIXED

**The problem (original notebook):** The feature selection used `df.select_dtypes(include=[np.number])`, which included post-outcome features only known AFTER the loan pays off or defaults:
- Payment history: `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`
- Outstanding principal: `out_prncp`, `out_prncp_inv`
- Bureau refreshes during servicing: `last_fico_range_high/low`, `last_credit_pull_d`
- Hardship/settlement fields: `hardship_flag`, `debt_settlement_flag`, `settlement_amount`, etc.

A quick check showed `total_rec_prncp` alone gave AUC ~0.91, and the full leaky feature set gave AUC 0.9996.

**The fix:** 
1. Explicit whitelist of application-time features in `config/features.yaml`
2. Feature guard module (`src/feature_guard.py`) to validate features before training
3. Comprehensive tests (`tests/test_feature_guard.py`) to prevent regression
4. Modified notebook to use `select_model_features()` instead of `select_dtypes()`

**Result:** AUC dropped from 0.9991 to 0.63 (realistic for consumer credit on application-time features).

### Pricing features add minimal signal

Including Lending Club's `grade`/`sub_grade`/`int_rate` (the platform's own risk tiers) adds only +0.0003 AUC. This confirms:
- These features encode Lending Club's risk assessment, not the outcome itself (so they're not "leakage" in the technical sense)
- But they add little beyond what's already captured by bureau attributes
- The original leakage was from post-outcome fields, not from pricing tiers

### Other limitations

- Calibration improves Brier score but works with the base model's discrimination power; honest discrimination (~0.63 AUC) yields realistic calibrated probabilities.
- **No** CECL/IFRS 9/regulatory capital treatment — see `docs/CECL_MAPPING.md`.
- Bundled data is **2014-era**; no claim about current market or any live portfolio.
- No out-of-time validation; all quality figures are from the same origination year as training. (Public Lending Club datasets for other years may exist on Kaggle or academic sources — not included here.)
- No model drift monitoring, override tracking, or governance documentation for production use.
- Low recall on Charged Off (4–6%) indicates the model struggles to identify defaults at this threshold; threshold/cost tradeoff should be tuned for the business context.

## Adverse action and review disposition

Loans in the **"review"** bucket require a documented disposition process for ECOA/Reg B compliance. See `docs/ADVERSE_ACTION.md` for workflow guidance and required disclosures.

## Reproduction

1. `pip install -r requirements.txt`
2. Run `Credit_Underwriting_Decisioning-Lending_Club.ipynb` end-to-end (clean kernel), or `pytest --run-notebook` (slow).
3. Record **git commit** and **data file** when reporting metrics.
