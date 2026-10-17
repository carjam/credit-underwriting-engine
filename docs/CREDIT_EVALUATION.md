# Credit-Focused Model Evaluation

## Why Credit Metrics Matter

Standard classification metrics (accuracy, recall at 0.5 cutoff) are **misleading for imbalanced credit data**:

- **Accuracy 78.9%** sounds low, but **approving every loan gives 81.1%** (the good-loan rate). Accuracy penalizes the model for declining bad loans.
- **Charged-Off recall 4%** at the 0.5 cutoff is irrelevant—lenders don't use 0.5; they set policy thresholds based on business economics (approval volume vs. loss).
- **The 0.5 cutoff is arbitrary** for imbalanced credit problems where the base rate is 19% defaults.

**Credit scoring uses different measures** that reflect risk separation and business value:

1. **ROC-AUC / Gini:** Overall discrimination power (AUC = 0.63 → Gini = 0.26)
2. **KS statistic:** Maximum separation between good and bad loan score distributions
3. **PR-AUC for minority class:** Precision-recall tradeoff for identifying defaults
4. **Decile analysis:** Default rate and lift by score band (business lens)
5. **Threshold sweep:** Approval rate vs. default rate at various policy cutoffs

---

## Current Model Performance (Honest Features, Post-Leakage-Fix)

### Primary Credit Metrics (Application-time features only, no pricing)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | **0.6301** | Moderate discrimination (industry: 0.60-0.75 typical) |
| **Gini** | **0.2602** | 2*AUC - 1; weak lift (industry: 0.40+ is moderate) |
| **KS Statistic** | **0.2235** | Weak separation (industry: 0.30+ is moderate) |
| **PR-AUC (Charged Off)** | **0.2452** | Above baseline (0.189), some lift on minority class |
| **Brier Score** | **0.1591** | Calibration quality |
| **Default Rate (p≥0.70)** | **18.0%** | Realistic for 2014 near-prime cohort |

**With pricing features (grade/sub_grade/int_rate):** AUC 0.6304, Gini 0.2608, KS 0.2168—**almost no improvement**, confirming bureau data already captures the credit signal.

---

## Decile Analysis: Risk Separation by Score Band

**Without pricing features:**

| Decile | Count | Score Range | Default Rate | Cumulative Default Capture | Lift vs. Portfolio |
|--------|-------|-------------|--------------|---------------------------|--------------------|
| 1 (best) | 118 | 0.933-1.000 | 4.2% | 2.2% | 0.22× |
| 2 | 118 | 0.915-0.933 | 13.6% | 9.3% | 0.72× |
| 3 | 118 | 0.911-0.915 | 11.0% | 15.1% | 0.58× |
| 4 | 118 | 0.881-0.911 | 13.6% | 22.2% | 0.72× |
| 5 | 118 | 0.859-0.881 | 22.0% | 33.8% | 1.16× |
| 6 | 118 | 0.851-0.859 | 22.0% | 45.3% | 1.16× |
| 7 | 118 | 0.817-0.850 | 22.0% | 56.9% | 1.16× |
| 8 | 118 | 0.777-0.817 | 24.6% | 69.8% | 1.30× |
| 9 | 118 | 0.683-0.777 | 30.5% | 85.8% | 1.61× |
| 10 (worst) | 127 | 0.061-0.683 | 25.2% | 100.0% | 1.33× |

**Key findings:**
- **Top decile (best loans):** 4.2% default rate (4× lower than portfolio average)
- **Bottom decile (worst loans):** 25-30% default rate (1.5× higher than average)
- **Cumulative capture:** Top 5 deciles capture 34% of defaults with 50% of loans
- **The model provides some risk separation**, but lift is modest (typical for weak AUC)

---

## Threshold Sweep: Business Tradeoff View

The policy threshold (approve if P(Fully Paid) ≥ 0.65) is driven by business economics, not a fixed cutoff. Threshold analysis shows:

| Approval Threshold | Approval Rate | Default Rate Among Approved | Expected Loss |
|--------------------|---------------|------------------------------|---------------|
| 0.50 | ~95% | ~19% (baseline) | High |
| 0.65 (policy default) | ~81% | ~18% | Moderate |
| 0.70 | ~88% | ~18% | Moderate |
| 0.80 | ~65% | ~15% | Lower |

**Break-even default rate:** ~67% (revenue $3k / LGD 45% × $10k avg loan). All thresholds above 0.50 are profitable on this data, but the margin narrows as approval rate increases.

---

## Classification Metrics (Secondary, for Reference)

These are **not** the right lens for credit decisions, but are included for completeness:

| Metric | Value | Note |
|--------|-------|------|
| Accuracy (0.5 cutoff) | 78.9% | Below 81.1% baseline (approve-all rate) |
| Precision (Fully Paid) | 81.1% | Slightly above baseline |
| Recall (Fully Paid) | 96.4% | Model approves most good loans |
| Recall (Charged Off) | 4.0% | **Irrelevant:** 0.5 cutoff not used in practice |
| Confusion Matrix (0.5 cutoff) | TN=9, FP=216, FN=35, TP=929 | Low TN reflects imbalance + cutoff mismatch |

**Why accuracy is misleading:** Approving every loan gives 81.1% accuracy (the good-loan rate). A model that declines bad loans will have *lower* accuracy by design, but higher profit.

**Why 0.5-cutoff recall is irrelevant:** Lenders set thresholds based on break-even economics and portfolio targets, not 0.5. A model with 4% Charged-Off recall at 0.5 might have 60%+ recall at a 0.30 threshold if the business tolerates higher defaults for volume.

---

## Interpretation: Weak But Honest

**AUC 0.63 / Gini 0.26 / KS 0.22 reflect weak credit discrimination.** This is honest performance on application-time features from a small 2014 sample. Industry benchmarks for consumer credit:

- **Weak:** Gini 0.20-0.40, KS 0.20-0.30
- **Moderate:** Gini 0.40-0.60, KS 0.30-0.40
- **Strong:** Gini 0.60+, KS 0.40+

**This model is weak but usable** for a prototype. The decile table shows ~4× separation between best and worst deciles, which is enough to demonstrate the decisioning framework (tiers, thresholds, EL, UL, CECL mapping).

**What would improve discrimination:**
- **More data:** 2014 only (5,942 loans); industry models train on 100k-1M+ loans
- **Out-of-time validation:** Test on 2015-2016 vintages to assess temporal stability
- **Feature engineering:** Debt service coverage, employment stability, regional indicators
- **Ensemble methods:** Blend multiple models to capture different risk patterns
- **Alternative data:** Rent payment history, utility bills, bank account activity (not in this dataset)

---

## Summary: Credit-First Evaluation

**Stop saying:** "78.9% accuracy, 4% Charged-Off recall"  
**Start saying:** "Gini 0.26, KS 0.22, 4× lift in top decile, 18% DR at policy threshold"

The first statement invites a wrong reading (below baseline accuracy, terrible minority recall). The second statement is honest about what the model delivers: weak but real risk separation, usable for business decisioning, room for improvement.

**The decisioning framework (tiers, thresholds, EL, UL, CECL gap mapping) is production-ready.** The model's weak discrimination is a data/sample-size issue, not a framework problem. A better model would slot in without changing the policy layer.

---

**References:**
- See `scripts/credit_evaluation.py` for full metric calculations
- See `src/credit_metrics.py` for Gini, KS, decile, and PSI implementations
- See `tests/test_credit_metrics.py` for validation and benchmarks
