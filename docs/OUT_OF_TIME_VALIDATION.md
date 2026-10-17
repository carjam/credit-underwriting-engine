# Out-of-Time Validation - Data Acquisition Instructions

## Status: Requires Manual Data Download

The current repository contains only **2014-era Lending Club loan data** (data/loans.csv, n=5,942 after filtering). Proper out-of-time validation requires training on 2014 and testing on 2015-2016 to assess temporal generalization.

## How to Obtain 2015-2016 Data

### Step 1: Download from Kaggle

1. **Visit**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. **Download**: `accepted_2007_to_2018Q4.csv.gz` (or the most recent Lending Club dataset)
3. **Extract** the gzip file to get `accepted_2007_to_2018Q4.csv` (~2-3 GB)

**Note:** You may need a Kaggle account to download. The dataset is free and publicly available.

### Step 2: Filter to 2015-2016 Vintage

Run this Python script to extract 2015-2016 loans:

```python
import pandas as pd

# Load full dataset
df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False)

# Convert issue date to datetime
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')

# Filter to 2015-2016 issue dates
df_oot = df[(df['issue_d'].dt.year >= 2015) & (df['issue_d'].dt.year <= 2016)]

print(f"Total rows in full dataset: {len(df)}")
print(f"Rows from 2015-2016: {len(df_oot)}")

# Save filtered dataset
df_oot.to_csv('data/loans_oot.csv', index=False)
print("Saved to data/loans_oot.csv")
```

### Step 3: Run OOT Validation

```bash
python scripts/run_oot_validation.py
```

The script will:
1. Train on 2014 data only (data/loans.csv)
2. Freeze the pipeline (model, calibration, feature preprocessing)
3. Score 2015-2016 data with the frozen pipeline
4. Report temporal drift metrics

---

## What OOT Validation Tests

**Purpose:** Measure whether a model trained on 2014 still works on 2015-2016 loans.

**Key Questions:**
- Does discrimination degrade over time? (ROC-AUC drift)
- Is the model still calibrated? (Brier score, reliability diagram)
- Has the population shifted? (PSI on score distribution)
- Do risk tiers still separate good/bad loans? (Decile analysis)

**What it does NOT do:**
- ❌ Retrain, retune, or recalibrate on 2015-2016 data
- ❌ Re-select features or adjust thresholds using later vintages
- ❌ Mix 2014 and 2015-2016 data for training

**The later vintage is TEST ONLY.** Any adjustment based on 2015-2016 data would invalidate the temporal stability test.

---

## Expected Metrics (Hypothetical - Run Required)

Once you provide the 2015-2016 data, the script will report:

| Metric | 2014 Held-Out | 2015-2016 OOT | Drift | Status |
|--------|---------------|---------------|-------|--------|
| ROC-AUC | 0.6301 | *TBD* | *TBD* | ✓ if \|drift\| < 0.05 |
| Gini | 0.2602 | *TBD* | *TBD* | — |
| KS Statistic | 0.2235 | *TBD* | *TBD* | — |
| Brier Score | 0.1591 | *TBD* | *TBD* | Lower is better |
| PSI (score dist) | — | *TBD* | — | ✓ if < 0.10 |

**Interpreting drift:**
- **AUC drift < 0.05:** Model discrimination is stable
- **AUC drift 0.05-0.10:** Moderate degradation, investigate
- **AUC drift > 0.10:** Significant degradation, model may be stale
- **PSI < 0.10:** No significant population shift
- **PSI 0.10-0.25:** Moderate shift, monitor
- **PSI ≥ 0.25:** Significant shift, consider retraining

---

## Why This Matters

In-sample splits (random 80/20) test generalization to **unseen loans from the same time period**. Out-of-time validation tests whether the model works when:

- **Economic conditions change** (interest rates, unemployment, housing prices)
- **Borrower behavior shifts** (credit-seeking patterns, default propensity)
- **Lending policies evolve** (underwriting standards, product mix)
- **Bureau scoring changes** (FICO methodology updates)

A model with AUC 0.63 on 2014 might have AUC 0.50 (random) on 2016 if it fit idiosyncrasies of the 2014 vintage rather than stable credit signals.

---

## Alternative Data Sources (If Kaggle is Unavailable)

If Kaggle access is blocked or the dataset is removed:

1. **Lending Club Statistics Archive** (if still available)
2. **Academic repositories** (some researchers mirror Lending Club data)
3. **Other consumer credit datasets** with similar schema (Prosper, Upstart)

**Requirement:** Must have the same schema (same feature columns) and binary loan outcome (Fully Paid / Charged Off). Do NOT use data from different lenders or different products—schema mismatches will break feature engineering.

---

## What NOT to Do

❌ **Do NOT simulate OOT validation** by time-splitting 2014 data (e.g., Q1-Q3 train, Q4 test). This is still in-sample for the calendar year and doesn't capture true temporal drift.

❌ **Do NOT use the later data for any training, tuning, or feature selection.** The OOT set must remain untouched until the final validation run.

❌ **Do NOT cherry-pick vintages** that happen to work well. Pick 2015-2016 in advance and report results honestly, even if performance drops.

---

## Current Status

**Data:** ❌ Not available in repository (requires manual download)  
**Script:** ✅ Ready (`scripts/run_oot_validation.py`)  
**Tests:** ✅ Credit metrics validated (`tests/test_credit_metrics.py`)  
**Documentation:** ✅ This file

**Next step:** Download 2015-2016 Lending Club data from Kaggle and run `scripts/run_oot_validation.py`.

---

**Last updated:** 2024 (awaiting OOT data download)
