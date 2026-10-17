# Out-of-Time Validation - TODO

## Status: Data Not Available in Repo

The current repository contains only **2014-era Lending Club loan data** (data/loans.csv, n=5,942 after filtering). Proper out-of-time validation requires training on one vintage and testing on a later one to assess temporal generalization.

## Why Out-of-Time Validation Matters

In-sample splits (random 80/20) test the model's ability to generalize to unseen loans from the **same time period**. Out-of-time validation tests whether the model works when:
- Economic conditions change (interest rates, unemployment, housing prices)
- Borrower behavior shifts
- Lending platform policies evolve
- Credit bureau scoring methodologies update

A model with AUC 0.63 on 2014 data might perform much worse (or better) on 2015-2016 vintages.

## What's Needed

### Ideal Setup
1. **Training vintage:** 2014 loans (current data/loans.csv)
2. **Validation vintage:** 2015 or 2016 Lending Club loans with the same schema
3. **Approach:** 
   - Train model (with feature whitelist) on 2014 data
   - Apply frozen model to 2015/2016 test set
   - Report ROC-AUC, calibration drift, score distribution shift

### Data Sources (to explore)
- **Kaggle:** Lending Club has published multi-year datasets; search for "Lending Club Loan Data" on Kaggle
- **Lending Club Statistics:** Historical loan data was publicly available through Lending Club's website (now part of LendingTree; check if archives are accessible)
- **Academic repositories:** Some researchers have mirrored Lending Club data for reproducibility

### What NOT to Do
- Do NOT simulate out-of-time validation by re-splitting the 2014 data with a time-based cutoff (e.g., Q1-Q3 train, Q4 test)
- This would still be in-sample for the calendar year and wouldn't capture true temporal drift
- Only use actual data from a different origination year

## TODO: Obtain 2015-2016 Data

1. Search Kaggle for "Lending Club" datasets with 2015-2016 vintages
2. Verify schema compatibility with data/loans.csv (column names, definitions)
3. Document provenance (URL, date accessed, any preprocessing)
4. Add script: `scripts/run_oot_validation.py` to:
   - Load 2014 as train
   - Load 2015-2016 as test
   - Apply same preprocessing + feature whitelist
   - Train model on 2014 only
   - Report metrics on 2015-2016
5. Update MODEL_CARD.md with OOT results

## Temporary Status

**Out-of-time validation is not included in this release.** The model quality metrics (ROC-AUC 0.63) reflect in-sample performance on 2014 data only. Any production use would require OOT validation on more recent vintages.

---
**Last updated:** 2024 (after target leakage fix)
