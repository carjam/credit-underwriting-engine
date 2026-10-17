# Portfolio Credit Risk Decisioning

This document describes how this repository elevates an existing underwriting ML notebook into a **portfolio-oriented credit risk decisioning narrative**.

## What the Decisioning Layer Adds

| Layer | Implementation | Notes |
|-------|---------------|-------|
| **SMOTE fix** | Split-first, SMOTE on training fold only | Corrects data leakage; test set reflects real 81.1% good-loan prevalence |
| **Target leakage fix** | Explicit feature whitelist (`config/features.yaml`) | Removed post-outcome features; AUC dropped from 0.999 (leaky) to 0.63 (honest) |
| **Probability calibration** | `CalibratedClassifierCV` (isotonic, cv=3) | Improves reliability; Brier score maintained at ~0.16 (realistic for this discrimination) |
| **Decision rules** | `decisions()` — approve / review / decline | Thresholds: approve ≥ 0.65, review 0.50–0.65, decline < 0.50 |
| **Risk tiers** | `risk_tier()` — prime / near-prime / subprime | Cutoffs: prime ≥ 0.70, near-prime 0.50–0.70, subprime < 0.50 |
| **Tier calibration** | `calibrate_tiers()` | Returns empirical DR per tier; validates monotone ordering |
| **Threshold simulation** | `threshold_sweep()` | Sweeps approval cutoffs; reports approval rate and default rate among approved |
| **Break-even DR** | `break_even_default_rate()` | Revenue / (LGD × avg_loan); annotated as red dashed line on threshold sweep chart |
| **Lifetime EL** | `lifetime_expected_loss()` | Accumulates PD × LGD × EAD_t over loan tenor (IFRS 9 lifetime ECL structure) |
| **Portfolio UL** | `unexpected_loss()` | Vasicek single-factor std-dev of aggregate loss (Basel retail ρ=0.12) |
| **SHAP** | Global beeswarm + one-loan waterfall | Feature importance and individual-loan explanation for adverse action context |

## Where to Run It

Open `Credit_Underwriting_Decisioning-Lending_Club.ipynb` and run all cells from a clean kernel. The Portfolio Credit Risk Decisioning Layer at the end depends on `best_model`, `calibrated_model`, `X_train`, `X_test`, and `y_test` being defined by the preceding cells.

## Measured Portfolio Outcomes (Honest Metrics, Post-Leakage-Fix)

The following are directional examples from a HistGradientBoosting model with application-time features only (no pricing) on the honest test set (n=1,189 loans, 81.1% good-loan prevalence):

| Metric | Typical range |
|---|---|
| ROC-AUC | 0.63 (realistic for consumer credit) |
| Default rate among high-score approvals (p≥0.70) | ~18% (realistic for 2014 near-prime cohort) |
| Break-even default rate | 66.7% (revenue $3k / LGD 45% × $10k) |
| Score distribution | 88.5% prime (≥0.70), 7.8% near-prime (0.50–0.70), 3.7% subprime (<0.50) |

> **Interpretation note:** The realistic default rate of ~18% among approved loans is consistent with consumer credit in the 2014 era. The original notebook showed "0.52% DR" due to target leakage from post-outcome features (payment history, outstanding balances, bureau refreshes, hardship/settlement flags). After fixing the leakage with an explicit whitelist of application-time features (`config/features.yaml`), the model now reflects genuine credit risk discrimination at application time.

**Key insight:** The break-even DR (66.7%) still exceeds the observed DR (~18%), indicating the portfolio would be profitable at this threshold — but the margin is realistic (not the implausible 26× RAROC from the leaky model). The capital framing (EL, UL, break-even) is structurally correct; fixing the input features yields meaningful business interpretation.

## Reuse, Not Rebuild

The decisioning functions in `src/decisioning.py` are pure, stateless functions that consume any array of `p_good` scores. Replacing the underlying model (after the leakage investigation) requires no changes to the policy layer.

## CECL and Compliance Context

The EL and UL outputs are **directional internal metrics**, not accounting provisions. See `docs/CECL_MAPPING.md` for the full ASC 326 / IFRS 9 gap map, and `docs/ADVERSE_ACTION.md` for the ECOA/Reg B adverse action workflow.
