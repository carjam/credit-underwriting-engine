"""
Credit-specific evaluation metrics for risk models.

Calculates metrics appropriate for credit scoring:
- Gini coefficient (2*AUC - 1)
- KS statistic (Kolmogorov-Smirnov)
- PR-AUC for minority class (Charged Off)
- Score decile analysis with lift
- Population Stability Index (PSI)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc_fn
from scipy import stats


def gini_coefficient(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Gini coefficient = 2*AUC - 1.
    
    Standard credit scoring metric. Ranges from 0 (no discrimination)
    to 1 (perfect separation). Industry benchmark:
    - 0.20-0.40: weak model
    - 0.40-0.60: moderate model
    - 0.60+: strong model
    """
    auc = roc_auc_score(y_true, y_proba)
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between cumulative
    distributions of good and bad loans.
    
    KS = max|F_good(score) - F_bad(score)|
    
    Industry benchmark:
    - 0.20-0.30: weak
    - 0.30-0.40: moderate
    - 0.40+: strong
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    
    # Split into good and bad
    good_scores = p[y == 1]
    bad_scores = p[y == 0]
    
    # KS test
    ks_stat, _ = stats.ks_2samp(good_scores, bad_scores)
    return float(ks_stat)


def pr_auc_minority_class(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Precision-Recall AUC for the minority class (Charged Off = 0).
    
    More informative than accuracy for imbalanced data.
    Baseline: proportion of minority class (e.g., 0.19 if 19% charge off).
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    
    # For minority class (0 = Charged Off), use 1 - p as score
    y_minority = (y == 0).astype(int)
    p_minority = 1 - p
    
    precision, recall, _ = precision_recall_curve(y_minority, p_minority)
    return float(pr_auc_fn(recall, precision))


def score_deciles(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Score decile (or quintile) analysis for credit risk.
    
    Returns table with:
    - decile: 1 (lowest risk) to n_bins (highest risk)
    - count: number of loans in decile
    - score_min, score_max: score range for decile
    - default_rate: empirical DR in decile
    - cum_default_capture: % of all defaults captured up to this decile
    - lift: DR in decile / overall DR
    
    Args:
        n_bins: 10 for deciles, 5 for quintiles, 20 for vigintiles, etc.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    
    # Define bins (highest scores = lowest risk = decile 1)
    # Sort descending so decile 1 is best loans
    sorted_idx = np.argsort(-p)
    y_sorted = y[sorted_idx]
    p_sorted = p[sorted_idx]
    
    n = len(y)
    bin_size = n // n_bins
    
    overall_dr = (y == 0).mean()
    total_defaults = (y == 0).sum()
    
    rows = []
    cumulative_defaults = 0
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
        
        y_bin = y_sorted[start_idx:end_idx]
        p_bin = p_sorted[start_idx:end_idx]
        
        count = len(y_bin)
        defaults_in_bin = (y_bin == 0).sum()
        cumulative_defaults += defaults_in_bin
        
        dr = defaults_in_bin / count if count > 0 else 0.0
        lift = dr / overall_dr if overall_dr > 0 else 0.0
        cum_capture = cumulative_defaults / total_defaults if total_defaults > 0 else 0.0
        
        rows.append({
            'decile': i + 1,
            'count': int(count),
            'score_min': float(p_bin.min()),
            'score_max': float(p_bin.max()),
            'default_rate': float(dr),
            'cum_default_capture': float(cum_capture),
            'lift': float(lift),
        })
    
    return pd.DataFrame(rows)


def population_stability_index(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI): measures drift in score distribution.
    
    PSI = Σ (current_pct - baseline_pct) × ln(current_pct / baseline_pct)
    
    Interpretation:
    - PSI < 0.10: no significant change
    - 0.10 ≤ PSI < 0.25: moderate change, investigate
    - PSI ≥ 0.25: significant change, model may be stale
    
    Used for out-of-time validation to detect population shift.
    """
    baseline = np.asarray(baseline_scores)
    current = np.asarray(current_scores)
    
    # Define bins based on baseline quantiles
    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # Ensure max value is included
    
    # Bin both distributions
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions
    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)
    
    # PSI calculation (handle zeros)
    psi = 0.0
    for b_pct, c_pct in zip(baseline_pct, current_pct):
        # Add small epsilon to avoid log(0)
        b_pct = max(b_pct, 1e-10)
        c_pct = max(c_pct, 1e-10)
        psi += (c_pct - b_pct) * np.log(c_pct / b_pct)
    
    return float(psi)


def credit_metrics_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute all credit-specific metrics in one call.
    
    Returns dict with:
    - roc_auc
    - gini
    - ks_statistic
    - pr_auc_minority (for Charged Off class)
    - baseline_dr (portfolio default rate)
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'gini': float(gini_coefficient(y, p)),
        'ks_statistic': float(ks_statistic(y, p)),
        'pr_auc_minority': float(pr_auc_minority_class(y, p)),
        'baseline_dr': float((y == 0).mean()),
    }
