"""
Out-of-time validation script for credit model temporal stability.

This script:
1. Trains on 2014 data (data/loans.csv)
2. Scores on later vintage (data/loans_oot.csv - 2015-2016)
3. Reports temporal drift metrics (ROC-AUC, KS, Brier, calibration, PSI)

The later vintage is TEST ONLY - never used for training, tuning, or calibration.
"""
import os
import sys
from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_selection import select_model_features
from src.credit_metrics import (
    gini_coefficient,
    ks_statistic,
    pr_auc_minority_class,
    score_deciles,
    population_stability_index,
    credit_metrics_summary,
)


def load_and_preprocess_data(file_path, vintage_name=""):
    """Load and minimally preprocess data."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"See docs/OUT_OF_TIME_VALIDATION.md for download instructions."
        )
    
    print(f"\n{'='*70}")
    print(f"Loading {vintage_name} data from {file_path}...")
    print(f"{'='*70}")
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"Loaded {len(df)} rows")
    
    # Filter to binary loan_status
    before_filter = len(df)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    print(f"After filtering to binary outcomes: {len(df)} rows ({before_filter - len(df)} dropped)")
    
    if len(df) == 0:
        raise ValueError("No loans with final outcome (Fully Paid or Charged Off) in data")
    
    # Map loan_status to binary (1 = good, 0 = bad)
    df['loan_status'] = (df['loan_status'] == 'Fully Paid').astype(int)
    print(f"Class balance: {df['loan_status'].value_counts(normalize=True).round(3).to_dict()}")
    
    # One-hot encode key categoricals
    if 'term' in df.columns:
        df['term'] = df['term'].str.strip().str.replace(' months', '')
        df = pd.concat([df, pd.get_dummies(df['term'], prefix='term')], axis=1)
        df.drop('term', axis=1, inplace=True)
    
    for col in ['home_ownership', 'purpose', 'verification_status']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Convert sub_grade to numeric
    if 'sub_grade' in df.columns:
        grade_map = {chr(ord('A')+i): i for i in range(7)}
        df['sub_grade_num'] = df['sub_grade'].apply(
            lambda x: grade_map.get(x[0], -1) * 5 + int(x[1]) if pd.notna(x) and len(x) == 2 else -1
        )
        df['sub_grade'] = df['sub_grade_num']
        df.drop('sub_grade_num', axis=1, inplace=True)
    
    if 'grade' in df.columns:
        grade_map = {chr(ord('A')+i): i for i in range(7)}
        df['grade'] = df['grade'].map(grade_map)
    
    # Convert emp_length to numeric
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].str.replace('< 1 year', '0')
        df['emp_length'] = df['emp_length'].str.replace(' years?', '', regex=True)
        df['emp_length'] = df['emp_length'].str.replace('\\+', '', regex=True)
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')
    
    print(f"{'='*70}\n")
    return df


def train_frozen_pipeline(X_train, y_train):
    """
    Train model on 2014 data and return frozen pipeline.
    
    Returns calibrated model that must NOT be retrained on OOT data.
    """
    print("\n" + "="*70)
    print("TRAINING FROZEN PIPELINE ON 2014 DATA")
    print("="*70)
    
    # Apply SMOTE to training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_smote.shape}")
    
    # Train model
    print(f"Training HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=100,
        learning_rate=0.1,
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Calibrate
    print("Calibrating with isotonic regression...")
    calibrated_model = CalibratedClassifierCV(
        model, cv=3, method='isotonic'
    )
    calibrated_model.fit(X_train_smote, y_train_smote)
    
    print("="*70 + "\n")
    return calibrated_model


def evaluate_on_test_set(model, X_test, y_test, set_name="Test"):
    """Evaluate frozen model on a test set."""
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {set_name.upper()} SET")
    print(f"{'='*70}")
    
    # Score with frozen model
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Credit metrics
    credit_metrics = credit_metrics_summary(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    
    # Decile analysis
    decile_table = score_deciles(y_test, y_proba, n_bins=10)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    print(f"\n--- CREDIT METRICS ---")
    print(f"ROC-AUC:                 {credit_metrics['roc_auc']:.4f}")
    print(f"Gini:                    {credit_metrics['gini']:.4f}")
    print(f"KS Statistic:            {credit_metrics['ks_statistic']:.4f}")
    print(f"PR-AUC (Charged Off):    {credit_metrics['pr_auc_minority']:.4f}")
    print(f"Brier Score:             {brier:.4f}")
    print(f"Baseline DR:             {credit_metrics['baseline_dr']:.2%}")
    
    print(f"\n--- DECILE ANALYSIS ---")
    print(decile_table[['decile', 'count', 'default_rate', 'lift']].to_string(index=False))
    
    print(f"\n{'='*70}\n")
    
    return {
        'credit_metrics': credit_metrics,
        'brier': float(brier),
        'decile_table': decile_table.to_dict('records'),
        'calibration': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        },
        'scores': y_proba.tolist(),
    }


def main():
    """Run out-of-time validation."""
    print("="*70)
    print("OUT-OF-TIME VALIDATION: TEMPORAL STABILITY TEST")
    print("="*70)
    print("\nTraining on 2014 -> Testing on 2015-2016 (frozen pipeline)")
    print("="*70)
    
    # Paths
    train_path = Path("data/loans.csv")
    oot_path = Path("data/loans_oot.csv")
    
    # Check if OOT data exists
    if not oot_path.exists():
        print(f"\n{'!'*70}")
        print("OOT DATA NOT FOUND")
        print(f"{'!'*70}")
        print(f"\nExpected file: {oot_path}")
        print("\nTo obtain 2015-2016 Lending Club data:")
        print("1. Go to Kaggle: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("2. Download 'accepted_2007_to_2018Q4.csv.gz' (or similar)")
        print("3. Extract and filter to 2015-2016 issue dates:")
        print("   ```python")
        print("   import pandas as pd")
        print("   df = pd.read_csv('accepted_2007_to_2018Q4.csv')")
        print("   df['issue_d'] = pd.to_datetime(df['issue_d'])")
        print("   df_oot = df[(df['issue_d'].dt.year >= 2015) & (df['issue_d'].dt.year <= 2016)]")
        print("   df_oot.to_csv('data/loans_oot.csv', index=False)")
        print("   ```")
        print("4. Place the filtered CSV at: data/loans_oot.csv")
        print("5. Rerun this script")
        print(f"\n{'!'*70}\n")
        return None
    
    # Load 2014 training data
    df_train_full = load_and_preprocess_data(train_path, "2014 TRAINING")
    y_train_full = df_train_full['loan_status'].astype(int)
    X_train_all = df_train_full.drop('loan_status', axis=1)
    
    # Load OOT test data (2015-2016)
    df_oot = load_and_preprocess_data(oot_path, "2015-2016 OOT")
    y_oot = df_oot['loan_status'].astype(int)
    X_oot_all = df_oot.drop('loan_status', axis=1)
    
    # Select features (same whitelist for both)
    print("Selecting application-time features (no pricing)...")
    X_train_full = select_model_features(X_train_all, include_pricing=False, verbose=False)
    X_oot = select_model_features(X_oot_all, include_pricing=False, verbose=False)
    
    # Ensure same columns (OOT may have different one-hot columns)
    train_cols = set(X_train_full.columns)
    oot_cols = set(X_oot.columns)
    
    missing_in_oot = train_cols - oot_cols
    extra_in_oot = oot_cols - train_cols
    
    if missing_in_oot:
        print(f"WARNING: {len(missing_in_oot)} columns missing in OOT data (adding as zeros)")
        for col in missing_in_oot:
            X_oot[col] = 0
    
    if extra_in_oot:
        print(f"WARNING: {len(extra_in_oot)} extra columns in OOT data (dropping)")
        X_oot = X_oot[[c for c in X_oot.columns if c in train_cols]]
    
    # Align column order
    X_oot = X_oot[X_train_full.columns]
    
    print(f"\nFinal feature count: {len(X_train_full.columns)}")
    print(f"Train set: {X_train_full.shape}")
    print(f"OOT set: {X_oot.shape}")
    
    # Split 2014 into train/test (for baseline in-sample metrics)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Train frozen pipeline on 2014 training data
    frozen_model = train_frozen_pipeline(X_train, y_train)
    
    # Evaluate on 2014 held-out test set (baseline)
    results_2014 = evaluate_on_test_set(frozen_model, X_test, y_test, set_name="2014 HELD-OUT")
    
    # Evaluate on OOT data (2015-2016) with FROZEN model
    results_oot = evaluate_on_test_set(frozen_model, X_oot, y_oot, set_name="2015-2016 OOT")
    
    # Population Stability Index (score distribution drift)
    psi = population_stability_index(
        np.array(results_2014['scores']),
        np.array(results_oot['scores']),
        n_bins=10
    )
    
    # Summary comparison
    print("\n" + "="*70)
    print("TEMPORAL STABILITY COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<30} {'2014 Held-Out':<20} {'2015-2016 OOT':<20} {'Drift':<15}")
    print("-" * 85)
    print(f"{'ROC-AUC':<30} {results_2014['credit_metrics']['roc_auc']:<20.4f} {results_oot['credit_metrics']['roc_auc']:<20.4f} {results_oot['credit_metrics']['roc_auc'] - results_2014['credit_metrics']['roc_auc']:>+.4f}")
    print(f"{'Gini':<30} {results_2014['credit_metrics']['gini']:<20.4f} {results_oot['credit_metrics']['gini']:<20.4f} {results_oot['credit_metrics']['gini'] - results_2014['credit_metrics']['gini']:>+.4f}")
    print(f"{'KS Statistic':<30} {results_2014['credit_metrics']['ks_statistic']:<20.4f} {results_oot['credit_metrics']['ks_statistic']:<20.4f} {results_oot['credit_metrics']['ks_statistic'] - results_2014['credit_metrics']['ks_statistic']:>+.4f}")
    print(f"{'Brier Score':<30} {results_2014['brier']:<20.4f} {results_oot['brier']:<20.4f} {results_oot['brier'] - results_2014['brier']:>+.4f}")
    print(f"{'Baseline DR':<30} {results_2014['credit_metrics']['baseline_dr']:<20.2%} {results_oot['credit_metrics']['baseline_dr']:<20.2%} {results_oot['credit_metrics']['baseline_dr'] - results_2014['credit_metrics']['baseline_dr']:>+.2%}")
    print(f"{'PSI (score distribution)':<30} {'—':<20} {psi:<20.4f} {'—':<15}")
    print("=" * 85)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if abs(results_oot['credit_metrics']['roc_auc'] - results_2014['credit_metrics']['roc_auc']) < 0.05:
        print("[OK] AUC drift < 0.05: Model discrimination is relatively stable")
    else:
        print("[WARNING] AUC drift >= 0.05: Significant temporal degradation detected")
    
    if psi < 0.10:
        print(f"[OK] PSI = {psi:.3f} < 0.10: No significant population shift")
    elif psi < 0.25:
        print(f"[MODERATE] PSI = {psi:.3f} in [0.10, 0.25): Moderate population shift, investigate")
    else:
        print(f"[HIGH] PSI = {psi:.3f} >= 0.25: Significant population shift, model may be stale")
    
    print("\n" + "="*70 + "\n")
    
    # Save results
    results = {
        '2014_held_out': results_2014,
        '2015_2016_oot': results_oot,
        'psi': float(psi),
        'drift': {
            'auc': float(results_oot['credit_metrics']['roc_auc'] - results_2014['credit_metrics']['roc_auc']),
            'gini': float(results_oot['credit_metrics']['gini'] - results_2014['credit_metrics']['gini']),
            'ks': float(results_oot['credit_metrics']['ks_statistic'] - results_2014['credit_metrics']['ks_statistic']),
            'brier': float(results_oot['brier'] - results_2014['brier']),
        },
    }
    
    output_path = Path("results_oot_validation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}\n")
    
    return results


if __name__ == '__main__':
    main()
