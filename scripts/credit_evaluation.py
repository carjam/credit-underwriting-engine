"""
Comprehensive credit-focused model evaluation.

Generates full evaluation report with credit-specific metrics,
decile tables, and threshold analysis for documentation.
"""
import os
import sys
from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
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
    credit_metrics_summary,
)


def load_and_preprocess_data():
    """Load and minimally preprocess data."""
    data_path = Path("data/loans.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    print(f"Loaded {len(df)} rows")
    
    # Filter to binary loan_status
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    print(f"After filtering to binary outcomes: {len(df)} rows")
    
    # Map loan_status to binary (1 = good, 0 = bad)
    df['loan_status'] = (df['loan_status'] == 'Fully Paid').astype(int)
    
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
    
    return df


def train_and_evaluate_full(X, y, include_pricing=False):
    """Train model and return comprehensive credit metrics."""
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE CREDIT EVALUATION (include_pricing={include_pricing})")
    print(f"{'='*70}")
    
    # Split BEFORE SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Test set class balance: {pd.Series(y_test).value_counts(normalize=True).round(3).to_dict()}")
    
    # Apply SMOTE to training set only
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
    
    # Evaluate on test set
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # === CREDIT-SPECIFIC METRICS ===
    credit_metrics = credit_metrics_summary(y_test, y_proba)
    
    # Classification metrics (for reference)
    accuracy = accuracy_score(y_test, y_pred)
    precision_paid = precision_score(y_test, y_pred, pos_label=1)
    recall_paid = recall_score(y_test, y_pred, pos_label=1)
    recall_charged_off = recall_score(y_test, y_pred, pos_label=0)
    brier = brier_score_loss(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Score distribution
    high_score = (y_proba >= 0.70).sum()
    mid_score = ((y_proba >= 0.50) & (y_proba < 0.70)).sum()
    low_score = (y_proba < 0.50).sum()
    
    # Default rate in high-score band
    approved_70 = y_proba >= 0.70
    dr_70 = (y_test[approved_70] == 0).mean() if approved_70.sum() > 0 else float('nan')
    
    # Decile analysis
    decile_table = score_deciles(y_test, y_proba, n_bins=10)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    
    print(f"\n{'='*70}")
    print(f"RESULTS (include_pricing={include_pricing})")
    print(f"{'='*70}")
    print(f"\n--- CREDIT-SPECIFIC METRICS (Primary) ---")
    print(f"ROC-AUC:                 {credit_metrics['roc_auc']:.4f}")
    print(f"Gini Coefficient:        {credit_metrics['gini']:.4f}  (2*AUC - 1)")
    print(f"KS Statistic:            {credit_metrics['ks_statistic']:.4f}")
    print(f"PR-AUC (Charged Off):    {credit_metrics['pr_auc_minority']:.4f}  (baseline={credit_metrics['baseline_dr']:.3f})")
    print(f"Brier Score:             {brier:.4f}")
    
    print(f"\n--- SCORE DISTRIBUTION ---")
    print(f"High score (>=0.70):     {high_score} loans ({high_score/len(y_test)*100:.1f}%)")
    print(f"Mid score (0.50-0.70):   {mid_score} loans ({mid_score/len(y_test)*100:.1f}%)")
    print(f"Low score (<0.50):       {low_score} loans ({low_score/len(y_test)*100:.1f}%)")
    print(f"Default Rate (p>=0.70):  {dr_70:.2%}")
    
    print(f"\n--- CLASSIFICATION METRICS (Secondary, for reference) ---")
    print(f"Accuracy (0.5 cutoff):   {accuracy:.4f}  (baseline={credit_metrics['baseline_dr']:.3f} if approve all)")
    print(f"Precision (Fully Paid):  {precision_paid:.4f}")
    print(f"Recall (Fully Paid):     {recall_paid:.4f}")
    print(f"Recall (Charged Off):    {recall_charged_off:.4f}")
    print(f"\nConfusion Matrix (0.5 cutoff):")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    
    print(f"\n--- DECILE ANALYSIS ---")
    print(decile_table.to_string(index=False))
    
    print(f"\n{'='*70}\n")
    
    return {
        'credit_metrics': credit_metrics,
        'classification_metrics': {
            'accuracy': float(accuracy),
            'precision_paid': float(precision_paid),
            'recall_paid': float(recall_paid),
            'recall_charged_off': float(recall_charged_off),
            'brier': float(brier),
        },
        'score_distribution': {
            'high_score_count': int(high_score),
            'mid_score_count': int(mid_score),
            'low_score_count': int(low_score),
            'dr_approved_70': float(dr_70),
        },
        'decile_table': decile_table.to_dict('records'),
        'calibration': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        },
        'n_features': X.shape[1],
    }


def main():
    """Run comprehensive evaluation and save results."""
    print("="*70)
    print("COMPREHENSIVE CREDIT-FOCUSED MODEL EVALUATION")
    print("="*70)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Extract target
    y = df['loan_status'].astype(int)
    X_all = df.drop('loan_status', axis=1)
    
    print(f"\nTotal features before selection: {len(X_all.columns)}")
    
    # Experiment 1: Without pricing features
    print("\n" + "="*70)
    print("EXPERIMENT 1: Application-time features only (NO pricing)")
    print("="*70)
    X_no_pricing = select_model_features(X_all, include_pricing=False, verbose=True)
    results_no_pricing = train_and_evaluate_full(X_no_pricing, y, include_pricing=False)
    
    # Experiment 2: With pricing features
    print("\n" + "="*70)
    print("EXPERIMENT 2: Application-time features + pricing (grade/sub_grade/int_rate)")
    print("="*70)
    X_with_pricing = select_model_features(X_all, include_pricing=True, verbose=True)
    results_with_pricing = train_and_evaluate_full(X_with_pricing, y, include_pricing=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Credit-Specific Metrics Comparison")
    print("="*70)
    print(f"\n{'Metric':<30} {'No Pricing':<15} {'With Pricing':<15}")
    print("-" * 70)
    print(f"{'ROC-AUC':<30} {results_no_pricing['credit_metrics']['roc_auc']:<15.4f} {results_with_pricing['credit_metrics']['roc_auc']:<15.4f}")
    print(f"{'Gini':<30} {results_no_pricing['credit_metrics']['gini']:<15.4f} {results_with_pricing['credit_metrics']['gini']:<15.4f}")
    print(f"{'KS Statistic':<30} {results_no_pricing['credit_metrics']['ks_statistic']:<15.4f} {results_with_pricing['credit_metrics']['ks_statistic']:<15.4f}")
    print(f"{'PR-AUC (Charged Off)':<30} {results_no_pricing['credit_metrics']['pr_auc_minority']:<15.4f} {results_with_pricing['credit_metrics']['pr_auc_minority']:<15.4f}")
    print(f"{'Brier':<30} {results_no_pricing['classification_metrics']['brier']:<15.4f} {results_with_pricing['classification_metrics']['brier']:<15.4f}")
    print("=" * 70)
    
    # Save results
    results = {
        'no_pricing': results_no_pricing,
        'with_pricing': results_with_pricing,
    }
    
    output_path = Path("results_credit_evaluation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
