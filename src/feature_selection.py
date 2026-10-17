"""
Feature selection helper for the credit underwriting notebook.

This module provides functions to select features based on the whitelist
defined in config/features.yaml, preventing target leakage.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.feature_guard import (
    get_allowed_features,
    validate_features_before_training,
    check_for_leakage,
)


def select_model_features(
    df: pd.DataFrame,
    include_pricing: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Select only allowed features from the dataframe for model training.
    
    This function:
    1. Gets the whitelist of allowed features (application-time only)
    2. Filters the dataframe to only include those features
    3. Selects only numeric columns (encoded categoricals are numeric after one-hot)
    4. Validates that no post-outcome features leaked through
    
    Args:
        df: DataFrame containing features (must NOT contain loan_status/target)
        include_pricing: If True, include grade/sub_grade/int_rate
        verbose: If True, print information about feature selection
    
    Returns:
        Filtered DataFrame with only allowed numeric features
    
    Raises:
        ValueError: If any post-outcome features are detected
    """
    # Get allowed feature names from config
    allowed_base = get_allowed_features(include_pricing=include_pricing)
    
    # Build set of allowed features including one-hot encoded versions
    # (e.g., if 'purpose' is allowed, then 'purpose_debt_consolidation' is too)
    allowed_set = set(allowed_base)
    
    # For each column in df, check if it matches an allowed base feature
    # or is a one-hot encoding of an allowed categorical
    # (column name like 'purpose_debt_consolidation' matches base 'purpose')
    allowed_columns = []
    for col in df.columns:
        # Direct match
        if col in allowed_set:
            allowed_columns.append(col)
            continue
        
        # Check if it's a one-hot encoded version
        # Split on underscore and check if any prefix is in allowed_base
        for base_feature in allowed_base:
            if col.startswith(base_feature + '_'):
                allowed_columns.append(col)
                break
    
    # Filter dataframe to only allowed columns
    df_filtered = df[allowed_columns].copy()
    
    # Keep only numeric columns (after one-hot encoding, all should be numeric)
    df_numeric = df_filtered.select_dtypes(include=[np.number]).copy()
    
    # Impute NaNs (SMOTE doesn't accept NaNs)
    df_numeric = df_numeric.fillna(df_numeric.median(numeric_only=True)).fillna(0)
    
    # Validate no leakage
    validate_features_before_training(df_numeric.columns.tolist(), raise_on_leak=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Feature Selection Summary (include_pricing={include_pricing})")
        print(f"{'='*70}")
        print(f"Original columns in df:     {len(df.columns)}")
        print(f"Allowed base features:      {len(allowed_base)}")
        print(f"Matched columns (inc. OH):  {len(allowed_columns)}")
        print(f"Final numeric features:     {len(df_numeric.columns)}")
        print(f"{'='*70}")
        
        # Show some example features
        sample_features = sorted(df_numeric.columns.tolist())[:20]
        print(f"Example features (first 20): {', '.join(sample_features)}")
        print(f"{'='*70}\n")
    
    return df_numeric


def validate_no_leakage_in_dataframe(df: pd.DataFrame) -> None:
    """
    Check that the dataframe doesn't contain any post-outcome features.
    
    Raises ValueError if leakage is detected.
    """
    result = check_for_leakage(df.columns.tolist())
    leaked = result["leaked_features"]
    
    if leaked:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED in dataframe!\n"
            f"Found {len(leaked)} post-outcome features:\n"
            + "\n".join(f"  - {f}" for f in sorted(leaked))
            + "\n\nThese columns should have been removed during preprocessing."
        )


def get_feature_summary(df: pd.DataFrame, include_pricing: bool = False) -> pd.DataFrame:
    """
    Get a summary of features in the dataframe compared to the whitelist.
    
    Returns a DataFrame with columns: feature, status (allowed/leaky/unknown)
    """
    allowed_base = set(get_allowed_features(include_pricing=include_pricing))
    blocklist = set(get_allowed_features(include_pricing=False))  # Get blocklist via import
    from src.feature_guard import get_blocklist
    blocklist = set(get_blocklist())
    
    rows = []
    for col in df.columns:
        if col in blocklist:
            status = "LEAKY (post-outcome)"
        elif col in allowed_base:
            status = "allowed"
        else:
            # Check if it's a one-hot encoded version
            is_oh_encoded = False
            for base_feature in allowed_base:
                if col.startswith(base_feature + '_'):
                    status = "allowed (one-hot)"
                    is_oh_encoded = True
                    break
            
            if not is_oh_encoded:
                status = "unknown"
        
        rows.append({"feature": col, "status": status})
    
    return pd.DataFrame(rows)
