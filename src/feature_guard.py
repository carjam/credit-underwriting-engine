"""
Feature guard to prevent target leakage in model training.

Enforces a whitelist of application-time features and blocks post-outcome features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_feature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load feature configuration from YAML file."""
    if config_path is None:
        # Default to config/features.yaml relative to this file's parent directory
        config_path = Path(__file__).parent.parent / "config" / "features.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_allowed_features(
    include_pricing: bool = False,
    config_path: str | Path | None = None,
) -> list[str]:
    """
    Get list of allowed feature names for model training.
    
    Args:
        include_pricing: If True, include grade/sub_grade/int_rate (Lending Club's
            own risk assessment). If False, exclude them to test pure application data.
        config_path: Path to features.yaml config file (defaults to config/features.yaml)
    
    Returns:
        List of allowed feature column names
    """
    config = load_feature_config(config_path)
    allowed = config.get("application_time_features", [])
    
    if include_pricing:
        allowed = allowed + config.get("pricing_features", [])
    
    return allowed


def get_blocklist() -> list[str]:
    """Get list of post-outcome features that must never reach the model."""
    config = load_feature_config()
    return config.get("post_outcome_blocklist", [])


def check_for_leakage(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Check if any post-outcome features are present in the feature list.
    
    Args:
        feature_names: List of feature column names to check
    
    Returns:
        Dict with 'leaked_features' key containing list of blocklisted features found.
        Empty list means no leakage detected.
    """
    blocklist = set(get_blocklist())
    leaked = [f for f in feature_names if f in blocklist]
    return {"leaked_features": leaked}


def filter_to_allowed_features(
    df_columns: list[str],
    include_pricing: bool = False,
    config_path: str | Path | None = None,
) -> list[str]:
    """
    Filter a list of column names to only allowed features.
    
    Returns the intersection of df_columns and the whitelist, preserving order.
    """
    allowed_set = set(get_allowed_features(include_pricing=include_pricing, config_path=config_path))
    return [col for col in df_columns if col in allowed_set]


def validate_features_before_training(
    feature_names: list[str],
    raise_on_leak: bool = True,
) -> None:
    """
    Validate that no post-outcome features are present before model training.
    
    Args:
        feature_names: List of feature column names about to be used for training
        raise_on_leak: If True, raise ValueError when leakage is detected.
            If False, just return (for inspection/testing).
    
    Raises:
        ValueError: If any post-outcome features are detected and raise_on_leak=True
    """
    result = check_for_leakage(feature_names)
    leaked = result["leaked_features"]
    
    if leaked and raise_on_leak:
        raise ValueError(
            f"TARGET LEAKAGE DETECTED: {len(leaked)} post-outcome features found in training data:\n"
            + "\n".join(f"  - {f}" for f in sorted(leaked))
            + "\n\nThese features are only known AFTER loan outcome and must be removed."
        )
