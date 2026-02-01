"""
Shared Data Validation Utilities for Notebooks.

Provides standardized validation functions for sales and rates data
that appear identically across product notebooks.

Design Notes:
    - Validation logic is shared to ensure consistency
    - Warning messages are educational (explain business impact)
    - Critical errors use fail-fast pattern
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation checks."""

    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def print_summary(self, name: str = "Data") -> None:
        """Print validation summary to console."""
        if self.errors:
            print(f"{name} validation FAILED:")
            for error in self.errors:
                print(f"  {error}")
        elif self.warnings:
            print(f"{name} validation warnings:")
            for warning in self.warnings:
                print(f"  Warning: {warning}")
        else:
            print(f"{name} input validation passed: All business rules satisfied")

        # Print stats
        if self.stats:
            print(f"\nValidated {name.lower()}:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")


def validate_sales_data(
    df: pd.DataFrame,
    model_config: Dict[str, Any],
    expected_record_range: tuple = (150, 200),
    rate_upper_bound: float = 0.20,
    min_date_range_days: int = 1000,
) -> ValidationResult:
    """
    Validate sales DataFrame for model training.

    Performs comprehensive business rule validation:
    1. Empty check (critical)
    2. Record count (warning)
    3. Required columns (critical)
    4. Null values in critical columns (critical)
    5. Positive sales values (critical)
    6. Date range coverage (warning)
    7. Feature value ranges (warning)

    Parameters
    ----------
    df : pd.DataFrame
        Sales DataFrame to validate.
    model_config : dict
        Model configuration containing features and target_variable.
    expected_record_range : tuple
        (min, max) expected record count.
    rate_upper_bound : float
        Maximum expected rate value (0.20 = 20%).
    min_date_range_days : int
        Minimum expected date range in days.

    Returns
    -------
    ValidationResult
        Dataclass with passed, warnings, errors, and stats.

    Raises
    ------
    ValueError
        If critical validation fails (empty data, missing columns, nulls).
    """
    result = ValidationResult(passed=True)

    # 1. Empty check (CRITICAL)
    if df.empty:
        raise ValueError(
            "CRITICAL: Sales dataset is empty after filtering. "
            "Business impact: Cannot proceed with price elasticity analysis. "
            "Required action: Verify final_dataset.parquet exists and contains data."
        )

    # 2. Record count check (WARNING)
    min_records, max_records = expected_record_range
    if not (min_records <= len(df) <= max_records):
        result.warnings.append(
            f"Sales record count {len(df)} outside expected range ({min_records}-{max_records}). "
            f"Expected ~{(min_records + max_records) // 2} records for historical analysis."
        )

    # 3. Required columns check (CRITICAL)
    required_cols = ["date", "sales"] + model_config["features"] + [model_config["target_variable"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CRITICAL: Missing required columns: {missing_cols}. "
            f"Business impact: Model training cannot proceed without these features. "
            f"Required action: Verify data pipeline output includes all model features."
        )

    # 4. Null values check (CRITICAL)
    for col in required_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"CRITICAL: Column '{col}' contains {null_count} null values. "
                f"Business impact: Bootstrap ensemble training will fail with missing data. "
                f"Required action: Verify data pipeline completeness for all lag features."
            )

    # 5. Positive sales check (CRITICAL)
    if (df["sales"] < 0).any():
        raise ValueError(
            "CRITICAL: Negative sales values detected. "
            "Business impact: Price elasticity predictions will be invalid. "
            "Required action: Verify data pipeline sales calculations."
        )

    # 6. Date range check (WARNING)
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range_days = (max_date - min_date).days

    if date_range_days < min_date_range_days:
        result.warnings.append(
            f"Date range only {date_range_days} days (expected ~{min_date_range_days}+ days). "
            f"Limited historical data may affect model quality."
        )

    # 7. Feature value ranges check (WARNING)
    rate_features = [f for f in required_cols if "rate" in f.lower() or "competitor" in f.lower()]
    for feat in rate_features:
        if feat in df.columns:
            min_val = df[feat].min()
            max_val = df[feat].max()
            if min_val < 0 or max_val > rate_upper_bound:
                result.warnings.append(
                    f"Feature '{feat}' has unusual range "
                    f"[{min_val:.2%}, {max_val:.2%}] (expected 0-{rate_upper_bound:.0%})"
                )

    # Collect stats
    result.stats = {
        "Records": f"{len(df):,}",
        "Columns": df.shape[1],
        "Date range": f"{min_date} to {max_date} ({date_range_days} days)",
        "Features validated": f"{len(required_cols)} required columns",
    }

    return result


def validate_rates_data(
    df: pd.DataFrame,
    expected_record_range: tuple = (2500, 3000),
    rate_upper_bound: float = 0.20,
    min_date_range_days: int = 2500,
) -> ValidationResult:
    """
    Validate WINK competitive rates DataFrame.

    Performs comprehensive validation:
    1. Empty check (critical)
    2. Record count (warning)
    3. Required rate columns (critical)
    4. C_weighted_mean completeness (critical)
    5. Rate ranges (warning)
    6. Date range coverage (warning)

    Parameters
    ----------
    df : pd.DataFrame
        WINK rates DataFrame to validate.
    expected_record_range : tuple
        (min, max) expected record count.
    rate_upper_bound : float
        Maximum expected rate value (0.20 = 20%).
    min_date_range_days : int
        Minimum expected date range in days.

    Returns
    -------
    ValidationResult
        Dataclass with passed, warnings, errors, and stats.

    Raises
    ------
    ValueError
        If critical validation fails.
    """
    result = ValidationResult(passed=True)

    # 1. Empty check (CRITICAL)
    if df.empty:
        raise ValueError(
            "CRITICAL: WINK competitive rates dataset is empty. "
            "Business impact: Cannot proceed with competitive analysis and rate scenarios. "
            "Required action: Verify WINK_competitive_rates.parquet exists and contains data."
        )

    # 2. Record count check (WARNING)
    min_records, max_records = expected_record_range
    if not (min_records <= len(df) <= max_records):
        result.warnings.append(
            f"WINK record count {len(df)} outside expected range ({min_records:,}-{max_records:,}). "
            f"Expected ~{(min_records + max_records) // 2:,} records for competitive rate history."
        )

    # 3. Required columns check (CRITICAL)
    required_rate_cols = ["date", "Prudential", "C_weighted_mean", "C_core"]
    missing_cols = [col for col in required_rate_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CRITICAL: Missing required rate columns: {missing_cols}. "
            f"Business impact: Competitive analysis and rate scenarios cannot be generated. "
            f"Required action: Verify data pipeline includes market share weighting step."
        )

    # 4. C_weighted_mean completeness (CRITICAL)
    null_count = df["C_weighted_mean"].isna().sum()
    if null_count > 0:
        raise ValueError(
            f"CRITICAL: C_weighted_mean contains {null_count} null values. "
            f"Business impact: Incomplete competitive rate coverage will affect scenario analysis. "
            f"Required action: Verify market share weights have complete date coverage."
        )

    # 5. Rate ranges check (WARNING)
    rate_cols = ["Prudential", "C_weighted_mean", "C_core"]
    for col in rate_cols:
        if col in df.columns:
            min_rate = df[col].min()
            max_rate = df[col].max()
            if min_rate < 0 or max_rate > rate_upper_bound:
                result.warnings.append(
                    f"{col} has unusual range "
                    f"[{min_rate:.2%}, {max_rate:.2%}] (expected 0-{rate_upper_bound:.0%})"
                )

    # 6. Date range check (WARNING)
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range_days = (max_date - min_date).days

    if date_range_days < min_date_range_days:
        result.warnings.append(
            f"Date range only {date_range_days} days (expected ~{min_date_range_days}+ days). "
            f"Limited competitive rate history may affect analysis quality."
        )

    # Collect stats
    result.stats = {
        "Records": f"{len(df):,}",
        "Columns": df.shape[1],
        "Date range": f"{min_date} to {max_date} ({date_range_days} days)",
        "C_weighted_mean range": (
            f"[{df['C_weighted_mean'].min():.2%}, {df['C_weighted_mean'].max():.2%}]"
        ),
        "Latest Prudential rate": f"{df['Prudential'].iloc[-1]:.2f}%",
        "Latest competitive rate": f"{df['C_weighted_mean'].iloc[-1]:.2f}%",
    }

    return result
