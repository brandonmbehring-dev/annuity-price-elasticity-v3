"""
Temporal Feature Engineering Module.

This module handles time-based feature engineering operations:
- Temporal indicator columns (year, quarter, month)
- Day of year extraction
- Holiday indicators
- Lag feature creation
- Polynomial interactions

Part of Phase 6.2 module split.

Module Architecture:
- competitive_features.py: Competitive rankings + WINK weighted mean
- engineering_integration.py: Data integration and merging
- engineering_temporal.py: Time-based feature engineering (this file)
- engineering_timeseries.py: Time series aggregation and creation
- engineering.py: Public API orchestrator

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# TEMPORAL INDICATOR COLUMNS
# =============================================================================


def create_temporal_indicator_columns(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Create temporal indicator columns (year, quarter, month)."""
    result = df.copy()

    if date_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Date column '{date_column}' not found in DataFrame. "
            f"Business impact: Cannot create seasonal features for time series modeling. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify date column exists and is properly formatted as datetime."
        )

    result['year'] = result[date_column].dt.year
    result['quarter'] = result[date_column].dt.quarter
    result['month'] = result[date_column].dt.month

    return result


# =============================================================================
# DAY OF YEAR EXTRACTION
# =============================================================================


def extract_day_of_year_column(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    """Extract day of year from date column."""
    result = df.copy()

    if date_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Date column '{date_column}' not found in DataFrame. "
            f"Business impact: Cannot extract day of year for seasonal feature engineering. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify date column exists and is properly formatted as datetime."
        )

    result[target_column] = result[date_column].dt.dayofyear

    return result


# =============================================================================
# HOLIDAY INDICATORS
# =============================================================================


def create_holiday_indicator_by_day_range(
    df: pd.DataFrame,
    day_column: str,
    start_day: int,
    end_day: int,
    holiday_column: str
) -> pd.DataFrame:
    """Create holiday indicator based on day of year range.

    CORRECTED LOGIC: Creates standard holiday encoding where:
    - holiday=1 for holiday periods (outside start_day to end_day range)
    - holiday=0 for non-holiday periods (between start_day and end_day)

    This matches manual implementation and standard convention where
    df[df["holiday"] == 0] filters TO non-holiday periods for training.
    """
    result = df.copy()

    if day_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Day column '{day_column}' not found in DataFrame. "
            f"Business impact: Cannot create holiday indicator, seasonal patterns will not be captured. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify day of year column was created in previous pipeline step."
        )

    # CORRECTED: 1 for holiday periods (outside start_day to end_day), 0 for non-holiday periods
    # This matches the manual logic: holiday=1 for days < 13 or > 359
    result[holiday_column] = ((result[day_column] < start_day) | (result[day_column] > end_day)).astype(int)

    return result


# =============================================================================
# LAG FEATURES
# =============================================================================


def create_lag_features_for_columns(
    df: pd.DataFrame,
    lag_configs: List[Dict[str, Any]],
    max_lag_periods: int,
    allow_inplace: bool = False
) -> pd.DataFrame:
    """Create lag features for specified columns with configurable direction."""
    result = df.copy() if not allow_inplace else df

    # Collect all new columns to add at once (avoids DataFrame fragmentation)
    new_columns: Dict[str, pd.Series] = {}

    for config in lag_configs:
        source_col = config['source_col']
        prefix = config['prefix']
        lag_direction = config['lag_direction']

        if source_col not in result.columns:
            continue

        # Create current period feature
        new_columns[f"{prefix}_current"] = result[source_col].copy()

        # Create backward lags (t1, t2, t3, ...)
        if lag_direction in ['both', 'backward']:
            for lag in range(1, max_lag_periods + 1):
                new_columns[f"{prefix}_t{lag}"] = result[source_col].shift(lag)

        # Create forward lags (lead1, lead2, lead3, ...)
        if lag_direction in ['both', 'forward']:
            for lag in range(1, max_lag_periods + 1):
                new_columns[f"{prefix}_lead{lag}"] = result[source_col].shift(-lag)

    # Add all new columns at once using pd.concat to avoid fragmentation
    if new_columns:
        new_cols_df = pd.DataFrame(new_columns, index=result.index)
        result = pd.concat([result, new_cols_df], axis=1)

    return result


# =============================================================================
# POLYNOMIAL INTERACTIONS
# =============================================================================


def create_polynomial_interaction_features(
    df: pd.DataFrame,
    base_columns: List[str],
    max_lag_periods: int,
    allow_inplace: bool = False
) -> pd.DataFrame:
    """Create polynomial and interaction features from base columns."""
    result = df.copy() if not allow_inplace else df

    # Collect all new columns to add at once (avoids DataFrame fragmentation)
    new_columns: Dict[str, pd.Series] = {}

    # Create squared terms for each base column at all lag periods
    for base_col in base_columns:
        # Current period
        current_col = f"{base_col}_current"
        if current_col in result.columns:
            new_columns[f"derived_{base_col}_squared_current"] = result[current_col] ** 2

        # Lagged periods
        for lag in range(1, max_lag_periods + 1):
            lag_col = f"{base_col}_t{lag}"
            if lag_col in result.columns:
                new_columns[f"derived_{base_col}_squared_t{lag}"] = result[lag_col] ** 2

    # Create interaction terms between different base columns
    if len(base_columns) >= 2:
        for i, col1 in enumerate(base_columns):
            for j, col2 in enumerate(base_columns[i+1:], i+1):
                # Current period interaction
                current_col1 = f"{col1}_current"
                current_col2 = f"{col2}_current"
                if current_col1 in result.columns and current_col2 in result.columns:
                    new_columns[f"derived_{col1}_{col2}_interaction_current"] = result[current_col1] * result[current_col2]

                # Lagged period interactions
                for lag in range(1, max_lag_periods + 1):
                    lag_col1 = f"{col1}_t{lag}"
                    lag_col2 = f"{col2}_t{lag}"
                    if lag_col1 in result.columns and lag_col2 in result.columns:
                        new_columns[f"derived_{col1}_{col2}_interaction_t{lag}"] = result[lag_col1] * result[lag_col2]

    # Add all new columns at once using pd.concat to avoid fragmentation
    if new_columns:
        new_cols_df = pd.DataFrame(new_columns, index=result.index)
        result = pd.concat([result, new_cols_df], axis=1)

    return result


# =============================================================================
# LOG TRANSFORMATION
# =============================================================================


def apply_log_plus_one_transformation(df: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    """Apply log(1 + x) transformation to source column."""
    result = df.copy()

    if source_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Source column '{source_column}' not found in DataFrame. "
            f"Business impact: Cannot apply log transformation for model normalization. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify source column exists and contains positive numeric values."
        )

    result[target_column] = np.log1p(result[source_column])

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Temporal indicators
    'create_temporal_indicator_columns',
    # Day of year
    'extract_day_of_year_column',
    # Holiday indicators
    'create_holiday_indicator_by_day_range',
    # Lag features
    'create_lag_features_for_columns',
    # Polynomial interactions
    'create_polynomial_interaction_features',
    # Log transformation
    'apply_log_plus_one_transformation',
]
