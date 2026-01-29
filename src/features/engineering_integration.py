"""
Data Integration and Merging Module for Feature Engineering.

This module handles data integration operations used in 00_clean_v0.ipynb:
- Date range creation
- DataFrame merging on date
- Forward fill operations
- CPI adjustments
- Rolling mean calculations
- Business day counting
- Frequency aggregation

Part of Phase 6.2 module split.

Module Architecture:
- competitive_features.py: Competitive rankings + WINK weighted mean
- engineering_integration.py: Data integration and merging (this file)
- engineering_temporal.py: Time-based feature engineering
- engineering_timeseries.py: Time series aggregation and creation
- engineering.py: Public API orchestrator

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

from typing import Dict, List

import pandas as pd


# =============================================================================
# DATE RANGE CREATION
# =============================================================================


def create_daily_date_range_dataframe(start_date: str, end_date: str) -> pd.DataFrame:
    """Create DataFrame with daily date range."""
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({'date': date_range})
    except Exception as e:
        raise ValueError(
            f"CRITICAL: Failed to create date range from {start_date} to {end_date}. "
            f"Business impact: Cannot establish temporal framework for time series analysis. "
            f"Error details: {e}. "
            f"Required action: Verify date format (YYYY-MM-DD) and ensure start_date <= end_date."
        )


# =============================================================================
# DATAFRAME MERGING
# =============================================================================


def merge_multiple_dataframes_on_date(base_df: pd.DataFrame, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple DataFrames on date column."""
    result = base_df.copy()

    for i, df in enumerate(dataframes):
        if 'date' not in df.columns:
            raise ValueError(
                f"CRITICAL: DataFrame {i} missing 'date' column for merge operation. "
                f"Business impact: Cannot integrate multiple data sources for unified time series. "
                f"Available columns in DataFrame {i}: {list(df.columns)}. "
                f"Required action: Ensure all input DataFrames have standardized 'date' column."
            )

        result = result.merge(df, on='date', how='left')

    return result


# =============================================================================
# FORWARD FILL OPERATIONS
# =============================================================================


def apply_forward_fill_to_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply forward fill to specified columns."""
    result = df.copy()

    for col in columns:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


# =============================================================================
# CPI ADJUSTMENTS
# =============================================================================


def apply_cpi_adjustment_to_sales(df: pd.DataFrame, cpi_column: str, sales_columns: List[str]) -> pd.DataFrame:
    """Apply CPI adjustment to sales columns."""
    result = df.copy()

    if cpi_column not in result.columns:
        raise ValueError(
            f"CRITICAL: CPI column '{cpi_column}' not found in DataFrame. "
            f"Business impact: Cannot apply inflation adjustment to sales data, revenue metrics will be distorted. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify CPI data extraction completed successfully and column naming matches configuration."
        )

    for col in sales_columns:
        if col in result.columns:
            result[col] = result[col] * result[cpi_column]

    return result


# =============================================================================
# ROLLING MEAN CALCULATIONS
# =============================================================================


def apply_rolling_mean_to_columns(df: pd.DataFrame, columns: List[str], window: int) -> pd.DataFrame:
    """Apply rolling mean to specified columns."""
    result = df.copy()

    for col in columns:
        if col in result.columns:
            result[col] = result[col].rolling(window=window, min_periods=1).mean()

    return result


# =============================================================================
# BUSINESS DAY COUNTING
# =============================================================================


def create_business_day_counter(df: pd.DataFrame, reference_column: str, counter_column: str) -> pd.DataFrame:
    """Create business day counter based on non-null reference column values."""
    result = df.copy()

    if reference_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Reference column '{reference_column}' not found in DataFrame. "
            f"Business impact: Cannot create business day counter for temporal analysis. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify reference column exists and matches expected sales data column."
        )

    result[counter_column] = (result[reference_column].notna() & (result[reference_column] > 0)).astype(int)

    return result


# =============================================================================
# FREQUENCY AGGREGATION
# =============================================================================


def aggregate_dataframe_by_frequency(
    df: pd.DataFrame,
    date_column: str,
    frequency: str,
    agg_dict: Dict[str, str]
) -> pd.DataFrame:
    """Aggregate DataFrame by time frequency with specified aggregation methods."""
    if date_column not in df.columns:
        raise ValueError(
            f"CRITICAL: Date column '{date_column}' not found in DataFrame. "
            f"Business impact: Cannot perform temporal aggregation, time series analysis will fail. "
            f"Available columns: {list(df.columns)}. "
            f"Required action: Verify date column exists with correct name in input DataFrame."
        )

    df_indexed = df.set_index(date_column)
    result = df_indexed.resample(frequency).agg(agg_dict).reset_index()

    return result


# =============================================================================
# DATA FILTERING
# =============================================================================


def filter_dataframe_by_mature_date(df: pd.DataFrame, date_column: str, cutoff_date: str) -> pd.DataFrame:
    """Filter DataFrame to only include dates before cutoff (mature data)."""
    if date_column not in df.columns:
        raise ValueError(
            f"CRITICAL: Date column '{date_column}' not found in DataFrame. "
            f"Business impact: Cannot filter to mature data, incomplete observations may corrupt analysis. "
            f"Available columns: {list(df.columns)}. "
            f"Required action: Verify date column exists and is properly named."
        )

    result = df[df[date_column] <= cutoff_date].copy()
    return result


def remove_final_row_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the final row from DataFrame."""
    if len(df) == 0:
        return df.copy()

    return df.iloc[:-1].copy()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Date range creation
    'create_daily_date_range_dataframe',
    # DataFrame merging
    'merge_multiple_dataframes_on_date',
    # Forward fill
    'apply_forward_fill_to_columns',
    # CPI adjustments
    'apply_cpi_adjustment_to_sales',
    # Rolling mean
    'apply_rolling_mean_to_columns',
    # Business day counting
    'create_business_day_counter',
    # Frequency aggregation
    'aggregate_dataframe_by_frequency',
    # Data filtering
    'filter_dataframe_by_mature_date',
    'remove_final_row_from_dataframe',
]
