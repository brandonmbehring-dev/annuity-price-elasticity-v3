"""
Unit Tests for Temporal Feature Engineering Module
==================================================

Tests for src/features/engineering_temporal.py covering:
- Temporal indicator columns (year, quarter, month)
- Day of year extraction
- Holiday indicators with corrected logic
- Lag feature creation (backward, forward, both)
- Polynomial interaction features
- Log transformation

Target: 85% coverage (218/257 lines)

Test Pattern:
- Follow pytest fixture pattern from tests/conftest.py
- Test basic functionality with simple data
- Test edge cases (missing columns, NaN, empty DataFrames)
- Verify immutability (original DataFrames unchanged)
- Test error handling and error messages

Author: Claude Code
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.engineering_temporal import (
    create_temporal_indicator_columns,
    extract_day_of_year_column,
    create_holiday_indicator_by_day_range,
    create_lag_features_for_columns,
    create_polynomial_interaction_features,
    apply_log_plus_one_transformation,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_temporal_data():
    """Simple DataFrame with date column for temporal feature testing."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    return pd.DataFrame({
        'date': dates,
        'value': range(365)
    })


@pytest.fixture
def lag_features_data():
    """Simple DataFrame for lag feature testing."""
    return pd.DataFrame({
        'sales': [100, 110, 120, 130, 140],
        'price': [5.0, 5.5, 6.0, 5.5, 5.0],
        'quantity': [20, 20, 20, 24, 28]
    })


@pytest.fixture
def polynomial_data():
    """DataFrame with lag columns for polynomial interaction testing."""
    return pd.DataFrame({
        'rate_current': [4.5, 4.6, 4.7],
        'rate_t1': [4.4, 4.5, 4.6],
        'rate_t2': [4.3, 4.4, 4.5],
        'median_current': [3.8, 3.9, 4.0],
        'median_t1': [3.7, 3.8, 3.9],
        'median_t2': [3.6, 3.7, 3.8],
    })


# =============================================================================
# TEMPORAL INDICATOR COLUMNS TESTS
# =============================================================================


def test_create_temporal_indicator_columns_basic(simple_temporal_data):
    """Test basic temporal indicator creation (year, quarter, month)."""
    result = create_temporal_indicator_columns(simple_temporal_data, 'date')

    # Should have new columns
    assert 'year' in result.columns
    assert 'quarter' in result.columns
    assert 'month' in result.columns

    # First row should be 2023-01-01
    assert result['year'].iloc[0] == 2023
    assert result['quarter'].iloc[0] == 1
    assert result['month'].iloc[0] == 1

    # Last row should be 2023-12-31
    assert result['year'].iloc[-1] == 2023
    assert result['quarter'].iloc[-1] == 4
    assert result['month'].iloc[-1] == 12


def test_create_temporal_indicator_columns_multiple_years():
    """Test temporal indicators across multiple years."""
    dates = pd.date_range('2022-06-01', '2024-06-01', freq='MS')  # Monthly start
    df = pd.DataFrame({'date': dates, 'value': range(len(dates))})

    result = create_temporal_indicator_columns(df, 'date')

    # Should span multiple years
    assert result['year'].min() == 2022
    assert result['year'].max() == 2024

    # Check quarters are correct
    assert 1 in result['quarter'].values
    assert 2 in result['quarter'].values
    assert 3 in result['quarter'].values
    assert 4 in result['quarter'].values


def test_create_temporal_indicator_columns_missing_date_column():
    """Test error when date column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})

    with pytest.raises(ValueError, match="Date column 'date' not found"):
        create_temporal_indicator_columns(df, 'date')


def test_create_temporal_indicator_columns_immutable(simple_temporal_data):
    """Test that original DataFrame is not modified."""
    original_cols = set(simple_temporal_data.columns)
    _ = create_temporal_indicator_columns(simple_temporal_data, 'date')

    # Original should be unchanged
    assert set(simple_temporal_data.columns) == original_cols
    assert 'year' not in simple_temporal_data.columns


# =============================================================================
# DAY OF YEAR EXTRACTION TESTS
# =============================================================================


def test_extract_day_of_year_basic(simple_temporal_data):
    """Test basic day of year extraction."""
    result = extract_day_of_year_column(simple_temporal_data, 'date', 'day_of_year')

    assert 'day_of_year' in result.columns

    # First day of year should be 1
    assert result['day_of_year'].iloc[0] == 1

    # Last day of year should be 365 (2023 is not a leap year)
    assert result['day_of_year'].iloc[-1] == 365


def test_extract_day_of_year_leap_year():
    """Test day of year extraction in leap year."""
    dates = pd.date_range('2024-01-01', periods=366, freq='D')  # 2024 is leap year
    df = pd.DataFrame({'date': dates, 'value': range(366)})

    result = extract_day_of_year_column(df, 'date', 'doy')

    # Should go up to 366
    assert result['doy'].max() == 366

    # Feb 29 should be day 60
    feb_29_idx = (df['date'].dt.month == 2) & (df['date'].dt.day == 29)
    assert result.loc[feb_29_idx, 'doy'].iloc[0] == 60


def test_extract_day_of_year_missing_date_column():
    """Test error when date column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})

    with pytest.raises(ValueError, match="Date column 'date' not found"):
        extract_day_of_year_column(df, 'date', 'doy')


def test_extract_day_of_year_immutable(simple_temporal_data):
    """Test that original DataFrame is not modified."""
    original_cols = set(simple_temporal_data.columns)
    _ = extract_day_of_year_column(simple_temporal_data, 'date', 'day_of_year')

    assert set(simple_temporal_data.columns) == original_cols


# =============================================================================
# HOLIDAY INDICATOR TESTS
# =============================================================================


def test_create_holiday_indicator_basic():
    """Test basic holiday indicator creation.

    CORRECTED LOGIC: holiday=1 for periods outside start_day to end_day range.
    """
    df = pd.DataFrame({'day_of_year': [1, 10, 15, 100, 360, 365]})

    # Non-holiday period: days 13-359
    # Holiday period: days 1-12 and 360-365
    result = create_holiday_indicator_by_day_range(
        df, 'day_of_year', start_day=13, end_day=359, holiday_column='holiday'
    )

    assert 'holiday' in result.columns

    # Days 1, 10 should be holiday (< 13)
    assert result['holiday'].iloc[0] == 1  # Day 1
    assert result['holiday'].iloc[1] == 1  # Day 10

    # Days 15, 100 should be non-holiday (between 13 and 359)
    assert result['holiday'].iloc[2] == 0  # Day 15
    assert result['holiday'].iloc[3] == 0  # Day 100

    # Days 360, 365 should be holiday (> 359)
    assert result['holiday'].iloc[4] == 1  # Day 360
    assert result['holiday'].iloc[5] == 1  # Day 365


def test_create_holiday_indicator_all_holiday():
    """Test when all days are holidays."""
    df = pd.DataFrame({'doy': [1, 2, 3, 364, 365]})

    # Very narrow non-holiday window (day 180 only)
    result = create_holiday_indicator_by_day_range(
        df, 'doy', start_day=180, end_day=180, holiday_column='is_holiday'
    )

    # All days should be holiday (not in range 180-180)
    assert (result['is_holiday'] == 1).all()


def test_create_holiday_indicator_all_non_holiday():
    """Test when all days are non-holidays."""
    df = pd.DataFrame({'doy': [50, 100, 150, 200, 250]})

    # Wide non-holiday window (1-365)
    result = create_holiday_indicator_by_day_range(
        df, 'doy', start_day=1, end_day=365, holiday_column='is_holiday'
    )

    # All days should be non-holiday (in range 1-365)
    assert (result['is_holiday'] == 0).all()


def test_create_holiday_indicator_missing_day_column():
    """Test error when day column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})

    with pytest.raises(ValueError, match="Day column 'doy' not found"):
        create_holiday_indicator_by_day_range(df, 'doy', 13, 359, 'holiday')


def test_create_holiday_indicator_immutable():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({'doy': [1, 100, 365]})
    original_cols = set(df.columns)

    _ = create_holiday_indicator_by_day_range(df, 'doy', 13, 359, 'holiday')

    assert set(df.columns) == original_cols


# =============================================================================
# LAG FEATURES TESTS
# =============================================================================


def test_create_lag_features_backward_only(lag_features_data):
    """Test backward lag creation (t1, t2, t3, ...)."""
    lag_configs = [
        {'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'backward'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    # Should have current and backward lags
    assert 'sales_current' in result.columns
    assert 'sales_t1' in result.columns
    assert 'sales_t2' in result.columns

    # Should NOT have forward lags
    assert 'sales_lead1' not in result.columns

    # Verify lag values
    assert result['sales_current'].iloc[0] == 100
    assert pd.isna(result['sales_t1'].iloc[0])  # First row has no lag-1
    assert result['sales_t1'].iloc[1] == 100  # Second row's t1 = first row's current


def test_create_lag_features_forward_only(lag_features_data):
    """Test forward lag creation (lead1, lead2, lead3, ...)."""
    lag_configs = [
        {'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'forward'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    # Should have current and forward lags
    assert 'sales_current' in result.columns
    assert 'sales_lead1' in result.columns
    assert 'sales_lead2' in result.columns

    # Should NOT have backward lags
    assert 'sales_t1' not in result.columns

    # Verify lead values
    assert result['sales_lead1'].iloc[0] == 110  # First row's lead1 = second row's value
    assert pd.isna(result['sales_lead1'].iloc[-1])  # Last row has no lead


def test_create_lag_features_both_directions(lag_features_data):
    """Test both backward and forward lags."""
    lag_configs = [
        {'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'both'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    # Should have all lag types
    assert 'sales_current' in result.columns
    assert 'sales_t1' in result.columns
    assert 'sales_t2' in result.columns
    assert 'sales_lead1' in result.columns
    assert 'sales_lead2' in result.columns


def test_create_lag_features_multiple_columns(lag_features_data):
    """Test lag creation for multiple columns."""
    lag_configs = [
        {'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'backward'},
        {'source_col': 'price', 'prefix': 'price', 'lag_direction': 'backward'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=1)

    # Should have lags for both columns
    assert 'sales_current' in result.columns
    assert 'sales_t1' in result.columns
    assert 'price_current' in result.columns
    assert 'price_t1' in result.columns


def test_create_lag_features_missing_column_skipped(lag_features_data):
    """Test that missing source columns are gracefully skipped."""
    lag_configs = [
        {'source_col': 'nonexistent', 'prefix': 'ne', 'lag_direction': 'backward'}
    ]

    result = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    # Should not create any lag columns for missing source
    assert 'ne_current' not in result.columns
    assert 'ne_t1' not in result.columns

    # Original columns should still exist
    assert 'sales' in result.columns


def test_create_lag_features_max_lag_periods():
    """Test max_lag_periods parameter controls number of lags."""
    df = pd.DataFrame({'value': range(10)})
    lag_configs = [{'source_col': 'value', 'prefix': 'v', 'lag_direction': 'backward'}]

    result = create_lag_features_for_columns(df, lag_configs, max_lag_periods=5)

    # Should have lags up to t5
    assert 'v_t1' in result.columns
    assert 'v_t5' in result.columns
    assert 'v_t6' not in result.columns


def test_create_lag_features_immutable(lag_features_data):
    """Test that original DataFrame is not modified."""
    original_cols = set(lag_features_data.columns)
    lag_configs = [{'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'backward'}]

    _ = create_lag_features_for_columns(lag_features_data, lag_configs, max_lag_periods=2)

    assert set(lag_features_data.columns) == original_cols


def test_create_lag_features_allow_inplace():
    """Test allow_inplace parameter avoids copy (performance optimization).

    Note: pd.concat at the end still returns a new DataFrame, but allow_inplace=True
    avoids the initial df.copy() operation for performance.
    """
    df = pd.DataFrame({'value': [1, 2, 3]})
    lag_configs = [{'source_col': 'value', 'prefix': 'v', 'lag_direction': 'backward'}]

    result = create_lag_features_for_columns(df, lag_configs, max_lag_periods=1, allow_inplace=True)

    # Result should have the lag columns
    assert 'v_current' in result.columns
    assert 'v_t1' in result.columns


# =============================================================================
# POLYNOMIAL INTERACTION FEATURES TESTS
# =============================================================================


def test_create_polynomial_squared_terms(polynomial_data):
    """Test polynomial squared term creation."""
    base_columns = ['rate', 'median']

    result = create_polynomial_interaction_features(
        polynomial_data, base_columns, max_lag_periods=2
    )

    # Should have squared terms for current and lags
    assert 'derived_rate_squared_current' in result.columns
    assert 'derived_rate_squared_t1' in result.columns
    assert 'derived_rate_squared_t2' in result.columns
    assert 'derived_median_squared_current' in result.columns

    # Verify squared values
    expected_squared = polynomial_data['rate_current'].iloc[0] ** 2
    assert np.isclose(result['derived_rate_squared_current'].iloc[0], expected_squared)


def test_create_polynomial_interaction_terms(polynomial_data):
    """Test polynomial interaction term creation."""
    base_columns = ['rate', 'median']

    result = create_polynomial_interaction_features(
        polynomial_data, base_columns, max_lag_periods=2
    )

    # Should have interaction terms
    assert 'derived_rate_median_interaction_current' in result.columns
    assert 'derived_rate_median_interaction_t1' in result.columns
    assert 'derived_rate_median_interaction_t2' in result.columns

    # Verify interaction values
    expected_interaction = polynomial_data['rate_current'].iloc[0] * polynomial_data['median_current'].iloc[0]
    assert np.isclose(result['derived_rate_median_interaction_current'].iloc[0], expected_interaction)


def test_create_polynomial_single_column_no_interactions():
    """Test that single column only creates squared terms, no interactions."""
    df = pd.DataFrame({
        'rate_current': [4.5, 4.6],
        'rate_t1': [4.4, 4.5]
    })

    result = create_polynomial_interaction_features(df, ['rate'], max_lag_periods=1)

    # Should have squared terms
    assert 'derived_rate_squared_current' in result.columns
    assert 'derived_rate_squared_t1' in result.columns

    # Should NOT have interaction terms (need at least 2 base columns)
    interaction_cols = [col for col in result.columns if 'interaction' in col]
    assert len(interaction_cols) == 0


def test_create_polynomial_three_columns_multiple_interactions():
    """Test interaction terms with 3 base columns (should create 3 pairs)."""
    df = pd.DataFrame({
        'a_current': [1, 2],
        'b_current': [3, 4],
        'c_current': [5, 6]
    })

    result = create_polynomial_interaction_features(df, ['a', 'b', 'c'], max_lag_periods=0)

    # Should have 3 interaction pairs: a*b, a*c, b*c
    assert 'derived_a_b_interaction_current' in result.columns
    assert 'derived_a_c_interaction_current' in result.columns
    assert 'derived_b_c_interaction_current' in result.columns

    # Verify one interaction
    assert result['derived_a_b_interaction_current'].iloc[0] == 1 * 3


def test_create_polynomial_missing_lag_columns_skipped(polynomial_data):
    """Test that missing lag columns are gracefully skipped."""
    # Remove some lag columns
    df = polynomial_data[['rate_current', 'rate_t1', 'median_current']].copy()

    result = create_polynomial_interaction_features(df, ['rate', 'median'], max_lag_periods=2)

    # Should create features for existing columns
    assert 'derived_rate_squared_current' in result.columns
    assert 'derived_rate_squared_t1' in result.columns

    # Should NOT create features for missing t2
    assert 'derived_rate_squared_t2' not in result.columns


def test_create_polynomial_immutable(polynomial_data):
    """Test that original DataFrame is not modified."""
    original_cols = set(polynomial_data.columns)

    _ = create_polynomial_interaction_features(polynomial_data, ['rate'], max_lag_periods=2)

    assert set(polynomial_data.columns) == original_cols


def test_create_polynomial_allow_inplace():
    """Test allow_inplace parameter avoids copy (performance optimization).

    Note: pd.concat at the end still returns a new DataFrame, but allow_inplace=True
    avoids the initial df.copy() operation for performance.
    """
    df = pd.DataFrame({'rate_current': [1, 2]})

    result = create_polynomial_interaction_features(
        df, ['rate'], max_lag_periods=0, allow_inplace=True
    )

    # Result should have the polynomial features
    assert 'derived_rate_squared_current' in result.columns


# =============================================================================
# LOG TRANSFORMATION TESTS
# =============================================================================


def test_apply_log_plus_one_basic():
    """Test basic log(1 + x) transformation."""
    df = pd.DataFrame({'value': [0, 1, 9, 99]})

    result = apply_log_plus_one_transformation(df, 'value', 'log_value')

    assert 'log_value' in result.columns

    # Verify log(1 + x) values
    assert np.isclose(result['log_value'].iloc[0], np.log1p(0))  # log(1)
    assert np.isclose(result['log_value'].iloc[1], np.log1p(1))  # log(2)
    assert np.isclose(result['log_value'].iloc[2], np.log1p(9))  # log(10)


def test_apply_log_plus_one_zero_values():
    """Test log transformation handles zero values correctly."""
    df = pd.DataFrame({'value': [0, 0, 0]})

    result = apply_log_plus_one_transformation(df, 'value', 'log_val')

    # log(1 + 0) = log(1) = 0
    assert (result['log_val'] == 0).all()


def test_apply_log_plus_one_large_values():
    """Test log transformation with large values."""
    df = pd.DataFrame({'value': [1000, 10000, 100000]})

    result = apply_log_plus_one_transformation(df, 'value', 'log_val')

    # Should produce reasonable log values
    assert result['log_val'].iloc[0] > 6  # log(1001) ≈ 6.9
    assert result['log_val'].iloc[1] > 9  # log(10001) ≈ 9.2
    assert result['log_val'].iloc[2] > 11  # log(100001) ≈ 11.5


def test_apply_log_plus_one_missing_source_column():
    """Test error when source column is missing."""
    df = pd.DataFrame({'value': [1, 2, 3]})

    with pytest.raises(ValueError, match="Source column 'nonexistent' not found"):
        apply_log_plus_one_transformation(df, 'nonexistent', 'log_val')


def test_apply_log_plus_one_immutable():
    """Test that original DataFrame is not modified."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    original_cols = set(df.columns)

    _ = apply_log_plus_one_transformation(df, 'value', 'log_val')

    assert set(df.columns) == original_cols
    assert 'log_val' not in df.columns


def test_apply_log_plus_one_with_nan():
    """Test log transformation with NaN values."""
    df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})

    result = apply_log_plus_one_transformation(df, 'value', 'log_val')

    # NaN should propagate through log transformation
    assert pd.isna(result['log_val'].iloc[1])
    assert not pd.isna(result['log_val'].iloc[0])
    assert not pd.isna(result['log_val'].iloc[2])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_temporal_pipeline():
    """Test full temporal feature engineering pipeline."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'sales': range(100, 200)
    })

    # Step 1: Create temporal indicators
    df = create_temporal_indicator_columns(df, 'date')
    assert 'year' in df.columns
    assert 'month' in df.columns

    # Step 2: Extract day of year
    df = extract_day_of_year_column(df, 'date', 'doy')
    assert 'doy' in df.columns

    # Step 3: Create holiday indicator
    df = create_holiday_indicator_by_day_range(df, 'doy', 13, 359, 'holiday')
    assert 'holiday' in df.columns

    # Step 4: Create lag features
    lag_configs = [{'source_col': 'sales', 'prefix': 'sales', 'lag_direction': 'backward'}]
    df = create_lag_features_for_columns(df, lag_configs, max_lag_periods=2)
    assert 'sales_t1' in df.columns

    # Step 5: Create polynomial features
    df = create_polynomial_interaction_features(df, ['sales'], max_lag_periods=2)
    assert 'derived_sales_squared_current' in df.columns

    # Step 6: Apply log transformation
    df = apply_log_plus_one_transformation(df, 'sales', 'log_sales')
    assert 'log_sales' in df.columns

    # Verify pipeline produced expected feature count
    assert len(df.columns) > 10  # Original + many derived features


def test_empty_dataframe_handling():
    """Test that all functions handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame({'date': pd.to_datetime([]), 'value': []})

    # Temporal indicators
    result = create_temporal_indicator_columns(empty_df, 'date')
    assert len(result) == 0
    assert 'year' in result.columns

    # Day of year
    result = extract_day_of_year_column(empty_df, 'date', 'doy')
    assert len(result) == 0
    assert 'doy' in result.columns


def test_nan_handling_in_lag_features():
    """Test that NaN values are handled correctly in lag features."""
    df = pd.DataFrame({
        'value': [1.0, np.nan, 3.0, 4.0, np.nan]
    })

    lag_configs = [{'source_col': 'value', 'prefix': 'v', 'lag_direction': 'backward'}]
    result = create_lag_features_for_columns(df, lag_configs, max_lag_periods=2)

    # NaN should propagate through lags
    assert pd.isna(result['v_t1'].iloc[2])  # lag of NaN
    assert result['v_t1'].iloc[1] == 1.0  # lag of non-NaN
