"""
Unit Tests for Inference Tableau Export Functions
=================================================

Tests for src/models/inference.py covering:
- Tableau export validation functions
- Sales momentum calculations
- DataFrame melting for Tableau
- Business context column addition

Target: 85% coverage for inference.py

Test Pattern:
- Test Tableau export pipeline
- Test validation functions
- Test helper functions
- Test edge cases

Author: Claude Code
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import pytest

from src.models.inference import (
    validate_melt_dataframe_inputs,
    calculate_sales_momentum,
    melt_dataframe_for_tableau,
    _melt_ci_to_long_format,
    _add_business_context_columns,
    _add_sales_momentum_column,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_confidence_intervals():
    """Simple confidence intervals DataFrame for testing."""
    return pd.DataFrame({
        'rate_change_in_basis_points': [-100, 0, 100],
        'range': ['lower', 'base', 'upper'],
        'predicted_sales': [95000, 100000, 105000],
        'ci_lower': [90000, 95000, 100000],
        'ci_upper': [100000, 105000, 110000]
    })


@pytest.fixture
def simple_sales_data():
    """Simple sales data DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'prudential_rate_current': [4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4],
        'competitor_mid_current': [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9],
        'sales_target_t2': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000],
        'sales_target_t3': [900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000]
    })


# =============================================================================
# VALIDATION TESTS
# =============================================================================


def test_validate_melt_inputs_basic(simple_confidence_intervals, simple_sales_data):
    """Test basic validation passes with valid inputs."""
    # Should not raise
    validate_melt_dataframe_inputs(
        confidence_intervals=simple_confidence_intervals,
        sales_data=simple_sales_data,
        prediction_date='2023-12-01',
        prudential_rate_col='prudential_rate_current',
        competitor_rate_col='competitor_mid_current',
        sales_lag_cols=['sales_target_t2', 'sales_target_t3'],
        sales_rounding_power=-7
    )


def test_validate_melt_inputs_empty_ci():
    """Test validation fails with empty confidence intervals."""
    empty_ci = pd.DataFrame()
    sales_data = pd.DataFrame({'prudential_rate_current': [4.5]})

    with pytest.raises(ValueError, match="Confidence intervals DataFrame cannot be empty"):
        validate_melt_dataframe_inputs(
            empty_ci, sales_data, '2023-12-01',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], -7
        )


def test_validate_melt_inputs_empty_sales(simple_confidence_intervals):
    """Test validation fails with empty sales data."""
    empty_sales = pd.DataFrame()

    with pytest.raises(ValueError, match="Sales data DataFrame cannot be empty"):
        validate_melt_dataframe_inputs(
            simple_confidence_intervals, empty_sales, '2023-12-01',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], -7
        )


def test_validate_melt_inputs_missing_ci_columns():
    """Test validation fails with missing CI columns."""
    bad_ci = pd.DataFrame({'wrong_column': [1, 2, 3]})
    sales_data = pd.DataFrame({
        'prudential_rate_current': [4.5],
        'competitor_mid_current': [4.0],
        'sales_target_t2': [1000000]
    })

    with pytest.raises(ValueError, match="Missing confidence interval columns"):
        validate_melt_dataframe_inputs(
            bad_ci, sales_data, '2023-12-01',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], -7
        )


def test_validate_melt_inputs_missing_sales_columns(simple_confidence_intervals):
    """Test validation fails with missing sales columns."""
    bad_sales = pd.DataFrame({'wrong_column': [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing sales data columns"):
        validate_melt_dataframe_inputs(
            simple_confidence_intervals, bad_sales, '2023-12-01',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], -7
        )


def test_validate_melt_inputs_invalid_date(simple_confidence_intervals, simple_sales_data):
    """Test validation fails with invalid date format."""
    with pytest.raises(ValueError):
        validate_melt_dataframe_inputs(
            simple_confidence_intervals, simple_sales_data, 'not-a-date',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], -7
        )


def test_validate_melt_inputs_positive_rounding_power(simple_confidence_intervals, simple_sales_data):
    """Test validation fails with positive rounding power."""
    with pytest.raises(ValueError, match="Sales rounding power should be negative"):
        validate_melt_dataframe_inputs(
            simple_confidence_intervals, simple_sales_data, '2023-12-01',
            'prudential_rate_current', 'competitor_mid_current',
            ['sales_target_t2'], 7  # Positive - wrong!
        )


# =============================================================================
# SALES MOMENTUM TESTS
# =============================================================================


def test_calculate_sales_momentum_basic(simple_sales_data):
    """Test basic sales momentum calculation."""
    result = calculate_sales_momentum(
        simple_sales_data,
        sales_lag_cols=['sales_target_t2', 'sales_target_t3'],
        sales_rounding_power=-7
    )

    # Last row: sales_target_t2 = 1900000, sales_target_t3 = 1800000
    # Sum = 3700000
    # Rounded to nearest 10M (-7 power): 0
    expected = 0  # np.round(3700000, -7) = 0
    assert result == expected


def test_calculate_sales_momentum_with_large_values():
    """Test sales momentum with values that round nicely."""
    df = pd.DataFrame({
        'sales_lag_2': [50000000],
        'sales_lag_1': [30000000]
    })

    result = calculate_sales_momentum(df, ['sales_lag_2', 'sales_lag_1'], -7)

    # Sum = 80000000
    # Rounded to nearest 10M: 80000000
    expected = 80000000
    assert result == expected


def test_calculate_sales_momentum_different_rounding():
    """Test sales momentum with different rounding powers."""
    df = pd.DataFrame({
        'sales_a': [1234567],
        'sales_b': [9876543]
    })

    # Round to nearest million (-6)
    result = calculate_sales_momentum(df, ['sales_a', 'sales_b'], -6)

    # Sum = 11111110
    # Round to nearest million: 11000000
    expected = 11000000
    assert result == expected


def test_calculate_sales_momentum_single_column():
    """Test sales momentum with single lag column."""
    df = pd.DataFrame({'sales': [5000000]})

    result = calculate_sales_momentum(df, ['sales'], -6)

    # 5000000 rounded to nearest million = 5000000
    assert result == 5000000


def test_calculate_sales_momentum_missing_column():
    """Test error when sales lag column is missing."""
    df = pd.DataFrame({'other_column': [1000000]})

    with pytest.raises(ValueError):
        calculate_sales_momentum(df, ['nonexistent_column'], -7)


# =============================================================================
# MELTING TESTS
# =============================================================================


def test_melt_ci_to_long_format(simple_confidence_intervals):
    """Test confidence interval melting to long format."""
    result = _melt_ci_to_long_format(simple_confidence_intervals)

    # Should have id_vars preserved and other columns melted
    assert 'rate_change_in_basis_points' in result.columns
    assert 'range' in result.columns
    assert 'output_type' in result.columns
    assert 'value' in result.columns

    # Should have 3 rows × 3 value columns = 9 rows
    expected_rows = 3 * 3  # 3 original rows × 3 melted columns
    assert len(result) == expected_rows


def test_melt_ci_preserves_id_vars(simple_confidence_intervals):
    """Test that id_vars are preserved in melting."""
    result = _melt_ci_to_long_format(simple_confidence_intervals)

    # Original id values should all appear in melted result
    original_rates = set(simple_confidence_intervals['rate_change_in_basis_points'])
    melted_rates = set(result['rate_change_in_basis_points'])
    assert original_rates == melted_rates


# =============================================================================
# BUSINESS CONTEXT TESTS
# =============================================================================


def test_add_business_context_columns(simple_sales_data):
    """Test adding business context columns."""
    df_melted = pd.DataFrame({'dummy': [1, 2, 3]})

    result = _add_business_context_columns(
        df_melted,
        simple_sales_data,
        prediction_date='2023-12-01',
        prudential_rate_col='prudential_rate_current',
        competitor_rate_col='competitor_mid_current'
    )

    # Should have new columns
    assert 'prediction_date' in result.columns
    assert 'Prudential Cap Rate' in result.columns
    assert 'Weighted Mean By Market Share of Competitors Cap Rate' in result.columns

    # Check values
    assert (result['prediction_date'] == '2023-12-01').all()
    assert result['Prudential Cap Rate'].iloc[0] == 5.4  # Last row
    assert result['Weighted Mean By Market Share of Competitors Cap Rate'].iloc[0] == 4.9  # Last row


def test_add_business_context_uses_last_row(simple_sales_data):
    """Test that business context uses the last row of sales data."""
    df_melted = pd.DataFrame({'dummy': [1]})

    result = _add_business_context_columns(
        df_melted,
        simple_sales_data,
        '2023-12-01',
        'prudential_rate_current',
        'competitor_mid_current'
    )

    # Should use last row values (index 9)
    assert result['Prudential Cap Rate'].iloc[0] == simple_sales_data['prudential_rate_current'].iloc[-1]
    assert result['Weighted Mean By Market Share of Competitors Cap Rate'].iloc[0] == simple_sales_data['competitor_mid_current'].iloc[-1]


# =============================================================================
# SALES MOMENTUM COLUMN TESTS
# =============================================================================


def test_add_sales_momentum_column(simple_sales_data):
    """Test adding sales momentum column."""
    df_melted = pd.DataFrame({'dummy': [1, 2, 3]})

    result = _add_sales_momentum_column(
        df_melted,
        simple_sales_data,
        sales_lag_cols=['sales_target_t2', 'sales_target_t3'],
        sales_rounding_power=-7
    )

    # Should have new column
    assert 'Previous Two Week Sales' in result.columns

    # All rows should have same momentum value
    assert len(result['Previous Two Week Sales'].unique()) == 1


def test_add_sales_momentum_column_value_correctness():
    """Test sales momentum column has correct calculated value."""
    sales_data = pd.DataFrame({
        'sales_lag_2': [50000000],
        'sales_lag_1': [30000000]
    })
    df_melted = pd.DataFrame({'dummy': [1]})

    result = _add_sales_momentum_column(
        df_melted, sales_data,
        ['sales_lag_2', 'sales_lag_1'], -7
    )

    # 50M + 30M = 80M, rounded to 10M = 80M
    assert result['Previous Two Week Sales'].iloc[0] == 80000000


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================


def test_melt_dataframe_for_tableau_basic(simple_confidence_intervals, simple_sales_data):
    """Test full Tableau export pipeline."""
    result = melt_dataframe_for_tableau(
        confidence_intervals=simple_confidence_intervals,
        sales_data=simple_sales_data,
        prediction_date='2023-12-01',
        prudential_rate_col='prudential_rate_current',
        competitor_rate_col='competitor_mid_current',
        sales_lag_cols=['sales_target_t2', 'sales_target_t3'],
        sales_rounding_power=-7
    )

    # Should have all expected columns
    assert 'rate_change_in_basis_points' in result.columns
    assert 'range' in result.columns
    assert 'output_type' in result.columns
    assert 'value' in result.columns
    assert 'prediction_date' in result.columns
    assert 'Prudential Cap Rate' in result.columns
    assert 'Weighted Mean By Market Share of Competitors Cap Rate' in result.columns
    assert 'Previous Two Week Sales' in result.columns


def test_melt_dataframe_for_tableau_default_sales_lags(simple_confidence_intervals, simple_sales_data):
    """Test Tableau export with default sales lag columns."""
    # Add default columns to sales data
    sales_data = simple_sales_data.copy()
    sales_data['sales_target_t2'] = [1000000] * len(sales_data)
    sales_data['sales_target_t3'] = [900000] * len(sales_data)

    result = melt_dataframe_for_tableau(
        confidence_intervals=simple_confidence_intervals,
        sales_data=sales_data,
        prediction_date='2023-12-01',
        sales_lag_cols=None  # Use defaults
    )

    # Should work with default columns
    assert 'Previous Two Week Sales' in result.columns


def test_melt_dataframe_for_tableau_shape(simple_confidence_intervals, simple_sales_data):
    """Test that melted DataFrame has expected shape."""
    result = melt_dataframe_for_tableau(
        confidence_intervals=simple_confidence_intervals,
        sales_data=simple_sales_data,
        prediction_date='2023-12-01'
    )

    # 3 CI rows × 3 value columns = 9 rows
    expected_rows = 3 * 3
    assert len(result) == expected_rows


def test_melt_dataframe_for_tableau_immutable(simple_confidence_intervals, simple_sales_data):
    """Test that input DataFrames are not modified."""
    ci_original = simple_confidence_intervals.copy()
    sales_original = simple_sales_data.copy()

    _ = melt_dataframe_for_tableau(
        simple_confidence_intervals,
        simple_sales_data,
        '2023-12-01'
    )

    # Originals should be unchanged
    pd.testing.assert_frame_equal(simple_confidence_intervals, ci_original)
    pd.testing.assert_frame_equal(simple_sales_data, sales_original)


def test_melt_dataframe_for_tableau_custom_columns():
    """Test Tableau export with custom column names."""
    ci = pd.DataFrame({
        'rate_change_in_basis_points': [-50, 0, 50],
        'range': ['low', 'mid', 'high'],
        'pred': [100, 110, 120]
    })
    sales = pd.DataFrame({
        'custom_pru_rate': [4.5],
        'custom_comp_rate': [4.0],
        'lag_a': [1000000],
        'lag_b': [900000]
    })

    result = melt_dataframe_for_tableau(
        ci, sales, '2023-12-01',
        prudential_rate_col='custom_pru_rate',
        competitor_rate_col='custom_comp_rate',
        sales_lag_cols=['lag_a', 'lag_b'],
        sales_rounding_power=-6
    )

    # Should use custom columns
    assert result['Prudential Cap Rate'].iloc[0] == 4.5
    assert result['Weighted Mean By Market Share of Competitors Cap Rate'].iloc[0] == 4.0


def test_melt_dataframe_for_tableau_preserves_rate_changes(simple_confidence_intervals, simple_sales_data):
    """Test that rate change values are preserved through melting."""
    result = melt_dataframe_for_tableau(
        simple_confidence_intervals,
        simple_sales_data,
        '2023-12-01'
    )

    # Original rate changes should all appear
    original_changes = set(simple_confidence_intervals['rate_change_in_basis_points'])
    melted_changes = set(result['rate_change_in_basis_points'])
    assert original_changes == melted_changes


def test_melt_dataframe_for_tableau_preserves_range_values(simple_confidence_intervals, simple_sales_data):
    """Test that range values are preserved through melting."""
    result = melt_dataframe_for_tableau(
        simple_confidence_intervals,
        simple_sales_data,
        '2023-12-01'
    )

    # Original ranges should all appear
    original_ranges = set(simple_confidence_intervals['range'])
    melted_ranges = set(result['range'])
    assert original_ranges == melted_ranges


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_melt_single_row_ci():
    """Test melting with single confidence interval row."""
    ci = pd.DataFrame({
        'rate_change_in_basis_points': [0],
        'range': ['base'],
        'pred': [100]
    })
    sales = pd.DataFrame({
        'prudential_rate_current': [4.5],
        'competitor_mid_current': [4.0],
        'sales_target_t2': [1000000],
        'sales_target_t3': [900000]
    })

    result = melt_dataframe_for_tableau(ci, sales, '2023-12-01')

    # 1 row × 1 value column = 1 row
    assert len(result) == 1


def test_melt_many_value_columns():
    """Test melting with many value columns."""
    ci = pd.DataFrame({
        'rate_change_in_basis_points': [0],
        'range': ['base'],
        'col1': [1],
        'col2': [2],
        'col3': [3],
        'col4': [4],
        'col5': [5]
    })
    sales = pd.DataFrame({
        'prudential_rate_current': [4.5],
        'competitor_mid_current': [4.0],
        'sales_target_t2': [1000000],
        'sales_target_t3': [900000]
    })

    result = melt_dataframe_for_tableau(ci, sales, '2023-12-01')

    # 1 row × 5 value columns = 5 rows
    assert len(result) == 5
    assert set(result['output_type']) == {'col1', 'col2', 'col3', 'col4', 'col5'}


def test_calculate_sales_momentum_zero_values():
    """Test sales momentum with zero values."""
    df = pd.DataFrame({
        'sales_1': [0],
        'sales_2': [0]
    })

    result = calculate_sales_momentum(df, ['sales_1', 'sales_2'], -6)
    assert result == 0


def test_calculate_sales_momentum_very_large_values():
    """Test sales momentum with very large values."""
    df = pd.DataFrame({
        'sales_1': [999999999],
        'sales_2': [999999999]
    })

    result = calculate_sales_momentum(df, ['sales_1', 'sales_2'], -7)

    # Sum = ~2B, rounded to 10M
    expected = int(np.round(1999999998, -7))
    assert result == expected
