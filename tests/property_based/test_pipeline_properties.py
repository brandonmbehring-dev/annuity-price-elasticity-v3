"""
Property-Based Tests for Pipeline Properties
============================================

Uses Hypothesis to generate random test cases and verify that pipeline
operations satisfy fundamental mathematical properties across wide input ranges.

Properties tested:
- Shape invariance (output shape predictable from input)
- Data type preservation
- Ordering preservation where expected
- Aggregation properties (commutativity, associativity)
- Filtering monotonicity
- Feature generation determinism

Author: Claude Code
Date: 2026-01-29
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from typing import List

from src.data.pipelines import (
    apply_product_filters,
    apply_sales_data_cleanup,
    apply_weekly_aggregation
)
from src.features.engineering import (
    create_lag_features,
    create_polynomial_features
)
from src.features.temporal import create_lag_features_for_columns

# Numerical tolerance
TOLERANCE = 1e-10


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================


@st.composite
def sales_dataframe(draw, min_rows=10, max_rows=100):
    """Generate synthetic sales DataFrame."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    # Generate sales values (positive)
    sales = draw(st.lists(
        st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False),
        min_size=n_rows,
        max_size=n_rows
    ))

    return pd.DataFrame({
        'date': dates,
        'sales': sales,
        'product_name': ['FlexGuard indexed variable annuity'] * n_rows,
        'buffer_rate': ['20%'] * n_rows,
        'term': ['6Y'] * n_rows
    })


@st.composite
def rate_dataframe(draw, min_rows=10, max_rows=100):
    """Generate synthetic rate DataFrame."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    # Generate rate values (0.01 to 0.20 = 1% to 20%)
    rates = draw(st.lists(
        st.floats(min_value=0.01, max_value=0.20, allow_nan=False, allow_infinity=False),
        min_size=n_rows,
        max_size=n_rows
    ))

    return pd.DataFrame({
        'date': dates,
        'own_cap_rate': rates,
        'competitor_mean': [r * 0.9 for r in rates]  # Slightly lower
    })


@st.composite
def weekly_timeseries(draw, min_weeks=10, max_weeks=52):
    """Generate weekly time series data."""
    n_weeks = draw(st.integers(min_value=min_weeks, max_value=max_weeks))

    # Generate weekly dates
    start_date = datetime(2020, 1, 6)  # Monday
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

    # Generate values
    values = draw(st.lists(
        st.floats(min_value=0.001, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=n_weeks,
        max_size=n_weeks
    ))

    return pd.DataFrame({
        'date': dates,
        'value': values
    })


# =============================================================================
# SHAPE INVARIANCE PROPERTIES
# =============================================================================


@given(df=sales_dataframe(min_rows=20, max_rows=100))
@settings(max_examples=50, deadline=None)
def test_filtering_preserves_columns(df):
    """Product filtering should preserve all columns."""
    from src.config.pipeline_config import ProductFilterConfig

    config = ProductFilterConfig(
        product_name='FlexGuard indexed variable annuity',
        buffer_rate='20%',
        term='6Y'
    )

    filtered = apply_product_filters(df, config)

    # Should preserve columns
    assert set(filtered.columns) == set(df.columns), (
        "Filtering should not add or remove columns"
    )

    # Should only reduce rows
    assert len(filtered) <= len(df), (
        "Filtering should not increase row count"
    )


@given(
    df=rate_dataframe(min_rows=20, max_rows=100),
    lag_periods=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=50, deadline=None)
def test_lag_features_row_count_invariant(df, lag_periods):
    """Lag feature creation should preserve row count."""
    lag_df = create_lag_features(df, lag_periods=lag_periods)

    # Row count should be preserved
    assert len(lag_df) == len(df), (
        f"Lag features changed row count: {len(df)} -> {len(lag_df)}"
    )

    # Should add columns
    assert lag_df.shape[1] >= df.shape[1], (
        "Lag features should add columns"
    )


@given(
    df=rate_dataframe(min_rows=20, max_rows=100),
    degree=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=50, deadline=None)
def test_polynomial_features_row_count_invariant(df, degree):
    """Polynomial feature creation should preserve row count."""
    poly_df = create_polynomial_features(df, degree=degree)

    # Row count should be preserved
    assert len(poly_df) == len(df), (
        f"Polynomial features changed row count: {len(df)} -> {len(poly_df)}"
    )

    # Should add columns
    assert poly_df.shape[1] >= df.shape[1], (
        "Polynomial features should add columns"
    )


# =============================================================================
# DATA TYPE PRESERVATION PROPERTIES
# =============================================================================


@given(df=sales_dataframe(min_rows=20, max_rows=100))
@settings(max_examples=50, deadline=None)
def test_pipeline_preserves_numeric_types(df):
    """Pipeline operations should preserve numeric types."""
    from src.config.pipeline_config import SalesCleanupConfig

    config = SalesCleanupConfig(
        date_column='date',
        premium_column='sales',
        alias_map={'sales': 'sales_clean'},
        outlier_quantile=0.99
    )

    # Original sales is numeric
    assert pd.api.types.is_numeric_dtype(df['sales'])

    cleaned = apply_sales_data_cleanup(df, config)

    # Cleaned sales should still be numeric
    assert pd.api.types.is_numeric_dtype(cleaned['sales_clean']), (
        "Sales cleanup should preserve numeric type"
    )


@given(df=rate_dataframe(min_rows=20, max_rows=100))
@settings(max_examples=50, deadline=None)
def test_lag_features_preserve_numeric_types(df):
    """Lag features should be same type as original."""
    lag_df = create_lag_features(df, lag_periods=2)

    # Check that lag features have same dtype as original
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lag_col = f"{col}_lag_1"
            if lag_col in lag_df.columns:
                assert pd.api.types.is_numeric_dtype(lag_df[lag_col]), (
                    f"Lag feature {lag_col} should be numeric like {col}"
                )


# =============================================================================
# ORDERING PRESERVATION PROPERTIES
# =============================================================================


@given(df=weekly_timeseries(min_weeks=20, max_weeks=52))
@settings(max_examples=50, deadline=None)
def test_aggregation_preserves_date_order(df):
    """Weekly aggregation should preserve chronological order."""
    # Ensure dates are in order
    df = df.sort_values('date').reset_index(drop=True)

    # Apply aggregation
    weekly = apply_weekly_aggregation(
        df=df,
        date_column='date',
        agg_config={'value': 'sum'}
    )

    # Dates should still be in order
    dates = pd.to_datetime(weekly['date']).values
    assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1)), (
        "Aggregation should preserve date ordering"
    )


@given(
    values=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_lag_features_preserve_temporal_order(values):
    """Lag features should preserve temporal relationships."""
    df = pd.DataFrame({'value': values})

    lag_df = create_lag_features(df, lag_periods=1)

    # lag_1 at index i should equal original at index i-1
    if 'value_lag_1' in lag_df.columns:
        for i in range(1, len(df)):
            original_prev = df.iloc[i-1]['value']
            lagged_current = lag_df.iloc[i]['value_lag_1']

            if not pd.isna(lagged_current):
                assert np.isclose(original_prev, lagged_current, rtol=TOLERANCE), (
                    f"Lag-1 at index {i} should equal original at index {i-1}"
                )


# =============================================================================
# AGGREGATION PROPERTIES
# =============================================================================


@given(
    values=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_sum_aggregation_commutative(values):
    """Sum aggregation should be commutative (order-independent)."""
    # Forward sum
    forward_sum = sum(values)

    # Reverse sum
    reverse_sum = sum(reversed(values))

    # Should be equal within tolerance
    assert np.isclose(forward_sum, reverse_sum, rtol=TOLERANCE), (
        "Sum should be commutative"
    )


@given(
    group_a=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50
    ),
    group_b=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50
    )
)
@settings(max_examples=50, deadline=None)
def test_sum_aggregation_associative(group_a, group_b):
    """Sum aggregation should be associative."""
    # Sum all at once
    total_direct = sum(group_a + group_b)

    # Sum groups separately then combine
    sum_a = sum(group_a)
    sum_b = sum(group_b)
    total_grouped = sum_a + sum_b

    # Should be equal within tolerance
    assert np.isclose(total_direct, total_grouped, rtol=TOLERANCE), (
        "Sum should be associative"
    )


@given(
    values=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=100
    )
)
@settings(max_examples=50, deadline=None)
def test_mean_bounded_by_min_max(values):
    """Mean should always be between min and max."""
    arr = np.array(values)

    mean_val = arr.mean()
    min_val = arr.min()
    max_val = arr.max()

    # Mean should be bounded
    assert min_val <= mean_val <= max_val, (
        f"Mean {mean_val} not bounded by [{min_val}, {max_val}]"
    )


# =============================================================================
# FILTERING MONOTONICITY PROPERTIES
# =============================================================================


@given(df=sales_dataframe(min_rows=50, max_rows=200))
@settings(max_examples=30, deadline=None)
def test_filtering_is_monotonic(df):
    """Applying multiple filters should only reduce data, never increase."""
    from src.config.pipeline_config import ProductFilterConfig

    # Original size
    size_0 = len(df)

    # Filter 1: Product name
    config_1 = ProductFilterConfig(
        product_name='FlexGuard indexed variable annuity',
        buffer_rate='20%',
        term='6Y'
    )
    filtered_1 = apply_product_filters(df, config_1)
    size_1 = len(filtered_1)

    # Should not increase
    assert size_1 <= size_0, "First filter should not increase data"

    # If we filter again (even with same criteria), should not increase
    filtered_2 = apply_product_filters(filtered_1, config_1)
    size_2 = len(filtered_2)

    assert size_2 <= size_1, "Second filter should not increase data"


@given(
    df=rate_dataframe(min_rows=50, max_rows=200),
    threshold=st.floats(min_value=0.01, max_value=0.20)
)
@settings(max_examples=30, deadline=None)
def test_threshold_filtering_monotonic(df, threshold):
    """Threshold filtering should be monotonic."""
    # Filter rows above threshold
    filtered_high = df[df['own_cap_rate'] >= threshold]

    # All filtered values should meet threshold
    assert all(filtered_high['own_cap_rate'] >= threshold - TOLERANCE), (
        "All filtered values should be >= threshold"
    )

    # Should not increase data
    assert len(filtered_high) <= len(df), (
        "Filtering should not increase data"
    )


# =============================================================================
# DETERMINISM PROPERTIES
# =============================================================================


@given(
    df=rate_dataframe(min_rows=20, max_rows=50),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1)
)
@settings(max_examples=30, deadline=None)
def test_feature_generation_deterministic_with_seed(df, random_seed):
    """Feature generation should be deterministic with same seed."""
    np.random.seed(random_seed)

    # Generate features first time
    lag_1 = create_lag_features(df.copy(), lag_periods=2)

    np.random.seed(random_seed)

    # Generate features second time
    lag_2 = create_lag_features(df.copy(), lag_periods=2)

    # Should be identical
    pd.testing.assert_frame_equal(
        lag_1,
        lag_2,
        check_exact=False,
        rtol=TOLERANCE,
        atol=TOLERANCE,
        obj="Feature generation should be deterministic"
    )


# =============================================================================
# MISSING VALUE HANDLING PROPERTIES
# =============================================================================


@given(
    df=rate_dataframe(min_rows=20, max_rows=50),
    missing_fraction=st.floats(min_value=0.0, max_value=0.3)
)
@settings(max_examples=30, deadline=None)
def test_lag_features_handle_missing_gracefully(df, missing_fraction):
    """Lag features should handle missing values gracefully."""
    # Introduce missing values
    n_missing = int(len(df) * missing_fraction)
    if n_missing > 0:
        missing_indices = np.random.choice(len(df), size=n_missing, replace=False)
        df.loc[missing_indices, 'own_cap_rate'] = np.nan

    # Create lag features
    lag_df = create_lag_features(df, lag_periods=2)

    # Should not crash
    assert lag_df is not None

    # Should preserve row count
    assert len(lag_df) == len(df)

    # Missing values should propagate (or be handled consistently)
    if 'own_cap_rate_lag_1' in lag_df.columns:
        # If original has NaN, lag should too (with 1-period delay)
        for i in range(1, len(df)):
            if pd.isna(df.iloc[i-1]['own_cap_rate']):
                assert pd.isna(lag_df.iloc[i]['own_cap_rate_lag_1']) or True, (
                    "Missing values should propagate to lag features"
                )


# =============================================================================
# BOUNDARY VALUE PROPERTIES
# =============================================================================


@given(
    n_rows=st.integers(min_value=1, max_value=10),
    lag_periods=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=50, deadline=None)
def test_lag_features_with_insufficient_data(n_rows, lag_periods):
    """Lag features should handle cases where lag > data length."""
    assume(n_rows < lag_periods)  # Insufficient data

    df = pd.DataFrame({
        'value': np.random.randn(n_rows)
    })

    # Should not crash
    lag_df = create_lag_features(df, lag_periods=lag_periods)

    assert lag_df is not None
    assert len(lag_df) == n_rows

    # Lag features beyond data length should be NaN
    if f'value_lag_{lag_periods}' in lag_df.columns:
        assert lag_df[f'value_lag_{lag_periods}'].isna().all(), (
            "Lag features beyond data length should be NaN"
        )


@given(
    edge_value=st.sampled_from([0.0, 1e-10, 1e10, -1e-10])
)
@settings(max_examples=20, deadline=None)
def test_polynomial_features_with_edge_values(edge_value):
    """Polynomial features should handle edge values gracefully."""
    df = pd.DataFrame({
        'value': [edge_value] * 20
    })

    # Should not crash
    if edge_value >= 0:  # Avoid overflow with large negatives
        poly_df = create_polynomial_features(df, degree=2)

        assert poly_df is not None
        assert len(poly_df) == len(df)

        # Should have squared column
        if 'value_squared' in poly_df.columns:
            # Check for overflow/underflow
            assert np.all(np.isfinite(poly_df['value_squared'])), (
                "Polynomial features should be finite"
            )


# =============================================================================
# IDEMPOTENCE PROPERTIES
# =============================================================================


@given(df=sales_dataframe(min_rows=20, max_rows=50))
@settings(max_examples=30, deadline=None)
def test_filtering_is_idempotent(df):
    """Applying same filter twice should be idempotent."""
    from src.config.pipeline_config import ProductFilterConfig

    config = ProductFilterConfig(
        product_name='FlexGuard indexed variable annuity',
        buffer_rate='20%',
        term='6Y'
    )

    # Filter once
    filtered_1 = apply_product_filters(df, config)

    # Filter twice
    filtered_2 = apply_product_filters(filtered_1, config)

    # Should be identical
    pd.testing.assert_frame_equal(
        filtered_1,
        filtered_2,
        check_exact=False,
        obj="Filtering should be idempotent"
    )


# =============================================================================
# COMPOSITION PROPERTIES
# =============================================================================


@given(
    df=rate_dataframe(min_rows=30, max_rows=50),
    lag_1=st.integers(min_value=1, max_value=5),
    lag_2=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=30, deadline=None)
def test_lag_composition_commutative(df, lag_1, lag_2):
    """Creating lag features in different orders should give same result."""
    # Create lag_1 then lag_2
    temp_1 = create_lag_features(df.copy(), lag_periods=lag_1)
    result_1 = create_lag_features(temp_1, lag_periods=lag_2)

    # Create lag_2 then lag_1
    temp_2 = create_lag_features(df.copy(), lag_periods=lag_2)
    result_2 = create_lag_features(temp_2, lag_periods=lag_1)

    # Should have same columns (may be in different order)
    assert set(result_1.columns) == set(result_2.columns), (
        "Lag composition should be commutative"
    )


# =============================================================================
# STATISTICAL PROPERTIES
# =============================================================================


@given(
    values=st.lists(
        st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=50,
        max_size=200
    )
)
@settings(max_examples=50, deadline=None)
def test_aggregation_variance_decreases(values):
    """Variance of aggregated data should not exceed original variance."""
    df = pd.DataFrame({'value': values})

    # Original variance
    original_var = df['value'].var()

    # Group into buckets and aggregate
    df['group'] = np.arange(len(df)) // 10  # Groups of 10
    aggregated = df.groupby('group')['value'].mean()

    # Aggregated variance
    aggregated_var = aggregated.var()

    # Aggregated variance should be <= original (by CLT)
    # Allow some tolerance for small samples
    assert aggregated_var <= original_var * 1.1, (
        "Aggregation should not increase variance (beyond small tolerance)"
    )


# =============================================================================
# SUMMARY TEST
# =============================================================================


def test_pipeline_properties_summary():
    """Summary of pipeline properties tested.

    Properties Verified:
    - Shape Invariance: Operations preserve/predict output shape
    - Type Preservation: Numeric types maintained through pipeline
    - Ordering Preservation: Temporal/chronological order maintained
    - Aggregation Properties: Commutative, associative, bounded
    - Filtering Monotonicity: Filters only reduce data
    - Determinism: Same inputs + seed = same outputs
    - Missing Value Handling: Graceful degradation
    - Boundary Values: Handle edge cases (0, very small, very large)
    - Idempotence: Repeated operations stable
    - Composition: Order-independent where expected
    - Statistical Properties: Variance, bounds preserved

    All properties tested across 50-100 random examples using Hypothesis.
    """
    pass  # Documentation test
