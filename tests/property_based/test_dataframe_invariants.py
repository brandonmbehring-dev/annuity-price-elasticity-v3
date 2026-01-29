"""
Property-Based Tests for DataFrame Invariants.

Tests invariants that should hold for all DataFrame transformations:
1. Shape preservation - operations should maintain expected dimensions
2. Type stability - dtypes should remain consistent
3. No NaN introduction - clean data should stay clean
4. Index preservation - transformations should preserve row identity
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from .conftest import price_elasticity_dataframe, feature_matrix


# =============================================================================
# SHAPE PRESERVATION
# =============================================================================

@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_column_selection_preserves_rows(df: pd.DataFrame):
    """Selecting columns should not change row count."""
    selected = df[["date", "own_rate"]]
    assert len(selected) == len(df), "Column selection changed row count"


@pytest.mark.property
@given(df=feature_matrix(n_features=5, min_rows=50, max_rows=100))
def test_transpose_shape_swap(df: pd.DataFrame):
    """Transpose should swap rows and columns."""
    transposed = df.T
    assert transposed.shape == (df.shape[1], df.shape[0]), "Transpose shape incorrect"


@pytest.mark.property
@given(df=price_elasticity_dataframe(min_rows=20, max_rows=50))
def test_copy_preserves_shape(df: pd.DataFrame):
    """Copy should preserve shape exactly."""
    copied = df.copy()
    assert copied.shape == df.shape, "Copy changed shape"


# =============================================================================
# TYPE STABILITY
# =============================================================================

@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_copy_preserves_dtypes(df: pd.DataFrame):
    """Copy should preserve column dtypes."""
    copied = df.copy()
    for col in df.columns:
        assert copied[col].dtype == df[col].dtype, f"dtype changed for {col}"


@pytest.mark.property
@given(df=feature_matrix())
def test_numeric_operations_preserve_numeric_type(df: pd.DataFrame):
    """Arithmetic operations on numeric columns should stay numeric."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            doubled = df[col] * 2
            assert pd.api.types.is_numeric_dtype(doubled), f"Lost numeric dtype for {col}"


@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_rate_columns_are_float(df: pd.DataFrame):
    """Rate columns should always be float."""
    assert pd.api.types.is_float_dtype(df["own_rate"]), "own_rate not float"
    assert pd.api.types.is_float_dtype(df["competitor_rate"]), "competitor_rate not float"


# =============================================================================
# NO NAN INTRODUCTION
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix())
def test_copy_no_nan_introduction(df: pd.DataFrame):
    """Copying clean data should not introduce NaN."""
    assume(not df.isna().any().any())  # Assume input is clean

    copied = df.copy()
    assert not copied.isna().any().any(), "Copy introduced NaN"


@pytest.mark.property
@given(df=feature_matrix())
def test_fillna_removes_all_nan(df: pd.DataFrame):
    """fillna should remove all NaN values."""
    filled = df.fillna(0)
    assert not filled.isna().any().any(), "fillna did not remove all NaN"


@pytest.mark.property
@given(
    df=feature_matrix(),
    scalar=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_scalar_multiply_no_nan(df: pd.DataFrame, scalar: float):
    """Multiplying clean data by finite scalar should not introduce NaN."""
    assume(not df.isna().any().any())

    result = df * scalar
    assert not result.isna().any().any(), "Scalar multiplication introduced NaN"


# =============================================================================
# INDEX PRESERVATION
# =============================================================================

@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_column_operation_preserves_index(df: pd.DataFrame):
    """Column operations should preserve the index."""
    original_index = df.index.tolist()
    df["new_col"] = df["own_rate"] * 100
    assert df.index.tolist() == original_index, "Index changed after column operation"


@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_reset_index_recoverable(df: pd.DataFrame):
    """Reset index should be recoverable."""
    original_shape = df.shape
    reset = df.reset_index(drop=True)
    assert len(reset) == len(df), "Reset index changed length"


# =============================================================================
# VALUE PRESERVATION
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix())
def test_copy_values_equal(df: pd.DataFrame):
    """Copied DataFrame should have identical values."""
    copied = df.copy()
    pd.testing.assert_frame_equal(df, copied)


@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_sort_preserves_values(df: pd.DataFrame):
    """Sorting should preserve all values (just reorder)."""
    sorted_df = df.sort_values("own_rate").reset_index(drop=True)

    # Same values, possibly different order
    assert set(df["own_rate"]) == set(sorted_df["own_rate"]), "Sort lost values"
    assert len(df) == len(sorted_df), "Sort changed length"


# =============================================================================
# CONCATENATION PROPERTIES
# =============================================================================

@pytest.mark.property
@given(
    df1=price_elasticity_dataframe(min_rows=5, max_rows=20),
    df2=price_elasticity_dataframe(min_rows=5, max_rows=20),
)
def test_concat_adds_rows(df1: pd.DataFrame, df2: pd.DataFrame):
    """Concatenating DataFrames should add their row counts."""
    combined = pd.concat([df1, df2], ignore_index=True)
    assert len(combined) == len(df1) + len(df2), "Concat row count incorrect"


@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_empty_concat_is_identity(df: pd.DataFrame):
    """Concatenating with empty DataFrame should be identity."""
    empty = pd.DataFrame(columns=df.columns)
    combined = pd.concat([df, empty], ignore_index=True)
    assert len(combined) == len(df), "Empty concat changed length"


# =============================================================================
# FILTERING PROPERTIES
# =============================================================================

@pytest.mark.property
@given(df=price_elasticity_dataframe(min_rows=20, max_rows=50))
def test_filter_reduces_or_preserves_rows(df: pd.DataFrame):
    """Filtering should never increase row count."""
    filtered = df[df["sales"] > 0]
    assert len(filtered) <= len(df), "Filter increased row count"


@pytest.mark.property
@given(df=price_elasticity_dataframe())
def test_true_filter_is_identity(df: pd.DataFrame):
    """Filtering on always-true condition should preserve all rows."""
    filtered = df[df["own_rate"] >= 0]  # Always true for valid rates
    assert len(filtered) == len(df), "True filter removed rows"


# =============================================================================
# AGGREGATION PROPERTIES
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix(min_rows=10, max_rows=50))
def test_mean_is_scalar(df: pd.DataFrame):
    """Column mean should produce scalar values."""
    means = df.mean()
    assert len(means) == df.shape[1], "Mean should have one value per column"


@pytest.mark.property
@given(df=feature_matrix())
def test_sum_consistent(df: pd.DataFrame):
    """Sum should be consistent across different calculation methods."""
    assume(not df.isna().any().any())

    for col in df.columns:
        df_sum = df[col].sum()
        list_sum = sum(df[col].tolist())
        assert abs(df_sum - list_sum) < 1e-6, f"Sum inconsistent for {col}"
