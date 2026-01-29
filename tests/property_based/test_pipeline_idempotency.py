"""
Property-Based Tests for Pipeline Idempotency.

Tests that pipelines produce consistent results:
1. Same input -> same output (determinism)
2. Re-running produces identical results
3. Order independence where expected
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from .conftest import price_elasticity_dataframe, feature_matrix


# =============================================================================
# DETERMINISM
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix(n_features=3, min_rows=20, max_rows=50))
def test_copy_idempotent(df: pd.DataFrame):
    """Copying a DataFrame multiple times should give identical results."""
    copy1 = df.copy()
    copy2 = df.copy()

    pd.testing.assert_frame_equal(copy1, copy2)


@pytest.mark.property
@given(df=feature_matrix())
def test_mean_deterministic(df: pd.DataFrame):
    """Mean calculation should be deterministic."""
    mean1 = df.mean()
    mean2 = df.mean()

    pd.testing.assert_series_equal(mean1, mean2)


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    )
)
def test_np_operations_deterministic(values: list):
    """NumPy operations should be deterministic."""
    arr = np.array(values)

    result1 = np.sum(arr)
    result2 = np.sum(arr)

    assert result1 == result2, "NumPy sum not deterministic"


# =============================================================================
# IDEMPOTENCY
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix())
def test_dropna_idempotent(df: pd.DataFrame):
    """Dropping NaN twice should be same as once."""
    once = df.dropna()
    twice = once.dropna()

    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.property
@given(df=feature_matrix())
def test_reset_index_idempotent(df: pd.DataFrame):
    """Reset index is idempotent."""
    once = df.reset_index(drop=True)
    twice = once.reset_index(drop=True)

    pd.testing.assert_frame_equal(once, twice)


@pytest.mark.property
@given(
    df=feature_matrix(),
    value=st.floats(min_value=-10, max_value=10, allow_nan=False),
)
def test_fillna_idempotent(df: pd.DataFrame, value: float):
    """fillna with a value is idempotent."""
    once = df.fillna(value)
    twice = once.fillna(value)

    pd.testing.assert_frame_equal(once, twice)


# =============================================================================
# ORDER INDEPENDENCE (WHERE EXPECTED)
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix())
def test_sum_order_independent(df: pd.DataFrame):
    """Sum should be independent of column order."""
    assume(not df.isna().any().any())

    sum_original = df.sum()

    # Reverse column order
    df_reversed = df[df.columns[::-1]]
    sum_reversed = df_reversed.sum()

    # Sums should be equal (just in different order)
    for col in df.columns:
        assert abs(sum_original[col] - sum_reversed[col]) < 1e-10


@pytest.mark.property
@given(df=feature_matrix(min_rows=10, max_rows=30))
def test_mean_order_independent(df: pd.DataFrame):
    """Mean should be independent of row order."""
    assume(not df.isna().any().any())

    mean_original = df.mean()

    # Shuffle rows
    df_shuffled = df.sample(frac=1.0, random_state=42)
    mean_shuffled = df_shuffled.mean()

    pd.testing.assert_series_equal(mean_original, mean_shuffled)


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=20,
    )
)
def test_sum_associative(values: list):
    """Sum should be associative (order of operations doesn't matter much)."""
    total = sum(values)
    reversed_total = sum(reversed(values))

    # Due to floating point, they might not be exactly equal
    assert abs(total - reversed_total) < 1e-10 * len(values)


# =============================================================================
# TRANSFORMATION CONSISTENCY
# =============================================================================

@pytest.mark.property
@given(
    df=feature_matrix(),
    scalar=st.floats(min_value=0.1, max_value=10, allow_nan=False),
)
def test_scale_then_unscale(df: pd.DataFrame, scalar: float):
    """Scaling then unscaling should recover original values."""
    assume(not df.isna().any().any())

    scaled = df * scalar
    unscaled = scaled / scalar

    for col in df.columns:
        for i in range(len(df)):
            assert abs(df[col].iloc[i] - unscaled[col].iloc[i]) < 1e-10


@pytest.mark.property
@given(df=feature_matrix())
def test_double_transpose_identity(df: pd.DataFrame):
    """Transposing twice should recover original DataFrame."""
    double_transposed = df.T.T

    pd.testing.assert_frame_equal(df, double_transposed)


# =============================================================================
# SEEDED RANDOMNESS
# =============================================================================

@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n=st.integers(min_value=10, max_value=100),
)
def test_seeded_random_deterministic(seed: int, n: int):
    """Seeded random operations should be deterministic."""
    np.random.seed(seed)
    random1 = np.random.randn(n)

    np.random.seed(seed)
    random2 = np.random.randn(n)

    np.testing.assert_array_equal(random1, random2)


@pytest.mark.property
@given(
    df=feature_matrix(min_rows=20, max_rows=50),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_seeded_sample_deterministic(df: pd.DataFrame, seed: int):
    """Seeded DataFrame sampling should be deterministic."""
    sample1 = df.sample(n=min(10, len(df)), random_state=seed)
    sample2 = df.sample(n=min(10, len(df)), random_state=seed)

    pd.testing.assert_frame_equal(sample1, sample2)


# =============================================================================
# PIPELINE COMPOSITION
# =============================================================================

@pytest.mark.property
@given(df=feature_matrix(min_rows=20, max_rows=50))
def test_filter_then_select_commutes(df: pd.DataFrame):
    """Filtering then selecting should equal selecting then filtering."""
    assume(len(df.columns) >= 2)

    col = df.columns[0]
    median = df[col].median()

    # Filter then select
    result1 = df[df[col] > median][[col]].copy()

    # Select then filter
    result2 = df[[col]][df[col] > median].copy()

    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.property
@given(
    df=feature_matrix(),
    scalar=st.floats(min_value=0.1, max_value=10, allow_nan=False),
)
def test_multiply_then_add_order(df: pd.DataFrame, scalar: float):
    """(df * a) + (df * b) should equal df * (a + b)."""
    assume(not df.isna().any().any())

    a, b = scalar, scalar / 2

    result1 = df * a + df * b
    result2 = df * (a + b)

    for col in df.columns:
        for i in range(len(df)):
            assert abs(result1[col].iloc[i] - result2[col].iloc[i]) < 1e-10
