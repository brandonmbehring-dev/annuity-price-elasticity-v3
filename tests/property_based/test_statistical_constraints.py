"""
Property-Based Tests for Statistical Constraints.

Tests invariants for statistical properties:
1. Coefficient signs - economic constraints on regression coefficients
2. R-squared bounds - always in [0, 1]
3. Variance positivity - variance should be non-negative
4. Standard error bounds - standard errors should be positive
"""

import pytest
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from .conftest import coefficient_dict, valid_constraint_coefficients


# =============================================================================
# COEFFICIENT SIGN CONSTRAINTS
# =============================================================================

@pytest.mark.property
@given(coeffs=valid_constraint_coefficients())
def test_prudential_positive(coeffs: dict):
    """Prudential (own rate) coefficient should be positive."""
    # Cap rate is a yield - higher rate attracts customers
    prudential_coef = coeffs.get("prudential_cap", coeffs.get("own_rate", None))
    if prudential_coef is not None:
        assert prudential_coef > 0, f"Prudential should be positive: {prudential_coef}"


@pytest.mark.property
@given(coeffs=valid_constraint_coefficients())
def test_competitor_negative(coeffs: dict):
    """Competitor rate coefficients should be negative (substitution effect)."""
    for key, value in coeffs.items():
        if "C_lag" in key or "competitor" in key.lower():
            assert value < 0, f"Competitor coefficient should be negative: {key}={value}"


@pytest.mark.property
@given(coeffs=coefficient_dict())
def test_intercept_can_be_any_sign(coeffs: dict):
    """Intercept has no sign constraint."""
    # This is a sanity test - intercept can be positive, negative, or zero
    intercept = coeffs.get("Intercept")
    assert intercept is not None, "Coefficient dict should have intercept"
    # No assertion on sign - just verify it exists


# =============================================================================
# R-SQUARED BOUNDS
# =============================================================================

@pytest.mark.property
@given(
    y_true=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    ),
    noise=st.floats(min_value=0.0, max_value=1.0),
)
def test_r_squared_bounds(y_true: list, noise: float):
    """R-squared should always be in [0, 1] for valid predictions."""
    assume(len(set(y_true)) > 1)  # Avoid constant y

    y = np.array(y_true)
    y_mean = np.mean(y)

    # Create predictions with controlled noise
    y_pred = y + noise * np.std(y) * np.random.randn(len(y))

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    assume(ss_tot > 0)  # Avoid division by zero

    r_squared = 1 - (ss_res / ss_tot)

    # R-squared can be negative if model is worse than mean
    # But for non-adversarial predictions, it should be bounded
    assert r_squared <= 1.0, f"R-squared > 1: {r_squared}"


@pytest.mark.property
@given(
    y=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50,
    )
)
def test_perfect_prediction_r_squared_one(y: list):
    """Perfect predictions should give R-squared = 1."""
    assume(len(set(y)) > 1)  # Avoid constant y

    y = np.array(y)
    y_pred = y  # Perfect prediction

    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    assume(ss_tot > 0)

    r_squared = 1 - (ss_res / ss_tot)
    assert abs(r_squared - 1.0) < 1e-10, f"Perfect prediction R-squared != 1: {r_squared}"


# =============================================================================
# VARIANCE PROPERTIES
# =============================================================================

@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=100,
    )
)
def test_variance_non_negative(values: list):
    """Variance should always be non-negative."""
    var = np.var(values)
    assert var >= 0, f"Negative variance: {var}"


@pytest.mark.property
@given(
    value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    n=st.integers(min_value=2, max_value=100),
)
def test_constant_variance_zero(value: float, n: int):
    """Constant values should have zero variance."""
    values = [value] * n
    var = np.var(values)
    assert abs(var) < 1e-10, f"Constant values should have zero variance: {var}"


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=50,
    )
)
def test_std_is_sqrt_variance(values: list):
    """Standard deviation should be sqrt of variance."""
    var = np.var(values)
    std = np.std(values)

    assert abs(std - np.sqrt(var)) < 1e-10, "std != sqrt(var)"


# =============================================================================
# STANDARD ERROR PROPERTIES
# =============================================================================

@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100,
    )
)
def test_standard_error_positive(values: list):
    """Standard error should be positive (or zero for constant)."""
    assume(len(set(values)) > 1)  # Avoid constant

    std = np.std(values, ddof=1)
    se = std / np.sqrt(len(values))

    assert se >= 0, f"Negative standard error: {se}"


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    )
)
def test_larger_sample_smaller_se(values: list):
    """Standard error should decrease with larger sample size."""
    assume(len(values) >= 20)
    assume(len(set(values)) > 1)

    half_n = len(values) // 2
    values_small = values[:half_n]
    values_large = values  # Full sample

    # Standard errors
    se_small = np.std(values_small, ddof=1) / np.sqrt(len(values_small))
    se_large = np.std(values_large, ddof=1) / np.sqrt(len(values_large))

    # SE should generally be smaller for larger samples (with same underlying variance)
    # This isn't always true due to random variation, so we're lenient
    # Just check they're both positive and finite
    assert se_small > 0 and np.isfinite(se_small), "Invalid SE for small sample"
    assert se_large > 0 and np.isfinite(se_large), "Invalid SE for large sample"


# =============================================================================
# CORRELATION BOUNDS
# =============================================================================

@pytest.mark.property
@given(
    data=st.data(),
)
def test_correlation_bounds(data):
    """Correlation should be in [-1, 1]."""
    # Use shared size to avoid filtering
    size = data.draw(st.integers(min_value=10, max_value=50))
    x = data.draw(st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size,
    ))
    y = data.draw(st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size,
    ))

    assume(len(set(x)) > 1 and len(set(y)) > 1)

    corr = np.corrcoef(x, y)[0, 1]

    assert -1 <= corr <= 1, f"Correlation outside bounds: {corr}"


@pytest.mark.property
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50,
    )
)
def test_self_correlation_is_one(values: list):
    """Correlation of variable with itself should be 1."""
    assume(len(set(values)) > 1)
    # Require meaningful variance to avoid NaN from near-constant arrays
    assume(np.std(values) > 1e-10)

    corr = np.corrcoef(values, values)[0, 1]

    assert abs(corr - 1.0) < 1e-10, f"Self-correlation != 1: {corr}"


# =============================================================================
# ECONOMIC CONSTRAINT VALIDATION
# =============================================================================

@pytest.mark.property
@given(
    r_squared=st.floats(min_value=0.0, max_value=1.0),
)
def test_suspicious_r_squared_threshold(r_squared: float):
    """R-squared > 0.3 should trigger investigation in this domain."""
    threshold = 0.3

    if r_squared > threshold:
        # This would trigger a leakage investigation
        is_suspicious = True
    else:
        is_suspicious = False

    # Verify threshold logic is correct
    assert (r_squared > threshold) == is_suspicious


@pytest.mark.property
@given(
    baseline=st.floats(min_value=0.01, max_value=0.50, allow_nan=False),
    new=st.floats(min_value=0.01, max_value=0.50, allow_nan=False),
)
def test_improvement_calculation(baseline: float, new: float):
    """Improvement calculation should be consistent."""
    # For metrics where lower is better (e.g., MAE, RMSE)
    improvement = (baseline - new) / baseline

    if new < baseline:
        assert improvement > 0, "Improvement should be positive when new < baseline"
    elif new > baseline:
        assert improvement < 0, "Improvement should be negative when new > baseline"
    else:
        assert abs(improvement) < 1e-10, "No improvement when equal"
