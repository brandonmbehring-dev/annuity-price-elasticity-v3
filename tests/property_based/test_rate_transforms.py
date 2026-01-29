"""
Property-Based Tests for Rate Transformations.

Tests invariants that should hold for all rate transformations:
1. Bounds preservation - rates stay within valid ranges
2. Monotonicity - higher input rates produce higher/lower outputs (as expected)
3. Invertibility - transformations that should be invertible are
4. Sign preservation - signs are preserved or flipped as expected
"""

import pytest
import numpy as np
from hypothesis import given, assume, settings
from hypothesis import strategies as st

# Import strategies from conftest
from .conftest import rate_values, rate_basis_points, rate_spread


# =============================================================================
# BOUNDS PRESERVATION
# =============================================================================

@pytest.mark.property
@given(rate=rate_values(min_rate=0.01, max_rate=0.20))
def test_rate_to_basis_points_bounds(rate: float):
    """Converting rate to basis points should produce value in valid range."""
    bp = rate * 10000
    assert 100 <= bp <= 2000, f"Basis points {bp} outside expected range"


@pytest.mark.property
@given(bp=rate_basis_points(min_bp=10, max_bp=2000))
def test_basis_points_to_rate_bounds(bp: int):
    """Converting basis points to rate should produce value in valid range."""
    rate = bp / 10000
    assert 0.001 <= rate <= 0.20, f"Rate {rate} outside expected range"


@pytest.mark.property
@given(rate=rate_values())
def test_rate_is_positive(rate: float):
    """Cap rates should always be positive (yield, not cost)."""
    assert rate > 0, "Cap rates must be positive"


# =============================================================================
# MONOTONICITY
# =============================================================================

@pytest.mark.property
@given(
    rate1=rate_values(min_rate=0.01, max_rate=0.10),
    rate2=rate_values(min_rate=0.11, max_rate=0.20),
)
def test_higher_rate_higher_bp(rate1: float, rate2: float):
    """Higher rate should convert to higher basis points."""
    assume(rate1 < rate2)
    bp1 = rate1 * 10000
    bp2 = rate2 * 10000
    assert bp1 < bp2, "Monotonicity violated: higher rate should give higher bp"


@pytest.mark.property
@given(
    own_rate=rate_values(),
    competitor_rate=rate_values(),
)
def test_spread_sign_consistency(own_rate: float, competitor_rate: float):
    """Spread sign should be consistent with rate comparison."""
    spread = (own_rate - competitor_rate) * 10000  # In basis points

    if own_rate > competitor_rate:
        assert spread > 0, "Positive spread when own > competitor"
    elif own_rate < competitor_rate:
        assert spread < 0, "Negative spread when own < competitor"
    else:
        assert spread == 0, "Zero spread when rates equal"


# =============================================================================
# INVERTIBILITY
# =============================================================================

@pytest.mark.property
@given(rate=rate_values())
def test_rate_bp_roundtrip(rate: float):
    """Rate -> BP -> Rate should be identity (within floating point precision)."""
    bp = rate * 10000
    recovered = bp / 10000
    assert abs(rate - recovered) < 1e-10, f"Roundtrip failed: {rate} -> {bp} -> {recovered}"


@pytest.mark.property
@given(bp=rate_basis_points())
def test_bp_rate_roundtrip(bp: int):
    """BP -> Rate -> BP should be identity."""
    rate = bp / 10000
    recovered_bp = rate * 10000
    assert abs(bp - recovered_bp) < 1e-10, f"Roundtrip failed: {bp} -> {rate} -> {recovered_bp}"


# =============================================================================
# SPREAD CALCULATIONS
# =============================================================================

@pytest.mark.property
@given(
    own_rate=rate_values(),
    competitor_rate=rate_values(),
)
def test_spread_is_difference(own_rate: float, competitor_rate: float):
    """Spread should be the difference between own and competitor rates."""
    spread_bp = (own_rate - competitor_rate) * 10000
    expected = own_rate * 10000 - competitor_rate * 10000

    assert abs(spread_bp - expected) < 1e-8, "Spread calculation inconsistent"


@pytest.mark.property
@given(
    rate=rate_values(),
    spread=rate_spread(),
)
def test_rate_plus_spread_bounds(rate: float, spread: int):
    """Adding spread to rate should still produce valid rate (or be rejected)."""
    new_rate = rate + spread / 10000

    # Rate could go negative or very high - this is expected in some scenarios
    # The key invariant is that the math is correct
    expected_spread = (new_rate - rate) * 10000
    assert abs(expected_spread - spread) < 1e-6, "Spread arithmetic error"


# =============================================================================
# LAG CONSISTENCY
# =============================================================================

@pytest.mark.property
@given(
    rates=st.lists(rate_values(), min_size=5, max_size=20),
    lag=st.integers(min_value=1, max_value=4),
)
def test_lag_reduces_series_length(rates: list, lag: int):
    """Lagging a series should reduce its effective length."""
    lagged = rates[lag:]
    original_aligned = rates[:-lag] if lag > 0 else rates

    # After alignment, lengths should match
    assert len(lagged) == len(original_aligned), "Lag alignment length mismatch"


@pytest.mark.property
@given(
    rates=st.lists(rate_values(), min_size=10, max_size=50),
)
def test_lag0_equals_current(rates: list):
    """Lag-0 should equal the current value (identity)."""
    lag0 = rates[0:]  # Lag 0
    assert lag0 == rates, "Lag-0 should be identity"


# =============================================================================
# AGGREGATION PROPERTIES
# =============================================================================

@pytest.mark.property
@given(
    rates=st.lists(rate_values(), min_size=3, max_size=10),
    weights=st.lists(st.floats(min_value=0.1, max_value=1.0), min_size=3, max_size=10),
)
def test_weighted_mean_bounds(rates: list, weights: list):
    """Weighted mean should be bounded by min and max of inputs (within float tolerance)."""
    assume(len(rates) == len(weights))
    assume(sum(weights) > 0)

    normalized_weights = [w / sum(weights) for w in weights]
    weighted_mean = sum(r * w for r, w in zip(rates, normalized_weights))

    # Allow small floating-point tolerance (1e-12)
    eps = 1e-12
    assert min(rates) - eps <= weighted_mean <= max(rates) + eps, "Weighted mean outside input bounds"


@pytest.mark.property
@given(rates=st.lists(rate_values(), min_size=1, max_size=10))
def test_simple_mean_bounds(rates: list):
    """Simple mean should be bounded by min and max of inputs (within float tolerance)."""
    mean = sum(rates) / len(rates)
    # Allow small floating-point tolerance (1e-12)
    eps = 1e-12
    assert min(rates) - eps <= mean <= max(rates) + eps, "Mean outside input bounds"


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

@pytest.mark.property
@given(rate=rate_values())
def test_no_nan_from_valid_rate(rate: float):
    """Valid rate input should never produce NaN."""
    bp = rate * 10000
    assert not np.isnan(bp), "NaN produced from valid input"


@pytest.mark.property
@given(
    rate1=rate_values(),
    rate2=rate_values(),
)
def test_spread_no_nan(rate1: float, rate2: float):
    """Spread calculation should never produce NaN from valid inputs."""
    spread = rate1 - rate2
    assert not np.isnan(spread), "NaN in spread calculation"
