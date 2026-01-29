"""
Hypothesis Strategies and Fixtures for Property-Based Testing.

Provides reusable strategies for generating:
- Rate values (cap rates, competitor rates)
- DataFrames matching expected schemas
- Feature configurations
- Time series data
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import strategies as st, settings, Verbosity
from datetime import datetime, timedelta
from typing import Dict, Any


# =============================================================================
# HYPOTHESIS SETTINGS
# =============================================================================

# Default settings for all property tests
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=50, deadline=None)
settings.register_profile("dev", max_examples=20, verbosity=Verbosity.verbose)
settings.register_profile("thorough", max_examples=500, deadline=None)

settings.load_profile("default")


# =============================================================================
# RATE VALUE STRATEGIES
# =============================================================================

@st.composite
def rate_values(draw, min_rate=0.01, max_rate=0.20):
    """Generate valid rate values (as decimals, e.g., 0.045 for 4.5%)."""
    return draw(st.floats(
        min_value=min_rate,
        max_value=max_rate,
        allow_nan=False,
        allow_infinity=False,
    ))


@st.composite
def rate_basis_points(draw, min_bp=10, max_bp=2000):
    """Generate rate values in basis points (10-2000 bp typical range)."""
    return draw(st.integers(min_value=min_bp, max_value=max_bp))


@st.composite
def rate_spread(draw, min_spread=-500, max_spread=500):
    """Generate spread values in basis points."""
    return draw(st.integers(min_value=min_spread, max_value=max_spread))


# =============================================================================
# DATAFRAME STRATEGIES
# =============================================================================

@st.composite
def price_elasticity_dataframe(draw, min_rows=10, max_rows=100):
    """Generate a DataFrame with schema matching price elasticity analysis."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate date range
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_rows)]

    # Generate rate data
    own_rates = [draw(rate_values()) for _ in range(n_rows)]
    competitor_rates = [draw(rate_values()) for _ in range(n_rows)]

    # Generate sales (positive integers)
    sales = draw(st.lists(
        st.integers(min_value=0, max_value=10000),
        min_size=n_rows,
        max_size=n_rows,
    ))

    return pd.DataFrame({
        "date": dates,
        "own_rate": own_rates,
        "competitor_rate": competitor_rates,
        "sales": sales,
    })


@st.composite
def feature_matrix(draw, n_features=5, min_rows=50, max_rows=200):
    """Generate a feature matrix for regression."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_feat = draw(st.integers(min_value=2, max_value=n_features))

    data = {}
    for i in range(n_feat):
        col_name = f"feature_{i}"
        values = draw(st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        ))
        data[col_name] = values

    return pd.DataFrame(data)


# =============================================================================
# COEFFICIENT STRATEGIES
# =============================================================================

@st.composite
def coefficient_dict(draw, n_features=5):
    """Generate a dictionary of model coefficients."""
    n = draw(st.integers(min_value=2, max_value=n_features))
    coeffs = {}

    for i in range(n):
        name = f"feature_{i}"
        value = draw(st.floats(min_value=-10, max_value=10, allow_nan=False))
        coeffs[name] = value

    # Always include intercept
    coeffs["Intercept"] = draw(st.floats(min_value=-100, max_value=100, allow_nan=False))

    return coeffs


@st.composite
def valid_constraint_coefficients(draw):
    """Generate coefficients that satisfy economic constraints."""
    # Own rate (prudential) should be positive
    prudential_coef = draw(st.floats(min_value=0.01, max_value=5.0))

    # Competitor rates should be negative (substitution effect)
    competitor_coefs = {
        f"C_lag_{i}": draw(st.floats(min_value=-5.0, max_value=-0.01))
        for i in range(1, 4)
    }

    return {
        "prudential_cap": prudential_coef,
        **competitor_coefs,
        "Intercept": draw(st.floats(min_value=-10, max_value=10, allow_nan=False)),
    }


# =============================================================================
# TIME SERIES STRATEGIES
# =============================================================================

@st.composite
def date_range_strategy(draw, min_days=30, max_days=365):
    """Generate a date range for time series analysis."""
    n_days = draw(st.integers(min_value=min_days, max_value=max_days))
    start_date = datetime(2023, 1, 1) + timedelta(days=draw(st.integers(0, 365)))
    return pd.date_range(start=start_date, periods=n_days, freq="D")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_rate():
    """Provide a sample rate value."""
    return 0.045  # 4.5%


@pytest.fixture
def sample_coefficients():
    """Provide sample coefficients satisfying constraints."""
    return {
        "prudential_cap": 0.5,
        "C_lag_1": -0.3,
        "C_lag_2": -0.2,
        "Intercept": 1.5,
    }


# =============================================================================
# EXPANDED STRATEGIES (Phase 5)
# =============================================================================

@st.composite
def sales_volume_strategy(draw, min_sales=0, max_sales=100000):
    """Generate realistic sales volume values."""
    return draw(st.integers(min_value=min_sales, max_value=max_sales))


@st.composite
def weekly_observations(draw, min_weeks=52, max_weeks=260):
    """Generate weekly time series observations (1-5 years)."""
    n_weeks = draw(st.integers(min_value=min_weeks, max_value=max_weeks))

    base_date = datetime(2020, 1, 6)  # Monday
    dates = [base_date + timedelta(weeks=i) for i in range(n_weeks)]

    return pd.DataFrame({
        'date': dates,
        'week_number': list(range(1, n_weeks + 1))
    })


@st.composite
def lag_configuration(draw, max_lags=10):
    """Generate lag feature configuration."""
    n_lags = draw(st.integers(min_value=1, max_value=max_lags))

    return {
        'max_lag_periods': n_lags,
        'lag_columns': draw(st.lists(
            st.sampled_from(['own_cap_rate', 'competitor_mean', 'macro_indicator']),
            min_size=1,
            max_size=3,
            unique=True
        ))
    }


@st.composite
def bootstrap_configuration(draw):
    """Generate bootstrap inference configuration."""
    return {
        'n_bootstrap': draw(st.integers(min_value=10, max_value=1000)),
        'confidence_level': draw(st.floats(min_value=0.90, max_value=0.99)),
        'random_state': draw(st.integers(min_value=0, max_value=2**31 - 1))
    }


@st.composite
def product_configuration(draw):
    """Generate product configuration."""
    return {
        'buffer_rate': draw(st.sampled_from(['10%', '20%', '30%'])),
        'term': draw(st.sampled_from(['6Y', '10Y'])),
        'product_name': 'FlexGuard indexed variable annuity'
    }


@st.composite
def outlier_threshold(draw):
    """Generate outlier detection threshold (quantile)."""
    return draw(st.floats(min_value=0.90, max_value=0.999))


@st.composite
def aggregation_config(draw):
    """Generate aggregation method configuration."""
    return {
        'method': draw(st.sampled_from(['weighted', 'top_n', 'median'])),
        'n_competitors': draw(st.integers(min_value=3, max_value=10)) if draw(st.booleans()) else None
    }


@st.composite
def polynomial_degree(draw):
    """Generate polynomial degree for feature engineering."""
    return draw(st.integers(min_value=2, max_value=4))


@st.composite
def feature_selection_config(draw):
    """Generate feature selection configuration."""
    return {
        'max_features': draw(st.integers(min_value=10, max_value=50)),
        'method': draw(st.sampled_from(['lasso', 'recursive', 'mutual_info'])),
        'threshold': draw(st.floats(min_value=0.001, max_value=0.1))
    }


@st.composite
def realistic_model_output(draw, n_observations=100):
    """Generate realistic model output for testing."""
    n_obs = draw(st.integers(min_value=50, max_value=n_observations))

    # Predictions (positive sales)
    predictions = draw(st.lists(
        st.floats(min_value=1000, max_value=100000, allow_nan=False, allow_infinity=False),
        min_size=n_obs,
        max_size=n_obs
    ))

    # Residuals (centered around 0)
    residuals = draw(st.lists(
        st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        min_size=n_obs,
        max_size=n_obs
    ))

    # Actual values (predictions + residuals, but keep positive)
    actuals = [max(pred + resid, 0) for pred, resid in zip(predictions, residuals)]

    return {
        'predictions': np.array(predictions),
        'actuals': np.array(actuals),
        'residuals': np.array(residuals)
    }


@st.composite
def confidence_interval_strategy(draw):
    """Generate confidence intervals for predictions."""
    point_estimate = draw(st.floats(min_value=1000, max_value=100000))

    # CI width as percentage of point estimate
    width_pct = draw(st.floats(min_value=0.05, max_value=0.30))
    width = point_estimate * width_pct

    return {
        'point': point_estimate,
        'lower': point_estimate - width,
        'upper': point_estimate + width
    }


@st.composite
def metric_values(draw):
    """Generate realistic model performance metrics."""
    return {
        'R²': draw(st.floats(min_value=0.50, max_value=0.99)),
        'MAPE': draw(st.floats(min_value=0.01, max_value=0.30)),
        'AIC': draw(st.floats(min_value=100, max_value=10000)),
        'BIC': draw(st.floats(min_value=100, max_value=10000))
    }
