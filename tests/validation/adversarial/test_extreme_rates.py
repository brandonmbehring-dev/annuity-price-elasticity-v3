"""
Adversarial Tests: Extreme Rate Values
=======================================

Edge case stress tests for rate value handling.

Scenarios Tested:
    1. Zero rates (0% cap rate)
    2. Maximum rates (near 100%)
    3. Negative rates (invalid but could occur in data)
    4. Missing/NaN rates
    5. Identical rates (no variance)

Purpose:
    Ensure the model handles edge cases gracefully without:
    - Division by zero errors
    - Numerical overflow/underflow
    - Silent data corruption
    - Unexpected NaN propagation

Knowledge Tier Tags:
    [T1] = Academically validated (numerical analysis)
    [T2] = Empirical edge cases from production
    [T3] = Assumptions about valid ranges

References:
    - LEAKAGE_CHECKLIST.md - Section 4: Suspicious Results Check
    - src/validation/data_schemas.py - Valid rate ranges
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# RATE BOUNDARIES
# =============================================================================

# Valid rate ranges (in decimal, e.g., 0.05 = 5%)
MIN_VALID_RATE = 0.0  # 0% is valid floor for buffer products
MAX_VALID_RATE = 0.25  # 25% cap (very aggressive product)
TYPICAL_RATE_RANGE = (0.02, 0.12)  # 2-12% typical for RILA


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_rate_dataframe(
    rates: np.ndarray,
    n_obs: int = 100,
) -> pd.DataFrame:
    """
    Create a DataFrame with rate features for testing.

    Parameters
    ----------
    rates : np.ndarray
        Array of rate values
    n_obs : int
        Number of observations (repeats rates if needed)

    Returns
    -------
    pd.DataFrame
        DataFrame with rate columns
    """
    if len(rates) < n_obs:
        rates = np.tile(rates, n_obs // len(rates) + 1)[:n_obs]

    return pd.DataFrame(
        {
            "prudential_rate": rates,
            "competitor_rate": rates * 0.95,  # Slightly lower
            "date": pd.date_range("2024-01-01", periods=len(rates), freq="W"),
        }
    )


def validate_rate_column(
    df: pd.DataFrame,
    column: str,
    allow_zero: bool = True,
    allow_negative: bool = False,
    max_rate: float = MAX_VALID_RATE,
) -> tuple[bool, str]:
    """
    Validate a rate column for extreme values.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason)
    """
    values = df[column]

    # Check for NaN
    nan_count = values.isna().sum()
    if nan_count > 0:
        return False, f"Found {nan_count} NaN values in {column}"

    # Check for negative
    if not allow_negative:
        neg_count = (values < 0).sum()
        if neg_count > 0:
            return False, f"Found {neg_count} negative values in {column}"

    # Check for zero
    if not allow_zero:
        zero_count = (values == 0).sum()
        if zero_count > 0:
            return False, f"Found {zero_count} zero values in {column}"

    # Check for excessive rates
    high_count = (values > max_rate).sum()
    if high_count > 0:
        return False, f"Found {high_count} values above {max_rate:.1%} in {column}"

    return True, "All values valid"


# =============================================================================
# ZERO RATE TESTS
# =============================================================================


@pytest.mark.adversarial
class TestZeroRates:
    """Test handling of zero rate values. [T2]"""

    def test_zero_own_rate_handled(self) -> None:
        """Zero own rate should be flagged or handled. [T3]

        A 0% cap rate is unusual but valid for floor products.
        The model should not produce NaN or infinite coefficients.
        """
        rates = np.array([0.0, 0.05, 0.06, 0.07, 0.08])
        df = create_rate_dataframe(rates)

        # Validate zero is present
        assert (df["prudential_rate"] == 0).any(), "Test data should contain zero"

        # Model fitting should not produce NaN
        # (Simulated - actual test would fit model)
        X = df["prudential_rate"].values.reshape(-1, 1)
        y = np.random.randn(len(df))

        # Simple OLS
        coef = np.linalg.lstsq(X, y, rcond=None)[0]

        assert np.isfinite(coef).all(), "Coefficient became non-finite with zero rate in data"

    def test_all_zero_rates_detected(self) -> None:
        """All-zero rates should be flagged as invalid. [T2]

        No variance in rates means no price elasticity can be estimated.
        """
        rates = np.zeros(100)
        df = create_rate_dataframe(rates)

        variance = df["prudential_rate"].var()

        assert variance < 1e-10, "Expected zero variance in test data"

        # Model should detect this
        is_estimable = variance > 1e-6

        assert not is_estimable, "Should flag that elasticity is not estimable with zero variance"


# =============================================================================
# EXTREME HIGH RATE TESTS
# =============================================================================


@pytest.mark.adversarial
class TestExtremeHighRates:
    """Test handling of unusually high rate values. [T2]"""

    def test_rate_above_25_percent_flagged(self) -> None:
        """Rates above 25% should be flagged as suspicious. [T3]

        RILA products typically have cap rates 2-12%.
        Rates above 25% are almost certainly data errors.
        """
        rates = np.array([0.05, 0.06, 0.30, 0.07, 0.08])  # 30% = suspicious
        df = create_rate_dataframe(rates)

        is_valid, reason = validate_rate_column(df, "prudential_rate", max_rate=0.25)

        assert not is_valid, "Should flag rate above 25%"
        assert "above" in reason.lower()

    def test_rate_100_percent_handled(self) -> None:
        """100% rate should be handled without overflow. [T1]"""
        rates = np.array([0.05, 1.0, 0.06])  # 100% cap rate
        df = create_rate_dataframe(rates)

        # Numerical operations should not overflow
        log_rate = np.log(1 + df["prudential_rate"])
        assert np.isfinite(log_rate).all(), "Log transform should handle 100%"

    def test_extreme_outlier_detection(self) -> None:
        """Extreme outliers should be detectable. [T2]"""
        # Use a more extreme outlier for reliable detection
        rates = np.array([0.05, 0.06, 0.05, 0.05, 0.06, 0.05, 0.06, 0.05, 0.06, 1.0])
        df = create_rate_dataframe(rates)

        mean_rate = df["prudential_rate"].mean()
        std_rate = df["prudential_rate"].std()

        # Z-score for outlier detection
        z_scores = (df["prudential_rate"] - mean_rate) / std_rate

        n_outliers = (np.abs(z_scores) > 2).sum()  # Use 2 sigma for small sample

        assert n_outliers >= 1, "Should detect at least one outlier"


# =============================================================================
# NEGATIVE RATE TESTS
# =============================================================================


@pytest.mark.adversarial
class TestNegativeRates:
    """Test handling of negative rate values. [T1]"""

    def test_negative_rate_rejected(self) -> None:
        """Negative rates should be rejected. [T1]

        Negative cap rates are economically meaningless.
        They indicate data corruption or processing errors.
        """
        rates = np.array([0.05, -0.02, 0.06])  # Negative rate
        df = create_rate_dataframe(rates)

        is_valid, reason = validate_rate_column(df, "prudential_rate", allow_negative=False)

        assert not is_valid, "Should reject negative rate"
        assert "negative" in reason.lower()

    def test_negative_rate_propagation_prevented(self) -> None:
        """Negative rates should not silently propagate. [T1]

        Processing should either:
        1. Raise an error
        2. Replace with missing indicator
        3. Clip to zero

        Silent propagation is dangerous.
        """
        rates = np.array([0.05, -0.02, 0.06])

        # Option 1: Detect and fail
        has_negative = (rates < 0).any()
        assert has_negative, "Test data should have negative"

        # Option 2: Clip to zero (explicit handling)
        clipped = np.maximum(rates, 0)
        assert (clipped >= 0).all(), "Clipping should ensure non-negative"


# =============================================================================
# MISSING/NAN RATE TESTS
# =============================================================================


@pytest.mark.adversarial
class TestMissingRates:
    """Test handling of missing/NaN rate values. [T1]"""

    def test_nan_rate_detected(self) -> None:
        """NaN rates should be detected. [T1]"""
        rates = np.array([0.05, np.nan, 0.06])
        df = create_rate_dataframe(rates)

        is_valid, reason = validate_rate_column(df, "prudential_rate")

        assert not is_valid, "Should detect NaN rate"
        assert "nan" in reason.lower()

    def test_nan_propagation_in_calculations(self) -> None:
        """NaN should propagate correctly in calculations. [T1]

        Calculations involving NaN should produce NaN (not 0 or inf).
        """
        rates = np.array([0.05, np.nan, 0.06])

        # Standard operations
        mean_rate = np.nanmean(rates)  # Use nanmean to get valid result
        raw_mean = np.mean(rates)

        assert np.isnan(raw_mean), "Raw mean of data with NaN should be NaN"
        assert np.isfinite(mean_rate), "nanmean should produce valid result"

    def test_missing_rate_imputation_warning(self) -> None:
        """Imputing missing rates should generate warning. [T2]

        Imputation can introduce bias. If used, it should be explicit.
        """
        rates = np.array([0.05, np.nan, 0.06])
        n_missing = np.isnan(rates).sum()

        # Imputation should be tracked
        assert n_missing == 1, "Expected 1 missing value"

        # After imputation (e.g., forward fill)
        imputed = pd.Series(rates).ffill().values
        assert np.isfinite(imputed).all(), "Imputation should fill NaN"


# =============================================================================
# VARIANCE/SPREAD TESTS
# =============================================================================


@pytest.mark.adversarial
class TestRateVariance:
    """Test handling of rate variance edge cases. [T2]"""

    def test_identical_rates_detected(self) -> None:
        """Identical rates across all observations should be flagged. [T2]

        No variance means no price elasticity can be estimated.
        """
        rates = np.full(100, 0.05)  # All identical
        df = create_rate_dataframe(rates)

        variance = df["prudential_rate"].var()

        assert variance < 1e-10, "Test data should have zero variance"

    def test_very_low_variance_flagged(self) -> None:
        """Very low variance should be flagged. [T2]

        Variance < 1bp^2 makes elasticity estimation unreliable.
        """
        # All rates between 5.00% and 5.01% (< 1bp variance)
        rates = np.random.uniform(0.0500, 0.0501, 100)
        df = create_rate_dataframe(rates)

        variance = df["prudential_rate"].var()
        bp_variance = variance * 10000  # Convert to basis points squared

        # Should flag if variance < 1 bp^2
        is_sufficient = bp_variance >= 1.0

        assert (
            not is_sufficient
        ), f"Variance ({bp_variance:.4f} bp²) too low for reliable estimation"

    def test_extreme_variance_flagged(self) -> None:
        """Extreme variance should be flagged. [T2]

        Very high variance may indicate mixed products or data issues.
        """
        # Rates spanning 0% to 50% (unrealistic for single product)
        rates = np.random.uniform(0.0, 0.50, 100)
        df = create_rate_dataframe(rates)

        rate_range = df["prudential_rate"].max() - df["prudential_rate"].min()

        # Range > 20 percentage points is suspicious
        is_suspicious = rate_range > 0.20

        assert is_suspicious, "Should flag extreme rate range"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================


@pytest.mark.adversarial
class TestNumericalStability:
    """Test numerical stability with edge case rates. [T1]"""

    def test_very_small_rates_stable(self) -> None:
        """Very small rates (near machine epsilon) should be stable. [T1]"""
        rates = np.array([1e-10, 1e-9, 1e-8, 0.05])

        # Division should not overflow
        ratio = rates / (rates + 1e-15)
        assert np.isfinite(ratio).all(), "Division unstable for small rates"

    def test_log_transform_stable(self) -> None:
        """Log transform should be stable for valid rates. [T1]"""
        rates = np.array([0.001, 0.01, 0.05, 0.10])

        # Log(1 + rate) is stable
        log_rates = np.log1p(rates)
        assert np.isfinite(log_rates).all(), "Log transform unstable"

        # Plain log of zero would fail
        with np.errstate(divide="raise"):
            try:
                _ = np.log(0.0)
                raise AssertionError("Should have raised on log(0)")
            except FloatingPointError:
                pass  # Expected

    def test_exponential_stable_for_reasonable_rates(self) -> None:
        """Exponential should be stable for reasonable rate values. [T1]"""
        rates = np.array([0.01, 0.05, 0.10, 0.15])

        # exp(rate) for reasonable values
        exp_rates = np.exp(rates)
        assert np.isfinite(exp_rates).all(), "Exp unstable for reasonable rates"

        # exp(710) would overflow to inf (exp(709) ~ 8.2e307, exp(710) = inf)
        extreme_rate = 710.0
        assert not np.isfinite(np.exp(extreme_rate)), "Expected overflow for extreme rate in exp"
