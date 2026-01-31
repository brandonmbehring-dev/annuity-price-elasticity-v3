"""
Statistical Invariant Property Tests
====================================

Property-based tests for statistical functions using Hypothesis.
These tests verify mathematical invariants that must hold regardless of input data.

Invariants Tested:
- Bonferroni correction always reduces alpha
- FDR correction bounded by original alpha * n_tests
- R² bounded in [-inf, 1]
- MAPE is always non-negative
- Confidence interval coverage properties

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from typing import List


# =============================================================================
# STATISTICAL CORRECTION INVARIANTS
# =============================================================================


class TestBonferroniInvariants:
    """Property tests for Bonferroni correction invariants."""

    @given(
        n_tests=st.integers(min_value=1, max_value=1000),
        alpha=st.floats(min_value=0.001, max_value=0.20, allow_nan=False)
    )
    def test_bonferroni_always_reduces_alpha(self, n_tests: int, alpha: float):
        """
        Invariant: Bonferroni-corrected alpha is always <= original alpha.

        Mathematical property: alpha_corrected = alpha / n_tests <= alpha (for n_tests >= 1)
        """
        corrected_alpha = alpha / n_tests
        assert corrected_alpha <= alpha, (
            f"Bonferroni correction increased alpha: "
            f"original={alpha}, corrected={corrected_alpha}, n_tests={n_tests}"
        )

    @given(
        n_tests=st.integers(min_value=1, max_value=1000),
        alpha=st.floats(min_value=0.001, max_value=0.20, allow_nan=False)
    )
    def test_bonferroni_corrected_alpha_positive(self, n_tests: int, alpha: float):
        """
        Invariant: Bonferroni-corrected alpha is always positive.
        """
        corrected_alpha = alpha / n_tests
        assert corrected_alpha > 0, (
            f"Bonferroni correction produced non-positive alpha: {corrected_alpha}"
        )

    @given(
        n_tests=st.integers(min_value=1, max_value=10000),
        alpha=st.floats(min_value=0.001, max_value=0.10, allow_nan=False)
    )
    def test_family_wise_error_rate_controlled(self, n_tests: int, alpha: float):
        """
        Invariant: Family-wise error rate is controlled at alpha level.

        If each test uses alpha/n_tests, the probability of at least one
        Type I error is bounded by alpha (Bonferroni inequality).
        """
        corrected_alpha = alpha / n_tests

        # FWER upper bound: 1 - (1 - alpha_corrected)^n_tests <= alpha
        # For small alpha_corrected, this approximates to n_tests * alpha_corrected = alpha
        fwer_upper_bound = n_tests * corrected_alpha
        assert fwer_upper_bound <= alpha + 1e-10, (
            f"FWER upper bound {fwer_upper_bound} exceeds alpha {alpha}"
        )


class TestFDRInvariants:
    """Property tests for False Discovery Rate invariants."""

    @given(
        pvalues=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        ),
        alpha=st.floats(min_value=0.01, max_value=0.20, allow_nan=False)
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_fdr_rejections_bounded(self, pvalues: np.ndarray, alpha: float):
        """
        Invariant: Number of FDR rejections <= n_tests.
        """
        assume(len(pvalues) > 0)
        assume(not np.any(np.isnan(pvalues)))

        n_tests = len(pvalues)

        # Benjamini-Hochberg procedure
        sorted_pvals = np.sort(pvalues)
        thresholds = alpha * np.arange(1, n_tests + 1) / n_tests

        # Count rejections
        rejections = np.sum(sorted_pvals <= thresholds)

        assert rejections <= n_tests, (
            f"FDR rejections ({rejections}) exceed n_tests ({n_tests})"
        )

    @given(
        n_tests=st.integers(min_value=1, max_value=100),
        alpha=st.floats(min_value=0.01, max_value=0.10, allow_nan=False)
    )
    def test_fdr_threshold_increases_with_rank(self, n_tests: int, alpha: float):
        """
        Invariant: BH threshold at rank k is k * alpha / n.

        Thresholds should increase monotonically with rank.
        """
        thresholds = [alpha * k / n_tests for k in range(1, n_tests + 1)]

        for i in range(1, len(thresholds)):
            assert thresholds[i] > thresholds[i-1], (
                f"BH thresholds not monotonically increasing: "
                f"threshold[{i-1}]={thresholds[i-1]}, threshold[{i}]={thresholds[i]}"
            )


# =============================================================================
# MODEL METRIC INVARIANTS
# =============================================================================


class TestRSquaredInvariants:
    """Property tests for R² metric invariants."""

    @given(
        n=st.integers(min_value=10, max_value=50)
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow], deadline=None)
    def test_r2_bounded_above_by_one(self, n: int):
        """
        Invariant: R² <= 1 always.

        R² = 1 - (SS_res / SS_tot), and SS_res >= 0, so R² <= 1.
        """
        # Generate data with known structure
        np.random.seed(n)
        y_true = np.random.randn(n) * 10 + 50
        y_pred = y_true + np.random.randn(n) * 5  # Predictions with noise

        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)

        assert r2 <= 1 + 1e-10, f"R² ({r2}) exceeds 1"

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_r2_equals_one_for_perfect_predictions(self, y_true: np.ndarray):
        """
        Invariant: R² = 1 when predictions equal actuals.
        """
        assume(len(y_true) >= 2)
        assume(not np.any(np.isnan(y_true)))
        assume(np.var(y_true) > 1e-10)

        from sklearn.metrics import r2_score

        # Perfect prediction: y_pred = y_true
        r2 = r2_score(y_true, y_true)

        np.testing.assert_almost_equal(
            r2, 1.0, decimal=10,
            err_msg="R² should equal 1 for perfect predictions"
        )

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=100),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_r2_zero_for_mean_prediction(self, y_true: np.ndarray):
        """
        Invariant: R² = 0 when predicting the mean for all observations.
        """
        assume(len(y_true) >= 2)
        assume(not np.any(np.isnan(y_true)))
        assume(np.var(y_true) > 1e-10)

        from sklearn.metrics import r2_score

        # Mean prediction
        y_pred = np.full_like(y_true, np.mean(y_true))
        r2 = r2_score(y_true, y_pred)

        np.testing.assert_almost_equal(
            r2, 0.0, decimal=10,
            err_msg="R² should equal 0 for mean predictions"
        )


class TestMAPEInvariants:
    """Property tests for MAPE metric invariants."""

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False)
        ),
        y_pred=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_mape_non_negative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Invariant: MAPE >= 0 always.

        MAPE is mean of absolute values, so it's always non-negative.
        """
        assume(len(y_true) == len(y_pred))
        assume(not np.any(np.isnan(y_true)))
        assume(not np.any(np.isnan(y_pred)))
        assume(np.all(y_true > 0))  # MAPE requires positive actuals

        # Manual MAPE calculation
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        assert mape >= 0, f"MAPE ({mape}) is negative"

    @given(
        y_true=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_mape_zero_for_perfect_predictions(self, y_true: np.ndarray):
        """
        Invariant: MAPE = 0 when predictions equal actuals.
        """
        assume(not np.any(np.isnan(y_true)))
        assume(np.all(y_true > 0))

        # Perfect prediction
        mape = np.mean(np.abs((y_true - y_true) / y_true)) * 100

        np.testing.assert_almost_equal(
            mape, 0.0, decimal=10,
            err_msg="MAPE should equal 0 for perfect predictions"
        )


# =============================================================================
# CONFIDENCE INTERVAL INVARIANTS
# =============================================================================


class TestConfidenceIntervalInvariants:
    """Property tests for confidence interval invariants."""

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=30, max_value=200),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        ),
        confidence_level=st.floats(min_value=0.80, max_value=0.99)
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_ci_lower_less_than_upper(self, values: np.ndarray, confidence_level: float):
        """
        Invariant: CI lower bound < CI upper bound (for non-constant data).
        """
        assume(not np.any(np.isnan(values)))
        assume(np.std(values) > 1e-10)  # Non-constant data

        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)

        assert lower <= upper, (
            f"CI lower bound ({lower}) > upper bound ({upper})"
        )

    @given(
        confidence_level=st.floats(min_value=0.80, max_value=0.99)
    )
    def test_higher_confidence_wider_interval(self, confidence_level: float):
        """
        Invariant: Higher confidence level produces wider interval.
        """
        np.random.seed(42)
        values = np.random.randn(100)

        # Lower confidence
        alpha_low = 1 - confidence_level
        lower_low = np.percentile(values, 100 * (alpha_low / 2))
        upper_low = np.percentile(values, 100 * (1 - alpha_low / 2))
        width_low = upper_low - lower_low

        # Higher confidence (99%)
        alpha_high = 0.01
        lower_high = np.percentile(values, 100 * (alpha_high / 2))
        upper_high = np.percentile(values, 100 * (1 - alpha_high / 2))
        width_high = upper_high - lower_high

        # 99% CI should be wider than lower confidence CI
        if confidence_level < 0.99:
            assert width_high >= width_low - 1e-10, (
                f"99% CI width ({width_high:.4f}) < {confidence_level:.0%} CI width ({width_low:.4f})"
            )


# =============================================================================
# DATA VALIDATION INVARIANTS
# =============================================================================


class TestDataValidationInvariants:
    """Property tests for data validation invariants."""

    @given(
        n_rows=st.integers(min_value=1, max_value=100),
        n_cols=st.integers(min_value=1, max_value=10)
    )
    def test_dataframe_shape_preserved(self, n_rows: int, n_cols: int):
        """
        Invariant: DataFrame shape is preserved through non-filtering operations.
        """
        df = pd.DataFrame(
            np.random.randn(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )

        # Non-filtering operation: standardization
        standardized = (df - df.mean()) / (df.std() + 1e-10)

        assert standardized.shape == df.shape, (
            f"Shape changed: original={df.shape}, after={standardized.shape}"
        )

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_standardization_produces_unit_variance(self, values: np.ndarray):
        """
        Invariant: Standardized data has unit variance (approximately).
        """
        assume(not np.any(np.isnan(values)))
        assume(np.std(values) > 1e-10)

        standardized = (values - np.mean(values)) / np.std(values)
        variance = np.var(standardized)

        np.testing.assert_almost_equal(
            variance, 1.0, decimal=5,
            err_msg=f"Standardized variance ({variance}) should be ~1"
        )

    @given(
        values=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        )
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_standardization_produces_zero_mean(self, values: np.ndarray):
        """
        Invariant: Standardized data has zero mean (approximately).

        Note: For constant arrays, standardization is undefined, so we require variance.
        """
        assume(not np.any(np.isnan(values)))
        assume(np.std(values) > 1e-10)  # Require non-constant data

        standardized = (values - np.mean(values)) / np.std(values)
        mean = np.mean(standardized)

        np.testing.assert_almost_equal(
            mean, 0.0, decimal=8,  # Relaxed for floating point precision
            err_msg=f"Standardized mean ({mean}) should be ~0"
        )
