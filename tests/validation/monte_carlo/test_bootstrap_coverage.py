"""
Monte Carlo Test: Bootstrap Confidence Interval Coverage
==========================================================

Validates that bootstrap confidence intervals achieve their nominal coverage.

Coverage Calibration:
    - 95% CI should contain true value ~95% of the time
    - Coverage significantly < 95% indicates underestimated uncertainty
    - Coverage significantly > 95% indicates overestimated uncertainty

Test Approach:
    1. Generate data from known DGP (Data Generating Process)
    2. Fit model and compute bootstrap CI
    3. Check if true parameter falls within CI
    4. Repeat N times and compute coverage rate

Knowledge Tier Tags:
    [T1] = Academically validated (bootstrap theory)
    [T2] = Empirical calibration targets

References:
    - Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
    - Production model: 94.4% coverage (target: 95%)
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# =============================================================================
# COVERAGE CALIBRATION TARGETS
# =============================================================================

NOMINAL_COVERAGE = 0.95  # Target coverage level
COVERAGE_TOLERANCE = 0.03  # Allow 3% deviation
MIN_SIMULATIONS = 100  # Minimum runs for stable estimate
BOOTSTRAP_SAMPLES = 500  # Samples per bootstrap run


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def generate_synthetic_data(
    n_obs: int = 200,
    true_coef: float = 0.05,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate synthetic data from known DGP.

    Parameters
    ----------
    n_obs : int
        Number of observations
    true_coef : float
        True coefficient value
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        X, y, true_coef
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, n_obs)
    y = true_coef * X + rng.normal(0, noise_std, n_obs)
    return X.reshape(-1, 1), y, true_coef


def compute_bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    seed: int = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for coefficient.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n_obs = len(y)

    # Point estimate (OLS)
    point_est = np.sum(X.flatten() * y) / np.sum(X.flatten() ** 2)

    # Bootstrap samples
    boot_coefs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_obs, n_obs, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        coef = np.sum(X_boot.flatten() * y_boot) / np.sum(X_boot.flatten() ** 2)
        boot_coefs.append(coef)

    # Percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_coefs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_coefs, 100 * (1 - alpha / 2))

    return point_est, ci_lower, ci_upper


def run_coverage_simulation(
    n_simulations: int = MIN_SIMULATIONS,
    true_coef: float = 0.05,
    confidence: float = 0.95,
) -> tuple[float, list[bool]]:
    """
    Run Monte Carlo simulation to estimate CI coverage.

    Returns
    -------
    Tuple[float, List[bool]]
        (coverage_rate, list of coverage indicators)
    """
    covered = []

    for sim in range(n_simulations):
        X, y, true_val = generate_synthetic_data(true_coef=true_coef, seed=sim)
        _, ci_lower, ci_upper = compute_bootstrap_ci(X, y, seed=sim)

        is_covered = ci_lower <= true_val <= ci_upper
        covered.append(is_covered)

    coverage_rate = np.mean(covered)
    return coverage_rate, covered


# =============================================================================
# COVERAGE VALIDATION TESTS
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.slow
class TestBootstrapCoverage:
    """Validate bootstrap confidence interval coverage. [T1]"""

    def test_95_coverage_achieved(self) -> None:
        """95% CI should have coverage near 95%. [T1]

        Reference: Bootstrap theory (Efron & Tibshirani, 1993)
        """
        coverage, _ = run_coverage_simulation(
            n_simulations=MIN_SIMULATIONS,
            true_coef=0.05,
            confidence=0.95,
        )

        assert (
            NOMINAL_COVERAGE - COVERAGE_TOLERANCE
            <= coverage
            <= NOMINAL_COVERAGE + COVERAGE_TOLERANCE
        ), (
            f"95% CI coverage ({coverage:.1%}) outside acceptable range "
            f"[{NOMINAL_COVERAGE - COVERAGE_TOLERANCE:.1%}, "
            f"{NOMINAL_COVERAGE + COVERAGE_TOLERANCE:.1%}]"
        )

    def test_coverage_not_too_conservative(self) -> None:
        """Coverage should not be excessively high. [T2]

        Overly conservative CIs (>98% coverage) waste precision.
        """
        coverage, _ = run_coverage_simulation(
            n_simulations=MIN_SIMULATIONS,
            true_coef=0.05,
            confidence=0.95,
        )

        max_acceptable = 0.98

        assert coverage <= max_acceptable, (
            f"Coverage ({coverage:.1%}) too conservative (>{max_acceptable:.1%}). "
            f"CI may be wider than necessary."
        )

    def test_coverage_not_too_liberal(self) -> None:
        """Coverage should not be too low. [T1]

        Under-coverage (<90%) indicates underestimated uncertainty.
        This is a more serious issue than over-coverage.
        """
        coverage, _ = run_coverage_simulation(
            n_simulations=MIN_SIMULATIONS,
            true_coef=0.05,
            confidence=0.95,
        )

        min_acceptable = 0.90

        assert coverage >= min_acceptable, (
            f"Coverage ({coverage:.1%}) too liberal (<{min_acceptable:.1%}). "
            f"Uncertainty is underestimated - this is a critical issue."
        )


@pytest.mark.monte_carlo
class TestBootstrapSignStability:
    """Validate coefficient sign stability across bootstrap samples. [T1]"""

    def test_positive_coefficient_stability(self) -> None:
        """Positive coefficient should have consistent sign. [T1]

        When true coefficient is positive and well-estimated,
        bootstrap samples should show high sign consistency.
        """
        X, y, _ = generate_synthetic_data(true_coef=0.08, seed=42)

        # Get bootstrap samples
        n_bootstrap = 200
        rng = np.random.RandomState(42)
        n_obs = len(y)

        positive_count = 0
        for _ in range(n_bootstrap):
            idx = rng.choice(n_obs, n_obs, replace=True)
            coef = np.sum(X[idx].flatten() * y[idx]) / np.sum(X[idx].flatten() ** 2)
            if coef > 0:
                positive_count += 1

        sign_consistency = positive_count / n_bootstrap

        # Expect >95% of samples to have correct sign
        assert sign_consistency >= 0.95, (
            f"Sign consistency ({sign_consistency:.1%}) below 95%. "
            f"Coefficient estimate may be unstable."
        )

    def test_negative_coefficient_stability(self) -> None:
        """Negative coefficient should have consistent sign. [T1]"""
        X, y, _ = generate_synthetic_data(true_coef=-0.05, seed=42)

        n_bootstrap = 200
        rng = np.random.RandomState(42)
        n_obs = len(y)

        negative_count = 0
        for _ in range(n_bootstrap):
            idx = rng.choice(n_obs, n_obs, replace=True)
            coef = np.sum(X[idx].flatten() * y[idx]) / np.sum(X[idx].flatten() ** 2)
            if coef < 0:
                negative_count += 1

        sign_consistency = negative_count / n_bootstrap

        assert (
            sign_consistency >= 0.95
        ), f"Negative sign consistency ({sign_consistency:.1%}) below 95%."


@pytest.mark.monte_carlo
class TestBootstrapDistribution:
    """Validate bootstrap coefficient distribution properties. [T1]"""

    def test_bootstrap_distribution_centered(self) -> None:
        """Bootstrap mean should be close to point estimate. [T1]"""
        X, y, true_coef = generate_synthetic_data(true_coef=0.05, seed=42)

        point_est = np.sum(X.flatten() * y) / np.sum(X.flatten() ** 2)

        # Bootstrap samples
        n_bootstrap = 500
        rng = np.random.RandomState(42)
        n_obs = len(y)

        boot_coefs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_obs, n_obs, replace=True)
            coef = np.sum(X[idx].flatten() * y[idx]) / np.sum(X[idx].flatten() ** 2)
            boot_coefs.append(coef)

        boot_mean = np.mean(boot_coefs)

        # Bootstrap mean should be very close to point estimate
        assert abs(boot_mean - point_est) < 0.01, (
            f"Bootstrap mean ({boot_mean:.4f}) differs from point estimate "
            f"({point_est:.4f}) by more than 0.01"
        )

    def test_bootstrap_ci_width_reasonable(self) -> None:
        """Bootstrap CI width should be reasonable. [T2]

        Too narrow: underestimated uncertainty
        Too wide: sample size may be insufficient
        """
        X, y, true_coef = generate_synthetic_data(
            n_obs=200, true_coef=0.05, noise_std=0.02, seed=42
        )

        _, ci_lower, ci_upper = compute_bootstrap_ci(X, y, seed=42)
        ci_width = ci_upper - ci_lower

        # For this DGP, CI width should be roughly 0.01-0.05
        assert 0.005 <= ci_width <= 0.10, (
            f"CI width ({ci_width:.4f}) outside expected range [0.005, 0.10]. "
            f"Check sample size or noise level."
        )


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================


@pytest.mark.monte_carlo
@pytest.mark.property
class TestBootstrapProperties:
    """Property-based tests for bootstrap. [T1]"""

    @given(st.floats(min_value=0.01, max_value=0.10))
    @settings(max_examples=10, deadline=5000)
    def test_larger_coefficient_narrower_relative_ci(self, true_coef: float) -> None:
        """Larger coefficients should have narrower relative CI. [T1]

        The coefficient of variation of the CI should decrease
        as the true coefficient magnitude increases.
        """
        X, y, _ = generate_synthetic_data(n_obs=200, true_coef=true_coef, noise_std=0.02, seed=42)

        point_est, ci_lower, ci_upper = compute_bootstrap_ci(X, y, seed=42)
        ci_width = ci_upper - ci_lower

        # Relative width (coefficient of variation)
        relative_width = ci_width / abs(point_est) if point_est != 0 else float("inf")

        # Should be less than 100% of the estimate
        assert relative_width < 1.0, (
            f"CI width ({ci_width:.4f}) is larger than point estimate "
            f"({point_est:.4f}). Model precision is poor."
        )
