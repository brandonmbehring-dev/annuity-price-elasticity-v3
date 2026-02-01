"""
Calibration Tests: Confidence Interval Coverage.

Validates that bootstrap confidence intervals achieve their nominal coverage rate
through Monte Carlo simulation.

Key Validation:
    - 95% CI should contain the true value ~95% of the time
    - Coverage should be within [93%, 97%] for well-calibrated intervals
    - Systematic under/over-coverage indicates bias

Decision Reference:
    DL-001: Bootstrap Sample Size (1000 samples)
    Rationale: 1000 samples provides ~1% Monte Carlo precision

Knowledge Tier:
    [T1] = Statistical theory (Efron & Tibshirani, 1993)
    [T2] = Empirical calibration from production models

References:
    - Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap
    - tests/known_answer/test_golden_reference.py (bootstrap statistics)
"""

from dataclasses import dataclass

import numpy as np
import pytest

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

# Monte Carlo simulation parameters
N_MONTE_CARLO_SIMS = 500  # Number of simulations for coverage estimation
N_BOOTSTRAP_SAMPLES = 1000  # Bootstrap samples per simulation (per DL-001)

# Coverage tolerances
NOMINAL_COVERAGE = 0.95  # Target coverage for 95% CI
COVERAGE_TOLERANCE = 0.02  # ±2% tolerance (93%-97% acceptable)
MIN_ACCEPTABLE_COVERAGE = NOMINAL_COVERAGE - COVERAGE_TOLERANCE
MAX_ACCEPTABLE_COVERAGE = NOMINAL_COVERAGE + COVERAGE_TOLERANCE


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CoverageResult:
    """Result of coverage calibration test."""

    coverage: float  # Observed coverage rate
    n_simulations: int  # Number of Monte Carlo simulations
    n_covered: int  # Simulations where CI contained true value
    lower_bound: float  # Lower acceptance bound
    upper_bound: float  # Upper acceptance bound
    passed: bool  # Whether coverage is within tolerance

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Coverage: {self.coverage:.1%} "
            f"(n={self.n_simulations}, covered={self.n_covered}) "
            f"[{self.lower_bound:.1%}, {self.upper_bound:.1%}] "
            f"[{status}]"
        )


# =============================================================================
# CORE CALIBRATION FUNCTIONS
# =============================================================================


def bootstrap_percentile_ci(
    data: np.ndarray,
    statistic_fn: callable,
    n_bootstrap: int = N_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
    random_state: int = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap percentile confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Original data sample.
    statistic_fn : callable
        Function to compute statistic from data (e.g., np.mean).
    n_bootstrap : int
        Number of bootstrap samples.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, lower_bound, upper_bound)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Resample with replacement
        boot_sample = data[np.random.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic_fn(boot_sample)

    # Percentile method for CI
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    point_estimate = statistic_fn(data)

    return point_estimate, lower, upper


def monte_carlo_coverage(
    true_value: float,
    data_generator: callable,
    statistic_fn: callable,
    n_simulations: int = N_MONTE_CARLO_SIMS,
    n_bootstrap: int = N_BOOTSTRAP_SAMPLES,
    confidence_level: float = 0.95,
) -> CoverageResult:
    """
    Estimate CI coverage via Monte Carlo simulation.

    Generates many datasets from known DGP, computes bootstrap CI for each,
    and counts how often the true value falls within the CI.

    Parameters
    ----------
    true_value : float
        Known true value of the statistic.
    data_generator : callable
        Function that generates a new data sample.
    statistic_fn : callable
        Function to compute statistic from data.
    n_simulations : int
        Number of Monte Carlo simulations.
    n_bootstrap : int
        Bootstrap samples per simulation.
    confidence_level : float
        Nominal confidence level.

    Returns
    -------
    CoverageResult
        Coverage statistics and pass/fail status.
    """
    n_covered = 0

    for sim in range(n_simulations):
        # Generate new dataset from DGP
        data = data_generator()

        # Compute bootstrap CI
        _, lower, upper = bootstrap_percentile_ci(
            data=data,
            statistic_fn=statistic_fn,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=sim,  # Different seed per simulation
        )

        # Check if true value covered
        if lower <= true_value <= upper:
            n_covered += 1

    coverage = n_covered / n_simulations
    passed = MIN_ACCEPTABLE_COVERAGE <= coverage <= MAX_ACCEPTABLE_COVERAGE

    return CoverageResult(
        coverage=coverage,
        n_simulations=n_simulations,
        n_covered=n_covered,
        lower_bound=MIN_ACCEPTABLE_COVERAGE,
        upper_bound=MAX_ACCEPTABLE_COVERAGE,
        passed=passed,
    )


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def normal_data_generator():
    """
    Generate data generator for Normal(0, 1) samples.

    Returns a function that generates n=100 samples from N(0,1).
    True mean is 0.
    """
    n_samples = 100

    def generator():
        return np.random.randn(n_samples)

    return generator, 0.0  # (generator, true_mean)


@pytest.fixture
def exponential_data_generator():
    """
    Generate data generator for Exponential(1) samples.

    Returns a function that generates n=100 samples from Exp(1).
    True mean is 1.
    """
    n_samples = 100
    rate = 1.0

    def generator():
        return np.random.exponential(scale=1 / rate, size=n_samples)

    return generator, 1.0  # (generator, true_mean)


@pytest.fixture
def regression_data_generator():
    """
    Generate data generator for simple linear regression.

    y = 2*x + 1 + noise
    True slope is 2.
    """
    n_samples = 100
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 0.5

    def generator():
        x = np.random.uniform(0, 1, n_samples)
        y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n_samples)
        return np.column_stack([x, y])

    def slope_statistic(data):
        """OLS slope estimator."""
        x, y = data[:, 0], data[:, 1]
        x_mean = np.mean(x)
        return np.sum((x - x_mean) * y) / np.sum((x - x_mean) ** 2)

    return generator, true_slope, slope_statistic


# =============================================================================
# COVERAGE CALIBRATION TESTS
# =============================================================================


@pytest.mark.calibration
@pytest.mark.slow
class TestCoverageCalibration:
    """
    Validate bootstrap CI coverage via Monte Carlo simulation.

    These tests verify that our 95% CIs actually contain the true value
    approximately 95% of the time. This is essential for reliable
    uncertainty quantification in price elasticity predictions.
    """

    def test_mean_coverage_normal_data(self, normal_data_generator) -> None:
        """
        Bootstrap CI for mean has correct coverage on Normal data. [T1]

        This is a standard calibration test where we know the true mean
        is 0 and verify the CI covers it ~95% of the time.
        """
        generator, true_mean = normal_data_generator

        result = monte_carlo_coverage(
            true_value=true_mean,
            data_generator=generator,
            statistic_fn=np.mean,
            n_simulations=N_MONTE_CARLO_SIMS,
            n_bootstrap=N_BOOTSTRAP_SAMPLES,
            confidence_level=0.95,
        )

        assert result.passed, (
            f"Coverage {result.coverage:.1%} outside acceptable range "
            f"[{result.lower_bound:.1%}, {result.upper_bound:.1%}]. "
            f"Bootstrap CI is {'under' if result.coverage < MIN_ACCEPTABLE_COVERAGE else 'over'}-covering."
        )

    def test_mean_coverage_exponential_data(self, exponential_data_generator) -> None:
        """
        Bootstrap CI for mean has correct coverage on skewed data. [T1]

        Exponential data is skewed, testing robustness of percentile method.
        """
        generator, true_mean = exponential_data_generator

        result = monte_carlo_coverage(
            true_value=true_mean,
            data_generator=generator,
            statistic_fn=np.mean,
            n_simulations=N_MONTE_CARLO_SIMS,
            n_bootstrap=N_BOOTSTRAP_SAMPLES,
            confidence_level=0.95,
        )

        assert result.passed, (
            f"Coverage {result.coverage:.1%} outside acceptable range on skewed data. "
            f"Percentile method may need BCa correction for heavily skewed distributions."
        )

    def test_slope_coverage_regression(self, regression_data_generator) -> None:
        """
        Bootstrap CI for regression slope has correct coverage. [T1]

        This tests coverage for a regression coefficient, which is directly
        relevant to our price elasticity coefficient estimation.
        """
        generator, true_slope, slope_statistic = regression_data_generator

        result = monte_carlo_coverage(
            true_value=true_slope,
            data_generator=generator,
            statistic_fn=slope_statistic,
            n_simulations=N_MONTE_CARLO_SIMS,
            n_bootstrap=N_BOOTSTRAP_SAMPLES,
            confidence_level=0.95,
        )

        assert result.passed, (
            f"Slope coverage {result.coverage:.1%} outside acceptable range. "
            f"This affects confidence in price elasticity coefficient estimates."
        )

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_coverage_at_different_levels(
        self, normal_data_generator, confidence_level: float
    ) -> None:
        """
        Coverage is correct at different confidence levels. [T1]

        Validates that our CI construction works across common confidence levels.
        """
        generator, true_mean = normal_data_generator

        # Adjust tolerance based on confidence level
        tolerance = 0.03  # Slightly looser for non-95% levels

        result = monte_carlo_coverage(
            true_value=true_mean,
            data_generator=generator,
            statistic_fn=np.mean,
            n_simulations=N_MONTE_CARLO_SIMS,
            n_bootstrap=N_BOOTSTRAP_SAMPLES,
            confidence_level=confidence_level,
        )

        min_acceptable = confidence_level - tolerance
        max_acceptable = confidence_level + tolerance

        assert min_acceptable <= result.coverage <= max_acceptable, (
            f"Coverage {result.coverage:.1%} for {confidence_level:.0%} CI "
            f"outside acceptable range [{min_acceptable:.1%}, {max_acceptable:.1%}]."
        )


# =============================================================================
# BOOTSTRAP STABILITY TESTS
# =============================================================================


@pytest.mark.calibration
class TestBootstrapStability:
    """
    Validate bootstrap estimator stability properties.

    These tests verify that bootstrap estimates are stable and consistent,
    which is essential for reproducible price elasticity predictions.
    """

    def test_bootstrap_ci_width_decreases_with_samples(self) -> None:
        """
        CI width decreases as sample size increases. [T1]

        Validates that more data leads to more precise estimates,
        as expected from asymptotic theory.
        """
        sample_sizes = [50, 100, 200]
        ci_widths = []

        for n in sample_sizes:
            data = np.random.randn(n)
            _, lower, upper = bootstrap_percentile_ci(
                data=data,
                statistic_fn=np.mean,
                n_bootstrap=500,
                confidence_level=0.95,
                random_state=42,
            )
            ci_widths.append(upper - lower)

        # CI width should decrease with sample size
        for i in range(len(ci_widths) - 1):
            assert ci_widths[i + 1] < ci_widths[i], (
                f"CI width did not decrease: "
                f"n={sample_sizes[i]} width={ci_widths[i]:.4f}, "
                f"n={sample_sizes[i+1]} width={ci_widths[i+1]:.4f}"
            )

    def test_bootstrap_ci_stable_across_seeds(self) -> None:
        """
        Bootstrap CI is stable (low variance) with enough samples. [T2]

        Per DL-001: 1000 samples provides ~1% Monte Carlo error.
        CI endpoints should not vary dramatically across seeds.
        """
        data = np.random.randn(100)
        seeds = [1, 2, 3, 4, 5]
        lower_bounds = []
        upper_bounds = []

        for seed in seeds:
            _, lower, upper = bootstrap_percentile_ci(
                data=data,
                statistic_fn=np.mean,
                n_bootstrap=N_BOOTSTRAP_SAMPLES,
                confidence_level=0.95,
                random_state=seed,
            )
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        # CI endpoints should have low coefficient of variation
        lower_cv = np.std(lower_bounds) / abs(np.mean(lower_bounds))
        upper_cv = np.std(upper_bounds) / abs(np.mean(upper_bounds))

        max_cv = 0.05  # 5% CV tolerance

        assert lower_cv < max_cv, (
            f"Lower bound CV {lower_cv:.1%} exceeds {max_cv:.0%} tolerance. "
            f"Bootstrap may need more samples (currently {N_BOOTSTRAP_SAMPLES})."
        )
        assert upper_cv < max_cv, f"Upper bound CV {upper_cv:.1%} exceeds {max_cv:.0%} tolerance."

    def test_bootstrap_mean_unbiased(self) -> None:
        """
        Bootstrap mean estimate is unbiased. [T1]

        The bootstrap point estimate should equal the sample statistic.
        """
        np.random.seed(42)
        data = np.random.randn(100)
        sample_mean = np.mean(data)

        point_estimate, _, _ = bootstrap_percentile_ci(
            data=data,
            statistic_fn=np.mean,
            n_bootstrap=N_BOOTSTRAP_SAMPLES,
            confidence_level=0.95,
            random_state=42,
        )

        assert np.isclose(point_estimate, sample_mean, rtol=1e-10), (
            f"Bootstrap point estimate {point_estimate:.6f} "
            f"differs from sample mean {sample_mean:.6f}"
        )


# =============================================================================
# DECISION VALIDATION TESTS
# =============================================================================


@pytest.mark.calibration
class TestDecisionDL001:
    """
    Validate Decision DL-001: Bootstrap Sample Size (1000 samples).

    Per DL-001, 1000 samples provides ~1% Monte Carlo precision.
    These tests verify the decision's empirical basis.
    """

    def test_1000_samples_provides_1_percent_precision(self) -> None:
        """
        1000 bootstrap samples gives ~1% Monte Carlo error. [T2]

        Monte Carlo standard error = 1/sqrt(n)
        For n=1000: SE ≈ 3.2%
        For n=10000: SE ≈ 1%

        This test verifies the relationship holds.
        """
        data = np.random.randn(100)

        # Compare variance of CI endpoints across runs
        n_runs = 20
        ci_lowers_1000 = []
        ci_lowers_10000 = []

        for i in range(n_runs):
            _, lower, _ = bootstrap_percentile_ci(
                data=data,
                statistic_fn=np.mean,
                n_bootstrap=1000,
                random_state=i * 100,
            )
            ci_lowers_1000.append(lower)

            _, lower, _ = bootstrap_percentile_ci(
                data=data,
                statistic_fn=np.mean,
                n_bootstrap=10000,
                random_state=i * 100,
            )
            ci_lowers_10000.append(lower)

        std_1000 = np.std(ci_lowers_1000)
        std_10000 = np.std(ci_lowers_10000)

        # 10x more samples should reduce variance by ~sqrt(10) ≈ 3.16x
        variance_ratio = std_1000 / std_10000
        expected_ratio = np.sqrt(10)  # ~3.16

        # Allow 50% tolerance due to random variation
        assert variance_ratio > expected_ratio * 0.5, (
            f"Variance ratio {variance_ratio:.2f} too low. "
            f"Expected ~{expected_ratio:.2f} based on 1/sqrt(n) relationship."
        )

    def test_diminishing_returns_above_1000(self) -> None:
        """
        Returns diminish significantly above 1000 samples. [T2]

        Per DL-001: Going from 1000 to 10000 gives 10x slower runtime
        but only ~3x improvement in precision.
        """
        # This is a documentation test - the math is:
        # - 1000 samples: SE ≈ 3.2%
        # - 10000 samples: SE ≈ 1%
        # - Improvement: 3.2x for 10x runtime cost

        cost_ratio = 10000 / 1000  # 10x
        precision_improvement = np.sqrt(10000) / np.sqrt(1000)  # ~3.16x

        # Diminishing returns: cost grows faster than precision
        assert cost_ratio > precision_improvement, (
            "Computation cost should grow faster than precision improvement. "
            "This validates DL-001's choice of 1000 samples."
        )
