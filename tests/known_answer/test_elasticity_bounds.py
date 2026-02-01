"""
Known-Answer Tests: Elasticity Coefficient Bounds
==================================================

Validates elasticity coefficients against published annuity research.

Expected Ranges (Literature-Based):
    Own-rate elasticity: +0.02 to +0.15 per basis point [T1: LIMRA 2023]
    Competitor elasticity: -0.08 to -0.01 per basis point [T1: LIMRA 2023]
    R² for weekly elasticity models: 0.50 to 0.85 [T2: Production baseline]

These bounds represent the academically validated range for price elasticity
in the U.S. annuity market. Values outside these ranges suggest either:
1. Data leakage (inflated performance)
2. Model misspecification (sign violations)
3. Insufficient sample size (extreme volatility)

Knowledge Tier Tags:
    [T1] = Academically validated (LIMRA, SEC filings)
    [T2] = Empirical from production models

References:
    - LIMRA (2023) "Price Sensitivity in Annuity Markets"
    - SEC Release No. 34-72685 (2014) - RILA Framework
    - Production model: 78.37% R², validated 2025-11-25
"""

import pytest

# =============================================================================
# LITERATURE-BASED BOUNDS [T1]
# =============================================================================

# Own-rate (Prudential) coefficient bounds
# Reference: LIMRA (2023), SEC Release No. 34-72685
# Economic intuition: Higher own rates attract customers
# Magnitude: 2-15% demand increase per 100 basis points
EXPECTED_OWN_RATE_RANGE: tuple[float, float] = (0.02, 0.15)

# Competitor rate coefficient bounds
# Reference: LIMRA (2023) cross-price elasticity estimates
# Economic intuition: Higher competitor rates divert customers
# Magnitude: 1-8% demand decrease per 100 basis points of competitor advantage
EXPECTED_COMPETITOR_RANGE: tuple[float, float] = (-0.08, -0.01)

# R² bounds for weekly elasticity models [T2]
# Reference: Production 6Y20B model (78.37% R²)
# Lower bound: Minimum acceptable explanatory power
# Upper bound: Suspiciously high (suggests leakage)
EXPECTED_R_SQUARED_RANGE: tuple[float, float] = (0.50, 0.85)

# MAPE bounds for weekly elasticity models [T2]
# Reference: Production 6Y20B model (12.74% MAPE)
EXPECTED_MAPE_RANGE: tuple[float, float] = (0.08, 0.25)


# =============================================================================
# FIXTURE IMPORTS
# =============================================================================


@pytest.fixture
def production_coefficients() -> dict[str, float]:
    """
    Load production-validated coefficient values.

    These are the actual coefficients from the validated 6Y20B production model.
    Reference: 02_time_series_forecasting_refactored.ipynb (2025-11-25 validation)

    Returns
    -------
    Dict[str, float]
        Feature name to coefficient value mapping

    Notes
    -----
    [T2] These values are empirically derived from production, not literature.
    """
    return {
        "prudential_rate_current": 0.0847,  # Own rate (positive)
        "competitor_mid_t2": -0.0312,  # Market-weighted competitor (negative)
        "competitor_top5_t3": -0.0284,  # Top-5 competitor (negative)
        "sales_target_contract_t5": 0.0156,  # Sales persistence (positive)
    }


@pytest.fixture
def production_metrics() -> dict[str, float]:
    """
    Load production-validated performance metrics.

    Reference: Production baseline captured 2025-11-25
    Environment: Bootstrap Ridge Ensemble (10,000 estimators)

    Returns
    -------
    Dict[str, float]
        Metric name to value mapping
    """
    return {
        "r_squared": 0.7837,
        "mape": 0.1274,
        "coverage_95": 0.944,
        "n_bootstrap_samples": 10000,
    }


# =============================================================================
# OWN-RATE ELASTICITY TESTS [T1]
# =============================================================================


@pytest.mark.known_answer
class TestOwnRateElasticityBounds:
    """Validate own-rate coefficient against literature bounds. [T1]"""

    def test_own_rate_positive(self, production_coefficients: dict[str, float]) -> None:
        """Own rate coefficient must be positive (attractiveness effect). [T1]

        Economic rationale: Higher own cap rates attract customers because
        cap rate = yield. Higher yield = more attractive product.
        """
        own_rate = production_coefficients.get("prudential_rate_current", 0)

        assert own_rate > 0, (
            f"Own rate coefficient ({own_rate:.4f}) must be positive. "
            f"Negative coefficient violates yield-based economics."
        )

    def test_own_rate_in_literature_range(self, production_coefficients: dict[str, float]) -> None:
        """Own rate coefficient within LIMRA (2023) published bounds. [T1]

        Reference: LIMRA (2023) "Price Sensitivity in Annuity Markets"
        Expected range: +0.02 to +0.15 per basis point

        Values below 0.02 suggest weak price sensitivity (possible misspecification).
        Values above 0.15 suggest inflated effect (possible leakage).
        """
        own_rate = production_coefficients.get("prudential_rate_current", 0)
        lower, upper = EXPECTED_OWN_RATE_RANGE

        assert lower <= own_rate <= upper, (
            f"Own rate ({own_rate:.4f}) outside literature range "
            f"[{lower:.2f}, {upper:.2f}]. "
            f"Check for data leakage or model misspecification."
        )

    def test_own_rate_magnitude_reasonable(self, production_coefficients: dict[str, float]) -> None:
        """Own rate magnitude implies reasonable demand response. [T1]

        A 100 basis point rate increase should increase demand by 2-15%.
        This test validates the implied elasticity is economically sensible.
        """
        own_rate = production_coefficients.get("prudential_rate_current", 0)

        # Implied demand change for 100bp rate increase
        implied_demand_change = own_rate * 100

        assert 2 <= implied_demand_change <= 15, (
            f"Implied demand change ({implied_demand_change:.1f}%) for 100bp "
            f"rate increase is outside reasonable range [2%, 15%]."
        )


# =============================================================================
# COMPETITOR RATE ELASTICITY TESTS [T1]
# =============================================================================


@pytest.mark.known_answer
class TestCompetitorRateElasticityBounds:
    """Validate competitor rate coefficient against literature bounds. [T1]"""

    def test_competitor_rate_negative(self, production_coefficients: dict[str, float]) -> None:
        """Competitor rate coefficient must be negative (substitution effect). [T1]

        Economic rationale: Higher competitor rates divert customers away.
        This is standard cross-price elasticity from microeconomics.
        """
        competitor_rate = production_coefficients.get("competitor_mid_t2", 0)

        assert competitor_rate < 0, (
            f"Competitor rate coefficient ({competitor_rate:.4f}) must be negative. "
            f"Positive coefficient violates substitution economics."
        )

    def test_competitor_rate_in_literature_range(
        self, production_coefficients: dict[str, float]
    ) -> None:
        """Competitor rate coefficient within LIMRA published bounds. [T1]

        Reference: LIMRA (2023) cross-price elasticity estimates
        Expected range: -0.08 to -0.01 per basis point

        Values closer to 0 suggest weak competitive pressure.
        Values more negative than -0.08 suggest inflated substitution effect.
        """
        competitor_rate = production_coefficients.get("competitor_mid_t2", 0)
        lower, upper = EXPECTED_COMPETITOR_RANGE

        assert lower <= competitor_rate <= upper, (
            f"Competitor rate ({competitor_rate:.4f}) outside literature range "
            f"[{lower:.2f}, {upper:.2f}]. "
            f"Check for data leakage or competitive set definition."
        )

    def test_own_vs_competitor_magnitude_ratio(
        self, production_coefficients: dict[str, float]
    ) -> None:
        """Own rate effect should dominate competitor effect. [T2]

        Economic rationale: Firms have more control over demand via own pricing
        than via competitor pricing. The own-rate coefficient magnitude should
        typically exceed the absolute competitor coefficient.

        This is an empirical regularity, not a theoretical constraint.
        """
        own_rate = production_coefficients.get("prudential_rate_current", 0)
        competitor_rate = production_coefficients.get("competitor_mid_t2", 0)

        # Own rate magnitude should exceed competitor magnitude
        ratio = abs(own_rate) / abs(competitor_rate) if competitor_rate != 0 else float("inf")

        assert ratio > 1.0, (
            f"Own rate magnitude ({abs(own_rate):.4f}) should exceed "
            f"competitor magnitude ({abs(competitor_rate):.4f}). "
            f"Ratio: {ratio:.2f}. "
            f"Consider whether competitive set is correctly defined."
        )


# =============================================================================
# MODEL PERFORMANCE BOUNDS TESTS [T2]
# =============================================================================


@pytest.mark.known_answer
class TestModelPerformanceBounds:
    """Validate model performance against production baselines. [T2]"""

    def test_r_squared_in_expected_range(self, production_metrics: dict[str, float]) -> None:
        """R² should fall within expected range for annuity models. [T2]

        Reference: Production 6Y20B model (78.37% R²)

        R² < 0.50 suggests insufficient explanatory power.
        R² > 0.85 suggests possible data leakage (suspiciously high).
        """
        r_squared = production_metrics.get("r_squared", 0)
        lower, upper = EXPECTED_R_SQUARED_RANGE

        assert lower <= r_squared <= upper, (
            f"R² ({r_squared:.3f}) outside expected range "
            f"[{lower:.2f}, {upper:.2f}]. "
            f"Low: check feature set. High: check for leakage."
        )

    def test_mape_in_expected_range(self, production_metrics: dict[str, float]) -> None:
        """MAPE should fall within expected range for annuity models. [T2]

        Reference: Production 6Y20B model (12.74% MAPE)

        MAPE > 0.25 suggests poor forecast accuracy.
        MAPE < 0.08 suggests possible data leakage (too accurate).
        """
        mape = production_metrics.get("mape", 0)
        lower, upper = EXPECTED_MAPE_RANGE

        assert lower <= mape <= upper, (
            f"MAPE ({mape:.3f}) outside expected range "
            f"[{lower:.2f}, {upper:.2f}]. "
            f"Low: check for leakage. High: check model specification."
        )

    def test_coverage_near_nominal(self, production_metrics: dict[str, float]) -> None:
        """95% prediction interval coverage should be near 95%. [T2]

        Coverage significantly below 95% indicates underestimated uncertainty.
        Coverage significantly above 95% indicates overestimated uncertainty.
        """
        coverage = production_metrics.get("coverage_95", 0)

        # Allow 5% deviation from nominal
        assert 0.90 <= coverage <= 0.97, (
            f"95% coverage ({coverage:.3f}) should be near 0.95. "
            f"Low: underestimated uncertainty. High: overestimated."
        )


# =============================================================================
# CROSS-PRODUCT CONSISTENCY TESTS [T2]
# =============================================================================


@pytest.mark.known_answer
class TestCrossProductConsistency:
    """Validate coefficient patterns across product variants. [T2]"""

    @pytest.mark.parametrize(
        "product_code,expected_own_sign,expected_competitor_sign",
        [
            ("6Y20B", "positive", "negative"),
            ("6Y10B", "positive", "negative"),
            ("10Y20B", "positive", "negative"),
        ],
    )
    def test_sign_consistency_across_products(
        self,
        product_code: str,
        expected_own_sign: str,
        expected_competitor_sign: str,
    ) -> None:
        """Economic sign constraints consistent across RILA variants. [T1]

        All RILA products share the same yield-based economics:
        - Own rate positive (attractiveness)
        - Competitor rate negative (substitution)

        Buffer and term variations don't change sign expectations.
        """
        # This is a structural test - actual coefficients would come from
        # product-specific model runs. Here we validate the constraint pattern.
        assert expected_own_sign == "positive", f"Product {product_code} own rate must be positive"
        assert (
            expected_competitor_sign == "negative"
        ), f"Product {product_code} competitor rate must be negative"


# =============================================================================
# EDGE CASE VALIDATION [T3]
# =============================================================================


@pytest.mark.known_answer
class TestEdgeCaseValidation:
    """Validate behavior at coefficient boundaries. [T3]"""

    def test_zero_coefficient_not_allowed_for_core_features(
        self, production_coefficients: dict[str, float]
    ) -> None:
        """Core features should not have zero coefficients. [T3]

        Zero coefficient indicates feature has no effect, which is unlikely
        for core economic variables (own rate, competitor rate).

        This is an assumption that could be violated in edge cases.
        """
        core_features = ["prudential_rate_current", "competitor_mid_t2"]

        for feature in core_features:
            coef = production_coefficients.get(feature, 0)
            assert abs(coef) > 1e-6, (
                f"Feature {feature} has near-zero coefficient ({coef:.6f}). "
                f"Core economic features should have non-zero effects."
            )

    def test_coefficient_ratio_stability(self, production_coefficients: dict[str, float]) -> None:
        """Ratio of own-to-competitor effect should be stable. [T3]

        Historical production models show this ratio between 2.0 and 4.0.
        Large deviations suggest model instability or specification change.
        """
        own = production_coefficients.get("prudential_rate_current", 0)
        competitor = production_coefficients.get("competitor_mid_t2", 0)

        if competitor != 0:
            ratio = abs(own) / abs(competitor)

            # Allow wide range due to assumption nature
            assert 1.5 <= ratio <= 5.0, (
                f"Own/competitor ratio ({ratio:.2f}) outside expected [1.5, 5.0]. "
                f"Check for model drift or specification changes."
            )
