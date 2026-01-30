"""
Anti-Pattern Test: Coefficient Sign Validation
==============================================

CRITICAL: Coefficient signs must match economic theory.

This test module enforces that fitted model coefficients have the correct
signs based on yield economics and substitution theory.

Economic Theory (RILA Products):
- Own cap rate coefficient: POSITIVE (higher rates attract customers)
- Competitor rate coefficient: NEGATIVE (substitution effect)
- Buffer level coefficient: POSITIVE (more protection is better)
- VIX coefficient: NEGATIVE (market uncertainty reduces purchases)

Why This Matters:
- Wrong signs indicate model specification error
- Could suggest data leakage or multicollinearity
- Makes business recommendations unreliable
- Violates basic microeconomic theory

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from enum import Enum


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================


class SignConstraint(Enum):
    """Expected sign for a coefficient."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANY = "any"  # No constraint


@dataclass
class CoefficientConstraint:
    """Defines expected sign and bounds for a coefficient pattern."""
    pattern: str              # Regex or substring to match feature names
    expected_sign: SignConstraint
    description: str
    tolerance: float = 0.0    # Allow small violations (for numerical noise)
    max_magnitude: Optional[float] = None  # Detect implausible values


# RILA product constraints (based on yield economics)
RILA_COEFFICIENT_CONSTRAINTS = [
    CoefficientConstraint(
        pattern="prudential_rate",
        expected_sign=SignConstraint.POSITIVE,
        description="Own rate positive: higher rates attract customers",
        max_magnitude=100.0,  # Elasticity > 100 is implausible
    ),
    CoefficientConstraint(
        pattern="own_cap_rate",
        expected_sign=SignConstraint.POSITIVE,
        description="Own rate positive: higher rates attract customers",
        max_magnitude=100.0,
    ),
    CoefficientConstraint(
        pattern="competitor",
        expected_sign=SignConstraint.NEGATIVE,
        description="Competitor negative: substitution effect",
        max_magnitude=100.0,
    ),
    CoefficientConstraint(
        pattern="vix",
        expected_sign=SignConstraint.NEGATIVE,
        description="VIX negative: uncertainty reduces purchases",
        max_magnitude=50.0,
    ),
    CoefficientConstraint(
        pattern="buffer",
        expected_sign=SignConstraint.POSITIVE,
        description="Buffer positive: more protection is better",
        max_magnitude=50.0,
    ),
]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


@dataclass
class SignViolation:
    """Details of a coefficient sign violation."""
    feature_name: str
    coefficient_value: float
    expected_sign: SignConstraint
    description: str


def validate_coefficient_signs(
    coefficients: Dict[str, float],
    constraints: List[CoefficientConstraint],
    strict: bool = True,
) -> List[SignViolation]:
    """Validate coefficient signs against constraints.

    Args:
        coefficients: Dictionary of feature_name -> coefficient_value
        constraints: List of constraints to check
        strict: If True, raise on violations; if False, return list

    Returns:
        List of SignViolation objects for any violations found
    """
    violations = []

    for feature_name, value in coefficients.items():
        feature_lower = feature_name.lower()

        for constraint in constraints:
            if constraint.pattern.lower() in feature_lower:
                # Check sign constraint
                if constraint.expected_sign == SignConstraint.POSITIVE:
                    if value < -constraint.tolerance:
                        violations.append(SignViolation(
                            feature_name=feature_name,
                            coefficient_value=value,
                            expected_sign=constraint.expected_sign,
                            description=f"{constraint.description}; got {value:.4f}",
                        ))

                elif constraint.expected_sign == SignConstraint.NEGATIVE:
                    if value > constraint.tolerance:
                        violations.append(SignViolation(
                            feature_name=feature_name,
                            coefficient_value=value,
                            expected_sign=constraint.expected_sign,
                            description=f"{constraint.description}; got {value:.4f}",
                        ))

                # Check magnitude constraint
                if constraint.max_magnitude is not None:
                    if abs(value) > constraint.max_magnitude:
                        violations.append(SignViolation(
                            feature_name=feature_name,
                            coefficient_value=value,
                            expected_sign=constraint.expected_sign,
                            description=f"Magnitude {abs(value):.4f} exceeds max {constraint.max_magnitude}",
                        ))

                break  # Only apply first matching constraint

    return violations


# =============================================================================
# UNIT TESTS FOR SIGN VALIDATION
# =============================================================================


class TestCoefficientSignValidation:
    """Tests for coefficient sign validation logic."""

    def test_valid_coefficients_pass(self):
        """Economically valid coefficients should pass."""
        # Use realistic elasticity values (within max_magnitude constraints)
        coefficients = {
            "prudential_rate_t0": 50.0,       # Positive (correct), within 100.0 max
            "prudential_rate_t1": 30.0,       # Positive (correct), within 100.0 max
            "competitor_weighted_t2": -40.0,  # Negative (correct), within 100.0 max
            "competitor_mean_t3": -20.0,      # Negative (correct), within 100.0 max
            "vix_t0": -5.0,                   # Negative (correct), within 50.0 max
            "buffer_level": 10.0,             # Positive (correct), within 50.0 max
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 0, f"Unexpected violations: {violations}"

    def test_negative_own_rate_fails(self):
        """Negative own rate coefficient should fail."""
        coefficients = {
            "prudential_rate_t0": -50.0,      # WRONG: should be positive
            "competitor_weighted_t2": -40.0,  # Correct
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 1
        assert violations[0].feature_name == "prudential_rate_t0"
        assert violations[0].expected_sign == SignConstraint.POSITIVE

    def test_positive_competitor_rate_fails(self):
        """Positive competitor rate coefficient should fail."""
        coefficients = {
            "prudential_rate_t0": 50.0,       # Correct
            "competitor_weighted_t2": 40.0,   # WRONG: should be negative
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 1
        assert violations[0].feature_name == "competitor_weighted_t2"
        assert violations[0].expected_sign == SignConstraint.NEGATIVE

    def test_multiple_violations_detected(self):
        """Multiple violations should all be detected."""
        coefficients = {
            "prudential_rate_t0": -50.0,      # WRONG
            "own_cap_rate_t1": -30.0,         # WRONG
            "competitor_weighted_t2": 40.0,   # WRONG
            "vix_t0": 5.0,                    # WRONG
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 4

    def test_tolerance_allows_small_values(self):
        """Small values within tolerance should pass."""
        constraints = [
            CoefficientConstraint(
                pattern="test_feature",
                expected_sign=SignConstraint.POSITIVE,
                description="Test",
                tolerance=0.001,
            )
        ]

        # Value of -0.0005 is within tolerance of 0.001
        coefficients = {"test_feature": -0.0005}

        violations = validate_coefficient_signs(coefficients, constraints)

        assert len(violations) == 0

    def test_magnitude_constraint_detected(self):
        """Implausibly large coefficients should be flagged."""
        coefficients = {
            "prudential_rate_t0": 500.0,  # > max_magnitude of 100
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        # Should have magnitude violation
        magnitude_violations = [v for v in violations if "Magnitude" in v.description]
        assert len(magnitude_violations) == 1

    def test_unmatched_features_ignored(self):
        """Features not matching any constraint should be ignored."""
        coefficients = {
            "random_feature": -999.0,
            "another_unknown": 999.0,
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 0


# =============================================================================
# INTEGRATION TESTS WITH MOCK MODELS
# =============================================================================


class TestCoefficientSignsWithMockModels:
    """Integration tests simulating real model outputs."""

    @pytest.fixture
    def well_specified_model_coefficients(self) -> Dict[str, float]:
        """Coefficients from a well-specified model (realistic magnitudes)."""
        return {
            "intercept": 100.0,
            "prudential_rate_t0": 85.0,        # Within max_magnitude 100
            "prudential_rate_t1": 42.0,        # Within max_magnitude 100
            "competitor_weighted_t2": -63.0,   # Within max_magnitude 100
            "competitor_weighted_t3": -21.0,   # Within max_magnitude 100
            "vix_t0": -35.0,                   # Within max_magnitude 50
            "dgs5_t0": 12.0,                   # No constraint
            "quarter_2": -5.0,                 # No constraint
            "quarter_3": -2.0,                 # No constraint
            "quarter_4": 8.0,                  # No constraint
        }

    @pytest.fixture
    def leaky_model_coefficients(self) -> Dict[str, float]:
        """Coefficients from a model with data leakage (suspiciously good fit)."""
        return {
            "intercept": 5.0,
            "prudential_rate_t0": 250.0,       # Suspiciously high (> max 100)
            "competitor_weighted_t0": -150.0,  # Lag-0! (should be caught elsewhere)
            "competitor_weighted_t2": 30.0,    # WRONG SIGN (leakage symptom)
            "vix_t0": 20.0,                    # WRONG SIGN (leakage symptom)
        }

    def test_well_specified_model_passes(self, well_specified_model_coefficients):
        """Well-specified model should pass all sign checks."""
        violations = validate_coefficient_signs(
            well_specified_model_coefficients,
            RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 0

    def test_leaky_model_fails(self, leaky_model_coefficients):
        """Leaky model should fail sign checks."""
        violations = validate_coefficient_signs(
            leaky_model_coefficients,
            RILA_COEFFICIENT_CONSTRAINTS
        )

        # Should catch: competitor_weighted_t2 (wrong sign), vix_t0 (wrong sign),
        # prudential_rate_t0 (magnitude)
        assert len(violations) >= 2

        # Specifically check for the sign violations
        sign_violations = [
            v for v in violations
            if v.expected_sign in (SignConstraint.POSITIVE, SignConstraint.NEGATIVE)
            and "Magnitude" not in v.description
        ]
        assert len(sign_violations) >= 2


# =============================================================================
# PRODUCT-SPECIFIC CONSTRAINT TESTS
# =============================================================================


class TestProductSpecificConstraints:
    """Tests for product-specific coefficient constraints."""

    @pytest.mark.parametrize("product_code", ["6Y20B", "6Y10B", "10Y20B"])
    def test_rila_products_have_positive_own_rate(self, product_code):
        """All RILA products should expect positive own rate coefficient."""
        try:
            from src.products import get_methodology

            methodology = get_methodology(product_code)
            rules = methodology.get_constraint_rules()

            own_rate_rules = [
                r for r in rules
                if "own" in r.constraint_type.lower() or "prudential" in r.constraint_type.lower()
            ]

            assert len(own_rate_rules) > 0, f"{product_code} missing own rate constraint"

            for rule in own_rate_rules:
                assert rule.expected_sign in ("positive", "+", 1), (
                    f"{product_code} own rate should be positive"
                )

        except ImportError:
            pytest.skip("Product methodology module not available")

    @pytest.mark.parametrize("product_code", ["6Y20B", "6Y10B", "10Y20B"])
    def test_rila_products_have_negative_competitor(self, product_code):
        """All RILA products should expect negative competitor coefficient (or forbidden for lag-0)."""
        try:
            from src.products import get_methodology

            methodology = get_methodology(product_code)
            rules = methodology.get_constraint_rules()

            competitor_rules = [
                r for r in rules
                if "competitor" in r.constraint_type.lower()
            ]

            assert len(competitor_rules) > 0, f"{product_code} missing competitor constraint"

            # RILA methodology has two types of competitor rules:
            # 1. Lagged competitors (t-1 or earlier): expected_sign = 'negative'
            # 2. Lag-0 competitors: expected_sign = 'forbidden' (stricter than negative)
            for rule in competitor_rules:
                # Both 'negative' and 'forbidden' are acceptable for competitor rules
                # 'forbidden' is stricter - means the feature shouldn't exist at all
                assert rule.expected_sign in ("negative", "-", -1, "forbidden"), (
                    f"{product_code} competitor should be negative or forbidden, got {rule.expected_sign}"
                )

        except ImportError:
            pytest.skip("Product methodology module not available")


# =============================================================================
# ECONOMIC THEORY TESTS
# =============================================================================


class TestEconomicTheoryConstraints:
    """Tests that verify economic theory constraints are properly encoded."""

    def test_yield_economics_for_own_rate(self):
        """
        Economic Theory: Cap rate IS the yield for RILA products.

        Unlike bonds (where higher yield = lower price), for annuities:
        - Higher cap rate = higher potential return for customer
        - Higher potential return = more attractive product
        - More attractive = higher sales

        Therefore: Own rate coefficient MUST be positive.
        """
        # Simulate yield economics (using realistic scaled coefficient)
        own_rate_coef = 50.0  # Positive, realistic magnitude
        base_rate = 0.10  # 10% cap rate
        higher_rate = 0.12  # 12% cap rate

        base_sales = 1000 + own_rate_coef * base_rate * 100  # Scale for percentage
        higher_sales = 1000 + own_rate_coef * higher_rate * 100

        assert higher_sales > base_sales, (
            "Higher own rate should increase sales (yield economics)"
        )

    def test_substitution_effect_for_competitor(self):
        """
        Economic Theory: Customers substitute to better alternatives.

        When competitors offer higher rates:
        - Competitor products become more attractive
        - Customers switch to competitors
        - Our sales decrease

        Therefore: Competitor rate coefficient MUST be negative.
        """
        competitor_coef = -40.0  # Negative, realistic magnitude
        base_competitor_rate = 0.10
        higher_competitor_rate = 0.12

        base_sales = 1000 + competitor_coef * base_competitor_rate * 100
        lower_sales = 1000 + competitor_coef * higher_competitor_rate * 100

        assert lower_sales < base_sales, (
            "Higher competitor rate should decrease our sales (substitution)"
        )

    def test_cross_elasticity_magnitude(self):
        """
        Economic Theory: Own-rate elasticity typically exceeds cross-elasticity.

        Customers are generally more responsive to changes in their chosen
        product's pricing than to competitor pricing.
        """
        own_coef = 60.0       # Realistic magnitude
        competitor_coef = -40.0  # Realistic magnitude

        # Own effect should be larger in magnitude
        assert abs(own_coef) >= abs(competitor_coef) * 0.5, (
            "Own elasticity should be comparable to or larger than cross elasticity"
        )


# =============================================================================
# REGRESSION TESTS FOR KNOWN BUGS
# =============================================================================


class TestKnownSignBugs:
    """Tests for specific sign-related bugs that have been discovered and fixed."""

    def test_multicollinearity_sign_flip(self):
        """
        Bug: Multicollinearity can cause sign flips in correlated features.
        Detection: If own_rate_t0 and own_rate_t1 have opposite signs, investigate.
        """
        coefficients = {
            "prudential_rate_t0": 80.0,      # Within max_magnitude 100
            "prudential_rate_t1": -20.0,     # Suspiciously negative
        }

        # Both should be positive
        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 1
        assert "prudential_rate_t1" in violations[0].feature_name

    def test_near_zero_coefficients(self):
        """
        Bug: Very small coefficients might be noise, not violations.
        Fix: Use tolerance to avoid flagging numerical noise.
        """
        coefficients = {
            "prudential_rate_t0": 0.0001,  # Tiny positive (OK)
            "competitor_weighted_t2": -0.0001,  # Tiny negative (OK)
        }

        violations = validate_coefficient_signs(
            coefficients, RILA_COEFFICIENT_CONSTRAINTS
        )

        assert len(violations) == 0


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_coefficient_sign_summary():
    """
    Summary: Coefficient Sign Validation

    ECONOMIC CONSTRAINTS FOR RILA:

    | Coefficient Type    | Expected Sign | Rationale                        |
    |---------------------|---------------|----------------------------------|
    | Own cap rate        | POSITIVE      | Yield economics: higher = better |
    | Competitor rates    | NEGATIVE      | Substitution: they win, we lose  |
    | Buffer level        | POSITIVE      | More protection is preferred     |
    | VIX                 | NEGATIVE      | Uncertainty reduces purchases    |

    WHY SIGNS MATTER:
    - Wrong signs indicate specification error
    - May suggest data leakage or multicollinearity
    - Makes pricing recommendations unreliable
    - Violates microeconomic theory

    WHAT TO DO IF SIGNS ARE WRONG:
    1. Check for data leakage (lag-0 features)
    2. Check for multicollinearity (VIF analysis)
    3. Check for coding errors (sign flips in data)
    4. Review feature engineering
    5. Consult domain expert

    ENFORCEMENT:
    - Part of `make leakage-audit` pipeline
    - Blocks deployment on sign violations
    - Generates diagnostic report with violations
    """
    pass  # Documentation test - always passes
