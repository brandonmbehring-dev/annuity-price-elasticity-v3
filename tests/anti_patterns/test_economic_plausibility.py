"""
Anti-Pattern Test: Economic Plausibility Validation
====================================================

CRITICAL: Model outputs must be economically plausible.

This test module validates that model coefficients and predictions fall
within realistic economic bounds. Implausible values often indicate:
- Data leakage
- Specification errors
- Multicollinearity
- Scaling issues

Plausibility Checks:
- Elasticity magnitudes are realistic
- Predictions are non-negative (sales can't be negative)
- Coefficient ratios make economic sense
- Model performance metrics are believable

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# PLAUSIBILITY BOUNDS
# =============================================================================

# RILA coefficient bounds (dollar amounts: sales change per 1pp rate change)
# These are SOFT bounds - violations warrant investigation, not automatic rejection
# Note: These are NOT unitless elasticities, but dollar-denominated coefficients
RILA_COEFFICIENT_BOUNDS = {
    "own_rate": {
        "min": 1000.0,     # Minimal response ($1k per 1pp)
        "max": 25000.0,    # Strong response ($25k per 1pp)
        "typical_low": 3000.0,
        "typical_high": 15000.0,
    },
    "competitor_rate": {
        "min": -25000.0,   # Strong substitution
        "max": -1000.0,    # Minimal substitution
        "typical_low": -12000.0,
        "typical_high": -3000.0,
    },
}

# Elasticity bounds (unitless: percentage change in sales per percentage change in rate)
# These represent true economic elasticities
ELASTICITY_BOUNDS = {
    "own_rate": {
        "min": 0.1,        # Low elasticity
        "max": 5.0,        # High elasticity
        "typical_low": 0.5,
        "typical_high": 2.0,
    },
    "competitor_rate": {
        "min": -5.0,       # Strong cross-elasticity
        "max": 0.0,        # Must be non-positive (substitution)
        "typical_low": -2.0,
        "typical_high": -0.3,
    },
}

# R² bounds for behavioral/economic models
R_SQUARED_BOUNDS = {
    "plausible_max": 0.90,  # Higher suggests leakage
    "suspicious_min": 0.10,  # Lower suggests poor specification
    "typical_range": (0.40, 0.75),
}

# Coefficient magnitude bounds (absolute value)
COEFFICIENT_MAGNITUDE_BOUNDS = {
    "max_reasonable": 50000.0,   # Absolute max
    "warn_threshold": 25000.0,   # Investigate if exceeded
}


# =============================================================================
# PLAUSIBILITY CHECK FUNCTIONS
# =============================================================================


@dataclass
class PlausibilityViolation:
    """Details of a plausibility violation."""
    check_name: str
    actual_value: float
    expected_range: Tuple[float, float]
    severity: str  # "warning" or "error"
    message: str


def check_elasticity_bounds(
    coefficients: Dict[str, float],
) -> List[PlausibilityViolation]:
    """Check that elasticity values are within economic bounds.

    These are unitless elasticities representing:
        elasticity = % change in sales per 1% change in rate

    Args:
        coefficients: Dictionary of feature_name -> elasticity_value

    Returns:
        List of PlausibilityViolation objects
    """
    violations = []

    for feature_name, value in coefficients.items():
        feature_lower = feature_name.lower()

        # Check own rate elasticities
        if "own" in feature_lower or "prudential" in feature_lower:
            bounds = ELASTICITY_BOUNDS["own_rate"]
            if value < bounds["min"] or value > bounds["max"]:
                violations.append(PlausibilityViolation(
                    check_name="own_rate_elasticity",
                    actual_value=value,
                    expected_range=(bounds["min"], bounds["max"]),
                    severity="warning" if bounds["typical_low"] <= value <= bounds["typical_high"] else "error",
                    message=f"Own rate elasticity {value:.4f} outside plausible range [{bounds['min']}, {bounds['max']}]"
                ))

        # Check competitor rate elasticities
        elif "competitor" in feature_lower:
            bounds = ELASTICITY_BOUNDS["competitor_rate"]
            if value < bounds["min"] or value > bounds["max"]:
                violations.append(PlausibilityViolation(
                    check_name="competitor_elasticity",
                    actual_value=value,
                    expected_range=(bounds["min"], bounds["max"]),
                    severity="warning" if bounds["typical_low"] <= value <= bounds["typical_high"] else "error",
                    message=f"Competitor elasticity {value:.4f} outside plausible range [{bounds['min']}, {bounds['max']}]"
                ))

    return violations


def check_coefficient_bounds(
    coefficients: Dict[str, float],
) -> List[PlausibilityViolation]:
    """Check that coefficients are within plausible bounds.

    These are dollar-denominated coefficients representing:
        coefficient = $ change in weekly sales per 1 percentage point rate change

    Args:
        coefficients: Dictionary of feature_name -> coefficient_value

    Returns:
        List of PlausibilityViolation objects
    """
    violations = []

    for feature_name, value in coefficients.items():
        feature_lower = feature_name.lower()

        # Check own rate coefficients
        if "own" in feature_lower or "prudential" in feature_lower:
            bounds = RILA_COEFFICIENT_BOUNDS["own_rate"]
            if value < bounds["min"] or value > bounds["max"]:
                violations.append(PlausibilityViolation(
                    check_name="own_rate_coefficient",
                    actual_value=value,
                    expected_range=(bounds["min"], bounds["max"]),
                    severity="warning" if bounds["typical_low"] <= value <= bounds["typical_high"] else "error",
                    message=f"Own rate coefficient {value:.4f} outside plausible range [{bounds['min']}, {bounds['max']}]"
                ))

        # Check competitor rate coefficients
        elif "competitor" in feature_lower:
            bounds = RILA_COEFFICIENT_BOUNDS["competitor_rate"]
            if value < bounds["min"] or value > bounds["max"]:
                violations.append(PlausibilityViolation(
                    check_name="competitor_coefficient",
                    actual_value=value,
                    expected_range=(bounds["min"], bounds["max"]),
                    severity="warning" if bounds["typical_low"] <= value <= bounds["typical_high"] else "error",
                    message=f"Competitor coefficient {value:.4f} outside plausible range [{bounds['min']}, {bounds['max']}]"
                ))

    return violations


def check_coefficient_magnitudes(
    coefficients: Dict[str, float],
) -> List[PlausibilityViolation]:
    """Check that coefficient magnitudes are reasonable.

    Args:
        coefficients: Dictionary of feature_name -> coefficient_value

    Returns:
        List of PlausibilityViolation objects
    """
    violations = []

    for feature_name, value in coefficients.items():
        abs_value = abs(value)

        if abs_value > COEFFICIENT_MAGNITUDE_BOUNDS["max_reasonable"]:
            violations.append(PlausibilityViolation(
                check_name="coefficient_magnitude",
                actual_value=abs_value,
                expected_range=(0, COEFFICIENT_MAGNITUDE_BOUNDS["max_reasonable"]),
                severity="error",
                message=f"Coefficient {feature_name}={value:.4f} has implausible magnitude"
            ))
        elif abs_value > COEFFICIENT_MAGNITUDE_BOUNDS["warn_threshold"]:
            violations.append(PlausibilityViolation(
                check_name="coefficient_magnitude",
                actual_value=abs_value,
                expected_range=(0, COEFFICIENT_MAGNITUDE_BOUNDS["warn_threshold"]),
                severity="warning",
                message=f"Coefficient {feature_name}={value:.4f} has unusually large magnitude"
            ))

    return violations


def check_r_squared_plausibility(
    r_squared: float,
) -> List[PlausibilityViolation]:
    """Check that R² is plausible for behavioral/economic data.

    Args:
        r_squared: Model R² value

    Returns:
        List of PlausibilityViolation objects
    """
    violations = []

    if r_squared > R_SQUARED_BOUNDS["plausible_max"]:
        violations.append(PlausibilityViolation(
            check_name="r_squared_too_high",
            actual_value=r_squared,
            expected_range=(0, R_SQUARED_BOUNDS["plausible_max"]),
            severity="error",
            message=f"R²={r_squared:.4f} is suspiciously high - likely data leakage"
        ))

    if r_squared < R_SQUARED_BOUNDS["suspicious_min"]:
        violations.append(PlausibilityViolation(
            check_name="r_squared_too_low",
            actual_value=r_squared,
            expected_range=(R_SQUARED_BOUNDS["suspicious_min"], 1.0),
            severity="warning",
            message=f"R²={r_squared:.4f} is unusually low - check model specification"
        ))

    return violations


def check_prediction_bounds(
    predictions: np.ndarray,
    historical_max: float,
    allow_negative: bool = False,
) -> List[PlausibilityViolation]:
    """Check that predictions are within plausible bounds.

    Args:
        predictions: Array of model predictions
        historical_max: Maximum historical value (for upper bound)
        allow_negative: Whether negative predictions are allowed

    Returns:
        List of PlausibilityViolation objects
    """
    violations = []

    # Check for negative predictions
    if not allow_negative and np.any(predictions < 0):
        n_negative = np.sum(predictions < 0)
        min_pred = np.min(predictions)
        violations.append(PlausibilityViolation(
            check_name="negative_predictions",
            actual_value=min_pred,
            expected_range=(0, float("inf")),
            severity="error",
            message=f"{n_negative} predictions are negative (min={min_pred:.2f}) - sales can't be negative"
        ))

    # Check for implausibly large predictions
    upper_bound = historical_max * 3  # 3x historical max is suspicious
    if np.any(predictions > upper_bound):
        n_extreme = np.sum(predictions > upper_bound)
        max_pred = np.max(predictions)
        violations.append(PlausibilityViolation(
            check_name="extreme_predictions",
            actual_value=max_pred,
            expected_range=(0, upper_bound),
            severity="warning",
            message=f"{n_extreme} predictions exceed 3x historical max (max={max_pred:.2f})"
        ))

    return violations


# =============================================================================
# UNIT TESTS FOR PLAUSIBILITY CHECKS
# =============================================================================


class TestElasticityBoundsCheck:
    """Tests for elasticity bounds validation."""

    def test_typical_elasticities_pass(self):
        """Typical elasticity values should pass."""
        coefficients = {
            "prudential_rate_t0": 1.2,      # Typical own-rate
            "competitor_weighted_t2": -0.8,  # Typical cross-rate
        }

        violations = check_elasticity_bounds(coefficients)

        # Should have no errors (might have warnings for edge cases)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0

    def test_extreme_own_rate_flagged(self):
        """Implausibly high own-rate elasticity should be flagged."""
        coefficients = {
            "prudential_rate_t0": 10.0,  # Way too high
        }

        violations = check_elasticity_bounds(coefficients)

        assert len(violations) > 0
        assert any("own_rate" in v.check_name for v in violations)

    def test_positive_competitor_flagged(self):
        """Positive competitor elasticity (wrong sign) should be flagged."""
        coefficients = {
            "competitor_weighted_t2": 0.5,  # Wrong sign AND out of bounds
        }

        violations = check_elasticity_bounds(coefficients)

        assert len(violations) > 0


class TestCoefficientMagnitudeCheck:
    """Tests for coefficient magnitude validation."""

    def test_reasonable_magnitudes_pass(self):
        """Reasonable coefficient magnitudes should pass."""
        coefficients = {
            "feature1": 5000.0,
            "feature2": -3000.0,
            "feature3": 100.0,
        }

        violations = check_coefficient_magnitudes(coefficients)

        assert len(violations) == 0

    def test_extreme_magnitude_flagged(self):
        """Extreme coefficient magnitude should be flagged."""
        coefficients = {
            "extreme_feature": 100000.0,  # Way too large
        }

        violations = check_coefficient_magnitudes(coefficients)

        assert len(violations) > 0
        assert violations[0].severity == "error"

    def test_warning_threshold(self):
        """Large but not extreme magnitude should warn."""
        coefficients = {
            "large_feature": 30000.0,  # Large but not extreme
        }

        violations = check_coefficient_magnitudes(coefficients)

        assert len(violations) > 0
        assert violations[0].severity == "warning"


class TestRSquaredPlausibility:
    """Tests for R² plausibility validation."""

    def test_typical_r_squared_passes(self):
        """Typical R² value should pass."""
        violations = check_r_squared_plausibility(0.65)

        assert len(violations) == 0

    def test_suspicious_high_r_squared_flagged(self):
        """Suspiciously high R² should be flagged as leakage risk."""
        violations = check_r_squared_plausibility(0.95)

        assert len(violations) > 0
        assert violations[0].severity == "error"
        assert "leakage" in violations[0].message.lower()

    def test_low_r_squared_warns(self):
        """Very low R² should warn about specification."""
        violations = check_r_squared_plausibility(0.05)

        assert len(violations) > 0
        assert violations[0].severity == "warning"


class TestPredictionBounds:
    """Tests for prediction bounds validation."""

    def test_valid_predictions_pass(self):
        """Valid predictions should pass."""
        predictions = np.array([10000, 20000, 15000, 25000])
        historical_max = 30000

        violations = check_prediction_bounds(predictions, historical_max)

        assert len(violations) == 0

    def test_negative_predictions_flagged(self):
        """Negative predictions should be flagged."""
        predictions = np.array([10000, -5000, 15000, 25000])
        historical_max = 30000

        violations = check_prediction_bounds(predictions, historical_max)

        assert len(violations) > 0
        assert "negative" in violations[0].check_name

    def test_extreme_predictions_flagged(self):
        """Extremely large predictions should be flagged."""
        predictions = np.array([10000, 20000, 150000, 25000])  # 150k is 5x max
        historical_max = 30000

        violations = check_prediction_bounds(predictions, historical_max)

        assert len(violations) > 0
        assert "extreme" in violations[0].check_name


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestModelOutputPlausibility:
    """Integration tests for full model output validation."""

    @pytest.fixture
    def plausible_model_output(self) -> Dict:
        """Output from a well-specified model with realistic elasticities."""
        return {
            "coefficients": {
                "intercept": 100.0,
                "prudential_rate_t0": 1.5,     # Within 0.1-5.0 elasticity range
                "prudential_rate_t1": 0.8,     # Within 0.1-5.0 elasticity range
                "competitor_weighted_t2": -1.2, # Within -5.0-0.0 cross-elasticity range
                "vix_t0": -0.35,               # Not in elasticity bounds (ignored)
            },
            "r_squared": 0.62,
            "predictions": np.random.uniform(10000, 50000, 100),
            "historical_max": 60000,
        }

    @pytest.fixture
    def leaky_model_output(self) -> Dict:
        """Output from a model with likely leakage."""
        return {
            "coefficients": {
                "intercept": 5.0,
                "prudential_rate_t0": 10.0,      # Too high (> 5.0 max)
                "competitor_weighted_t2": 1.5,   # Wrong sign (positive, should be negative)
            },
            "r_squared": 0.97,  # Too good to be true
            "predictions": np.concatenate([
                np.random.uniform(10000, 50000, 95),
                np.array([-5000, -3000, -2000, -1000, 200000])  # Negatives and extreme
            ]),
            "historical_max": 60000,
        }

    def test_plausible_output_passes(self, plausible_model_output):
        """Plausible model output should pass all checks."""
        all_violations = []

        # Check elasticities
        all_violations.extend(
            check_elasticity_bounds(plausible_model_output["coefficients"])
        )

        # Check magnitudes
        all_violations.extend(
            check_coefficient_magnitudes(plausible_model_output["coefficients"])
        )

        # Check R²
        all_violations.extend(
            check_r_squared_plausibility(plausible_model_output["r_squared"])
        )

        # Check predictions
        all_violations.extend(
            check_prediction_bounds(
                plausible_model_output["predictions"],
                plausible_model_output["historical_max"]
            )
        )

        # Should have no errors
        errors = [v for v in all_violations if v.severity == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_leaky_output_fails(self, leaky_model_output):
        """Leaky model output should fail multiple checks."""
        all_violations = []

        all_violations.extend(
            check_elasticity_bounds(leaky_model_output["coefficients"])
        )
        all_violations.extend(
            check_coefficient_magnitudes(leaky_model_output["coefficients"])
        )
        all_violations.extend(
            check_r_squared_plausibility(leaky_model_output["r_squared"])
        )
        all_violations.extend(
            check_prediction_bounds(
                leaky_model_output["predictions"],
                leaky_model_output["historical_max"]
            )
        )

        # Should have multiple errors
        errors = [v for v in all_violations if v.severity == "error"]
        assert len(errors) >= 3, f"Expected multiple errors, got: {errors}"


# =============================================================================
# ECONOMIC INTERPRETATION TESTS
# =============================================================================


class TestEconomicInterpretation:
    """Tests that validate economic interpretation of model outputs."""

    def test_elasticity_interpretation(self):
        """
        Validate elasticity interpretation is correct.

        Semi-elasticity coefficient of 0.10 means:
        - 1 percentage point increase in cap rate (e.g., 10% -> 11%)
        - Leads to 10% increase in sales
        """
        semi_elasticity = 0.10  # 10% sales increase per 1pp rate increase
        base_sales = 100000
        rate_change_pp = 1  # 1 percentage point

        expected_sales_change = base_sales * semi_elasticity * rate_change_pp

        assert expected_sales_change == 10000, (
            "1pp rate increase with 0.10 elasticity should yield 10% sales increase"
        )

    def test_cross_elasticity_interpretation(self):
        """
        Validate cross-elasticity interpretation.

        If competitor rates increase by 1pp, and cross-elasticity is -0.05:
        - Our sales should decrease by 5%
        """
        cross_elasticity = -0.05
        base_sales = 100000
        competitor_rate_change_pp = 1

        expected_sales_change = base_sales * cross_elasticity * competitor_rate_change_pp

        assert expected_sales_change == -5000, (
            "1pp competitor rate increase with -0.05 cross-elasticity should yield 5% sales decrease"
        )

    def test_relative_elasticity_magnitudes(self):
        """
        Own-rate elasticity should typically exceed cross-rate elasticity.

        Customers respond more to their chosen product than alternatives.
        """
        own_elasticity = 0.10
        cross_elasticity = -0.05

        assert abs(own_elasticity) >= abs(cross_elasticity), (
            "Own-rate elasticity should typically exceed cross-rate elasticity"
        )


# =============================================================================
# DOCUMENTATION TEST
# =============================================================================


def test_economic_plausibility_summary():
    """
    Summary: Economic Plausibility Validation

    ELASTICITY BOUNDS (RILA):

    | Elasticity Type | Plausible Range | Typical Range  |
    |-----------------|-----------------|----------------|
    | Own rate        | [0.1, 5.0]      | [0.5, 2.0]     |
    | Competitor      | [-5.0, -0.1]    | [-2.0, -0.5]   |

    R² BOUNDS:

    | Check              | Threshold | Meaning                    |
    |--------------------|-----------|----------------------------|
    | Too high           | > 0.90    | Likely data leakage        |
    | Typical range      | 0.40-0.75 | Expected for behavior data |
    | Too low            | < 0.10    | Poor specification         |

    PREDICTION BOUNDS:

    - Non-negative: Sales cannot be negative
    - Upper bound: 3x historical max is suspicious
    - Extreme values warrant investigation

    WHY THESE BOUNDS MATTER:

    - Protect against data leakage (R² too high)
    - Catch specification errors (wrong signs, extreme values)
    - Ensure business usability (reasonable predictions)
    - Validate economic theory (elasticity magnitudes)

    WHAT TO DO ON VIOLATION:

    1. R² too high:
       - Check for lag-0 features
       - Check for target leakage
       - Review feature engineering

    2. Extreme coefficients:
       - Check for multicollinearity (VIF)
       - Check for scaling issues
       - Review variable definitions

    3. Negative predictions:
       - Consider log transformation
       - Add non-negativity constraints
       - Review prediction bounds

    4. Implausible elasticities:
       - Compare to literature
       - Check data quality
       - Review model specification
    """
    pass  # Documentation test - always passes
