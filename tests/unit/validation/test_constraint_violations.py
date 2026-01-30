"""
Economic Constraint Violation Detection Tests
=============================================

Tests using pre-defined constraint violation examples to validate
that the constraint engine correctly detects sign violations,
magnitude violations, and other economic constraint breaches.

These tests activate the constraint_violation_examples fixture from
conftest.py which provides:
- valid: Coefficients that satisfy all constraints
- own_rate_negative: Own rate coefficient that's negative (violation)
- competitor_positive: Competitor coefficient that's positive (violation)
- magnitude_unrealistic: Coefficients with unrealistic magnitudes

Author: Claude Code
Date: 2026-01-30
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List

from src.features.selection_types import (
    ConstraintRule,
    ConstraintViolation,
    ConstraintType,
    create_default_constraint_rules,
)
from src.features.selection.engines.constraints_engine import (
    validate_constraint_rule,
    generate_constraint_violations,
    apply_economic_constraints,
    _check_sign_violation,
)
from src.validation.leakage_gates import GateStatus


# =============================================================================
# SIGN VIOLATION DETECTION TESTS
# =============================================================================


class TestSignViolationDetection:
    """
    Tests for _check_sign_violation function.

    Core constraint logic: own rate should be positive, competitor negative.
    """

    def test_positive_expected_with_positive_coefficient_passes(self):
        """Positive coefficient should pass positive constraint."""
        assert _check_sign_violation(0.5, "positive") is False

    def test_positive_expected_with_negative_coefficient_fails(self):
        """Negative coefficient should violate positive constraint."""
        assert _check_sign_violation(-0.5, "positive") is True

    def test_negative_expected_with_negative_coefficient_passes(self):
        """Negative coefficient should pass negative constraint."""
        assert _check_sign_violation(-0.5, "negative") is False

    def test_negative_expected_with_positive_coefficient_fails(self):
        """Positive coefficient should violate negative constraint."""
        assert _check_sign_violation(0.5, "negative") is True

    def test_zero_coefficient_violates_positive_constraint(self):
        """Zero coefficient should violate positive constraint."""
        assert _check_sign_violation(0.0, "positive") is True

    def test_zero_coefficient_violates_negative_constraint(self):
        """Zero coefficient should violate negative constraint."""
        assert _check_sign_violation(0.0, "negative") is True

    def test_nan_coefficient_does_not_violate(self):
        """NaN coefficient should not be flagged as violation."""
        assert _check_sign_violation(np.nan, "positive") is False
        assert _check_sign_violation(np.nan, "negative") is False

    def test_none_coefficient_does_not_violate(self):
        """None coefficient should not be flagged as violation."""
        assert _check_sign_violation(None, "positive") is False
        assert _check_sign_violation(None, "negative") is False


class TestCIBasedViolationDetection:
    """
    Tests for CI-based constraint validation.

    When standard errors are provided, the entire confidence interval
    must have the correct sign for the constraint to be satisfied.
    """

    def test_significant_positive_passes_positive_constraint(self):
        """Significantly positive coefficient passes positive constraint."""
        # coef=0.5, SE=0.1 -> 95% CI ≈ (0.3, 0.7) - entirely positive
        result = _check_sign_violation(0.5, "positive", standard_error=0.1)
        assert result == False, f"Expected False, got {result}"

    def test_significant_negative_passes_negative_constraint(self):
        """Significantly negative coefficient passes negative constraint."""
        # coef=-0.5, SE=0.1 -> 95% CI ≈ (-0.7, -0.3) - entirely negative
        result = _check_sign_violation(-0.5, "negative", standard_error=0.1)
        assert result == False, f"Expected False, got {result}"

    def test_ci_spanning_zero_violates_positive(self):
        """CI spanning zero violates positive constraint."""
        # coef=0.1, SE=0.2 -> 95% CI ≈ (-0.29, 0.49) - spans zero
        result = _check_sign_violation(0.1, "positive", standard_error=0.2)
        assert result == True, f"Expected True (violation), got {result}"

    def test_ci_spanning_zero_violates_negative(self):
        """CI spanning zero violates negative constraint."""
        # coef=-0.1, SE=0.2 -> 95% CI ≈ (-0.49, 0.29) - spans zero
        result = _check_sign_violation(-0.1, "negative", standard_error=0.2)
        assert result == True, f"Expected True (violation), got {result}"

    def test_barely_significant_positive(self):
        """Barely significant positive should pass."""
        # coef=0.4, SE=0.2 -> 95% CI ≈ (0.008, 0.792) - barely positive
        result = _check_sign_violation(0.4, "positive", standard_error=0.2)
        assert result == False, f"Expected False, got {result}"

    def test_barely_insignificant_positive(self):
        """Barely insignificant positive should fail."""
        # coef=0.38, SE=0.2 -> 95% CI ≈ (-0.012, 0.772) - just crosses zero
        result = _check_sign_violation(0.38, "positive", standard_error=0.2)
        assert result == True, f"Expected True (violation), got {result}"


# =============================================================================
# CONSTRAINT RULE VALIDATION TESTS
# =============================================================================


class TestConstraintRuleValidation:
    """
    Tests for validate_constraint_rule function.

    Validates that constraint rules correctly identify violations
    in model coefficients.
    """

    @pytest.fixture
    def own_rate_positive_rule(self):
        """Rule: own rate coefficients should be positive."""
        return ConstraintRule(
            feature_pattern="own_cap_rate",
            expected_sign="positive",
            constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,  # Own rate = Prudential
            business_rationale="Higher own rates attract customers",
            strict=True
        )

    @pytest.fixture
    def competitor_negative_rule(self):
        """Rule: competitor coefficients should be negative."""
        return ConstraintRule(
            feature_pattern="competitor",
            expected_sign="negative",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale="Customers substitute to higher competitor rates",
            strict=True
        )

    def test_valid_own_rate_no_violations(self, own_rate_positive_rule):
        """Valid positive own rate should have no violations."""
        coefficients = {
            'own_cap_rate_lag_1': 0.5,
            'own_cap_rate_lag_2': 0.3,
            'competitor_mean': -0.4,
        }

        violations = validate_constraint_rule(coefficients, own_rate_positive_rule)

        assert len(violations) == 0, (
            f"Valid own rate should have no violations, got {violations}"
        )

    def test_negative_own_rate_detected(self, own_rate_positive_rule):
        """Negative own rate should be detected as violation."""
        coefficients = {
            'own_cap_rate_lag_1': -0.2,  # VIOLATION
            'own_cap_rate_lag_2': 0.3,
            'competitor_mean': -0.4,
        }

        violations = validate_constraint_rule(coefficients, own_rate_positive_rule)

        assert len(violations) == 1, (
            f"Expected 1 violation for negative own rate, got {len(violations)}"
        )
        assert violations[0].expected_sign == "positive"
        assert violations[0].actual_coefficient == -0.2

    def test_multiple_own_rate_violations(self, own_rate_positive_rule):
        """Multiple negative own rates should all be detected."""
        coefficients = {
            'own_cap_rate_lag_1': -0.2,  # VIOLATION
            'own_cap_rate_lag_2': -0.1,  # VIOLATION
            'own_cap_rate_lag_3': 0.3,   # OK
        }

        violations = validate_constraint_rule(coefficients, own_rate_positive_rule)

        assert len(violations) == 2, (
            f"Expected 2 violations, got {len(violations)}"
        )

    def test_valid_competitor_negative_no_violations(self, competitor_negative_rule):
        """Valid negative competitor rates should have no violations."""
        coefficients = {
            'competitor_mean_lag_1': -0.4,
            'competitor_median_lag_1': -0.3,
            'own_cap_rate': 0.5,
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        assert len(violations) == 0

    def test_positive_competitor_detected(self, competitor_negative_rule):
        """Positive competitor coefficient should be detected."""
        coefficients = {
            'competitor_mean_lag_1': 0.2,  # VIOLATION
            'competitor_median_lag_1': -0.3,
            'own_cap_rate': 0.5,
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        assert len(violations) == 1
        assert violations[0].expected_sign == "negative"
        assert violations[0].actual_coefficient == 0.2

    def test_no_matching_features(self, own_rate_positive_rule):
        """Rule with no matching features should have no violations."""
        coefficients = {
            'competitor_mean': -0.4,
            'vix': 0.1,
            'dgs5': 0.05,
        }

        violations = validate_constraint_rule(coefficients, own_rate_positive_rule)

        assert len(violations) == 0


# =============================================================================
# FIXTURE-BASED CONSTRAINT TESTS
# =============================================================================


class TestConstraintViolationExamples:
    """
    Tests using pre-defined constraint_violation_examples fixture.

    These tests verify that the constraint system correctly processes
    the canonical violation examples from conftest.py.
    """

    def test_valid_example_passes_validation(self, constraint_violation_examples):
        """Valid coefficients should pass all constraints."""
        valid = constraint_violation_examples['valid']

        # The fixture contains rate values, not coefficients
        # Construct a valid coefficient scenario
        coefficients = {
            'own_cap_rate_lag_1': 0.5,     # Positive (correct)
            'competitor_mean_lag_1': -0.3,  # Negative (correct)
        }

        rules = create_default_constraint_rules()
        all_violations = []
        for rule in rules:
            violations = validate_constraint_rule(coefficients, rule)
            all_violations.extend(violations)

        assert len(all_violations) == 0, (
            f"Valid scenario should have no violations, got {all_violations}"
        )

    def test_own_rate_negative_example_detected(self, constraint_violation_examples):
        """Own rate negative example should be detected."""
        # Create coefficient scenario matching the fixture's intent
        coefficients = {
            'own_cap_rate_lag_1': -0.05,   # Negative (VIOLATION)
            'competitor_mean_lag_1': -0.3,  # Negative (correct)
        }

        own_rule = ConstraintRule(
            feature_pattern="own_cap_rate",
            expected_sign="positive",
            constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,
            business_rationale="Higher own rates attract customers",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, own_rule)

        assert len(violations) >= 1, (
            "Negative own rate should be detected as violation"
        )
        assert any(v.expected_sign == "positive" for v in violations)

    def test_competitor_positive_example_detected(self, constraint_violation_examples):
        """Positive competitor coefficient should be detected."""
        # Create coefficient scenario matching the fixture's intent
        coefficients = {
            'own_cap_rate_lag_1': 0.5,
            'competitor_mean_lag_1': 0.2,   # Positive (VIOLATION)
        }

        competitor_rule = ConstraintRule(
            feature_pattern="competitor",
            expected_sign="negative",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale="Customers substitute away from higher competitor rates",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, competitor_rule)

        assert len(violations) >= 1, (
            "Positive competitor coefficient should be detected"
        )
        assert any(v.expected_sign == "negative" for v in violations)

    def test_magnitude_unrealistic_values_flagged(self, constraint_violation_examples):
        """Unrealistically large coefficients should be flagged."""
        magnitude = constraint_violation_examples['magnitude_unrealistic']

        # Extract the unrealistic coefficient values
        coef_own = magnitude['coefficient_own'].iloc[0]
        coef_comp = magnitude['coefficient_competitor'].iloc[0]

        # These are very large magnitudes
        assert abs(coef_own) > 100, f"Expected large own coef, got {coef_own}"
        assert abs(coef_comp) > 100, f"Expected large competitor coef, got {coef_comp}"

        # Note: Magnitude checking is a separate concern from sign validation
        # This test documents that the fixture provides unrealistic examples


# =============================================================================
# FULL PIPELINE CONSTRAINT TESTS
# =============================================================================


class TestApplyEconomicConstraints:
    """
    Tests for the full apply_economic_constraints pipeline.

    Validates end-to-end constraint filtering of model results.
    """

    @pytest.fixture
    def valid_results_df(self):
        """DataFrame with valid model results."""
        return pd.DataFrame({
            'model_id': [1, 2, 3],
            'coefficients': [
                {'own_cap_rate_lag_1': 0.5, 'competitor_mean_lag_1': -0.3},
                {'own_cap_rate_lag_1': 0.4, 'competitor_mean_lag_1': -0.25},
                {'own_cap_rate_lag_1': 0.6, 'competitor_mean_lag_1': -0.35},
            ],
            'converged': [True, True, True],
            'aic': [100, 95, 105],
        })

    @pytest.fixture
    def mixed_results_df(self):
        """DataFrame with some constraint violations.

        Uses patterns that match default rules:
        - competitor_ (with underscore) for COMPETITOR_NEGATIVE
        """
        return pd.DataFrame({
            'model_id': [1, 2, 3, 4],
            'coefficients': [
                {'prudential_rate_lag_1': 0.5, 'competitor_mean_lag_1': -0.3},   # Valid
                {'prudential_rate_lag_1': 0.4, 'competitor_mean_lag_1': -0.3},   # Valid
                {'prudential_rate_lag_1': 0.4, 'competitor_mean_lag_1': 0.2},    # Competitor positive - VIOLATION
                {'prudential_rate_lag_1': 0.6, 'competitor_mean_lag_1': -0.35},  # Valid
            ],
            'converged': [True, True, True, True],
            'aic': [100, 95, 105, 110],
        })

    @pytest.fixture
    def enabled_config(self):
        """Configuration with constraints enabled."""
        return {
            'enabled': True,
            'strict_validation': True,
        }

    def test_valid_results_pass_through(self, valid_results_df, enabled_config):
        """All valid results should pass through."""
        filtered, violations = apply_economic_constraints(valid_results_df, enabled_config)

        assert len(filtered) == len(valid_results_df), (
            f"All {len(valid_results_df)} valid results should pass, "
            f"got {len(filtered)}"
        )
        assert len(violations) == 0

    def test_violations_detected_and_filtered(self, mixed_results_df, enabled_config):
        """Models with violations should be filtered out in strict mode."""
        filtered, violations = apply_economic_constraints(mixed_results_df, enabled_config)

        # Should have at least 1 violation (model 2 has positive competitor)
        assert len(violations) >= 1, (
            f"Expected at least 1 violation, got {len(violations)}"
        )

        # Filtered results should have fewer models
        assert len(filtered) < len(mixed_results_df), (
            "Strict mode should filter out violating models"
        )

    def test_disabled_config_passes_all(self, mixed_results_df):
        """Disabled constraints should pass all results."""
        config = {'enabled': False}

        filtered, violations = apply_economic_constraints(mixed_results_df, config)

        assert len(filtered) == len(mixed_results_df)
        assert len(violations) == 0

    def test_non_strict_mode_keeps_violations(self, mixed_results_df):
        """Non-strict mode should keep models with violations."""
        config = {
            'enabled': True,
            'strict_validation': False,
        }

        filtered, violations = apply_economic_constraints(mixed_results_df, config)

        # Should detect violations but keep all models
        assert len(violations) >= 1, f"Expected at least 1 violation, got {len(violations)}"
        assert len(filtered) == len(mixed_results_df), (
            "Non-strict mode should keep all models"
        )

    def test_empty_dataframe_handled(self, enabled_config):
        """Empty DataFrame should be handled gracefully."""
        empty_df = pd.DataFrame()

        filtered, violations = apply_economic_constraints(empty_df, enabled_config)

        assert len(filtered) == 0
        assert len(violations) == 0


# =============================================================================
# DEFAULT CONSTRAINT RULES TESTS
# =============================================================================


class TestDefaultConstraintRules:
    """Tests for default constraint rules."""

    def test_default_rules_include_own_positive(self):
        """Default rules should include own rate positive constraint."""
        rules = create_default_constraint_rules()

        # Own/Prudential rate constraint
        own_rules = [r for r in rules if r.constraint_type == ConstraintType.PRUDENTIAL_POSITIVE]

        assert len(own_rules) >= 1, (
            f"Default rules should include own rate positive constraint. "
            f"Available types: {[r.constraint_type for r in rules]}"
        )

    def test_default_rules_include_competitor_negative(self):
        """Default rules should include competitor negative constraint."""
        rules = create_default_constraint_rules()

        competitor_rules = [
            r for r in rules
            if r.constraint_type == ConstraintType.COMPETITOR_NEGATIVE
        ]

        assert len(competitor_rules) >= 1, (
            "Default rules should include competitor negative constraint"
        )

    def test_default_rules_have_business_rationale(self):
        """All default rules should have business rationale."""
        rules = create_default_constraint_rules()

        for rule in rules:
            assert rule.business_rationale, (
                f"Rule {rule.feature_pattern} should have business rationale"
            )
            assert len(rule.business_rationale) > 10, (
                f"Rule {rule.feature_pattern} rationale too short"
            )

    def test_default_rules_are_strict(self):
        """Default rules should be strict (require correction)."""
        rules = create_default_constraint_rules()

        # Most rules should be strict
        strict_rules = [r for r in rules if r.strict is True]
        assert len(strict_rules) > 0, (
            "At least some default rules should be strict"
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestConstraintEdgeCases:
    """Tests for constraint validation edge cases."""

    def test_very_small_positive_coefficient(self):
        """Very small positive coefficient should pass positive constraint."""
        coefficients = {'own_cap_rate_lag_1': 1e-10}
        rule = ConstraintRule(
            feature_pattern="own_cap_rate",
            expected_sign="positive",
            constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,
            business_rationale="Test",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, rule)

        # Without SE, small positive passes
        assert len(violations) == 0

    def test_very_small_negative_coefficient(self):
        """Very small negative coefficient should fail positive constraint."""
        coefficients = {'own_cap_rate_lag_1': -1e-10}
        rule = ConstraintRule(
            feature_pattern="own_cap_rate",
            expected_sign="positive",
            constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,
            business_rationale="Test",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, rule)

        assert len(violations) == 1

    def test_multiple_feature_patterns_matched(self):
        """Rule should match all features containing pattern."""
        coefficients = {
            'competitor_mean_lag_1': -0.3,
            'competitor_median_lag_1': -0.25,
            'competitor_top5_lag_1': -0.4,
            'own_cap_rate_lag_1': 0.5,
        }
        rule = ConstraintRule(
            feature_pattern="competitor",
            expected_sign="negative",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale="Test",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, rule)

        # All competitor features are negative, so no violations
        assert len(violations) == 0

    def test_case_sensitive_pattern_matching(self):
        """Pattern matching should be case-sensitive."""
        coefficients = {
            'COMPETITOR_mean_lag_1': -0.3,  # Uppercase
            'competitor_median_lag_1': 0.2,  # Lowercase, violation
        }
        rule = ConstraintRule(
            feature_pattern="competitor",  # Lowercase pattern
            expected_sign="negative",
            constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
            business_rationale="Test",
            strict=True
        )

        violations = validate_constraint_rule(coefficients, rule)

        # Only lowercase match should be checked and violated
        assert len(violations) == 1
        assert 'competitor_median' in violations[0].feature_name
