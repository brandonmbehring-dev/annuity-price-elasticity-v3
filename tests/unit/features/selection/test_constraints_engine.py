"""
Tests for Constraints Engine - Economic Validation Module.

Tests cover:
- validate_constraint_rule: Single rule validation
- generate_constraint_violations: Comprehensive violation detection
- apply_economic_constraints: Main filtering function

Design Principles:
- Real assertions about correctness
- Test happy path + error cases + edge cases
- Business rule validation testing

Author: Claude Code
Date: 2026-01-23
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import asdict

from src.features.selection.engines.constraints_engine import (
    validate_constraint_rule,
    generate_constraint_violations,
    apply_economic_constraints,
)
from src.features.selection_types import (
    ConstraintRule,
    ConstraintViolation,
    ConstraintType,
    EconomicConstraintConfig,
    create_default_constraint_rules,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def competitor_negative_rule():
    """Create competitor negative constraint rule."""
    return ConstraintRule(
        feature_pattern="competitor_",
        expected_sign="negative",
        constraint_type=ConstraintType.COMPETITOR_NEGATIVE,
        business_rationale="Higher competitor rates should reduce own sales",
        strict=True
    )


@pytest.fixture
def prudential_positive_rule():
    """Create prudential positive constraint rule."""
    return ConstraintRule(
        feature_pattern="prudential_rate",
        expected_sign="positive",
        constraint_type=ConstraintType.PRUDENTIAL_POSITIVE,
        business_rationale="Higher own rates should increase sales",
        strict=True
    )


@pytest.fixture
def warning_rule():
    """Create non-strict rule that generates warnings instead of errors."""
    return ConstraintRule(
        feature_pattern="test_",
        expected_sign="positive",
        constraint_type=ConstraintType.AUTOREGRESSIVE_POSITIVE,
        business_rationale="Test rule",
        strict=False  # Generates warnings, not errors
    )


@pytest.fixture
def valid_coefficients():
    """Create coefficients that satisfy default constraints."""
    return {
        'Intercept': 5.0,
        'competitor_mid_t1': -0.5,  # Negative - satisfies rule
        'competitor_mid_t2': -0.3,  # Negative - satisfies rule
        'prudential_rate_current': 0.8,  # Positive - satisfies rule
        'sales_target_t1': 0.6,  # Positive
    }


@pytest.fixture
def violating_coefficients():
    """Create coefficients that violate constraints."""
    return {
        'Intercept': 5.0,
        'competitor_mid_t1': 0.5,  # POSITIVE - VIOLATION!
        'competitor_mid_t2': -0.3,  # Negative - OK
        'prudential_rate_current': -0.2,  # NEGATIVE - VIOLATION!
        'sales_target_t1': 0.6,  # Positive - OK
    }


@pytest.fixture
def results_dataframe_valid(valid_coefficients):
    """Create results DataFrame with valid models."""
    return pd.DataFrame([
        {
            'features': 'competitor_mid_t1 + prudential_rate_current',
            'aic': 100.0,
            'converged': True,
            'coefficients': valid_coefficients,
        },
        {
            'features': 'competitor_mid_t2 + prudential_rate_current',
            'aic': 105.0,
            'converged': True,
            'coefficients': valid_coefficients,
        },
    ])


@pytest.fixture
def results_dataframe_mixed(valid_coefficients, violating_coefficients):
    """Create results DataFrame with mix of valid and violating models."""
    return pd.DataFrame([
        {
            'features': 'model_0_valid',
            'aic': 100.0,
            'converged': True,
            'coefficients': valid_coefficients,
        },
        {
            'features': 'model_1_violating',
            'aic': 95.0,  # Better AIC but violates constraints
            'converged': True,
            'coefficients': violating_coefficients,
        },
        {
            'features': 'model_2_valid',
            'aic': 102.0,
            'converged': True,
            'coefficients': valid_coefficients,
        },
    ])


@pytest.fixture
def constraint_config_enabled():
    """Create enabled constraint config."""
    return {
        'enabled': True,
        'strict_validation': True,
    }


@pytest.fixture
def constraint_config_disabled():
    """Create disabled constraint config."""
    return {
        'enabled': False,
    }


@pytest.fixture
def constraint_config_non_strict():
    """Create non-strict constraint config (warnings only)."""
    return {
        'enabled': True,
        'strict_validation': False,
    }


# =============================================================================
# Tests for validate_constraint_rule
# =============================================================================

class TestValidateConstraintRule:
    """Test suite for validate_constraint_rule function."""

    def test_returns_empty_list_when_satisfied(
        self, competitor_negative_rule, valid_coefficients
    ):
        """Verify empty list returned when constraint is satisfied."""
        violations = validate_constraint_rule(
            valid_coefficients,
            competitor_negative_rule
        )

        assert violations == []

    def test_detects_sign_violation(
        self, competitor_negative_rule, violating_coefficients
    ):
        """Verify violation detected for wrong sign."""
        violations = validate_constraint_rule(
            violating_coefficients,
            competitor_negative_rule
        )

        # Should detect violation for competitor_mid_t1 (positive when should be negative)
        assert len(violations) >= 1

        violation = violations[0]
        assert isinstance(violation, ConstraintViolation)
        assert 'competitor' in violation.feature_name.lower()

    def test_violation_contains_correct_metadata(
        self, competitor_negative_rule, violating_coefficients
    ):
        """Verify violation contains correct metadata."""
        violations = validate_constraint_rule(
            violating_coefficients,
            competitor_negative_rule
        )

        violation = violations[0]
        assert violation.expected_sign == "negative"
        assert violation.constraint_type == ConstraintType.COMPETITOR_NEGATIVE
        assert violation.business_rationale == competitor_negative_rule.business_rationale

    def test_strict_rule_generates_error_severity(
        self, competitor_negative_rule, violating_coefficients
    ):
        """Verify strict rule generates ERROR severity."""
        violations = validate_constraint_rule(
            violating_coefficients,
            competitor_negative_rule
        )

        assert violations[0].violation_severity == "ERROR"

    def test_non_strict_rule_generates_warning_severity(
        self, warning_rule
    ):
        """Verify non-strict rule generates WARNING severity."""
        coefficients = {'test_feature': -1.0}  # Negative when should be positive

        violations = validate_constraint_rule(coefficients, warning_rule)

        assert len(violations) == 1
        assert violations[0].violation_severity == "WARNING"

    def test_no_matching_features_returns_empty(self, competitor_negative_rule):
        """Verify empty list when no features match pattern."""
        coefficients = {
            'Intercept': 1.0,
            'other_feature': 0.5,
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        assert violations == []

    def test_intercept_is_excluded(self, competitor_negative_rule):
        """Verify Intercept is excluded from validation."""
        # Even if Intercept contained 'competitor_', it should be excluded
        coefficients = {
            'Intercept': 10.0,  # Large positive - would violate if checked
            'competitor_mid': -0.5,
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        # Should be empty (Intercept excluded, competitor_mid is OK)
        assert violations == []

    def test_handles_nan_coefficient(self, competitor_negative_rule):
        """Verify NaN coefficient is handled gracefully."""
        coefficients = {
            'competitor_mid': np.nan,
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        # NaN should not cause violation
        assert violations == []

    def test_multiple_violations_detected(self, competitor_negative_rule):
        """Verify multiple violations from same rule."""
        coefficients = {
            'competitor_a': 0.5,  # Positive - violation
            'competitor_b': 0.3,  # Positive - violation
            'competitor_c': -0.2,  # Negative - OK
        }

        violations = validate_constraint_rule(coefficients, competitor_negative_rule)

        assert len(violations) == 2


# =============================================================================
# Tests for generate_constraint_violations
# =============================================================================

class TestGenerateConstraintViolations:
    """Test suite for generate_constraint_violations function."""

    def test_returns_empty_for_valid_models(self, results_dataframe_valid):
        """Verify empty list for models that satisfy constraints."""
        rules = create_default_constraint_rules()

        violations = generate_constraint_violations(
            results_dataframe_valid,
            rules
        )

        assert violations == []

    def test_detects_violations_across_models(self, results_dataframe_mixed):
        """Verify violations detected across multiple models."""
        rules = create_default_constraint_rules()

        violations = generate_constraint_violations(
            results_dataframe_mixed,
            rules
        )

        # Should detect violations in model 1
        assert len(violations) > 0

        # Violations should include model index
        violation_names = [v.feature_name for v in violations]
        assert any('Model_1' in name for name in violation_names)

    def test_handles_empty_dataframe(self):
        """Verify empty DataFrame returns empty violations."""
        empty_df = pd.DataFrame()
        rules = create_default_constraint_rules()

        violations = generate_constraint_violations(empty_df, rules)

        assert violations == []

    def test_handles_missing_coefficients_column(self):
        """Verify missing coefficients column handled gracefully."""
        df = pd.DataFrame([{'features': 'a + b', 'aic': 100.0}])
        rules = create_default_constraint_rules()

        violations = generate_constraint_violations(df, rules)

        assert violations == []

    def test_skips_non_converged_models(self):
        """Verify non-converged models are skipped."""
        df = pd.DataFrame([
            {
                'features': 'test',
                'aic': 100.0,
                'converged': False,  # Should be skipped
                'coefficients': {'competitor_mid': 1.0},  # Would violate
            }
        ])
        rules = create_default_constraint_rules()

        violations = generate_constraint_violations(df, rules)

        assert violations == []


# =============================================================================
# Tests for apply_economic_constraints
# =============================================================================

class TestApplyEconomicConstraints:
    """Test suite for apply_economic_constraints function."""

    def test_returns_tuple(
        self, results_dataframe_valid, constraint_config_enabled
    ):
        """Verify function returns tuple of (DataFrame, violations)."""
        result = apply_economic_constraints(
            results_dataframe_valid,
            constraint_config_enabled
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_disabled_config_returns_all_results(
        self, results_dataframe_mixed, constraint_config_disabled
    ):
        """Verify disabled config returns all results unchanged."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_disabled
        )

        assert len(valid_df) == len(results_dataframe_mixed)
        assert violations == []

    def test_strict_mode_filters_violating_models(
        self, results_dataframe_mixed, constraint_config_enabled
    ):
        """Verify strict mode removes violating models."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_enabled
        )

        # Should have fewer models after filtering
        assert len(valid_df) < len(results_dataframe_mixed)

        # Model 1 (the violating one) should be removed
        assert len(valid_df) == 2  # Models 0 and 2 remain

    def test_non_strict_mode_keeps_all_models(
        self, results_dataframe_mixed, constraint_config_non_strict
    ):
        """Verify non-strict mode keeps all models (just warns)."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_non_strict
        )

        # All models should be preserved
        assert len(valid_df) == len(results_dataframe_mixed)

        # But violations should still be detected
        assert len(violations) > 0

    def test_violations_returned_even_when_filtering(
        self, results_dataframe_mixed, constraint_config_enabled
    ):
        """Verify violations are returned even when models are filtered."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_enabled
        )

        assert len(violations) > 0

    def test_valid_models_not_filtered(
        self, results_dataframe_valid, constraint_config_enabled
    ):
        """Verify valid models are preserved."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_valid,
            constraint_config_enabled
        )

        assert len(valid_df) == len(results_dataframe_valid)
        assert violations == []

    def test_empty_dataframe_handled(self, constraint_config_enabled):
        """Verify empty DataFrame handled gracefully."""
        empty_df = pd.DataFrame()

        valid_df, violations = apply_economic_constraints(
            empty_df,
            constraint_config_enabled
        )

        assert len(valid_df) == 0
        assert violations == []

    def test_missing_coefficients_column_handled(self, constraint_config_enabled):
        """Verify missing coefficients column handled gracefully."""
        df = pd.DataFrame([{'features': 'a + b', 'aic': 100.0}])

        valid_df, violations = apply_economic_constraints(
            df,
            constraint_config_enabled
        )

        assert len(valid_df) == 1  # Returns original
        assert violations == []


# =============================================================================
# Tests for Default Constraint Rules
# =============================================================================

class TestDefaultConstraintRules:
    """Test suite for default constraint rule configuration."""

    def test_default_rules_exist(self):
        """Verify default rules are created."""
        rules = create_default_constraint_rules()

        assert len(rules) > 0

    def test_competitor_negative_rule_exists(self):
        """Verify competitor negative rule in defaults."""
        rules = create_default_constraint_rules()

        competitor_rules = [
            r for r in rules
            if r.constraint_type == ConstraintType.COMPETITOR_NEGATIVE
        ]

        assert len(competitor_rules) >= 1
        assert competitor_rules[0].expected_sign == "negative"

    def test_prudential_positive_rule_exists(self):
        """Verify prudential positive rule in defaults."""
        rules = create_default_constraint_rules()

        prudential_rules = [
            r for r in rules
            if r.constraint_type == ConstraintType.PRUDENTIAL_POSITIVE
        ]

        assert len(prudential_rules) >= 1
        assert prudential_rules[0].expected_sign == "positive"

    def test_all_rules_have_business_rationale(self):
        """Verify all default rules have business rationale."""
        rules = create_default_constraint_rules()

        for rule in rules:
            assert rule.business_rationale is not None
            assert len(rule.business_rationale) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestConstraintsEngineIntegration:
    """Integration tests for constraints engine."""

    def test_full_constraint_validation_flow(
        self, results_dataframe_mixed, constraint_config_enabled
    ):
        """Test complete constraint validation flow."""
        # Apply constraints
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_enabled
        )

        # Verify results
        assert len(valid_df) < len(results_dataframe_mixed)  # Some filtered
        assert len(violations) > 0  # Violations detected

        # Verify filtered models are actually valid
        rules = create_default_constraint_rules()
        recheck_violations = generate_constraint_violations(valid_df, rules)

        # Remaining models should have no violations
        # (some might be warnings from non-strict rules)
        error_violations = [
            v for v in recheck_violations
            if v.violation_severity == "ERROR"
        ]
        assert len(error_violations) == 0

    def test_constraint_filtering_preserves_best_valid_model(
        self, results_dataframe_mixed, constraint_config_enabled
    ):
        """Verify best valid model is preserved after filtering."""
        valid_df, violations = apply_economic_constraints(
            results_dataframe_mixed,
            constraint_config_enabled
        )

        # Find best AIC among valid models
        best_aic = valid_df['aic'].min()

        # Model 0 had AIC=100 and was valid
        # Model 1 had better AIC=95 but was invalid
        # Model 2 had AIC=102 and was valid
        # Best valid should be 100.0
        assert best_aic == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
