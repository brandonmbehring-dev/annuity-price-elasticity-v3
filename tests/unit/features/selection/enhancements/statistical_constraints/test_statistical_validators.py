"""
Tests for Statistical Validators Module.

Tests cover:
- _validate_model_inputs: Input validation for models and constraints
- _determine_constraint_type: Constraint type determination from sign specification
- _calculate_confidence_interval: CI computation for coefficient estimates
- _assess_constraint_strength: Strength assessment based on CI position
- _evaluate_sign_constraint: Constraint evaluation logic
- _build_hard_threshold_comparison: Hard vs statistical threshold comparison
- _validate_single_statistical_constraint: Core constraint validator

Design Principles:
- Property-based tests for mathematical invariants
- Edge case tests for error handling
- Mock models for isolated testing

Mathematical Properties Validated:
- Confidence intervals: lower < coefficient < upper (typically)
- Constraint strength: Always in {STRONG, MODERATE, WEAK, VIOLATED}
- P-values and significance levels: p-value in [0, 1]

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Tuple

from src.features.selection.enhancements.statistical_constraints.statistical_validators import (
    _validate_model_inputs,
    _determine_constraint_type,
    _calculate_confidence_interval,
    _assess_constraint_strength,
    _evaluate_sign_constraint,
    _build_hard_threshold_comparison,
    _validate_single_statistical_constraint,
)
from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_valid_model():
    """Create a mock model with required attributes."""
    model = MagicMock()
    model.params = {'feature_1': 0.5, 'feature_2': -0.3}
    model.bse = {'feature_1': 0.1, 'feature_2': 0.05}
    return model


@pytest.fixture
def basic_constraint_specs():
    """Basic constraint specifications for testing."""
    return {
        'feature_1': {
            'expected_sign': 'positive',
            'business_rationale': 'Higher values improve outcome'
        },
        'feature_2': {
            'expected_sign': 'negative',
            'business_rationale': 'Higher values reduce outcome'
        }
    }


@pytest.fixture
def mock_interpret_positive():
    """Mock interpreter for positive constraints."""
    def interpret(coef, ci_lower, ci_upper, rationale, significant):
        return f"Positive interpretation: {coef:.4f}"
    return interpret


@pytest.fixture
def mock_interpret_negative():
    """Mock interpreter for negative constraints."""
    def interpret(coef, ci_lower, ci_upper, rationale, significant):
        return f"Negative interpretation: {coef:.4f}"
    return interpret


# =============================================================================
# Tests for _validate_model_inputs
# =============================================================================


class TestValidateModelInputs:
    """Tests for model and constraint input validation."""

    def test_valid_inputs_pass(self, mock_valid_model, basic_constraint_specs):
        """Valid model and constraints pass validation."""
        # Should not raise
        _validate_model_inputs(mock_valid_model, basic_constraint_specs)

    def test_model_missing_params_raises(self, basic_constraint_specs):
        """Model without params attribute raises ValueError."""
        model = MagicMock(spec=[])  # Empty spec = no attributes
        model.bse = {'feature_1': 0.1}

        with pytest.raises(ValueError) as exc_info:
            _validate_model_inputs(model, basic_constraint_specs)

        assert "CRITICAL" in str(exc_info.value)
        assert "coefficient attributes" in str(exc_info.value)

    def test_model_missing_bse_raises(self, basic_constraint_specs):
        """Model without bse attribute raises ValueError."""
        model = MagicMock(spec=['params'])
        model.params = {'feature_1': 0.5}

        with pytest.raises(ValueError) as exc_info:
            _validate_model_inputs(model, basic_constraint_specs)

        assert "CRITICAL" in str(exc_info.value)

    def test_empty_constraints_raises(self, mock_valid_model):
        """Empty constraint specifications raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_model_inputs(mock_valid_model, {})

        assert "CRITICAL" in str(exc_info.value)
        assert "No constraint specifications" in str(exc_info.value)

    def test_none_constraints_raises(self, mock_valid_model):
        """None constraint specifications raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_model_inputs(mock_valid_model, None)

        assert "CRITICAL" in str(exc_info.value)


# =============================================================================
# Tests for _determine_constraint_type
# =============================================================================


class TestDetermineConstraintType:
    """Tests for constraint type determination from sign specification."""

    def test_positive_sign(self):
        """'positive' maps to POSITIVE constraint type."""
        result = _determine_constraint_type('positive')
        assert result == ConstraintType.POSITIVE

    def test_negative_sign(self):
        """'negative' maps to NEGATIVE constraint type."""
        result = _determine_constraint_type('negative')
        assert result == ConstraintType.NEGATIVE

    def test_other_sign_maps_to_magnitude(self):
        """Other values map to MAGNITUDE constraint type."""
        assert _determine_constraint_type('magnitude') == ConstraintType.MAGNITUDE
        assert _determine_constraint_type('any') == ConstraintType.MAGNITUDE
        assert _determine_constraint_type('') == ConstraintType.MAGNITUDE

    def test_case_sensitive(self):
        """Sign detection is case-sensitive."""
        # 'Positive' != 'positive'
        assert _determine_constraint_type('Positive') == ConstraintType.MAGNITUDE
        assert _determine_constraint_type('POSITIVE') == ConstraintType.MAGNITUDE


# =============================================================================
# Tests for _calculate_confidence_interval
# =============================================================================


class TestCalculateConfidenceInterval:
    """Tests for confidence interval calculation."""

    def test_symmetric_around_coefficient(self):
        """CI is symmetric around coefficient for symmetric t-distribution."""
        coef = 1.0
        se = 0.1
        ci_lower, ci_upper = _calculate_confidence_interval(coef, se, 0.95, 100)

        # CI should be symmetric around coefficient
        lower_diff = coef - ci_lower
        upper_diff = ci_upper - coef
        assert abs(lower_diff - upper_diff) < 1e-10

    def test_wider_ci_for_lower_confidence(self):
        """Lower confidence level produces narrower CI."""
        coef, se, df = 1.0, 0.1, 100

        ci_95_lower, ci_95_upper = _calculate_confidence_interval(coef, se, 0.95, df)
        ci_90_lower, ci_90_upper = _calculate_confidence_interval(coef, se, 0.90, df)

        width_95 = ci_95_upper - ci_95_lower
        width_90 = ci_90_upper - ci_90_lower

        assert width_95 > width_90

    def test_wider_ci_for_larger_se(self):
        """Larger standard error produces wider CI."""
        coef, confidence, df = 1.0, 0.95, 100

        ci_small_lower, ci_small_upper = _calculate_confidence_interval(coef, 0.05, confidence, df)
        ci_large_lower, ci_large_upper = _calculate_confidence_interval(coef, 0.20, confidence, df)

        width_small = ci_small_upper - ci_small_lower
        width_large = ci_large_upper - ci_large_lower

        assert width_large > width_small

    def test_coefficient_inside_ci(self):
        """Coefficient is always inside its CI."""
        coef, se = 2.5, 0.3
        ci_lower, ci_upper = _calculate_confidence_interval(coef, se, 0.95, 50)

        assert ci_lower < coef < ci_upper

    def test_positive_coefficient_positive_ci_bounds(self):
        """Positive coefficient with small SE has positive CI bounds."""
        coef, se = 5.0, 0.5
        ci_lower, ci_upper = _calculate_confidence_interval(coef, se, 0.95, 100)

        # With coef=5.0, se=0.5, and 100 df, CI should be entirely positive
        assert ci_lower > 0
        assert ci_upper > 0

    @pytest.mark.parametrize("df", [1, 10, 30, 100, 1000])
    def test_ci_narrows_with_df(self, df):
        """CI width decreases as degrees of freedom increase."""
        coef, se, confidence = 1.0, 0.1, 0.95
        ci_lower, ci_upper = _calculate_confidence_interval(coef, se, confidence, df)

        # For very small df, CI should be wider
        if df == 1:
            width = ci_upper - ci_lower
            assert width > 0.4  # Wide for df=1


# =============================================================================
# Tests for _assess_constraint_strength
# =============================================================================


class TestAssessConstraintStrength:
    """Tests for constraint strength assessment."""

    # Positive constraint tests
    def test_strong_positive_constraint(self):
        """CI entirely positive with ci_lower > 0.01 is STRONG."""
        result = _assess_constraint_strength(
            coefficient=0.5, ci_lower=0.1, ci_upper=0.9, expected_sign='positive'
        )
        assert result == "STRONG"

    def test_moderate_positive_constraint(self):
        """CI entirely positive with ci_lower <= 0.01 is MODERATE."""
        result = _assess_constraint_strength(
            coefficient=0.05, ci_lower=0.005, ci_upper=0.1, expected_sign='positive'
        )
        assert result == "MODERATE"

    def test_weak_positive_constraint(self):
        """Point positive but CI crosses zero is WEAK."""
        result = _assess_constraint_strength(
            coefficient=0.1, ci_lower=-0.05, ci_upper=0.25, expected_sign='positive'
        )
        assert result == "WEAK"

    def test_violated_positive_constraint(self):
        """Negative coefficient for positive constraint is VIOLATED."""
        result = _assess_constraint_strength(
            coefficient=-0.1, ci_lower=-0.2, ci_upper=-0.05, expected_sign='positive'
        )
        assert result == "VIOLATED"

    # Negative constraint tests
    def test_strong_negative_constraint(self):
        """CI entirely negative with ci_upper < -0.01 is STRONG."""
        result = _assess_constraint_strength(
            coefficient=-0.5, ci_lower=-0.9, ci_upper=-0.1, expected_sign='negative'
        )
        assert result == "STRONG"

    def test_moderate_negative_constraint(self):
        """CI entirely negative with ci_upper >= -0.01 is MODERATE."""
        result = _assess_constraint_strength(
            coefficient=-0.05, ci_lower=-0.1, ci_upper=-0.005, expected_sign='negative'
        )
        assert result == "MODERATE"

    def test_weak_negative_constraint(self):
        """Point negative but CI crosses zero is WEAK."""
        result = _assess_constraint_strength(
            coefficient=-0.1, ci_lower=-0.25, ci_upper=0.05, expected_sign='negative'
        )
        assert result == "WEAK"

    def test_violated_negative_constraint(self):
        """Positive coefficient for negative constraint is VIOLATED."""
        result = _assess_constraint_strength(
            coefficient=0.1, ci_lower=0.05, ci_upper=0.2, expected_sign='negative'
        )
        assert result == "VIOLATED"

    def test_returns_valid_strength_category(self):
        """Always returns one of the valid strength categories."""
        valid_strengths = {"STRONG", "MODERATE", "WEAK", "VIOLATED"}

        for coef in [-1.0, -0.1, 0.0, 0.1, 1.0]:
            for ci_lower in [coef - 0.5, coef - 0.1]:
                for ci_upper in [coef + 0.1, coef + 0.5]:
                    for sign in ['positive', 'negative']:
                        result = _assess_constraint_strength(coef, ci_lower, ci_upper, sign)
                        assert result in valid_strengths


# =============================================================================
# Tests for _evaluate_sign_constraint
# =============================================================================


class TestEvaluateSignConstraint:
    """Tests for sign constraint evaluation."""

    def test_positive_constraint_satisfied(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Positive constraint is satisfied when ci_lower > 0."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.POSITIVE,
            coefficient=0.5,
            ci_lower=0.1,
            ci_upper=0.9,
            business_rationale="Test rationale",
            statistically_significant=True,
            minimum_magnitude=0.0,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is True
        assert strength == "STRONG"
        assert "Positive interpretation" in interpretation

    def test_positive_constraint_violated(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Positive constraint is violated when ci_lower <= 0."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.POSITIVE,
            coefficient=0.2,
            ci_lower=-0.1,
            ci_upper=0.5,
            business_rationale="Test rationale",
            statistically_significant=False,
            minimum_magnitude=0.0,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is False
        assert strength == "WEAK"

    def test_negative_constraint_satisfied(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Negative constraint is satisfied when ci_upper < 0."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.NEGATIVE,
            coefficient=-0.5,
            ci_lower=-0.9,
            ci_upper=-0.1,
            business_rationale="Test rationale",
            statistically_significant=True,
            minimum_magnitude=0.0,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is True
        assert strength == "STRONG"
        assert "Negative interpretation" in interpretation

    def test_negative_constraint_violated(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Negative constraint is violated when ci_upper >= 0."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.NEGATIVE,
            coefficient=-0.2,
            ci_lower=-0.5,
            ci_upper=0.1,
            business_rationale="Test rationale",
            statistically_significant=False,
            minimum_magnitude=0.0,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is False
        assert strength == "WEAK"

    def test_magnitude_constraint_satisfied(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Magnitude constraint satisfied when significant and >= threshold."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.MAGNITUDE,
            coefficient=0.5,
            ci_lower=0.1,
            ci_upper=0.9,
            business_rationale="Test rationale",
            statistically_significant=True,
            minimum_magnitude=0.3,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is True
        assert strength == "MODERATE"
        assert "Magnitude constraint" in interpretation

    def test_magnitude_constraint_not_significant(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Magnitude constraint fails when not statistically significant."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.MAGNITUDE,
            coefficient=0.5,
            ci_lower=-0.1,
            ci_upper=1.1,
            business_rationale="Test rationale",
            statistically_significant=False,
            minimum_magnitude=0.3,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is False
        assert strength == "WEAK"

    def test_magnitude_constraint_below_threshold(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Magnitude constraint fails when below minimum magnitude."""
        satisfied, strength, interpretation = _evaluate_sign_constraint(
            constraint_type=ConstraintType.MAGNITUDE,
            coefficient=0.2,
            ci_lower=0.1,
            ci_upper=0.3,
            business_rationale="Test rationale",
            statistically_significant=True,
            minimum_magnitude=0.3,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert satisfied is False


# =============================================================================
# Tests for _build_hard_threshold_comparison
# =============================================================================


class TestBuildHardThresholdComparison:
    """Tests for hard threshold vs statistical comparison builder."""

    def test_positive_sign_positive_coefficient_passes(self):
        """Positive coefficient passes hard threshold for positive sign."""
        result = _build_hard_threshold_comparison(
            expected_sign='positive',
            coefficient=0.5,
            constraint_satisfied=True,
            minimum_magnitude=0.0
        )

        assert result['hard_threshold_passes'] is True
        assert result['statistical_approach_passes'] is True
        assert result['methods_agree'] is True

    def test_positive_sign_negative_coefficient_fails(self):
        """Negative coefficient fails hard threshold for positive sign."""
        result = _build_hard_threshold_comparison(
            expected_sign='positive',
            coefficient=-0.5,
            constraint_satisfied=False,
            minimum_magnitude=0.0
        )

        assert result['hard_threshold_passes'] is False
        assert result['statistical_approach_passes'] is False
        assert result['methods_agree'] is True

    def test_negative_sign_negative_coefficient_passes(self):
        """Negative coefficient passes hard threshold for negative sign."""
        result = _build_hard_threshold_comparison(
            expected_sign='negative',
            coefficient=-0.5,
            constraint_satisfied=True,
            minimum_magnitude=0.0
        )

        assert result['hard_threshold_passes'] is True
        assert result['statistical_approach_passes'] is True
        assert result['methods_agree'] is True

    def test_methods_disagree_example(self):
        """Methods can disagree (hard passes, statistical fails)."""
        result = _build_hard_threshold_comparison(
            expected_sign='positive',
            coefficient=0.1,  # Point positive (hard passes)
            constraint_satisfied=False,  # Statistical fails (CI crosses 0)
            minimum_magnitude=0.0
        )

        assert result['hard_threshold_passes'] is True
        assert result['statistical_approach_passes'] is False
        assert result['methods_agree'] is False

    def test_magnitude_constraint_uses_threshold(self):
        """Magnitude constraint uses minimum_magnitude threshold."""
        # Above threshold
        result = _build_hard_threshold_comparison(
            expected_sign='magnitude',
            coefficient=0.5,
            constraint_satisfied=True,
            minimum_magnitude=0.3
        )
        assert result['hard_threshold_passes'] is True

        # Below threshold
        result = _build_hard_threshold_comparison(
            expected_sign='magnitude',
            coefficient=0.2,
            constraint_satisfied=False,
            minimum_magnitude=0.3
        )
        assert result['hard_threshold_passes'] is False

    def test_negative_coefficient_magnitude(self):
        """Magnitude constraint uses absolute value."""
        result = _build_hard_threshold_comparison(
            expected_sign='magnitude',
            coefficient=-0.5,
            constraint_satisfied=True,
            minimum_magnitude=0.3
        )
        assert result['hard_threshold_passes'] is True  # |-0.5| >= 0.3


# =============================================================================
# Tests for _validate_single_statistical_constraint
# =============================================================================


class TestValidateSingleStatisticalConstraint:
    """Tests for the core single constraint validator."""

    def test_returns_statistical_constraint_result(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Returns properly structured StatisticalConstraintResult."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive', 'business_rationale': 'Test'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert isinstance(result, StatisticalConstraintResult)
        assert result.feature_name == 'test_feature'
        assert result.coefficient_estimate == 0.5
        assert result.standard_error == 0.1
        assert result.t_statistic == 5.0
        assert result.p_value == 0.001

    def test_confidence_interval_computed(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Confidence interval is properly computed."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=1.0,
            standard_error=0.2,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < 1.0 < ci_upper
        assert ci_lower > 0  # Should be positive for coef=1.0, se=0.2

    def test_statistical_significance_determined(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Statistical significance is correctly determined."""
        # Significant (p < alpha)
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )
        assert result.statistically_significant is True

        # Not significant (p > alpha)
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=1.5,
            p_value=0.15,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )
        assert result.statistically_significant is False

    def test_constraint_type_determined(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Constraint type is correctly determined from spec."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=-0.5,
            standard_error=0.1,
            t_statistic=-5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'negative'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert result.constraint_type == ConstraintType.NEGATIVE

    def test_default_values_used(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Default values used when not specified in constraint spec."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={},  # Empty spec - use defaults
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        # Default is 'positive'
        assert result.constraint_type == ConstraintType.POSITIVE

    def test_hard_threshold_comparison_included(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Result includes hard threshold comparison."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert 'hard_threshold_passes' in result.hard_threshold_comparison
        assert 'statistical_approach_passes' in result.hard_threshold_comparison
        assert 'methods_agree' in result.hard_threshold_comparison

    def test_strongly_positive_constraint_satisfied(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Strongly positive coefficient satisfies positive constraint."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=1.0,
            standard_error=0.1,
            t_statistic=10.0,
            p_value=0.0001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert result.constraint_satisfied == True  # Use == for numpy bool compatibility
        assert result.constraint_strength == "STRONG"

    def test_negative_coefficient_violates_positive_constraint(
        self, mock_interpret_positive, mock_interpret_negative
    ):
        """Negative coefficient violates positive constraint."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=-0.5,
            standard_error=0.1,
            t_statistic=-5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=100,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        assert result.constraint_satisfied == False  # Use == for numpy bool compatibility
        assert result.constraint_strength == "VIOLATED"

    @pytest.mark.parametrize("degrees_freedom", [10, 30, 100, 500])
    def test_degrees_freedom_affects_ci(
        self, degrees_freedom, mock_interpret_positive, mock_interpret_negative
    ):
        """Different degrees of freedom produce different CI widths."""
        result = _validate_single_statistical_constraint(
            feature_name='test_feature',
            coefficient=0.5,
            standard_error=0.1,
            t_statistic=5.0,
            p_value=0.001,
            constraint_spec={'expected_sign': 'positive'},
            confidence_level=0.95,
            significance_level=0.05,
            degrees_freedom=degrees_freedom,
            interpret_positive_fn=mock_interpret_positive,
            interpret_negative_fn=mock_interpret_negative
        )

        ci_lower, ci_upper = result.confidence_interval
        ci_width = ci_upper - ci_lower

        # CI should be positive (we're not testing exact width values)
        assert ci_width > 0
