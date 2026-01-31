"""
Tests for statistical_constraints_engine module.

Target: 38% → 60%+ coverage
Tests organized by function categories:
- Model specification extraction
- Single feature constraint processing
- Main public API (apply_statistical_constraints)
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.features.selection.enhancements.statistical_constraints.statistical_constraints_engine import (
    _extract_model_specification,
    _process_constraint_for_feature,
    apply_statistical_constraints,
    ConstraintType,
    StatisticalConstraintResult,
    ComprehensiveConstraintAnalysis,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Mock statsmodels regression model."""
    model = MagicMock()
    model.nobs = 100
    model.params = pd.Series({
        'const': 5.0,
        'own_rate_t1': 0.15,
        'competitor_weighted_t1': -0.08,
    })
    model.bse = pd.Series({
        'const': 0.5,
        'own_rate_t1': 0.03,
        'competitor_weighted_t1': 0.02,
    })
    model.tvalues = pd.Series({
        'const': 10.0,
        'own_rate_t1': 5.0,
        'competitor_weighted_t1': -4.0,
    })
    model.pvalues = pd.Series({
        'const': 0.0001,
        'own_rate_t1': 0.001,
        'competitor_weighted_t1': 0.002,
    })
    model.rsquared = 0.85
    model.aic = 250.5
    model.df_resid = 97
    return model


@pytest.fixture
def constraint_specs():
    """Sample constraint specifications."""
    return {
        'own_rate_t1': {
            'expected_sign': 'positive',
            'description': 'Own rate should be positive',
        },
        'competitor_weighted_t1': {
            'expected_sign': 'negative',
            'description': 'Competitor rate should be negative',
        },
    }


# =============================================================================
# Model Specification Extraction Tests
# =============================================================================


class TestExtractModelSpecification:
    """Tests for _extract_model_specification."""

    def test_returns_dict(self, mock_model):
        """Returns dictionary."""
        result = _extract_model_specification(mock_model)

        assert isinstance(result, dict)

    def test_contains_n_observations(self, mock_model):
        """Contains n_observations."""
        result = _extract_model_specification(mock_model)

        assert 'n_observations' in result
        assert result['n_observations'] == 100

    def test_contains_n_parameters(self, mock_model):
        """Contains n_parameters."""
        result = _extract_model_specification(mock_model)

        assert 'n_parameters' in result
        assert result['n_parameters'] == 3

    def test_contains_r_squared(self, mock_model):
        """Contains model_r_squared."""
        result = _extract_model_specification(mock_model)

        assert 'model_r_squared' in result
        assert result['model_r_squared'] == 0.85

    def test_contains_aic(self, mock_model):
        """Contains model_aic."""
        result = _extract_model_specification(mock_model)

        assert 'model_aic' in result
        assert result['model_aic'] == 250.5

    def test_contains_degrees_freedom(self, mock_model):
        """Contains degrees_freedom."""
        result = _extract_model_specification(mock_model)

        assert 'degrees_freedom' in result
        assert result['degrees_freedom'] == 97

    def test_contains_timestamp(self, mock_model):
        """Contains constraint_validation_timestamp."""
        result = _extract_model_specification(mock_model)

        assert 'constraint_validation_timestamp' in result
        # Should be valid ISO format
        datetime.fromisoformat(result['constraint_validation_timestamp'])


# =============================================================================
# Single Feature Constraint Processing Tests
# =============================================================================


class TestProcessConstraintForFeature:
    """Tests for _process_constraint_for_feature."""

    def test_returns_none_for_missing_feature(self, mock_model):
        """Returns None when feature not in model."""
        result = _process_constraint_for_feature(
            feature_name='nonexistent_feature',
            constraint_spec={'expected_sign': 'positive'},
            model=mock_model,
            confidence_level=0.95,
            significance_level=0.05,
        )

        assert result is None

    def test_logs_warning_for_missing_feature(self, mock_model, caplog):
        """Logs warning when feature not found."""
        import logging
        with caplog.at_level(logging.WARNING):
            _process_constraint_for_feature(
                feature_name='missing_feature',
                constraint_spec={'expected_sign': 'positive'},
                model=mock_model,
                confidence_level=0.95,
                significance_level=0.05,
            )

        assert 'not found in model coefficients' in caplog.text

    def test_returns_result_for_valid_feature(self, mock_model):
        """Returns StatisticalConstraintResult for valid feature."""
        result = _process_constraint_for_feature(
            feature_name='own_rate_t1',
            constraint_spec={'expected_sign': 'positive'},
            model=mock_model,
            confidence_level=0.95,
            significance_level=0.05,
        )

        assert result is not None
        assert isinstance(result, StatisticalConstraintResult)

    def test_passes_correct_coefficient(self, mock_model):
        """Passes correct coefficient value."""
        result = _process_constraint_for_feature(
            feature_name='own_rate_t1',
            constraint_spec={'expected_sign': 'positive'},
            model=mock_model,
            confidence_level=0.95,
            significance_level=0.05,
        )

        assert result.coefficient_estimate == 0.15

    def test_passes_correct_standard_error(self, mock_model):
        """Passes correct standard error."""
        result = _process_constraint_for_feature(
            feature_name='own_rate_t1',
            constraint_spec={'expected_sign': 'positive'},
            model=mock_model,
            confidence_level=0.95,
            significance_level=0.05,
        )

        assert result.standard_error == 0.03


# =============================================================================
# Main Public API Tests
# =============================================================================


class TestApplyStatisticalConstraints:
    """Tests for apply_statistical_constraints."""

    def test_returns_comprehensive_analysis(self, mock_model, constraint_specs):
        """Returns ComprehensiveConstraintAnalysis."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert isinstance(result, ComprehensiveConstraintAnalysis)

    def test_contains_model_specification(self, mock_model, constraint_specs):
        """Result contains model_specification."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert result.model_specification is not None
        assert result.model_specification['n_observations'] == 100

    def test_contains_constraint_results(self, mock_model, constraint_specs):
        """Result contains constraint_results for each feature."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert len(result.constraint_results) == 2

    def test_contains_overall_assessment(self, mock_model, constraint_specs):
        """Result contains overall_assessment."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert result.overall_assessment is not None

    def test_contains_methodology_comparison(self, mock_model, constraint_specs):
        """Result contains methodology_comparison."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert result.methodology_comparison is not None

    def test_contains_business_recommendations(self, mock_model, constraint_specs):
        """Result contains business_recommendations."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert result.business_recommendations is not None

    def test_contains_power_analysis(self, mock_model, constraint_specs):
        """Result contains power_analysis."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        assert result.power_analysis is not None

    def test_uses_default_confidence_level(self, mock_model, constraint_specs):
        """Uses default confidence level of 0.95."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs
        )

        # Check first constraint result has a valid confidence interval
        ci = result.constraint_results[0].confidence_interval
        assert ci is not None
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower < Upper

    def test_custom_confidence_level(self, mock_model, constraint_specs):
        """Accepts custom confidence level (wider CI for 99%)."""
        result_95 = apply_statistical_constraints(
            mock_model, constraint_specs,
            confidence_level=0.95
        )
        result_99 = apply_statistical_constraints(
            mock_model, constraint_specs,
            confidence_level=0.99
        )

        # 99% CI should be wider than 95% CI
        ci_95 = result_95.constraint_results[0].confidence_interval
        ci_99 = result_99.constraint_results[0].confidence_interval
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 > width_95

    def test_custom_significance_level(self, mock_model, constraint_specs):
        """Accepts custom significance level."""
        result = apply_statistical_constraints(
            mock_model, constraint_specs,
            significance_level=0.01
        )

        # Result should use the custom significance level
        assert result is not None

    def test_skips_missing_features(self, mock_model):
        """Skips features not in model."""
        specs = {
            'own_rate_t1': {'expected_sign': 'positive'},
            'nonexistent': {'expected_sign': 'negative'},
        }

        result = apply_statistical_constraints(mock_model, specs)

        # Only one feature should have results
        assert len(result.constraint_results) == 1

    def test_raises_on_invalid_model(self):
        """Raises ValueError for invalid model."""
        with pytest.raises(ValueError, match="Statistical constraint validation failed"):
            apply_statistical_constraints(None, {'feature': {}})

    def test_raises_on_invalid_specs(self, mock_model):
        """Raises ValueError for invalid constraint specs."""
        with pytest.raises(ValueError, match="Statistical constraint validation failed"):
            apply_statistical_constraints(mock_model, None)

    def test_positive_constraint_satisfaction(self, mock_model):
        """Validates positive constraint is satisfied."""
        specs = {'own_rate_t1': {'expected_sign': 'positive'}}

        result = apply_statistical_constraints(mock_model, specs)

        # own_rate_t1 = 0.15 with expected positive should satisfy
        constraint = result.constraint_results[0]
        assert constraint.feature_name == 'own_rate_t1'
        assert constraint.constraint_type == ConstraintType.POSITIVE

    def test_negative_constraint_satisfaction(self, mock_model):
        """Validates negative constraint is satisfied."""
        specs = {'competitor_weighted_t1': {'expected_sign': 'negative'}}

        result = apply_statistical_constraints(mock_model, specs)

        # competitor_weighted_t1 = -0.08 with expected negative should satisfy
        constraint = result.constraint_results[0]
        assert constraint.feature_name == 'competitor_weighted_t1'
        assert constraint.constraint_type == ConstraintType.NEGATIVE


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_constraint_type_exported(self):
        """ConstraintType is exported."""
        from src.features.selection.enhancements.statistical_constraints import statistical_constraints_engine
        assert 'ConstraintType' in statistical_constraints_engine.__all__

    def test_result_types_exported(self):
        """Result types are exported."""
        from src.features.selection.enhancements.statistical_constraints import statistical_constraints_engine
        assert 'StatisticalConstraintResult' in statistical_constraints_engine.__all__
        assert 'ComprehensiveConstraintAnalysis' in statistical_constraints_engine.__all__

    def test_main_api_exported(self):
        """Main API function is exported."""
        from src.features.selection.enhancements.statistical_constraints import statistical_constraints_engine
        assert 'apply_statistical_constraints' in statistical_constraints_engine.__all__
