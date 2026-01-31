"""
Tests for comparative_analysis module.

Target: 39% → 60%+ coverage
Tests organized by function categories:
- Input validation
- Default comparison metrics
- All comparisons orchestration
- Methodology comparison building
- Main public API
"""

import pytest
from unittest.mock import patch, MagicMock

from src.features.selection.comparison.comparative_analysis import (
    _validate_comparison_inputs,
    _get_default_comparison_metrics,
    _run_all_comparisons,
    _build_methodology_comparison,
    compare_methodologies,
    MethodologyComparison,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_baseline_results():
    """Sample baseline methodology results."""
    return {
        'selected_model': {
            'features': 'feature_a + feature_b',
            'aic': 250.5,
            'r_squared': 0.75,
        },
        'performance_metrics': {
            'r_squared': 0.75,
            'mape': 12.5,
            'generalization_gap': 0.05,
        },
        'diagnostic_results': {
            'autocorrelation': 'passed',
            'heteroscedasticity': 'passed',
        },
    }


@pytest.fixture
def sample_enhanced_results():
    """Sample enhanced methodology results."""
    return {
        'selected_model': {
            'features': 'feature_a + feature_b + feature_c',
            'aic': 245.0,
            'r_squared': 0.82,
        },
        'performance_metrics': {
            'r_squared': 0.82,
            'mape': 10.2,
            'generalization_gap': 0.03,
        },
        'diagnostic_results': {
            'autocorrelation': 'passed',
            'heteroscedasticity': 'passed',
            'multicollinearity': 'passed',
        },
        'statistical_validation': {
            'bootstrap_stability': 0.95,
            'information_ratio': 0.52,
        },
    }


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateComparisonInputs:
    """Tests for _validate_comparison_inputs."""

    def test_raises_on_none_baseline(self, sample_enhanced_results):
        """Raises ValueError when baseline is None."""
        with pytest.raises(ValueError, match="baseline_results cannot be None"):
            _validate_comparison_inputs(None, sample_enhanced_results)

    def test_raises_on_none_enhanced(self, sample_baseline_results):
        """Raises ValueError when enhanced is None."""
        with pytest.raises(ValueError, match="enhanced_results cannot be None"):
            _validate_comparison_inputs(sample_baseline_results, None)

    def test_raises_on_non_dict_baseline(self, sample_enhanced_results):
        """Raises ValueError when baseline is not dict."""
        with pytest.raises(ValueError, match="baseline_results must be a dictionary"):
            _validate_comparison_inputs("not a dict", sample_enhanced_results)

    def test_raises_on_non_dict_enhanced(self, sample_baseline_results):
        """Raises ValueError when enhanced is not dict."""
        with pytest.raises(ValueError, match="enhanced_results must be a dictionary"):
            _validate_comparison_inputs(sample_baseline_results, "not a dict")

    def test_accepts_valid_inputs(self, sample_baseline_results, sample_enhanced_results):
        """Accepts valid dict inputs without error."""
        # Should not raise
        _validate_comparison_inputs(sample_baseline_results, sample_enhanced_results)

    def test_accepts_empty_dicts(self):
        """Accepts empty dicts (validation passes)."""
        # Should not raise - empty dicts are still dicts
        _validate_comparison_inputs({}, {})


# =============================================================================
# Default Metrics Tests
# =============================================================================


class TestGetDefaultComparisonMetrics:
    """Tests for _get_default_comparison_metrics."""

    def test_returns_list(self):
        """Returns a list."""
        result = _get_default_comparison_metrics()

        assert isinstance(result, list)

    def test_contains_model_selection_consistency(self):
        """Contains model_selection_consistency metric."""
        result = _get_default_comparison_metrics()

        assert 'model_selection_consistency' in result

    def test_contains_performance_metrics(self):
        """Contains performance_metrics."""
        result = _get_default_comparison_metrics()

        assert 'performance_metrics' in result

    def test_contains_statistical_validation(self):
        """Contains statistical_validation."""
        result = _get_default_comparison_metrics()

        assert 'statistical_validation' in result

    def test_contains_production_readiness(self):
        """Contains production_readiness."""
        result = _get_default_comparison_metrics()

        assert 'production_readiness' in result

    def test_contains_business_impact(self):
        """Contains business_impact."""
        result = _get_default_comparison_metrics()

        assert 'business_impact' in result

    def test_has_five_metrics(self):
        """Returns exactly 5 default metrics."""
        result = _get_default_comparison_metrics()

        assert len(result) == 5


# =============================================================================
# All Comparisons Orchestration Tests
# =============================================================================


class TestRunAllComparisons:
    """Tests for _run_all_comparisons."""

    def test_returns_tuple_of_five(self, sample_baseline_results, sample_enhanced_results):
        """Returns tuple of 5 comparison results."""
        result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_calls_performance_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Calls _compare_performance_metrics."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._compare_performance_metrics',
            return_value={'test': 'performance'}
        ) as mock_perf:
            result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

            mock_perf.assert_called_once()
            assert result[0] == {'test': 'performance'}

    def test_calls_model_selection_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Calls _compare_model_selection."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._compare_model_selection',
            return_value={'test': 'selection'}
        ) as mock_sel:
            result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

            mock_sel.assert_called_once()
            assert result[1] == {'test': 'selection'}

    def test_calls_statistical_validation_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Calls _compare_statistical_validation."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._compare_statistical_validation',
            return_value={'test': 'validation'}
        ) as mock_val:
            result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

            mock_val.assert_called_once()
            assert result[2] == {'test': 'validation'}

    def test_calls_production_readiness_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Calls _compare_production_readiness."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._compare_production_readiness',
            return_value={'test': 'readiness'}
        ) as mock_ready:
            result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

            mock_ready.assert_called_once()
            assert result[3] == {'test': 'readiness'}

    def test_calls_business_impact_analysis(self, sample_baseline_results, sample_enhanced_results):
        """Calls _analyze_business_impact."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._analyze_business_impact',
            return_value={'test': 'business'}
        ) as mock_biz:
            result = _run_all_comparisons(sample_baseline_results, sample_enhanced_results)

            mock_biz.assert_called_once()
            assert result[4] == {'test': 'business'}


# =============================================================================
# Build Methodology Comparison Tests
# =============================================================================


class TestBuildMethodologyComparison:
    """Tests for _build_methodology_comparison."""

    def test_returns_methodology_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Returns MethodologyComparison dataclass."""
        comparisons = (
            {'perf': 'data'},
            {'selection': 'data'},
            {'validation': 'data'},
            {'readiness': 'data'},
            {'business': 'data'},
        )
        recommendations = {'primary': 'Use enhanced'}

        result = _build_methodology_comparison(
            sample_baseline_results, sample_enhanced_results,
            comparisons, recommendations
        )

        assert isinstance(result, MethodologyComparison)

    def test_assigns_baseline_results(self, sample_baseline_results, sample_enhanced_results):
        """Assigns baseline_results correctly."""
        comparisons = ({}, {}, {}, {}, {})
        recommendations = {}

        result = _build_methodology_comparison(
            sample_baseline_results, sample_enhanced_results,
            comparisons, recommendations
        )

        assert result.baseline_results == sample_baseline_results

    def test_assigns_enhanced_results(self, sample_baseline_results, sample_enhanced_results):
        """Assigns enhanced_results correctly."""
        comparisons = ({}, {}, {}, {}, {})
        recommendations = {}

        result = _build_methodology_comparison(
            sample_baseline_results, sample_enhanced_results,
            comparisons, recommendations
        )

        assert result.enhanced_results == sample_enhanced_results

    def test_assigns_performance_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Assigns performance_comparison correctly."""
        comparisons = ({'perf': 'test'}, {}, {}, {}, {})
        recommendations = {}

        result = _build_methodology_comparison(
            sample_baseline_results, sample_enhanced_results,
            comparisons, recommendations
        )

        assert result.performance_comparison == {'perf': 'test'}

    def test_assigns_recommendations(self, sample_baseline_results, sample_enhanced_results):
        """Assigns recommendations correctly."""
        comparisons = ({}, {}, {}, {}, {})
        recommendations = {'primary': 'Adopt enhanced', 'risk': 'Low'}

        result = _build_methodology_comparison(
            sample_baseline_results, sample_enhanced_results,
            comparisons, recommendations
        )

        assert result.recommendations == recommendations


# =============================================================================
# Main Public API Tests
# =============================================================================


class TestCompareMethodologies:
    """Tests for compare_methodologies."""

    def test_returns_methodology_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Returns MethodologyComparison."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert isinstance(result, MethodologyComparison)

    def test_validates_inputs(self, sample_enhanced_results):
        """Validates inputs before proceeding."""
        with pytest.raises(ValueError, match="baseline_results cannot be None"):
            compare_methodologies(None, sample_enhanced_results)

    def test_contains_baseline_results(self, sample_baseline_results, sample_enhanced_results):
        """Result contains baseline_results."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.baseline_results == sample_baseline_results

    def test_contains_enhanced_results(self, sample_baseline_results, sample_enhanced_results):
        """Result contains enhanced_results."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.enhanced_results == sample_enhanced_results

    def test_contains_performance_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Result contains performance_comparison."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.performance_comparison is not None

    def test_contains_model_selection_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Result contains model_selection_comparison."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.model_selection_comparison is not None

    def test_contains_statistical_validation_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Result contains statistical_validation_comparison."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.statistical_validation_comparison is not None

    def test_contains_production_readiness_comparison(self, sample_baseline_results, sample_enhanced_results):
        """Result contains production_readiness_comparison."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.production_readiness_comparison is not None

    def test_contains_business_impact_analysis(self, sample_baseline_results, sample_enhanced_results):
        """Result contains business_impact_analysis."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.business_impact_analysis is not None

    def test_contains_recommendations(self, sample_baseline_results, sample_enhanced_results):
        """Result contains recommendations."""
        result = compare_methodologies(sample_baseline_results, sample_enhanced_results)

        assert result.recommendations is not None

    def test_raises_on_comparison_failure(self, sample_baseline_results):
        """Raises ValueError on comparison failure."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._run_all_comparisons',
            side_effect=Exception("Comparison failed")
        ):
            with pytest.raises(ValueError, match="Methodology comparison failed"):
                compare_methodologies(sample_baseline_results, {})

    def test_error_includes_business_impact(self, sample_baseline_results):
        """Error message includes business impact context."""
        with patch(
            'src.features.selection.comparison.comparative_analysis._run_all_comparisons',
            side_effect=Exception("Test error")
        ):
            with pytest.raises(ValueError, match="Business impact"):
                compare_methodologies(sample_baseline_results, {})


# =============================================================================
# MethodologyComparison Dataclass Tests
# =============================================================================


class TestMethodologyComparisonDataclass:
    """Tests for MethodologyComparison dataclass."""

    def test_creates_instance(self):
        """Creates instance with all required fields."""
        result = MethodologyComparison(
            baseline_results={'a': 1},
            enhanced_results={'b': 2},
            performance_comparison={'c': 3},
            model_selection_comparison={'d': 4},
            statistical_validation_comparison={'e': 5},
            production_readiness_comparison={'f': 6},
            business_impact_analysis={'g': 7},
            recommendations={'h': 'i'},
        )

        assert result.baseline_results == {'a': 1}
        assert result.enhanced_results == {'b': 2}
        assert result.recommendations == {'h': 'i'}

    def test_all_fields_accessible(self):
        """All fields are accessible."""
        result = MethodologyComparison(
            baseline_results={},
            enhanced_results={},
            performance_comparison={},
            model_selection_comparison={},
            statistical_validation_comparison={},
            production_readiness_comparison={},
            business_impact_analysis={},
            recommendations={},
        )

        # All fields should be accessible without error
        _ = result.baseline_results
        _ = result.enhanced_results
        _ = result.performance_comparison
        _ = result.model_selection_comparison
        _ = result.statistical_validation_comparison
        _ = result.production_readiness_comparison
        _ = result.business_impact_analysis
        _ = result.recommendations


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_main_function_exported(self):
        """Main function is in __all__."""
        from src.features.selection.comparison import comparative_analysis
        assert 'compare_methodologies' in comparative_analysis.__all__

    def test_dataclass_exported(self):
        """MethodologyComparison is in __all__."""
        from src.features.selection.comparison import comparative_analysis
        assert 'MethodologyComparison' in comparative_analysis.__all__

    def test_validation_exported(self):
        """Validation function is in __all__."""
        from src.features.selection.comparison import comparative_analysis
        assert '_validate_comparison_inputs' in comparative_analysis.__all__
