"""
Tests for src/features/selection/comparison/comparison_metrics.py

Comprehensive test coverage for methodology comparison metrics:
- Performance metric extraction (baseline and enhanced)
- Model selection comparison and consistency analysis
- Statistical validation comparison and rigor assessment
- Production readiness comparison and confidence scoring

Target: 0% → 60%+ coverage
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.features.selection.comparison.comparison_metrics import (
    # Performance extraction
    _extract_baseline_performance,
    _extract_enhanced_performance,
    _compute_metric_improvements,
    _identify_enhanced_exclusive_benefits,
    _compare_performance_metrics,
    # Model selection
    _extract_baseline_selection,
    _extract_enhanced_selection,
    _analyze_selection_consistency,
    _get_selection_criteria_comparison,
    _compare_model_selection,
    _calculate_feature_overlap,
    # Statistical validation
    _get_default_validation_comparison,
    _compare_statistical_validation,
    _extract_baseline_validation,
    _extract_enhanced_validation,
    _identify_validation_improvements,
    _assess_statistical_rigor,
    _calculate_statistical_rigor_score,
    # Production readiness
    _get_baseline_readiness,
    _extract_enhanced_readiness,
    _identify_readiness_improvements,
    _assess_deployment_confidence,
    _compare_production_readiness,
    _calculate_confidence_score,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def baseline_results() -> Dict[str, Any]:
    """Standard baseline results with valid_results_sorted."""
    return {
        'valid_results_sorted': pd.DataFrame({
            'aic': [500.0, 510.0, 520.0],
            'r_squared': [0.75, 0.72, 0.70],
            'features': ['feat1 + feat2', 'feat1 + feat3', 'feat2 + feat3'],
        }),
        'selected_model': {
            'aic': 500.0,
            'r_squared': 0.75,
            'features': 'feat1 + feat2',
            'n_features': 2,
        },
        'bootstrap_results': [{'stability': 'STABLE'}],
        'constraint_validation': True,
    }


@pytest.fixture
def enhanced_results() -> Dict[str, Any]:
    """Standard enhanced results with full validation."""
    return {
        'valid_results_sorted': pd.DataFrame({
            'aic': [490.0, 505.0, 515.0],
            'r_squared': [0.78, 0.74, 0.71],
            'features': ['feat1 + feat2', 'feat1 + feat3', 'feat2 + feat3'],
        }),
        'selected_model': {
            'aic': 490.0,
            'r_squared': 0.78,
            'features': 'feat1 + feat2',
            'n_features': 2,
        },
        'temporal_validation': {
            'train_performance': {'r_squared': 0.80, 'mse': 100.0},
            'test_performance': {'r_squared': 0.75, 'mse': 120.0},
            'generalization_gap': 0.05,
            'production_assessment': {
                'confidence_level': 'HIGH',
                'recommendations': ['Deploy with monitoring'],
            },
        },
        'multiple_testing_correction': {'method': 'bonferroni'},
        'block_bootstrap': {'n_samples': 1000},
        'regression_diagnostics': {'overall_assessment': 'PASSED'},
        'statistical_constraints': {'ci_based': True},
    }


@pytest.fixture
def empty_results() -> Dict[str, Any]:
    """Empty results for edge case testing."""
    return {}


@pytest.fixture
def minimal_baseline() -> Dict[str, Any]:
    """Minimal baseline with just selected_model."""
    return {
        'selected_model': {
            'features': 'feat1',
            'aic': 600.0,
            'n_features': 1,
        }
    }


# =============================================================================
# Tests for Performance Metric Extraction
# =============================================================================


class TestExtractBaselinePerformance:
    """Tests for _extract_baseline_performance."""

    def test_extracts_from_valid_results(self, baseline_results):
        """1.1: Extracts AIC and R² from valid_results_sorted."""
        perf = _extract_baseline_performance(baseline_results)

        assert perf['aic'] == 500.0
        assert perf['r_squared'] == 0.75

    def test_extracts_from_selected_model(self, minimal_baseline):
        """1.2: Falls back to selected_model when valid_results_sorted missing."""
        perf = _extract_baseline_performance(minimal_baseline)

        assert perf['aic'] == 600.0

    def test_empty_results_returns_empty_dict(self, empty_results):
        """1.3: Empty results returns empty performance dict."""
        perf = _extract_baseline_performance(empty_results)

        assert perf == {}

    def test_handles_empty_valid_results(self):
        """1.4: Handles empty valid_results_sorted DataFrame."""
        results = {'valid_results_sorted': pd.DataFrame()}
        perf = _extract_baseline_performance(results)

        assert 'aic' not in perf


class TestExtractEnhancedPerformance:
    """Tests for _extract_enhanced_performance."""

    def test_extracts_base_metrics(self, enhanced_results):
        """2.1: Extracts base AIC and R² metrics."""
        perf = _extract_enhanced_performance(enhanced_results)

        assert perf['aic'] == 490.0
        assert perf['r_squared'] == 0.78

    def test_extracts_temporal_validation_metrics(self, enhanced_results):
        """2.2: Extracts test performance metrics."""
        perf = _extract_enhanced_performance(enhanced_results)

        assert perf['test_r_squared'] == 0.75
        assert perf['test_mse'] == 120.0

    def test_calculates_generalization_gap(self, enhanced_results):
        """2.3: Calculates generalization gap from train/test."""
        perf = _extract_enhanced_performance(enhanced_results)

        # 0.80 - 0.75 = 0.05
        assert abs(perf['generalization_gap'] - 0.05) < 1e-10

    def test_empty_results_returns_empty_dict(self, empty_results):
        """2.4: Empty results returns empty dict."""
        perf = _extract_enhanced_performance(empty_results)

        assert perf == {}


class TestComputeMetricImprovements:
    """Tests for _compute_metric_improvements."""

    def test_computes_aic_improvement(self):
        """3.1: Computes AIC improvement (lower is better)."""
        baseline = {'aic': 500.0}
        enhanced = {'aic': 490.0}

        improvements = _compute_metric_improvements(baseline, enhanced)

        assert improvements['aic']['improved'] is True
        assert improvements['aic']['difference'] == -10.0

    def test_computes_r_squared_improvement(self):
        """3.2: Computes R² improvement (higher is better)."""
        baseline = {'r_squared': 0.70}
        enhanced = {'r_squared': 0.78}

        improvements = _compute_metric_improvements(baseline, enhanced)

        assert improvements['r_squared']['improved'] is True
        assert abs(improvements['r_squared']['difference'] - 0.08) < 1e-10

    def test_handles_no_improvement(self):
        """3.3: Correctly marks no improvement."""
        baseline = {'aic': 490.0, 'r_squared': 0.80}
        enhanced = {'aic': 500.0, 'r_squared': 0.70}

        improvements = _compute_metric_improvements(baseline, enhanced)

        assert improvements['aic']['improved'] is False
        assert improvements['r_squared']['improved'] is False

    def test_handles_nan_values(self):
        """3.4: Skips NaN values."""
        baseline = {'aic': np.nan}
        enhanced = {'aic': 500.0}

        improvements = _compute_metric_improvements(baseline, enhanced)

        assert 'aic' not in improvements


class TestIdentifyEnhancedExclusiveBenefits:
    """Tests for _identify_enhanced_exclusive_benefits."""

    def test_identifies_all_benefits(self, enhanced_results):
        """4.1: Identifies all enhanced-only benefits."""
        benefits = _identify_enhanced_exclusive_benefits(enhanced_results)

        assert benefits['out_of_sample_validation'] is True
        assert benefits['multiple_testing_correction'] is True
        assert benefits['block_bootstrap'] is True
        assert benefits['regression_diagnostics'] is True
        assert benefits['statistical_constraints'] is True

    def test_empty_results_no_benefits(self, empty_results):
        """4.2: Empty results shows no benefits."""
        benefits = _identify_enhanced_exclusive_benefits(empty_results)

        assert benefits['out_of_sample_validation'] is False
        assert benefits['multiple_testing_correction'] is False


class TestComparePerformanceMetrics:
    """Tests for _compare_performance_metrics orchestration."""

    def test_returns_full_structure(self, baseline_results, enhanced_results):
        """5.1: Returns complete comparison structure."""
        comparison = _compare_performance_metrics(baseline_results, enhanced_results)

        assert 'baseline_performance' in comparison
        assert 'enhanced_performance' in comparison
        assert 'improvements' in comparison
        assert 'enhanced_exclusive_benefits' in comparison

    def test_handles_exception_gracefully(self):
        """5.2: Handles exceptions gracefully."""
        # Create results that will cause issues
        comparison = _compare_performance_metrics(None, None)

        assert comparison.get('comparison_failed', False) is True


# =============================================================================
# Tests for Model Selection Comparison
# =============================================================================


class TestExtractBaselineSelection:
    """Tests for _extract_baseline_selection."""

    def test_extracts_features_and_selection(self, baseline_results):
        """6.1: Extracts features and selection details."""
        features, selection = _extract_baseline_selection(baseline_results)

        assert features == 'feat1 + feat2'
        assert selection['aic'] == 500.0
        assert selection['n_features'] == 2

    def test_default_values_when_missing(self, empty_results):
        """6.2: Returns defaults when selected_model missing."""
        features, selection = _extract_baseline_selection(empty_results)

        assert features == 'Unknown'
        assert selection['method'] == 'AIC minimization'


class TestExtractEnhancedSelection:
    """Tests for _extract_enhanced_selection."""

    def test_extracts_features_and_selection(self, enhanced_results):
        """7.1: Extracts features and selection details."""
        features, selection = _extract_enhanced_selection(enhanced_results)

        assert features == 'feat1 + feat2'
        assert selection['aic'] == 490.0
        assert selection['validation_method'] == 'Temporal train/test split'


class TestAnalyzeSelectionConsistency:
    """Tests for _analyze_selection_consistency."""

    def test_high_consistency(self):
        """8.1: Identifies HIGH consistency for identical features."""
        consistency = _analyze_selection_consistency('feat1 + feat2', 'feat1 + feat2')

        assert consistency['consistency_level'] == 'HIGH'
        assert consistency['same_model_selected'] is True
        assert consistency['feature_overlap'] == 1.0

    def test_moderate_consistency(self):
        """8.2: Identifies MODERATE consistency for partial overlap."""
        # feat1+feat2 vs feat1+feat2+feat3 has better overlap
        consistency = _analyze_selection_consistency('feat1 + feat2 + feat3', 'feat1 + feat2 + feat4')

        assert consistency['consistency_level'] == 'MODERATE'
        assert 0.5 <= consistency['feature_overlap'] < 0.8

    def test_low_consistency(self):
        """8.3: Identifies LOW consistency for no overlap."""
        consistency = _analyze_selection_consistency('feat1 + feat2', 'feat3 + feat4')

        assert consistency['consistency_level'] == 'LOW'
        assert consistency['feature_overlap'] == 0.0


class TestCalculateFeatureOverlap:
    """Tests for _calculate_feature_overlap."""

    def test_identical_features(self):
        """9.1: Returns 1.0 for identical features."""
        overlap = _calculate_feature_overlap('a + b + c', 'a + b + c')
        assert overlap == 1.0

    def test_no_overlap(self):
        """9.2: Returns 0.0 for no overlap."""
        overlap = _calculate_feature_overlap('a + b', 'c + d')
        assert overlap == 0.0

    def test_partial_overlap(self):
        """9.3: Returns correct Jaccard similarity."""
        # {a, b} and {a, c} → intersection=1, union=3 → 0.333...
        overlap = _calculate_feature_overlap('a + b', 'a + c')
        assert abs(overlap - 1/3) < 0.01

    def test_handles_non_string_input(self):
        """9.4: Returns 0.0 for invalid input."""
        overlap = _calculate_feature_overlap(None, 'a + b')
        assert overlap == 0.0


class TestGetSelectionCriteriaComparison:
    """Tests for _get_selection_criteria_comparison."""

    def test_returns_comparison_structure(self):
        """10.1: Returns expected structure."""
        comparison = _get_selection_criteria_comparison()

        assert 'baseline_criteria' in comparison
        assert 'enhanced_criteria' in comparison
        assert 'enhanced_advantages' in comparison
        assert len(comparison['enhanced_advantages']) > 0


class TestCompareModelSelection:
    """Tests for _compare_model_selection orchestration."""

    def test_returns_full_structure(self, baseline_results, enhanced_results):
        """11.1: Returns complete comparison structure."""
        comparison = _compare_model_selection(baseline_results, enhanced_results)

        assert 'baseline_selection' in comparison
        assert 'enhanced_selection' in comparison
        assert 'consistency_analysis' in comparison
        assert 'selection_criteria_comparison' in comparison


# =============================================================================
# Tests for Statistical Validation Comparison
# =============================================================================


class TestGetDefaultValidationComparison:
    """Tests for _get_default_validation_comparison."""

    def test_returns_correct_structure(self):
        """12.1: Returns expected default structure."""
        default = _get_default_validation_comparison()

        assert 'baseline_validation' in default
        assert 'enhanced_validation' in default
        assert 'validation_improvements' in default
        assert 'statistical_rigor_assessment' in default


class TestExtractBaselineValidation:
    """Tests for _extract_baseline_validation."""

    def test_basic_level_no_validation(self, empty_results):
        """13.1: Returns BASIC level for empty results."""
        validation = _extract_baseline_validation(empty_results)

        assert validation['validation_level'] == 'BASIC'
        assert validation['out_of_sample_validation'] is False

    def test_moderate_level_with_bootstrap(self, baseline_results):
        """13.2: Returns MODERATE level with bootstrap."""
        validation = _extract_baseline_validation(baseline_results)

        assert validation['validation_level'] == 'MODERATE'
        assert 'Standard Bootstrap' in str(validation['methods_used'])


class TestExtractEnhancedValidation:
    """Tests for _extract_enhanced_validation."""

    def test_high_level_with_temporal(self, enhanced_results):
        """14.1: Returns HIGH or RIGOROUS level with full validation."""
        validation = _extract_enhanced_validation(enhanced_results)

        assert validation['validation_level'] in ['HIGH', 'RIGOROUS']
        assert validation['out_of_sample_validation'] is True
        assert validation['multiple_testing_correction'] is True


class TestIdentifyValidationImprovements:
    """Tests for _identify_validation_improvements."""

    def test_identifies_oos_improvement(self):
        """15.1: Identifies OOS validation improvement."""
        baseline = {}  # No temporal validation
        enhanced = {'temporal_validation': True}

        improvements = _identify_validation_improvements(baseline, enhanced)

        assert any('out-of-sample' in imp.lower() for imp in improvements)

    def test_identifies_multiple_testing_improvement(self):
        """15.2: Identifies multiple testing improvement."""
        baseline = {}
        enhanced = {'multiple_testing_correction': True}

        improvements = _identify_validation_improvements(baseline, enhanced)

        assert any('multiple testing' in imp.lower() for imp in improvements)


class TestAssessStatisticalRigor:
    """Tests for _assess_statistical_rigor."""

    def test_calculates_improvement(self, baseline_results, enhanced_results):
        """16.1: Calculates rigor improvement."""
        assessment = _assess_statistical_rigor(baseline_results, enhanced_results)

        assert assessment['enhanced_rigor_score'] > assessment['baseline_rigor_score']
        assert assessment['rigor_improvement'] > 0


class TestCalculateStatisticalRigorScore:
    """Tests for _calculate_statistical_rigor_score."""

    def test_base_score_with_selected_model(self, baseline_results):
        """17.1: Awards points for selected model."""
        score = _calculate_statistical_rigor_score(baseline_results)

        assert score >= 20  # Base 20 points for selected model

    def test_max_score_with_full_validation(self, enhanced_results):
        """17.2: Awards high score for full validation."""
        score = _calculate_statistical_rigor_score(enhanced_results)

        assert score >= 80  # Should be high with all features

    def test_empty_results_zero_score(self, empty_results):
        """17.3: Returns 0 for empty results."""
        score = _calculate_statistical_rigor_score(empty_results)

        assert score == 0


class TestCompareStatisticalValidation:
    """Tests for _compare_statistical_validation orchestration."""

    def test_returns_full_structure(self, baseline_results, enhanced_results):
        """18.1: Returns complete comparison structure."""
        comparison = _compare_statistical_validation(baseline_results, enhanced_results)

        assert 'baseline_validation' in comparison
        assert 'enhanced_validation' in comparison
        assert 'validation_improvements' in comparison
        assert 'statistical_rigor_assessment' in comparison


# =============================================================================
# Tests for Production Readiness Comparison
# =============================================================================


class TestGetBaselineReadiness:
    """Tests for _get_baseline_readiness."""

    def test_returns_low_confidence(self):
        """19.1: Returns LOW confidence baseline."""
        readiness = _get_baseline_readiness()

        assert readiness['confidence_level'] == 'LOW'
        assert readiness['generalization_evidence'] is False


class TestExtractEnhancedReadiness:
    """Tests for _extract_enhanced_readiness."""

    def test_extracts_full_readiness(self, enhanced_results):
        """20.1: Extracts full readiness from enhanced results."""
        readiness = _extract_enhanced_readiness(enhanced_results)

        assert readiness['generalization_evidence'] is True
        assert readiness['assumption_validation'] is True
        assert readiness['confidence_level'] == 'HIGH'

    def test_handles_missing_fields(self, empty_results):
        """20.2: Handles missing fields gracefully."""
        readiness = _extract_enhanced_readiness(empty_results)

        assert readiness['generalization_evidence'] is False
        assert readiness['confidence_level'] == 'UNKNOWN'


class TestIdentifyReadinessImprovements:
    """Tests for _identify_readiness_improvements."""

    def test_identifies_generalization_improvement(self):
        """21.1: Identifies generalization evidence improvement."""
        baseline = {
            'generalization_evidence': False,
            'assumption_validation': False,
            'statistical_significance': False,
        }
        enhanced = {
            'generalization_evidence': True,
            'assumption_validation': True,
            'statistical_significance': True,
        }

        improvements = _identify_readiness_improvements(baseline, enhanced)

        assert any('generalization' in imp.lower() for imp in improvements)


class TestAssessDeploymentConfidence:
    """Tests for _assess_deployment_confidence."""

    def test_significant_improvement(self):
        """22.1: Identifies SIGNIFICANT improvement for 40+ points."""
        assessment = _assess_deployment_confidence(30, 80)

        assert assessment['confidence_interpretation'] == 'SIGNIFICANT IMPROVEMENT'
        assert assessment['confidence_improvement'] == 50

    def test_moderate_improvement(self):
        """22.2: Identifies MODERATE improvement for 20-39 points."""
        assessment = _assess_deployment_confidence(30, 55)

        assert assessment['confidence_interpretation'] == 'MODERATE IMPROVEMENT'

    def test_minimal_improvement(self):
        """22.3: Identifies MINIMAL improvement for <20 points."""
        assessment = _assess_deployment_confidence(30, 45)

        assert assessment['confidence_interpretation'] == 'MINIMAL IMPROVEMENT'


class TestCalculateConfidenceScore:
    """Tests for _calculate_confidence_score."""

    def test_base_score(self):
        """23.1: Returns base score of 30 for empty readiness."""
        score = _calculate_confidence_score({})

        assert score == 30

    def test_full_score(self):
        """23.2: Returns high score for full readiness."""
        readiness = {
            'generalization_evidence': True,
            'assumption_validation': True,
            'statistical_significance': True,
            'confidence_level': 'HIGH',
        }
        score = _calculate_confidence_score(readiness)

        assert score >= 95  # 30 + 30 + 20 + 15 + 5 = 100


class TestCompareProductionReadiness:
    """Tests for _compare_production_readiness orchestration."""

    def test_returns_full_structure(self, baseline_results, enhanced_results):
        """24.1: Returns complete comparison structure."""
        comparison = _compare_production_readiness(baseline_results, enhanced_results)

        assert 'baseline_readiness' in comparison
        assert 'enhanced_readiness' in comparison
        assert 'readiness_improvements' in comparison
        assert 'deployment_confidence' in comparison


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for comparison metrics."""

    def test_full_comparison_workflow(self, baseline_results, enhanced_results):
        """25.1: Full comparison workflow produces consistent results."""
        # Performance comparison
        perf = _compare_performance_metrics(baseline_results, enhanced_results)
        assert not perf.get('comparison_failed', False)

        # Model selection comparison
        model = _compare_model_selection(baseline_results, enhanced_results)
        assert not model.get('comparison_failed', False)

        # Statistical validation comparison
        stat = _compare_statistical_validation(baseline_results, enhanced_results)
        assert not stat.get('comparison_failed', False)

        # Production readiness comparison
        prod = _compare_production_readiness(baseline_results, enhanced_results)
        assert not prod.get('comparison_failed', False)

    def test_enhanced_always_better_or_equal(self, baseline_results, enhanced_results):
        """25.2: Enhanced methodology scores better or equal in all areas."""
        rigor = _assess_statistical_rigor(baseline_results, enhanced_results)
        assert rigor['enhanced_rigor_score'] >= rigor['baseline_rigor_score']

        deployment = _assess_deployment_confidence(30, 80)  # Typical scores
        assert deployment['confidence_improvement'] >= 0
