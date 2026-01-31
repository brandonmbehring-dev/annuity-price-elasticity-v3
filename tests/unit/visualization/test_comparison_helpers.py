"""
Tests for src.visualization.comparison_helpers module.

Tests pure utility functions for model comparison visualizations:
- Pareto frontier calculation
- Bootstrap metrics extraction
- Constraint summary computation
- Model selection summary generation

Target coverage: 90%+
"""

from collections import namedtuple
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from src.visualization.comparison_helpers import (
    compute_constraint_summary,
    compute_stability_ranking,
    create_model_selection_summary,
    extract_bootstrap_metrics,
    extract_sign_consistency_data,
    extract_uncertainty_data,
    find_pareto_frontier,
    sort_bootstrap_by_stability,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_top_models():
    """Sample top models DataFrame for testing."""
    return pd.DataFrame({
        'features': ['feat_a+feat_b', 'feat_c+feat_d', 'feat_e'],
        'aic': [100.5, 105.2, 98.7],
        'r_squared': [0.85, 0.82, 0.87],
        'coefficients': [
            {'const': 1.0, 'prudential_rate': 0.5, 'competitor_avg': -0.3},
            {'const': 0.8, 'prudential_rate': 0.4, 'competitor_rate': -0.2},
            {'const': 1.2, 'prudential_rate': -0.1, 'competitor_rate': 0.1},  # violations
        ]
    })


@pytest.fixture
def sample_bootstrap_results_dict():
    """Sample bootstrap results in dict format."""
    return [
        {'features': 'feat_a+feat_b', 'aic_stability_cv': 0.05, 'successful_fits': 98},
        {'features': 'feat_c+feat_d', 'aic_stability_cv': 0.08, 'successful_fits': 95},
        {'features': 'feat_e', 'aic_stability_cv': 0.03, 'successful_fits': 100},
    ]


@pytest.fixture
def sample_bootstrap_results_namedtuple():
    """Sample bootstrap results in namedtuple format."""
    BootstrapResult = namedtuple('BootstrapResult', ['model_features', 'stability_metrics'])
    return [
        BootstrapResult('feat_a+feat_b', {'aic_cv': 0.05, 'successful_fit_rate': 0.98}),
        BootstrapResult('feat_c+feat_d', {'aic_cv': 0.08, 'successful_fit_rate': 0.95}),
        BootstrapResult('feat_e', {'aic_cv': 0.03, 'successful_fit_rate': 1.0}),
    ]


@pytest.fixture
def sample_coefficient_stability():
    """Sample coefficient stability data."""
    return {
        'feat_a+feat_b': {
            'prudential_rate': {'sign_consistency': 0.95, 'mean': 0.5, 'cv': 0.1},
            'competitor_avg': {'sign_consistency': 0.90, 'mean': -0.3, 'cv': 0.15},
        },
        'feat_c+feat_d': {
            'prudential_rate': {'sign_consistency': 0.88, 'mean': 0.4, 'cv': 0.2},
            'competitor_rate': {'sign_consistency': 0.92, 'mean': -0.2, 'cv': 0.12},
        },
        'feat_e': {
            'prudential_rate': {'sign_consistency': 0.70, 'mean': -0.1, 'cv': 0.5},
        },
    }


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame."""
    return pd.DataFrame({
        'features': ['model_a', 'model_b', 'model_c'],
        'aic': [100.0, 102.5, 105.0],
        'r_squared': [0.85, 0.83, 0.80],
    })


# =============================================================================
# PARETO FRONTIER TESTS
# =============================================================================


class TestFindParetoFrontier:
    """Tests for find_pareto_frontier function."""

    def test_basic_pareto_frontier(self):
        """Test basic Pareto frontier with clear optimal points."""
        # Points: minimize x (AIC), maximize y (R²)
        # Index:  0    1    2    3    4
        x_values = np.array([100, 105, 98, 110, 95])
        y_values = np.array([0.85, 0.82, 0.87, 0.80, 0.88])

        result = find_pareto_frontier(x_values, y_values)

        # Point at index 4 (95, 0.88) dominates all others:
        # - dominates 0 (100, 0.85): 95 < 100 and 0.88 > 0.85
        # - dominates 1 (105, 0.82): 95 < 105 and 0.88 > 0.82
        # - dominates 2 (98, 0.87): 95 < 98 and 0.88 > 0.87
        # - dominates 3 (110, 0.80): 95 < 110 and 0.88 > 0.80
        assert result[4] == True  # (95, 0.88) - best on both dimensions
        # Point 2 (98, 0.87) is dominated by point 4
        assert result[2] == False
        # Only point 4 should be Pareto optimal
        assert np.sum(result) == 1

    def test_single_point_pareto(self):
        """Test single point is always Pareto optimal."""
        x_values = np.array([100.0])
        y_values = np.array([0.85])

        result = find_pareto_frontier(x_values, y_values)

        assert len(result) == 1
        assert result[0] == True

    def test_two_points_pareto(self):
        """Test two points where one dominates the other."""
        # Point 0: (100, 0.85), Point 1: (90, 0.90)
        # Point 1 dominates Point 0 (lower x, higher y)
        x_values = np.array([100.0, 90.0])
        y_values = np.array([0.85, 0.90])

        result = find_pareto_frontier(x_values, y_values)

        assert result[0] == False  # Dominated
        assert result[1] == True   # Dominates

    def test_two_nondominated_points(self):
        """Test two points that don't dominate each other."""
        # Point 0: (100, 0.90), Point 1: (90, 0.85)
        # Neither dominates: Point 0 has higher y, Point 1 has lower x
        x_values = np.array([100.0, 90.0])
        y_values = np.array([0.90, 0.85])

        result = find_pareto_frontier(x_values, y_values)

        assert result[0] == True  # Not dominated
        assert result[1] == True  # Not dominated

    def test_all_points_on_pareto_frontier(self):
        """Test case where all points are Pareto optimal."""
        # Diagonal trade-off: as x decreases, y also decreases
        x_values = np.array([100.0, 90.0, 80.0])
        y_values = np.array([0.90, 0.85, 0.80])

        result = find_pareto_frontier(x_values, y_values)

        assert all(result)  # All points are Pareto optimal

    def test_many_dominated_points(self):
        """Test case with many dominated points."""
        x_values = np.array([100, 110, 120, 90, 95])
        y_values = np.array([0.85, 0.75, 0.70, 0.90, 0.88])

        result = find_pareto_frontier(x_values, y_values)

        # Point at index 3 (90, 0.90) dominates most
        assert result[3] == True
        # Point at index 4 (95, 0.88) might be on frontier
        assert result[4] == False  # Dominated by (90, 0.90)

    def test_identical_points(self):
        """Test handling of identical points."""
        x_values = np.array([100.0, 100.0, 100.0])
        y_values = np.array([0.85, 0.85, 0.85])

        result = find_pareto_frontier(x_values, y_values)

        # At least one should be on frontier (algorithm dependent)
        assert np.sum(result) >= 1

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        x_values = np.array([])
        y_values = np.array([])

        result = find_pareto_frontier(x_values, y_values)

        assert len(result) == 0


# =============================================================================
# BOOTSTRAP METRICS EXTRACTION TESTS
# =============================================================================


class TestExtractBootstrapMetrics:
    """Tests for extract_bootstrap_metrics function."""

    def test_extract_from_dict_format(self, sample_bootstrap_results_dict):
        """Test extraction from dict format bootstrap results."""
        names, cvs, rates = extract_bootstrap_metrics(sample_bootstrap_results_dict)

        assert names == ['feat_a+feat_b', 'feat_c+feat_d', 'feat_e']
        assert cvs == [0.05, 0.08, 0.03]
        assert rates == [0.98, 0.95, 1.0]

    def test_extract_from_namedtuple_format(self, sample_bootstrap_results_namedtuple):
        """Test extraction from namedtuple format bootstrap results."""
        names, cvs, rates = extract_bootstrap_metrics(sample_bootstrap_results_namedtuple)

        assert names == ['feat_a+feat_b', 'feat_c+feat_d', 'feat_e']
        assert cvs == [0.05, 0.08, 0.03]
        assert rates == [0.98, 0.95, 1.0]

    def test_extract_empty_results(self):
        """Test extraction from empty results list."""
        names, cvs, rates = extract_bootstrap_metrics([])

        assert names == []
        assert cvs == []
        assert rates == []

    def test_extract_missing_keys(self):
        """Test extraction when dict keys are missing."""
        results = [
            {'other_key': 'value'},  # Missing features, aic_stability_cv, successful_fits
        ]

        names, cvs, rates = extract_bootstrap_metrics(results)

        # Should use defaults
        assert names == ['Unknown']
        assert cvs == [0]
        assert rates == [1.0]  # 100/100

    def test_mixed_formats(self):
        """Test extraction from mixed dict and namedtuple formats."""
        BootstrapResult = namedtuple('BootstrapResult', ['model_features', 'stability_metrics'])

        results = [
            {'features': 'dict_model', 'aic_stability_cv': 0.1, 'successful_fits': 90},
            BootstrapResult('tuple_model', {'aic_cv': 0.2, 'successful_fit_rate': 0.85}),
        ]

        names, cvs, rates = extract_bootstrap_metrics(results)

        assert len(names) == 2
        assert 'dict_model' in names
        assert 'tuple_model' in names


class TestComputeStabilityRanking:
    """Tests for compute_stability_ranking function."""

    def test_basic_ranking(self, sample_top_models, sample_bootstrap_results_dict):
        """Test basic stability ranking computation."""
        result = compute_stability_ranking(sample_top_models, sample_bootstrap_results_dict)

        assert result is not None
        assert len(result) == len(sample_top_models)
        # Rankings should be 1, 2, 3 based on CV values

    def test_ranking_with_missing_bootstrap(self, sample_top_models):
        """Test ranking when bootstrap results don't match models."""
        bootstrap_results = [
            {'features': 'unknown_model', 'aic_stability_cv': 0.1},
        ]

        result = compute_stability_ranking(sample_top_models, bootstrap_results)

        # All models should get default rank (tied at 999)
        assert result is not None
        assert len(result) == len(sample_top_models)

    def test_ranking_with_namedtuple(self, sample_top_models, sample_bootstrap_results_namedtuple):
        """Test ranking with namedtuple format bootstrap results."""
        result = compute_stability_ranking(sample_top_models, sample_bootstrap_results_namedtuple)

        assert result is not None
        assert len(result) == len(sample_top_models)


class TestSortBootstrapByStability:
    """Tests for sort_bootstrap_by_stability function."""

    def test_sort_dict_format(self, sample_bootstrap_results_dict):
        """Test sorting dict format by stability."""
        sorted_results = sort_bootstrap_by_stability(sample_bootstrap_results_dict, top_n=3)

        assert len(sorted_results) == 3
        # First element should have lowest CV
        assert sorted_results[0][0] == 0.03  # feat_e
        assert sorted_results[1][0] == 0.05  # feat_a+feat_b
        assert sorted_results[2][0] == 0.08  # feat_c+feat_d

    def test_sort_namedtuple_format(self, sample_bootstrap_results_namedtuple):
        """Test sorting namedtuple format by stability."""
        sorted_results = sort_bootstrap_by_stability(sample_bootstrap_results_namedtuple, top_n=3)

        assert len(sorted_results) == 3
        # First element should have lowest CV
        assert sorted_results[0][0] == 0.03

    def test_sort_with_top_n_limit(self, sample_bootstrap_results_dict):
        """Test sorting respects top_n limit."""
        sorted_results = sort_bootstrap_by_stability(sample_bootstrap_results_dict, top_n=2)

        assert len(sorted_results) == 2

    def test_sort_empty_results(self):
        """Test sorting empty results."""
        sorted_results = sort_bootstrap_by_stability([], top_n=5)

        assert sorted_results == []

    def test_sort_unknown_format(self):
        """Test sorting with unknown object format."""
        results = [object(), object()]  # Unknown format

        sorted_results = sort_bootstrap_by_stability(results, top_n=2)

        # Should all have inf CV
        assert len(sorted_results) == 2
        assert sorted_results[0][0] == float('inf')


# =============================================================================
# CONSTRAINT SUMMARY TESTS
# =============================================================================


class TestComputeConstraintSummary:
    """Tests for compute_constraint_summary function."""

    def test_basic_constraint_summary(self, sample_top_models):
        """Test basic constraint summary computation."""
        summary = compute_constraint_summary(sample_top_models)

        assert len(summary) == 3
        assert all('model' in item for item in summary)
        assert all('pass_rate' in item for item in summary)
        assert all('violations' in item for item in summary)
        assert all('total_checks' in item for item in summary)

    def test_constraint_violations_detected(self, sample_top_models):
        """Test that violations are correctly detected."""
        summary = compute_constraint_summary(sample_top_models)

        # Model at index 2 has violations:
        # - prudential_rate is negative (should be positive)
        # - competitor_rate is positive (should be negative)
        third_model = summary[2]
        assert third_model['violations'] == 2

    def test_constraint_passes(self, sample_top_models):
        """Test models that pass constraints."""
        summary = compute_constraint_summary(sample_top_models)

        # First model should have no violations
        first_model = summary[0]
        assert first_model['violations'] == 0
        assert first_model['pass_rate'] == 1.0

    def test_empty_coefficients(self):
        """Test handling of empty coefficients."""
        models = pd.DataFrame({
            'features': ['empty_model'],
            'coefficients': [{}]
        })

        summary = compute_constraint_summary(models)

        assert len(summary) == 1
        assert summary[0]['total_checks'] == 0
        # Formula: (0 - 0) / max(1, 0) = 0.0
        assert summary[0]['pass_rate'] == 0.0

    def test_const_excluded(self):
        """Test that const coefficient is excluded from checks."""
        models = pd.DataFrame({
            'features': ['const_only'],
            'coefficients': [{'const': 1.0}]
        })

        summary = compute_constraint_summary(models)

        assert summary[0]['total_checks'] == 0

    def test_model_name_truncation(self):
        """Test model name is truncated to 20 characters."""
        models = pd.DataFrame({
            'features': ['this_is_a_very_long_model_name_that_exceeds_twenty_chars'],
            'coefficients': [{'feat': 0.1}]
        })

        summary = compute_constraint_summary(models)

        assert len(summary[0]['model']) == 20


# =============================================================================
# SIGN CONSISTENCY DATA TESTS
# =============================================================================


class TestExtractSignConsistencyData:
    """Tests for extract_sign_consistency_data function."""

    def test_basic_extraction(self, sample_coefficient_stability, sample_top_models):
        """Test basic sign consistency extraction."""
        features, consistency, means = extract_sign_consistency_data(
            sample_coefficient_stability, sample_top_models
        )

        assert len(features) > 0
        assert len(features) == len(consistency) == len(means)
        # All consistency values should be between 0 and 1
        assert all(0 <= c <= 1 for c in consistency)
        # All means should be non-negative (absolute values)
        assert all(m >= 0 for m in means)

    def test_only_matching_models(self, sample_coefficient_stability, sample_top_models):
        """Test that only matching models are included."""
        # Add non-matching model to stability data
        stability = sample_coefficient_stability.copy()
        stability['non_matching_model'] = {'feat': {'sign_consistency': 0.5, 'mean': 0.1}}

        features, _, _ = extract_sign_consistency_data(stability, sample_top_models)

        # Non-matching model should not be included
        assert not any('non_matching' in f for f in features)

    def test_empty_stability(self, sample_top_models):
        """Test with empty coefficient stability."""
        features, consistency, means = extract_sign_consistency_data({}, sample_top_models)

        assert features == []
        assert consistency == []
        assert means == []

    def test_missing_stats_keys(self, sample_top_models):
        """Test handling of missing stats keys."""
        stability = {
            'feat_a+feat_b': {
                'some_feature': {}  # Missing sign_consistency and mean
            }
        }

        features, consistency, means = extract_sign_consistency_data(stability, sample_top_models)

        assert len(features) == 1
        assert consistency[0] == 0  # Default
        assert means[0] == 0  # Default


class TestExtractUncertaintyData:
    """Tests for extract_uncertainty_data function."""

    def test_basic_extraction(self, sample_coefficient_stability, sample_top_models):
        """Test basic uncertainty data extraction."""
        data = extract_uncertainty_data(sample_coefficient_stability, sample_top_models)

        assert len(data) > 0
        assert all('feature' in item for item in data)
        assert all('model' in item for item in data)
        assert all('cv' in item for item in data)
        assert all('uncertainty_score' in item for item in data)

    def test_uncertainty_score_capped(self, sample_top_models):
        """Test uncertainty score is capped at 2.0."""
        stability = {
            'feat_a+feat_b': {
                'feature': {'cv': 10.0, 'mean': 0.1}  # Very high CV
            }
        }

        data = extract_uncertainty_data(stability, sample_top_models)

        assert len(data) == 1
        assert data[0]['cv'] == 10.0  # Original preserved
        assert data[0]['uncertainty_score'] == 2.0  # Capped at 2.0

    def test_model_name_truncation(self, sample_coefficient_stability, sample_top_models):
        """Test model name is truncated with ellipsis."""
        data = extract_uncertainty_data(sample_coefficient_stability, sample_top_models)

        # Model names should end with '...'
        assert all(item['model'].endswith('...') for item in data)

    def test_empty_stability(self, sample_top_models):
        """Test with empty coefficient stability."""
        data = extract_uncertainty_data({}, sample_top_models)

        assert data == []


# =============================================================================
# MODEL SELECTION SUMMARY TESTS
# =============================================================================


class TestCreateModelSelectionSummary:
    """Tests for create_model_selection_summary function."""

    def test_basic_summary(
        self, sample_aic_results, sample_bootstrap_results_dict
    ):
        """Test basic summary creation."""
        info_criteria = [{'model': 'a', 'aic': 100}]

        summary = create_model_selection_summary(
            sample_aic_results,
            info_criteria,
            sample_bootstrap_results_dict
        )

        assert 'MODEL SELECTION DECISION SUMMARY' in summary
        assert 'Analysis Scope' in summary
        assert 'Best AIC Performance' in summary
        assert 'Most Stable Model' in summary
        assert 'Selection Methodology' in summary

    def test_summary_with_empty_aic(self, sample_bootstrap_results_dict):
        """Test summary with empty AIC results."""
        summary = create_model_selection_summary(
            pd.DataFrame(),
            [],
            sample_bootstrap_results_dict
        )

        assert 'MODEL SELECTION DECISION SUMMARY' in summary
        # Should not crash, just skip best AIC section

    def test_summary_with_empty_bootstrap(self, sample_aic_results):
        """Test summary with empty bootstrap results."""
        summary = create_model_selection_summary(
            sample_aic_results,
            [],
            []
        )

        assert 'MODEL SELECTION DECISION SUMMARY' in summary
        # Should not crash, just skip stability section

    def test_summary_with_namedtuple_bootstrap(
        self, sample_aic_results, sample_bootstrap_results_namedtuple
    ):
        """Test summary creation with namedtuple bootstrap results."""
        summary = create_model_selection_summary(
            sample_aic_results,
            [],
            sample_bootstrap_results_namedtuple
        )

        assert 'Most Stable Model' in summary
        assert 'feat_e' in summary  # Most stable (CV=0.03)

    def test_summary_includes_counts(self, sample_aic_results, sample_bootstrap_results_dict):
        """Test summary includes correct counts."""
        info_criteria = [{'a': 1}, {'b': 2}]

        summary = create_model_selection_summary(
            sample_aic_results,
            info_criteria,
            sample_bootstrap_results_dict
        )

        assert 'Total Models Evaluated: 3' in summary
        assert 'Information Criteria Models: 2' in summary
        assert 'Bootstrap Stability Models: 3' in summary

    def test_summary_best_aic_details(self, sample_aic_results, sample_bootstrap_results_dict):
        """Test summary includes best AIC model details."""
        summary = create_model_selection_summary(
            sample_aic_results,
            [],
            sample_bootstrap_results_dict
        )

        assert 'model_a' in summary
        assert '100.0000' in summary  # AIC Score
        assert '0.8500' in summary  # R-squared

    def test_summary_methodology_checklist(self, sample_aic_results, sample_bootstrap_results_dict):
        """Test summary includes methodology checklist."""
        summary = create_model_selection_summary(
            sample_aic_results,
            [],
            sample_bootstrap_results_dict
        )

        assert '[PASS] AIC-based model quality' in summary
        assert '[PASS] Economic constraint validation' in summary
        assert '[PASS] Bootstrap stability' in summary
        assert '[PASS] Multi-criteria robustness' in summary

    def test_summary_recommendation(self, sample_aic_results, sample_bootstrap_results_dict):
        """Test summary includes recommendation."""
        summary = create_model_selection_summary(
            sample_aic_results,
            [],
            sample_bootstrap_results_dict
        )

        assert 'Recommendation:' in summary
        assert 'integrated selection approach' in summary


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases across all helper functions."""

    def test_pareto_with_nan_values(self):
        """Test Pareto frontier handling of NaN values."""
        x_values = np.array([100.0, np.nan, 95.0])
        y_values = np.array([0.85, 0.90, 0.88])

        # Should handle NaN gracefully (behavior depends on implementation)
        result = find_pareto_frontier(x_values, y_values)
        assert len(result) == 3

    def test_bootstrap_with_zero_successful_fits(self):
        """Test bootstrap extraction with zero successful fits."""
        results = [
            {'features': 'model', 'aic_stability_cv': 0.1, 'successful_fits': 0}
        ]

        names, cvs, rates = extract_bootstrap_metrics(results)

        assert rates[0] == 0.0

    def test_constraint_with_mixed_case_keys(self):
        """Test constraint summary with mixed case coefficient names."""
        models = pd.DataFrame({
            'features': ['model'],
            'coefficients': [{
                'COMPETITOR_rate': 0.5,  # Upper case, violation
                'Prudential_Rate': -0.1,  # Mixed case, violation
            }]
        })

        summary = compute_constraint_summary(models)

        # Both should be detected as violations
        assert summary[0]['violations'] == 2

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        n_points = 100
        x_values = np.random.rand(n_points) * 100
        y_values = np.random.rand(n_points)

        result = find_pareto_frontier(x_values, y_values)

        assert len(result) == n_points
        assert np.sum(result) > 0  # At least some Pareto optimal points
