"""
Tests for Multiple Testing Correction Orchestrator Module.

Tests cover:
- Method comparison functionality
- Re-exported functions from split modules
- Recommendation generation
- Integration between Bonferroni, FDR, and reduced search space

Design Principles:
- Integration tests for orchestration
- Unit tests for helper functions
- Verify re-exports work correctly

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np

from src.features.selection.enhancements.multiple_testing.multiple_testing_correction import (
    # Re-exported types
    MultipleTestingResults,
    # Re-exported functions
    apply_bonferroni_correction,
    apply_fdr_correction,
    create_reduced_search_space,
    # Orchestrator helpers
    _build_bonferroni_method_summary,
    _build_fdr_method_summary,
    _build_reduced_space_method_summary,
    _recommend_conservative_approach,
    _recommend_liberal_approach,
    _recommend_primary_approach,
    _generate_method_recommendations,
    # Main comparison function
    compare_correction_methods,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model_results_df():
    """Create sample model results DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'features': [f'model_{i}' for i in range(n)],
        'aic': 100 + np.random.exponential(scale=5, size=n),
        'r_squared': np.random.uniform(0.5, 0.9, n),
    })


@pytest.fixture
def large_model_results():
    """Create larger model results for realistic testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'features': [f'model_{i}' for i in range(n)],
        'aic': 100 + np.random.exponential(scale=5, size=n),
        'r_squared': np.random.uniform(0.5, 0.9, n),
    })


@pytest.fixture
def bonf_results(model_results_df):
    """Pre-computed Bonferroni results."""
    return apply_bonferroni_correction(model_results_df, 'aic')


@pytest.fixture
def fdr_results(model_results_df):
    """Pre-computed FDR results."""
    return apply_fdr_correction(model_results_df, 'aic')


@pytest.fixture
def reduced_space():
    """Pre-computed reduced search space."""
    return create_reduced_search_space(
        candidate_features=['f1', 'f2', 'f3', 'f4', 'f5'],
        max_combinations=20
    )


# =============================================================================
# Tests for Re-Exported Functions
# =============================================================================


class TestReExports:
    """Tests that re-exported functions work correctly."""

    def test_bonferroni_available(self, model_results_df):
        """Test that apply_bonferroni_correction is available."""
        result = apply_bonferroni_correction(model_results_df, 'aic')
        assert isinstance(result, MultipleTestingResults)

    def test_fdr_available(self, model_results_df):
        """Test that apply_fdr_correction is available."""
        result = apply_fdr_correction(model_results_df, 'aic')
        assert isinstance(result, MultipleTestingResults)

    def test_reduced_space_available(self):
        """Test that create_reduced_search_space is available."""
        result = create_reduced_search_space(['f1', 'f2', 'f3'])
        assert isinstance(result, dict)
        assert 'combinations' in result


# =============================================================================
# Tests for Method Summary Builders
# =============================================================================


class TestBuildBonferroniMethodSummary:
    """Tests for Bonferroni method summary builder."""

    def test_returns_dict_with_required_keys(self, bonf_results, model_results_df):
        """Test that summary has required keys."""
        summary = _build_bonferroni_method_summary(bonf_results, len(model_results_df))

        assert 'significant_models' in summary
        assert 'corrected_alpha' in summary
        assert 'rejection_rate' in summary
        assert 'statistical_power' in summary
        assert 'interpretation' in summary

    def test_significant_models_is_int(self, bonf_results, model_results_df):
        """Test that significant_models is an integer."""
        summary = _build_bonferroni_method_summary(bonf_results, len(model_results_df))

        assert isinstance(summary['significant_models'], int)

    def test_rejection_rate_in_range(self, bonf_results, model_results_df):
        """Test that rejection rate is in [0, 1]."""
        summary = _build_bonferroni_method_summary(bonf_results, len(model_results_df))

        assert 0 <= summary['rejection_rate'] <= 1


class TestBuildFDRMethodSummary:
    """Tests for FDR method summary builder."""

    def test_returns_dict_with_required_keys(self, fdr_results, model_results_df):
        """Test that summary has required keys."""
        summary = _build_fdr_method_summary(fdr_results, len(model_results_df))

        assert 'significant_models' in summary
        assert 'corrected_alpha' in summary
        assert 'rejection_rate' in summary
        assert 'interpretation' in summary

    def test_interpretation_mentions_fdr(self, fdr_results, model_results_df):
        """Test that interpretation mentions FDR control."""
        summary = _build_fdr_method_summary(fdr_results, len(model_results_df))

        assert 'false discovery' in summary['interpretation'].lower()


class TestBuildReducedSpaceMethodSummary:
    """Tests for reduced search space method summary builder."""

    def test_returns_dict_with_required_keys(self, reduced_space):
        """Test that summary has required keys."""
        summary = _build_reduced_space_method_summary(reduced_space)

        assert 'combinations_tested' in summary
        assert 'reduction_factor' in summary
        assert 'multiple_testing_eliminated' in summary
        assert 'effective_alpha' in summary
        assert 'interpretation' in summary

    def test_interpretation_mentions_domain(self, reduced_space):
        """Test that interpretation mentions domain knowledge."""
        summary = _build_reduced_space_method_summary(reduced_space)

        assert 'domain' in summary['interpretation'].lower()


# =============================================================================
# Tests for Recommendation Generators
# =============================================================================


class TestRecommendConservativeApproach:
    """Tests for conservative approach recommendation."""

    def test_sufficient_models_recommends_bonferroni(self, bonf_results):
        """Test recommendation when sufficient significant models."""
        # Ensure we have >= 3 significant models
        if len(bonf_results.significant_models) >= 3:
            rec = _recommend_conservative_approach(bonf_results)
            assert 'bonferroni' in rec.lower()

    def test_few_models_warns_stringent(self, model_results_df):
        """Test recommendation when few significant models."""
        # Use very strict alpha to get few models
        strict_results = apply_bonferroni_correction(model_results_df, 'aic', alpha=0.001)

        rec = _recommend_conservative_approach(strict_results)
        # Should either recommend or warn about stringency
        assert 'bonferroni' in rec.lower() or 'stringent' in rec.lower()


class TestRecommendLiberalApproach:
    """Tests for liberal approach recommendation."""

    def test_small_space_recommends_reduced(self, reduced_space):
        """Test recommendation for small reduced space."""
        rec = _recommend_liberal_approach(reduced_space)

        # Should mention reduced search space
        assert 'reduced' in rec.lower()

    def test_large_space_needs_correction(self):
        """Test recommendation for large reduced space."""
        large_space = create_reduced_search_space(
            ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
            max_combinations=100
        )

        rec = _recommend_liberal_approach(large_space)
        # Either recommend or note need for correction
        assert 'reduced' in rec.lower() or 'correction' in rec.lower()


class TestRecommendPrimaryApproach:
    """Tests for primary approach recommendation."""

    def test_returns_string(self, fdr_results, reduced_space):
        """Test that function returns a string recommendation."""
        rec = _recommend_primary_approach(fdr_results, reduced_space)

        assert isinstance(rec, str)
        assert len(rec) > 0

    def test_recommends_one_method(self, fdr_results, reduced_space):
        """Test that recommendation mentions a specific method."""
        rec = _recommend_primary_approach(fdr_results, reduced_space)

        methods = ['fdr', 'bonferroni', 'reduced']
        assert any(m in rec.lower() for m in methods)


class TestGenerateMethodRecommendations:
    """Tests for full method recommendation generation."""

    def test_returns_dict_with_all_keys(self, bonf_results, fdr_results, reduced_space):
        """Test that all recommendation types are present."""
        recs = _generate_method_recommendations(
            bonf_results, fdr_results, reduced_space, alpha=0.05
        )

        assert 'conservative' in recs
        assert 'balanced' in recs
        assert 'liberal' in recs
        assert 'primary' in recs

    def test_all_recommendations_are_strings(self, bonf_results, fdr_results, reduced_space):
        """Test that all recommendations are strings."""
        recs = _generate_method_recommendations(
            bonf_results, fdr_results, reduced_space, alpha=0.05
        )

        for key, value in recs.items():
            assert isinstance(value, str)


# =============================================================================
# Tests for compare_correction_methods
# =============================================================================


class TestCompareCorrectMethods:
    """Tests for main comparison function."""

    def test_returns_dict(self, model_results_df):
        """Test that function returns dict."""
        result = compare_correction_methods(model_results_df)

        assert isinstance(result, dict)

    def test_contains_timestamp(self, model_results_df):
        """Test that result contains timestamp."""
        result = compare_correction_methods(model_results_df)

        assert 'comparison_timestamp' in result

    def test_contains_original_model_count(self, model_results_df):
        """Test that original model count is included."""
        result = compare_correction_methods(model_results_df)

        assert result['original_model_count'] == len(model_results_df)

    def test_contains_all_methods(self, model_results_df):
        """Test that all three methods are compared."""
        result = compare_correction_methods(model_results_df)

        assert 'methods' in result
        assert 'bonferroni' in result['methods']
        assert 'fdr_bh' in result['methods']
        assert 'reduced_space' in result['methods']

    def test_contains_recommendations(self, model_results_df):
        """Test that recommendations are included."""
        result = compare_correction_methods(model_results_df)

        assert 'recommendations' in result

    def test_handles_empty_results(self):
        """Test handling of empty model results."""
        empty_df = pd.DataFrame()

        result = compare_correction_methods(empty_df)

        # Should have error flag
        assert 'error' in result or 'comparison_failed' in result

    def test_custom_alpha(self, model_results_df):
        """Test that custom alpha is used."""
        result = compare_correction_methods(model_results_df, alpha=0.01)

        # Should complete without error
        assert 'methods' in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for multiple testing correction."""

    def test_full_workflow(self, large_model_results):
        """Test complete comparison workflow."""
        result = compare_correction_methods(large_model_results)

        # Verify structure
        assert 'comparison_timestamp' in result
        assert 'original_model_count' in result
        assert 'methods' in result
        assert 'recommendations' in result

        # Verify all methods present
        for method in ['bonferroni', 'fdr_bh', 'reduced_space']:
            assert method in result['methods']

        # Verify recommendations present
        for rec_type in ['conservative', 'balanced', 'liberal', 'primary']:
            assert rec_type in result['recommendations']

    def test_methods_process_same_data(self, model_results_df):
        """Test that all methods process the same input."""
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        # Both should have same n_tests
        assert bonf.n_tests == fdr.n_tests == len(model_results_df)

    def test_comparison_consistent_with_individual(self, model_results_df):
        """Test that comparison results match individual method results."""
        # Run individual methods
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        # Run comparison
        comparison = compare_correction_methods(model_results_df)

        # Check consistency
        assert comparison['methods']['bonferroni']['significant_models'] == len(bonf.significant_models)
        assert comparison['methods']['fdr_bh']['significant_models'] == len(fdr.significant_models)
