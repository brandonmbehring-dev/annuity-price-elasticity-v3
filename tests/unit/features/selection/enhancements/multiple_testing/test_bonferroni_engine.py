"""
Tests for Bonferroni Correction Engine.

Tests cover:
- Input validation (empty results, missing columns)
- Bonferroni threshold calculation (mathematical correctness)
- Significance test application (split logic)
- Impact assessment (rejection rates, business interpretation)
- Power analysis (statistical power estimation)
- Full workflow integration (apply_bonferroni_correction)

Design Principles:
- Property-based testing for mathematical invariants
- Edge cases for numerical stability
- Business context validation

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from src.features.selection.enhancements.multiple_testing.bonferroni_engine import (
    _validate_bonferroni_inputs,
    _calculate_bonferroni_threshold,
    _apply_bonferroni_significance,
    _calculate_correction_impact,
    _interpret_rejection_rate,
    _estimate_statistical_power,
    _build_bonferroni_results,
    apply_bonferroni_correction,
)
from src.features.selection.enhancements.multiple_testing.multiple_testing_types import (
    MultipleTestingResults,
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
        'model_id': [f'model_{i}' for i in range(n)],
        'aic': 100 + np.random.exponential(scale=5, size=n),
        'r_squared': np.random.uniform(0.5, 0.9, n),
    })


@pytest.fixture
def large_model_results():
    """Create larger model results (793 models like production)."""
    np.random.seed(42)
    n = 793
    return pd.DataFrame({
        'model_id': [f'model_{i}' for i in range(n)],
        'aic': 100 + np.random.exponential(scale=5, size=n),
    })


@pytest.fixture
def small_model_results():
    """Create small model results for edge case testing."""
    return pd.DataFrame({
        'model_id': ['model_0', 'model_1', 'model_2'],
        'aic': [100.0, 100.5, 105.0],
    })


# =============================================================================
# Tests for _validate_bonferroni_inputs
# =============================================================================


class TestValidateBonferroniInputs:
    """Tests for input validation."""

    def test_empty_results_raises(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError) as excinfo:
            _validate_bonferroni_inputs(empty_df, 'aic')

        assert "CRITICAL" in str(excinfo.value)
        assert "No model results" in str(excinfo.value)
        assert "Business impact" in str(excinfo.value)

    def test_missing_aic_column_raises(self, model_results_df):
        """Test that missing AIC column raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            _validate_bonferroni_inputs(model_results_df, 'nonexistent_column')

        assert "CRITICAL" in str(excinfo.value)
        assert "nonexistent_column" in str(excinfo.value)
        assert "Available columns" in str(excinfo.value)

    def test_valid_inputs_pass(self, model_results_df):
        """Test that valid inputs do not raise."""
        # Should not raise
        _validate_bonferroni_inputs(model_results_df, 'aic')

    def test_custom_aic_column_name(self):
        """Test validation with custom AIC column name."""
        df = pd.DataFrame({'custom_aic': [100, 105, 110]})

        # Should not raise
        _validate_bonferroni_inputs(df, 'custom_aic')


# =============================================================================
# Tests for _calculate_bonferroni_threshold
# =============================================================================


class TestCalculateBonferroniThreshold:
    """Tests for Bonferroni threshold calculation."""

    def test_50_tests(self):
        """Test threshold calculation for 50 tests."""
        corrected_alpha, critical_value = _calculate_bonferroni_threshold(50, 0.05)

        assert corrected_alpha == pytest.approx(0.001, rel=1e-6)
        assert critical_value > 0

    def test_100_tests(self):
        """Test threshold calculation for 100 tests."""
        corrected_alpha, critical_value = _calculate_bonferroni_threshold(100, 0.05)

        assert corrected_alpha == pytest.approx(0.0005, rel=1e-6)
        assert critical_value > 0

    def test_793_tests(self):
        """Test threshold for production scenario (793 models)."""
        corrected_alpha, critical_value = _calculate_bonferroni_threshold(793, 0.05)

        expected = 0.05 / 793
        assert corrected_alpha == pytest.approx(expected, rel=1e-6)
        assert corrected_alpha < 0.0001  # Very small

    def test_single_test(self):
        """Test threshold with single test (no correction needed)."""
        corrected_alpha, critical_value = _calculate_bonferroni_threshold(1, 0.05)

        assert corrected_alpha == 0.05
        assert critical_value > 0

    def test_different_alpha(self):
        """Test threshold with different alpha levels."""
        corrected_01, _ = _calculate_bonferroni_threshold(100, 0.01)
        corrected_05, _ = _calculate_bonferroni_threshold(100, 0.05)

        assert corrected_01 < corrected_05
        assert corrected_01 == pytest.approx(0.0001, rel=1e-6)

    def test_critical_value_increases_with_stringency(self):
        """Test that critical value increases as alpha decreases."""
        _, cv_lenient = _calculate_bonferroni_threshold(10, 0.05)
        _, cv_strict = _calculate_bonferroni_threshold(100, 0.05)

        # More tests = smaller alpha = larger critical value
        assert cv_strict > cv_lenient


# =============================================================================
# Tests for _apply_bonferroni_significance
# =============================================================================


class TestApplyBonferroniSignificance:
    """Tests for significance test application."""

    def test_splits_correctly(self, model_results_df):
        """Test that results are correctly split into significant/rejected."""
        # Use a moderate critical value
        _, critical_value = _calculate_bonferroni_threshold(50, 0.05)

        significant, rejected = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value
        )

        # Should sum to total
        assert len(significant) + len(rejected) == len(model_results_df)

    def test_aic_diff_column_added(self, model_results_df):
        """Test that aic_diff column is added."""
        _, critical_value = _calculate_bonferroni_threshold(50, 0.05)

        significant, rejected = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value
        )

        assert 'aic_diff' in significant.columns
        assert 'bonferroni_significant' in significant.columns

    def test_best_model_always_significant(self, model_results_df):
        """Test that best model (min AIC) is always significant."""
        _, critical_value = _calculate_bonferroni_threshold(50, 0.05)

        significant, _ = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value
        )

        # Find index of min AIC
        min_aic_idx = model_results_df['aic'].idxmin()

        # Best model should have aic_diff = 0
        assert len(significant) > 0
        assert significant['aic_diff'].min() == 0

    def test_single_model_significant(self):
        """Test single model is always significant."""
        df = pd.DataFrame({'aic': [100.0]})
        critical_value = 10.0

        significant, rejected = _apply_bonferroni_significance(df, 'aic', critical_value)

        assert len(significant) == 1
        assert len(rejected) == 0

    def test_critical_value_determines_split(self, small_model_results):
        """Test that critical value correctly determines split point."""
        # AIC values: 100.0, 100.5, 105.0
        # Diffs: 0, 0.5, 5.0

        # Critical value of 1.0 should include first two
        significant, rejected = _apply_bonferroni_significance(
            small_model_results, 'aic', 1.0
        )
        assert len(significant) == 2
        assert len(rejected) == 1

        # Critical value of 0.3 should include only best
        significant, rejected = _apply_bonferroni_significance(
            small_model_results, 'aic', 0.3
        )
        assert len(significant) == 1
        assert len(rejected) == 2


# =============================================================================
# Tests for _calculate_correction_impact
# =============================================================================


class TestCalculateCorrectionImpact:
    """Tests for impact assessment."""

    def test_high_rejection_rate(self):
        """Test impact assessment with high rejection rate (>80%)."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=10,
            method='bonferroni'
        )

        assert impact['models_rejected'] == 90
        assert impact['models_retained'] == 10
        assert impact['rejection_rate_percent'] == 90.0
        assert impact['retention_rate_percent'] == 10.0
        assert impact['correction_stringency'] == 'HIGH'

    def test_moderate_rejection_rate(self):
        """Test impact assessment with moderate rejection rate (50-80%)."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=35,
            method='bonferroni'
        )

        assert impact['correction_stringency'] == 'MODERATE'

    def test_low_rejection_rate(self):
        """Test impact assessment with low rejection rate (<50%)."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=70,
            method='bonferroni'
        )

        assert impact['correction_stringency'] == 'LOW'

    def test_no_rejections(self):
        """Test impact when no models rejected."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=100,
            method='bonferroni'
        )

        assert impact['models_rejected'] == 0
        assert impact['rejection_rate_percent'] == 0.0
        assert impact['correction_stringency'] == 'LOW'

    def test_business_impact_included(self):
        """Test that business impact interpretation is included."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=50,
            method='bonferroni'
        )

        assert 'business_impact' in impact
        assert isinstance(impact['business_impact'], str)
        assert len(impact['business_impact']) > 0


# =============================================================================
# Tests for _interpret_rejection_rate
# =============================================================================


class TestInterpretRejectionRate:
    """Tests for rejection rate interpretation."""

    def test_very_stringent(self):
        """Test interpretation for >90% rejection."""
        interpretation = _interpret_rejection_rate(0.95, 'bonferroni')

        assert 'stringent' in interpretation.lower()
        assert 'bonferroni' in interpretation

    def test_moderately_stringent(self):
        """Test interpretation for 70-90% rejection."""
        interpretation = _interpret_rejection_rate(0.75, 'bonferroni')

        assert 'moderately stringent' in interpretation.lower()

    def test_conservative(self):
        """Test interpretation for 30-70% rejection."""
        interpretation = _interpret_rejection_rate(0.50, 'bonferroni')

        assert 'conservative' in interpretation.lower()

    def test_minimal_impact(self):
        """Test interpretation for <30% rejection."""
        interpretation = _interpret_rejection_rate(0.20, 'bonferroni')

        assert 'minimal impact' in interpretation.lower()


# =============================================================================
# Tests for _estimate_statistical_power
# =============================================================================


class TestEstimateStatisticalPower:
    """Tests for power analysis."""

    def test_power_in_valid_range(self):
        """Test that power is always in [0, 1]."""
        power = _estimate_statistical_power(100, 0.05, effect_size=0.1)

        assert 0 <= power['statistical_power'] <= 1
        assert 0 <= power['type_ii_error_rate'] <= 1

    def test_power_plus_error_equals_one(self):
        """Test that power + type II error = 1."""
        power = _estimate_statistical_power(100, 0.05, effect_size=0.1)

        total = power['statistical_power'] + power['type_ii_error_rate']
        assert total == pytest.approx(1.0, rel=1e-6)

    def test_power_interpretation_categories(self):
        """Test power interpretation categories."""
        # High power scenario
        high_power = _estimate_statistical_power(10, 0.05, effect_size=0.5)
        # Low power scenario (many tests, small effect)
        low_power = _estimate_statistical_power(1000, 0.0001, effect_size=0.01)

        assert high_power['power_interpretation'] in ['HIGH', 'MODERATE', 'LOW']
        assert low_power['power_interpretation'] in ['HIGH', 'MODERATE', 'LOW']

    def test_effect_size_included(self):
        """Test that assumed effect size is included."""
        power = _estimate_statistical_power(100, 0.05, effect_size=0.15)

        assert power['effect_size_assumed'] == 0.15


# =============================================================================
# Tests for _build_bonferroni_results
# =============================================================================


class TestBuildBonferroniResults:
    """Tests for result builder."""

    def test_returns_correct_type(self, model_results_df):
        """Test that function returns MultipleTestingResults."""
        significant = model_results_df.head(5).copy()
        rejected = model_results_df.tail(45).copy()

        result = _build_bonferroni_results(
            alpha=0.05,
            corrected_alpha=0.001,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected,
            min_significant_models=1
        )

        assert isinstance(result, MultipleTestingResults)

    def test_method_is_bonferroni(self, model_results_df):
        """Test that method is correctly set to bonferroni."""
        significant = model_results_df.head(5).copy()
        rejected = model_results_df.tail(45).copy()

        result = _build_bonferroni_results(
            alpha=0.05,
            corrected_alpha=0.001,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected,
            min_significant_models=1
        )

        assert result.method == 'bonferroni'

    def test_alpha_values_preserved(self, model_results_df):
        """Test that original and corrected alpha are preserved."""
        significant = model_results_df.head(5).copy()
        rejected = model_results_df.tail(45).copy()

        result = _build_bonferroni_results(
            alpha=0.05,
            corrected_alpha=0.001,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected,
            min_significant_models=1
        )

        assert result.original_alpha == 0.05
        assert result.corrected_alpha == 0.001
        assert result.n_tests == 50

    def test_warns_when_too_few_significant(self, model_results_df):
        """Test warning when significant models below minimum."""
        significant = model_results_df.head(0).copy()  # No significant models
        rejected = model_results_df.copy()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _build_bonferroni_results(
                alpha=0.05,
                corrected_alpha=0.0001,
                n_tests=50,
                significant_models=significant,
                rejected_models=rejected,
                min_significant_models=3
            )

            # Should have warning about overly conservative
            assert len(w) >= 1
            assert "conservative" in str(w[0].message).lower()


# =============================================================================
# Tests for apply_bonferroni_correction (Public API)
# =============================================================================


class TestApplyBonferroniCorrection:
    """Tests for main public function."""

    def test_full_workflow(self, model_results_df):
        """Test complete Bonferroni correction workflow."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        assert isinstance(result, MultipleTestingResults)
        assert result.method == 'bonferroni'
        assert result.n_tests == len(model_results_df)
        assert len(result.significant_models) + len(result.rejected_models) == len(model_results_df)

    def test_corrected_alpha_less_than_original(self, model_results_df):
        """Test that corrected alpha is less than original."""
        result = apply_bonferroni_correction(model_results_df, 'aic', alpha=0.05)

        assert result.corrected_alpha < result.original_alpha

    def test_custom_alpha(self, model_results_df):
        """Test with custom alpha level."""
        result_05 = apply_bonferroni_correction(model_results_df, 'aic', alpha=0.05)
        result_01 = apply_bonferroni_correction(model_results_df, 'aic', alpha=0.01)

        # Stricter alpha = smaller corrected alpha
        assert result_01.corrected_alpha < result_05.corrected_alpha

    def test_correction_impact_included(self, model_results_df):
        """Test that correction impact is included."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        assert 'models_rejected' in result.correction_impact
        assert 'models_retained' in result.correction_impact
        assert 'correction_stringency' in result.correction_impact

    def test_statistical_power_included(self, model_results_df):
        """Test that statistical power is included."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        assert 'statistical_power' in result.statistical_power
        assert 'type_ii_error_rate' in result.statistical_power

    def test_empty_raises(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            apply_bonferroni_correction(pd.DataFrame(), 'aic')

    def test_missing_column_raises(self, model_results_df):
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError):
            apply_bonferroni_correction(model_results_df, 'nonexistent')

    def test_production_size(self, large_model_results):
        """Test with production-size dataset (793 models)."""
        result = apply_bonferroni_correction(large_model_results, 'aic')

        assert result.n_tests == 793
        assert result.corrected_alpha == pytest.approx(0.05 / 793, rel=1e-6)

    def test_min_significant_models(self, model_results_df):
        """Test min_significant_models parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_bonferroni_correction(
                model_results_df, 'aic', min_significant_models=100  # Impossible to achieve
            )

            # Should still return results
            assert isinstance(result, MultipleTestingResults)


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestBonferroniProperties:
    """Property-based tests for mathematical invariants."""

    def test_significant_subset_of_original(self, model_results_df):
        """Property: significant models ⊆ original models."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        # All significant model indices should be in original
        sig_indices = set(result.significant_models.index)
        orig_indices = set(model_results_df.index)
        assert sig_indices.issubset(orig_indices)

    def test_rejected_subset_of_original(self, model_results_df):
        """Property: rejected models ⊆ original models."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        rej_indices = set(result.rejected_models.index)
        orig_indices = set(model_results_df.index)
        assert rej_indices.issubset(orig_indices)

    def test_significant_and_rejected_partition(self, model_results_df):
        """Property: significant ∩ rejected = ∅ and significant ∪ rejected = original."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        sig_indices = set(result.significant_models.index)
        rej_indices = set(result.rejected_models.index)
        orig_indices = set(model_results_df.index)

        # Disjoint
        assert sig_indices.isdisjoint(rej_indices)

        # Union equals original
        assert sig_indices.union(rej_indices) == orig_indices

    def test_best_model_always_significant(self, model_results_df):
        """Property: model with minimum AIC is always significant."""
        result = apply_bonferroni_correction(model_results_df, 'aic')

        min_aic_idx = model_results_df['aic'].idxmin()
        assert min_aic_idx in result.significant_models.index

    def test_corrected_alpha_equals_formula(self, model_results_df):
        """Property: corrected_alpha = alpha / n_tests."""
        alpha = 0.05
        result = apply_bonferroni_correction(model_results_df, 'aic', alpha=alpha)

        expected = alpha / len(model_results_df)
        assert result.corrected_alpha == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("n_models", [10, 50, 100, 500])
    def test_correction_stringency_increases_with_n(self, n_models):
        """Property: more models = more stringent correction."""
        np.random.seed(42)
        df = pd.DataFrame({
            'aic': 100 + np.random.exponential(scale=5, size=n_models)
        })

        result = apply_bonferroni_correction(df, 'aic')

        # Corrected alpha should decrease with more models
        assert result.corrected_alpha == pytest.approx(0.05 / n_models, rel=1e-6)
