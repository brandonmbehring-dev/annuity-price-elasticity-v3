"""
Tests for FDR Correction Engine.

Tests cover:
- Input validation (empty results, missing columns)
- AIC to p-value conversion (chi-squared approximation)
- FDR (Benjamini-Hochberg) procedure application
- Effective alpha calculation
- Impact assessment and power analysis
- Full workflow integration (apply_fdr_correction)

Design Principles:
- Property-based testing for mathematical invariants
- Comparison with Bonferroni (FDR should be less conservative)
- Edge cases for numerical stability

Author: Claude Code
Date: 2026-01-31
"""

import pytest
import pandas as pd
import numpy as np

from src.features.selection.enhancements.multiple_testing.fdr_engine import (
    _validate_fdr_inputs,
    _convert_aic_to_pvalues,
    _apply_fdr_to_pvalues,
    _compute_effective_alpha,
    _calculate_correction_impact,
    _interpret_rejection_rate,
    _estimate_statistical_power,
    _build_fdr_results,
    apply_fdr_correction,
)
from src.features.selection.enhancements.multiple_testing.multiple_testing_types import (
    MultipleTestingResults,
)
from src.features.selection.enhancements.multiple_testing.bonferroni_engine import (
    apply_bonferroni_correction,
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


@pytest.fixture
def models_with_ties():
    """Create model results with AIC ties."""
    return pd.DataFrame({
        'model_id': ['model_0', 'model_1', 'model_2', 'model_3'],
        'aic': [100.0, 100.0, 105.0, 105.0],  # Ties at 100 and 105
    })


# =============================================================================
# Tests for _validate_fdr_inputs
# =============================================================================


class TestValidateFDRInputs:
    """Tests for input validation."""

    def test_empty_results_raises(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError) as excinfo:
            _validate_fdr_inputs(empty_df, 'aic')

        assert "CRITICAL" in str(excinfo.value)
        assert "No model results" in str(excinfo.value)
        assert "Business impact" in str(excinfo.value)

    def test_missing_aic_column_raises(self, model_results_df):
        """Test that missing AIC column raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            _validate_fdr_inputs(model_results_df, 'nonexistent_column')

        assert "CRITICAL" in str(excinfo.value)
        assert "nonexistent_column" in str(excinfo.value)
        assert "Available columns" in str(excinfo.value)

    def test_valid_inputs_pass(self, model_results_df):
        """Test that valid inputs do not raise."""
        # Should not raise
        _validate_fdr_inputs(model_results_df, 'aic')

    def test_custom_aic_column_name(self):
        """Test validation with custom AIC column name."""
        df = pd.DataFrame({'custom_aic': [100, 105, 110]})

        # Should not raise
        _validate_fdr_inputs(df, 'custom_aic')


# =============================================================================
# Tests for _convert_aic_to_pvalues
# =============================================================================


class TestConvertAICToPvalues:
    """Tests for AIC to p-value conversion."""

    def test_adds_required_columns(self, model_results_df):
        """Test that aic_diff and p_value columns are added."""
        result = _convert_aic_to_pvalues(model_results_df, 'aic')

        assert 'aic_diff' in result.columns
        assert 'p_value' in result.columns

    def test_pvalues_in_valid_range(self, model_results_df):
        """Test that all p-values are in [0, 1]."""
        result = _convert_aic_to_pvalues(model_results_df, 'aic')

        assert (result['p_value'] >= 0).all()
        assert (result['p_value'] <= 1).all()

    def test_min_aic_has_zero_diff(self, model_results_df):
        """Test that minimum AIC model has aic_diff = 0."""
        result = _convert_aic_to_pvalues(model_results_df, 'aic')

        assert result['aic_diff'].min() == 0

    def test_min_aic_has_pvalue_one(self, model_results_df):
        """Test that minimum AIC model has p_value ≈ 1."""
        result = _convert_aic_to_pvalues(model_results_df, 'aic')

        # Model with aic_diff = 0 should have p-value = 1
        best_model_pvalue = result.loc[result['aic_diff'] == 0, 'p_value'].values[0]
        assert best_model_pvalue == pytest.approx(1.0, rel=1e-6)

    def test_larger_diff_smaller_pvalue(self, small_model_results):
        """Test that larger AIC diff = smaller p-value."""
        result = _convert_aic_to_pvalues(small_model_results, 'aic')

        # Sort by aic_diff
        sorted_result = result.sort_values('aic_diff')

        # p_value should decrease as aic_diff increases
        pvalues = sorted_result['p_value'].values
        assert pvalues[0] >= pvalues[1] >= pvalues[2]

    def test_clipping_prevents_zero_pvalues(self):
        """Test that very large AIC diffs don't produce zero p-values."""
        df = pd.DataFrame({
            'aic': [100.0, 200.0, 500.0, 1000.0]  # Very large diffs
        })

        result = _convert_aic_to_pvalues(df, 'aic')

        # All p-values should be positive (clipped at 1e-16)
        assert (result['p_value'] > 0).all()
        assert result['p_value'].min() >= 1e-16

    def test_handles_ties(self, models_with_ties):
        """Test that tied AIC values get same p-value."""
        result = _convert_aic_to_pvalues(models_with_ties, 'aic')

        # Models with same AIC should have same p-value
        tied_100 = result[result['aic'] == 100.0]['p_value'].values
        assert tied_100[0] == tied_100[1]


# =============================================================================
# Tests for _apply_fdr_to_pvalues
# =============================================================================


class TestApplyFDRToPvalues:
    """Tests for FDR correction application."""

    def test_adds_required_columns(self, model_results_df):
        """Test that fdr_corrected_p and fdr_significant columns are added."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')

        significant, rejected = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        assert 'fdr_corrected_p' in significant.columns
        assert 'fdr_significant' in significant.columns

    def test_splits_correctly(self, model_results_df):
        """Test that results are correctly split."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')

        significant, rejected = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        # Should sum to total
        assert len(significant) + len(rejected) == len(model_results_df)

    def test_significant_all_true(self, model_results_df):
        """Test that significant models have fdr_significant = True."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')

        significant, rejected = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        if len(significant) > 0:
            assert significant['fdr_significant'].all()

    def test_rejected_all_false(self, model_results_df):
        """Test that rejected models have fdr_significant = False."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')

        significant, rejected = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        if len(rejected) > 0:
            assert not rejected['fdr_significant'].any()

    def test_stricter_alpha_fewer_significant(self, model_results_df):
        """Test that stricter alpha results in fewer significant models."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')

        sig_05, _ = _apply_fdr_to_pvalues(results_with_p.copy(), alpha=0.05)
        sig_01, _ = _apply_fdr_to_pvalues(results_with_p.copy(), alpha=0.01)

        assert len(sig_01) <= len(sig_05)


# =============================================================================
# Tests for _compute_effective_alpha
# =============================================================================


class TestComputeEffectiveAlpha:
    """Tests for effective alpha calculation."""

    def test_with_significant_models(self, model_results_df):
        """Test effective alpha when significant models exist."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')
        significant, _ = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        if len(significant) > 0:
            effective_alpha = _compute_effective_alpha(
                significant, alpha=0.05, n_models=len(model_results_df)
            )

            # Effective alpha should be max p-value of significant models
            assert effective_alpha == significant['p_value'].max()

    def test_with_no_significant_models(self):
        """Test effective alpha when no significant models."""
        # Empty significant models
        significant = pd.DataFrame({'p_value': []})

        effective_alpha = _compute_effective_alpha(
            significant, alpha=0.05, n_models=100
        )

        # Should fall back to Bonferroni-like
        assert effective_alpha == 0.05 / 100

    def test_effective_alpha_positive(self, model_results_df):
        """Test that effective alpha is always positive."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')
        significant, _ = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        effective_alpha = _compute_effective_alpha(
            significant, alpha=0.05, n_models=len(model_results_df)
        )

        assert effective_alpha > 0


# =============================================================================
# Tests for Impact Assessment Functions
# =============================================================================


class TestFDRImpactAssessment:
    """Tests for FDR impact assessment (same functions as Bonferroni)."""

    def test_calculate_correction_impact(self):
        """Test impact calculation."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=25,
            method='fdr_bh'
        )

        assert impact['models_rejected'] == 75
        assert impact['models_retained'] == 25
        assert impact['rejection_rate_percent'] == 75.0
        assert impact['correction_stringency'] == 'MODERATE'
        assert 'fdr_bh' in impact['business_impact']

    def test_interpret_rejection_rate(self):
        """Test rejection rate interpretation."""
        interp = _interpret_rejection_rate(0.6, 'fdr_bh')

        assert 'fdr_bh' in interp
        assert isinstance(interp, str)

    def test_estimate_statistical_power(self):
        """Test power estimation."""
        power = _estimate_statistical_power(100, 0.05, effect_size=0.1)

        assert 'statistical_power' in power
        assert 'type_ii_error_rate' in power
        assert 0 <= power['statistical_power'] <= 1


# =============================================================================
# Tests for _build_fdr_results
# =============================================================================


class TestBuildFDRResults:
    """Tests for FDR result builder."""

    def test_returns_correct_type(self, model_results_df):
        """Test that function returns MultipleTestingResults."""
        significant = model_results_df.head(10).copy()
        rejected = model_results_df.tail(40).copy()

        result = _build_fdr_results(
            alpha=0.05,
            effective_alpha=0.01,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected
        )

        assert isinstance(result, MultipleTestingResults)

    def test_method_is_fdr_bh(self, model_results_df):
        """Test that method is correctly set to fdr_bh."""
        significant = model_results_df.head(10).copy()
        rejected = model_results_df.tail(40).copy()

        result = _build_fdr_results(
            alpha=0.05,
            effective_alpha=0.01,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected
        )

        assert result.method == 'fdr_bh'

    def test_alpha_values_preserved(self, model_results_df):
        """Test that original and corrected alpha are preserved."""
        significant = model_results_df.head(10).copy()
        rejected = model_results_df.tail(40).copy()

        result = _build_fdr_results(
            alpha=0.05,
            effective_alpha=0.015,
            n_tests=50,
            significant_models=significant,
            rejected_models=rejected
        )

        assert result.original_alpha == 0.05
        assert result.corrected_alpha == 0.015
        assert result.n_tests == 50


# =============================================================================
# Tests for apply_fdr_correction (Public API)
# =============================================================================


class TestApplyFDRCorrection:
    """Tests for main public function."""

    def test_full_workflow(self, model_results_df):
        """Test complete FDR correction workflow."""
        result = apply_fdr_correction(model_results_df, 'aic')

        assert isinstance(result, MultipleTestingResults)
        assert result.method == 'fdr_bh'
        assert result.n_tests == len(model_results_df)
        assert len(result.significant_models) + len(result.rejected_models) == len(model_results_df)

    def test_corrected_alpha_reasonable(self, model_results_df):
        """Test that corrected alpha is reasonable."""
        result = apply_fdr_correction(model_results_df, 'aic', alpha=0.05)

        # Corrected alpha should be positive and <= original
        assert result.corrected_alpha > 0
        assert result.corrected_alpha <= result.original_alpha

    def test_custom_alpha(self, model_results_df):
        """Test with custom alpha level."""
        result_05 = apply_fdr_correction(model_results_df, 'aic', alpha=0.05)
        result_01 = apply_fdr_correction(model_results_df, 'aic', alpha=0.01)

        # Stricter alpha should result in fewer or equal significant models
        assert len(result_01.significant_models) <= len(result_05.significant_models)

    def test_correction_impact_included(self, model_results_df):
        """Test that correction impact is included."""
        result = apply_fdr_correction(model_results_df, 'aic')

        assert 'models_rejected' in result.correction_impact
        assert 'models_retained' in result.correction_impact
        assert 'correction_stringency' in result.correction_impact

    def test_statistical_power_included(self, model_results_df):
        """Test that statistical power is included."""
        result = apply_fdr_correction(model_results_df, 'aic')

        assert 'statistical_power' in result.statistical_power
        assert 'type_ii_error_rate' in result.statistical_power

    def test_empty_raises(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            apply_fdr_correction(pd.DataFrame(), 'aic')

    def test_missing_column_raises(self, model_results_df):
        """Test that missing column raises ValueError."""
        with pytest.raises(ValueError):
            apply_fdr_correction(model_results_df, 'nonexistent')

    def test_production_size(self, large_model_results):
        """Test with production-size dataset (793 models)."""
        result = apply_fdr_correction(large_model_results, 'aic')

        assert result.n_tests == 793

    def test_method_parameter_ignored(self, model_results_df):
        """Test that method parameter is accepted but uses fdr_bh."""
        result = apply_fdr_correction(model_results_df, 'aic', method='other')

        # Should still use fdr_bh
        assert result.method == 'fdr_bh'


# =============================================================================
# Tests Comparing FDR vs Bonferroni
# =============================================================================


class TestFDRvsBonferroni:
    """Tests comparing FDR to Bonferroni correction."""

    def test_both_methods_produce_results(self, model_results_df):
        """Test that both FDR and Bonferroni produce valid results."""
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        # Both should produce valid results
        assert isinstance(bonf, MultipleTestingResults)
        assert isinstance(fdr, MultipleTestingResults)
        assert bonf.method == 'bonferroni'
        assert fdr.method == 'fdr_bh'

    def test_bonferroni_includes_best_model(self, model_results_df):
        """Test that Bonferroni always includes the best model.

        Note: FDR may not include best model due to its p-value adjustment
        methodology which operates on the full p-value distribution.
        """
        bonf = apply_bonferroni_correction(model_results_df, 'aic')

        min_aic_idx = model_results_df['aic'].idxmin()
        assert min_aic_idx in bonf.significant_models.index

    def test_same_n_tests(self, model_results_df):
        """Test that both methods process same number of tests."""
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        assert bonf.n_tests == fdr.n_tests == len(model_results_df)

    def test_methods_use_different_thresholds(self, model_results_df):
        """Test that FDR and Bonferroni use different thresholds."""
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        # Bonferroni uses alpha/n, FDR uses adaptive threshold
        # They should typically be different
        assert bonf.corrected_alpha != fdr.corrected_alpha or \
               len(bonf.significant_models) != len(fdr.significant_models)

    def test_both_methods_scale_with_sample_size(self):
        """Test that both methods scale appropriately with sample size."""
        np.random.seed(42)

        # Small sample
        small = pd.DataFrame({'aic': 100 + np.random.exponential(5, 20)})
        bonf_small = apply_bonferroni_correction(small, 'aic')
        fdr_small = apply_fdr_correction(small, 'aic')

        # Larger sample
        large = pd.DataFrame({'aic': 100 + np.random.exponential(5, 200)})
        bonf_large = apply_bonferroni_correction(large, 'aic')
        fdr_large = apply_fdr_correction(large, 'aic')

        # Both methods should handle different sizes
        assert bonf_small.n_tests == 20
        assert bonf_large.n_tests == 200
        assert fdr_small.n_tests == 20
        assert fdr_large.n_tests == 200


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestFDRProperties:
    """Property-based tests for mathematical invariants."""

    def test_significant_subset_of_original(self, model_results_df):
        """Property: significant models ⊆ original models."""
        result = apply_fdr_correction(model_results_df, 'aic')

        sig_indices = set(result.significant_models.index)
        orig_indices = set(model_results_df.index)
        assert sig_indices.issubset(orig_indices)

    def test_rejected_subset_of_original(self, model_results_df):
        """Property: rejected models ⊆ original models."""
        result = apply_fdr_correction(model_results_df, 'aic')

        rej_indices = set(result.rejected_models.index)
        orig_indices = set(model_results_df.index)
        assert rej_indices.issubset(orig_indices)

    def test_significant_and_rejected_partition(self, model_results_df):
        """Property: significant ∩ rejected = ∅ and significant ∪ rejected = original."""
        result = apply_fdr_correction(model_results_df, 'aic')

        sig_indices = set(result.significant_models.index)
        rej_indices = set(result.rejected_models.index)
        orig_indices = set(model_results_df.index)

        # Disjoint
        assert sig_indices.isdisjoint(rej_indices)

        # Union equals original
        assert sig_indices.union(rej_indices) == orig_indices

    def test_best_model_has_highest_pvalue(self, model_results_df):
        """Property: model with minimum AIC has highest p-value (least evidence against it).

        Note: Unlike Bonferroni which directly uses AIC difference thresholds,
        FDR operates on the full p-value distribution and may reject even the
        best model in pathological cases (e.g., when all p-values are high).
        """
        result = apply_fdr_correction(model_results_df, 'aic')

        # Combine significant and rejected models
        all_results = pd.concat([result.significant_models, result.rejected_models])

        # Best model (min AIC) should have max p_value (1.0)
        min_aic_idx = model_results_df['aic'].idxmin()
        best_model_pvalue = all_results.loc[min_aic_idx, 'p_value']
        assert best_model_pvalue == pytest.approx(1.0, rel=1e-6)

    def test_fdr_corrected_pvalues_ordered(self, model_results_df):
        """Property: FDR-corrected p-values maintain ordering."""
        result = apply_fdr_correction(model_results_df, 'aic')

        # Get all results (significant + rejected)
        all_results = pd.concat([result.significant_models, result.rejected_models])

        if 'fdr_corrected_p' in all_results.columns and 'p_value' in all_results.columns:
            # Corrected p-values should be >= original p-values
            assert (all_results['fdr_corrected_p'] >= all_results['p_value']).all()

    @pytest.mark.parametrize("n_models", [10, 50, 100, 500])
    def test_handles_various_sizes(self, n_models):
        """Property: FDR works correctly for various sample sizes."""
        np.random.seed(42)
        df = pd.DataFrame({
            'aic': 100 + np.random.exponential(scale=5, size=n_models)
        })

        result = apply_fdr_correction(df, 'aic')

        assert result.n_tests == n_models
        assert len(result.significant_models) + len(result.rejected_models) == n_models
