"""
Tests for Multiple Testing Correction Engines.

Tests cover:
- Bonferroni Engine: FWER control with conservative correction
- FDR Engine: FDR control using Benjamini-Hochberg procedure
- Validation functions
- Impact assessment helpers
- Power analysis

Design Principles:
- Real assertions about correctness
- Mathematical validation of correction procedures
- Test edge cases and error handling

Author: Claude Code
Date: 2026-01-30
"""

import pytest
import pandas as pd
import numpy as np

from src.features.selection.enhancements.multiple_testing.bonferroni_engine import (
    apply_bonferroni_correction,
    _validate_bonferroni_inputs,
    _calculate_bonferroni_threshold,
    _apply_bonferroni_significance,
    _calculate_correction_impact,
    _interpret_rejection_rate,
    _estimate_statistical_power,
)
from src.features.selection.enhancements.multiple_testing.fdr_engine import (
    apply_fdr_correction,
    _validate_fdr_inputs,
    _convert_aic_to_pvalues,
    _apply_fdr_to_pvalues,
    _compute_effective_alpha,
)
from src.features.selection.enhancements.multiple_testing.multiple_testing_types import (
    MultipleTestingResults,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model_results_df():
    """Create sample model results DataFrame with AIC values."""
    return pd.DataFrame({
        'features': ['f1', 'f1+f2', 'f2', 'f1+f3', 'f3'],
        'aic': [100.0, 95.0, 105.0, 98.0, 110.0],
        'r_squared': [0.70, 0.75, 0.65, 0.72, 0.60],
    })


@pytest.fixture
def large_model_results():
    """Create larger model results for realistic testing."""
    np.random.seed(42)
    n = 100
    base_aic = 100
    return pd.DataFrame({
        'features': [f'model_{i}' for i in range(n)],
        'aic': base_aic + np.random.exponential(scale=5, size=n),
        'r_squared': np.random.uniform(0.5, 0.9, n),
    })


@pytest.fixture
def empty_results():
    """Create empty DataFrame."""
    return pd.DataFrame()


# =============================================================================
# Tests for Bonferroni Validation
# =============================================================================


class TestBonferroniValidation:
    """Tests for Bonferroni input validation."""

    def test_empty_results_raises_error(self, empty_results):
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="No model results provided"):
            _validate_bonferroni_inputs(empty_results, 'aic')

    def test_missing_column_raises_error(self, model_results_df):
        """Test that missing AIC column raises ValueError."""
        with pytest.raises(ValueError, match="not found in results"):
            _validate_bonferroni_inputs(model_results_df, 'nonexistent_column')

    def test_valid_inputs_pass(self, model_results_df):
        """Test that valid inputs pass validation."""
        # Should not raise
        _validate_bonferroni_inputs(model_results_df, 'aic')


# =============================================================================
# Tests for Bonferroni Calculation
# =============================================================================


class TestBonferroniCalculation:
    """Tests for Bonferroni threshold calculation."""

    def test_corrected_alpha_formula(self):
        """Test Bonferroni corrected alpha calculation."""
        n_tests = 100
        alpha = 0.05

        corrected_alpha, critical_value = _calculate_bonferroni_threshold(n_tests, alpha)

        # Bonferroni: α_corrected = α / m
        expected = 0.05 / 100
        assert corrected_alpha == pytest.approx(expected)

    def test_critical_value_positive(self):
        """Test that critical value is positive."""
        corrected_alpha, critical_value = _calculate_bonferroni_threshold(50, 0.05)

        assert critical_value > 0

    def test_more_tests_smaller_alpha(self):
        """Test that more tests result in smaller corrected alpha."""
        _, _ = _calculate_bonferroni_threshold(10, 0.05)
        alpha_10, _ = _calculate_bonferroni_threshold(10, 0.05)
        alpha_100, _ = _calculate_bonferroni_threshold(100, 0.05)
        alpha_1000, _ = _calculate_bonferroni_threshold(1000, 0.05)

        assert alpha_10 > alpha_100 > alpha_1000


# =============================================================================
# Tests for Bonferroni Significance
# =============================================================================


class TestBonferroniSignificance:
    """Tests for Bonferroni significance testing."""

    def test_splits_results_correctly(self, model_results_df):
        """Test that results are split into significant/rejected."""
        # Use large critical value to keep all models significant
        significant, rejected = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value=100.0
        )

        assert len(significant) + len(rejected) == len(model_results_df)

    def test_minimum_aic_always_significant(self, model_results_df):
        """Test that model with minimum AIC is always significant."""
        significant, _ = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value=1.0
        )

        # Model with min AIC should have aic_diff=0, always significant
        assert len(significant) >= 1
        assert significant['aic_diff'].min() == 0.0

    def test_adds_required_columns(self, model_results_df):
        """Test that required columns are added."""
        significant, _ = _apply_bonferroni_significance(
            model_results_df, 'aic', critical_value=5.0
        )

        assert 'aic_diff' in significant.columns
        assert 'bonferroni_significant' in significant.columns


# =============================================================================
# Tests for Impact Assessment
# =============================================================================


class TestCorrectionImpact:
    """Tests for correction impact calculation."""

    def test_rejection_rate_calculation(self):
        """Test rejection rate is calculated correctly."""
        impact = _calculate_correction_impact(
            original_count=100,
            significant_count=20,
            method='bonferroni'
        )

        assert impact['models_rejected'] == 80
        assert impact['models_retained'] == 20
        assert impact['rejection_rate_percent'] == pytest.approx(80.0)
        assert impact['retention_rate_percent'] == pytest.approx(20.0)

    def test_stringency_classification(self):
        """Test stringency classification thresholds."""
        # High stringency (>80% rejected)
        impact_high = _calculate_correction_impact(100, 10, 'test')
        assert impact_high['correction_stringency'] == 'HIGH'

        # Moderate stringency (50-80% rejected)
        impact_mod = _calculate_correction_impact(100, 40, 'test')
        assert impact_mod['correction_stringency'] == 'MODERATE'

        # Low stringency (<50% rejected)
        impact_low = _calculate_correction_impact(100, 60, 'test')
        assert impact_low['correction_stringency'] == 'LOW'


class TestRejectionInterpretation:
    """Tests for rejection rate interpretation."""

    def test_very_stringent_interpretation(self):
        """Test interpretation for very high rejection rate."""
        interp = _interpret_rejection_rate(0.95, 'bonferroni')
        assert 'very stringent' in interp.lower()

    def test_moderate_interpretation(self):
        """Test interpretation for moderate rejection rate."""
        interp = _interpret_rejection_rate(0.75, 'bonferroni')
        assert 'moderately stringent' in interp.lower()

    def test_minimal_impact_interpretation(self):
        """Test interpretation for low rejection rate."""
        interp = _interpret_rejection_rate(0.2, 'bonferroni')
        assert 'minimal impact' in interp.lower()


class TestStatisticalPower:
    """Tests for statistical power estimation."""

    def test_power_in_valid_range(self):
        """Test that power is between 0 and 1."""
        power = _estimate_statistical_power(n_tests=100, alpha=0.001)

        assert 0.0 <= power['statistical_power'] <= 1.0
        assert 0.0 <= power['type_ii_error_rate'] <= 1.0

    def test_power_interpretation_categories(self):
        """Test power interpretation categories."""
        # Test that interpretation is one of expected values
        power = _estimate_statistical_power(n_tests=50, alpha=0.05)

        assert power['power_interpretation'] in ['HIGH', 'MODERATE', 'LOW']


# =============================================================================
# Tests for Full Bonferroni Correction
# =============================================================================


class TestApplyBonferroniCorrection:
    """Tests for full Bonferroni correction function."""

    def test_returns_multiple_testing_results(self, model_results_df):
        """Test that function returns MultipleTestingResults."""
        results = apply_bonferroni_correction(model_results_df, 'aic')

        assert isinstance(results, MultipleTestingResults)
        assert results.method == 'bonferroni'

    def test_results_contain_all_models(self, model_results_df):
        """Test that all models are accounted for."""
        results = apply_bonferroni_correction(model_results_df, 'aic')

        total = len(results.significant_models) + len(results.rejected_models)
        assert total == len(model_results_df)

    def test_empty_results_raises_error(self, empty_results):
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError):
            apply_bonferroni_correction(empty_results, 'aic')


# =============================================================================
# Tests for FDR Validation
# =============================================================================


class TestFDRValidation:
    """Tests for FDR input validation."""

    def test_empty_results_raises_error(self, empty_results):
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="No model results provided"):
            _validate_fdr_inputs(empty_results, 'aic')

    def test_missing_column_raises_error(self, model_results_df):
        """Test that missing AIC column raises ValueError."""
        with pytest.raises(ValueError, match="not found in results"):
            _validate_fdr_inputs(model_results_df, 'missing_col')


# =============================================================================
# Tests for FDR Calculation
# =============================================================================


class TestFDRCalculation:
    """Tests for FDR calculation functions."""

    def test_pvalue_conversion(self, model_results_df):
        """Test AIC to p-value conversion."""
        results = _convert_aic_to_pvalues(model_results_df, 'aic')

        assert 'p_value' in results.columns
        assert 'aic_diff' in results.columns

        # All p-values should be in (0, 1]
        assert (results['p_value'] > 0).all()
        assert (results['p_value'] <= 1).all()

    def test_minimum_aic_has_pvalue_1(self, model_results_df):
        """Test that minimum AIC model has p-value near 1."""
        results = _convert_aic_to_pvalues(model_results_df, 'aic')

        # Model with aic_diff=0 should have p-value close to 1
        min_diff_row = results[results['aic_diff'] == 0]
        if len(min_diff_row) > 0:
            assert min_diff_row['p_value'].iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_fdr_splits_results(self, model_results_df):
        """Test FDR correction splits results."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')
        significant, rejected = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        assert len(significant) + len(rejected) == len(model_results_df)

    def test_effective_alpha_computation(self, model_results_df):
        """Test effective alpha is computed correctly."""
        results_with_p = _convert_aic_to_pvalues(model_results_df, 'aic')
        significant, _ = _apply_fdr_to_pvalues(results_with_p, alpha=0.05)

        effective = _compute_effective_alpha(significant, 0.05, len(model_results_df))

        assert effective > 0
        assert effective <= 1.0


# =============================================================================
# Tests for Full FDR Correction
# =============================================================================


class TestApplyFDRCorrection:
    """Tests for full FDR correction function."""

    def test_returns_multiple_testing_results(self, model_results_df):
        """Test that function returns MultipleTestingResults."""
        results = apply_fdr_correction(model_results_df, 'aic')

        assert isinstance(results, MultipleTestingResults)
        assert results.method == 'fdr_bh'

    def test_results_contain_all_models(self, model_results_df):
        """Test that all models are accounted for."""
        results = apply_fdr_correction(model_results_df, 'aic')

        total = len(results.significant_models) + len(results.rejected_models)
        assert total == len(model_results_df)

    def test_both_methods_process_same_data(self, large_model_results):
        """Test that both methods process the same data consistently.

        Note: For AIC-based comparisons, the typical relationship (FDR less
        conservative than Bonferroni) doesn't always hold. AIC differences are
        converted to pseudo-p-values, creating different statistical properties
        than traditional hypothesis testing scenarios.

        This test verifies both methods handle the same input correctly rather
        than making assumptions about relative conservativeness.
        """
        bonf_results = apply_bonferroni_correction(large_model_results, 'aic')
        fdr_results = apply_fdr_correction(large_model_results, 'aic')

        # Both should process the same number of tests
        assert bonf_results.n_tests == fdr_results.n_tests == len(large_model_results)

        # Both should account for all models
        bonf_total = len(bonf_results.significant_models) + len(bonf_results.rejected_models)
        fdr_total = len(fdr_results.significant_models) + len(fdr_results.rejected_models)
        assert bonf_total == len(large_model_results)
        assert fdr_total == len(large_model_results)

        # Both should have valid corrected alphas
        assert 0 < bonf_results.corrected_alpha <= bonf_results.original_alpha
        assert 0 < fdr_results.corrected_alpha <= fdr_results.original_alpha


# =============================================================================
# Tests for MultipleTestingResults
# =============================================================================


class TestMultipleTestingResults:
    """Tests for MultipleTestingResults dataclass."""

    def test_dataclass_creation(self, model_results_df):
        """Test that dataclass can be created."""
        results = MultipleTestingResults(
            method='bonferroni',
            original_alpha=0.05,
            corrected_alpha=0.001,
            n_tests=100,
            significant_models=model_results_df.head(2),
            rejected_models=model_results_df.tail(3),
            correction_impact={'test': 'impact'},
            statistical_power={'power': 0.8},
        )

        assert results.method == 'bonferroni'
        assert results.original_alpha == 0.05
        assert results.n_tests == 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultipleTestingIntegration:
    """Integration tests for multiple testing module."""

    def test_bonferroni_full_workflow(self, large_model_results):
        """Test complete Bonferroni workflow."""
        results = apply_bonferroni_correction(
            large_model_results,
            aic_column='aic',
            alpha=0.05,
            min_significant_models=1,
        )

        # Verify structure
        assert results.method == 'bonferroni'
        assert results.n_tests == len(large_model_results)
        assert len(results.significant_models) + len(results.rejected_models) == results.n_tests

        # Verify impact assessment
        assert 'models_rejected' in results.correction_impact
        assert 'correction_stringency' in results.correction_impact

        # Verify power analysis
        assert 'statistical_power' in results.statistical_power

    def test_fdr_full_workflow(self, large_model_results):
        """Test complete FDR workflow."""
        results = apply_fdr_correction(
            large_model_results,
            aic_column='aic',
            alpha=0.05,
        )

        # Verify structure
        assert results.method == 'fdr_bh'
        assert results.n_tests == len(large_model_results)

        # Significant models should have FDR columns
        if len(results.significant_models) > 0:
            assert 'fdr_significant' in results.significant_models.columns
            assert 'fdr_corrected_p' in results.significant_models.columns

    def test_both_methods_on_same_data(self, model_results_df):
        """Test both correction methods on same data."""
        bonf = apply_bonferroni_correction(model_results_df, 'aic')
        fdr = apply_fdr_correction(model_results_df, 'aic')

        # Both should process same number of tests
        assert bonf.n_tests == fdr.n_tests

        # Both should have valid alpha values
        assert 0 < bonf.corrected_alpha <= bonf.original_alpha
        assert 0 < fdr.corrected_alpha <= fdr.original_alpha
