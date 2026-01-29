"""
Tests for Bootstrap Engine - Stability Analysis Module.

Tests cover:
- run_bootstrap_stability: Main entry point for bootstrap analysis
- calculate_bootstrap_metrics: Single model bootstrap evaluation
- assess_model_stability: Stability classification

Design Principles:
- Real assertions about statistical correctness
- Test happy path + error cases + edge cases
- Use small n_samples for fast test execution

Note: Bootstrap tests use n_samples=30 for speed while maintaining
statistical validity (>30 samples for central limit theorem).

Author: Claude Code
Date: 2026-01-23
"""

import pytest
import pandas as pd
import numpy as np
from typing import List

from src.features.selection.engines.bootstrap_engine import (
    run_bootstrap_stability,
    calculate_bootstrap_metrics,
    assess_model_stability,
)
from src.features.selection_types import BootstrapResult


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Create simple regression data for bootstrap testing."""
    np.random.seed(42)
    n = 100

    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    noise = np.random.randn(n) * 0.5

    y = 2.0 * x1 + 1.0 * x2 + noise

    return pd.DataFrame({
        'target': y,
        'feature_1': x1,
        'feature_2': x2,
    })


@pytest.fixture
def valid_models_df():
    """Create valid models DataFrame for bootstrap analysis."""
    return pd.DataFrame([
        {
            'features': 'feature_1 + feature_2',
            'aic': 100.0,
            'r_squared': 0.85,
            'converged': True,
        },
        {
            'features': 'feature_1',
            'aic': 110.0,
            'r_squared': 0.70,
            'converged': True,
        },
    ])


@pytest.fixture
def bootstrap_config_enabled():
    """Create enabled bootstrap config with minimal samples for speed."""
    return {
        'enabled': True,
        'n_samples': 30,  # Minimal for testing
        'models_to_analyze': 2,
    }


@pytest.fixture
def bootstrap_config_disabled():
    """Create disabled bootstrap config."""
    return {
        'enabled': False,
        'n_samples': 30,
        'models_to_analyze': 2,
    }


@pytest.fixture
def noisy_data():
    """Create data with high variance (unstable model)."""
    np.random.seed(123)
    n = 80

    x1 = np.random.randn(n)
    noise = np.random.randn(n) * 5.0  # High noise

    y = 0.5 * x1 + noise  # Weak signal

    return pd.DataFrame({
        'target': y,
        'feature_1': x1,
    })


# =============================================================================
# Tests for assess_model_stability
# =============================================================================

class TestAssessModelStability:
    """Test suite for assess_model_stability function."""

    def test_stable_assessment(self):
        """Test STABLE assessment for low CV values."""
        result = assess_model_stability(
            aic_cv=0.003,  # Very low
            r2_cv=0.05,    # Very low
            successful_fits=95,
            total_attempts=100
        )

        assert result == "STABLE"

    def test_moderate_assessment(self):
        """Test MODERATE assessment for medium CV values."""
        result = assess_model_stability(
            aic_cv=0.007,  # Medium
            r2_cv=0.15,    # Medium
            successful_fits=90,
            total_attempts=100
        )

        assert result == "MODERATE"

    def test_unstable_assessment(self):
        """Test UNSTABLE assessment for high CV values."""
        result = assess_model_stability(
            aic_cv=0.02,   # High
            r2_cv=0.3,     # High
            successful_fits=85,
            total_attempts=100
        )

        assert result == "UNSTABLE"

    def test_failed_assessment_low_success_rate(self):
        """Test FAILED assessment for low success rate."""
        result = assess_model_stability(
            aic_cv=0.001,  # Would be stable
            r2_cv=0.01,
            successful_fits=40,  # < 50%
            total_attempts=100
        )

        assert result == "FAILED"

    def test_boundary_stable_moderate(self):
        """Test boundary between STABLE and MODERATE."""
        # Just below STABLE threshold
        stable = assess_model_stability(
            aic_cv=0.004,
            r2_cv=0.09,
            successful_fits=90,
            total_attempts=100
        )

        # Just above STABLE threshold
        moderate = assess_model_stability(
            aic_cv=0.006,
            r2_cv=0.11,
            successful_fits=90,
            total_attempts=100
        )

        assert stable == "STABLE"
        assert moderate == "MODERATE"

    def test_zero_attempts_returns_failed(self):
        """Test zero attempts returns FAILED."""
        result = assess_model_stability(
            aic_cv=0.001,
            r2_cv=0.01,
            successful_fits=0,
            total_attempts=0
        )

        assert result == "FAILED"


# =============================================================================
# Tests for calculate_bootstrap_metrics
# =============================================================================

class TestCalculateBootstrapMetrics:
    """Test suite for calculate_bootstrap_metrics function."""

    def test_returns_bootstrap_result(self, simple_data):
        """Verify function returns BootstrapResult."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=30
        )

        assert isinstance(result, BootstrapResult)

    def test_bootstrap_aics_populated(self, simple_data):
        """Verify bootstrap AICs are populated."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.70,
            n_samples=30
        )

        assert len(result.bootstrap_aics) > 0
        assert len(result.bootstrap_aics) >= 15  # At least 50% success

    def test_bootstrap_r2_values_in_range(self, simple_data):
        """Verify bootstrap R-squared values are reasonable."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=30
        )

        # R² values should be between -inf and 1 (can be negative for bad fits)
        for r2 in result.bootstrap_r2_values:
            assert r2 <= 1.0
            assert np.isfinite(r2)

    def test_stability_coefficients_positive(self, simple_data):
        """Verify stability coefficients are positive."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=30
        )

        assert result.aic_stability_coefficient >= 0
        assert result.r2_stability_coefficient >= 0

    def test_confidence_intervals_exist(self, simple_data):
        """Verify confidence intervals are calculated."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.70,
            n_samples=30
        )

        assert 'confidence_intervals' in dir(result)
        ci = result.confidence_intervals

        # Should have multiple confidence levels
        assert len(ci) > 0

        # Each CI should have aic_lower, aic_upper
        for level, values in ci.items():
            assert 'aic_lower' in values
            assert 'aic_upper' in values
            assert values['aic_lower'] <= values['aic_upper']

    def test_stability_assessment_assigned(self, simple_data):
        """Verify stability assessment is assigned."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=30
        )

        assert result.stability_assessment in [
            "STABLE", "MODERATE", "UNSTABLE", "FAILED"
        ]

    def test_successful_fits_tracked(self, simple_data):
        """Verify successful fits are tracked."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.70,
            n_samples=30
        )

        assert result.successful_fits > 0
        assert result.total_attempts == 30
        assert result.successful_fits <= result.total_attempts

    def test_model_features_stored(self, simple_data):
        """Verify model features are stored correctly."""
        features = ['feature_1', 'feature_2']
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=features,
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=30
        )

        assert result.model_features == 'feature_1 + feature_2'

    # Error cases

    def test_missing_feature_raises_error(self, simple_data):
        """Test that missing feature raises error."""
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            calculate_bootstrap_metrics(
                data=simple_data,
                model_features=['nonexistent_feature'],
                target_variable='target',
                original_aic=100.0,
                original_r2=0.70,
                n_samples=30
            )


# =============================================================================
# Tests for run_bootstrap_stability
# =============================================================================

class TestRunBootstrapStability:
    """Test suite for run_bootstrap_stability function."""

    def test_returns_list_of_results(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Verify function returns list of BootstrapResult."""
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target'
        )

        assert isinstance(results, list)
        assert all(isinstance(r, BootstrapResult) for r in results)

    def test_correct_number_of_models_analyzed(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Verify correct number of models are analyzed."""
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target'
        )

        expected_count = min(
            bootstrap_config_enabled['models_to_analyze'],
            len(valid_models_df)
        )
        assert len(results) == expected_count

    def test_disabled_config_returns_empty(
        self, simple_data, valid_models_df, bootstrap_config_disabled
    ):
        """Verify disabled config returns empty list."""
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_disabled,
            target_variable='target'
        )

        assert results == []

    def test_analyzes_top_models_by_order(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Verify models are analyzed in order from DataFrame."""
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target'
        )

        # First result should be for first model (feature_1 + feature_2)
        assert 'feature_1' in results[0].model_features
        assert 'feature_2' in results[0].model_features

    def test_handles_single_model(self, simple_data):
        """Test with single model in DataFrame."""
        single_model_df = pd.DataFrame([{
            'features': 'feature_1',
            'aic': 110.0,
            'r_squared': 0.70,
            'converged': True,
        }])

        config = {'enabled': True, 'n_samples': 30, 'models_to_analyze': 5}

        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=single_model_df,
            config=config,
            target_variable='target'
        )

        assert len(results) == 1

    def test_models_to_analyze_caps_at_available(self, simple_data, valid_models_df):
        """Verify models_to_analyze is capped at available models."""
        config = {'enabled': True, 'n_samples': 30, 'models_to_analyze': 100}

        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=config,
            target_variable='target'
        )

        # Should only analyze 2 models (all available)
        assert len(results) == len(valid_models_df)


# =============================================================================
# Tests for Statistical Properties
# =============================================================================

class TestBootstrapStatisticalProperties:
    """Test statistical properties of bootstrap analysis."""

    def test_bootstrap_aic_mean_near_original(self, simple_data):
        """Verify bootstrap AIC mean is near original AIC."""
        # First fit original model to get true AIC
        import statsmodels.formula.api as smf
        model = smf.ols('target ~ feature_1 + feature_2', data=simple_data).fit()
        original_aic = model.aic
        original_r2 = model.rsquared

        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=original_aic,
            original_r2=original_r2,
            n_samples=50  # More samples for this test
        )

        bootstrap_mean = np.mean(result.bootstrap_aics)

        # Bootstrap mean should be reasonably close to original
        # Allow 20% deviation due to sampling variation
        relative_diff = abs(bootstrap_mean - original_aic) / abs(original_aic)
        assert relative_diff < 0.2, f"Bootstrap mean {bootstrap_mean} too far from original {original_aic}"

    def test_stable_model_has_reasonable_cv(self, simple_data):
        """Verify stable data produces reasonable coefficient of variation."""
        result = calculate_bootstrap_metrics(
            data=simple_data,
            model_features=['feature_1', 'feature_2'],
            target_variable='target',
            original_aic=100.0,
            original_r2=0.85,
            n_samples=50
        )

        # Good data should have CV < 0.2 (relaxed for small sample bootstrap)
        assert result.aic_stability_coefficient < 0.2
        # R² stability should be tighter
        assert result.r2_stability_coefficient < 0.05

    def test_noisy_model_has_higher_cv(self, noisy_data):
        """Verify noisy data produces higher coefficient of variation."""
        result = calculate_bootstrap_metrics(
            data=noisy_data,
            model_features=['feature_1'],
            target_variable='target',
            original_aic=200.0,
            original_r2=0.1,
            n_samples=50
        )

        # Noisy data should have higher variability in R² stability
        # (AIC might still be stable if sample size is fixed)
        assert result.r2_stability_coefficient > result.aic_stability_coefficient * 0.5

    def test_reproducibility_with_same_seed(self, simple_data):
        """Verify bootstrap results are reproducible (same base seed)."""
        params = {
            'data': simple_data,
            'model_features': ['feature_1'],
            'target_variable': 'target',
            'original_aic': 100.0,
            'original_r2': 0.70,
            'n_samples': 30
        }

        result1 = calculate_bootstrap_metrics(**params)
        result2 = calculate_bootstrap_metrics(**params)

        # Results should be identical with same parameters (deterministic seeding)
        assert result1.bootstrap_aics == result2.bootstrap_aics


# =============================================================================
# Integration Tests
# =============================================================================

class TestBootstrapEngineIntegration:
    """Integration tests for bootstrap engine."""

    def test_end_to_end_stability_analysis(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Test complete stability analysis flow."""
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target'
        )

        # All results should be valid
        for result in results:
            assert result.successful_fits > 0
            assert len(result.bootstrap_aics) > 0
            assert result.stability_assessment in [
                "STABLE", "MODERATE", "UNSTABLE", "FAILED"
            ]

    def test_best_model_not_failed(self, simple_data, valid_models_df):
        """Verify best model (lowest AIC) completes analysis successfully."""
        config = {'enabled': True, 'n_samples': 50, 'models_to_analyze': 2}

        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=config,
            target_variable='target'
        )

        # First model should complete successfully (not FAILED)
        assert results[0].stability_assessment != "FAILED"
        assert results[0].successful_fits >= 25  # At least 50% success


# =============================================================================
# Tests for Config Validation
# =============================================================================

class TestValidateBootstrapConfig:
    """Test suite for _validate_bootstrap_config function."""

    def test_valid_config_returns_parameters(self, valid_models_df):
        """Test valid config returns correct parameters."""
        from src.features.selection.engines.bootstrap_engine import _validate_bootstrap_config

        config = {'n_samples': 50, 'models_to_analyze': 5}
        n_samples, models_to_analyze = _validate_bootstrap_config(config, valid_models_df)

        assert n_samples == 50
        assert models_to_analyze == 2  # Capped at len(valid_models_df)

    def test_default_values_used(self, valid_models_df):
        """Test default values when not specified."""
        from src.features.selection.engines.bootstrap_engine import _validate_bootstrap_config

        config = {}  # Empty config
        n_samples, models_to_analyze = _validate_bootstrap_config(config, valid_models_df)

        assert n_samples == 100  # Default
        assert models_to_analyze == 2  # min(10, len(valid_models_df))

    def test_invalid_config_type_raises(self, valid_models_df):
        """Test non-dict config raises ValueError."""
        from src.features.selection.engines.bootstrap_engine import _validate_bootstrap_config

        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_bootstrap_config("not a dict", valid_models_df)

    def test_insufficient_samples_raises(self, valid_models_df):
        """Test too few samples raises ValueError."""
        from src.features.selection.engines.bootstrap_engine import _validate_bootstrap_config

        config = {'n_samples': 3}  # < 5
        with pytest.raises(ValueError, match="Insufficient bootstrap samples"):
            _validate_bootstrap_config(config, valid_models_df)

    def test_empty_models_df_raises(self):
        """Test empty models DataFrame raises ValueError."""
        from src.features.selection.engines.bootstrap_engine import _validate_bootstrap_config

        config = {'n_samples': 50}
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="No models to analyze"):
            _validate_bootstrap_config(config, empty_df)


# =============================================================================
# Tests for Bootstrap Summary
# =============================================================================

class TestPrintBootstrapSummary:
    """Test suite for _print_bootstrap_summary function."""

    def test_print_summary_no_error(self, simple_data, valid_models_df, bootstrap_config_enabled, capsys):
        """Test summary printing completes without error."""
        from src.features.selection.engines.bootstrap_engine import _print_bootstrap_summary

        # Generate real results
        results = run_bootstrap_stability(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target'
        )

        # Should not raise
        _print_bootstrap_summary(results)

        captured = capsys.readouterr()
        assert "Bootstrap Stability Analysis Results" in captured.out
        assert "Models analyzed" in captured.out

    def test_summary_counts_assessments(self, capsys):
        """Test summary correctly counts stability assessments."""
        from src.features.selection.engines.bootstrap_engine import _print_bootstrap_summary
        from src.features.selection_types import BootstrapResult

        # Create mock results with known assessments
        mock_results = [
            BootstrapResult(
                model_name="Model 1", model_features="x1",
                bootstrap_aics=[100, 101, 99], bootstrap_r2_values=[0.8, 0.79, 0.81],
                original_aic=100.0, original_r2=0.8,
                aic_stability_coefficient=0.01, r2_stability_coefficient=0.01,
                confidence_intervals={}, successful_fits=30, total_attempts=30,
                stability_assessment="STABLE"
            ),
            BootstrapResult(
                model_name="Model 2", model_features="x2",
                bootstrap_aics=[100, 101, 99], bootstrap_r2_values=[0.8, 0.79, 0.81],
                original_aic=100.0, original_r2=0.8,
                aic_stability_coefficient=0.01, r2_stability_coefficient=0.01,
                confidence_intervals={}, successful_fits=30, total_attempts=30,
                stability_assessment="STABLE"
            ),
            BootstrapResult(
                model_name="Model 3", model_features="x3",
                bootstrap_aics=[100, 101, 99], bootstrap_r2_values=[0.8, 0.79, 0.81],
                original_aic=100.0, original_r2=0.8,
                aic_stability_coefficient=0.01, r2_stability_coefficient=0.01,
                confidence_intervals={}, successful_fits=30, total_attempts=30,
                stability_assessment="MODERATE"
            ),
        ]

        _print_bootstrap_summary(mock_results)

        captured = capsys.readouterr()
        assert "3" in captured.out  # 3 models
        assert "STABLE: 2" in captured.out
        assert "MODERATE: 1" in captured.out


# =============================================================================
# Tests for Visualization Data Preparation
# =============================================================================

class TestPrepareBootstrapVizData:
    """Test suite for _prepare_bootstrap_viz_data function."""

    @pytest.fixture
    def bootstrap_results_for_viz(self):
        """Create bootstrap results for visualization testing."""
        from src.features.selection_types import BootstrapResult

        return [
            BootstrapResult(
                model_name="Model 1",
                model_features="feature_1 + feature_2",
                bootstrap_aics=[100.0, 101.5, 99.2, 100.8, 98.7] * 6,  # 30 samples
                bootstrap_r2_values=[0.85, 0.84, 0.86, 0.85, 0.84] * 6,
                original_aic=100.0,
                original_r2=0.85,
                aic_stability_coefficient=0.01,
                r2_stability_coefficient=0.01,
                confidence_intervals={},
                successful_fits=30,
                total_attempts=30,
                stability_assessment="STABLE"
            ),
            BootstrapResult(
                model_name="Model 2",
                model_features="feature_1",
                bootstrap_aics=[110.0, 112.0, 108.0, 111.0, 109.0] * 6,
                bootstrap_r2_values=[0.70, 0.68, 0.72, 0.71, 0.69] * 6,
                original_aic=110.0,
                original_r2=0.70,
                aic_stability_coefficient=0.015,
                r2_stability_coefficient=0.02,
                confidence_intervals={},
                successful_fits=30,
                total_attempts=30,
                stability_assessment="MODERATE"
            ),
        ]

    def test_prepare_viz_data_returns_dataframe(self, bootstrap_results_for_viz):
        """Test that viz data preparation returns DataFrame."""
        from src.features.selection.engines.bootstrap_engine import _prepare_bootstrap_viz_data

        df, models_order = _prepare_bootstrap_viz_data(bootstrap_results_for_viz, n_models_display=2)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(models_order, list)

    def test_viz_data_has_required_columns(self, bootstrap_results_for_viz):
        """Test DataFrame has required columns for visualization."""
        from src.features.selection.engines.bootstrap_engine import _prepare_bootstrap_viz_data

        df, _ = _prepare_bootstrap_viz_data(bootstrap_results_for_viz, n_models_display=2)

        required_cols = ['model', 'model_features', 'bootstrap_aic', 'bootstrap_r2',
                        'original_aic', 'original_r2', 'stability_assessment']
        for col in required_cols:
            assert col in df.columns

    def test_viz_data_respects_n_models_display(self, bootstrap_results_for_viz):
        """Test that n_models_display limits output."""
        from src.features.selection.engines.bootstrap_engine import _prepare_bootstrap_viz_data

        df, models_order = _prepare_bootstrap_viz_data(bootstrap_results_for_viz, n_models_display=1)

        assert len(models_order) == 1
        assert df['model'].unique().tolist() == ['Model 1']

    def test_viz_data_models_order_correct(self, bootstrap_results_for_viz):
        """Test models order is correct."""
        from src.features.selection.engines.bootstrap_engine import _prepare_bootstrap_viz_data

        df, models_order = _prepare_bootstrap_viz_data(bootstrap_results_for_viz, n_models_display=2)

        assert models_order == ['Model 1', 'Model 2']

    def test_empty_results_raises(self):
        """Test empty results raises ValueError."""
        from src.features.selection.engines.bootstrap_engine import _prepare_bootstrap_viz_data

        with pytest.raises(ValueError, match="No visualization data"):
            _prepare_bootstrap_viz_data([], n_models_display=5)


# =============================================================================
# Tests for Stability Color Mapping
# =============================================================================

class TestGetStabilityColor:
    """Test suite for _get_stability_color function."""

    def test_stable_returns_green(self):
        """Test STABLE returns green color."""
        from src.features.selection.engines.bootstrap_engine import _get_stability_color
        import seaborn as sns

        color = _get_stability_color("STABLE")
        expected = sns.color_palette("deep")[2]  # Green

        assert color == expected

    def test_moderate_returns_orange(self):
        """Test MODERATE returns orange color."""
        from src.features.selection.engines.bootstrap_engine import _get_stability_color
        import seaborn as sns

        color = _get_stability_color("MODERATE")
        expected = sns.color_palette("deep")[1]  # Orange

        assert color == expected

    def test_unstable_returns_red(self):
        """Test UNSTABLE returns red color."""
        from src.features.selection.engines.bootstrap_engine import _get_stability_color
        import seaborn as sns

        color = _get_stability_color("UNSTABLE")
        expected = sns.color_palette("deep")[3]  # Red

        assert color == expected

    def test_unknown_returns_red(self):
        """Test unknown status returns red (default)."""
        from src.features.selection.engines.bootstrap_engine import _get_stability_color
        import seaborn as sns

        color = _get_stability_color("FAILED")
        expected = sns.color_palette("deep")[3]  # Red

        assert color == expected


# =============================================================================
# Tests for Visualization Functions
# =============================================================================

class TestBootstrapVisualization:
    """Test suite for bootstrap visualization functions."""

    @pytest.fixture
    def viz_bootstrap_df(self):
        """Create bootstrap DataFrame for visualization testing."""
        data = []
        for model_num, (stability, aic_base) in enumerate([
            ("STABLE", 100.0), ("MODERATE", 110.0)
        ], 1):
            for i in range(30):
                data.append({
                    'model': f'Model {model_num}',
                    'model_features': f'feature_{model_num}',
                    'bootstrap_aic': aic_base + np.random.normal(0, 2),
                    'bootstrap_r2': 0.8 + np.random.normal(0, 0.02),
                    'original_aic': aic_base,
                    'original_r2': 0.8,
                    'stability_assessment': stability
                })
        return pd.DataFrame(data)

    def test_create_violin_plot_returns_figure(self, viz_bootstrap_df):
        """Test violin plot creation returns Figure."""
        from src.features.selection.engines.bootstrap_engine import _create_violin_plot
        import matplotlib.pyplot as plt

        models_order = ['Model 1', 'Model 2']
        fig = _create_violin_plot(viz_bootstrap_df, models_order)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_boxplot_returns_figure(self, viz_bootstrap_df):
        """Test boxplot creation returns Figure."""
        from src.features.selection.engines.bootstrap_engine import _create_bootstrap_boxplot
        import matplotlib.pyplot as plt

        models_order = ['Model 1', 'Model 2']
        fig = _create_bootstrap_boxplot(viz_bootstrap_df, models_order, fig_width=12)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_prepare_boxplot_data_structure(self, viz_bootstrap_df):
        """Test boxplot data preparation structure."""
        from src.features.selection.engines.bootstrap_engine import _prepare_boxplot_data

        models_order = ['Model 1', 'Model 2']
        boxplot_data, model_labels, colors = _prepare_boxplot_data(viz_bootstrap_df, models_order)

        assert len(boxplot_data) == 2
        assert len(model_labels) == 2
        assert len(colors) == 2
        assert all(isinstance(d, np.ndarray) for d in boxplot_data)

    def test_render_violin_kde_no_error(self, viz_bootstrap_df):
        """Test violin KDE rendering completes without error."""
        from src.features.selection.engines.bootstrap_engine import _render_violin_kde
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        model_data = viz_bootstrap_df[viz_bootstrap_df['model'] == 'Model 1']['bootstrap_aic'].values

        # Should not raise
        _render_violin_kde(ax, model_data, i=0, stability="STABLE", original_aic=100.0)

        plt.close(fig)

    def test_format_violin_axes_no_error(self, viz_bootstrap_df):
        """Test violin axes formatting completes without error."""
        from src.features.selection.engines.bootstrap_engine import _format_violin_axes
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        models_order = ['Model 1', 'Model 2']

        # Should not raise
        _format_violin_axes(ax, viz_bootstrap_df, models_order)

        assert ax.get_xlabel() == 'AIC Score (Lower is Better)'
        plt.close(fig)

    def test_render_boxplot_no_error(self, viz_bootstrap_df):
        """Test boxplot rendering completes without error."""
        from src.features.selection.engines.bootstrap_engine import _render_boxplot, _prepare_boxplot_data
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        models_order = ['Model 1', 'Model 2']
        boxplot_data, model_labels, colors = _prepare_boxplot_data(viz_bootstrap_df, models_order)

        # Should not raise
        _render_boxplot(ax, boxplot_data, model_labels, colors)

        plt.close(fig)


# =============================================================================
# Tests for Create Bootstrap Visualizations
# =============================================================================

class TestCreateBootstrapVisualizations:
    """Test suite for _create_bootstrap_visualizations function."""

    @pytest.fixture
    def bootstrap_results_for_viz_creation(self):
        """Create bootstrap results for visualization creation testing."""
        from src.features.selection_types import BootstrapResult

        np.random.seed(42)
        return [
            BootstrapResult(
                model_name=f"Model {i+1}",
                model_features=f"feature_{i+1}",
                bootstrap_aics=(100.0 + i*10 + np.random.normal(0, 2, 30)).tolist(),
                bootstrap_r2_values=(0.8 - i*0.1 + np.random.normal(0, 0.02, 30)).tolist(),
                original_aic=100.0 + i*10,
                original_r2=0.8 - i*0.1,
                aic_stability_coefficient=0.01 + i*0.005,
                r2_stability_coefficient=0.02 + i*0.01,
                confidence_intervals={},
                successful_fits=30,
                total_attempts=30,
                stability_assessment=["STABLE", "MODERATE", "UNSTABLE"][min(i, 2)]
            )
            for i in range(3)
        ]

    def test_creates_visualizations_dict(self, bootstrap_results_for_viz_creation):
        """Test visualization creation returns dict with figures."""
        from src.features.selection.engines.bootstrap_engine import _create_bootstrap_visualizations
        import matplotlib.pyplot as plt

        config = {'n_models_display': 3, 'fig_width': 12}
        result = _create_bootstrap_visualizations(bootstrap_results_for_viz_creation, config)

        assert isinstance(result, dict)
        assert 'violin_plot' in result or 'boxplot' in result

        for fig in result.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)

    def test_empty_results_raises(self):
        """Test empty results raises ValueError."""
        from src.features.selection.engines.bootstrap_engine import _create_bootstrap_visualizations

        with pytest.raises(ValueError, match="No bootstrap results"):
            _create_bootstrap_visualizations([], {})


# =============================================================================
# Tests for Core Bootstrap Analysis with Visualizations
# =============================================================================

class TestRunCoreBootstrapAnalysis:
    """Test suite for run_core_bootstrap_analysis function."""

    def test_returns_results_and_visualizations(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Test function returns results and visualizations."""
        from src.features.selection.engines.bootstrap_engine import run_core_bootstrap_analysis
        import matplotlib.pyplot as plt

        results, viz = run_core_bootstrap_analysis(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target',
            create_visualizations=True
        )

        assert isinstance(results, list)
        assert len(results) > 0
        assert viz is not None

        # Clean up figures
        if viz:
            for fig in viz.values():
                if isinstance(fig, plt.Figure):
                    plt.close(fig)

    def test_disabled_returns_empty(self, simple_data, valid_models_df):
        """Test disabled config returns empty results."""
        from src.features.selection.engines.bootstrap_engine import run_core_bootstrap_analysis

        config = {'enabled': False}
        results, viz = run_core_bootstrap_analysis(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=config,
            target_variable='target',
            create_visualizations=True
        )

        assert results == []
        assert viz == {}

    def test_no_visualizations_when_disabled(
        self, simple_data, valid_models_df, bootstrap_config_enabled
    ):
        """Test no visualizations created when disabled."""
        from src.features.selection.engines.bootstrap_engine import run_core_bootstrap_analysis

        results, viz = run_core_bootstrap_analysis(
            data=simple_data,
            valid_models_df=valid_models_df,
            config=bootstrap_config_enabled,
            target_variable='target',
            create_visualizations=False
        )

        assert len(results) > 0
        assert viz is None


# =============================================================================
# Tests for Edge Cases and Error Handling
# =============================================================================

class TestBootstrapEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_stability_near_zero_mean(self):
        """Test stability coefficient computation with near-zero mean."""
        from src.features.selection.engines.bootstrap_engine import _compute_single_stability_coefficient

        # Values with near-zero mean but positive original
        values = [-0.001, 0.001, -0.0005, 0.0008, -0.0003]
        original = 100.0

        result = _compute_single_stability_coefficient(values, original)

        # Should use original value for scaling
        assert np.isfinite(result)
        assert result >= 0

    def test_compute_stability_both_near_zero(self):
        """Test stability coefficient with both mean and original near zero."""
        from src.features.selection.engines.bootstrap_engine import _compute_single_stability_coefficient

        # Values and original both near zero
        values = [0.00001, -0.00001, 0.000005, -0.000005]
        original = 0.0000001

        result = _compute_single_stability_coefficient(values, original)

        # Should use IQR-based measure, returning inf if median is near zero
        assert np.isfinite(result) or result == float('inf')

    def test_single_model_bootstrap_error_handling(self, simple_data):
        """Test error propagation from single model bootstrap."""
        from src.features.selection.engines.bootstrap_engine import _run_single_model_bootstrap

        # Create model row with invalid feature
        model_row = pd.Series({
            'features': 'nonexistent_column',
            'aic': 100.0,
            'r_squared': 0.8
        })

        with pytest.raises((ValueError, RuntimeError)):
            _run_single_model_bootstrap(simple_data, model_row, 'target', n_samples=30, idx=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
