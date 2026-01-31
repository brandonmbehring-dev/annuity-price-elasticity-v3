"""
Tests for src.visualization.visualization module.

Tests AIC feature selection visualization and model performance charts:
- plot_aic_model_comparison
- plot_feature_selection_progression
- plot_economic_constraint_validation
- plot_model_diagnostics
- create_aic_summary_report
- Helper functions for each plot type

Target coverage: 70%+
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.visualization import (
    _compute_aic_progression_by_feature_count,
    _create_aic_histogram,
    _create_aic_progression_line_plot,
    _create_aic_scores_bar_plot,
    _create_convergence_pie,
    _create_feature_correlation_heatmap,
    _create_feature_count_chart,
    _create_feature_variance_bar_chart,
    _create_performance_tradeoff_chart,
    _create_rsquared_feature_scatter,
    _create_selected_features_bar_chart,
    _create_summary_text_section,
    _create_target_distribution_histogram,
    _create_target_vs_top_feature_scatter,
    _extract_coefficient_averages,
    _filter_valid_models_for_comparison,
    _plot_coefficient_bar,
    _prepare_clean_model_data,
    _validate_aic_results_for_comparison,
    _validate_aic_results_for_progression,
    _validate_diagnostics_inputs,
    create_aic_summary_report,
    plot_aic_model_comparison,
    plot_economic_constraint_validation,
    plot_feature_selection_progression,
    plot_model_diagnostics,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes for testing."""
    ax = MagicMock(spec=plt.Axes)
    ax.transAxes = 'transAxes_mock'
    return ax


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame with all required columns."""
    return pd.DataFrame({
        'features': ['feat_a+feat_b', 'feat_c', 'feat_d+feat_e', 'feat_f'],
        'aic_score': [100.5, 105.2, 98.7, 110.0],
        'r_squared': [0.85, 0.82, 0.87, 0.80],
        'n_features': [2, 1, 2, 1],
        'model_converged': [True, True, True, False],
    })


@pytest.fixture
def sample_aic_results_with_inf():
    """Sample AIC results with infinite values."""
    return pd.DataFrame({
        'features': ['feat_a', 'feat_b', 'feat_c'],
        'aic_score': [100.5, np.inf, 105.2],
        'r_squared': [0.85, 0.82, 0.87],
        'n_features': [1, 1, 1],
        'model_converged': [True, True, True],
    })


@pytest.fixture
def sample_selected_features():
    """Sample selected features list."""
    return ['prudential_rate', 'competitor_avg', 'market_volatility']


@pytest.fixture
def sample_feature_coefficients():
    """Sample feature coefficients dictionary."""
    return {
        'feat_a+feat_b': {
            'Intercept': 1.0,
            'competitor_rate': -0.5,
            'prudential_rate': 0.3,
        },
        'feat_c': {
            'Intercept': 0.8,
            'competitor_avg': -0.4,
            'prudential_rate': 0.2,
        },
    }


@pytest.fixture
def sample_model_data():
    """Sample model data DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'prudential_rate': np.random.normal(0.05, 0.01, 100),
        'competitor_avg': np.random.normal(0.04, 0.01, 100),
        'market_volatility': np.random.normal(0.2, 0.05, 100),
        'sales_target': np.random.normal(1000, 100, 100),
    })


@pytest.fixture
def sample_metadata():
    """Sample metadata dictionary."""
    return {
        'best_r_squared': 0.87,
        'n_observations': 1000,
        'total_models_tested': 500,
        'valid_models': 450,
    }


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================


class TestValidateAicResultsForComparison:
    """Tests for _validate_aic_results_for_comparison function."""

    def test_valid_results_pass(self, sample_aic_results):
        """Test valid results don't raise."""
        _validate_aic_results_for_comparison(sample_aic_results)
        # No exception raised

    def test_empty_results_raise(self):
        """Test empty results raise ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="CRITICAL: AIC results are empty"):
            _validate_aic_results_for_comparison(empty_df)

    def test_missing_columns_raise(self):
        """Test missing columns raise ValueError."""
        incomplete_df = pd.DataFrame({
            'features': ['a', 'b'],
            'aic_score': [100, 105],
            # Missing r_squared and n_features
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_aic_results_for_comparison(incomplete_df)


class TestFilterValidModelsForComparison:
    """Tests for _filter_valid_models_for_comparison function."""

    def test_filters_converged_models(self, sample_aic_results):
        """Test filters to converged models only."""
        result = _filter_valid_models_for_comparison(sample_aic_results, max_models=10)

        # Should exclude non-converged model
        assert len(result) == 3
        assert 'feat_f' not in result['features'].values

    def test_filters_infinite_aic(self, sample_aic_results_with_inf):
        """Test filters out infinite AIC values."""
        result = _filter_valid_models_for_comparison(sample_aic_results_with_inf, max_models=10)

        assert len(result) == 2
        assert not any(np.isinf(result['aic_score']))

    def test_respects_max_models(self, sample_aic_results):
        """Test respects max_models limit."""
        result = _filter_valid_models_for_comparison(sample_aic_results, max_models=2)

        assert len(result) == 2

    def test_no_valid_models_raise(self):
        """Test raises when no valid models."""
        all_failed = pd.DataFrame({
            'features': ['a', 'b'],
            'aic_score': [100, 105],
            'r_squared': [0.85, 0.82],
            'n_features': [1, 1],
            'model_converged': [False, False],
        })

        with pytest.raises(ValueError, match="No valid converged models"):
            _filter_valid_models_for_comparison(all_failed, max_models=10)


class TestValidateAicResultsForProgression:
    """Tests for _validate_aic_results_for_progression function."""

    def test_valid_results_pass(self, sample_aic_results):
        """Test valid results don't raise."""
        _validate_aic_results_for_progression(sample_aic_results)

    def test_empty_results_raise(self):
        """Test empty results raise ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="CRITICAL: No AIC results provided"):
            _validate_aic_results_for_progression(empty_df)


class TestComputeAicProgressionByFeatureCount:
    """Tests for _compute_aic_progression_by_feature_count function."""

    def test_computes_progression(self, sample_aic_results):
        """Test computes progression by feature count."""
        result = _compute_aic_progression_by_feature_count(sample_aic_results)

        assert 'n_features' in result.columns
        assert 'aic_score' in result.columns
        # Should have best AIC for each feature count
        assert result.loc[result['n_features'] == 2, 'aic_score'].iloc[0] == 98.7

    def test_no_valid_models_raise(self):
        """Test raises when no valid models."""
        all_failed = pd.DataFrame({
            'features': ['a'],
            'aic_score': [np.inf],
            'n_features': [1],
            'model_converged': [True],
        })

        with pytest.raises(ValueError, match="No valid models found"):
            _compute_aic_progression_by_feature_count(all_failed)


class TestValidateDiagnosticsInputs:
    """Tests for _validate_diagnostics_inputs function."""

    def test_valid_inputs_pass(self, sample_model_data, sample_selected_features):
        """Test valid inputs don't raise."""
        _validate_diagnostics_inputs(
            sample_model_data,
            sample_selected_features[:2],  # Only use features in data
            'sales_target'
        )

    def test_missing_target_raise(self, sample_model_data, sample_selected_features):
        """Test raises when target not in data."""
        with pytest.raises(ValueError, match="Target variable .* not found"):
            _validate_diagnostics_inputs(
                sample_model_data,
                sample_selected_features[:2],
                'nonexistent_target'
            )

    def test_missing_features_raise(self, sample_model_data):
        """Test raises when features not in data."""
        with pytest.raises(ValueError, match="Missing features in data"):
            _validate_diagnostics_inputs(
                sample_model_data,
                ['nonexistent_feature'],
                'sales_target'
            )


class TestPrepareCleanModelData:
    """Tests for _prepare_clean_model_data function."""

    def test_prepares_clean_data(self, sample_model_data):
        """Test prepares clean data."""
        features = ['prudential_rate', 'competitor_avg']
        result = _prepare_clean_model_data(sample_model_data, features, 'sales_target')

        assert len(result.columns) == 3  # 2 features + target
        assert 'prudential_rate' in result.columns
        assert 'sales_target' in result.columns

    def test_insufficient_data_raise(self):
        """Test raises when insufficient data after cleaning."""
        small_data = pd.DataFrame({
            'feat': [1, 2, np.nan, np.nan, np.nan],
            'target': [10, 20, 30, 40, 50],
        })

        with pytest.raises(ValueError, match="Insufficient data after cleaning"):
            _prepare_clean_model_data(small_data, ['feat'], 'target')


# =============================================================================
# HELPER PLOT FUNCTION TESTS
# =============================================================================


class TestCreateAicScoresBarPlot:
    """Tests for _create_aic_scores_bar_plot function."""

    def test_creates_bar_plot(self, mock_axes, sample_aic_results):
        """Test creates bar plot."""
        _create_aic_scores_bar_plot(mock_axes, sample_aic_results, 'Test Title')

        mock_axes.bar.assert_called_once()
        mock_axes.set_title.assert_called_once()
        mock_axes.set_xlabel.assert_called_once()
        mock_axes.set_ylabel.assert_called_once()


class TestCreateRsquaredFeatureScatter:
    """Tests for _create_rsquared_feature_scatter function."""

    @patch('src.visualization.visualization.plt')
    def test_creates_scatter(self, mock_plt, mock_axes, sample_aic_results):
        """Test creates scatter plot."""
        _create_rsquared_feature_scatter(mock_axes, sample_aic_results)

        mock_axes.scatter.assert_called_once()
        mock_plt.colorbar.assert_called_once()


class TestCreateAicProgressionLinePlot:
    """Tests for _create_aic_progression_line_plot function."""

    def test_creates_line_plot(self, mock_axes, sample_aic_results):
        """Test creates line plot with horizontal reference line."""
        progression = pd.DataFrame({'n_features': [1, 2], 'aic_score': [105, 98]})

        _create_aic_progression_line_plot(mock_axes, progression, 98.0)

        mock_axes.plot.assert_called_once()
        mock_axes.axhline.assert_called_once()
        mock_axes.legend.assert_called_once()


class TestCreateSelectedFeatureBarChart:
    """Tests for _create_selected_features_bar_chart function."""

    def test_creates_bar_chart(self, mock_axes, sample_selected_features):
        """Test creates horizontal bar chart."""
        _create_selected_features_bar_chart(mock_axes, sample_selected_features, 98.0)

        mock_axes.barh.assert_called_once()
        mock_axes.set_yticks.assert_called_once()
        mock_axes.set_yticklabels.assert_called_once()


class TestExtractCoefficientAverages:
    """Tests for _extract_coefficient_averages function."""

    def test_extracts_averages(self, sample_feature_coefficients):
        """Test extracts coefficient averages."""
        comp_coeffs, prud_coeffs = _extract_coefficient_averages(sample_feature_coefficients)

        assert len(comp_coeffs) == 2
        assert len(prud_coeffs) == 2
        # Competitor coefficients should be negative (average of -0.5 and -0.4)
        assert all(c <= 0 for c in comp_coeffs)
        # Prudential coefficients should be positive
        assert all(p >= 0 for p in prud_coeffs)

    def test_empty_coefficients(self):
        """Test handles empty coefficients."""
        comp_coeffs, prud_coeffs = _extract_coefficient_averages({})

        assert comp_coeffs == []
        assert prud_coeffs == []


class TestPlotCoefficientBar:
    """Tests for _plot_coefficient_bar function."""

    def test_plots_with_correct_coloring(self, mock_axes):
        """Test plots with correct economic coloring."""
        coeffs = [-0.5, -0.3, 0.1]  # Last one violates expected negative

        _plot_coefficient_bar(mock_axes, coeffs, False, 'Test Title')

        mock_axes.bar.assert_called_once()
        mock_axes.axhline.assert_called_once()


class TestCreateFeatureCorrelationHeatmap:
    """Tests for _create_feature_correlation_heatmap function."""

    @patch('src.visualization.visualization.sns')
    def test_creates_heatmap(self, mock_sns, mock_axes, sample_model_data):
        """Test creates heatmap."""
        features = ['prudential_rate', 'competitor_avg']

        _create_feature_correlation_heatmap(mock_axes, sample_model_data, features)

        mock_sns.heatmap.assert_called_once()


class TestCreateTargetVsTopFeatureScatter:
    """Tests for _create_target_vs_top_feature_scatter function."""

    def test_creates_scatter(self, mock_axes, sample_model_data):
        """Test creates scatter plot."""
        features = ['prudential_rate', 'competitor_avg']

        _create_target_vs_top_feature_scatter(
            mock_axes, sample_model_data, features, 'sales_target'
        )

        mock_axes.scatter.assert_called_once()

    def test_empty_features(self, mock_axes, sample_model_data):
        """Test handles empty features list."""
        _create_target_vs_top_feature_scatter(
            mock_axes, sample_model_data, [], 'sales_target'
        )

        # Should not create scatter with empty features
        mock_axes.scatter.assert_not_called()


class TestCreateFeatureVarianceBarChart:
    """Tests for _create_feature_variance_bar_chart function."""

    def test_creates_bar_chart(self, mock_axes, sample_model_data):
        """Test creates variance bar chart."""
        features = ['prudential_rate', 'competitor_avg']

        _create_feature_variance_bar_chart(mock_axes, sample_model_data, features)

        mock_axes.barh.assert_called_once()


class TestCreateTargetDistributionHistogram:
    """Tests for _create_target_distribution_histogram function."""

    def test_creates_histogram(self, mock_axes, sample_model_data):
        """Test creates histogram."""
        _create_target_distribution_histogram(mock_axes, sample_model_data, 'sales_target')

        mock_axes.hist.assert_called_once()


class TestCreateSummaryTextSection:
    """Tests for _create_summary_text_section function."""

    def test_creates_text(self, mock_axes, sample_selected_features, sample_metadata):
        """Test creates summary text."""
        _create_summary_text_section(mock_axes, 98.0, sample_selected_features, sample_metadata)

        mock_axes.axis.assert_called_with('off')
        mock_axes.text.assert_called_once()


class TestCreateAicHistogram:
    """Tests for _create_aic_histogram function."""

    def test_creates_histogram(self, mock_axes, sample_aic_results):
        """Test creates AIC histogram."""
        _create_aic_histogram(mock_axes, sample_aic_results, 98.0)

        mock_axes.hist.assert_called_once()
        mock_axes.axvline.assert_called_once()


class TestCreateFeatureCountChart:
    """Tests for _create_feature_count_chart function."""

    def test_creates_chart(self, mock_axes, sample_aic_results, sample_selected_features):
        """Test creates feature count chart."""
        _create_feature_count_chart(mock_axes, sample_aic_results, sample_selected_features)

        mock_axes.bar.assert_called_once()
        mock_axes.axvline.assert_called_once()


class TestCreateConvergencePie:
    """Tests for _create_convergence_pie function."""

    def test_creates_pie_chart(self, mock_axes, sample_aic_results):
        """Test creates convergence pie chart."""
        _create_convergence_pie(mock_axes, sample_aic_results)

        mock_axes.pie.assert_called_once()


class TestCreatePerformanceTradeoffChart:
    """Tests for _create_performance_tradeoff_chart function."""

    def test_creates_dual_axis_chart(self, mock_axes):
        """Test creates dual-axis performance chart."""
        mock_twin = MagicMock()
        mock_axes.twinx.return_value = mock_twin
        mock_axes.get_legend_handles_labels.return_value = (['h1'], ['l1'])
        mock_twin.get_legend_handles_labels.return_value = (['h2'], ['l2'])

        grouped = pd.DataFrame({
            'n_features': [1, 2, 3],
            'aic_score': [110, 105, 100],
            'r_squared': [0.80, 0.85, 0.87],
        })

        _create_performance_tradeoff_chart(mock_axes, grouped)

        mock_axes.plot.assert_called_once()
        mock_twin.plot.assert_called_once()


# =============================================================================
# MAIN PLOT FUNCTION TESTS
# =============================================================================


class TestPlotAicModelComparison:
    """Tests for plot_aic_model_comparison function."""

    @patch('src.visualization.visualization.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results):
        """Test returns matplotlib figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_ax2 = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, (mock_ax1, mock_ax2))

        result = plot_aic_model_comparison(sample_aic_results)

        assert result == mock_figure

    @patch('src.visualization.visualization.plt')
    def test_empty_results_raise(self, mock_plt):
        """Test raises with empty results."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            plot_aic_model_comparison(empty_df)


class TestPlotFeatureSelectionProgression:
    """Tests for plot_feature_selection_progression function."""

    @patch('src.visualization.visualization.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results, sample_selected_features):
        """Test returns matplotlib figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_ax2 = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, (mock_ax1, mock_ax2))

        result = plot_feature_selection_progression(
            sample_aic_results, sample_selected_features, 98.0
        )

        assert result == mock_figure


class TestPlotEconomicConstraintValidation:
    """Tests for plot_economic_constraint_validation function."""

    @patch('src.visualization.visualization.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results, sample_feature_coefficients):
        """Test returns matplotlib figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_ax2 = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, (mock_ax1, mock_ax2))

        result = plot_economic_constraint_validation(
            sample_aic_results, sample_feature_coefficients
        )

        assert result == mock_figure

    @patch('src.visualization.visualization.plt')
    def test_empty_inputs_raise(self, mock_plt):
        """Test raises with empty inputs."""
        with pytest.raises(ValueError, match="Insufficient data"):
            plot_economic_constraint_validation(pd.DataFrame(), {})


class TestPlotModelDiagnostics:
    """Tests for plot_model_diagnostics function."""

    @patch('src.visualization.visualization.plt')
    @patch('src.visualization.visualization.sns')
    def test_returns_figure(self, mock_sns, mock_plt, sample_model_data):
        """Test returns matplotlib figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        # Create proper 2D array of mock axes
        mock_ax00 = MagicMock(spec=plt.Axes)
        mock_ax01 = MagicMock(spec=plt.Axes)
        mock_ax10 = MagicMock(spec=plt.Axes)
        mock_ax11 = MagicMock(spec=plt.Axes)
        mock_axes = MagicMock()
        mock_axes.__getitem__ = lambda self, key: {
            (0, 0): mock_ax00,
            (0, 1): mock_ax01,
            (1, 0): mock_ax10,
            (1, 1): mock_ax11,
        }[key]
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        features = ['prudential_rate', 'competitor_avg']
        result = plot_model_diagnostics(sample_model_data, features, 'sales_target')

        assert result == mock_figure


class TestCreateAicSummaryReport:
    """Tests for create_aic_summary_report function."""

    @patch('src.visualization.visualization.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results, sample_selected_features, sample_metadata):
        """Test returns matplotlib figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_plt.figure.return_value = mock_figure
        mock_figure.add_gridspec.return_value = MagicMock()

        # Create mock axes that return proper values
        mock_ax = MagicMock(spec=plt.Axes)
        mock_ax.transAxes = 'transAxes_mock'
        mock_ax.twinx.return_value = MagicMock(spec=plt.Axes)
        mock_ax.get_legend_handles_labels.return_value = (['h1'], ['l1'])
        mock_ax.twinx.return_value.get_legend_handles_labels.return_value = (['h2'], ['l2'])
        mock_figure.add_subplot.return_value = mock_ax

        result = create_aic_summary_report(
            sample_aic_results, sample_selected_features, 98.0, sample_metadata
        )

        assert result == mock_figure

    @patch('src.visualization.visualization.plt')
    def test_empty_results_raise(self, mock_plt, sample_selected_features, sample_metadata):
        """Test raises with empty results."""
        with pytest.raises(ValueError, match="CRITICAL: No AIC results"):
            create_aic_summary_report(
                pd.DataFrame(), sample_selected_features, 98.0, sample_metadata
            )


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases across visualization functions."""

    def test_coefficient_extraction_no_matching_keys(self):
        """Test coefficient extraction with no matching keys."""
        coeffs = {
            'model': {'other_key': 0.5, 'another_key': -0.3}
        }

        comp_coeffs, prud_coeffs = _extract_coefficient_averages(coeffs)

        # Should return 0 for no matching keys
        assert comp_coeffs[0] == 0
        assert prud_coeffs[0] == 0

    @patch('src.visualization.visualization.plt')
    def test_progression_with_single_feature_count(self, mock_plt):
        """Test progression with models all having same feature count."""
        results = pd.DataFrame({
            'features': ['a', 'b', 'c'],
            'aic_score': [100, 105, 110],
            'r_squared': [0.85, 0.82, 0.80],
            'n_features': [2, 2, 2],  # All same
            'model_converged': [True, True, True],
        })

        progression = _compute_aic_progression_by_feature_count(results)

        # Should have only one row
        assert len(progression) == 1
        assert progression.iloc[0]['aic_score'] == 100  # Best AIC

    def test_filter_all_infinite_aic(self):
        """Test filter when all models have infinite AIC."""
        all_inf = pd.DataFrame({
            'features': ['a', 'b'],
            'aic_score': [np.inf, np.inf],
            'r_squared': [0.85, 0.82],
            'n_features': [1, 2],
            'model_converged': [True, True],
        })

        with pytest.raises(ValueError, match="No valid converged models"):
            _filter_valid_models_for_comparison(all_inf, max_models=10)
