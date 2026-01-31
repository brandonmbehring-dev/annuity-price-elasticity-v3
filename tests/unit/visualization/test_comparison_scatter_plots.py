"""
Tests for src.visualization.comparison_scatter_plots module.

Tests scatter plot visualizations for model comparison:
- AIC vs R² scatter plot
- Information criteria heatmap
- Bootstrap stability scatter
- Model complexity distribution
- Feature frequency analysis
- Ranking correlation matrix
- Decision summary

Target coverage: 70%+
"""

from collections import namedtuple
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.comparison_scatter_plots import (
    plot_aic_vs_r2_scatter,
    plot_bootstrap_stability_scatter,
    plot_complexity_distribution,
    plot_decision_summary,
    plot_feature_frequency,
    plot_information_criteria_heatmap,
    plot_ranking_correlation_matrix,
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
def default_colors():
    """Default color scheme for plots."""
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'highlight': '#d62728',
        'success': '#2ca02c',
        'light_gray': '#f0f0f0',
    }


@pytest.fixture
def sample_top_models():
    """Sample top models DataFrame."""
    return pd.DataFrame({
        'features': ['feat_a+feat_b', 'feat_c+feat_d', 'feat_e', 'feat_f+feat_g'],
        'aic': [100.5, 105.2, 98.7, 110.0],
        'r_squared': [0.85, 0.82, 0.87, 0.80],
        'n_features': [2, 2, 1, 2],
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
def sample_information_criteria_results():
    """Sample information criteria results with namedtuple format."""
    InfoCriteriaResult = namedtuple(
        'InfoCriteriaResult',
        ['model_features', 'criteria_values']
    )
    return [
        InfoCriteriaResult(
            'feat_a+feat_b',
            {'aic': 100.0, 'bic': 105.0, 'hqic': 102.0, 'caic': 108.0}
        ),
        InfoCriteriaResult(
            'feat_c+feat_d',
            {'aic': 102.0, 'bic': 107.0, 'hqic': 104.0, 'caic': 110.0}
        ),
    ]


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame."""
    return pd.DataFrame({
        'features': ['model_a', 'model_b', 'model_c'],
        'aic': [100.0, 102.5, 105.0],
        'r_squared': [0.85, 0.83, 0.80],
    })


# =============================================================================
# AIC VS R² SCATTER TESTS
# =============================================================================


class TestPlotAicVsR2Scatter:
    """Tests for plot_aic_vs_r2_scatter function."""

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_basic_scatter(self, mock_plt, mock_axes, sample_top_models, default_colors):
        """Test basic AIC vs R² scatter plot."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_aic_vs_r2_scatter(mock_axes, sample_top_models, default_colors)

        # Verify scatter was created (twice: main + pareto)
        assert mock_axes.scatter.call_count == 2
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_scatter_colorbar(self, mock_plt, mock_axes, sample_top_models, default_colors):
        """Test colorbar is created for feature count."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_aic_vs_r2_scatter(mock_axes, sample_top_models, default_colors)

        mock_plt.colorbar.assert_called_once()
        mock_colorbar.set_label.assert_called_with('Number of Features')

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_scatter_pareto_highlight(self, mock_plt, mock_axes, sample_top_models, default_colors):
        """Test Pareto optimal models are highlighted."""
        mock_plt.colorbar.return_value = MagicMock()

        plot_aic_vs_r2_scatter(mock_axes, sample_top_models, default_colors)

        # Second scatter call is for Pareto optimal points
        calls = mock_axes.scatter.call_args_list
        assert len(calls) == 2
        pareto_call = calls[1]
        assert pareto_call[1]['edgecolors'] == default_colors['highlight']

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_scatter_legend(self, mock_plt, mock_axes, sample_top_models, default_colors):
        """Test legend is created."""
        mock_plt.colorbar.return_value = MagicMock()

        plot_aic_vs_r2_scatter(mock_axes, sample_top_models, default_colors)

        mock_axes.legend.assert_called_once()


# =============================================================================
# INFORMATION CRITERIA HEATMAP TESTS
# =============================================================================


class TestPlotInformationCriteriaHeatmap:
    """Tests for plot_information_criteria_heatmap function."""

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_basic_heatmap(
        self, mock_plt, mock_axes, sample_information_criteria_results, default_colors
    ):
        """Test basic information criteria heatmap."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_information_criteria_heatmap(
            mock_axes, sample_information_criteria_results, default_colors
        )

        # Verify imshow was called
        mock_axes.imshow.assert_called_once()
        # Verify colorbar
        mock_plt.colorbar.assert_called_once()

    def test_heatmap_no_results(self, mock_axes, default_colors):
        """Test heatmap with empty results."""
        plot_information_criteria_heatmap(mock_axes, [], default_colors)

        # Should show placeholder
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Not Available' in text_content

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_heatmap_model_name_truncation(
        self, mock_plt, mock_axes, default_colors
    ):
        """Test long model names are truncated."""
        InfoCriteriaResult = namedtuple(
            'InfoCriteriaResult', ['model_features', 'criteria_values']
        )
        results = [
            InfoCriteriaResult(
                'this_is_a_very_long_model_name_that_exceeds_limit',
                {'aic': 100.0, 'bic': 105.0, 'hqic': 102.0, 'caic': 108.0}
            )
        ]
        mock_plt.colorbar.return_value = MagicMock()

        plot_information_criteria_heatmap(mock_axes, results, default_colors)

        # Check ytick labels are truncated
        call_args = mock_axes.set_yticklabels.call_args
        labels = call_args[0][0]
        assert '...' in labels[0]

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_heatmap_criteria_labels(
        self, mock_plt, mock_axes, sample_information_criteria_results, default_colors
    ):
        """Test criteria column labels."""
        mock_plt.colorbar.return_value = MagicMock()

        plot_information_criteria_heatmap(
            mock_axes, sample_information_criteria_results, default_colors
        )

        # Check xticklabels
        call_args = mock_axes.set_xticklabels.call_args
        labels = call_args[0][0]
        assert labels == ['AIC', 'BIC', 'HQIC', 'CAIC']


# =============================================================================
# BOOTSTRAP STABILITY SCATTER TESTS
# =============================================================================


class TestPlotBootstrapStabilityScatter:
    """Tests for plot_bootstrap_stability_scatter function."""

    def test_basic_scatter(self, mock_axes, sample_bootstrap_results_dict, default_colors):
        """Test basic bootstrap stability scatter."""
        plot_bootstrap_stability_scatter(
            mock_axes, sample_bootstrap_results_dict, default_colors
        )

        # Verify scatter was created
        mock_axes.scatter.assert_called_once()
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_scatter_no_results(self, mock_axes, default_colors):
        """Test scatter with empty bootstrap results."""
        plot_bootstrap_stability_scatter(mock_axes, [], default_colors)

        # Should show placeholder
        mock_axes.text.assert_called_once()
        call_args = mock_axes.text.call_args
        text_content = call_args[0][2]
        assert 'Not Available' in text_content

    def test_scatter_annotations(self, mock_axes, sample_bootstrap_results_dict, default_colors):
        """Test top 3 stable models are annotated."""
        plot_bootstrap_stability_scatter(
            mock_axes, sample_bootstrap_results_dict, default_colors
        )

        # Should have 3 annotations (top 3 most stable)
        assert mock_axes.annotate.call_count == 3

    def test_scatter_grid(self, mock_axes, sample_bootstrap_results_dict, default_colors):
        """Test grid is enabled."""
        plot_bootstrap_stability_scatter(
            mock_axes, sample_bootstrap_results_dict, default_colors
        )

        mock_axes.grid.assert_called_once()


# =============================================================================
# COMPLEXITY DISTRIBUTION TESTS
# =============================================================================


class TestPlotComplexityDistribution:
    """Tests for plot_complexity_distribution function."""

    def test_basic_distribution(self, mock_axes, sample_top_models, default_colors):
        """Test basic complexity distribution plot."""
        # Need to mock twinx
        mock_twin = MagicMock()
        mock_axes.twinx.return_value = mock_twin

        plot_complexity_distribution(mock_axes, sample_top_models, default_colors)

        # Verify bar chart
        mock_axes.bar.assert_called_once()
        # Verify twin axis for R²
        mock_axes.twinx.assert_called_once()

    def test_distribution_single_complexity(self, mock_axes, default_colors):
        """Test distribution with single complexity level."""
        models = pd.DataFrame({
            'features': ['a', 'b', 'c'],
            'aic': [100, 102, 104],
            'r_squared': [0.85, 0.83, 0.81],
            'n_features': [2, 2, 2],  # All same complexity
        })

        plot_complexity_distribution(mock_axes, models, default_colors)

        # Should return early if only one complexity level
        mock_axes.bar.assert_not_called()

    def test_distribution_twin_axis_labels(self, mock_axes, sample_top_models, default_colors):
        """Test twin axis labels are set."""
        mock_twin = MagicMock()
        mock_axes.twinx.return_value = mock_twin

        plot_complexity_distribution(mock_axes, sample_top_models, default_colors)

        mock_twin.set_ylabel.assert_called()
        mock_twin.tick_params.assert_called()

    def test_distribution_count_labels(self, mock_axes, sample_top_models, default_colors):
        """Test count labels are added to bars."""
        mock_twin = MagicMock()
        mock_axes.twinx.return_value = mock_twin

        # Mock bar returns
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 100
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_axes.bar.return_value = [mock_bar, mock_bar]

        plot_complexity_distribution(mock_axes, sample_top_models, default_colors)

        # Should add text labels
        assert mock_axes.text.call_count >= 1


# =============================================================================
# RANKING CORRELATION MATRIX TESTS
# =============================================================================


class TestPlotRankingCorrelationMatrix:
    """Tests for plot_ranking_correlation_matrix function."""

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_basic_correlation(
        self, mock_plt, mock_axes, sample_top_models, sample_bootstrap_results_dict
    ):
        """Test basic ranking correlation matrix."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_ranking_correlation_matrix(
            mock_axes, sample_top_models, sample_bootstrap_results_dict
        )

        # Verify imshow was called
        mock_axes.imshow.assert_called_once()
        # Verify colorbar
        mock_plt.colorbar.assert_called_once()

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_correlation_without_bootstrap(self, mock_plt, mock_axes, sample_top_models):
        """Test correlation without bootstrap results."""
        mock_colorbar = MagicMock()
        mock_plt.colorbar.return_value = mock_colorbar

        plot_ranking_correlation_matrix(mock_axes, sample_top_models, [])

        # Should still work with AIC and R² only
        mock_axes.imshow.assert_called_once()

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_correlation_text_annotations(
        self, mock_plt, mock_axes, sample_top_models, sample_bootstrap_results_dict
    ):
        """Test correlation values are annotated."""
        mock_plt.colorbar.return_value = MagicMock()

        plot_ranking_correlation_matrix(
            mock_axes, sample_top_models, sample_bootstrap_results_dict
        )

        # Should have text annotations for correlation values
        assert mock_axes.text.call_count >= 4  # At least 2x2 matrix


# =============================================================================
# FEATURE FREQUENCY TESTS
# =============================================================================


class TestPlotFeatureFrequency:
    """Tests for plot_feature_frequency function."""

    def test_basic_frequency(self, mock_axes, sample_top_models, default_colors):
        """Test basic feature frequency plot."""
        plot_feature_frequency(mock_axes, sample_top_models, default_colors)

        # Verify horizontal bar chart
        mock_axes.barh.assert_called_once()
        # Verify labels
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_frequency_empty_models(self, mock_axes, default_colors):
        """Test frequency with empty models."""
        empty_models = pd.DataFrame({
            'features': [],
            'aic': [],
        })

        plot_feature_frequency(mock_axes, empty_models, default_colors)

        # Should return early
        mock_axes.barh.assert_not_called()

    def test_frequency_sorted(self, mock_axes, default_colors):
        """Test features are sorted by frequency."""
        models = pd.DataFrame({
            'features': ['a+b', 'a+c', 'a+d'],  # 'a' appears 3 times
            'aic': [100, 102, 104],
        })

        plot_feature_frequency(mock_axes, models, default_colors)

        # Check ytick labels - 'a' should be first (most frequent)
        call_args = mock_axes.set_yticklabels.call_args
        labels = call_args[0][0]
        assert labels[0] == 'a'

    def test_frequency_truncates_names(self, mock_axes, default_colors):
        """Test long feature names are truncated."""
        models = pd.DataFrame({
            'features': ['this_is_a_very_long_feature_name_exceeding_limit'],
            'aic': [100],
        })

        plot_feature_frequency(mock_axes, models, default_colors)

        call_args = mock_axes.set_yticklabels.call_args
        labels = call_args[0][0]
        # Truncated to 20 chars + '...'
        assert '...' in labels[0]

    def test_frequency_labels(self, mock_axes, sample_top_models, default_colors):
        """Test frequency count labels are added."""
        mock_bar = MagicMock()
        mock_bar.get_y.return_value = 0
        mock_bar.get_height.return_value = 0.8
        mock_axes.barh.return_value = [mock_bar] * 10

        plot_feature_frequency(mock_axes, sample_top_models, default_colors)

        # Should add text labels for each bar
        assert mock_axes.text.call_count > 0


# =============================================================================
# DECISION SUMMARY TESTS
# =============================================================================


class TestPlotDecisionSummary:
    """Tests for plot_decision_summary function."""

    def test_basic_summary(
        self, mock_axes, sample_aic_results, sample_bootstrap_results_dict, default_colors
    ):
        """Test basic decision summary plot."""
        mock_summary_func = MagicMock(return_value='Summary text')

        plot_decision_summary(
            mock_axes,
            sample_aic_results,
            [],
            sample_bootstrap_results_dict,
            default_colors,
            mock_summary_func
        )

        # Axis should be turned off
        mock_axes.axis.assert_called_with('off')
        # Summary function should be called
        mock_summary_func.assert_called_once()
        # Text should be added
        mock_axes.text.assert_called_once()

    def test_summary_calls_function_correctly(
        self, mock_axes, sample_aic_results, sample_bootstrap_results_dict, default_colors
    ):
        """Test summary function is called with correct arguments."""
        mock_summary_func = MagicMock(return_value='Summary')
        info_criteria = [{'key': 'value'}]

        plot_decision_summary(
            mock_axes,
            sample_aic_results,
            info_criteria,
            sample_bootstrap_results_dict,
            default_colors,
            mock_summary_func
        )

        # Verify correct arguments passed
        mock_summary_func.assert_called_once_with(
            sample_aic_results, info_criteria, sample_bootstrap_results_dict
        )

    def test_summary_text_styling(
        self, mock_axes, sample_aic_results, sample_bootstrap_results_dict, default_colors
    ):
        """Test text has correct styling."""
        mock_summary_func = MagicMock(return_value='Summary')

        plot_decision_summary(
            mock_axes,
            sample_aic_results,
            [],
            sample_bootstrap_results_dict,
            default_colors,
            mock_summary_func
        )

        call_args = mock_axes.text.call_args
        # Check font family is monospace
        assert call_args[1]['fontfamily'] == 'monospace'
        # Check bbox uses light gray
        assert call_args[1]['bbox']['facecolor'] == default_colors['light_gray']


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases across scatter plot functions."""

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_aic_scatter_single_model(self, mock_plt, mock_axes, default_colors):
        """Test AIC scatter with single model."""
        mock_plt.colorbar.return_value = MagicMock()

        models = pd.DataFrame({
            'features': ['single'],
            'aic': [100],
            'r_squared': [0.85],
            'n_features': [1],
        })

        plot_aic_vs_r2_scatter(mock_axes, models, default_colors)

        # Should still create scatter
        assert mock_axes.scatter.call_count >= 1

    def test_bootstrap_scatter_with_namedtuple(self, mock_axes, default_colors):
        """Test bootstrap scatter with namedtuple format."""
        BootstrapResult = namedtuple(
            'BootstrapResult', ['model_features', 'stability_metrics']
        )
        results = [
            BootstrapResult('model_a', {'aic_cv': 0.05, 'successful_fit_rate': 0.95}),
            BootstrapResult('model_b', {'aic_cv': 0.08, 'successful_fit_rate': 0.90}),
        ]

        plot_bootstrap_stability_scatter(mock_axes, results, default_colors)

        # Should still work with namedtuple format
        mock_axes.scatter.assert_called_once()

    @patch('src.visualization.comparison_scatter_plots.plt')
    def test_ranking_correlation_single_ranking(self, mock_plt, mock_axes):
        """Test correlation with insufficient ranking data."""
        import warnings
        models = pd.DataFrame({
            'features': ['a'],
            'aic': [100],
            'r_squared': [0.85],
        })

        # Suppress expected numpy warnings for edge case (single-element correlation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plot_ranking_correlation_matrix(mock_axes, models, [])

        # Should still attempt to plot
        # (at least AIC and R² rankings)
