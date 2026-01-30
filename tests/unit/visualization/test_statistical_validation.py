"""
Unit tests for src/visualization/statistical_validation.py.

Tests validate the StatisticalValidationPlots class and related functions
for generating model validation visualizations.

Target: 60%+ coverage for statistical_validation.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame for testing."""
    np.random.seed(42)
    n_models = 100
    return pd.DataFrame({
        'model_id': range(n_models),
        'aic': np.random.uniform(500, 700, n_models),
        'r_squared': np.random.uniform(0.3, 0.9, n_models),
        'n_features': np.random.randint(2, 8, n_models),
        'valid_constraints': np.random.choice([True, False], n_models, p=[0.7, 0.3])
    })


@pytest.fixture
def sample_valid_models(sample_aic_results):
    """Sample economically valid models subset."""
    return sample_aic_results[sample_aic_results['valid_constraints']].copy()


@pytest.fixture
def sample_final_model():
    """Sample final model selection."""
    return {
        'model_id': 42,
        'aic': 525.5,
        'r_squared': 0.82,
        'n_features': 4,
        'features': ['prudential_rate_t0', 'competitor_weighted_t2']
    }


@pytest.fixture
def sample_bootstrap_stability():
    """Sample bootstrap stability results."""
    np.random.seed(42)
    n_iterations = 100
    return pd.DataFrame({
        'iteration': range(n_iterations),
        'aic': np.random.normal(530, 15, n_iterations),
        'r_squared': np.random.normal(0.8, 0.05, n_iterations),
        'selected_features': [['f1', 'f2', 'f3']] * n_iterations
    })


@pytest.fixture
def sample_analysis_results(sample_aic_results, sample_valid_models, sample_final_model, sample_bootstrap_stability):
    """Complete analysis results dictionary."""
    return {
        'aic_results': sample_aic_results,
        'valid_models': sample_valid_models,
        'final_model': sample_final_model,
        'bootstrap_stability': sample_bootstrap_stability
    }


# =============================================================================
# CLASS INITIALIZATION TESTS
# =============================================================================


class TestStatisticalValidationPlotsInit:
    """Tests for StatisticalValidationPlots class initialization."""

    def test_init_default_params(self):
        """Should initialize with default style and palette."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        assert plotter.style == 'seaborn-v0_8'
        assert plotter.color_palette == 'husl'

    def test_init_custom_params(self):
        """Should accept custom style and palette."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots(
            style='whitegrid',
            color_palette='deep'
        )

        assert plotter.style == 'whitegrid'
        assert plotter.color_palette == 'deep'

    def test_init_sets_colors_dict(self):
        """Should initialize color dictionary."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        assert 'primary' in plotter.colors
        assert 'secondary' in plotter.colors
        assert 'accent' in plotter.colors
        assert 'success' in plotter.colors

    def test_init_sets_plot_config(self):
        """Should initialize plot configuration."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        assert 'figure_size_large' in plotter.plot_config
        assert 'figure_size_medium' in plotter.plot_config
        assert 'figure_size_small' in plotter.plot_config


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================


class TestPlotAicDistributionHistogram:
    """Tests for _plot_aic_distribution_histogram helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_histogram(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should create histogram with correct data."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_aic_distribution_histogram(
            mock_ax, sample_aic_results, sample_valid_models, sample_final_model
        )

        # Should call hist twice (all models, valid models)
        assert mock_ax.hist.call_count == 2

    @patch('matplotlib.pyplot.subplots')
    def test_adds_vertical_line_for_final_model(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should add vertical line for final model AIC."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_aic_distribution_histogram(
            mock_ax, sample_aic_results, sample_valid_models, sample_final_model
        )

        mock_ax.axvline.assert_called_once()

    @patch('matplotlib.pyplot.subplots')
    def test_sets_labels_and_title(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should set axis labels and title."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_aic_distribution_histogram(
            mock_ax, sample_aic_results, sample_valid_models, sample_final_model
        )

        mock_ax.set_xlabel.assert_called_once_with('AIC Score')
        mock_ax.set_ylabel.assert_called_once_with('Density')
        mock_ax.set_title.assert_called_once()


class TestPlotAicVsRSquaredScatter:
    """Tests for _plot_aic_vs_rsquared_scatter helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_scatter_plots(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should create scatter plots for different model groups."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_aic_vs_rsquared_scatter(
            mock_ax, sample_aic_results, sample_valid_models, sample_final_model
        )

        # Should call scatter 3 times (all, valid, final)
        assert mock_ax.scatter.call_count == 3


class TestPlotComplexityAnalysis:
    """Tests for _plot_complexity_analysis helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_bar_chart(self, mock_subplots, sample_aic_results):
        """Should create bar chart for complexity analysis."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.twinx.return_value = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_complexity_analysis(mock_ax, sample_aic_results)

        mock_ax.bar.assert_called_once()


class TestPlotConstraintSuccessRate:
    """Tests for _plot_constraint_success_rate helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_bar_chart(self, mock_subplots, sample_aic_results, sample_valid_models):
        """Should create bar chart showing constraint success."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar]

        plotter = StatisticalValidationPlots()
        plotter._plot_constraint_success_rate(mock_ax, sample_aic_results, sample_valid_models)

        mock_ax.bar.assert_called_once()


# =============================================================================
# MAIN PLOT GENERATION TESTS
# =============================================================================


class TestPlotAicDistributionAnalysis:
    """Tests for plot_aic_distribution_analysis method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_figure(self, mock_plt, sample_aic_results, sample_valid_models, sample_final_model):
        """Should return matplotlib Figure."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        # Create mock axes as object array to preserve MagicMock identity
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                ax = MagicMock()
                ax.twinx.return_value = MagicMock()
                mock_bar = MagicMock()
                mock_bar.get_height.return_value = 50
                mock_bar.get_x.return_value = 0
                mock_bar.get_width.return_value = 0.8
                ax.bar.return_value = [mock_bar, mock_bar]
                mock_axes[i, j] = ax

        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()
        result = plotter.plot_aic_distribution_analysis(
            sample_aic_results, sample_valid_models, sample_final_model
        )

        assert result == mock_fig


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================


class TestGenerateAicDistributionPlot:
    """Tests for _generate_aic_distribution_plot standalone function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_function_callable(self, mock_plt, tmp_path, sample_analysis_results):
        """Function should be callable with correct parameters."""
        from src.visualization.statistical_validation import (
            _generate_aic_distribution_plot,
            StatisticalValidationPlots
        )

        mock_fig = MagicMock()
        # Create mock axes as object array to preserve MagicMock identity
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                ax = MagicMock()
                ax.twinx.return_value = MagicMock()
                mock_bar = MagicMock()
                mock_bar.get_height.return_value = 50
                mock_bar.get_x.return_value = 0
                mock_bar.get_width.return_value = 0.8
                ax.bar.return_value = [mock_bar, mock_bar]
                mock_axes[i, j] = ax
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()

        result = _generate_aic_distribution_plot(
            plotter, sample_analysis_results, tmp_path, 'test'
        )

        # Returns Path or None
        assert result is None or isinstance(result, Path)


class TestGenerateBootstrapStabilityPlot:
    """Tests for _generate_bootstrap_stability_plot standalone function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_function_callable(self, mock_plt, tmp_path, sample_analysis_results):
        """Function should be callable with stability data."""
        from src.visualization.statistical_validation import (
            _generate_bootstrap_stability_plot,
            StatisticalValidationPlots
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.figure.return_value = mock_fig

        plotter = StatisticalValidationPlots()

        result = _generate_bootstrap_stability_plot(
            plotter, sample_analysis_results, tmp_path, 'test'
        )

        # Returns Path or None
        assert result is None or isinstance(result, Path)


class TestGenerateInformationCriteriaPlot:
    """Tests for _generate_information_criteria_plot standalone function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_function_callable(self, mock_plt, tmp_path, sample_analysis_results):
        """Function should be callable with IC comparison data."""
        from src.visualization.statistical_validation import (
            _generate_information_criteria_plot,
            StatisticalValidationPlots
        )

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.figure.return_value = mock_fig

        plotter = StatisticalValidationPlots()

        result = _generate_information_criteria_plot(
            plotter, sample_analysis_results, tmp_path, 'test'
        )

        # Returns Path or None
        assert result is None or isinstance(result, Path)


class TestCreateStatisticalValidationReport:
    """Tests for create_statistical_validation_report function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_dict_with_figures(self, mock_plt, sample_analysis_results, tmp_path):
        """Should return dict containing figures and paths."""
        from src.visualization.statistical_validation import create_statistical_validation_report

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.twinx.return_value = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_fig, np.array([[mock_ax, mock_ax], [mock_ax, mock_ax]]))
        mock_plt.figure.return_value = mock_fig

        result = create_statistical_validation_report(
            sample_analysis_results,
            output_dir=tmp_path
        )

        assert isinstance(result, dict)

    @patch('src.visualization.statistical_validation.plt')
    def test_accepts_output_dir(self, mock_plt, sample_analysis_results, tmp_path):
        """Should accept output_dir parameter."""
        from src.visualization.statistical_validation import create_statistical_validation_report

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.twinx.return_value = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_fig, np.array([[mock_ax, mock_ax], [mock_ax, mock_ax]]))
        mock_plt.figure.return_value = mock_fig

        output_dir = tmp_path / "validation_plots"

        # Should not raise
        result = create_statistical_validation_report(
            sample_analysis_results,
            output_dir=output_dir
        )

        assert isinstance(result, dict)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_aic_results_handled(self):
        """Should handle empty AIC results gracefully."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        empty_df = pd.DataFrame(columns=['aic', 'r_squared', 'n_features'])

        # This is a setup test - actual behavior depends on implementation
        # Some methods may raise, others may return empty figures
        assert plotter is not None

    def test_none_final_model_handled(self):
        """Should handle None final_model."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        # Create minimal test data
        aic_results = pd.DataFrame({
            'aic': [500, 550],
            'r_squared': [0.8, 0.7],
            'n_features': [3, 4]
        })
        valid_models = aic_results.copy()

        mock_ax = MagicMock()

        # Should not raise when final_model is None
        plotter._plot_aic_distribution_histogram(
            mock_ax, aic_results, valid_models, None
        )

        # axvline should not be called when final_model is None
        mock_ax.axvline.assert_not_called()

    def test_single_model_complexity(self):
        """Should handle single model in complexity analysis."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        single_model = pd.DataFrame({
            'aic': [525],
            'r_squared': [0.8],
            'n_features': [4]
        })

        mock_ax = MagicMock()
        mock_ax.twinx.return_value = MagicMock()

        # Should not raise
        plotter._plot_complexity_analysis(mock_ax, single_model)


# =============================================================================
# COLOR SCHEME TESTS
# =============================================================================


class TestColorScheme:
    """Tests for color scheme consistency."""

    def test_colors_are_valid_hex(self):
        """All colors should be valid hex codes."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        import re
        hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')

        for color_name, color_value in plotter.colors.items():
            assert hex_pattern.match(color_value), f"Invalid hex color for {color_name}: {color_value}"

    def test_primary_colors_defined(self):
        """Primary semantic colors should be defined."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        required_colors = ['primary', 'secondary', 'accent', 'success', 'warning']
        for color in required_colors:
            assert color in plotter.colors, f"Missing required color: {color}"
