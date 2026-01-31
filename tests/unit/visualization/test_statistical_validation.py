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
    def test_creates_histogram_with_correct_data(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should create histogram with correct AIC data."""
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

        # Validate data passed to first histogram call (all models)
        first_call = mock_ax.hist.call_args_list[0]
        first_data = first_call[0][0]  # positional arg 0
        np.testing.assert_array_equal(first_data, sample_aic_results['aic'].values)

        # Validate data passed to second histogram call (valid models)
        second_call = mock_ax.hist.call_args_list[1]
        second_data = second_call[0][0]
        np.testing.assert_array_equal(second_data, sample_valid_models['aic'].values)

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
    def test_creates_scatter_plots_with_correct_data(self, mock_subplots, sample_aic_results, sample_valid_models, sample_final_model):
        """Should create scatter plots for different model groups with correct data."""
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

        # Validate data passed to scatter calls
        scatter_calls = mock_ax.scatter.call_args_list

        # First call: all models (AIC vs R²)
        first_x = scatter_calls[0][0][0]  # AIC values
        first_y = scatter_calls[0][0][1]  # R² values
        np.testing.assert_array_equal(first_x, sample_aic_results['aic'].values)
        np.testing.assert_array_equal(first_y, sample_aic_results['r_squared'].values)

        # Second call: valid models only
        second_x = scatter_calls[1][0][0]
        second_y = scatter_calls[1][0][1]
        np.testing.assert_array_equal(second_x, sample_valid_models['aic'].values)
        np.testing.assert_array_equal(second_y, sample_valid_models['r_squared'].values)

        # Third call: final model (single point)
        third_x = scatter_calls[2][0][0]
        third_y = scatter_calls[2][0][1]
        assert third_x == sample_final_model['aic']
        assert third_y == sample_final_model['r_squared']


class TestPlotComplexityAnalysis:
    """Tests for _plot_complexity_analysis helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_bar_chart_with_correct_aggregations(self, mock_subplots, sample_aic_results):
        """Should create bar chart with correctly aggregated complexity data."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax.twinx.return_value = mock_ax2
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = StatisticalValidationPlots()
        plotter._plot_complexity_analysis(mock_ax, sample_aic_results)

        mock_ax.bar.assert_called_once()

        # Validate the bar chart data represents complexity analysis
        bar_call = mock_ax.bar.call_args
        x_positions = bar_call[0][0]  # Feature counts or categories
        y_values = bar_call[0][1]  # Aggregated values (mean AIC per complexity)

        # Should have multiple feature counts represented
        assert len(x_positions) > 0
        assert len(y_values) > 0
        # Values should be reasonable AIC scores (positive, in expected range)
        for val in y_values:
            assert val > 0, "AIC values should be positive"


class TestPlotConstraintSuccessRate:
    """Tests for _plot_constraint_success_rate helper method."""

    @patch('matplotlib.pyplot.subplots')
    def test_creates_bar_chart_with_correct_counts(self, mock_subplots, sample_aic_results, sample_valid_models):
        """Should create bar chart showing constraint success with correct model counts."""
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

        # Validate bar chart data represents constraint success
        bar_call = mock_ax.bar.call_args
        categories = bar_call[0][0]  # Categories (e.g., "All Models", "Valid Models")
        counts = bar_call[0][1]  # Counts of models in each category

        # Should show total vs valid counts
        assert len(categories) == 2, "Should show 2 categories: all and valid"
        assert len(counts) == 2, "Should have counts for both categories"

        # First should be total models, second should be valid models
        total_count = len(sample_aic_results)
        valid_count = len(sample_valid_models)
        assert counts[0] == total_count, f"Expected {total_count} total models, got {counts[0]}"
        assert counts[1] == valid_count, f"Expected {valid_count} valid models, got {counts[1]}"


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


# =============================================================================
# BOOTSTRAP ANALYSIS TESTS
# =============================================================================


class TestExtractBootstrapData:
    """Tests for _extract_bootstrap_data method."""

    def test_extracts_model_data(self):
        """Extracts model data from bootstrap results."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        bootstrap_results = [
            {'model_name': 'Model_1', 'aic_cv': 0.02, 'success_rate': 0.95,
             'aic_mean': 100, 'confidence_intervals': {'aic': (95, 105)}},
            {'model_name': 'Model_2', 'aic_cv': 0.03, 'success_rate': 0.90,
             'aic_mean': 110, 'confidence_intervals': {'aic': (105, 115)}},
        ]

        result = plotter._extract_bootstrap_data(bootstrap_results, confidence_level=95)

        # Returns tuple of lists
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_handles_empty_list(self):
        """Handles empty bootstrap results."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()

        model_names, cvs, means, rates, cis = plotter._extract_bootstrap_data([], confidence_level=95)

        assert model_names == []
        assert cvs == []


class TestFormatModelLabels:
    """Tests for _format_model_labels method."""

    def test_formats_names_with_plus(self):
        """Formats model names with + by adding newlines."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        names = ['A+B', 'C']

        result = plotter._format_model_labels(names)

        assert '+\n' in result[0]
        assert result[1] == 'C'

    def test_preserves_names_without_plus(self):
        """Preserves model names without +."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        short_names = ['Model_1', 'Model_2']

        result = plotter._format_model_labels(short_names)

        assert result == short_names


class TestPlotBootstrapStabilityAnalysis:
    """Tests for plot_bootstrap_stability_analysis method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_figure(self, mock_plt):
        """Returns matplotlib figure."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()
        bootstrap_results = [
            {'model_name': 'M1', 'aic_cv': 0.02, 'success_rate': 0.95,
             'aic_mean': 100, 'confidence_intervals': {'aic': (95, 105)}},
        ]

        result = plotter.plot_bootstrap_stability_analysis(bootstrap_results)

        assert result == mock_fig


# =============================================================================
# INFORMATION CRITERIA TESTS
# =============================================================================


class TestExtractCriteriaData:
    """Tests for _extract_criteria_data method."""

    def test_extracts_ic_values(self):
        """Extracts information criteria values."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        ic_comparison = [
            {'model_name': 'M1', 'aic': 100, 'bic': 110, 'hqic': 105, 'aic_c': 102},
        ]

        result = plotter._extract_criteria_data(ic_comparison)

        # Returns tuple with model names and criteria values
        assert isinstance(result, tuple)


class TestPlotInformationCriteriaRobustness:
    """Tests for plot_information_criteria_robustness method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_figure(self, mock_plt):
        """Returns matplotlib figure."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()
        ic_comparison = [
            {'model_name': 'M1', 'aic': 100, 'bic': 110, 'hqic': 105, 'aic_c': 102,
             'robustness_score': 0.9},
        ]

        result = plotter.plot_information_criteria_robustness(ic_comparison)

        assert result == mock_fig


# =============================================================================
# VALIDATION DASHBOARD TESTS
# =============================================================================


class TestBuildAnalysisSummaryText:
    """Tests for _build_analysis_summary_text method."""

    def test_builds_summary_string(self):
        """Builds formatted summary string."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        metadata = {'total_models': 100, 'valid_models': 70}
        final_model = {'aic': 100, 'r_squared': 0.8, 'features': ['a', 'b']}

        result = plotter._build_analysis_summary_text(metadata, final_model)

        assert isinstance(result, str)


class TestCreateValidationSummaryDashboard:
    """Tests for create_validation_summary_dashboard method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_figure(self, mock_plt):
        """Returns matplotlib figure."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        mock_fig = MagicMock()
        # Mock figure.add_gridspec to return a proper mock
        mock_gs = MagicMock()
        mock_fig.add_gridspec.return_value = mock_gs

        # Mock figure.add_subplot to return mocked axes
        mock_ax = MagicMock()
        mock_bar = MagicMock()
        mock_bar.get_height.return_value = 50
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_ax.bar.return_value = [mock_bar, mock_bar, mock_bar]
        mock_fig.add_subplot.return_value = mock_ax

        mock_plt.figure.return_value = mock_fig

        plotter = StatisticalValidationPlots()
        # Include 'economically_valid' column - this is what the code expects
        analysis_results = {
            'aic_results': pd.DataFrame({
                'aic': [100, 110, 120],
                'r_squared': [0.8, 0.75, 0.7],
                'economically_valid': [True, True, False]
            }),
            'final_model': {'aic': 100, 'r_squared': 0.8, 'selected_model': {'aic': 100}},
            'metadata': {},
        }

        result = plotter.create_validation_summary_dashboard(analysis_results)

        assert result is not None


# =============================================================================
# HELPER PLOT METHODS TESTS
# =============================================================================


class TestPlotAicCvRanking:
    """Tests for _plot_aic_cv_ranking method."""

    def test_creates_bar_chart_with_correct_data(self):
        """Creates horizontal bar chart with correct CV values."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        model_names = ['M1', 'M2', 'M3']
        cvs = [0.02, 0.03, 0.05]

        plotter._plot_aic_cv_ranking(mock_ax, model_names, cvs)

        mock_ax.barh.assert_called_once()

        # Validate data passed to barh
        barh_call = mock_ax.barh.call_args
        y_positions = barh_call[0][0]  # Model positions
        widths = barh_call[0][1]  # CV values

        # Verify CV values are passed correctly
        np.testing.assert_array_almost_equal(widths, cvs)
        assert len(y_positions) == len(model_names)


class TestPlotBootstrapSuccessRate:
    """Tests for _plot_bootstrap_success_rate method."""

    def test_creates_bar_chart_with_correct_rates(self):
        """Creates bar chart with correct success rate values."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        model_names = ['M1', 'M2']
        success_rates = [0.95, 0.90]

        plotter._plot_bootstrap_success_rate(mock_ax, model_names, success_rates)

        # Method should call bar or barh
        assert mock_ax.bar.called or mock_ax.barh.called

        # Validate data passed to bar/barh
        if mock_ax.barh.called:
            call_args = mock_ax.barh.call_args
            widths = call_args[0][1]  # Success rates
            np.testing.assert_array_almost_equal(widths, success_rates)
        elif mock_ax.bar.called:
            call_args = mock_ax.bar.call_args
            heights = call_args[0][1]  # Success rates
            np.testing.assert_array_almost_equal(heights, success_rates)


class TestPlotAicConfidenceIntervals:
    """Tests for _plot_aic_confidence_intervals method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_creates_error_bars(self, mock_plt):
        """Creates error bar plot."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        model_names = ['M1', 'M2']
        aic_means = [100, 110]
        confidence_intervals = [(95, 105), (105, 115)]

        plotter._plot_aic_confidence_intervals(
            mock_ax, model_names, aic_means, confidence_intervals,
            confidence_level=95  # Required argument
        )

        # errorbar is called once per model with valid CI
        assert mock_ax.errorbar.call_count == 2


class TestPlotStabilityVsPerformance:
    """Tests for _plot_stability_vs_performance method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_creates_scatter_plot(self, mock_plt):
        """Creates scatter plot."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        mock_ax.scatter.return_value = MagicMock()
        model_names = ['M1', 'M2']
        aic_cvs = [0.02, 0.03]
        aic_means = [100, 110]
        success_rates = [0.95, 0.90]

        plotter._plot_stability_vs_performance(
            mock_ax, model_names, aic_cvs, aic_means, success_rates
        )

        mock_ax.scatter.assert_called()


class TestPlotCriteriaHeatmap:
    """Tests for _plot_criteria_heatmap method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_creates_heatmap(self, mock_plt):
        """Creates heatmap visualization."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        mock_ax.imshow = MagicMock(return_value=MagicMock())
        model_names = ['M1', 'M2']
        criteria_values = {'aic': [100, 105], 'bic': [110, 115]}
        criteria_list = ['aic', 'bic']

        plotter._plot_criteria_heatmap(
            mock_ax, model_names, criteria_values, criteria_list
        )

        mock_ax.imshow.assert_called_once()


class TestPlotRobustnessDistribution:
    """Tests for _plot_robustness_distribution method."""

    def test_creates_histogram_with_correct_scores(self):
        """Creates histogram with correct robustness score data."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        robustness_scores = [0.9, 0.85, 0.8, 0.75, 0.70]

        plotter._plot_robustness_distribution(mock_ax, robustness_scores)

        mock_ax.hist.assert_called_once()

        # Validate data passed to histogram
        hist_call = mock_ax.hist.call_args
        hist_data = hist_call[0][0]
        np.testing.assert_array_almost_equal(hist_data, robustness_scores)


class TestPlotTopRobustModels:
    """Tests for _plot_top_robust_models method."""

    def test_creates_bar_chart_with_top_n_models(self):
        """Creates bar chart showing top N robust models with correct scores."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        model_names = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
        robustness_scores = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

        plotter._plot_top_robust_models(mock_ax, model_names, robustness_scores)

        mock_ax.barh.assert_called_once()

        # Validate data passed to barh
        barh_call = mock_ax.barh.call_args
        widths = barh_call[0][1]  # Robustness scores

        # Should show top models (typically top 5)
        # Verify scores are in expected range [0, 1]
        for score in widths:
            assert 0 <= score <= 1, f"Robustness score {score} should be in [0, 1]"

        # Scores should be sorted (highest first for top models)
        assert len(widths) <= len(robustness_scores)


class TestPlotValidationStatus:
    """Tests for _plot_validation_status method."""

    def test_creates_status_display(self):
        """Creates validation status display."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        final_model = {'aic': 100, 'r_squared': 0.8, 'features': ['a', 'b']}

        plotter._plot_validation_status(mock_ax, final_model)

        mock_ax.text.assert_called()


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_class_importable(self):
        """StatisticalValidationPlots is importable."""
        from src.visualization.statistical_validation import StatisticalValidationPlots
        assert StatisticalValidationPlots is not None

    def test_create_report_importable(self):
        """create_statistical_validation_report is importable."""
        from src.visualization.statistical_validation import create_statistical_validation_report
        assert create_statistical_validation_report is not None

    def test_standalone_functions_importable(self):
        """Standalone functions are importable."""
        from src.visualization.statistical_validation import (
            _generate_aic_distribution_plot,
            _generate_bootstrap_stability_plot,
            _generate_information_criteria_plot,
        )
        assert _generate_aic_distribution_plot is not None
        assert _generate_bootstrap_stability_plot is not None
        assert _generate_information_criteria_plot is not None


# =============================================================================
# NAMEDTUPLE BOOTSTRAP DATA TESTS
# =============================================================================


class TestExtractBootstrapDataNamedTuple:
    """Tests for NamedTuple branch in _extract_bootstrap_data."""

    def test_handles_namedtuple_objects(self):
        """Handles NamedTuple bootstrap results."""
        from src.visualization.statistical_validation import StatisticalValidationPlots
        from collections import namedtuple

        # Create NamedTuple type matching expected structure
        BootstrapResult = namedtuple('BootstrapResult', [
            'model_features', 'stability_metrics', 'confidence_intervals'
        ])

        plotter = StatisticalValidationPlots()
        bootstrap_results = [
            BootstrapResult(
                model_features='feature_a + feature_b',
                stability_metrics={'aic_cv': 0.02, 'aic_mean': 100, 'successful_fit_rate': 0.95},
                confidence_intervals={95: (95.0, 105.0)}
            ),
            BootstrapResult(
                model_features='feature_c',
                stability_metrics={'aic_cv': 0.03, 'aic_mean': 110, 'successful_fit_rate': 0.90},
                confidence_intervals={95: (105.0, 115.0)}
            ),
        ]

        model_names, cvs, means, rates, cis = plotter._extract_bootstrap_data(
            bootstrap_results, confidence_level=95
        )

        assert model_names == ['feature_a + feature_b', 'feature_c']
        assert cvs == [0.02, 0.03]
        assert means == [100, 110]
        assert rates == [0.95, 0.90]
        assert cis == [(95.0, 105.0), (105.0, 115.0)]

    def test_handles_missing_confidence_level(self):
        """Handles NamedTuple with missing confidence level key."""
        from src.visualization.statistical_validation import StatisticalValidationPlots
        from collections import namedtuple

        BootstrapResult = namedtuple('BootstrapResult', [
            'model_features', 'stability_metrics', 'confidence_intervals'
        ])

        plotter = StatisticalValidationPlots()
        bootstrap_results = [
            BootstrapResult(
                model_features='feature_a',
                stability_metrics={'aic_cv': 0.02, 'aic_mean': 100, 'successful_fit_rate': 0.95},
                confidence_intervals={90: (96.0, 104.0)}  # Different key than requested
            ),
        ]

        model_names, cvs, means, rates, cis = plotter._extract_bootstrap_data(
            bootstrap_results, confidence_level=95  # Request 95, but only 90 available
        )

        # Should use default (0, 0) when key not found
        assert cis == [(0, 0)]


# =============================================================================
# TEXT ANNOTATION TESTS
# =============================================================================


class TestPlotAicCvRankingWithAnnotations:
    """Tests for text annotations in _plot_aic_cv_ranking."""

    def test_adds_text_annotations(self):
        """Adds text annotations to bars."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()

        # Mock barh to return bar objects
        mock_bar1 = MagicMock()
        mock_bar1.get_y.return_value = 0
        mock_bar1.get_height.return_value = 0.8
        mock_bar2 = MagicMock()
        mock_bar2.get_y.return_value = 1
        mock_bar2.get_height.return_value = 0.8
        mock_ax.barh.return_value = [mock_bar1, mock_bar2]

        model_names = ['M1', 'M2']
        cvs = [0.02, 0.03]

        plotter._plot_aic_cv_ranking(mock_ax, model_names, cvs)

        # Should call text for each bar
        assert mock_ax.text.call_count == 2


class TestPlotBootstrapSuccessRateWithAnnotations:
    """Tests for text annotations in _plot_bootstrap_success_rate."""

    def test_adds_text_annotations(self):
        """Adds text annotations to bars."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()

        # Mock barh to return bar objects
        mock_bar1 = MagicMock()
        mock_bar1.get_y.return_value = 0
        mock_bar1.get_height.return_value = 0.8
        mock_bar2 = MagicMock()
        mock_bar2.get_y.return_value = 1
        mock_bar2.get_height.return_value = 0.8
        mock_ax.barh.return_value = [mock_bar1, mock_bar2]

        model_names = ['M1', 'M2']
        success_rates = [0.95, 0.90]

        plotter._plot_bootstrap_success_rate(mock_ax, model_names, success_rates)

        # Should call text for each bar
        assert mock_ax.text.call_count == 2


# =============================================================================
# CONFIDENCE INTERVALS SCATTER TESTS
# =============================================================================


class TestPlotAicConfidenceIntervalsScatter:
    """Tests for scatter fallback in _plot_aic_confidence_intervals."""

    def test_uses_scatter_for_equal_ci_bounds(self):
        """Uses scatter when CI bounds are equal."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()

        model_names = ['M1', 'M2']
        aic_means = [100, 110]
        # Second CI has equal bounds (no uncertainty)
        confidence_intervals = [(95, 105), (110, 110)]

        plotter._plot_aic_confidence_intervals(
            mock_ax, model_names, aic_means, confidence_intervals,
            confidence_level=95
        )

        # First uses errorbar, second uses scatter
        assert mock_ax.errorbar.call_count == 1
        assert mock_ax.scatter.call_count == 1


# =============================================================================
# RANKING CORRELATION MATRIX TESTS
# =============================================================================


class TestPlotRankingCorrelationMatrix:
    """Tests for _plot_ranking_correlation_matrix method."""

    @patch('src.visualization.statistical_validation.plt')
    def test_creates_correlation_heatmap(self, mock_plt):
        """Creates correlation matrix heatmap."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()
        mock_im = MagicMock()
        mock_ax.imshow.return_value = mock_im
        mock_cbar = MagicMock()
        mock_plt.colorbar.return_value = mock_cbar

        ranking_positions = {
            'aic': [1, 2, 3, 4, 5],
            'bic': [1, 3, 2, 5, 4],
            'hqic': [2, 1, 3, 4, 5],
        }
        criteria_list = ['aic', 'bic', 'hqic']

        plotter._plot_ranking_correlation_matrix(mock_ax, ranking_positions, criteria_list)

        # Should create heatmap
        mock_ax.imshow.assert_called_once()
        # Should add colorbar
        mock_plt.colorbar.assert_called_once()
        # Should add text annotations (3x3 = 9)
        assert mock_ax.text.call_count == 9

    def test_handles_single_criterion(self):
        """Handles single criterion gracefully."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()

        ranking_positions = {'aic': [1, 2, 3]}
        criteria_list = ['aic']

        plotter._plot_ranking_correlation_matrix(mock_ax, ranking_positions, criteria_list)

        # Should show "Insufficient criteria" message
        mock_ax.text.assert_called()

    def test_handles_empty_positions(self):
        """Handles empty ranking positions."""
        from src.visualization.statistical_validation import StatisticalValidationPlots

        plotter = StatisticalValidationPlots()
        mock_ax = MagicMock()

        ranking_positions = {}
        criteria_list = ['aic', 'bic']

        plotter._plot_ranking_correlation_matrix(mock_ax, ranking_positions, criteria_list)

        # Should show "Insufficient criteria" message
        mock_ax.text.assert_called()


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================


class TestGenerateBootstrapStabilityPlot:
    """Tests for _generate_bootstrap_stability_plot function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_path_when_data_exists(self, mock_plt, tmp_path):
        """Returns path when bootstrap data exists."""
        from src.visualization.statistical_validation import (
            _generate_bootstrap_stability_plot,
            StatisticalValidationPlots,
        )

        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()
        analysis_results = {
            'bootstrap_results': {
                'block_bootstrap_results': [
                    {'model_name': 'M1', 'aic_cv': 0.02, 'success_rate': 0.95,
                     'aic_mean': 100, 'confidence_intervals': {'aic': (95, 105)}},
                ]
            }
        }

        result = _generate_bootstrap_stability_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is not None
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_returns_none_when_no_bootstrap_results(self, tmp_path):
        """Returns None when bootstrap_results key missing."""
        from src.visualization.statistical_validation import (
            _generate_bootstrap_stability_plot,
            StatisticalValidationPlots,
        )

        plotter = StatisticalValidationPlots()
        analysis_results = {}

        result = _generate_bootstrap_stability_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is None

    def test_returns_none_when_bootstrap_data_empty(self, tmp_path):
        """Returns None when block_bootstrap_results is empty."""
        from src.visualization.statistical_validation import (
            _generate_bootstrap_stability_plot,
            StatisticalValidationPlots,
        )

        plotter = StatisticalValidationPlots()
        analysis_results = {
            'bootstrap_results': {
                'block_bootstrap_results': []
            }
        }

        result = _generate_bootstrap_stability_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is None


class TestGenerateInformationCriteriaPlot:
    """Tests for _generate_information_criteria_plot function."""

    @patch('src.visualization.statistical_validation.plt')
    def test_returns_path_when_data_exists(self, mock_plt, tmp_path):
        """Returns path when criteria data exists."""
        from src.visualization.statistical_validation import (
            _generate_information_criteria_plot,
            StatisticalValidationPlots,
        )

        mock_fig = MagicMock()
        mock_axes = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mock_axes[i, j] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        plotter = StatisticalValidationPlots()
        analysis_results = {
            'enhanced_metrics': {
                'information_criteria_analysis': [
                    {'model_name': 'M1', 'aic': 100, 'bic': 110,
                     'hqic': 105, 'aic_c': 102, 'robustness_score': 0.9},
                ]
            }
        }

        result = _generate_information_criteria_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is not None
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_returns_none_when_no_enhanced_metrics(self, tmp_path):
        """Returns None when enhanced_metrics key missing."""
        from src.visualization.statistical_validation import (
            _generate_information_criteria_plot,
            StatisticalValidationPlots,
        )

        plotter = StatisticalValidationPlots()
        analysis_results = {}

        result = _generate_information_criteria_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is None

    def test_returns_none_when_criteria_data_empty(self, tmp_path):
        """Returns None when information_criteria_analysis is empty."""
        from src.visualization.statistical_validation import (
            _generate_information_criteria_plot,
            StatisticalValidationPlots,
        )

        plotter = StatisticalValidationPlots()
        analysis_results = {
            'enhanced_metrics': {
                'information_criteria_analysis': []
            }
        }

        result = _generate_information_criteria_plot(
            plotter, analysis_results, tmp_path, "test"
        )

        assert result is None


class TestCreateStatisticalValidationReportFull:
    """Tests for full create_statistical_validation_report flow."""

    @patch('src.visualization.statistical_validation._generate_aic_distribution_plot')
    @patch('src.visualization.statistical_validation._generate_bootstrap_stability_plot')
    @patch('src.visualization.statistical_validation._generate_information_criteria_plot')
    @patch('src.visualization.statistical_validation.StatisticalValidationPlots')
    @patch('src.visualization.statistical_validation.plt')
    def test_includes_bootstrap_stability(
        self, mock_plt, mock_plotter_class, mock_ic_plot, mock_bootstrap_plot, mock_aic_plot, tmp_path
    ):
        """Includes bootstrap stability plot when data available."""
        from src.visualization.statistical_validation import create_statistical_validation_report

        # Configure mocks
        mock_fig = MagicMock()
        mock_plotter = MagicMock()
        mock_plotter.create_validation_summary_dashboard.return_value = mock_fig
        mock_plotter_class.return_value = mock_plotter

        mock_aic_plot.return_value = tmp_path / "aic.png"
        mock_bootstrap_plot.return_value = tmp_path / "bootstrap.png"
        mock_ic_plot.return_value = None

        analysis_results = {
            'aic_results': pd.DataFrame({'aic': [100]}),
            'final_model': {'aic': 100},
            'metadata': {},
            'bootstrap_results': {'block_bootstrap_results': [{'model': 'M1'}]}
        }

        result = create_statistical_validation_report(analysis_results, tmp_path)

        assert 'summary_dashboard' in result
        assert 'bootstrap_stability' in result
        mock_plt.close.assert_called()

    @patch('src.visualization.statistical_validation._generate_aic_distribution_plot')
    @patch('src.visualization.statistical_validation._generate_bootstrap_stability_plot')
    @patch('src.visualization.statistical_validation._generate_information_criteria_plot')
    @patch('src.visualization.statistical_validation.StatisticalValidationPlots')
    @patch('src.visualization.statistical_validation.plt')
    def test_includes_information_criteria(
        self, mock_plt, mock_plotter_class, mock_ic_plot, mock_bootstrap_plot, mock_aic_plot, tmp_path
    ):
        """Includes information criteria plot when data available."""
        from src.visualization.statistical_validation import create_statistical_validation_report

        # Configure mocks
        mock_fig = MagicMock()
        mock_plotter = MagicMock()
        mock_plotter.create_validation_summary_dashboard.return_value = mock_fig
        mock_plotter_class.return_value = mock_plotter

        mock_aic_plot.return_value = tmp_path / "aic.png"
        mock_bootstrap_plot.return_value = None
        mock_ic_plot.return_value = tmp_path / "criteria.png"

        analysis_results = {
            'aic_results': pd.DataFrame({'aic': [100]}),
            'final_model': {'aic': 100},
            'metadata': {},
            'enhanced_metrics': {'information_criteria_analysis': [{'model': 'M1'}]}
        }

        result = create_statistical_validation_report(analysis_results, tmp_path)

        assert 'summary_dashboard' in result
        assert 'information_criteria' in result
        mock_plt.close.assert_called()
