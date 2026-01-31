"""
Tests for src.visualization.forecasting_atomic_plots module.

Tests atomic plotting operations for time series forecasting:
- Bootstrap forecast plots
- Model benchmark comparison
- MAPE analysis
- Comprehensive analysis dashboard
- Volatility analysis
- Performance summary

Target coverage: 60%+
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.forecasting_atomic_plots import (
    _extract_volatility_metrics,
    _format_comparison_legend,
    _format_forecast_axes,
    _plot_bootstrap_comparison,
    _plot_comparison_panel,
    _plot_forecast_panel,
    _plot_mape_line,
    _plot_mape_panel,
    _plot_metric_bars_with_labels,
    _plot_scatter_reference,
    create_bootstrap_forecast_plot_atomic,
    create_comprehensive_analysis_plot_atomic,
    create_mape_analysis_plot_atomic,
    create_model_benchmark_comparison_atomic,
    create_performance_summary_plot_atomic,
    create_volatility_analysis_plot_atomic,
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
def sample_dates():
    """Sample dates array."""
    return pd.date_range('2025-01-01', periods=10, freq='W').values


@pytest.fixture
def sample_y_true():
    """Sample true values array."""
    return np.array([100, 105, 98, 110, 103, 115, 108, 120, 112, 118])


@pytest.fixture
def sample_bootstrap_data():
    """Sample bootstrap data DataFrame."""
    dates = pd.date_range('2025-01-01', periods=10, freq='W')
    return pd.DataFrame({
        'date': np.tile(dates, 5),
        'y_bootstrap': np.random.normal(100, 10, 50),
        'output': ['Model'] * 50
    })


@pytest.fixture
def sample_combined_bootstrap_data():
    """Sample combined bootstrap data with Model and Benchmark."""
    dates = pd.date_range('2025-01-01', periods=10, freq='W')
    model_data = pd.DataFrame({
        'date': np.tile(dates, 5),
        'y_bootstrap': np.random.normal(100, 10, 50),
        'output': ['Model'] * 50
    })
    benchmark_data = pd.DataFrame({
        'date': np.tile(dates, 5),
        'y_bootstrap': np.random.normal(95, 15, 50),
        'output': ['Benchmark'] * 50
    })
    return pd.concat([model_data, benchmark_data], ignore_index=True)


@pytest.fixture
def sample_forecast_df():
    """Sample forecast DataFrame with all required columns."""
    dates = pd.date_range('2025-01-01', periods=10, freq='W')
    return pd.DataFrame({
        'date': dates,
        'y_true': np.random.normal(100, 10, 10),
        'y_pred_model': np.random.normal(100, 8, 10),
        'y_pred_benchmark': np.random.normal(100, 12, 10),
        '13_week_MAPE_model': np.random.uniform(5, 15, 10),
        '13_week_MAPE_benchmark': np.random.uniform(8, 20, 10),
    })


@pytest.fixture
def sample_volatility_metrics():
    """Sample volatility metrics dictionary."""
    return {
        'model_r2_standard': 0.85,
        'model_r2_weighted': 0.82,
        'model_mape_standard': 8.5,
        'model_mape_weighted': 9.2,
    }


@pytest.fixture
def sample_performance_summary():
    """Sample performance summary dictionary."""
    return {
        'model_r2': 0.85,
        'benchmark_r2': 0.72,
        'model_mape': 8.5,
        'benchmark_mape': 12.3,
    }


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestFormatForecastAxes:
    """Tests for _format_forecast_axes helper."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_basic_formatting(self, mock_plt, mock_axes):
        """Test basic axes formatting."""
        _format_forecast_axes(mock_axes, 'Test Title')

        mock_axes.grid.assert_called_once()
        mock_axes.set_title.assert_called_once()
        mock_axes.set_xlabel.assert_called_once()
        mock_axes.set_ylabel.assert_called_once()
        mock_axes.legend.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_title_set_correctly(self, mock_plt, mock_axes):
        """Test title is set correctly."""
        _format_forecast_axes(mock_axes, 'My Custom Title')

        call_args = mock_axes.set_title.call_args
        assert call_args[0][0] == 'My Custom Title'


class TestPlotScatterReference:
    """Tests for _plot_scatter_reference helper."""

    def test_basic_scatter(self, mock_axes, sample_dates, sample_y_true):
        """Test basic scatter plot creation."""
        _plot_scatter_reference(mock_axes, sample_dates, sample_y_true)

        mock_axes.scatter.assert_called_once()

    def test_custom_label(self, mock_axes, sample_dates, sample_y_true):
        """Test custom label is used."""
        _plot_scatter_reference(mock_axes, sample_dates, sample_y_true, label='Custom Label')

        call_args = mock_axes.scatter.call_args
        assert call_args[1]['label'] == 'Custom Label'


class TestPlotBootstrapComparison:
    """Tests for _plot_bootstrap_comparison helper."""

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_basic_lineplot(self, mock_sns, mock_axes, sample_combined_bootstrap_data):
        """Test seaborn lineplot is called."""
        _plot_bootstrap_comparison(mock_axes, sample_combined_bootstrap_data)

        mock_sns.lineplot.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_correct_columns(self, mock_sns, mock_axes, sample_combined_bootstrap_data):
        """Test correct columns are used."""
        _plot_bootstrap_comparison(mock_axes, sample_combined_bootstrap_data)

        call_args = mock_sns.lineplot.call_args
        assert call_args[1]['x'] == 'date'
        assert call_args[1]['y'] == 'y_bootstrap'
        assert call_args[1]['hue'] == 'output'


class TestFormatComparisonLegend:
    """Tests for _format_comparison_legend helper."""

    def test_legend_labels_mapped(self, mock_axes):
        """Test legend labels are mapped to business-friendly names."""
        mock_axes.get_legend_handles_labels.return_value = (
            ['handle1', 'handle2'],
            ['Model', 'Benchmark']
        )

        _format_comparison_legend(mock_axes)

        mock_axes.legend.assert_called_once()
        call_args = mock_axes.legend.call_args
        labels = call_args[0][1]
        assert 'Bootstrap Ridge Model' in labels
        assert 'Rolling Average Benchmark' in labels


class TestPlotMapeLine:
    """Tests for _plot_mape_line helper."""

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_plots_when_column_exists(self, mock_sns, mock_axes, sample_forecast_df):
        """Test line is plotted when column exists."""
        _plot_mape_line(mock_axes, sample_forecast_df, '13_week_MAPE_model', 'blue', 'Model')

        mock_sns.lineplot.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_no_plot_when_column_missing(self, mock_sns, mock_axes, sample_forecast_df):
        """Test no plot when column doesn't exist."""
        _plot_mape_line(mock_axes, sample_forecast_df, 'nonexistent_column', 'blue', 'Label')

        mock_sns.lineplot.assert_not_called()


class TestExtractVolatilityMetrics:
    """Tests for _extract_volatility_metrics helper."""

    def test_basic_extraction(self, sample_volatility_metrics):
        """Test basic metric extraction."""
        r2_data, mape_data = _extract_volatility_metrics(sample_volatility_metrics)

        assert 'Standard Weighting' in r2_data
        assert 'Volatility Weighting' in r2_data
        assert r2_data['Standard Weighting'] == 0.85
        assert r2_data['Volatility Weighting'] == 0.82

    def test_missing_keys_default_to_zero(self):
        """Test missing keys default to zero."""
        r2_data, mape_data = _extract_volatility_metrics({})

        assert r2_data['Standard Weighting'] == 0
        assert mape_data['Volatility Weighting'] == 0


class TestPlotMetricBarsWithLabels:
    """Tests for _plot_metric_bars_with_labels helper."""

    def test_basic_bars(self, mock_axes):
        """Test basic bar plot creation."""
        data = {'A': 0.5, 'B': 0.7}
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_axes.bar.return_value = [mock_bar, mock_bar]

        _plot_metric_bars_with_labels(mock_axes, data, 'Title', 'Y Label', '{:.2f}', 0.01)

        mock_axes.bar.assert_called_once()
        mock_axes.set_title.assert_called_once()
        mock_axes.set_ylabel.assert_called_once()

    def test_labels_added(self, mock_axes):
        """Test value labels are added to bars."""
        data = {'A': 0.5, 'B': 0.7}
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_axes.bar.return_value = [mock_bar, mock_bar]

        _plot_metric_bars_with_labels(mock_axes, data, 'Title', 'Y Label', '{:.2f}', 0.01)

        # Text called for each bar
        assert mock_axes.text.call_count == 2


class TestPlotComparisonPanel:
    """Tests for _plot_comparison_panel helper."""

    def test_basic_comparison(self, mock_axes):
        """Test basic comparison panel."""
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_axes.bar.return_value = [mock_bar, mock_bar]

        _plot_comparison_panel(mock_axes, 0.85, 0.72, 'Title', 'Y Label', '{:.4f}', 0.01)

        mock_axes.bar.assert_called_once()
        mock_axes.set_title.assert_called_with('Title', fontsize=12)


# =============================================================================
# ATOMIC PLOT CREATION TESTS
# =============================================================================


class TestCreateBootstrapForecastPlotAtomic:
    """Tests for create_bootstrap_forecast_plot_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_returns_figure(self, mock_sns, mock_plt, sample_dates, sample_y_true, sample_bootstrap_data):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        result = create_bootstrap_forecast_plot_atomic(
            sample_dates, sample_y_true, sample_bootstrap_data
        )

        assert result == mock_figure
        mock_figure.tight_layout.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_custom_title(self, mock_sns, mock_plt, sample_dates, sample_y_true, sample_bootstrap_data):
        """Test custom title is used."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        create_bootstrap_forecast_plot_atomic(
            sample_dates, sample_y_true, sample_bootstrap_data,
            title='Custom Title'
        )

        # Title should be set via _format_forecast_axes
        mock_axes.set_title.assert_called()


class TestCreateModelBenchmarkComparisonAtomic:
    """Tests for create_model_benchmark_comparison_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_returns_figure(
        self, mock_sns, mock_plt, sample_dates, sample_y_true, sample_combined_bootstrap_data
    ):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_axes.get_legend_handles_labels.return_value = (['h1', 'h2'], ['Model', 'Benchmark'])
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        result = create_model_benchmark_comparison_atomic(
            sample_dates, sample_y_true, sample_combined_bootstrap_data
        )

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_scatter_and_lineplot_called(
        self, mock_sns, mock_plt, sample_dates, sample_y_true, sample_combined_bootstrap_data
    ):
        """Test both scatter and lineplot are called."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_axes.get_legend_handles_labels.return_value = (['h1'], ['l1'])
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        create_model_benchmark_comparison_atomic(
            sample_dates, sample_y_true, sample_combined_bootstrap_data
        )

        mock_axes.scatter.assert_called_once()
        mock_sns.lineplot.assert_called_once()


class TestCreateMapeAnalysisPlotAtomic:
    """Tests for create_mape_analysis_plot_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_returns_figure(self, mock_sns, mock_plt, sample_forecast_df):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        result = create_mape_analysis_plot_atomic(sample_forecast_df)

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_both_mape_lines_plotted(self, mock_sns, mock_plt, sample_forecast_df):
        """Test both model and benchmark MAPE lines are plotted."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        create_mape_analysis_plot_atomic(sample_forecast_df)

        # Two calls to lineplot (model and benchmark)
        assert mock_sns.lineplot.call_count == 2


class TestCreateComprehensiveAnalysisPlotAtomic:
    """Tests for create_comprehensive_analysis_plot_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_returns_figure(
        self, mock_sns, mock_plt, sample_forecast_df, sample_combined_bootstrap_data
    ):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = [MagicMock(spec=plt.Axes), MagicMock(spec=plt.Axes)]
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        result = create_comprehensive_analysis_plot_atomic(
            sample_forecast_df, sample_combined_bootstrap_data
        )

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_two_panels_created(
        self, mock_sns, mock_plt, sample_forecast_df, sample_combined_bootstrap_data
    ):
        """Test two panels (forecast and MAPE) are created."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = [MagicMock(spec=plt.Axes), MagicMock(spec=plt.Axes)]
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        create_comprehensive_analysis_plot_atomic(
            sample_forecast_df, sample_combined_bootstrap_data
        )

        # Verify subplots called with 2 rows
        mock_plt.subplots.assert_called_once()
        call_args = mock_plt.subplots.call_args
        assert call_args[0][0] == 2  # 2 rows


class TestCreateVolatilityAnalysisPlotAtomic:
    """Tests for create_volatility_analysis_plot_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_returns_figure(self, mock_plt, sample_volatility_metrics, sample_forecast_df):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax0 = MagicMock(spec=plt.Axes)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_ax0.bar.return_value = [mock_bar, mock_bar]
        mock_ax1.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_figure, [mock_ax0, mock_ax1])

        result = create_volatility_analysis_plot_atomic(
            sample_volatility_metrics, sample_forecast_df
        )

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_two_panels_r2_and_mape(self, mock_plt, sample_volatility_metrics, sample_forecast_df):
        """Test R2 and MAPE panels are created."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax0 = MagicMock(spec=plt.Axes)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_ax0.bar.return_value = [mock_bar, mock_bar]
        mock_ax1.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_figure, [mock_ax0, mock_ax1])

        create_volatility_analysis_plot_atomic(
            sample_volatility_metrics, sample_forecast_df
        )

        # Both axes should have bar called
        mock_ax0.bar.assert_called_once()
        mock_ax1.bar.assert_called_once()


class TestCreatePerformanceSummaryPlotAtomic:
    """Tests for create_performance_summary_plot_atomic function."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_returns_figure(self, mock_plt, sample_performance_summary):
        """Test that function returns a figure."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax0 = MagicMock(spec=plt.Axes)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_ax0.bar.return_value = [mock_bar, mock_bar]
        mock_ax1.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_figure, [mock_ax0, mock_ax1])

        result = create_performance_summary_plot_atomic(sample_performance_summary)

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_r2_and_mape_comparisons(self, mock_plt, sample_performance_summary):
        """Test R2 and MAPE comparison panels."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax0 = MagicMock(spec=plt.Axes)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_ax0.bar.return_value = [mock_bar, mock_bar]
        mock_ax1.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_figure, [mock_ax0, mock_ax1])

        create_performance_summary_plot_atomic(sample_performance_summary)

        # Subplots should be 1 row, 2 columns
        mock_plt.subplots.assert_called_once()
        call_args = mock_plt.subplots.call_args
        assert call_args[0] == (1, 2)


# =============================================================================
# PANEL HELPER TESTS
# =============================================================================


class TestPlotForecastPanel:
    """Tests for _plot_forecast_panel helper."""

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_with_y_true(self, mock_sns, mock_axes, sample_forecast_df, sample_combined_bootstrap_data):
        """Test panel with y_true column."""
        _plot_forecast_panel(mock_axes, sample_forecast_df, sample_combined_bootstrap_data)

        mock_axes.scatter.assert_called_once()
        mock_sns.lineplot.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_without_y_true(self, mock_sns, mock_axes, sample_combined_bootstrap_data):
        """Test panel without y_true column."""
        df = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=5)})

        _plot_forecast_panel(mock_axes, df, sample_combined_bootstrap_data)

        # No scatter for actual values
        mock_axes.scatter.assert_not_called()
        mock_sns.lineplot.assert_called_once()


class TestPlotMapePanel:
    """Tests for _plot_mape_panel helper."""

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_with_both_columns(self, mock_sns, mock_axes, sample_forecast_df):
        """Test panel with both MAPE columns."""
        _plot_mape_panel(mock_axes, sample_forecast_df)

        # Both MAPE lines should be plotted
        assert mock_sns.lineplot.call_count == 2

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_with_only_model_column(self, mock_sns, mock_axes):
        """Test panel with only model MAPE column."""
        df = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            '13_week_MAPE_model': [5, 6, 7, 8, 9]
        })

        _plot_mape_panel(mock_axes, df)

        mock_sns.lineplot.assert_called_once()

    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_with_no_mape_columns(self, mock_sns, mock_axes):
        """Test panel with no MAPE columns."""
        df = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=5)})

        _plot_mape_panel(mock_axes, df)

        mock_sns.lineplot.assert_not_called()


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases for forecasting atomic plots."""

    @patch('src.visualization.forecasting_atomic_plots.plt')
    @patch('src.visualization.forecasting_atomic_plots.sns')
    def test_empty_bootstrap_data(self, mock_sns, mock_plt, sample_dates, sample_y_true):
        """Test with empty bootstrap data."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_axes = MagicMock(spec=plt.Axes)
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        empty_bootstrap = pd.DataFrame({'date': [], 'y_bootstrap': []})

        # Should not crash
        result = create_bootstrap_forecast_plot_atomic(
            sample_dates, sample_y_true, empty_bootstrap
        )

        assert result == mock_figure

    @patch('src.visualization.forecasting_atomic_plots.plt')
    def test_missing_performance_keys(self, mock_plt):
        """Test performance summary with missing keys."""
        mock_figure = MagicMock(spec=plt.Figure)
        mock_ax0 = MagicMock(spec=plt.Axes)
        mock_ax1 = MagicMock(spec=plt.Axes)
        mock_bar = MagicMock()
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        mock_bar.get_height.return_value = 0.5
        mock_ax0.bar.return_value = [mock_bar, mock_bar]
        mock_ax1.bar.return_value = [mock_bar, mock_bar]
        mock_plt.subplots.return_value = (mock_figure, [mock_ax0, mock_ax1])

        # Empty dict - all values should default to 0
        result = create_performance_summary_plot_atomic({})

        assert result == mock_figure

    def test_volatility_extraction_empty_dict(self):
        """Test volatility extraction with empty dict."""
        r2_data, mape_data = _extract_volatility_metrics({})

        assert r2_data['Standard Weighting'] == 0
        assert r2_data['Volatility Weighting'] == 0
        assert mape_data['Standard Weighting'] == 0
        assert mape_data['Volatility Weighting'] == 0
