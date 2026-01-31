"""
Tests for stability_visualizations module.

Target: 10% → 60%+ coverage
Tests organized by function categories:
- Win rate visualization functions
- Information ratio visualization functions
- Figure creation functions
- Main visualization function
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.features.selection.stability.stability_visualizations import (
    # Win rate plots
    _plot_win_rate_bar_chart,
    _plot_win_rate_distribution,
    # IR plots
    _plot_ir_rankings,
    _plot_risk_return_scatter,
    _plot_consistency_scatter,
    _plot_excess_aic_distributions,
    # Figure creation
    _create_win_rate_figure,
    _create_ir_figure,
    # Main function
    create_advanced_visualizations,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_win_rate_results():
    """Sample win rate analysis results."""
    return [
        {'model': 'Model_A', 'win_rate_pct': 35.5},
        {'model': 'Model_B', 'win_rate_pct': 22.0},
        {'model': 'Model_C', 'win_rate_pct': 15.3},
        {'model': 'Model_D', 'win_rate_pct': 8.2},
        {'model': 'Model_E', 'win_rate_pct': 5.0},
    ]


@pytest.fixture
def sample_ir_results():
    """Sample information ratio analysis results."""
    return [
        {
            'model_name': 'Model_A',
            'information_ratio': 0.85,
            'mean_excess': 10.5,
            'std_excess': 12.3,
            'success_rate': 75.0,
            'excess_aics': np.random.normal(10, 12, 100).tolist()
        },
        {
            'model_name': 'Model_B',
            'information_ratio': 0.45,
            'mean_excess': 5.2,
            'std_excess': 11.5,
            'success_rate': 60.0,
            'excess_aics': np.random.normal(5, 11, 100).tolist()
        },
        {
            'model_name': 'Model_C',
            'information_ratio': 0.15,
            'mean_excess': 1.5,
            'std_excess': 10.0,
            'success_rate': 52.0,
            'excess_aics': np.random.normal(1, 10, 100).tolist()
        },
        {
            'model_name': 'Model_D',
            'information_ratio': -0.10,
            'mean_excess': -1.2,
            'std_excess': 12.0,
            'success_rate': 45.0,
            'excess_aics': np.random.normal(-1, 12, 100).tolist()
        },
    ]


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes."""
    ax = MagicMock()
    ax.bar.return_value = [MagicMock()]
    ax.barh.return_value = [MagicMock()]
    ax.hist.return_value = ([MagicMock()], [MagicMock()], [MagicMock()])
    ax.scatter.return_value = MagicMock()
    ax.axvline.return_value = MagicMock()
    ax.axhline.return_value = MagicMock()
    return ax


# =============================================================================
# Win Rate Visualization Tests
# =============================================================================


class TestPlotWinRateBarChart:
    """Tests for _plot_win_rate_bar_chart."""

    def test_creates_bar_chart(self, mock_axes, sample_win_rate_results):
        """Creates bar chart with correct data."""
        _plot_win_rate_bar_chart(mock_axes, sample_win_rate_results)

        mock_axes.bar.assert_called_once()
        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_sets_correct_labels(self, mock_axes, sample_win_rate_results):
        """Sets correct axis labels."""
        _plot_win_rate_bar_chart(mock_axes, sample_win_rate_results)

        # Check xlabel call
        xlabel_call = mock_axes.set_xlabel.call_args[0][0]
        assert 'Models' in xlabel_call or 'Ranked' in xlabel_call

    def test_color_coding_by_rate(self, mock_axes, sample_win_rate_results):
        """Uses color coding based on win rate threshold."""
        _plot_win_rate_bar_chart(mock_axes, sample_win_rate_results)

        # Bar should be called with colors
        call_kwargs = mock_axes.bar.call_args
        assert 'color' in call_kwargs.kwargs or len(call_kwargs.args) >= 3


class TestPlotWinRateDistribution:
    """Tests for _plot_win_rate_distribution."""

    def test_creates_histogram(self, mock_axes, sample_win_rate_results):
        """Creates histogram with correct data."""
        _plot_win_rate_distribution(mock_axes, sample_win_rate_results)

        mock_axes.hist.assert_called_once()
        mock_axes.axvline.assert_called()  # Mean line

    def test_shows_mean_line(self, mock_axes, sample_win_rate_results):
        """Shows mean value as vertical line."""
        _plot_win_rate_distribution(mock_axes, sample_win_rate_results)

        # Should add mean line
        mock_axes.axvline.assert_called()

    def test_adds_legend(self, mock_axes, sample_win_rate_results):
        """Adds legend showing mean value."""
        _plot_win_rate_distribution(mock_axes, sample_win_rate_results)

        mock_axes.legend.assert_called()


# =============================================================================
# Information Ratio Visualization Tests
# =============================================================================


class TestPlotIrRankings:
    """Tests for _plot_ir_rankings."""

    def test_creates_horizontal_bar_chart(self, mock_axes, sample_ir_results):
        """Creates horizontal bar chart."""
        _plot_ir_rankings(mock_axes, sample_ir_results)

        mock_axes.barh.assert_called_once()

    def test_limits_to_15_models(self, mock_axes):
        """Limits display to top 15 models."""
        # Create 20 models
        large_results = [
            {'model_name': f'Model_{i}', 'information_ratio': 1.0 - i * 0.05}
            for i in range(20)
        ]

        _plot_ir_rankings(mock_axes, large_results)

        # Should only plot 15
        call_args = mock_axes.barh.call_args[0]
        assert len(call_args[0]) <= 15

    def test_adds_threshold_lines(self, mock_axes, sample_ir_results):
        """Adds IR threshold reference lines."""
        _plot_ir_rankings(mock_axes, sample_ir_results)

        # Should add multiple threshold lines
        assert mock_axes.axvline.call_count >= 3


class TestPlotRiskReturnScatter:
    """Tests for _plot_risk_return_scatter."""

    def test_creates_scatter_plot(self, mock_axes, sample_ir_results):
        """Creates scatter plot with correct data."""
        with patch('matplotlib.pyplot.colorbar') as mock_cbar:
            mock_cbar.return_value = MagicMock()
            _plot_risk_return_scatter(mock_axes, sample_ir_results)

        mock_axes.scatter.assert_called_once()

    def test_uses_ir_for_colors(self, mock_axes, sample_ir_results):
        """Uses information ratio for color coding."""
        with patch('matplotlib.pyplot.colorbar') as mock_cbar:
            mock_cbar.return_value = MagicMock()
            _plot_risk_return_scatter(mock_axes, sample_ir_results)

        call_kwargs = mock_axes.scatter.call_args.kwargs
        assert 'c' in call_kwargs  # Color values
        assert 'cmap' in call_kwargs  # Colormap

    def test_adds_reference_lines(self, mock_axes, sample_ir_results):
        """Adds benchmark and average risk reference lines."""
        with patch('matplotlib.pyplot.colorbar') as mock_cbar:
            mock_cbar.return_value = MagicMock()
            _plot_risk_return_scatter(mock_axes, sample_ir_results)

        mock_axes.axhline.assert_called()
        mock_axes.axvline.assert_called()


class TestPlotConsistencyScatter:
    """Tests for _plot_consistency_scatter."""

    def test_creates_scatter_plot(self, mock_axes, sample_ir_results):
        """Creates scatter plot."""
        _plot_consistency_scatter(mock_axes, sample_ir_results)

        mock_axes.scatter.assert_called_once()

    def test_plots_success_rate_vs_ir(self, mock_axes, sample_ir_results):
        """Plots success rate on x-axis, IR on y-axis."""
        _plot_consistency_scatter(mock_axes, sample_ir_results)

        # Check axis labels
        xlabel_call = mock_axes.set_xlabel.call_args[0][0]
        ylabel_call = mock_axes.set_ylabel.call_args[0][0]

        assert 'Success' in xlabel_call or 'Rate' in xlabel_call
        assert 'Information Ratio' in ylabel_call or 'IR' in ylabel_call

    def test_adds_50_percent_reference(self, mock_axes, sample_ir_results):
        """Adds 50% success rate reference line."""
        _plot_consistency_scatter(mock_axes, sample_ir_results)

        # Should have vertical line at 50
        axvline_calls = mock_axes.axvline.call_args_list
        assert any(call[0][0] == 50 for call in axvline_calls)


class TestPlotExcessAicDistributions:
    """Tests for _plot_excess_aic_distributions."""

    def test_plots_top_3_models(self, mock_axes, sample_ir_results):
        """Plots distributions for top 3 models only."""
        _plot_excess_aic_distributions(mock_axes, sample_ir_results)

        # Should call hist 3 times (top 3 models)
        assert mock_axes.hist.call_count == 3

    def test_adds_benchmark_line(self, mock_axes, sample_ir_results):
        """Adds benchmark reference line at zero."""
        _plot_excess_aic_distributions(mock_axes, sample_ir_results)

        axvline_calls = mock_axes.axvline.call_args_list
        assert any(call[0][0] == 0 for call in axvline_calls)

    def test_uses_density_normalization(self, mock_axes, sample_ir_results):
        """Uses density normalization for histogram."""
        _plot_excess_aic_distributions(mock_axes, sample_ir_results)

        # Check that density=True is passed
        for call in mock_axes.hist.call_args_list:
            assert call.kwargs.get('density') == True  # noqa: E712


# =============================================================================
# Figure Creation Tests
# =============================================================================


class TestCreateWinRateFigure:
    """Tests for _create_win_rate_figure."""

    def test_creates_figure_with_subplots(self, sample_win_rate_results):
        """Creates figure with two subplots."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            fig = _create_win_rate_figure(sample_win_rate_results, 16, 6)

            assert fig is mock_fig
            mock_subplots.assert_called_once_with(1, 2, figsize=(16, 6))

    def test_handles_creation_error(self, sample_win_rate_results):
        """Returns None on creation error."""
        with patch('matplotlib.pyplot.subplots', side_effect=Exception("Test error")):
            fig = _create_win_rate_figure(sample_win_rate_results, 16, 6)

            assert fig is None


class TestCreateIrFigure:
    """Tests for _create_ir_figure."""

    def test_creates_figure_with_2x2_subplots(self, sample_ir_results):
        """Creates figure with 2x2 subplot grid."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar') as mock_cbar:
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            mock_cbar.return_value = MagicMock()

            fig = _create_ir_figure(sample_ir_results, 16, 6)

            assert fig is mock_fig
            mock_subplots.assert_called_once_with(2, 2, figsize=(16, 12))

    def test_handles_creation_error(self, sample_ir_results):
        """Returns None on creation error."""
        with patch('matplotlib.pyplot.subplots', side_effect=Exception("Test error")):
            fig = _create_ir_figure(sample_ir_results, 16, 6)

            assert fig is None


# =============================================================================
# Main Visualization Function Tests
# =============================================================================


class TestCreateAdvancedVisualizations:
    """Tests for create_advanced_visualizations."""

    def test_raises_without_any_results(self):
        """Raises ValueError when no results provided."""
        with pytest.raises(ValueError, match="No analysis results available"):
            create_advanced_visualizations()

    def test_creates_win_rate_visualization(self, sample_win_rate_results):
        """Creates win rate visualization when results provided."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_win_rate_figure'
        ) as mock_create:
            mock_create.return_value = MagicMock()

            result = create_advanced_visualizations(
                win_rate_results=sample_win_rate_results
            )

            assert 'win_rate_analysis' in result
            mock_create.assert_called_once()

    def test_creates_ir_visualization(self, sample_ir_results):
        """Creates IR visualization when results provided."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_ir_figure'
        ) as mock_create:
            mock_create.return_value = MagicMock()

            result = create_advanced_visualizations(ir_results=sample_ir_results)

            assert 'information_ratio_analysis' in result
            mock_create.assert_called_once()

    def test_creates_both_visualizations(
        self, sample_win_rate_results, sample_ir_results
    ):
        """Creates both visualizations when both results provided."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_win_rate_figure'
        ) as mock_win, \
             patch(
            'src.features.selection.stability.stability_visualizations._create_ir_figure'
        ) as mock_ir:
            mock_win.return_value = MagicMock()
            mock_ir.return_value = MagicMock()

            result = create_advanced_visualizations(
                win_rate_results=sample_win_rate_results,
                ir_results=sample_ir_results
            )

            assert 'win_rate_analysis' in result
            assert 'information_ratio_analysis' in result

    def test_uses_default_config(self, sample_win_rate_results):
        """Uses default config when not provided."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_win_rate_figure'
        ) as mock_create:
            mock_create.return_value = MagicMock()

            create_advanced_visualizations(
                win_rate_results=sample_win_rate_results,
                config=None
            )

            # Should use default width=16, height=6
            call_args = mock_create.call_args[0]
            assert call_args[1] == 16  # fig_width
            assert call_args[2] == 6   # fig_height

    def test_uses_custom_config(self, sample_win_rate_results):
        """Uses custom config when provided."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_win_rate_figure'
        ) as mock_create:
            mock_create.return_value = MagicMock()

            create_advanced_visualizations(
                win_rate_results=sample_win_rate_results,
                config={'fig_width': 20, 'fig_height': 8}
            )

            call_args = mock_create.call_args[0]
            assert call_args[1] == 20  # Custom fig_width
            assert call_args[2] == 8   # Custom fig_height

    def test_skips_failed_visualization(self, sample_win_rate_results):
        """Skips visualization that fails to create."""
        with patch(
            'src.features.selection.stability.stability_visualizations._create_win_rate_figure'
        ) as mock_create:
            mock_create.return_value = None  # Simulate failure

            result = create_advanced_visualizations(
                win_rate_results=sample_win_rate_results
            )

            assert 'win_rate_analysis' not in result
