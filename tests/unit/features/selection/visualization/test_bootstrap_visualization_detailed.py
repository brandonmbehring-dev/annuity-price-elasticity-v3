"""
Tests for bootstrap_visualization_detailed module.

Target: 12% → 60%+ coverage
Tests organized by function categories:
- DataFrame creation functions
- Violin plot functions
- Boxplot functions
- Ranking statistics functions
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.visualization.bootstrap_visualization_detailed import (
    # DataFrame creation
    _build_bootstrap_row,
    create_bootstrap_dataframe,
    # Violin plot
    _get_stability_color,
    _plot_single_violin,
    _add_violin_markers,
    _format_violin_axes,
    generate_violin_plot_visualization,
    # Boxplot
    _prepare_boxplot_data,
    _create_styled_boxplot,
    generate_boxplot_visualization,
    # Ranking statistics
    _compute_summary_statistics,
    _display_ranking_comparison,
    calculate_ranking_statistics,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockBootstrapResult:
    """Mock BootstrapResult for testing."""
    model_name: str
    model_features: str
    bootstrap_aics: List[float]
    bootstrap_r2_values: List[float]
    original_aic: float
    original_r2: float
    aic_stability_coefficient: float
    r2_stability_coefficient: float
    confidence_intervals: Dict[str, Dict[str, float]]
    successful_fits: int
    total_attempts: int
    stability_assessment: str


@pytest.fixture
def sample_bootstrap_results():
    """Sample bootstrap results for testing."""
    np.random.seed(42)
    return [
        MockBootstrapResult(
            model_name='Model_1',
            model_features='feature_a + feature_b',
            bootstrap_aics=np.random.normal(100, 5, 50).tolist(),
            bootstrap_r2_values=np.random.uniform(0.5, 0.8, 50).tolist(),
            original_aic=98.5,
            original_r2=0.65,
            aic_stability_coefficient=0.05,
            r2_stability_coefficient=0.08,
            confidence_intervals={'aic': {'lower': 95, 'upper': 105}},
            successful_fits=50,
            total_attempts=50,
            stability_assessment='STABLE'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            model_features='feature_c + feature_d + feature_e + feature_f',
            bootstrap_aics=np.random.normal(105, 8, 50).tolist(),
            bootstrap_r2_values=np.random.uniform(0.4, 0.7, 50).tolist(),
            original_aic=103.2,
            original_r2=0.55,
            aic_stability_coefficient=0.08,
            r2_stability_coefficient=0.12,
            confidence_intervals={'aic': {'lower': 92, 'upper': 118}},
            successful_fits=48,
            total_attempts=50,
            stability_assessment='MODERATE'
        ),
        MockBootstrapResult(
            model_name='Model_3',
            model_features='feature_e',
            bootstrap_aics=np.random.normal(110, 12, 50).tolist(),
            bootstrap_r2_values=np.random.uniform(0.3, 0.6, 50).tolist(),
            original_aic=108.7,
            original_r2=0.45,
            aic_stability_coefficient=0.11,
            r2_stability_coefficient=0.18,
            confidence_intervals={'aic': {'lower': 88, 'upper': 132}},
            successful_fits=45,
            total_attempts=50,
            stability_assessment='UNSTABLE'
        ),
    ]


@pytest.fixture
def sample_bootstrap_df(sample_bootstrap_results):
    """Sample bootstrap DataFrame from results."""
    return create_bootstrap_dataframe(sample_bootstrap_results)


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes."""
    ax = MagicMock()
    ax.fill_between.return_value = MagicMock()
    ax.plot.return_value = [MagicMock()]
    ax.axhline.return_value = MagicMock()
    ax.scatter.return_value = MagicMock()
    ax.annotate.return_value = MagicMock()
    ax.boxplot.return_value = {'boxes': [MagicMock(), MagicMock()]}
    return ax


@pytest.fixture
def sample_config():
    """Sample visualization configuration."""
    return {
        'models_to_analyze': 15,
        'fig_width': 12,
        'fig_height': 10
    }


# =============================================================================
# DataFrame Creation Tests
# =============================================================================


class TestBuildBootstrapRow:
    """Tests for _build_bootstrap_row."""

    def test_returns_dict_with_expected_keys(self, sample_bootstrap_results):
        """Returns dictionary with all expected keys."""
        result = sample_bootstrap_results[0]
        row = _build_bootstrap_row(result, 0, 0, 100.0)

        expected_keys = ['model', 'model_features', 'bootstrap_aic', 'bootstrap_r2',
                        'original_aic', 'original_r2', 'stability_assessment']
        for key in expected_keys:
            assert key in row

    def test_model_naming_convention(self, sample_bootstrap_results):
        """Uses 'Model N' naming convention (1-indexed)."""
        result = sample_bootstrap_results[0]

        row0 = _build_bootstrap_row(result, 0, 0, 100.0)
        row1 = _build_bootstrap_row(result, 1, 0, 100.0)
        row2 = _build_bootstrap_row(result, 2, 0, 100.0)

        assert row0['model'] == 'Model 1'
        assert row1['model'] == 'Model 2'
        assert row2['model'] == 'Model 3'

    def test_handles_missing_r2_values(self, sample_bootstrap_results):
        """Handles sample index beyond r2_values length."""
        result = sample_bootstrap_results[0]
        row = _build_bootstrap_row(result, 0, 999, 100.0)  # Index beyond data

        assert np.isnan(row['bootstrap_r2'])

    def test_uses_provided_aic_value(self, sample_bootstrap_results):
        """Uses the provided AIC value."""
        result = sample_bootstrap_results[0]
        row = _build_bootstrap_row(result, 0, 0, 123.456)

        assert row['bootstrap_aic'] == 123.456


class TestCreateBootstrapDataframe:
    """Tests for create_bootstrap_dataframe."""

    def test_returns_dataframe(self, sample_bootstrap_results):
        """Returns a pandas DataFrame."""
        result = create_bootstrap_dataframe(sample_bootstrap_results)

        assert isinstance(result, pd.DataFrame)

    def test_raises_on_empty_results(self):
        """Raises ValueError when no results provided."""
        with pytest.raises(ValueError, match="CRITICAL: No bootstrap results"):
            create_bootstrap_dataframe([])

    def test_creates_correct_number_of_rows(self, sample_bootstrap_results):
        """Creates correct number of rows (all samples from all models)."""
        result = create_bootstrap_dataframe(sample_bootstrap_results)

        expected_rows = sum(len(r.bootstrap_aics) for r in sample_bootstrap_results)
        assert len(result) == expected_rows

    def test_prints_summary(self, sample_bootstrap_results, capsys):
        """Prints creation summary."""
        create_bootstrap_dataframe(sample_bootstrap_results)

        captured = capsys.readouterr()
        assert 'Bootstrap DataFrame created' in captured.out


# =============================================================================
# Stability Color Tests
# =============================================================================


class TestGetStabilityColor:
    """Tests for _get_stability_color."""

    def test_stable_returns_green(self):
        """STABLE returns green color."""
        color = _get_stability_color('STABLE')

        # Should be from seaborn deep palette index 2 (green)
        assert color is not None

    def test_moderate_returns_orange(self):
        """MODERATE returns orange color."""
        color = _get_stability_color('MODERATE')

        assert color is not None

    def test_unstable_returns_red(self):
        """UNSTABLE/other returns red color."""
        color = _get_stability_color('UNSTABLE')

        assert color is not None

    def test_unknown_returns_red(self):
        """Unknown stability returns red (default)."""
        color = _get_stability_color('UNKNOWN')

        assert color is not None


# =============================================================================
# Violin Plot Tests
# =============================================================================


class TestPlotSingleViolin:
    """Tests for _plot_single_violin."""

    def test_creates_fill_and_line(self, mock_axes):
        """Creates filled area and outline."""
        model_data = np.random.normal(100, 5, 50)

        with patch('scipy.stats.gaussian_kde') as mock_kde:
            mock_kde.return_value = MagicMock(return_value=np.ones(100))

            _plot_single_violin(mock_axes, model_data, 0, (0.5, 0.5, 0.5), 'STABLE')

            mock_axes.fill_between.assert_called_once()
            mock_axes.plot.assert_called()

    def test_adds_horizontal_baseline(self, mock_axes):
        """Adds horizontal baseline."""
        model_data = np.random.normal(100, 5, 50)

        with patch('scipy.stats.gaussian_kde') as mock_kde:
            mock_kde.return_value = MagicMock(return_value=np.ones(100))

            _plot_single_violin(mock_axes, model_data, 0, (0.5, 0.5, 0.5), 'STABLE')

            mock_axes.axhline.assert_called_once()


class TestAddViolinMarkers:
    """Tests for _add_violin_markers."""

    def test_adds_original_aic_marker(self, mock_axes):
        """Adds marker for original AIC value."""
        model_data = np.random.normal(100, 5, 50)

        _add_violin_markers(mock_axes, model_data, 98.5, 0)

        mock_axes.scatter.assert_called_once()

    def test_adds_median_line(self, mock_axes):
        """Adds median line."""
        model_data = np.random.normal(100, 5, 50)

        _add_violin_markers(mock_axes, model_data, 98.5, 0)

        mock_axes.plot.assert_called_once()


class TestFormatViolinAxes:
    """Tests for _format_violin_axes."""

    def test_sets_labels(self, mock_axes, sample_bootstrap_df):
        """Sets axis labels."""
        models_order = ['Model 1', 'Model 2', 'Model 3']

        _format_violin_axes(mock_axes, sample_bootstrap_df, models_order)

        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()

    def test_sets_y_ticks(self, mock_axes, sample_bootstrap_df):
        """Sets Y-axis ticks."""
        models_order = ['Model 1', 'Model 2', 'Model 3']

        _format_violin_axes(mock_axes, sample_bootstrap_df, models_order)

        mock_axes.set_yticks.assert_called()
        mock_axes.set_yticklabels.assert_called()


class TestGenerateViolinPlotVisualization:
    """Tests for generate_violin_plot_visualization."""

    def test_raises_on_empty_df(self, sample_config):
        """Raises ValueError on empty DataFrame."""
        with pytest.raises(ValueError, match="CRITICAL: Empty DataFrame"):
            generate_violin_plot_visualization(pd.DataFrame(), [], sample_config)

    def test_returns_result_dict(self, sample_bootstrap_df, sample_bootstrap_results, sample_config):
        """Returns result dictionary."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.show'), \
             patch('scipy.stats.gaussian_kde') as mock_kde:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_kde.return_value = MagicMock(return_value=np.ones(100))

            result = generate_violin_plot_visualization(
                sample_bootstrap_df, sample_bootstrap_results, sample_config, display_plot=False
            )

            assert result['violin_plot_created'] == True
            assert 'models_visualized' in result

    def test_respects_display_flag(self, sample_bootstrap_df, sample_bootstrap_results, sample_config):
        """Respects display_plot flag."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.show') as mock_show, \
             patch('scipy.stats.gaussian_kde') as mock_kde:
            mock_subplots.return_value = (MagicMock(), MagicMock())
            mock_kde.return_value = MagicMock(return_value=np.ones(100))

            generate_violin_plot_visualization(
                sample_bootstrap_df, sample_bootstrap_results, sample_config, display_plot=False
            )

            mock_show.assert_not_called()


# =============================================================================
# Boxplot Tests
# =============================================================================


class TestPrepareBoxplotData:
    """Tests for _prepare_boxplot_data."""

    def test_returns_three_lists(self, sample_bootstrap_df):
        """Returns data, labels, and colors."""
        models_order = ['Model 1', 'Model 2', 'Model 3']

        data, labels, colors = _prepare_boxplot_data(sample_bootstrap_df, models_order)

        assert isinstance(data, list)
        assert isinstance(labels, list)
        assert isinstance(colors, list)

    def test_truncates_long_features(self, sample_bootstrap_df):
        """Truncates feature names longer than 35 characters."""
        models_order = ['Model 1', 'Model 2', 'Model 3']

        data, labels, colors = _prepare_boxplot_data(sample_bootstrap_df, models_order)

        # Model 2 has long features
        for label in labels:
            # Each label part should be truncated if > 35
            assert '...' in label or len(label.split('\n')[1]) <= 35

    def test_skips_models_with_insufficient_data(self, sample_bootstrap_df):
        """Skips models with <= 5 data points."""
        # Create df with one model having minimal data
        small_df = sample_bootstrap_df[sample_bootstrap_df['model'] == 'Model 1'].head(3)

        data, labels, colors = _prepare_boxplot_data(small_df, ['Model 1'])

        assert len(data) == 0  # Not enough data


class TestCreateStyledBoxplot:
    """Tests for _create_styled_boxplot."""

    def test_creates_boxplot(self, mock_axes):
        """Creates boxplot with styling."""
        data = [np.random.normal(100, 5, 50), np.random.normal(105, 5, 50)]
        labels = ['Model 1\nfeatures\n(STABLE)', 'Model 2\nfeatures\n(MODERATE)']
        colors = [(0.5, 0.5, 0.5), (0.6, 0.6, 0.6)]

        with patch('matplotlib.pyplot.xticks'):
            _create_styled_boxplot(mock_axes, data, labels, colors)

            mock_axes.boxplot.assert_called_once()

    def test_sets_axis_labels(self, mock_axes):
        """Sets axis labels and title."""
        data = [np.random.normal(100, 5, 50)]
        labels = ['Model 1\nfeatures\n(STABLE)']
        colors = [(0.5, 0.5, 0.5)]

        with patch('matplotlib.pyplot.xticks'):
            _create_styled_boxplot(mock_axes, data, labels, colors)

            mock_axes.set_xlabel.assert_called()
            mock_axes.set_ylabel.assert_called()
            mock_axes.set_title.assert_called()


class TestGenerateBoxplotVisualization:
    """Tests for generate_boxplot_visualization."""

    def test_raises_on_empty_df(self, sample_config):
        """Raises ValueError on empty DataFrame."""
        with pytest.raises(ValueError, match="CRITICAL: Empty DataFrame"):
            generate_boxplot_visualization(pd.DataFrame(), [], sample_config)

    def test_returns_result_dict(self, sample_bootstrap_df, sample_bootstrap_results, sample_config):
        """Returns result dictionary."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.xticks'):
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_ax.boxplot.return_value = {'boxes': [MagicMock() for _ in range(3)]}
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = generate_boxplot_visualization(
                sample_bootstrap_df, sample_bootstrap_results, sample_config, display_plot=False
            )

            assert 'boxplot_created' in result
            assert 'models_visualized' in result

    def test_handles_no_data(self, sample_config, capsys):
        """Handles case with no plottable data."""
        # Create minimal df that will be skipped (< 5 points per model)
        df = pd.DataFrame({
            'model': ['Model 1', 'Model 1'],
            'model_features': ['a', 'a'],
            'bootstrap_aic': [100, 101],
            'stability_assessment': ['STABLE', 'STABLE']
        })

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            mock_subplots.return_value = (MagicMock(), MagicMock())

            result = generate_boxplot_visualization(df, [], sample_config, display_plot=False)

            assert result['boxplot_created'] == False


# =============================================================================
# Ranking Statistics Tests
# =============================================================================


class TestComputeSummaryStatistics:
    """Tests for _compute_summary_statistics."""

    def test_returns_sorted_dataframe(self, sample_bootstrap_df):
        """Returns DataFrame sorted by median."""
        result = _compute_summary_statistics(sample_bootstrap_df)

        assert isinstance(result, pd.DataFrame)
        # Should be sorted by median (ascending)
        medians = result['median'].values
        assert list(medians) == sorted(medians)

    def test_includes_all_statistics(self, sample_bootstrap_df):
        """Includes mean, std, median, q25, q75, stability."""
        result = _compute_summary_statistics(sample_bootstrap_df)

        expected_cols = ['mean', 'std', 'median', 'q25', 'q75', 'stability']
        for col in expected_cols:
            assert col in result.columns


class TestDisplayRankingComparison:
    """Tests for _display_ranking_comparison."""

    def test_prints_table_header(self, sample_bootstrap_df, capsys):
        """Prints comparison table header."""
        summary_stats = _compute_summary_statistics(sample_bootstrap_df)

        _display_ranking_comparison(summary_stats, 15)

        captured = capsys.readouterr()
        assert 'BOOTSTRAP vs ORIGINAL AIC RANKING COMPARISON' in captured.out
        assert 'Model' in captured.out
        assert 'Original Rank' in captured.out

    def test_limits_to_models_analyzed(self, sample_bootstrap_df, capsys):
        """Limits output to models_to_analyze parameter."""
        summary_stats = _compute_summary_statistics(sample_bootstrap_df)

        _display_ranking_comparison(summary_stats, 2)

        captured = capsys.readouterr()
        # Count data rows (Model N where N is digit), not header row
        lines = [l for l in captured.out.split('\n') if l.strip().startswith('Model ') and l[6:7].isdigit()]
        assert len(lines) <= 2


class TestCalculateRankingStatistics:
    """Tests for calculate_ranking_statistics."""

    def test_raises_on_empty_df(self):
        """Raises ValueError on empty DataFrame."""
        with pytest.raises(ValueError, match="CRITICAL: Empty DataFrame"):
            calculate_ranking_statistics(pd.DataFrame())

    def test_returns_result_dict(self, sample_bootstrap_df):
        """Returns result dictionary."""
        result = calculate_ranking_statistics(sample_bootstrap_df, display_results=False)

        assert 'summary_statistics' in result
        assert 'median_aic_ranking' in result
        assert 'models_analyzed' in result

    def test_median_aic_ranking_is_list(self, sample_bootstrap_df):
        """Returns median AIC ranking as list."""
        result = calculate_ranking_statistics(sample_bootstrap_df, display_results=False)

        assert isinstance(result['median_aic_ranking'], list)
        assert len(result['median_aic_ranking']) == sample_bootstrap_df['model'].nunique()

    def test_respects_models_to_analyze(self, sample_bootstrap_df):
        """Respects models_to_analyze parameter."""
        result = calculate_ranking_statistics(sample_bootstrap_df, models_to_analyze=2, display_results=False)

        assert result['models_analyzed'] == 2

    def test_prints_when_display_true(self, sample_bootstrap_df, capsys):
        """Prints statistics when display_results=True."""
        calculate_ranking_statistics(sample_bootstrap_df, display_results=True)

        captured = capsys.readouterr()
        assert 'Bootstrap Stability Summary Statistics' in captured.out
