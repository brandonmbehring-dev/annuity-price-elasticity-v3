"""
Tests for bootstrap_visualization_analysis module.

Target: 12% → 60%+ coverage
Tests organized by function categories:
- Data preparation functions
- AIC distribution visualizations
- Stability comparison visualizations
- R² distribution visualizations
- Insights generation
- Main analysis function
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.visualization.bootstrap_visualization_analysis import (
    # Data preparation
    prepare_bootstrap_visualization_data,
    _extract_original_aics,
    # AIC distribution
    _setup_aic_axis,
    create_aic_distribution_visualizations,
    # Stability comparison
    _compute_model_aic_statistics,
    _create_mean_vs_variability_plot,
    _create_consistency_plot,
    create_stability_comparison_visualizations,
    # R² distribution
    _plot_r2_violin,
    _plot_r2_vs_aic,
    create_r2_distribution_visualizations,
    # Insights
    _calculate_model_statistics,
    _calculate_stability_rankings,
    _generate_insights_summary,
    _build_visualization_results,
    generate_visualization_insights,
    # Main functions
    _display_data_summary,
    _create_all_visualizations,
    _build_analysis_results,
    run_bootstrap_visualization_analysis,
    run_notebook_visualization_analysis,
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
            stability_assessment='Stable'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            model_features='feature_c + feature_d',
            bootstrap_aics=np.random.normal(105, 8, 50).tolist(),
            bootstrap_r2_values=np.random.uniform(0.4, 0.7, 50).tolist(),
            original_aic=103.2,
            original_r2=0.55,
            aic_stability_coefficient=0.08,
            r2_stability_coefficient=0.12,
            confidence_intervals={'aic': {'lower': 92, 'upper': 118}},
            successful_fits=48,
            total_attempts=50,
            stability_assessment='Moderate'
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
            stability_assessment='Unstable'
        ),
    ]


@pytest.fixture
def sample_bootstrap_df(sample_bootstrap_results):
    """Sample bootstrap DataFrame from results."""
    return prepare_bootstrap_visualization_data(sample_bootstrap_results)


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes."""
    ax = MagicMock()
    ax.bar.return_value = [MagicMock()]
    ax.scatter.return_value = MagicMock()
    ax.axhline.return_value = MagicMock()
    ax.axvline.return_value = MagicMock()
    ax.plot.return_value = [MagicMock()]
    ax.annotate.return_value = MagicMock()
    ax.text.return_value = MagicMock()
    return ax


# =============================================================================
# Data Preparation Tests
# =============================================================================


class TestPrepareBootstrapVisualizationData:
    """Tests for prepare_bootstrap_visualization_data."""

    def test_returns_dataframe(self, sample_bootstrap_results):
        """Returns a pandas DataFrame."""
        result = prepare_bootstrap_visualization_data(sample_bootstrap_results)

        assert isinstance(result, pd.DataFrame)

    def test_raises_on_empty_results(self):
        """Raises ValueError when no results provided."""
        with pytest.raises(ValueError, match="No bootstrap results provided"):
            prepare_bootstrap_visualization_data([])

    def test_creates_long_format_data(self, sample_bootstrap_results):
        """Creates long-format data with one row per bootstrap sample."""
        result = prepare_bootstrap_visualization_data(sample_bootstrap_results)

        # 3 models * 50 samples = 150 rows
        expected_rows = sum(len(r.bootstrap_aics) for r in sample_bootstrap_results)
        assert len(result) == expected_rows

    def test_includes_required_columns(self, sample_bootstrap_results):
        """Includes all required columns."""
        result = prepare_bootstrap_visualization_data(sample_bootstrap_results)

        required_cols = ['model', 'model_features', 'bootstrap_aic', 'bootstrap_r2',
                        'original_aic', 'original_r2', 'stability_assessment', 'model_index']
        for col in required_cols:
            assert col in result.columns

    def test_respects_max_models(self, sample_bootstrap_results):
        """Respects max_models parameter."""
        result = prepare_bootstrap_visualization_data(sample_bootstrap_results, max_models=2)

        assert result['model'].nunique() == 2

    def test_model_naming_convention(self, sample_bootstrap_results):
        """Uses 'Model N' naming convention."""
        result = prepare_bootstrap_visualization_data(sample_bootstrap_results)

        expected_names = ['Model 1', 'Model 2', 'Model 3']
        assert set(result['model'].unique()) == set(expected_names)


class TestExtractOriginalAics:
    """Tests for _extract_original_aics."""

    def test_returns_model_names_and_aics(self, sample_bootstrap_df):
        """Returns model names array and original AICs list."""
        names, aics = _extract_original_aics(sample_bootstrap_df)

        assert len(names) == 3
        assert len(aics) == 3

    def test_original_aics_match_input(self, sample_bootstrap_df):
        """Original AICs match input data."""
        names, aics = _extract_original_aics(sample_bootstrap_df)

        # Check that values are correct
        for name, aic in zip(names, aics):
            expected = sample_bootstrap_df[sample_bootstrap_df['model'] == name]['original_aic'].iloc[0]
            assert aic == expected


# =============================================================================
# AIC Distribution Tests
# =============================================================================


class TestSetupAicAxis:
    """Tests for _setup_aic_axis."""

    def test_violin_plot_mode(self, mock_axes, sample_bootstrap_df):
        """Creates violin plot in violin mode."""
        with patch('seaborn.violinplot') as mock_violin:
            _setup_aic_axis(mock_axes, sample_bootstrap_df, 'Test Title', 'violin')
            mock_violin.assert_called_once()

    def test_box_plot_mode(self, mock_axes, sample_bootstrap_df):
        """Creates box plot in box mode."""
        with patch('seaborn.boxplot') as mock_box:
            _setup_aic_axis(mock_axes, sample_bootstrap_df, 'Test Title', 'box')
            mock_box.assert_called_once()

    def test_sets_axis_labels(self, mock_axes, sample_bootstrap_df):
        """Sets axis labels and title."""
        with patch('seaborn.violinplot'):
            _setup_aic_axis(mock_axes, sample_bootstrap_df, 'Test Title', 'violin')

            mock_axes.set_title.assert_called()
            mock_axes.set_xlabel.assert_called()
            mock_axes.set_ylabel.assert_called()


class TestCreateAicDistributionVisualizations:
    """Tests for create_aic_distribution_visualizations."""

    def test_raises_on_empty_df(self):
        """Raises ValueError on empty DataFrame."""
        with pytest.raises(ValueError, match="Bootstrap visualization data is empty"):
            create_aic_distribution_visualizations(pd.DataFrame())

    def test_returns_figure_dict(self, sample_bootstrap_df):
        """Returns dictionary with figure."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('seaborn.violinplot'), \
             patch('seaborn.boxplot'):
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            result = create_aic_distribution_visualizations(sample_bootstrap_df)

            assert 'aic_distribution' in result
            assert result['aic_distribution'] is mock_fig

    def test_uses_config_dimensions(self, sample_bootstrap_df):
        """Uses config for figure dimensions."""
        config = {'fig_width': 24, 'fig_height': 16}

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('seaborn.violinplot'), \
             patch('seaborn.boxplot'):
            mock_subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))

            create_aic_distribution_visualizations(sample_bootstrap_df, config)

            call_kwargs = mock_subplots.call_args.kwargs
            assert call_kwargs['figsize'] == (24, 16)


# =============================================================================
# Stability Comparison Tests
# =============================================================================


class TestComputeModelAicStatistics:
    """Tests for _compute_model_aic_statistics."""

    def test_returns_dataframe(self, sample_bootstrap_df):
        """Returns a DataFrame."""
        result = _compute_model_aic_statistics(sample_bootstrap_df)

        assert isinstance(result, pd.DataFrame)

    def test_includes_statistics_columns(self, sample_bootstrap_df):
        """Includes required statistics columns."""
        result = _compute_model_aic_statistics(sample_bootstrap_df)

        expected_cols = ['mean_aic', 'std_aic', 'median_aic', 'original_aic']
        for col in expected_cols:
            assert col in result.columns

    def test_one_row_per_model(self, sample_bootstrap_df):
        """Returns one row per model."""
        result = _compute_model_aic_statistics(sample_bootstrap_df)

        assert len(result) == sample_bootstrap_df['model'].nunique()


class TestCreateMeanVsVariabilityPlot:
    """Tests for _create_mean_vs_variability_plot."""

    def test_creates_scatter_plot(self, mock_axes, sample_bootstrap_df):
        """Creates scatter plot."""
        model_stats = _compute_model_aic_statistics(sample_bootstrap_df)

        with patch('matplotlib.pyplot.colorbar'):
            _create_mean_vs_variability_plot(mock_axes, model_stats)

            mock_axes.scatter.assert_called_once()

    def test_adds_model_annotations(self, mock_axes, sample_bootstrap_df):
        """Adds annotations for each model."""
        model_stats = _compute_model_aic_statistics(sample_bootstrap_df)

        with patch('matplotlib.pyplot.colorbar'):
            _create_mean_vs_variability_plot(mock_axes, model_stats)

            assert mock_axes.annotate.call_count == len(model_stats)


class TestCreateConsistencyPlot:
    """Tests for _create_consistency_plot."""

    def test_creates_scatter_and_reference_line(self, mock_axes, sample_bootstrap_df):
        """Creates scatter plot with reference line."""
        model_stats = _compute_model_aic_statistics(sample_bootstrap_df)

        _create_consistency_plot(mock_axes, model_stats)

        mock_axes.scatter.assert_called_once()
        mock_axes.plot.assert_called_once()  # Reference line

    def test_adds_legend(self, mock_axes, sample_bootstrap_df):
        """Adds legend."""
        model_stats = _compute_model_aic_statistics(sample_bootstrap_df)

        _create_consistency_plot(mock_axes, model_stats)

        mock_axes.legend.assert_called()


class TestCreateStabilityComparisonVisualizations:
    """Tests for create_stability_comparison_visualizations."""

    def test_raises_on_empty_df(self):
        """Raises ValueError on empty DataFrame."""
        with pytest.raises(ValueError, match="Bootstrap visualization data is empty"):
            create_stability_comparison_visualizations(pd.DataFrame())

    def test_returns_figure_dict(self, sample_bootstrap_df):
        """Returns dictionary with figure."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'):
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            result = create_stability_comparison_visualizations(sample_bootstrap_df)

            assert 'stability_comparison' in result


# =============================================================================
# R² Distribution Tests
# =============================================================================


class TestPlotR2Violin:
    """Tests for _plot_r2_violin."""

    def test_creates_violin_plot(self, mock_axes, sample_bootstrap_df):
        """Creates violin plot for R² values."""
        with patch('seaborn.violinplot'):
            _plot_r2_violin(mock_axes, sample_bootstrap_df)

            mock_axes.scatter.assert_called()  # Original R² markers


class TestPlotR2VsAic:
    """Tests for _plot_r2_vs_aic."""

    def test_creates_scatter_plot(self, mock_axes, sample_bootstrap_df):
        """Creates scatter plot."""
        _plot_r2_vs_aic(mock_axes, sample_bootstrap_df)

        mock_axes.scatter.assert_called_once()

    def test_adds_correlation_annotation(self, mock_axes, sample_bootstrap_df):
        """Adds correlation annotation."""
        _plot_r2_vs_aic(mock_axes, sample_bootstrap_df)

        mock_axes.text.assert_called_once()


class TestCreateR2DistributionVisualizations:
    """Tests for create_r2_distribution_visualizations."""

    def test_returns_empty_on_empty_df(self):
        """Returns empty dict on empty DataFrame."""
        result = create_r2_distribution_visualizations(pd.DataFrame())

        assert result == {}

    def test_returns_empty_if_no_r2_column(self):
        """Returns empty dict if no bootstrap_r2 column."""
        df = pd.DataFrame({'model': ['A'], 'bootstrap_aic': [100]})
        result = create_r2_distribution_visualizations(df)

        assert result == {}

    def test_returns_figure_dict(self, sample_bootstrap_df):
        """Returns dictionary with figure."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('seaborn.violinplot'):
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            result = create_r2_distribution_visualizations(sample_bootstrap_df)

            assert 'r2_distribution' in result


# =============================================================================
# Insights Generation Tests
# =============================================================================


class TestCalculateModelStatistics:
    """Tests for _calculate_model_statistics."""

    def test_returns_dataframe(self, sample_bootstrap_df):
        """Returns a DataFrame with aggregated statistics."""
        result = _calculate_model_statistics(sample_bootstrap_df)

        assert isinstance(result, pd.DataFrame)

    def test_includes_aic_statistics(self, sample_bootstrap_df):
        """Includes AIC statistics columns."""
        result = _calculate_model_statistics(sample_bootstrap_df)

        expected_cols = ['mean_bootstrap_aic', 'std_bootstrap_aic', 'median_bootstrap_aic']
        for col in expected_cols:
            assert col in result.columns


class TestCalculateStabilityRankings:
    """Tests for _calculate_stability_rankings."""

    def test_returns_list_of_dicts(self, sample_bootstrap_df):
        """Returns list of model dictionaries."""
        model_stats = _calculate_model_statistics(sample_bootstrap_df)
        result = _calculate_stability_rankings(model_stats)

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_sorted_by_combined_score(self, sample_bootstrap_df):
        """Results are sorted by combined score."""
        model_stats = _calculate_model_statistics(sample_bootstrap_df)
        result = _calculate_stability_rankings(model_stats)

        scores = [r['combined_score'] for r in result]
        assert scores == sorted(scores)


class TestGenerateInsightsSummary:
    """Tests for _generate_insights_summary."""

    def test_returns_string(self, sample_bootstrap_df):
        """Returns a string summary."""
        model_stats = _calculate_model_statistics(sample_bootstrap_df)
        model_stats['aic_stability_coefficient'] = 1 / (
            model_stats['std_bootstrap_aic'] / model_stats['mean_bootstrap_aic']
        )

        result = _generate_insights_summary(model_stats)

        assert isinstance(result, str)

    def test_includes_key_information(self, sample_bootstrap_df):
        """Includes key analysis information."""
        model_stats = _calculate_model_statistics(sample_bootstrap_df)
        model_stats['aic_stability_coefficient'] = 1 / (
            model_stats['std_bootstrap_aic'] / model_stats['mean_bootstrap_aic']
        )

        result = _generate_insights_summary(model_stats)

        assert 'Models Analyzed' in result
        assert 'Most Stable Model' in result
        assert 'Best Average Performance' in result


class TestBuildVisualizationResults:
    """Tests for _build_visualization_results."""

    def test_returns_complete_results_dict(self, sample_bootstrap_df):
        """Returns complete results dictionary."""
        model_stats = _calculate_model_statistics(sample_bootstrap_df)
        model_stats['aic_stability_coefficient'] = 1.0

        result = _build_visualization_results(
            sample_bootstrap_df,
            model_stats,
            [{'model': 'Model 1', 'score': 1.0}],
            'Test summary'
        )

        assert 'visualization_insights' in result
        assert 'model_stability_ranking' in result
        assert 'statistical_summary' in result
        assert 'analysis_metadata' in result


class TestGenerateVisualizationInsights:
    """Tests for generate_visualization_insights."""

    def test_returns_insights_for_valid_data(self, sample_bootstrap_df):
        """Returns insights for valid data."""
        result = generate_visualization_insights(sample_bootstrap_df)

        assert 'visualization_insights' in result
        assert 'model_stability_ranking' in result
        assert 'statistical_summary' in result

    def test_handles_empty_dataframe(self):
        """Handles empty DataFrame gracefully."""
        result = generate_visualization_insights(pd.DataFrame())

        assert 'visualization_insights' in result
        assert 'No data available' in result['visualization_insights']
        assert result['model_stability_ranking'] == []


# =============================================================================
# Main Analysis Function Tests
# =============================================================================


class TestDisplayDataSummary:
    """Tests for _display_data_summary."""

    def test_prints_summary(self, sample_bootstrap_df, capsys):
        """Prints data summary."""
        _display_data_summary(sample_bootstrap_df)

        captured = capsys.readouterr()
        assert 'Bootstrap Visualization Data Summary' in captured.out
        assert 'Models' in captured.out


class TestCreateAllVisualizations:
    """Tests for _create_all_visualizations."""

    def test_returns_visualization_dict(self, sample_bootstrap_df):
        """Returns dictionary of visualizations."""
        with patch('src.features.selection.visualization.bootstrap_visualization_analysis.create_aic_distribution_visualizations') as mock_aic, \
             patch('src.features.selection.visualization.bootstrap_visualization_analysis.create_stability_comparison_visualizations') as mock_stab, \
             patch('src.features.selection.visualization.bootstrap_visualization_analysis.create_r2_distribution_visualizations') as mock_r2, \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):
            mock_aic.return_value = {'aic': MagicMock()}
            mock_stab.return_value = {'stability': MagicMock()}
            mock_r2.return_value = {'r2': MagicMock()}

            result = _create_all_visualizations(sample_bootstrap_df, {})

            assert 'aic' in result
            assert 'stability' in result
            assert 'r2' in result

    def test_handles_visualization_error(self, sample_bootstrap_df, capsys):
        """Handles visualization creation errors gracefully."""
        with patch('src.features.selection.visualization.bootstrap_visualization_analysis.create_aic_distribution_visualizations',
                   side_effect=Exception("Test error")):
            result = _create_all_visualizations(sample_bootstrap_df, {})

            assert result == {}
            captured = capsys.readouterr()
            assert 'WARNING' in captured.out


class TestBuildAnalysisResults:
    """Tests for _build_analysis_results."""

    def test_returns_complete_results(self, sample_bootstrap_df):
        """Returns complete analysis results."""
        insights = {'test': 'insights'}
        visualizations = {'fig': MagicMock()}

        result = _build_analysis_results(sample_bootstrap_df, insights, visualizations, 15)

        assert 'visualization_data' in result
        assert 'insights' in result
        assert 'visualizations' in result
        assert 'analysis_metadata' in result


class TestRunBootstrapVisualizationAnalysis:
    """Tests for run_bootstrap_visualization_analysis."""

    def test_raises_on_empty_results(self):
        """Raises ValueError on empty results."""
        with pytest.raises(ValueError, match="No bootstrap results provided"):
            run_bootstrap_visualization_analysis([])

    def test_returns_analysis_results(self, sample_bootstrap_results, capsys):
        """Returns analysis results."""
        with patch('src.features.selection.visualization.bootstrap_visualization_analysis._create_all_visualizations') as mock_viz:
            mock_viz.return_value = {}

            result = run_bootstrap_visualization_analysis(
                sample_bootstrap_results,
                display_results=True,
                create_visualizations=False
            )

            assert 'visualization_data' in result
            assert 'insights' in result

    def test_respects_return_detailed_flag(self, sample_bootstrap_results):
        """Respects return_detailed flag."""
        with patch('src.features.selection.visualization.bootstrap_visualization_analysis._create_all_visualizations') as mock_viz:
            mock_viz.return_value = {}

            result = run_bootstrap_visualization_analysis(
                sample_bootstrap_results,
                display_results=False,
                create_visualizations=False,
                return_detailed=False
            )

            assert 'visualization_data' in result
            assert 'insights' not in result  # Not detailed

    def test_handles_analysis_error(self, capsys):
        """Handles analysis errors."""
        # Create invalid data that will cause error
        bad_result = MagicMock()
        bad_result.bootstrap_aics = None  # Will cause error

        with pytest.raises(RuntimeError, match="Visualization analysis failed"):
            run_bootstrap_visualization_analysis([bad_result])


class TestRunNotebookVisualizationAnalysis:
    """Tests for run_notebook_visualization_analysis."""

    def test_returns_dataframe_for_valid_data(self, sample_bootstrap_results):
        """Returns DataFrame for valid data."""
        with patch('src.features.selection.visualization.bootstrap_visualization_analysis._create_all_visualizations') as mock_viz:
            mock_viz.return_value = {}

            result = run_notebook_visualization_analysis(sample_bootstrap_results)

            assert isinstance(result, pd.DataFrame)

    def test_returns_empty_df_for_empty_results(self, capsys):
        """Returns empty DataFrame for empty results."""
        result = run_notebook_visualization_analysis([])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
