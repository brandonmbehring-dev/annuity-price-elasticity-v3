"""
Tests for src.visualization.model_comparison module.

Target: 23% → 80%+ coverage
Tests organized by class/function:
- ModelComparisonPlots class
- create_model_comparison_report function
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass
from typing import List, Dict, Any

from src.visualization.model_comparison import (
    ModelComparisonPlots,
    create_model_comparison_report,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_aic_results():
    """Sample AIC results DataFrame."""
    return pd.DataFrame({
        'model_name': [f'Model_{i}' for i in range(15)],
        'model_features': ['A + B', 'A + C', 'B + C', 'A + B + C', 'A',
                          'B', 'C', 'A + B + D', 'A + C + D', 'B + C + D',
                          'A + D', 'B + D', 'C + D', 'D', 'A + B + C + D'],
        'aic': np.linspace(100, 150, 15),
        'r_squared': np.linspace(0.85, 0.70, 15),
        'n_features': [2, 2, 2, 3, 1, 1, 1, 3, 3, 3, 2, 2, 2, 1, 4],
        'converged': [True] * 15,
    })


@pytest.fixture
def sample_information_criteria():
    """Sample information criteria results."""
    return [
        {'model_name': 'Model_0', 'aic': 100, 'bic': 105, 'hqic': 102},
        {'model_name': 'Model_1', 'aic': 105, 'bic': 110, 'hqic': 107},
        {'model_name': 'Model_2', 'aic': 110, 'bic': 115, 'hqic': 112},
    ]


@pytest.fixture
def sample_bootstrap_results():
    """Sample bootstrap results."""
    return [
        {
            'model_name': 'Model_0',
            'model_features': 'A + B',
            'bootstrap_aics': list(np.random.normal(100, 2, 100)),
            'cv': 0.02,
            'stability_assessment': 'STABLE',
        },
        {
            'model_name': 'Model_1',
            'model_features': 'A + C',
            'bootstrap_aics': list(np.random.normal(105, 3, 100)),
            'cv': 0.03,
            'stability_assessment': 'STABLE',
        },
        {
            'model_name': 'Model_2',
            'model_features': 'B + C',
            'bootstrap_aics': list(np.random.normal(110, 5, 100)),
            'cv': 0.05,
            'stability_assessment': 'MODERATE',
        },
    ]


@pytest.fixture
def sample_coefficient_stability():
    """Sample coefficient stability analysis."""
    return {
        'feature_a': {
            'mean': 0.15,
            'std': 0.02,
            'sign_consistency': 0.95,
            'ci_lower': 0.11,
            'ci_upper': 0.19,
        },
        'feature_b': {
            'mean': -0.08,
            'std': 0.01,
            'sign_consistency': 0.98,
            'ci_lower': -0.10,
            'ci_upper': -0.06,
        },
    }


@pytest.fixture
def mock_figure():
    """Mock matplotlib figure."""
    fig = MagicMock()
    fig.savefig = MagicMock()
    return fig


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes."""
    ax = MagicMock()
    ax.scatter = MagicMock()
    ax.bar = MagicMock()
    ax.hist = MagicMock()
    ax.text = MagicMock()
    ax.set_xlabel = MagicMock()
    ax.set_ylabel = MagicMock()
    ax.set_title = MagicMock()
    ax.axis = MagicMock()
    return ax


# =============================================================================
# ModelComparisonPlots Class Tests
# =============================================================================


class TestModelComparisonPlotsInit:
    """Tests for ModelComparisonPlots.__init__."""

    def test_default_style(self):
        """Uses default seaborn-v0_8 style."""
        plotter = ModelComparisonPlots()

        assert plotter.style == 'seaborn-v0_8'

    def test_custom_style(self):
        """Accepts custom style."""
        plotter = ModelComparisonPlots(style='ggplot')

        assert plotter.style == 'ggplot'

    def test_colors_initialized(self):
        """Colors dictionary is initialized."""
        plotter = ModelComparisonPlots()

        assert 'primary' in plotter.colors
        assert 'secondary' in plotter.colors
        assert 'success' in plotter.colors
        assert plotter.colors['primary'] == '#1f77b4'

    def test_plot_config_initialized(self):
        """Plot configuration is initialized."""
        plotter = ModelComparisonPlots()

        assert 'figure_size_large' in plotter.plot_config
        assert 'figure_size_standard' in plotter.plot_config
        assert plotter.plot_config['figure_size_large'] == (16, 12)


class TestCreateMultiCriteriaComparisonMatrix:
    """Tests for create_multi_criteria_comparison_matrix."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.gridspec.GridSpec')
    def test_returns_figure(self, mock_gs, mock_fig_func, sample_aic_results,
                            sample_information_criteria, sample_bootstrap_results):
        """Returns matplotlib figure."""
        mock_fig = MagicMock()
        mock_fig.add_subplot = MagicMock(return_value=MagicMock())
        mock_fig_func.return_value = mock_fig
        mock_gs.return_value = MagicMock()

        plotter = ModelComparisonPlots()

        with patch.object(plotter, '_populate_comparison_matrix_subplots'):
            result = plotter.create_multi_criteria_comparison_matrix(
                sample_aic_results, sample_information_criteria, sample_bootstrap_results
            )

        assert result is not None

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.gridspec.GridSpec')
    def test_saves_to_path(self, mock_gs, mock_fig_func, sample_aic_results,
                           sample_information_criteria, sample_bootstrap_results, tmp_path):
        """Saves figure to provided path."""
        mock_fig = MagicMock()
        mock_fig.add_subplot = MagicMock(return_value=MagicMock())
        mock_fig.savefig = MagicMock()
        mock_fig_func.return_value = mock_fig
        mock_gs.return_value = MagicMock()

        plotter = ModelComparisonPlots()
        save_path = tmp_path / "matrix.png"

        with patch.object(plotter, '_populate_comparison_matrix_subplots'):
            plotter.create_multi_criteria_comparison_matrix(
                sample_aic_results, sample_information_criteria, sample_bootstrap_results,
                save_path=save_path
            )

        mock_fig.savefig.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.gridspec.GridSpec')
    def test_uses_top_15_models(self, mock_gs, mock_fig_func, sample_aic_results,
                                 sample_information_criteria, sample_bootstrap_results):
        """Uses top 15 models from aic_results."""
        mock_fig = MagicMock()
        mock_fig.add_subplot = MagicMock(return_value=MagicMock())
        mock_fig_func.return_value = mock_fig
        mock_gs.return_value = MagicMock()

        plotter = ModelComparisonPlots()

        with patch.object(plotter, '_populate_comparison_matrix_subplots') as mock_populate:
            plotter.create_multi_criteria_comparison_matrix(
                sample_aic_results, sample_information_criteria, sample_bootstrap_results
            )

            # Check that top_models was passed (first positional after fig, gs)
            call_args = mock_populate.call_args[0]
            top_models = call_args[2]  # Third positional arg
            assert len(top_models) == 15


class TestCreateBootstrapDistributionComparison:
    """Tests for create_bootstrap_distribution_comparison."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_returns_figure(self, mock_tight, mock_subplots, sample_bootstrap_results):
        """Returns matplotlib figure."""
        mock_fig = MagicMock()
        # Create proper array of MagicMocks that flatten() works on
        mock_ax_list = [MagicMock() for _ in range(6)]
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = mock_ax_list
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()

        with patch('src.visualization.model_comparison.sort_bootstrap_by_stability',
                   return_value=[(0.02, sample_bootstrap_results[0])]), \
             patch('src.visualization.model_comparison.plot_dict_format_bootstrap'):
            result = plotter.create_bootstrap_distribution_comparison(sample_bootstrap_results)

        assert result is not None

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_handles_empty_results(self, mock_tight, mock_subplots):
        """Handles empty bootstrap results gracefully."""
        mock_fig = MagicMock()
        mock_ax_list = [MagicMock() for _ in range(6)]
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = mock_ax_list
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()
        result = plotter.create_bootstrap_distribution_comparison([])

        assert result is not None
        # Should show "No Bootstrap Data Available" text on each ax
        for ax in mock_ax_list:
            ax.text.assert_called()

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_saves_to_path(self, mock_tight, mock_subplots, sample_bootstrap_results, tmp_path):
        """Saves figure to provided path."""
        mock_fig = MagicMock()
        mock_ax_list = [MagicMock() for _ in range(6)]
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = mock_ax_list
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()
        save_path = tmp_path / "bootstrap.png"

        with patch('src.visualization.model_comparison.sort_bootstrap_by_stability',
                   return_value=[(0.02, sample_bootstrap_results[0])]), \
             patch('src.visualization.model_comparison.plot_dict_format_bootstrap'):
            plotter.create_bootstrap_distribution_comparison(
                sample_bootstrap_results, save_path=save_path
            )

        mock_fig.savefig.assert_called_once()

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_custom_top_n(self, mock_tight, mock_subplots, sample_bootstrap_results):
        """Accepts custom top_n_models parameter."""
        mock_fig = MagicMock()
        mock_ax_list = [MagicMock() for _ in range(6)]
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = mock_ax_list
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()

        with patch('src.visualization.model_comparison.sort_bootstrap_by_stability',
                   return_value=[(0.02, sample_bootstrap_results[0])]) as mock_sort, \
             patch('src.visualization.model_comparison.plot_dict_format_bootstrap'):
            plotter.create_bootstrap_distribution_comparison(
                sample_bootstrap_results, top_n_models=3
            )

            mock_sort.assert_called_with(sample_bootstrap_results, 3)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_handles_dict_format_bootstrap(self, mock_tight, mock_subplots, sample_bootstrap_results):
        """Handles dict format bootstrap results."""
        mock_fig = MagicMock()
        mock_ax_list = [MagicMock() for _ in range(6)]
        mock_axes = MagicMock()
        mock_axes.flatten.return_value = mock_ax_list
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()
        sorted_results = [(0.02, sample_bootstrap_results[0])]

        with patch('src.visualization.model_comparison.sort_bootstrap_by_stability',
                   return_value=sorted_results):
            with patch('src.visualization.model_comparison.plot_dict_format_bootstrap') as mock_plot:
                plotter.create_bootstrap_distribution_comparison(sample_bootstrap_results)

                mock_plot.assert_called_once()


class TestCreateFeatureCoefficientAnalysis:
    """Tests for create_feature_coefficient_analysis."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_returns_figure(self, mock_tight, mock_subplots, sample_aic_results,
                            sample_coefficient_stability):
        """Returns matplotlib figure."""
        mock_fig = MagicMock()
        mock_axes = ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()

        with patch('src.visualization.model_comparison.plot_coefficient_heatmap'), \
             patch('src.visualization.model_comparison.plot_sign_consistency_scatter'), \
             patch('src.visualization.model_comparison.plot_economic_constraint_validation'), \
             patch('src.visualization.model_comparison.plot_coefficient_uncertainty'):
            result = plotter.create_feature_coefficient_analysis(
                sample_aic_results, sample_coefficient_stability
            )

        assert result is not None

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_saves_to_path(self, mock_tight, mock_subplots, sample_aic_results,
                           sample_coefficient_stability, tmp_path):
        """Saves figure to provided path."""
        mock_fig = MagicMock()
        mock_axes = ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()
        save_path = tmp_path / "coefficients.png"

        with patch('src.visualization.model_comparison.plot_coefficient_heatmap'), \
             patch('src.visualization.model_comparison.plot_sign_consistency_scatter'), \
             patch('src.visualization.model_comparison.plot_economic_constraint_validation'), \
             patch('src.visualization.model_comparison.plot_coefficient_uncertainty'):
            plotter.create_feature_coefficient_analysis(
                sample_aic_results, sample_coefficient_stability, save_path=save_path
            )

        mock_fig.savefig.assert_called_once()

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.tight_layout')
    def test_custom_top_n(self, mock_tight, mock_subplots, sample_aic_results,
                          sample_coefficient_stability):
        """Accepts custom top_n_models parameter."""
        mock_fig = MagicMock()
        mock_axes = ((MagicMock(), MagicMock()), (MagicMock(), MagicMock()))
        mock_subplots.return_value = (mock_fig, mock_axes)

        plotter = ModelComparisonPlots()

        with patch('src.visualization.model_comparison.plot_coefficient_heatmap') as mock_heatmap, \
             patch('src.visualization.model_comparison.plot_sign_consistency_scatter'), \
             patch('src.visualization.model_comparison.plot_economic_constraint_validation'), \
             patch('src.visualization.model_comparison.plot_coefficient_uncertainty'):
            plotter.create_feature_coefficient_analysis(
                sample_aic_results, sample_coefficient_stability, top_n_models=5
            )

            # Check that top_models has 5 rows
            call_args = mock_heatmap.call_args[0]
            top_models = call_args[1]  # Second positional arg
            assert len(top_models) == 5


class TestBackwardCompatibilityMethods:
    """Tests for backward compatibility wrapper methods."""

    def test_find_pareto_frontier_wrapper(self):
        """_find_pareto_frontier delegates to comparison_helpers."""
        plotter = ModelComparisonPlots()

        with patch('src.visualization.comparison_helpers.find_pareto_frontier',
                   return_value=[0, 2, 4]) as mock_func:
            result = plotter._find_pareto_frontier([1, 2, 3], [3, 2, 1])

            mock_func.assert_called_once_with([1, 2, 3], [3, 2, 1])
            assert result == [0, 2, 4]

    def test_create_model_selection_summary_wrapper(self):
        """_create_model_selection_summary delegates to helper."""
        plotter = ModelComparisonPlots()

        with patch('src.visualization.model_comparison.create_model_selection_summary',
                   return_value="Summary text") as mock_func:
            result = plotter._create_model_selection_summary({}, [], [])

            mock_func.assert_called_once()
            assert result == "Summary text"

    def test_extract_bootstrap_metrics_wrapper(self):
        """_extract_bootstrap_metrics delegates to comparison_helpers."""
        plotter = ModelComparisonPlots()

        with patch('src.visualization.comparison_helpers.extract_bootstrap_metrics',
                   return_value={'metrics': 'data'}) as mock_func:
            result = plotter._extract_bootstrap_metrics([])

            mock_func.assert_called_once_with([])
            assert result == {'metrics': 'data'}


# =============================================================================
# create_model_comparison_report Function Tests
# =============================================================================


class TestCreateModelComparisonReport:
    """Tests for create_model_comparison_report function."""

    def test_creates_output_directory(self, tmp_path):
        """Creates output directory if it doesn't exist."""
        output_dir = tmp_path / "reports" / "nested"

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_bootstrap_distribution_comparison',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            create_model_comparison_report({'aic_results': pd.DataFrame()}, output_dir)

        assert output_dir.exists()

    def test_returns_dict(self, tmp_path, sample_aic_results, sample_bootstrap_results):
        """Returns dictionary of report files."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {'block_bootstrap_results': sample_bootstrap_results},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_bootstrap_distribution_comparison',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        assert isinstance(result, dict)

    def test_generates_multi_criteria_matrix(self, tmp_path, sample_aic_results):
        """Generates multi-criteria matrix when aic_results non-empty."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {'block_bootstrap_results': []},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()) as mock_matrix, \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        mock_matrix.assert_called_once()
        assert 'multi_criteria_matrix' in result

    def test_generates_bootstrap_distributions(self, tmp_path, sample_aic_results,
                                               sample_bootstrap_results):
        """Generates bootstrap distributions when data available."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {'block_bootstrap_results': sample_bootstrap_results},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_bootstrap_distribution_comparison',
                          return_value=MagicMock()) as mock_bootstrap, \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        mock_bootstrap.assert_called_once()
        assert 'bootstrap_distributions' in result

    def test_generates_coefficient_analysis(self, tmp_path, sample_aic_results):
        """Generates coefficient analysis when aic_results non-empty."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {
                'block_bootstrap_results': [],
                'coefficient_stability_analysis': {'feature_a': {'mean': 0.1}},
            },
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()) as mock_coeff, \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        mock_coeff.assert_called_once()
        assert 'coefficient_analysis' in result

    def test_uses_custom_file_prefix(self, tmp_path, sample_aic_results):
        """Uses custom file prefix for output files."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {'block_bootstrap_results': []},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()) as mock_matrix, \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(
                analysis_results, tmp_path, file_prefix="custom_prefix"
            )

        # Check save_path contains custom prefix
        call_kwargs = mock_matrix.call_args[1]
        assert 'custom_prefix' in str(call_kwargs['save_path'])

    def test_handles_empty_aic_results(self, tmp_path):
        """Handles empty aic_results gracefully."""
        analysis_results = {
            'aic_results': pd.DataFrame(),
            'enhanced_metrics': {},
            'bootstrap_results': {},
        }

        result = create_model_comparison_report(analysis_results, tmp_path)

        assert isinstance(result, dict)
        assert 'multi_criteria_matrix' not in result

    def test_handles_exception(self, tmp_path, sample_aic_results):
        """Handles exceptions gracefully and returns partial results."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {},
            'bootstrap_results': {},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          side_effect=Exception("Plot failed")), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        assert isinstance(result, dict)

    def test_skips_bootstrap_when_empty(self, tmp_path, sample_aic_results):
        """Skips bootstrap generation when no data."""
        analysis_results = {
            'aic_results': sample_aic_results,
            'enhanced_metrics': {'information_criteria_analysis': []},
            'bootstrap_results': {'block_bootstrap_results': []},
        }

        with patch.object(ModelComparisonPlots, 'create_multi_criteria_comparison_matrix',
                          return_value=MagicMock()), \
             patch.object(ModelComparisonPlots, 'create_bootstrap_distribution_comparison',
                          return_value=MagicMock()) as mock_bootstrap, \
             patch.object(ModelComparisonPlots, 'create_feature_coefficient_analysis',
                          return_value=MagicMock()), \
             patch('matplotlib.pyplot.close'):
            result = create_model_comparison_report(analysis_results, tmp_path)

        mock_bootstrap.assert_not_called()
        assert 'bootstrap_distributions' not in result


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_model_comparison_plots_exported(self):
        """ModelComparisonPlots is exported."""
        from src.visualization import model_comparison
        assert 'ModelComparisonPlots' in model_comparison.__all__

    def test_create_report_function_exported(self):
        """create_model_comparison_report is exported."""
        from src.visualization import model_comparison
        assert 'create_model_comparison_report' in model_comparison.__all__
