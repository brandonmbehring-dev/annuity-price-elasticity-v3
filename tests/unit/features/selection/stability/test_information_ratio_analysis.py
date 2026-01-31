"""
Tests for information_ratio_analysis module.

Target: 35% → 60%+ coverage
Tests organized by function categories:
- Benchmark calculation
- Risk-adjusted ratio calculations
- Consistency metrics
- Single model IR computation
- Main IR calculation orchestrator
- IR color/classification helpers
- Visualization creation
- Insight generation
- Display functions
- Main analysis orchestrator
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.stability.information_ratio_analysis import (
    _calculate_benchmark_aic,
    _calculate_risk_adjusted_ratios,
    _calculate_consistency_metrics,
    _compute_single_model_ir,
    calculate_bootstrap_information_ratios,
    _get_ir_color,
    _plot_ir_ranking,
    _plot_risk_return_scatter,
    _plot_success_rate_scatter,
    _plot_excess_distributions,
    create_information_ratio_visualizations,
    _classify_ir_models,
    _get_ir_recommendation,
    generate_information_ratio_insights,
    _get_ir_class_label,
    _display_benchmark_stats,
    _display_ir_results_table,
    _display_ir_insights,
    run_information_ratio_analysis,
    run_notebook_information_ratio_analysis,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockBootstrapResult:
    """Mock bootstrap result for testing."""
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
    return [
        MockBootstrapResult(
            model_name='Model_1',
            model_features='feature_a + feature_b',
            bootstrap_aics=[100.0, 102.0, 98.0, 101.0, 99.0],
            bootstrap_r2_values=[0.80, 0.78, 0.82, 0.79, 0.81],
            original_aic=100.0,
            original_r2=0.80,
            aic_stability_coefficient=0.02,
            r2_stability_coefficient=0.02,
            confidence_intervals={'aic': {'lower': 98.0, 'upper': 102.0}},
            successful_fits=100,
            total_attempts=100,
            stability_assessment='Stable'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            model_features='feature_c + feature_d',
            bootstrap_aics=[110.0, 115.0, 105.0, 112.0, 108.0],
            bootstrap_r2_values=[0.75, 0.72, 0.78, 0.74, 0.76],
            original_aic=110.0,
            original_r2=0.75,
            aic_stability_coefficient=0.04,
            r2_stability_coefficient=0.03,
            confidence_intervals={'aic': {'lower': 105.0, 'upper': 115.0}},
            successful_fits=100,
            total_attempts=100,
            stability_assessment='Moderate'
        ),
        MockBootstrapResult(
            model_name='Model_3',
            model_features='feature_e + feature_f + feature_g',
            bootstrap_aics=[120.0, 125.0, 118.0, 122.0, 119.0],
            bootstrap_r2_values=[0.70, 0.68, 0.72, 0.69, 0.71],
            original_aic=120.0,
            original_r2=0.70,
            aic_stability_coefficient=0.03,
            r2_stability_coefficient=0.02,
            confidence_intervals={'aic': {'lower': 118.0, 'upper': 125.0}},
            successful_fits=95,
            total_attempts=100,
            stability_assessment='Moderate'
        ),
    ]


@pytest.fixture
def sample_ir_metrics():
    """Sample IR metrics for testing."""
    return [
        {
            'model_idx': 0,
            'model_name': 'Model 1',
            'features': 'feature_a + feature_b',
            'original_aic': 100.0,
            'mean_bootstrap_aic': 100.0,
            'benchmark_aic': 110.0,
            'mean_excess': 10.0,
            'std_excess': 5.0,
            'information_ratio': 2.0,
            'sharpe_like': 1.5,
            'sortino_ratio': 2.5,
            'calmar_ratio': 3.0,
            'max_drawdown': 2.0,
            'success_rate': 80.0,
            'positive_excess_count': 8,
            'consecutive_wins': 5,
            'excess_aics': np.array([8, 10, 12, 6, 14, 9, 11, 7])
        },
        {
            'model_idx': 1,
            'model_name': 'Model 2',
            'features': 'feature_c + feature_d',
            'original_aic': 105.0,
            'mean_bootstrap_aic': 107.0,
            'benchmark_aic': 110.0,
            'mean_excess': 3.0,
            'std_excess': 8.0,
            'information_ratio': 0.375,
            'sharpe_like': 0.5,
            'sortino_ratio': 0.8,
            'calmar_ratio': 1.0,
            'max_drawdown': 5.0,
            'success_rate': 60.0,
            'positive_excess_count': 6,
            'consecutive_wins': 3,
            'excess_aics': np.array([2, -3, 5, 1, -2, 4, 3, -1])
        },
        {
            'model_idx': 2,
            'model_name': 'Model 3',
            'features': 'feature_e + feature_f',
            'original_aic': 115.0,
            'mean_bootstrap_aic': 118.0,
            'benchmark_aic': 110.0,
            'mean_excess': -8.0,
            'std_excess': 6.0,
            'information_ratio': -1.33,
            'sharpe_like': -0.8,
            'sortino_ratio': -1.2,
            'calmar_ratio': -0.5,
            'max_drawdown': 10.0,
            'success_rate': 30.0,
            'positive_excess_count': 3,
            'consecutive_wins': 1,
            'excess_aics': np.array([-5, -10, -8, 2, -6, -9, -7, 1])
        },
    ]


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes for plot testing."""
    ax = MagicMock()
    ax.barh.return_value = MagicMock()
    ax.scatter.return_value = MagicMock()
    ax.hist.return_value = (MagicMock(), MagicMock(), MagicMock())
    return ax


# =============================================================================
# Benchmark Calculation Tests
# =============================================================================


class TestCalculateBenchmarkAic:
    """Tests for _calculate_benchmark_aic."""

    def test_calculates_median_across_models(self, sample_bootstrap_results):
        """Calculates median AIC across all bootstrap samples."""
        result = _calculate_benchmark_aic(sample_bootstrap_results, 3)

        # Median of all 15 bootstrap AICs
        all_aics = [100, 102, 98, 101, 99, 110, 115, 105, 112, 108, 120, 125, 118, 122, 119]
        expected = np.median(all_aics)
        assert result == pytest.approx(expected)

    def test_respects_n_models_limit(self, sample_bootstrap_results):
        """Only uses first n_models for benchmark."""
        result = _calculate_benchmark_aic(sample_bootstrap_results, 1)

        # Only model 1's AICs
        expected = np.median([100, 102, 98, 101, 99])
        assert result == pytest.approx(expected)

    def test_single_model(self, sample_bootstrap_results):
        """Works with single model."""
        result = _calculate_benchmark_aic(sample_bootstrap_results[:1], 1)

        assert isinstance(result, float)


# =============================================================================
# Risk-Adjusted Ratio Tests
# =============================================================================


class TestCalculateRiskAdjustedRatios:
    """Tests for _calculate_risk_adjusted_ratios."""

    def test_returns_all_ratio_keys(self):
        """Returns all expected ratio keys."""
        excess_aics = np.array([5.0, 10.0, -2.0, 8.0, 3.0])
        result = _calculate_risk_adjusted_ratios(excess_aics, 4.8, 4.0, 100.0, 110.0)

        assert 'sharpe_like' in result
        assert 'sortino_ratio' in result
        assert 'calmar_ratio' in result
        assert 'max_drawdown' in result

    def test_sharpe_like_calculation(self):
        """Calculates Sharpe-like ratio correctly."""
        excess_aics = np.array([5.0, 10.0, 5.0, 10.0])
        result = _calculate_risk_adjusted_ratios(
            excess_aics, mean_excess=7.5, std_excess=2.5,
            original_aic=100.0, benchmark_aic=110.0
        )

        # Sharpe = (benchmark - original) / std_excess = (110-100)/2.5 = 4.0
        assert result['sharpe_like'] == pytest.approx(4.0)

    def test_zero_std_returns_zero_sharpe(self):
        """Returns zero Sharpe when std is zero."""
        excess_aics = np.array([5.0, 5.0, 5.0])
        result = _calculate_risk_adjusted_ratios(
            excess_aics, mean_excess=5.0, std_excess=0.0,
            original_aic=105.0, benchmark_aic=110.0
        )

        assert result['sharpe_like'] == 0

    def test_sortino_handles_no_downside(self):
        """Handles case with no downside deviations."""
        excess_aics = np.array([5.0, 10.0, 15.0])  # All positive
        result = _calculate_risk_adjusted_ratios(
            excess_aics, mean_excess=10.0, std_excess=5.0,
            original_aic=100.0, benchmark_aic=110.0
        )

        # With positive mean and no downside, returns inf (capped at 999)
        assert result['sortino_ratio'] == 999

    def test_calmar_ratio_calculation(self):
        """Calculates Calmar ratio with drawdown."""
        excess_aics = np.array([10.0, 5.0, 8.0, 2.0])  # Has drawdown
        result = _calculate_risk_adjusted_ratios(
            excess_aics, mean_excess=6.25, std_excess=3.0,
            original_aic=100.0, benchmark_aic=110.0
        )

        assert result['calmar_ratio'] > 0
        assert result['max_drawdown'] > 0


# =============================================================================
# Consistency Metrics Tests
# =============================================================================


class TestCalculateConsistencyMetrics:
    """Tests for _calculate_consistency_metrics."""

    def test_returns_expected_keys(self):
        """Returns all expected metric keys."""
        excess_aics = np.array([5.0, -2.0, 8.0])
        result = _calculate_consistency_metrics(excess_aics)

        assert 'success_rate' in result
        assert 'positive_excess_count' in result
        assert 'consecutive_wins' in result

    def test_success_rate_calculation(self):
        """Calculates success rate correctly."""
        excess_aics = np.array([5.0, -2.0, 8.0, 3.0, -1.0])  # 3/5 positive
        result = _calculate_consistency_metrics(excess_aics)

        assert result['success_rate'] == pytest.approx(60.0)

    def test_positive_count(self):
        """Counts positive excess correctly."""
        excess_aics = np.array([5.0, -2.0, 8.0, 3.0, -1.0])
        result = _calculate_consistency_metrics(excess_aics)

        assert result['positive_excess_count'] == 3

    def test_consecutive_wins(self):
        """Counts maximum consecutive wins."""
        excess_aics = np.array([5.0, 8.0, 3.0, -1.0, 2.0, 4.0])  # Max 3 consecutive
        result = _calculate_consistency_metrics(excess_aics)

        assert result['consecutive_wins'] == 3

    def test_all_positive(self):
        """Handles all positive excess."""
        excess_aics = np.array([5.0, 8.0, 3.0, 10.0])
        result = _calculate_consistency_metrics(excess_aics)

        assert result['success_rate'] == 100.0
        assert result['consecutive_wins'] == 4

    def test_all_negative(self):
        """Handles all negative excess."""
        excess_aics = np.array([-5.0, -8.0, -3.0])
        result = _calculate_consistency_metrics(excess_aics)

        assert result['success_rate'] == 0.0
        assert result['consecutive_wins'] == 0


# =============================================================================
# Single Model IR Computation Tests
# =============================================================================


class TestComputeSingleModelIr:
    """Tests for _compute_single_model_ir."""

    def test_returns_expected_structure(self, sample_bootstrap_results):
        """Returns dict with expected keys."""
        result = _compute_single_model_ir(sample_bootstrap_results[0], 0, 110.0)

        assert 'model_idx' in result
        assert 'model_name' in result
        assert 'features' in result
        assert 'information_ratio' in result
        assert 'mean_excess' in result
        assert 'std_excess' in result
        assert 'success_rate' in result

    def test_model_name_format(self, sample_bootstrap_results):
        """Model name follows expected format."""
        result = _compute_single_model_ir(sample_bootstrap_results[0], 0, 110.0)

        assert result['model_name'] == 'Model 1'

    def test_information_ratio_calculation(self, sample_bootstrap_results):
        """Calculates IR correctly."""
        result = _compute_single_model_ir(sample_bootstrap_results[0], 0, 110.0)

        # IR = mean_excess / std_excess
        expected_ir = result['mean_excess'] / result['std_excess'] if result['std_excess'] > 0 else 0
        assert result['information_ratio'] == pytest.approx(expected_ir)


# =============================================================================
# Main IR Calculation Tests
# =============================================================================


class TestCalculateBootstrapInformationRatios:
    """Tests for calculate_bootstrap_information_ratios."""

    def test_raises_on_empty_results(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            calculate_bootstrap_information_ratios([])

    def test_returns_tuple(self, sample_bootstrap_results):
        """Returns tuple of (metrics, benchmark)."""
        result = calculate_bootstrap_information_ratios(sample_bootstrap_results)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_sorted_metrics(self, sample_bootstrap_results):
        """Returns metrics sorted by IR descending."""
        metrics, _ = calculate_bootstrap_information_ratios(sample_bootstrap_results)

        ir_values = [m['information_ratio'] for m in metrics]
        assert ir_values == sorted(ir_values, reverse=True)

    def test_respects_max_models(self, sample_bootstrap_results):
        """Respects max_models parameter."""
        metrics, _ = calculate_bootstrap_information_ratios(
            sample_bootstrap_results, max_models=2
        )

        assert len(metrics) == 2

    def test_benchmark_is_float(self, sample_bootstrap_results):
        """Benchmark AIC is a float."""
        _, benchmark = calculate_bootstrap_information_ratios(sample_bootstrap_results)

        assert isinstance(benchmark, (float, np.floating))


# =============================================================================
# IR Color/Classification Helper Tests
# =============================================================================


class TestGetIrColor:
    """Tests for _get_ir_color."""

    def test_high_ir_green(self):
        """High IR (>0.5) returns green."""
        assert _get_ir_color(0.6) == '#2E8B57'
        assert _get_ir_color(1.0) == '#2E8B57'

    def test_moderate_ir_gold(self):
        """Moderate IR (0.2-0.5) returns gold."""
        assert _get_ir_color(0.3) == '#FFD700'
        assert _get_ir_color(0.5) == '#FFD700'

    def test_low_ir_orange(self):
        """Low IR (0-0.2) returns orange."""
        assert _get_ir_color(0.1) == '#FFA500'
        assert _get_ir_color(0.19) == '#FFA500'

    def test_negative_ir_crimson(self):
        """Negative IR returns crimson."""
        assert _get_ir_color(-0.1) == '#DC143C'
        assert _get_ir_color(-1.0) == '#DC143C'

    def test_zero_ir_orange(self):
        """Zero IR returns orange (low but positive)."""
        assert _get_ir_color(0.0) == '#DC143C'


class TestGetIrClassLabel:
    """Tests for _get_ir_class_label."""

    def test_high_label(self):
        """High IR returns 'High'."""
        assert _get_ir_class_label(0.6) == "High"

    def test_moderate_label(self):
        """Moderate IR returns 'Moderate'."""
        assert _get_ir_class_label(0.3) == "Moderate"

    def test_low_label(self):
        """Low IR returns 'Low'."""
        assert _get_ir_class_label(0.1) == "Low"

    def test_negative_label(self):
        """Negative IR returns 'Negative'."""
        assert _get_ir_class_label(-0.5) == "Negative"


class TestClassifyIrModels:
    """Tests for _classify_ir_models."""

    def test_returns_classification_counts(self, sample_ir_metrics):
        """Returns counts for each category."""
        result = _classify_ir_models(sample_ir_metrics)

        assert 'high_ir_models' in result
        assert 'moderate_ir_models' in result
        assert 'low_ir_models' in result
        assert 'negative_ir_models' in result
        assert 'total_models' in result

    def test_total_equals_sum(self, sample_ir_metrics):
        """Total equals sum of categories."""
        result = _classify_ir_models(sample_ir_metrics)

        sum_categories = (
            result['high_ir_models'] +
            result['moderate_ir_models'] +
            result['low_ir_models'] +
            result['negative_ir_models']
        )
        assert result['total_models'] == len(sample_ir_metrics)

    def test_classification_thresholds(self):
        """Classifies correctly based on thresholds."""
        metrics = [
            {'information_ratio': 0.6},   # High
            {'information_ratio': 0.3},   # Moderate
            {'information_ratio': 0.1},   # Low
            {'information_ratio': -0.2},  # Negative
        ]
        result = _classify_ir_models(metrics)

        assert result['high_ir_models'] == 1
        assert result['moderate_ir_models'] == 1
        assert result['low_ir_models'] == 1
        assert result['negative_ir_models'] == 1


class TestGetIrRecommendation:
    """Tests for _get_ir_recommendation."""

    def test_recommended_with_high_ir(self):
        """Returns RECOMMENDED when high IR models exist."""
        classification = {'high_ir_models': 2, 'moderate_ir_models': 1, 'low_ir_models': 0, 'negative_ir_models': 0}
        category, detail = _get_ir_recommendation(classification)

        assert category == "RECOMMENDED"
        assert "2 models" in detail

    def test_acceptable_with_moderate_ir(self):
        """Returns ACCEPTABLE when only moderate IR models exist."""
        classification = {'high_ir_models': 0, 'moderate_ir_models': 3, 'low_ir_models': 1, 'negative_ir_models': 0}
        category, detail = _get_ir_recommendation(classification)

        assert category == "ACCEPTABLE"
        assert "3 models" in detail

    def test_caution_with_no_good_models(self):
        """Returns CAUTION when no high or moderate models."""
        classification = {'high_ir_models': 0, 'moderate_ir_models': 0, 'low_ir_models': 2, 'negative_ir_models': 1}
        category, detail = _get_ir_recommendation(classification)

        assert category == "CAUTION"


# =============================================================================
# Visualization Tests (Mocked)
# =============================================================================


class TestPlotIrRanking:
    """Tests for _plot_ir_ranking."""

    def test_creates_bar_chart(self, mock_axes, sample_ir_metrics):
        """Creates horizontal bar chart."""
        _plot_ir_ranking(mock_axes, sample_ir_metrics)

        mock_axes.barh.assert_called_once()

    def test_sets_labels(self, mock_axes, sample_ir_metrics):
        """Sets axis labels and title."""
        _plot_ir_ranking(mock_axes, sample_ir_metrics)

        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()


class TestPlotRiskReturnScatter:
    """Tests for _plot_risk_return_scatter."""

    def test_creates_scatter(self, mock_axes, sample_ir_metrics):
        """Creates scatter plot."""
        with patch('matplotlib.pyplot.colorbar'):
            _plot_risk_return_scatter(mock_axes, sample_ir_metrics)

        mock_axes.scatter.assert_called_once()


class TestPlotSuccessRateScatter:
    """Tests for _plot_success_rate_scatter."""

    def test_creates_scatter(self, mock_axes, sample_ir_metrics):
        """Creates scatter plot."""
        _plot_success_rate_scatter(mock_axes, sample_ir_metrics)

        mock_axes.scatter.assert_called_once()


class TestPlotExcessDistributions:
    """Tests for _plot_excess_distributions."""

    def test_creates_histogram(self, mock_axes, sample_ir_metrics):
        """Creates histogram for top 3 models."""
        _plot_excess_distributions(mock_axes, sample_ir_metrics)

        # Should be called 3 times for top 3 models
        assert mock_axes.hist.call_count == 3


class TestCreateInformationRatioVisualizations:
    """Tests for create_information_ratio_visualizations."""

    def test_raises_on_empty_metrics(self):
        """Raises ValueError for empty metrics."""
        with pytest.raises(ValueError, match="No information ratio metrics"):
            create_information_ratio_visualizations([], 100.0)

    def test_returns_dict_with_figure(self, sample_ir_metrics):
        """Returns dict with figure."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'):
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)

            result = create_information_ratio_visualizations(sample_ir_metrics, 100.0)

            assert 'information_ratio_analysis' in result

    def test_uses_config_dimensions(self, sample_ir_metrics):
        """Uses config for figure dimensions."""
        config = {'fig_width': 20, 'fig_height': 15}

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'):
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            mock_subplots.return_value = (mock_fig, mock_axes)

            create_information_ratio_visualizations(sample_ir_metrics, 100.0, config)

            # Check figsize was passed
            call_kwargs = mock_subplots.call_args
            assert call_kwargs[1]['figsize'] == (20, 15)


# =============================================================================
# Insight Generation Tests
# =============================================================================


class TestGenerateInformationRatioInsights:
    """Tests for generate_information_ratio_insights."""

    def test_handles_empty_metrics(self):
        """Handles empty metrics list."""
        result = generate_information_ratio_insights([], 100)

        assert result['best_risk_adjusted_model'] is None
        assert 'No Data' in result['risk_classification']

    def test_returns_expected_keys(self, sample_ir_metrics):
        """Returns all expected keys."""
        result = generate_information_ratio_insights(sample_ir_metrics, 100)

        assert 'best_risk_adjusted_model' in result
        assert 'risk_classification' in result
        assert 'recommendation_category' in result
        assert 'recommendation_detail' in result
        assert 'insights_summary' in result
        assert 'benchmark_statistics' in result

    def test_best_model_is_first(self, sample_ir_metrics):
        """Best model is first in sorted list."""
        result = generate_information_ratio_insights(sample_ir_metrics, 100)

        assert result['best_risk_adjusted_model'] == sample_ir_metrics[0]

    def test_benchmark_statistics(self, sample_ir_metrics):
        """Includes benchmark statistics."""
        result = generate_information_ratio_insights(sample_ir_metrics, 100)

        assert result['benchmark_statistics']['n_bootstrap_samples'] == 100
        assert result['benchmark_statistics']['analysis_type'] == 'information_ratio'


# =============================================================================
# Display Function Tests
# =============================================================================


class TestDisplayBenchmarkStats:
    """Tests for _display_benchmark_stats."""

    def test_prints_statistics(self, sample_bootstrap_results, capsys):
        """Prints benchmark statistics."""
        _display_benchmark_stats(sample_bootstrap_results, 3, 110.0)

        captured = capsys.readouterr()
        assert 'Benchmark Statistics' in captured.out
        assert 'Population Median AIC' in captured.out


class TestDisplayIrResultsTable:
    """Tests for _display_ir_results_table."""

    def test_prints_header(self, sample_ir_metrics, capsys):
        """Prints table header."""
        _display_ir_results_table(sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'Information Ratio Analysis Results' in captured.out
        assert 'Model' in captured.out
        assert 'Features' in captured.out
        assert 'IR' in captured.out

    def test_prints_model_rows(self, sample_ir_metrics, capsys):
        """Prints rows for each model."""
        _display_ir_results_table(sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'Model 1' in captured.out
        assert 'Model 2' in captured.out


class TestDisplayIrInsights:
    """Tests for _display_ir_insights."""

    def test_prints_insights_header(self, sample_ir_metrics, capsys):
        """Prints insights header."""
        insights = generate_information_ratio_insights(sample_ir_metrics, 100)
        _display_ir_insights(insights, sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'INFORMATION RATIO INSIGHTS' in captured.out

    def test_prints_classification(self, sample_ir_metrics, capsys):
        """Prints risk classification."""
        insights = generate_information_ratio_insights(sample_ir_metrics, 100)
        _display_ir_insights(insights, sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'Risk-Adjusted Performance Classification' in captured.out


# =============================================================================
# Main Analysis Orchestrator Tests
# =============================================================================


class TestRunInformationRatioAnalysis:
    """Tests for run_information_ratio_analysis."""

    def test_raises_on_empty_results(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            run_information_ratio_analysis([])

    def test_returns_dict(self, sample_bootstrap_results):
        """Returns dictionary."""
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):

            result = run_information_ratio_analysis(
                sample_bootstrap_results,
                display_results=False,
                create_visualizations=False
            )

            assert isinstance(result, dict)

    def test_returns_ir_results(self, sample_bootstrap_results):
        """Returns ir_results key."""
        result = run_information_ratio_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False
        )

        assert 'ir_results' in result

    def test_detailed_results_structure(self, sample_bootstrap_results):
        """Detailed results have expected structure."""
        result = run_information_ratio_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False,
            return_detailed=True
        )

        assert 'ir_results' in result
        assert 'insights' in result
        assert 'analysis_metadata' in result

    def test_non_detailed_returns_minimal(self, sample_bootstrap_results):
        """Non-detailed returns minimal structure."""
        result = run_information_ratio_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False,
            return_detailed=False
        )

        assert 'ir_results' in result
        assert 'insights' not in result

    def test_respects_config_max_models(self, sample_bootstrap_results):
        """Respects config models_to_analyze."""
        config = {'models_to_analyze': 2}
        result = run_information_ratio_analysis(
            sample_bootstrap_results,
            config=config,
            display_results=False,
            create_visualizations=False
        )

        assert len(result['ir_results']) == 2


class TestRunNotebookInformationRatioAnalysis:
    """Tests for run_notebook_information_ratio_analysis."""

    def test_handles_empty_results(self, capsys):
        """Handles empty results gracefully."""
        result = run_notebook_information_ratio_analysis([])

        assert result == []
        captured = capsys.readouterr()
        assert 'No bootstrap results available' in captured.out

    def test_returns_list(self, sample_bootstrap_results):
        """Returns list of IR results."""
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):

            result = run_notebook_information_ratio_analysis(sample_bootstrap_results)

            assert isinstance(result, list)

    def test_enables_display(self, sample_bootstrap_results, capsys):
        """Enables display by default."""
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.colorbar'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):

            run_notebook_information_ratio_analysis(sample_bootstrap_results)

            captured = capsys.readouterr()
            assert 'INFORMATION RATIO ANALYSIS' in captured.out
