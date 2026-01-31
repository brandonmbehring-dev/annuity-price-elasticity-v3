"""
Tests for bootstrap_win_rate_analysis module.

Target: 45% → 70%+ coverage
Tests organized by function categories:
- Bootstrap matrix building
- Win count computation
- Win rate calculation
- Color helpers
- Visualization creation
- Insights generation
- Display functions
- Main analysis orchestration
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.stability.bootstrap_win_rate_analysis import (
    _build_bootstrap_matrix,
    _compute_win_counts,
    calculate_model_win_rates,
    _get_win_rate_color,
    _plot_win_rate_bars,
    _plot_win_rate_distribution,
    create_win_rate_visualizations,
    _assess_selection_confidence,
    generate_win_rate_insights,
    _display_win_rate_table,
    _validate_bootstrap_inputs,
    _display_insights_output,
    _create_and_display_visualizations,
    _build_detailed_results,
    run_bootstrap_win_rate_analysis,
    run_notebook_win_rate_analysis,
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
            model_features='feature_e + feature_f',
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
def sample_win_rate_results():
    """Sample win rate results for testing."""
    return [
        {'model': 'Model 1', 'features': 'a + b', 'win_rate_pct': 60.0, 'win_count': 60,
         'original_aic': 100.0, 'median_bootstrap_aic': 100.0},
        {'model': 'Model 2', 'features': 'c + d', 'win_rate_pct': 25.0, 'win_count': 25,
         'original_aic': 105.0, 'median_bootstrap_aic': 107.0},
        {'model': 'Model 3', 'features': 'e + f', 'win_rate_pct': 15.0, 'win_count': 15,
         'original_aic': 110.0, 'median_bootstrap_aic': 112.0},
    ]


@pytest.fixture
def mock_axes():
    """Mock matplotlib axes."""
    ax = MagicMock()
    ax.bar.return_value = MagicMock()
    ax.hist.return_value = (MagicMock(), MagicMock(), MagicMock())
    return ax


# =============================================================================
# Bootstrap Matrix Building Tests
# =============================================================================


class TestBuildBootstrapMatrix:
    """Tests for _build_bootstrap_matrix."""

    def test_returns_tuple(self, sample_bootstrap_results):
        """Returns tuple of (matrix, names)."""
        result = _build_bootstrap_matrix(sample_bootstrap_results, 3)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_shape(self, sample_bootstrap_results):
        """Matrix has correct shape (n_samples x n_models)."""
        matrix, _ = _build_bootstrap_matrix(sample_bootstrap_results, 3)

        assert matrix.shape == (5, 3)  # 5 samples, 3 models

    def test_model_names(self, sample_bootstrap_results):
        """Returns correct model names."""
        _, names = _build_bootstrap_matrix(sample_bootstrap_results, 3)

        assert names == ['Model 1', 'Model 2', 'Model 3']

    def test_matrix_values(self, sample_bootstrap_results):
        """Matrix contains correct AIC values."""
        matrix, _ = _build_bootstrap_matrix(sample_bootstrap_results, 2)

        # First model's AICs
        assert matrix[0, 0] == 100.0
        assert matrix[1, 0] == 102.0

    def test_respects_n_models(self, sample_bootstrap_results):
        """Respects n_models limit."""
        matrix, names = _build_bootstrap_matrix(sample_bootstrap_results, 2)

        assert matrix.shape[1] == 2
        assert len(names) == 2


# =============================================================================
# Win Count Computation Tests
# =============================================================================


class TestComputeWinCounts:
    """Tests for _compute_win_counts."""

    def test_returns_array(self):
        """Returns numpy array."""
        matrix = np.array([[100, 110], [105, 102], [98, 99]])
        result = _compute_win_counts(matrix)

        assert isinstance(result, np.ndarray)

    def test_sum_equals_samples(self):
        """Sum of wins equals number of samples."""
        matrix = np.array([[100, 110], [105, 102], [98, 99]])
        result = _compute_win_counts(matrix)

        assert result.sum() == 3  # 3 samples

    def test_correct_winner(self):
        """Identifies correct winner in each sample."""
        # Model 1 always wins (lower AIC)
        matrix = np.array([[100, 200], [100, 200], [100, 200]])
        result = _compute_win_counts(matrix)

        assert result[0] == 3  # Model 1 wins all
        assert result[1] == 0  # Model 2 wins none

    def test_mixed_winners(self):
        """Handles mixed winners."""
        matrix = np.array([[100, 110], [110, 100], [100, 110]])
        result = _compute_win_counts(matrix)

        assert result[0] == 2  # Model 1 wins 2
        assert result[1] == 1  # Model 2 wins 1


# =============================================================================
# Win Rate Calculation Tests
# =============================================================================


class TestCalculateModelWinRates:
    """Tests for calculate_model_win_rates."""

    def test_raises_on_empty(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            calculate_model_win_rates([])

    def test_returns_list(self, sample_bootstrap_results):
        """Returns list of dicts."""
        result = calculate_model_win_rates(sample_bootstrap_results)

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_sorted_by_win_rate(self, sample_bootstrap_results):
        """Results are sorted by win rate descending."""
        result = calculate_model_win_rates(sample_bootstrap_results)

        rates = [r['win_rate_pct'] for r in result]
        assert rates == sorted(rates, reverse=True)

    def test_contains_expected_keys(self, sample_bootstrap_results):
        """Results contain expected keys."""
        result = calculate_model_win_rates(sample_bootstrap_results)

        expected_keys = ['model', 'features', 'win_rate_pct', 'win_count',
                        'original_aic', 'median_bootstrap_aic']
        assert all(key in result[0] for key in expected_keys)

    def test_respects_max_models(self, sample_bootstrap_results):
        """Respects max_models parameter."""
        result = calculate_model_win_rates(sample_bootstrap_results, max_models=2)

        assert len(result) == 2

    def test_win_rates_sum_to_100(self, sample_bootstrap_results):
        """Win rates sum to 100%."""
        result = calculate_model_win_rates(sample_bootstrap_results)

        total = sum(r['win_rate_pct'] for r in result)
        assert total == pytest.approx(100.0)


# =============================================================================
# Color Helper Tests
# =============================================================================


class TestGetWinRateColor:
    """Tests for _get_win_rate_color."""

    def test_high_rate_green(self):
        """High rate (>20%) returns green."""
        assert _get_win_rate_color(25.0) == '#2E8B57'
        assert _get_win_rate_color(50.0) == '#2E8B57'

    def test_medium_rate_orange(self):
        """Medium rate (10-20%) returns orange."""
        assert _get_win_rate_color(15.0) == '#FF6B35'
        assert _get_win_rate_color(20.0) == '#FF6B35'

    def test_low_rate_blue(self):
        """Low rate (<10%) returns blue."""
        assert _get_win_rate_color(5.0) == '#6B73FF'
        assert _get_win_rate_color(10.0) == '#6B73FF'


# =============================================================================
# Visualization Tests
# =============================================================================


class TestPlotWinRateBars:
    """Tests for _plot_win_rate_bars."""

    def test_creates_bar_chart(self, mock_axes, sample_win_rate_results):
        """Creates bar chart."""
        _plot_win_rate_bars(mock_axes, sample_win_rate_results)

        mock_axes.bar.assert_called_once()

    def test_sets_labels(self, mock_axes, sample_win_rate_results):
        """Sets axis labels."""
        _plot_win_rate_bars(mock_axes, sample_win_rate_results)

        mock_axes.set_xlabel.assert_called()
        mock_axes.set_ylabel.assert_called()
        mock_axes.set_title.assert_called()


class TestPlotWinRateDistribution:
    """Tests for _plot_win_rate_distribution."""

    def test_creates_histogram(self, mock_axes, sample_win_rate_results):
        """Creates histogram."""
        _plot_win_rate_distribution(mock_axes, sample_win_rate_results)

        mock_axes.hist.assert_called_once()

    def test_adds_mean_line(self, mock_axes, sample_win_rate_results):
        """Adds vertical mean line."""
        _plot_win_rate_distribution(mock_axes, sample_win_rate_results)

        mock_axes.axvline.assert_called()


class TestCreateWinRateVisualizations:
    """Tests for create_win_rate_visualizations."""

    def test_raises_on_empty(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No win rate results"):
            create_win_rate_visualizations([], 100)

    def test_returns_dict_with_figure(self, sample_win_rate_results):
        """Returns dict with figure."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            mock_fig = MagicMock()
            mock_axes = (MagicMock(), MagicMock())
            mock_subplots.return_value = (mock_fig, mock_axes)

            result = create_win_rate_visualizations(sample_win_rate_results, 100)

            assert 'win_rate_chart' in result

    def test_uses_config_dimensions(self, sample_win_rate_results):
        """Uses config for figure dimensions."""
        config = {'fig_width': 20, 'fig_height': 10}

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'):
            mock_fig = MagicMock()
            mock_subplots.return_value = (mock_fig, (MagicMock(), MagicMock()))

            create_win_rate_visualizations(sample_win_rate_results, 100, config)

            call_kwargs = mock_subplots.call_args
            assert call_kwargs[1]['figsize'] == (20, 10)


# =============================================================================
# Confidence Assessment Tests
# =============================================================================


class TestAssessSelectionConfidence:
    """Tests for _assess_selection_confidence."""

    def test_high_confidence(self):
        """High rate (>30%) gives high confidence."""
        confidence, rationale = _assess_selection_confidence(35.0)

        assert confidence == 'High'
        assert '35.0%' in rationale

    def test_moderate_confidence(self):
        """Medium rate (20-30%) gives moderate confidence."""
        confidence, rationale = _assess_selection_confidence(25.0)

        assert confidence == 'Moderate'
        assert 'reasonable consistency' in rationale

    def test_low_confidence(self):
        """Low rate (<20%) gives low confidence."""
        confidence, rationale = _assess_selection_confidence(15.0)

        assert confidence == 'Low'
        assert 'high uncertainty' in rationale


# =============================================================================
# Insights Generation Tests
# =============================================================================


class TestGenerateWinRateInsights:
    """Tests for generate_win_rate_insights."""

    def test_handles_empty_results(self):
        """Handles empty results."""
        result = generate_win_rate_insights([], 100)

        assert result['top_performer'] is None
        assert result['selection_confidence'] == 'No Data'

    def test_returns_expected_keys(self, sample_win_rate_results):
        """Returns all expected keys."""
        result = generate_win_rate_insights(sample_win_rate_results, 100)

        expected_keys = ['top_performer', 'competitive_models', 'selection_confidence',
                        'insights_summary', 'analysis_metadata']
        assert all(key in result for key in expected_keys)

    def test_top_performer_is_first(self, sample_win_rate_results):
        """Top performer is first result."""
        result = generate_win_rate_insights(sample_win_rate_results, 100)

        assert result['top_performer'] == sample_win_rate_results[0]

    def test_competitive_models_threshold(self, sample_win_rate_results):
        """Competitive models are those with >10% win rate."""
        result = generate_win_rate_insights(sample_win_rate_results, 100)

        # Model 1 (60%), Model 2 (25%), Model 3 (15%) all > 10%
        assert len(result['competitive_models']) == 3

    def test_metadata_content(self, sample_win_rate_results):
        """Metadata contains expected fields."""
        result = generate_win_rate_insights(sample_win_rate_results, 100)

        assert result['analysis_metadata']['n_bootstrap_samples'] == 100
        assert result['analysis_metadata']['analysis_type'] == 'bootstrap_win_rate'


# =============================================================================
# Display Function Tests
# =============================================================================


class TestDisplayWinRateTable:
    """Tests for _display_win_rate_table."""

    def test_prints_header(self, sample_win_rate_results, capsys):
        """Prints table header."""
        _display_win_rate_table(sample_win_rate_results, 100)

        captured = capsys.readouterr()
        assert 'Bootstrap Win Rate Analysis' in captured.out
        assert 'Rank' in captured.out
        assert 'Model' in captured.out

    def test_prints_model_rows(self, sample_win_rate_results, capsys):
        """Prints rows for each model."""
        _display_win_rate_table(sample_win_rate_results, 100)

        captured = capsys.readouterr()
        assert 'Model 1' in captured.out
        assert 'Model 2' in captured.out


class TestValidateBootstrapInputs:
    """Tests for _validate_bootstrap_inputs."""

    def test_raises_on_empty(self):
        """Raises ValueError for empty list."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            _validate_bootstrap_inputs([])

    def test_raises_on_empty_aics(self, sample_bootstrap_results):
        """Raises ValueError when AICs are empty."""
        sample_bootstrap_results[0].bootstrap_aics = []

        with pytest.raises(ValueError, match="no AIC values"):
            _validate_bootstrap_inputs(sample_bootstrap_results)

    def test_accepts_valid(self, sample_bootstrap_results):
        """Accepts valid bootstrap results."""
        _validate_bootstrap_inputs(sample_bootstrap_results)  # Should not raise


class TestDisplayInsightsOutput:
    """Tests for _display_insights_output."""

    def test_prints_header(self, capsys):
        """Prints insights header."""
        insights = {'insights_summary': 'Test summary'}

        _display_insights_output(insights)

        captured = capsys.readouterr()
        assert 'WIN RATE INSIGHTS' in captured.out


# =============================================================================
# Main Analysis Tests
# =============================================================================


class TestRunBootstrapWinRateAnalysis:
    """Tests for run_bootstrap_win_rate_analysis."""

    def test_raises_on_empty(self):
        """Raises ValueError for empty results."""
        with pytest.raises(ValueError, match="No bootstrap results"):
            run_bootstrap_win_rate_analysis([])

    def test_returns_dict(self, sample_bootstrap_results):
        """Returns dictionary."""
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):

            result = run_bootstrap_win_rate_analysis(
                sample_bootstrap_results,
                display_results=False,
                create_visualizations=False
            )

            assert isinstance(result, dict)

    def test_contains_win_rate_results(self, sample_bootstrap_results):
        """Result contains win_rate_results."""
        result = run_bootstrap_win_rate_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False
        )

        assert 'win_rate_results' in result

    def test_detailed_results(self, sample_bootstrap_results):
        """Detailed results contain all sections."""
        result = run_bootstrap_win_rate_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False,
            return_detailed=True
        )

        assert 'win_rate_results' in result
        assert 'insights' in result
        assert 'analysis_metadata' in result

    def test_minimal_results(self, sample_bootstrap_results):
        """Non-detailed results are minimal."""
        result = run_bootstrap_win_rate_analysis(
            sample_bootstrap_results,
            display_results=False,
            create_visualizations=False,
            return_detailed=False
        )

        assert 'win_rate_results' in result
        assert 'insights' not in result

    def test_respects_config(self, sample_bootstrap_results):
        """Respects config parameters."""
        config = {'models_to_analyze': 2}

        result = run_bootstrap_win_rate_analysis(
            sample_bootstrap_results,
            config=config,
            display_results=False,
            create_visualizations=False
        )

        assert len(result['win_rate_results']) == 2


class TestRunNotebookWinRateAnalysis:
    """Tests for run_notebook_win_rate_analysis."""

    def test_handles_empty(self, capsys):
        """Handles empty results."""
        result = run_notebook_win_rate_analysis([])

        assert result == []
        captured = capsys.readouterr()
        assert 'No bootstrap results available' in captured.out

    def test_returns_list(self, sample_bootstrap_results):
        """Returns list of win rate results."""
        with patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):

            result = run_notebook_win_rate_analysis(sample_bootstrap_results)

            assert isinstance(result, list)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateAndDisplayVisualizations:
    """Tests for _create_and_display_visualizations."""

    def test_returns_dict(self, sample_win_rate_results):
        """Returns dict of visualizations."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.show'):
            mock_fig = MagicMock()
            mock_fig.number = 1
            mock_subplots.return_value = (mock_fig, (MagicMock(), MagicMock()))

            result = _create_and_display_visualizations(
                sample_win_rate_results, 100, None
            )

            assert isinstance(result, dict)

    def test_handles_exception(self, sample_win_rate_results, capsys):
        """Handles visualization exception gracefully."""
        with patch(
            'src.features.selection.stability.bootstrap_win_rate_analysis.create_win_rate_visualizations',
            side_effect=Exception("Viz failed")
        ):
            result = _create_and_display_visualizations(
                sample_win_rate_results, 100, None
            )

            assert result == {}
            captured = capsys.readouterr()
            assert 'WARNING' in captured.out


class TestBuildDetailedResults:
    """Tests for _build_detailed_results."""

    def test_returns_dict(self, sample_win_rate_results):
        """Returns dict with all sections."""
        insights = {'top_performer': sample_win_rate_results[0]}
        visualizations = {}

        result = _build_detailed_results(
            sample_win_rate_results, insights, visualizations, 100, 15
        )

        assert 'win_rate_results' in result
        assert 'insights' in result
        assert 'visualizations' in result
        assert 'analysis_metadata' in result

    def test_metadata_content(self, sample_win_rate_results):
        """Metadata contains expected fields."""
        result = _build_detailed_results(
            sample_win_rate_results, {}, {}, 100, 15
        )

        assert result['analysis_metadata']['n_models_analyzed'] == 3
        assert result['analysis_metadata']['n_bootstrap_samples'] == 100
        assert result['analysis_metadata']['max_models_configured'] == 15
        assert result['analysis_metadata']['analysis_complete'] == True
