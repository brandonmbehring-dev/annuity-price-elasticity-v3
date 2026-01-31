"""
Tests for stability_win_rate module.

Target: 14% → 80%+ coverage
Tests organized by function categories:
- Matrix construction
- Win count calculations
- Results building
- Display functions
- Main analysis function
"""

import numpy as np
import pytest
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.stability.stability_win_rate import (
    # Matrix construction
    _build_bootstrap_aic_matrix,
    _calculate_win_counts,
    # Results construction
    _build_win_rate_results,
    # Display functions
    _print_win_rate_table,
    _print_win_rate_insights,
    # Main function
    calculate_bootstrap_win_rates,
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
            bootstrap_aics=np.random.normal(100, 5, 100).tolist(),
            bootstrap_r2_values=np.random.uniform(0.5, 0.8, 100).tolist(),
            original_aic=98.5,
            original_r2=0.65,
            aic_stability_coefficient=0.05,
            r2_stability_coefficient=0.08,
            confidence_intervals={'aic': {'lower': 95, 'upper': 105}},
            successful_fits=100,
            total_attempts=100,
            stability_assessment='Stable'
        ),
        MockBootstrapResult(
            model_name='Model_2',
            model_features='feature_c + feature_d',
            bootstrap_aics=np.random.normal(105, 8, 100).tolist(),
            bootstrap_r2_values=np.random.uniform(0.4, 0.7, 100).tolist(),
            original_aic=103.2,
            original_r2=0.55,
            aic_stability_coefficient=0.08,
            r2_stability_coefficient=0.12,
            confidence_intervals={'aic': {'lower': 92, 'upper': 118}},
            successful_fits=98,
            total_attempts=100,
            stability_assessment='Moderate'
        ),
        MockBootstrapResult(
            model_name='Model_3',
            model_features='feature_e',
            bootstrap_aics=np.random.normal(110, 12, 100).tolist(),
            bootstrap_r2_values=np.random.uniform(0.3, 0.6, 100).tolist(),
            original_aic=108.7,
            original_r2=0.45,
            aic_stability_coefficient=0.11,
            r2_stability_coefficient=0.18,
            confidence_intervals={'aic': {'lower': 88, 'upper': 132}},
            successful_fits=95,
            total_attempts=100,
            stability_assessment='Unstable'
        ),
    ]


@pytest.fixture
def sample_win_rate_results():
    """Sample win rate results for display testing."""
    return [
        {
            'model': 'Model 1',
            'features': 'feature_a + feature_b',
            'win_rate_pct': 65.0,
            'win_count': 65,
            'original_aic': 98.5,
            'median_bootstrap_aic': 99.2,
        },
        {
            'model': 'Model 2',
            'features': 'feature_c + feature_d',
            'win_rate_pct': 25.0,
            'win_count': 25,
            'original_aic': 103.2,
            'median_bootstrap_aic': 104.1,
        },
        {
            'model': 'Model 3',
            'features': 'feature_e',
            'win_rate_pct': 10.0,
            'win_count': 10,
            'original_aic': 108.7,
            'median_bootstrap_aic': 109.5,
        },
    ]


# =============================================================================
# Matrix Construction Tests
# =============================================================================


class TestBuildBootstrapAicMatrix:
    """Tests for _build_bootstrap_aic_matrix."""

    def test_returns_matrix_and_names(self, sample_bootstrap_results):
        """Returns tuple of matrix and model names."""
        matrix, names = _build_bootstrap_aic_matrix(sample_bootstrap_results)

        assert isinstance(matrix, np.ndarray)
        assert isinstance(names, list)

    def test_matrix_shape_correct(self, sample_bootstrap_results):
        """Matrix has correct shape (samples x models)."""
        matrix, _ = _build_bootstrap_aic_matrix(sample_bootstrap_results)

        # 100 samples, 3 models
        assert matrix.shape == (100, 3)

    def test_model_names_format(self, sample_bootstrap_results):
        """Model names follow 'Model N' format."""
        _, names = _build_bootstrap_aic_matrix(sample_bootstrap_results)

        assert names == ['Model 1', 'Model 2', 'Model 3']

    def test_matrix_contains_aics(self, sample_bootstrap_results):
        """Matrix contains bootstrap AICs from results."""
        matrix, _ = _build_bootstrap_aic_matrix(sample_bootstrap_results)

        # First model's AICs should be in first column
        expected_aics = np.array(sample_bootstrap_results[0].bootstrap_aics)
        np.testing.assert_array_almost_equal(matrix[:, 0], expected_aics)

    def test_single_model(self):
        """Handles single model correctly."""
        single_result = [
            MockBootstrapResult(
                model_name='Only_Model',
                model_features='feature_x',
                bootstrap_aics=[100.0, 101.0, 102.0],
                bootstrap_r2_values=[0.5, 0.6, 0.7],
                original_aic=100.5,
                original_r2=0.6,
                aic_stability_coefficient=0.01,
                r2_stability_coefficient=0.1,
                confidence_intervals={},
                successful_fits=3,
                total_attempts=3,
                stability_assessment='Stable'
            )
        ]

        matrix, names = _build_bootstrap_aic_matrix(single_result)

        assert matrix.shape == (3, 1)
        assert names == ['Model 1']


class TestCalculateWinCounts:
    """Tests for _calculate_win_counts."""

    def test_returns_array(self):
        """Returns numpy array of win counts."""
        matrix = np.array([
            [100, 105, 110],
            [102, 101, 108],
            [98, 103, 107],
        ])

        result = _calculate_win_counts(matrix)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_correct_win_counts(self):
        """Calculates correct win counts (lower AIC wins)."""
        matrix = np.array([
            [100, 105, 110],  # Model 0 wins
            [102, 101, 108],  # Model 1 wins
            [98, 103, 107],   # Model 0 wins
        ])

        result = _calculate_win_counts(matrix)

        assert result[0] == 2  # Model 0: 2 wins
        assert result[1] == 1  # Model 1: 1 win
        assert result[2] == 0  # Model 2: 0 wins

    def test_total_wins_equal_samples(self):
        """Total win counts equals number of samples."""
        matrix = np.random.random((50, 5))

        result = _calculate_win_counts(matrix)

        assert result.sum() == 50

    def test_tie_goes_to_first(self):
        """In ties, first model wins (argmin behavior)."""
        matrix = np.array([
            [100, 100, 110],  # Tie between 0 and 1, 0 wins
        ])

        result = _calculate_win_counts(matrix)

        assert result[0] == 1
        assert result[1] == 0

    def test_single_sample(self):
        """Handles single sample correctly."""
        matrix = np.array([[105, 100, 110]])  # Model 1 wins

        result = _calculate_win_counts(matrix)

        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 0


# =============================================================================
# Results Construction Tests
# =============================================================================


class TestBuildWinRateResults:
    """Tests for _build_win_rate_results."""

    def test_returns_list_of_dicts(self, sample_bootstrap_results):
        """Returns list of dictionaries."""
        model_names = ['Model 1', 'Model 2', 'Model 3']
        win_rates = np.array([50.0, 30.0, 20.0])
        win_counts = np.array([50, 30, 20])

        result = _build_win_rate_results(
            sample_bootstrap_results, model_names, win_rates, win_counts
        )

        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_contains_required_keys(self, sample_bootstrap_results):
        """Each result contains required keys."""
        model_names = ['Model 1', 'Model 2', 'Model 3']
        win_rates = np.array([50.0, 30.0, 20.0])
        win_counts = np.array([50, 30, 20])

        result = _build_win_rate_results(
            sample_bootstrap_results, model_names, win_rates, win_counts
        )

        expected_keys = ['model', 'features', 'win_rate_pct', 'win_count',
                         'original_aic', 'median_bootstrap_aic']
        for r in result:
            for key in expected_keys:
                assert key in r

    def test_sorted_by_win_rate_descending(self, sample_bootstrap_results):
        """Results sorted by win rate (highest first)."""
        model_names = ['Model 1', 'Model 2', 'Model 3']
        win_rates = np.array([20.0, 50.0, 30.0])  # Unsorted
        win_counts = np.array([20, 50, 30])

        result = _build_win_rate_results(
            sample_bootstrap_results, model_names, win_rates, win_counts
        )

        rates = [r['win_rate_pct'] for r in result]
        assert rates == sorted(rates, reverse=True)

    def test_includes_original_features(self, sample_bootstrap_results):
        """Includes original features from bootstrap results."""
        model_names = ['Model 1', 'Model 2', 'Model 3']
        win_rates = np.array([50.0, 30.0, 20.0])
        win_counts = np.array([50, 30, 20])

        result = _build_win_rate_results(
            sample_bootstrap_results, model_names, win_rates, win_counts
        )

        # Check features are preserved (order may change due to sorting)
        all_features = {r['features'] for r in result}
        expected_features = {'feature_a + feature_b', 'feature_c + feature_d', 'feature_e'}
        assert all_features == expected_features


# =============================================================================
# Display Function Tests
# =============================================================================


class TestPrintWinRateTable:
    """Tests for _print_win_rate_table."""

    def test_prints_table_header(self, sample_win_rate_results, capsys):
        """Prints table with header."""
        _print_win_rate_table(sample_win_rate_results, 3, 100)

        captured = capsys.readouterr()
        assert 'Bootstrap Win Rate Analysis' in captured.out
        assert 'Rank' in captured.out
        assert 'Model' in captured.out
        assert 'Win Rate' in captured.out

    def test_shows_sample_count(self, sample_win_rate_results, capsys):
        """Shows number of models and samples."""
        _print_win_rate_table(sample_win_rate_results, 3, 100)

        captured = capsys.readouterr()
        assert '3 Models' in captured.out
        assert '100 samples' in captured.out

    def test_prints_all_models(self, sample_win_rate_results, capsys):
        """Prints data for all models."""
        _print_win_rate_table(sample_win_rate_results, 3, 100)

        captured = capsys.readouterr()
        assert 'Model 1' in captured.out
        assert 'Model 2' in captured.out
        assert 'Model 3' in captured.out

    def test_shows_win_rates(self, sample_win_rate_results, capsys):
        """Shows win rate percentages."""
        _print_win_rate_table(sample_win_rate_results, 3, 100)

        captured = capsys.readouterr()
        # Format produces "65.0     %" due to column widths
        assert '65.0' in captured.out
        assert '25.0' in captured.out
        assert '10.0' in captured.out


class TestPrintWinRateInsights:
    """Tests for _print_win_rate_insights."""

    def test_prints_insights_header(self, sample_win_rate_results, capsys):
        """Prints insights section header."""
        _print_win_rate_insights(sample_win_rate_results, 3)

        captured = capsys.readouterr()
        assert 'WIN RATE INSIGHTS' in captured.out

    def test_identifies_top_performer(self, sample_win_rate_results, capsys):
        """Identifies top performing model."""
        _print_win_rate_insights(sample_win_rate_results, 3)

        captured = capsys.readouterr()
        assert 'Top Performer' in captured.out
        assert 'Model 1' in captured.out
        assert '65.0%' in captured.out

    def test_counts_competitive_models(self, sample_win_rate_results, capsys):
        """Counts models with >10% win rate."""
        _print_win_rate_insights(sample_win_rate_results, 3)

        captured = capsys.readouterr()
        assert 'Competitive Models' in captured.out
        # 2 models have >10% (10.0% is not >10%)
        assert '2 out of 3' in captured.out

    def test_high_confidence_threshold(self, capsys):
        """High confidence when top performer >30%."""
        results = [
            {'model': 'Model 1', 'features': 'a', 'win_rate_pct': 45.0, 'win_count': 45,
             'original_aic': 100, 'median_bootstrap_aic': 101},
            {'model': 'Model 2', 'features': 'b', 'win_rate_pct': 5.0, 'win_count': 5,
             'original_aic': 110, 'median_bootstrap_aic': 111},
        ]

        _print_win_rate_insights(results, 2)

        captured = capsys.readouterr()
        assert 'High' in captured.out

    def test_moderate_confidence_threshold(self, capsys):
        """Moderate confidence when top performer 20-30%."""
        results = [
            {'model': 'Model 1', 'features': 'a', 'win_rate_pct': 25.0, 'win_count': 25,
             'original_aic': 100, 'median_bootstrap_aic': 101},
            {'model': 'Model 2', 'features': 'b', 'win_rate_pct': 20.0, 'win_count': 20,
             'original_aic': 110, 'median_bootstrap_aic': 111},
        ]

        _print_win_rate_insights(results, 2)

        captured = capsys.readouterr()
        assert 'Moderate' in captured.out

    def test_low_confidence_threshold(self, capsys):
        """Low confidence when top performer <20%."""
        results = [
            {'model': 'Model 1', 'features': 'a', 'win_rate_pct': 15.0, 'win_count': 15,
             'original_aic': 100, 'median_bootstrap_aic': 101},
            {'model': 'Model 2', 'features': 'b', 'win_rate_pct': 10.0, 'win_count': 10,
             'original_aic': 110, 'median_bootstrap_aic': 111},
        ]

        _print_win_rate_insights(results, 2)

        captured = capsys.readouterr()
        assert 'Low' in captured.out


# =============================================================================
# Main Analysis Function Tests
# =============================================================================


class TestCalculateBootstrapWinRates:
    """Tests for calculate_bootstrap_win_rates."""

    def test_raises_with_empty_results(self):
        """Raises ValueError when no results provided."""
        with pytest.raises(ValueError, match="No bootstrap results available"):
            calculate_bootstrap_win_rates([])

    def test_returns_sorted_results(self, sample_bootstrap_results, capsys):
        """Returns results sorted by win rate."""
        result = calculate_bootstrap_win_rates(sample_bootstrap_results)

        rates = [r['win_rate_pct'] for r in result]
        assert rates == sorted(rates, reverse=True)

    def test_returns_all_models(self, sample_bootstrap_results, capsys):
        """Returns results for all input models."""
        result = calculate_bootstrap_win_rates(sample_bootstrap_results)

        assert len(result) == len(sample_bootstrap_results)

    def test_prints_analysis_header(self, sample_bootstrap_results, capsys):
        """Prints analysis header."""
        calculate_bootstrap_win_rates(sample_bootstrap_results)

        captured = capsys.readouterr()
        assert 'BOOTSTRAP WIN RATE ANALYSIS' in captured.out

    def test_results_contain_required_keys(self, sample_bootstrap_results, capsys):
        """Each result has required keys."""
        result = calculate_bootstrap_win_rates(sample_bootstrap_results)

        required_keys = ['model', 'features', 'win_rate_pct', 'win_count']
        for r in result:
            for key in required_keys:
                assert key in r

    def test_win_rates_sum_to_100(self, sample_bootstrap_results, capsys):
        """Win rates sum to 100%."""
        result = calculate_bootstrap_win_rates(sample_bootstrap_results)

        total = sum(r['win_rate_pct'] for r in result)
        assert abs(total - 100.0) < 0.01

    def test_win_counts_sum_to_samples(self, sample_bootstrap_results, capsys):
        """Win counts sum to number of samples."""
        result = calculate_bootstrap_win_rates(sample_bootstrap_results)

        total_wins = sum(r['win_count'] for r in result)
        n_samples = len(sample_bootstrap_results[0].bootstrap_aics)
        assert total_wins == n_samples
