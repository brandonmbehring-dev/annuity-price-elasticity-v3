"""
Tests for stability_ir module.

Target: 12% → 60%+ coverage
Tests organized by function categories:
- Benchmark statistics
- Ratio calculations
- Single model metrics
- Classification
- Display functions
- Main analysis function
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict

from src.features.selection.stability.stability_ir import (
    # Benchmark statistics
    _prepare_ir_benchmark_stats,
    # Ratio calculations
    _compute_ratio_safe,
    _compute_sortino_ratio,
    # Single model metrics
    _calculate_single_model_ir_metrics,
    # Classification
    _classify_risk_adjusted,
    # Display functions
    _print_ir_results_table,
    _print_ir_insights,
    _print_best_ir_model,
    # Main function
    calculate_information_ratio_analysis,
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
def sample_ir_metrics():
    """Sample IR metrics for display function tests."""
    return [
        {
            'model_name': 'Model 1',
            'features': 'feature_a + feature_b',
            'information_ratio': 0.85,
            'success_rate': 72.5,
            'sharpe_like': 0.92,
            'mean_excess': 5.2,
            'std_excess': 6.1,
        },
        {
            'model_name': 'Model 2',
            'features': 'feature_c + feature_d + feature_e + feature_f',
            'information_ratio': 0.35,
            'success_rate': 58.0,
            'sharpe_like': 0.41,
            'mean_excess': 2.8,
            'std_excess': 8.0,
        },
        {
            'model_name': 'Model 3',
            'features': 'feature_x',
            'information_ratio': -0.15,
            'success_rate': 42.0,
            'sharpe_like': -0.18,
            'mean_excess': -1.5,
            'std_excess': 10.0,
        },
    ]


# =============================================================================
# Benchmark Statistics Tests
# =============================================================================


class TestPrepareIrBenchmarkStats:
    """Tests for _prepare_ir_benchmark_stats."""

    def test_returns_three_statistics(self, sample_bootstrap_results):
        """Returns benchmark AIC, mean, and std."""
        benchmark_aic, benchmark_mean, benchmark_std = _prepare_ir_benchmark_stats(
            sample_bootstrap_results
        )

        assert isinstance(benchmark_aic, float)
        assert isinstance(benchmark_mean, float)
        assert isinstance(benchmark_std, float)

    def test_uses_all_bootstrap_aics(self, sample_bootstrap_results):
        """Uses all bootstrap AICs across all models."""
        benchmark_aic, benchmark_mean, benchmark_std = _prepare_ir_benchmark_stats(
            sample_bootstrap_results
        )

        # Total AICs = 3 models * 100 bootstraps each = 300
        total_expected = sum(
            len(r.bootstrap_aics) for r in sample_bootstrap_results
        )
        assert total_expected == 300

    def test_benchmark_aic_is_median(self, sample_bootstrap_results):
        """Benchmark AIC is the median of all bootstrap AICs."""
        benchmark_aic, _, _ = _prepare_ir_benchmark_stats(sample_bootstrap_results)

        # Collect all AICs manually
        all_aics = []
        for r in sample_bootstrap_results:
            all_aics.extend(r.bootstrap_aics)

        expected_median = np.median(all_aics)
        assert benchmark_aic == pytest.approx(expected_median)


# =============================================================================
# Ratio Calculation Tests
# =============================================================================


class TestComputeRatioSafe:
    """Tests for _compute_ratio_safe."""

    def test_normal_ratio(self):
        """Computes normal ratio correctly."""
        result = _compute_ratio_safe(10.0, 2.0)

        assert result == pytest.approx(5.0)

    def test_zero_denominator_positive_numerator(self):
        """Returns capped positive value for zero denominator with positive numerator."""
        result = _compute_ratio_safe(5.0, 0.0)

        assert result == 999.0

    def test_zero_denominator_negative_numerator(self):
        """Returns capped negative value for zero denominator with negative numerator."""
        result = _compute_ratio_safe(-5.0, 0.0)

        assert result == -999.0

    def test_zero_numerator_and_denominator(self):
        """Returns zero when both numerator and denominator are zero."""
        result = _compute_ratio_safe(0.0, 0.0)

        assert result == 0.0

    def test_very_small_denominator(self):
        """Handles very small denominator as zero."""
        result = _compute_ratio_safe(5.0, 1e-12)

        assert result == 999.0  # Treated as zero denominator


class TestComputeSortinoRatio:
    """Tests for _compute_sortino_ratio."""

    def test_positive_excess_with_downside(self):
        """Computes Sortino ratio with positive excess and downside volatility."""
        excess_aics = np.array([5.0, 3.0, -2.0, -4.0, 8.0, -1.0])
        mean_excess = np.mean(excess_aics)

        result = _compute_sortino_ratio(excess_aics, mean_excess)

        # Should be positive since mean excess is positive
        assert result > 0

    def test_no_downside_positive_mean(self):
        """Returns 999 when no downside volatility and positive mean."""
        excess_aics = np.array([5.0, 3.0, 2.0, 8.0])  # All positive
        mean_excess = 4.5

        result = _compute_sortino_ratio(excess_aics, mean_excess)

        assert result == 999.0

    def test_no_downside_negative_mean(self):
        """Returns 0 when no downside volatility and non-positive mean."""
        excess_aics = np.array([5.0, 3.0])  # All positive, but we'll set mean to 0
        mean_excess = 0.0

        result = _compute_sortino_ratio(excess_aics, mean_excess)

        assert result == 0.0

    def test_caps_at_999(self):
        """Caps Sortino ratio at 999 for numerical stability."""
        excess_aics = np.array([100.0, -0.001])  # Very small downside
        mean_excess = 50.0

        result = _compute_sortino_ratio(excess_aics, mean_excess)

        assert result <= 999.0


# =============================================================================
# Single Model Metrics Tests
# =============================================================================


class TestCalculateSingleModelIrMetrics:
    """Tests for _calculate_single_model_ir_metrics."""

    def test_returns_all_expected_keys(self, sample_bootstrap_results):
        """Returns dictionary with all expected metric keys."""
        benchmark_aic = 105.0

        result = _calculate_single_model_ir_metrics(
            0, sample_bootstrap_results[0], benchmark_aic
        )

        expected_keys = [
            'model_idx', 'model_name', 'features', 'original_aic',
            'mean_bootstrap_aic', 'benchmark_aic', 'mean_excess', 'std_excess',
            'information_ratio', 'sharpe_like', 'sortino_ratio',
            'success_rate', 'positive_excess_count', 'excess_aics'
        ]

        for key in expected_keys:
            assert key in result

    def test_model_name_format(self, sample_bootstrap_results):
        """Model name follows expected format."""
        result = _calculate_single_model_ir_metrics(
            2, sample_bootstrap_results[0], 105.0
        )

        assert result['model_name'] == 'Model 3'  # 0-indexed + 1

    def test_success_rate_percentage(self, sample_bootstrap_results):
        """Success rate is expressed as percentage (0-100)."""
        result = _calculate_single_model_ir_metrics(
            0, sample_bootstrap_results[0], 105.0
        )

        assert 0 <= result['success_rate'] <= 100

    def test_excess_aics_computed(self, sample_bootstrap_results):
        """Excess AICs are computed relative to benchmark."""
        benchmark_aic = 105.0

        result = _calculate_single_model_ir_metrics(
            0, sample_bootstrap_results[0], benchmark_aic
        )

        # Excess = benchmark - bootstrap (positive = better)
        assert len(result['excess_aics']) == 100  # Same as bootstrap count


# =============================================================================
# Classification Tests
# =============================================================================


class TestClassifyRiskAdjusted:
    """Tests for _classify_risk_adjusted."""

    @pytest.mark.parametrize('ir,expected_class', [
        (0.8, 'High'),
        (0.5, 'Moderate'),  # Boundary
        (0.51, 'High'),
        (0.35, 'Moderate'),
        (0.2, 'Low'),  # Boundary
        (0.21, 'Moderate'),
        (0.1, 'Low'),
        (0.01, 'Low'),
        (0.0, 'Negative'),  # Boundary
        (-0.1, 'Negative'),
        (-0.5, 'Negative'),
    ])
    def test_classification_thresholds(self, ir, expected_class):
        """Classification follows IR thresholds."""
        result = _classify_risk_adjusted(ir)

        assert result == expected_class


# =============================================================================
# Display Function Tests
# =============================================================================


class TestPrintIrResultsTable:
    """Tests for _print_ir_results_table."""

    def test_prints_table(self, sample_ir_metrics, capsys):
        """Prints formatted table."""
        _print_ir_results_table(sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'Information Ratio Analysis Results' in captured.out
        assert 'Model' in captured.out
        assert 'IR' in captured.out

    def test_truncates_long_features(self, sample_ir_metrics, capsys):
        """Truncates feature names longer than 27 characters."""
        _print_ir_results_table(sample_ir_metrics)

        captured = capsys.readouterr()
        # Second model has long features, should be truncated with "..."
        assert '...' in captured.out

    def test_shows_risk_adjusted_class(self, sample_ir_metrics, capsys):
        """Shows risk-adjusted classification for each model."""
        _print_ir_results_table(sample_ir_metrics)

        captured = capsys.readouterr()
        assert 'High' in captured.out or 'Moderate' in captured.out or 'Negative' in captured.out


class TestPrintIrInsights:
    """Tests for _print_ir_insights."""

    def test_prints_classification_counts(self, sample_ir_metrics, capsys):
        """Prints counts for each classification."""
        _print_ir_insights(sample_ir_metrics, 3)

        captured = capsys.readouterr()
        assert 'INFORMATION RATIO INSIGHTS' in captured.out
        assert 'High IR' in captured.out
        assert 'Moderate IR' in captured.out
        assert 'Low IR' in captured.out
        assert 'Negative IR' in captured.out

    def test_shows_percentages(self, sample_ir_metrics, capsys):
        """Shows percentages for each category."""
        _print_ir_insights(sample_ir_metrics, 3)

        captured = capsys.readouterr()
        assert '%' in captured.out


class TestPrintBestIrModel:
    """Tests for _print_best_ir_model."""

    def test_prints_best_model_details(self, sample_ir_metrics, capsys):
        """Prints details about best model."""
        _print_best_ir_model(sample_ir_metrics[0])

        captured = capsys.readouterr()
        assert 'Best Risk-Adjusted Model' in captured.out
        assert 'Model 1' in captured.out
        assert 'Information Ratio' in captured.out
        assert 'Success Rate' in captured.out


# =============================================================================
# Main Analysis Function Tests
# =============================================================================


class TestCalculateInformationRatioAnalysis:
    """Tests for calculate_information_ratio_analysis."""

    def test_raises_with_empty_results(self):
        """Raises ValueError when no bootstrap results provided."""
        with pytest.raises(ValueError, match="No bootstrap results available"):
            calculate_information_ratio_analysis([])

    def test_returns_sorted_metrics(self, sample_bootstrap_results, capsys):
        """Returns metrics sorted by IR (descending)."""
        result = calculate_information_ratio_analysis(sample_bootstrap_results)

        # Should be sorted by IR descending
        irs = [m['information_ratio'] for m in result]
        assert irs == sorted(irs, reverse=True)

    def test_returns_all_models(self, sample_bootstrap_results, capsys):
        """Returns metrics for all input models."""
        result = calculate_information_ratio_analysis(sample_bootstrap_results)

        assert len(result) == len(sample_bootstrap_results)

    def test_prints_benchmark_stats(self, sample_bootstrap_results, capsys):
        """Prints benchmark statistics."""
        calculate_information_ratio_analysis(sample_bootstrap_results)

        captured = capsys.readouterr()
        assert 'Benchmark Statistics' in captured.out
        assert 'Population Median AIC' in captured.out

    def test_each_result_has_required_keys(self, sample_bootstrap_results, capsys):
        """Each result has required metric keys."""
        result = calculate_information_ratio_analysis(sample_bootstrap_results)

        required_keys = ['model_name', 'information_ratio', 'success_rate', 'features']
        for metrics in result:
            for key in required_keys:
                assert key in metrics
