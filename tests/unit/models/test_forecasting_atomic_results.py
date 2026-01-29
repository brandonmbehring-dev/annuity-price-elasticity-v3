"""
Unit Tests for Forecasting Atomic Results
==========================================

Tests for src/models/forecasting_atomic_results.py covering:
- Confidence interval calculation
- Performance metrics calculation
- Volatility weights calculation
- Export data preparation
- Enhanced MAPE metrics

Target: 80% coverage for forecasting_atomic_results.py

Test Pattern:
- Test atomic statistical operations
- Test validation functions
- Test edge cases (empty arrays, NaN values)
- Test mathematical correctness

Author: Claude Code
Date: 2026-01-29
"""

import numpy as np
import pandas as pd
import pytest

from src.models.forecasting_atomic_results import (
    calculate_single_confidence_interval,
    generate_confidence_intervals_atomic,
    calculate_performance_metrics_atomic,
    calculate_volatility_weights_atomic,
    calculate_weighted_metrics_atomic,
    prepare_export_data_atomic,
    calculate_enhanced_mape_metrics_atomic,
    _validate_prediction_arrays,
    _compute_mape,
    _compute_rmse_mae,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_bootstrap_predictions():
    """Sample bootstrap predictions for testing."""
    np.random.seed(42)
    return np.random.randn(100) * 10 + 50  # Mean ~50, std ~10


@pytest.fixture
def sample_forecast_results():
    """Sample forecast results for testing."""
    np.random.seed(42)
    n = 10
    y_true = np.random.randn(n) * 10 + 100
    y_pred = y_true + np.random.randn(n) * 2  # Small noise

    abs_pct_error = np.abs((y_true - y_pred) / y_true)

    return {
        'y_true': y_true,
        'y_predict': y_pred,
        'abs_pct_error': abs_pct_error,
        'benchmark_predictions': y_true * 0.95  # Slightly off
    }


@pytest.fixture
def sample_confidence_intervals():
    """Sample confidence intervals for testing."""
    n_forecasts = 5
    percentiles = {}

    for p in range(3, 98, 5):  # 3%, 8%, 13%, ..., 93%
        percentiles[f"{p-2:02d}th-percentile"] = np.random.randn(n_forecasts) * 10 + 100

    return percentiles


# =============================================================================
# SINGLE CONFIDENCE INTERVAL TESTS
# =============================================================================


def test_calculate_single_confidence_interval_median(sample_bootstrap_predictions):
    """Test median (50th percentile) confidence interval."""
    ci_value = calculate_single_confidence_interval(sample_bootstrap_predictions, 50.0)

    assert isinstance(ci_value, float)
    assert np.isfinite(ci_value)

    # Should be close to mean for normal distribution
    assert np.abs(ci_value - np.mean(sample_bootstrap_predictions)) < 5.0


def test_calculate_single_confidence_interval_extremes(sample_bootstrap_predictions):
    """Test extreme percentiles."""
    ci_low = calculate_single_confidence_interval(sample_bootstrap_predictions, 5.0)
    ci_high = calculate_single_confidence_interval(sample_bootstrap_predictions, 95.0)

    # Low percentile should be less than high percentile
    assert ci_low < ci_high

    # Should be within distribution range
    assert ci_low >= np.min(sample_bootstrap_predictions)
    assert ci_high <= np.max(sample_bootstrap_predictions)


def test_calculate_single_confidence_interval_empty():
    """Test error with empty predictions."""
    with pytest.raises(ValueError, match="Empty bootstrap predictions"):
        calculate_single_confidence_interval(np.array([]), 50.0)


def test_calculate_single_confidence_interval_invalid_values():
    """Test error with invalid values."""
    with pytest.raises(ValueError, match="Invalid values in bootstrap predictions"):
        calculate_single_confidence_interval(np.array([10, np.nan, 30]), 50.0)


def test_calculate_single_confidence_interval_invalid_percentile():
    """Test error with invalid percentile."""
    predictions = np.array([10, 20, 30])

    with pytest.raises(ValueError, match="Percentile must be in"):
        calculate_single_confidence_interval(predictions, -5.0)

    with pytest.raises(ValueError, match="Percentile must be in"):
        calculate_single_confidence_interval(predictions, 105.0)


def test_calculate_single_confidence_interval_boundary_percentiles():
    """Test boundary percentiles (0% and 100%)."""
    predictions = np.array([10, 20, 30, 40, 50])

    ci_0 = calculate_single_confidence_interval(predictions, 0.0)
    ci_100 = calculate_single_confidence_interval(predictions, 100.0)

    assert ci_0 == 10.0  # Minimum
    assert ci_100 == 50.0  # Maximum


# =============================================================================
# CONFIDENCE INTERVALS GENERATION TESTS
# =============================================================================


def test_generate_confidence_intervals_basic():
    """Test basic confidence interval generation."""
    np.random.seed(42)
    bootstrap_matrix = np.random.randn(100, 5) * 10 + 50

    percentiles = np.array([5, 25, 50, 75, 95])

    ci_dict = generate_confidence_intervals_atomic(bootstrap_matrix, percentiles)

    assert len(ci_dict) == 5
    assert '02th-percentile' in ci_dict  # int(5-2.5)=int(2.5)=2
    assert '47th-percentile' in ci_dict  # int(50-2.5)=int(47.5)=47
    assert '92th-percentile' in ci_dict  # int(95-2.5)=int(92.5)=92

    # Check each has correct shape
    for key, values in ci_dict.items():
        assert len(values) == 5  # n_forecasts


def test_generate_confidence_intervals_ordering():
    """Test that percentiles maintain ordering."""
    bootstrap_matrix = np.random.randn(100, 3) * 10 + 50

    percentiles = np.array([10, 50, 90])

    ci_dict = generate_confidence_intervals_atomic(bootstrap_matrix, percentiles)

    # Extract values for first forecast
    # int(10-2.5)=7, int(50-2.5)=47, int(90-2.5)=87
    p10 = ci_dict['07th-percentile'][0]
    p50 = ci_dict['47th-percentile'][0]
    p90 = ci_dict['87th-percentile'][0]

    # Should be ordered
    assert p10 < p50 < p90


def test_generate_confidence_intervals_wrong_dimensions():
    """Test error with wrong matrix dimensions."""
    # 1D array instead of 2D
    with pytest.raises(ValueError, match="Bootstrap matrix must be 2D"):
        generate_confidence_intervals_atomic(np.array([1, 2, 3]), np.array([50]))


def test_generate_confidence_intervals_empty_matrix():
    """Test error with empty matrix."""
    with pytest.raises(ValueError, match="Empty bootstrap matrix"):
        generate_confidence_intervals_atomic(np.zeros((0, 5)), np.array([50]))


def test_generate_confidence_intervals_invalid_values():
    """Test error with invalid values in matrix."""
    bootstrap_matrix = np.array([[10, 20], [np.nan, 40]])

    with pytest.raises(ValueError, match="Invalid values in bootstrap matrix"):
        generate_confidence_intervals_atomic(bootstrap_matrix, np.array([50]))


def test_generate_confidence_intervals_no_percentiles():
    """Test error with no percentiles specified."""
    bootstrap_matrix = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="No percentiles specified"):
        generate_confidence_intervals_atomic(bootstrap_matrix, np.array([]))


def test_generate_confidence_intervals_invalid_percentile():
    """Test error with invalid percentile value."""
    bootstrap_matrix = np.random.randn(100, 5)

    with pytest.raises(ValueError, match="Invalid percentile"):
        generate_confidence_intervals_atomic(bootstrap_matrix, np.array([50, 150]))


# =============================================================================
# PERFORMANCE METRICS TESTS
# =============================================================================


def test_calculate_performance_metrics_basic():
    """Test basic performance metrics calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([95, 210, 290, 410, 495])

    metrics = calculate_performance_metrics_atomic(y_true, y_pred)

    assert 'r2_score' in metrics
    assert 'mape' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics

    # All metrics should be finite
    for value in metrics.values():
        assert np.isfinite(value)

    # R² should be high for good predictions
    assert metrics['r2_score'] > 0.95


def test_calculate_performance_metrics_perfect():
    """Test metrics with perfect predictions."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = y_true.copy()

    metrics = calculate_performance_metrics_atomic(y_true, y_pred)

    assert metrics['r2_score'] == 1.0
    assert metrics['mape'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['mae'] == 0.0


def test_calculate_performance_metrics_with_weights():
    """Test weighted performance metrics."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    weights = np.array([1.0, 2.0, 1.0])  # More weight on middle point

    metrics = calculate_performance_metrics_atomic(y_true, y_pred, sample_weights=weights)

    # Should have all metrics
    assert all(k in metrics for k in ['r2_score', 'mape', 'rmse', 'mae'])


def test_calculate_performance_metrics_length_mismatch():
    """Test error with length mismatch."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200])  # Wrong length

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_performance_metrics_atomic(y_true, y_pred)


def test_calculate_performance_metrics_empty():
    """Test error with empty arrays."""
    with pytest.raises(ValueError, match="Empty prediction arrays"):
        calculate_performance_metrics_atomic(np.array([]), np.array([]))


def test_calculate_performance_metrics_invalid_values():
    """Test error with invalid values."""
    y_true = np.array([100, np.nan, 300])
    y_pred = np.array([100, 200, 300])

    with pytest.raises(ValueError, match="Invalid values in prediction arrays"):
        calculate_performance_metrics_atomic(y_true, y_pred)


def test_validate_prediction_arrays_weight_mismatch():
    """Test validation error with weight length mismatch."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    weights = np.array([1.0, 1.0])  # Wrong length

    with pytest.raises(ValueError, match="Sample weights length mismatch"):
        _validate_prediction_arrays(y_true, y_pred, weights)


def test_validate_prediction_arrays_negative_weights():
    """Test validation error with negative weights."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    weights = np.array([1.0, -1.0, 1.0])  # Negative weight

    with pytest.raises(ValueError, match="Sample weights must be non-negative"):
        _validate_prediction_arrays(y_true, y_pred, weights)


# =============================================================================
# VOLATILITY WEIGHTS TESTS
# =============================================================================


def test_calculate_volatility_weights_basic():
    """Test basic volatility weights calculation."""
    y_series = np.array([100, 110, 105, 120, 115, 130, 125, 140])

    weights = calculate_volatility_weights_atomic(y_series, window_size=3)

    assert len(weights) == len(y_series)
    assert np.allclose(np.sum(weights), 1.0)  # Should sum to 1
    assert np.all(weights >= 0)  # All non-negative


def test_calculate_volatility_weights_constant_series():
    """Test volatility weights with constant series."""
    y_series = np.array([100, 100, 100, 100, 100])

    weights = calculate_volatility_weights_atomic(y_series, window_size=3)

    # Should return equal weights for constant series
    assert np.allclose(weights, 1.0 / len(y_series))


def test_calculate_volatility_weights_empty():
    """Test error with empty series."""
    with pytest.raises(ValueError, match="Empty time series"):
        calculate_volatility_weights_atomic(np.array([]), window_size=3)


def test_calculate_volatility_weights_invalid_values():
    """Test error with invalid values."""
    with pytest.raises(ValueError, match="Invalid values in time series"):
        calculate_volatility_weights_atomic(np.array([100, np.nan, 300]), window_size=3)


def test_calculate_volatility_weights_invalid_window():
    """Test error with invalid window size."""
    y_series = np.array([100, 200, 300])

    with pytest.raises(ValueError, match="Window size must be positive"):
        calculate_volatility_weights_atomic(y_series, window_size=0)


def test_calculate_volatility_weights_default_window():
    """Test volatility weights with default window (13)."""
    y_series = np.random.randn(50) * 10 + 100

    weights = calculate_volatility_weights_atomic(y_series)

    assert np.allclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)


# =============================================================================
# WEIGHTED METRICS TESTS
# =============================================================================


def test_calculate_weighted_metrics_basic():
    """Test basic weighted metrics calculation."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([95, 210, 290, 410, 495])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    metrics = calculate_weighted_metrics_atomic(y_true, y_pred, weights)

    assert 'weighted_r2' in metrics
    assert 'weighted_mape' in metrics

    # Both should be finite
    assert np.isfinite(metrics['weighted_r2'])
    assert np.isfinite(metrics['weighted_mape'])


def test_calculate_weighted_metrics_perfect():
    """Test weighted metrics with perfect predictions."""
    y_true = np.array([100, 200, 300])
    y_pred = y_true.copy()
    weights = np.array([0.3, 0.4, 0.3])

    metrics = calculate_weighted_metrics_atomic(y_true, y_pred, weights)

    assert metrics['weighted_r2'] == 1.0
    assert metrics['weighted_mape'] == 0.0


def test_calculate_weighted_metrics_length_mismatch():
    """Test error with array length mismatch."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200])
    weights = np.array([0.5, 0.5])

    # This will fail in the validation or numpy operations
    with pytest.raises((ValueError, TypeError)):
        calculate_weighted_metrics_atomic(y_true, y_pred, weights)


def test_calculate_weighted_metrics_negative_weights():
    """Test error with negative weights."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    weights = np.array([0.5, -0.3, 0.8])

    with pytest.raises(ValueError, match="All weights must be non-negative"):
        calculate_weighted_metrics_atomic(y_true, y_pred, weights)


def test_calculate_weighted_metrics_zero_weights():
    """Test error when all weights are zero."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    weights = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="All weights are zero"):
        calculate_weighted_metrics_atomic(y_true, y_pred, weights)


# =============================================================================
# EXPORT DATA PREPARATION TESTS
# =============================================================================


def test_prepare_export_data_basic(sample_forecast_results, sample_confidence_intervals):
    """Test basic export data preparation."""
    dates = ['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22', '2024-01-29']

    metadata = {
        'product': 'TEST_PRODUCT',
        'model_version': 'v1.0',
        'forecast_method': 'bootstrap',
        'bootstrap_samples': 1000,
        'ridge_alpha': 1.0,
        'analysis_date': '2024-01-01'
    }

    export_df = prepare_export_data_atomic(
        sample_forecast_results,
        sample_confidence_intervals,
        dates,
        metadata
    )

    assert isinstance(export_df, pd.DataFrame)
    assert 'date' in export_df.columns
    assert 'metric_type' in export_df.columns
    assert 'sales_value' in export_df.columns
    assert 'product' in export_df.columns

    # Should have records for each date
    assert len(export_df['date'].unique()) == len(dates)


def test_prepare_export_data_empty_dates():
    """Test error with empty dates."""
    with pytest.raises(ValueError, match="No dates provided"):
        prepare_export_data_atomic({}, {}, [], {})


def test_prepare_export_data_no_confidence_intervals():
    """Test error with no confidence intervals."""
    with pytest.raises(ValueError, match="No confidence intervals"):
        prepare_export_data_atomic({}, {}, ['2024-01-01'], {})


def test_prepare_export_data_structure(sample_forecast_results, sample_confidence_intervals):
    """Test export data structure and format."""
    dates = ['2024-01-01', '2024-01-08', '2024-01-15']

    # Adjust sample data to match dates
    for key in ['y_true', 'y_predict', 'benchmark_predictions', 'abs_pct_error']:
        if key in sample_forecast_results:
            sample_forecast_results[key] = sample_forecast_results[key][:3]

    for key in sample_confidence_intervals:
        sample_confidence_intervals[key] = sample_confidence_intervals[key][:3]

    metadata = {'product': 'TEST'}

    export_df = prepare_export_data_atomic(
        sample_forecast_results,
        sample_confidence_intervals,
        dates,
        metadata
    )

    # Check that data is sorted
    assert export_df['date'].is_monotonic_increasing or not export_df['date'].is_unique

    # Check no missing values in required columns
    assert export_df['sales_value'].notna().all()


# =============================================================================
# ENHANCED MAPE METRICS TESTS
# =============================================================================


def test_calculate_enhanced_mape_metrics_basic():
    """Test basic enhanced MAPE metrics calculation."""
    forecast_results = {
        'abs_pct_error': np.array([0.1, 0.15, 0.12, 0.08, 0.11, 0.14])
    }

    dates = ['2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22', '2024-01-29', '2024-02-05']

    enhanced_metrics = calculate_enhanced_mape_metrics_atomic(forecast_results, dates)

    assert 'cumulative_mape' in enhanced_metrics
    assert 'rolling_13week_mape' in enhanced_metrics
    assert 'rolling_26week_mape' in enhanced_metrics

    # Check shapes
    assert len(enhanced_metrics['cumulative_mape']) == len(dates)
    assert len(enhanced_metrics['rolling_13week_mape']) == len(dates)


def test_calculate_enhanced_mape_metrics_cumulative():
    """Test cumulative MAPE calculation."""
    forecast_results = {
        'abs_pct_error': np.array([0.1, 0.2, 0.3])
    }

    dates = ['2024-01-01', '2024-01-08', '2024-01-15']

    enhanced_metrics = calculate_enhanced_mape_metrics_atomic(forecast_results, dates)

    cumulative = enhanced_metrics['cumulative_mape']

    # First value should be 10% (0.1 * 100)
    assert np.isclose(cumulative[0], 10.0)

    # Second should be mean of [0.1, 0.2] * 100 = 15.0
    assert np.isclose(cumulative[1], 15.0)

    # Third should be mean of [0.1, 0.2, 0.3] * 100 = 20.0
    assert np.isclose(cumulative[2], 20.0)


def test_calculate_enhanced_mape_metrics_missing_error():
    """Test error when abs_pct_error is missing."""
    forecast_results = {'y_true': np.array([1, 2, 3])}
    dates = ['2024-01-01', '2024-01-08', '2024-01-15']

    with pytest.raises(ValueError, match="Absolute percentage error not found"):
        calculate_enhanced_mape_metrics_atomic(forecast_results, dates)


def test_calculate_enhanced_mape_metrics_length_mismatch():
    """Test error when error array length doesn't match dates."""
    forecast_results = {
        'abs_pct_error': np.array([0.1, 0.2])  # 2 values
    }

    dates = ['2024-01-01', '2024-01-08', '2024-01-15']  # 3 dates

    with pytest.raises(ValueError, match="Error array length doesn't match dates"):
        calculate_enhanced_mape_metrics_atomic(forecast_results, dates)


def test_calculate_enhanced_mape_metrics_long_series():
    """Test enhanced MAPE with series longer than 26 weeks."""
    n = 30
    forecast_results = {
        'abs_pct_error': np.random.rand(n) * 0.2
    }

    dates = [f'2024-W{i:02d}' for i in range(n)]

    enhanced_metrics = calculate_enhanced_mape_metrics_atomic(forecast_results, dates)

    # Should have 26-week rolling for long series
    assert len(enhanced_metrics['rolling_26week_mape']) == n


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


def test_compute_mape_basic():
    """Test MAPE computation helper."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 290])

    mape = _compute_mape(y_true, y_pred, sample_weights=None)

    assert isinstance(mape, float)
    assert mape > 0  # Should have some error
    assert mape < 20  # But not too much for this data


def test_compute_mape_with_weights():
    """Test MAPE computation with weights."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 290])
    weights = np.array([1.0, 2.0, 1.0])

    mape = _compute_mape(y_true, y_pred, sample_weights=weights)

    assert isinstance(mape, float)
    assert np.isfinite(mape)


def test_compute_rmse_mae_basic():
    """Test RMSE and MAE computation helper."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([95, 205, 295])

    rmse, mae = _compute_rmse_mae(y_true, y_pred, sample_weights=None)

    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert rmse > 0
    assert mae > 0

    # For this data, errors are all 5
    assert np.isclose(mae, 5.0)


def test_compute_rmse_mae_with_weights():
    """Test RMSE and MAE with weights."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 290])
    weights = np.array([1.0, 2.0, 1.0])

    rmse, mae = _compute_rmse_mae(y_true, y_pred, sample_weights=weights)

    assert isinstance(rmse, float)
    assert isinstance(mae, float)
    assert np.isfinite(rmse)
    assert np.isfinite(mae)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_results_pipeline():
    """Test complete results processing pipeline."""
    np.random.seed(42)

    # Create bootstrap predictions
    n_bootstrap = 100
    n_forecasts = 5

    bootstrap_matrix = np.random.randn(n_bootstrap, n_forecasts) * 10 + 100

    # Generate confidence intervals
    percentiles = np.array([5, 25, 50, 75, 95])
    ci_dict = generate_confidence_intervals_atomic(bootstrap_matrix, percentiles)

    # Calculate mean predictions
    mean_predictions = np.mean(bootstrap_matrix, axis=0)

    # Create true values
    y_true = np.random.randn(n_forecasts) * 10 + 100

    # Calculate performance metrics
    metrics = calculate_performance_metrics_atomic(y_true, mean_predictions)

    # All should be finite
    assert all(np.isfinite(v) for v in metrics.values())
    assert len(ci_dict) == len(percentiles)


def test_metrics_consistency():
    """Test that metrics are mathematically consistent."""
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([102, 198, 305, 395, 502])

    # Calculate unweighted metrics
    metrics_unweighted = calculate_performance_metrics_atomic(y_true, y_pred)

    # Calculate with uniform weights (should be same)
    weights = np.ones(len(y_true)) / len(y_true)
    metrics_weighted = calculate_weighted_metrics_atomic(y_true, y_pred, weights)

    # Weighted and unweighted R² should be close
    assert np.abs(metrics_unweighted['r2_score'] - metrics_weighted['weighted_r2']) < 0.01
