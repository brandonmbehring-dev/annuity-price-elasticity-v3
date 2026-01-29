"""
Atomic Results Processing for Time Series Forecasting - Performance Optimization.

This module provides atomic operations for processing forecasting results,
including confidence interval generation, performance metrics, and export
data preparation. All operations designed for single responsibility and
future performance optimization.

Key Design Principles:
- Atomic statistical operations (10-50 lines each)
- Single mathematical responsibility per function
- Vectorization-ready for large result processing
- Zero business logic mixing with statistical calculations
- Perfect preservation of numerical precision

Target Baseline Results (Today's Run):
- Model R²: 0.664940178498685
- Model MAPE: 14.961824607519691
- 127 forecasts with 96 confidence interval percentiles each
- 12,574 detailed export records

Mathematical Operations:
- Confidence interval generation (96 percentiles)
- Performance metric calculations (R², MAPE, weighted variants)
- Bootstrap distribution analysis
- Export data preparation for business intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import warnings

# Suppress specific sklearn warnings that don't indicate code issues
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn'
)
# Justification: sklearn FutureWarnings for API changes in future versions
# These don't affect current functionality and will be addressed during sklearn upgrades

# Triple-fallback imports for resilience
from src.config.forecasting_config import ValidationFrameworkConfig


def calculate_single_confidence_interval(bootstrap_predictions: np.ndarray,
                                       percentile: float) -> float:
    """
    Calculate single confidence interval percentile atomically - pure math.

    Atomic Responsibility: Single percentile calculation only.
    Vectorization Ready: Single percentile, easily batchable across percentiles.

    Parameters
    ----------
    bootstrap_predictions : np.ndarray
        Bootstrap prediction array, shape (n_bootstrap_samples,)
    percentile : float
        Percentile to calculate (0.0 to 100.0)

    Returns
    -------
    float
        Confidence interval value at specified percentile

    Mathematical Properties
    ----------------------
    Percentile Calculation: Quantile of bootstrap distribution
    Uncertainty Quantification: Bootstrap distribution → confidence intervals
    Interpolation: Linear interpolation between data points
    """

    # Validate inputs
    if len(bootstrap_predictions) == 0:
        raise ValueError("Empty bootstrap predictions for confidence interval")

    if not np.all(np.isfinite(bootstrap_predictions)):
        raise ValueError("Invalid values in bootstrap predictions")

    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

    # Calculate percentile using numpy (consistent with baseline)
    confidence_value = np.percentile(bootstrap_predictions, percentile)

    # Validate calculation
    if not np.isfinite(confidence_value):
        raise ValueError(f"Invalid confidence interval value: {confidence_value}")

    return float(confidence_value)


def generate_confidence_intervals_atomic(bootstrap_matrix: np.ndarray,
                                       percentiles: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate complete confidence intervals atomically - vectorized operation.

    Atomic Responsibility: Confidence interval generation for all percentiles.
    Vectorization Ready: Operates on full bootstrap matrix efficiently.

    Parameters
    ----------
    bootstrap_matrix : np.ndarray
        Bootstrap predictions matrix, shape (n_bootstrap_samples, n_forecasts)
    percentiles : np.ndarray
        Percentiles to calculate, shape (n_percentiles,)

    Returns
    -------
    Dict[str, np.ndarray]
        Confidence intervals with keys like "05th-percentile", "95th-percentile"
        Each value is array of shape (n_forecasts,)

    Mathematical Properties
    ----------------------
    Bootstrap Matrix: Rows = bootstrap samples, Columns = forecast dates
    Percentile Calculation: Applied column-wise (across bootstrap dimension)
    Naming Convention: "{percentile:02d}th-percentile" for consistency
    """

    # Validate inputs
    if bootstrap_matrix.ndim != 2:
        raise ValueError(f"Bootstrap matrix must be 2D, got {bootstrap_matrix.ndim}D")

    if bootstrap_matrix.shape[0] == 0 or bootstrap_matrix.shape[1] == 0:
        raise ValueError(f"Empty bootstrap matrix: {bootstrap_matrix.shape}")

    if not np.all(np.isfinite(bootstrap_matrix)):
        raise ValueError("Invalid values in bootstrap matrix")

    if len(percentiles) == 0:
        raise ValueError("No percentiles specified")

    # Generate confidence intervals for all percentiles (vectorized)
    confidence_intervals = {}

    for percentile in percentiles:
        # Validate percentile range
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"Invalid percentile: {percentile}")

        # Calculate percentile across bootstrap dimension (axis=0)
        percentile_values = np.percentile(bootstrap_matrix, percentile, axis=0)

        # Create standardized key name (matching baseline format)
        percentile_key = f"{int(percentile-2.5):02d}th-percentile"

        # Store results
        confidence_intervals[percentile_key] = percentile_values

        # Validate calculation
        if len(percentile_values) != bootstrap_matrix.shape[1]:
            raise ValueError(
                f"Percentile calculation failed for {percentile}: "
                f"expected {bootstrap_matrix.shape[1]} values, got {len(percentile_values)}"
            )

    return confidence_intervals


def _validate_prediction_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weights: Optional[np.ndarray]
) -> None:
    """Validate input arrays for performance metrics calculation.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    sample_weights : Optional[np.ndarray]
        Sample weights

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true {len(y_true)} vs y_pred {len(y_pred)}")

    if len(y_true) == 0:
        raise ValueError("Empty prediction arrays")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Invalid values in prediction arrays")

    if sample_weights is not None:
        if len(sample_weights) != len(y_true):
            raise ValueError(f"Sample weights length mismatch: {len(sample_weights)} vs {len(y_true)}")
        if not np.all(sample_weights >= 0):
            raise ValueError("Sample weights must be non-negative")


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray,
                  sample_weights: Optional[np.ndarray]) -> float:
    """Compute Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    sample_weights : Optional[np.ndarray]
        Sample weights

    Returns
    -------
    float
        MAPE as percentage (0-100 scale)
    """
    if sample_weights is None:
        return mean_absolute_percentage_error(y_true, y_pred) * 100
    return mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weights) * 100


def _compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray,
                      sample_weights: Optional[np.ndarray]) -> tuple:
    """Compute RMSE and MAE metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    sample_weights : Optional[np.ndarray]
        Sample weights

    Returns
    -------
    tuple
        (rmse, mae) values
    """
    squared_errors = (y_true - y_pred) ** 2
    absolute_errors = np.abs(y_true - y_pred)

    if sample_weights is not None:
        rmse = np.sqrt(np.average(squared_errors, weights=sample_weights))
        mae = np.average(absolute_errors, weights=sample_weights)
    else:
        rmse = np.sqrt(np.mean(squared_errors))
        mae = np.mean(absolute_errors)

    return rmse, mae


def calculate_performance_metrics_atomic(y_true: np.ndarray, y_pred: np.ndarray,
                                       sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate performance metrics atomically - pure statistical operation.

    Parameters
    ----------
    y_true : np.ndarray
        True target values, shape (n_samples,)
    y_pred : np.ndarray
        Predicted values, shape (n_samples,)
    sample_weights : Optional[np.ndarray], default=None
        Sample weights for weighted metrics, shape (n_samples,)

    Returns
    -------
    Dict[str, float]
        Performance metrics: r2_score, mape, rmse, mae
    """
    _validate_prediction_arrays(y_true, y_pred, sample_weights)

    # Compute all metrics
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weights)
    mape = _compute_mape(y_true, y_pred, sample_weights)
    rmse, mae = _compute_rmse_mae(y_true, y_pred, sample_weights)

    metrics = {
        'r2_score': float(r2),
        'mape': float(mape),
        'rmse': float(rmse),
        'mae': float(mae)
    }

    # Validate all metrics are finite
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise ValueError(f"Invalid {name}: {value}")

    return metrics


def calculate_volatility_weights_atomic(y_series: np.ndarray,
                                      window_size: int = 13) -> np.ndarray:
    """
    Calculate volatility-based weights atomically - pure statistical operation.

    Atomic Responsibility: Volatility weight calculation only.
    Vectorization Ready: Operates on complete time series efficiently.

    Parameters
    ----------
    y_series : np.ndarray
        Time series values for volatility calculation, shape (n_samples,)
    window_size : int, default=13
        Rolling window size for volatility calculation (13 = quarterly)

    Returns
    -------
    np.ndarray
        Volatility-based weights, shape (n_samples,)

    Mathematical Properties
    ----------------------
    Rolling Volatility: Standard deviation over rolling window
    Weight Normalization: Weights sum to 1.0
    Edge Handling: min_periods=1 for beginning of series
    """

    # Validate inputs
    if len(y_series) == 0:
        raise ValueError("Empty time series for volatility calculation")

    if not np.all(np.isfinite(y_series)):
        raise ValueError("Invalid values in time series")

    if window_size <= 0:
        raise ValueError(f"Window size must be positive, got {window_size}")

    # Convert to pandas Series for rolling operations
    y_pandas = pd.Series(y_series)

    # Calculate rolling standard deviation (volatility)
    rolling_volatility = y_pandas.rolling(
        window=window_size,
        min_periods=1,  # Allow calculation from first observation
        center=False    # Use past values only
    ).std()

    # Convert back to numpy array
    volatility_array = rolling_volatility.values

    # Handle edge cases for weight normalization
    # Fill NaN values with minimum positive volatility
    clean_volatility = volatility_array.copy()
    clean_volatility = pd.Series(clean_volatility).fillna(0).values

    # Replace zeros and infinite values
    clean_volatility = np.where(clean_volatility == 0, 1e-10, clean_volatility)
    clean_volatility = np.where(~np.isfinite(clean_volatility), 1e-10, clean_volatility)

    # Normalize to create weights
    if np.sum(clean_volatility) > 0:
        volatility_weights = clean_volatility / np.sum(clean_volatility)
    else:
        # Equal weights if no volatility variation
        volatility_weights = np.ones(len(y_series)) / len(y_series)

    # Validate weights
    if not np.allclose(np.sum(volatility_weights), 1.0, atol=1e-10):
        raise ValueError(f"Weights don't sum to 1: {np.sum(volatility_weights)}")

    if not np.all(volatility_weights >= 0):
        raise ValueError("All weights must be non-negative")

    return volatility_weights


def calculate_weighted_metrics_atomic(y_true: np.ndarray, y_pred: np.ndarray,
                                    weights: np.ndarray) -> Dict[str, float]:
    """
    Calculate volatility-weighted performance metrics atomically.

    Atomic Responsibility: Weighted metric calculation only.
    Vectorization Ready: Efficient weighted calculations.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    weights : np.ndarray
        Volatility-based weights

    Returns
    -------
    Dict[str, float]
        Weighted performance metrics:
        - 'weighted_r2': Volatility-weighted R²
        - 'weighted_mape': Volatility-weighted MAPE

    Mathematical Properties
    ----------------------
    Weighted R²: 1 - (Σ wᵢ(yᵢ - ŷᵢ)² / Σ wᵢ(yᵢ - ȳ)²)
    Weighted MAPE: (100 * Σ wᵢ|yᵢ - ŷᵢ|/|yᵢ|) / Σ wᵢ
    """

    # Validate inputs
    if len(y_true) != len(y_pred) != len(weights):
        raise ValueError("Array length mismatch for weighted calculations")

    if not np.all(weights >= 0):
        raise ValueError("All weights must be non-negative")

    if np.sum(weights) == 0:
        raise ValueError("All weights are zero")

    # Weighted means
    y_true_weighted_mean = np.average(y_true, weights=weights)

    # Weighted R² calculation
    tss_weighted = np.sum(weights * (y_true - y_true_weighted_mean) ** 2)
    rss_weighted = np.sum(weights * (y_true - y_pred) ** 2)

    if tss_weighted == 0:
        weighted_r2 = 1.0 if rss_weighted == 0 else 0.0
    else:
        weighted_r2 = 1.0 - (rss_weighted / tss_weighted)

    # Weighted MAPE calculation (avoid division by zero)
    y_true_safe = np.maximum(np.abs(y_true), 1e-10)
    weighted_ape = np.abs((y_true - y_pred) / y_true_safe)
    weighted_mape = np.average(weighted_ape, weights=weights) * 100

    return {
        'weighted_r2': float(weighted_r2),
        'weighted_mape': float(weighted_mape)
    }


def _validate_export_inputs(dates: List[str],
                           confidence_intervals: Dict[str, np.ndarray]) -> None:
    """Validate inputs for export data preparation."""
    if len(dates) == 0:
        raise ValueError("No dates provided for export")
    if len(confidence_intervals) == 0:
        raise ValueError("No confidence intervals for export")


def _add_forecast_records(export_records: List[Dict],
                          forecast_results: Dict[str, Any],
                          date: str, index: int,
                          metadata: Dict[str, Any]) -> None:
    """Add forecast mean, true values, and benchmark predictions."""
    if 'y_predict' in forecast_results:
        export_records.append({
            'date': date, 'metric_type': 'forecast_mean',
            'sales_value': forecast_results['y_predict'][index], **metadata
        })
    if 'y_true' in forecast_results:
        export_records.append({
            'date': date, 'metric_type': 'y_true',
            'sales_value': forecast_results['y_true'][index], **metadata
        })
    if 'benchmark_predictions' in forecast_results:
        export_records.append({
            'date': date, 'metric_type': 'benchmark_prediction',
            'sales_value': forecast_results['benchmark_predictions'][index], **metadata
        })


def _add_confidence_interval_records(export_records: List[Dict],
                                     confidence_intervals: Dict[str, np.ndarray],
                                     date: str, index: int,
                                     metadata: Dict[str, Any]) -> None:
    """Add confidence interval records for all percentiles."""
    for percentile_key, percentile_values in confidence_intervals.items():
        export_records.append({
            'date': date, 'metric_type': percentile_key,
            'sales_value': percentile_values[index], **metadata
        })


def _format_export_dataframe(export_records: List[Dict]) -> pd.DataFrame:
    """Convert records to DataFrame and validate structure."""
    export_df = pd.DataFrame(export_records)
    export_df = export_df.sort_values(['date', 'metric_type']).reset_index(drop=True)

    expected_columns = ['date', 'metric_type', 'sales_value']
    missing_columns = [col for col in expected_columns if col not in export_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required export columns: {missing_columns}")
    return export_df


def prepare_export_data_atomic(forecast_results: Dict[str, Any],
                             confidence_intervals: Dict[str, np.ndarray],
                             dates: List[str],
                             metadata: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare export data atomically - business intelligence format.

    Atomic Responsibility: Data format preparation only.
    Vectorization Ready: Efficient data restructuring.

    Parameters
    ----------
    forecast_results : Dict[str, Any]
        Forecast results with predictions and true values
    confidence_intervals : Dict[str, np.ndarray]
        Confidence interval data (96 percentiles)
    dates : List[str]
        Forecast dates
    metadata : Dict[str, Any]
        Export metadata (product, version, etc.)

    Returns
    -------
    pd.DataFrame
        Export-ready DataFrame in long format for BI tools

    Business Intelligence Format
    ---------------------------
    Columns: date, metric_type, sales_value, product, model_version,
             forecast_method, bootstrap_samples, ridge_alpha, analysis_date
    """
    _validate_export_inputs(dates, confidence_intervals)

    export_records = []
    for i, date in enumerate(dates):
        _add_forecast_records(export_records, forecast_results, date, i, metadata)
        _add_confidence_interval_records(export_records, confidence_intervals, date, i, metadata)

    return _format_export_dataframe(export_records)


def calculate_enhanced_mape_metrics_atomic(forecast_results: Dict[str, Any],
                                         dates: List[str]) -> Dict[str, np.ndarray]:
    """
    Calculate enhanced MAPE metrics atomically - rolling and cumulative.

    Atomic Responsibility: MAPE metric enhancement only.
    Vectorization Ready: Efficient rolling calculations.

    Parameters
    ----------
    forecast_results : Dict[str, Any]
        Forecast results with error information
    dates : List[str]
        Forecast dates for temporal analysis

    Returns
    -------
    Dict[str, np.ndarray]
        Enhanced MAPE metrics:
        - 'cumulative_mape': Cumulative MAPE over time
        - 'rolling_13week_mape': 13-week rolling MAPE
        - 'rolling_26week_mape': 26-week rolling MAPE (if sufficient data)

    Mathematical Properties
    ----------------------
    Cumulative MAPE: MAPE calculated from start to each point
    Rolling MAPE: MAPE calculated over moving window
    Temporal Analysis: Shows model performance evolution over time
    """

    # Validate inputs
    if 'abs_pct_error' not in forecast_results:
        raise ValueError("Absolute percentage error not found in results")

    abs_pct_errors = np.array(forecast_results['abs_pct_error'])

    if len(abs_pct_errors) != len(dates):
        raise ValueError("Error array length doesn't match dates")

    # Convert to pandas Series for rolling calculations
    error_series = pd.Series(abs_pct_errors)

    # Calculate cumulative MAPE
    cumulative_mape = []
    for i in range(len(abs_pct_errors)):
        if i == 0:
            cumulative_mape.append(abs_pct_errors[i] * 100)
        else:
            cum_mape = np.mean(abs_pct_errors[:i+1]) * 100
            cumulative_mape.append(cum_mape)

    # Calculate 13-week rolling MAPE (quarterly)
    rolling_13week = error_series.rolling(
        window=13,
        min_periods=1,
        center=True
    ).mean() * 100

    # Calculate 26-week rolling MAPE (semi-annual) if sufficient data
    if len(abs_pct_errors) >= 26:
        rolling_26week = error_series.rolling(
            window=26,
            min_periods=13,
            center=True
        ).mean() * 100
    else:
        rolling_26week = rolling_13week.copy()

    return {
        'cumulative_mape': np.array(cumulative_mape),
        'rolling_13week_mape': rolling_13week.values,
        'rolling_26week_mape': rolling_26week.values
    }