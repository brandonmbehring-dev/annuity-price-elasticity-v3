"""
Temporal Validation Engine for Feature Selection.

This module provides statistically rigorous temporal validation for time series
feature selection, addressing the critical lack of out-of-sample validation
identified in the mathematical analysis report.

Key Functions:
- create_temporal_splits: Time-aware train/test splitting
- validate_temporal_structure: Temporal integrity checks
- evaluate_out_of_sample_performance: Generalization assessment
- calculate_generalization_metrics: Performance gap analysis

Critical Statistical Issues Addressed:
- Issue #2: Complete absence of out-of-sample validation (CRITICAL)
- Provides actual evidence of model generalization
- Prevents overfitting through proper temporal validation
- Enables production readiness assessment

Design Principles:
- Time-aware splitting (no data leakage)
- Configurable split ratios with business calendar awareness
- Comprehensive performance tracking
- Integration with existing pipeline architecture

Mathematical Foundation:
- Temporal split: Training set [1, t-h], Test set [t-h+1, T]
- No future information leakage in model selection
- Proper out-of-sample performance evaluation
- Generalization gap assessment: |R²_train - R²_test|
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """
    Container for temporal split results.

    Attributes
    ----------
    train_data : pd.DataFrame
        Training dataset (time periods 1 to t-h)
    test_data : pd.DataFrame
        Test dataset (time periods t-h+1 to T)
    train_period : Tuple[str, str]
        Training period (start_date, end_date)
    test_period : Tuple[str, str]
        Test period (start_date, end_date)
    split_ratio : float
        Actual split ratio achieved
    n_train_obs : int
        Number of training observations
    n_test_obs : int
        Number of test observations
    validation_checks : Dict[str, bool]
        Temporal integrity validation results
    """
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    split_ratio: float
    n_train_obs: int
    n_test_obs: int
    validation_checks: Dict[str, bool]


def _validate_data_size_for_split(data: pd.DataFrame,
                                  ensure_minimum_test_size: int) -> None:
    """
    Validate dataset has sufficient observations for temporal split.

    Raises ValueError if insufficient data for meaningful train/test split.
    """
    if len(data) < ensure_minimum_test_size + 10:
        raise ValueError(
            f"CRITICAL: Insufficient data for temporal split. "
            f"Need at least {ensure_minimum_test_size + 10} observations, got {len(data)}. "
            f"Business impact: Cannot validate model generalization. "
            f"Required action: Increase dataset size or reduce minimum test size."
        )


def _prepare_datetime_data(data: pd.DataFrame,
                          date_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data with proper datetime handling and sorting.

    Returns working data copy and date series for split calculations.
    """
    working_data = data.copy()

    if date_column in working_data.columns:
        working_data[date_column] = pd.to_datetime(working_data[date_column])
        working_data = working_data.sort_values(date_column).reset_index(drop=True)
        date_series = working_data[date_column]
    elif isinstance(working_data.index, pd.DatetimeIndex):
        working_data = working_data.sort_index()
        date_series = working_data.index.to_series()
    else:
        warnings.warn(
            "No date column or datetime index found. "
            "Assuming sequential time series ordering."
        )
        date_range = pd.date_range(start='2022-01-01', periods=len(working_data), freq='W')
        working_data['date'] = date_range
        date_series = working_data['date']  # Must be Series for .iloc access in downstream code

    return working_data, date_series


def _calculate_split_point(n_total: int,
                          split_ratio: float,
                          ensure_minimum_test_size: int) -> Tuple[int, float]:
    """
    Calculate training set size and actual split ratio.

    Adjusts split ratio if needed to ensure minimum test set size.
    """
    n_train = int(n_total * split_ratio)

    if (n_total - n_train) < ensure_minimum_test_size:
        n_train = n_total - ensure_minimum_test_size
        actual_split_ratio = n_train / n_total
        warnings.warn(
            f"Adjusted split ratio from {split_ratio:.3f} to {actual_split_ratio:.3f} "
            f"to ensure minimum test size ({ensure_minimum_test_size} observations)"
        )
    else:
        actual_split_ratio = split_ratio

    return n_train, actual_split_ratio


def _extract_period_strings(date_series: pd.Series,
                           n_train: int) -> Tuple[str, str, str, str]:
    """Extract formatted date strings for train/test periods."""
    train_start = date_series.iloc[0].strftime('%Y-%m-%d')
    train_end = date_series.iloc[n_train - 1].strftime('%Y-%m-%d')
    test_start = date_series.iloc[n_train].strftime('%Y-%m-%d')
    test_end = date_series.iloc[-1].strftime('%Y-%m-%d')
    return train_start, train_end, test_start, test_end


def create_temporal_splits(data: pd.DataFrame,
                         split_ratio: float = 0.80,
                         date_column: str = "date",
                         ensure_minimum_test_size: int = 20,
                         business_calendar_aware: bool = True) -> TemporalSplit:
    """
    Create time-aware train/test splits for feature selection validation.

    Implements proper temporal splitting to address Issue #2 (complete absence
    of out-of-sample validation) from the mathematical analysis report.

    Raises ValueError if insufficient data for meaningful train/test split.
    """
    _validate_data_size_for_split(data, ensure_minimum_test_size)

    working_data, date_series = _prepare_datetime_data(data, date_column)

    n_train, actual_split_ratio = _calculate_split_point(
        len(working_data), split_ratio, ensure_minimum_test_size
    )

    train_data = working_data.iloc[:n_train].copy()
    test_data = working_data.iloc[n_train:].copy()

    train_start, train_end, test_start, test_end = _extract_period_strings(
        date_series, n_train
    )

    validation_checks = _validate_temporal_split_integrity(
        train_data, test_data, train_end, test_start
    )

    return TemporalSplit(
        train_data=train_data,
        test_data=test_data,
        train_period=(train_start, train_end),
        test_period=(test_start, test_end),
        split_ratio=actual_split_ratio,
        n_train_obs=len(train_data),
        n_test_obs=len(test_data),
        validation_checks=validation_checks
    )


def _validate_temporal_split_integrity(train_data: pd.DataFrame,
                                     test_data: pd.DataFrame,
                                     train_end: str,
                                     test_start: str) -> Dict[str, bool]:
    """Validate temporal split integrity to prevent data leakage."""
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    gap_days = (test_start_dt - train_end_dt).days
    total_obs = len(train_data) + len(test_data)
    train_ratio = len(train_data) / total_obs

    return {
        'no_temporal_overlap': train_end < test_start,
        'sequential_periods': gap_days <= 7,
        'train_data_non_empty': len(train_data) > 0,
        'test_data_non_empty': len(test_data) > 0,
        'consistent_features': set(train_data.columns) == set(test_data.columns),
        'reasonable_split': 0.6 <= train_ratio <= 0.9
    }


def _get_or_fit_model(model_results: Dict[str, Any],
                     temporal_split: TemporalSplit,
                     target_variable: str,
                     features: List[str]) -> Any:
    """Get trained model from results or refit on training data."""
    if 'model' in model_results:
        return model_results['model']

    import statsmodels.formula.api as smf
    train_data = temporal_split.train_data
    formula = f"{target_variable} ~ {' + '.join(features)}"
    return smf.ols(formula, data=train_data).fit()


def _calculate_test_metrics(trained_model: Any,
                           test_data: pd.DataFrame,
                           target_variable: str) -> Tuple[np.ndarray, float, float]:
    """Calculate predictions, MSE, and R-squared on test data."""
    test_predictions = trained_model.predict(test_data)
    test_actual = test_data[target_variable]

    test_mse = np.mean((test_actual - test_predictions) ** 2)
    ss_res = np.sum((test_actual - test_predictions) ** 2)
    ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
    test_r_squared = 1 - (ss_res / ss_tot)

    return test_predictions, float(test_mse), float(test_r_squared)


def _build_performance_results(trained_model: Any,
                               test_predictions: np.ndarray,
                               test_data: pd.DataFrame,
                               target_variable: str,
                               test_mse: float,
                               test_r_squared: float,
                               train_r_squared: float,
                               temporal_split: TemporalSplit,
                               features: List[str]) -> Dict[str, Any]:
    """Build comprehensive performance results dictionary."""
    generalization_gap = train_r_squared - test_r_squared
    test_actual = test_data[target_variable]

    return {
        'test_r_squared': test_r_squared,
        'test_mse': test_mse,
        'train_r_squared': train_r_squared,
        'generalization_gap': float(generalization_gap),
        'generalization_gap_percent': float(generalization_gap / train_r_squared * 100),
        'performance_assessment': _assess_generalization_quality(
            train_r_squared, test_r_squared, generalization_gap
        ),
        'test_period': temporal_split.test_period,
        'n_test_observations': temporal_split.n_test_obs,
        'features_evaluated': features,
        'model_coefficients': dict(trained_model.params),
        'test_predictions': test_predictions.tolist(),
        'test_residuals': (test_actual - test_predictions).tolist()
    }


def evaluate_out_of_sample_performance(model_results: Dict[str, Any],
                                     temporal_split: TemporalSplit,
                                     target_variable: str,
                                     features: List[str]) -> Dict[str, Any]:
    """
    Evaluate model performance on out-of-sample test data.

    Addresses Issue #2 by providing actual evidence of model generalization.
    Critical for production readiness assessment.

    Raises ValueError if model evaluation fails or data inconsistencies found.
    """
    try:
        trained_model = _get_or_fit_model(
            model_results, temporal_split, target_variable, features
        )

        test_predictions, test_mse, test_r_squared = _calculate_test_metrics(
            trained_model, temporal_split.test_data, target_variable
        )

        train_r_squared = model_results.get('r_squared', trained_model.rsquared)

        return _build_performance_results(
            trained_model, test_predictions, temporal_split.test_data,
            target_variable, test_mse, test_r_squared, train_r_squared,
            temporal_split, features
        )

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Out-of-sample evaluation failed. "
            f"Business impact: Cannot assess model generalization for production. "
            f"Required action: Check model specification and data consistency. "
            f"Original error: {e}"
        ) from e


def _classify_gap_quality(gap: float) -> Tuple[str, str, bool, str]:
    """Classify generalization gap into quality rating with recommendation."""
    if gap <= 0.05:
        return ("EXCELLENT", "HIGH", True,
                "Model shows excellent generalization. Safe for production.")
    elif gap <= 0.10:
        return ("GOOD", "MODERATE", True,
                "Model shows good generalization. Acceptable for production with monitoring.")
    elif gap <= 0.20:
        return ("FAIR", "LOW", False,
                "Model shows concerning overfitting. Consider regularization or more data.")
    else:
        return ("POOR", "VERY LOW", False,
                "Model severely overfitted. Requires major revisions before production.")


def _assess_generalization_quality(train_r2: float,
                                 test_r2: float,
                                 gap: float) -> Dict[str, Any]:
    """
    Assess quality of model generalization.

    Single responsibility: Performance assessment only.
    """
    quality, confidence, production_ready, recommendation = _classify_gap_quality(gap)

    # Override for failed model (negative R-squared)
    if test_r2 < 0:
        quality = "FAILED"
        confidence = "NONE"
        production_ready = False
        recommendation = "Model performs worse than naive mean prediction. Complete redesign needed."

    return {
        'quality_rating': quality,
        'confidence_level': confidence,
        'production_ready': production_ready,
        'recommendation': recommendation,
        'generalization_gap_threshold': gap,
        'literature_reference': "Hawkins (2004), Copas (1983) - typical degradation 30-40%"
    }


def _check_frequency_consistency(dates: pd.Series,
                                 results: Dict[str, Any]) -> None:
    """Check and record frequency consistency in validation results."""
    if len(dates) <= 1:
        return

    date_diffs = dates.diff().dropna()
    mode_diff = date_diffs.mode()
    if len(mode_diff) > 0:
        consistent_frequency = (date_diffs == mode_diff.iloc[0]).mean() > 0.8
        results['checks']['frequency_consistency'] = consistent_frequency
        results['modal_frequency'] = str(mode_diff.iloc[0])
    else:
        results['checks']['frequency_consistency'] = False


def _check_missing_periods(dates: pd.Series,
                          expected_frequency: str,
                          results: Dict[str, Any]) -> None:
    """Check and record missing periods in validation results."""
    if expected_frequency != "W":
        return

    expected_dates = pd.date_range(start=dates.min(), end=dates.max(), freq='W')
    missing_dates = expected_dates.difference(dates)
    results['checks']['no_missing_periods'] = len(missing_dates) == 0
    results['missing_periods_count'] = len(missing_dates)


def validate_temporal_structure(data: pd.DataFrame,
                              date_column: str = "date",
                              expected_frequency: str = "W") -> Dict[str, Any]:
    """
    Validate temporal structure integrity for time series analysis.

    Ensures data meets requirements for temporal validation and bootstrap analysis.
    """
    validation_results = {
        'temporal_validation_timestamp': datetime.now().isoformat(),
        'dataset_shape': data.shape,
        'checks': {}
    }

    try:
        if date_column not in data.columns:
            validation_results['checks']['date_column_exists'] = False
            return validation_results

        dates = pd.to_datetime(data[date_column])
        validation_results['checks']['date_column_exists'] = True
        validation_results['checks']['datetime_conversion_successful'] = True
        validation_results['checks']['chronological_order'] = dates.is_monotonic_increasing

        _check_frequency_consistency(dates, validation_results)
        _check_missing_periods(dates, expected_frequency, validation_results)

        validation_results['validation_passed'] = all(validation_results['checks'].values())
        return validation_results

    except Exception as e:
        validation_results['validation_error'] = str(e)
        validation_results['validation_passed'] = False
        return validation_results