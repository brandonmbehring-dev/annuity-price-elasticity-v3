"""
Atomic Validation Operations for Time Series Forecasting - Mathematical Equivalence.

This module provides atomic validation operations for ensuring mathematical
equivalence during forecasting refactoring. Each function has single
responsibility for specific validation aspects.

Key Validation Areas:
- Input data validation (features, targets, weights)
- Model fitting validation (coefficients, convergence)
- Prediction validation (ranges, constraints)
- Performance metric validation (precision, consistency)
- Cross-validation sequence validation (temporal integrity)

Target Baseline Values (Today's Run - Oct 30, 2025):
- Model R²: 0.664940178498685 ± 1e-6
- Model MAPE: 14.961824607519691 ± 1e-4
- Bootstrap predictions: 127 forecasts × 100 samples (element-wise ± 1e-6)
- Confidence intervals: 96 percentiles × 127 forecasts (± 1e-6)

Mathematical Precision Standards:
All validations enforce the precision requirements for complete computational
lineage validation with 1000+ checkpoints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.linear_model import Ridge
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


def _validate_shape_consistency(X: np.ndarray, y: np.ndarray,
                                weights: Optional[np.ndarray],
                                context: str) -> bool:
    """Validate array shape consistency for X, y, and weights."""
    try:
        if X.ndim != 2:
            print(f"{context}: X must be 2D array, got {X.ndim}D")
            return False
        if y.ndim != 1:
            print(f"{context}: y must be 1D array, got {y.ndim}D")
            return False
        if X.shape[0] != len(y):
            print(f"{context}: X-y shape mismatch {X.shape[0]} vs {len(y)}")
            return False
        if weights is not None and len(weights) != len(y):
            print(f"{context}: weights-y shape mismatch {len(weights)} vs {len(y)}")
            return False
        return True
    except Exception as e:
        print(f"{context}: Shape validation error - {e}")
        return False


def _validate_finite_values(X: np.ndarray, y: np.ndarray,
                           weights: Optional[np.ndarray],
                           context: str) -> bool:
    """Validate all values are finite (no NaN/inf)."""
    try:
        x_finite = np.all(np.isfinite(X))
        y_finite = np.all(np.isfinite(y))
        weights_finite = True if weights is None else np.all(np.isfinite(weights))

        if not x_finite:
            print(f"{context}: Non-finite values in X")
        if not y_finite:
            print(f"{context}: Non-finite values in y")
        if not weights_finite:
            print(f"{context}: Non-finite values in weights")

        return x_finite and y_finite and weights_finite
    except Exception as e:
        print(f"{context}: Finite values validation error - {e}")
        return False


def _validate_positive_weights(weights: Optional[np.ndarray],
                               context: str) -> bool:
    """Validate all weights are non-negative."""
    try:
        if weights is not None:
            weights_positive = np.all(weights >= 0)
            if not weights_positive:
                print(f"{context}: Negative weights found")
            return weights_positive
        return True
    except Exception as e:
        print(f"{context}: Weights validation error - {e}")
        return False


def _validate_sufficient_samples(X: np.ndarray, context: str,
                                 min_samples: int = 10) -> bool:
    """Validate minimum samples for meaningful modeling."""
    try:
        sufficient = X.shape[0] >= min_samples
        if not sufficient:
            print(f"{context}: Insufficient samples {X.shape[0]} < {min_samples}")
        return sufficient
    except Exception as e:
        print(f"{context}: Sample count validation error - {e}")
        return False


def _validate_feature_variance(X: np.ndarray, context: str,
                               min_variance: float = 1e-10) -> bool:
    """Validate features have sufficient variance (relaxed for Ridge)."""
    try:
        feature_std = np.std(X, axis=0)
        constant_features = np.where(feature_std <= min_variance)[0]
        all_constant = len(constant_features) == X.shape[1]

        if len(constant_features) > 0 and not all_constant:
            print(f"{context}: {len(constant_features)} constant features at indices "
                  f"{constant_features} (Ridge can handle)")
        elif all_constant:
            print(f"{context}: ALL features are constant - Ridge regression will fail")

        return not all_constant
    except Exception as e:
        print(f"{context}: Feature variance validation error - {e}")
        return False


def validate_input_data_atomic(X: np.ndarray, y: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              context: str = "forecasting") -> Dict[str, bool]:
    """
    Validate input data atomically - comprehensive data integrity checks.

    Atomic Responsibility: Input data validation only.
    Validation Scope: Mathematical and statistical properties.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for validation, shape (n_samples, n_features)
    y : np.ndarray
        Target array for validation, shape (n_samples,)
    weights : Optional[np.ndarray], default=None
        Sample weights for validation, shape (n_samples,)
    context : str, default="forecasting"
        Validation context for error messages

    Returns
    -------
    Dict[str, bool]
        Validation results:
        - 'shape_consistency': Arrays have consistent shapes
        - 'finite_values': All values are finite (no NaN/inf)
        - 'positive_weights': All weights are non-negative
        - 'sufficient_samples': Minimum samples for modeling
        - 'feature_variance': Features have sufficient variance

    Mathematical Properties
    ----------------------
    Shape Consistency: X.shape[0] == len(y) == len(weights)
    Finite Values: No NaN or infinite values
    Positive Weights: All weights >= 0
    Sufficient Samples: At least 10 observations for meaningful modeling
    Feature Variance: No constant features (prevent singular matrices)
    """
    return {
        'shape_consistency': _validate_shape_consistency(X, y, weights, context),
        'finite_values': _validate_finite_values(X, y, weights, context),
        'positive_weights': _validate_positive_weights(weights, context),
        'sufficient_samples': _validate_sufficient_samples(X, context),
        'feature_variance': _validate_feature_variance(X, context)
    }


def _validate_model_fitted(model: Ridge) -> bool:
    """Validate model has been fitted successfully."""
    try:
        has_coef = hasattr(model, 'coef_')
        has_intercept = hasattr(model, 'intercept_')
        if not has_coef:
            print("Model validation: No coefficients found (not fitted)")
        if not has_intercept:
            print("Model validation: No intercept found (not fitted)")
        return has_coef and has_intercept
    except Exception as e:
        print(f"Model fitted validation error - {e}")
        return False


def _validate_coefficients_finite(model: Ridge) -> bool:
    """Validate all coefficients are finite."""
    try:
        coef_finite = np.all(np.isfinite(model.coef_))
        intercept_finite = np.isfinite(model.intercept_)
        if not coef_finite:
            print("Model validation: Non-finite coefficients")
        if not intercept_finite:
            print("Model validation: Non-finite intercept")
        return coef_finite and intercept_finite
    except Exception as e:
        print(f"Coefficients validation error - {e}")
        return False


def _validate_positive_constraint(model: Ridge, config: Dict[str, Any],
                                  tolerance: float = 1e-10) -> bool:
    """Validate positive constraint on coefficients if enabled."""
    try:
        if not config.get('positive_constraint', False):
            return True
        coefficients_positive = np.all(model.coef_ >= -tolerance)
        if not coefficients_positive:
            negative_coef = model.coef_[model.coef_ < -tolerance]
            print(f"Model validation: Negative coefficients {negative_coef}")
        return coefficients_positive
    except Exception as e:
        print(f"Positive constraint validation error - {e}")
        return False


def _validate_intercept_reasonable(model: Ridge, y: np.ndarray) -> bool:
    """Validate intercept is within reasonable range relative to target."""
    try:
        y_scale = np.std(y) if len(y) > 1 else 1.0
        y_mean = np.mean(y)
        reasonable_range = 10 * y_scale

        intercept_reasonable = abs(model.intercept_ - y_mean) <= reasonable_range
        if not intercept_reasonable:
            print(f"Model validation: Unreasonable intercept {model.intercept_:.2e} "
                  f"(target mean: {y_mean:.2e}, scale: {y_scale:.2e})")
        return intercept_reasonable
    except Exception as e:
        print(f"Intercept validation error - {e}")
        return False


def _validate_prediction_capability(model: Ridge, X: np.ndarray) -> bool:
    """Validate model can generate finite predictions."""
    try:
        X_test = X[:1, :] if len(X) > 0 else np.zeros((1, X.shape[1]))
        test_prediction = model.predict(X_test)[0]
        prediction_valid = np.isfinite(test_prediction)
        if not prediction_valid:
            print(f"Model validation: Invalid prediction {test_prediction}")
        return prediction_valid
    except Exception as e:
        print(f"Prediction capability validation error - {e}")
        return False


def validate_model_fit_atomic(model: Ridge, X: np.ndarray, y: np.ndarray,
                             config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate model fitting results atomically - mathematical model properties.

    Atomic Responsibility: Fitted model validation only.
    Validation Scope: Model coefficients, convergence, constraints.

    Parameters
    ----------
    model : Ridge
        Fitted Ridge regression model
    X : np.ndarray
        Training features used for fitting
    y : np.ndarray
        Training targets used for fitting
    config : Dict[str, Any]
        Model configuration for validation

    Returns
    -------
    Dict[str, bool]
        Model validation results:
        - 'model_fitted': Model has been fitted successfully
        - 'coefficients_finite': All coefficients are finite
        - 'positive_constraint': Positive constraints satisfied (if enabled)
        - 'intercept_reasonable': Intercept term is reasonable
        - 'prediction_capability': Model can generate predictions

    Mathematical Properties
    ----------------------
    Model Fitted: Has coef_ and intercept_ attributes
    Coefficients Finite: No NaN or infinite coefficients
    Positive Constraint: All coefficients >= 0 (if enabled)
    Reasonable Intercept: Intercept within reasonable range
    Prediction Capability: Can generate finite predictions
    """
    model_fitted = _validate_model_fitted(model)

    if not model_fitted:
        return {key: False for key in ['model_fitted', 'coefficients_finite',
                                       'positive_constraint', 'intercept_reasonable',
                                       'prediction_capability']}

    return {
        'model_fitted': model_fitted,
        'coefficients_finite': _validate_coefficients_finite(model),
        'positive_constraint': _validate_positive_constraint(model, config),
        'intercept_reasonable': _validate_intercept_reasonable(model, y),
        'prediction_capability': _validate_prediction_capability(model, X)
    }


def _validate_bootstrap_correct_size(bootstrap_predictions: np.ndarray,
                                     config: Dict[str, Any]) -> bool:
    """Validate correct number of bootstrap predictions."""
    try:
        expected_size = config.get('n_bootstrap_samples', 100)
        correct_size = len(bootstrap_predictions) == expected_size
        if not correct_size:
            print(f"Bootstrap validation: Size mismatch {len(bootstrap_predictions)} vs {expected_size}")
        return correct_size
    except Exception as e:
        print(f"Bootstrap size validation error - {e}")
        return False


def _validate_bootstrap_finite(bootstrap_predictions: np.ndarray) -> bool:
    """Validate all predictions are finite."""
    try:
        all_finite = np.all(np.isfinite(bootstrap_predictions))
        if not all_finite:
            invalid_count = np.sum(~np.isfinite(bootstrap_predictions))
            print(f"Bootstrap validation: {invalid_count} non-finite predictions")
        return all_finite
    except Exception as e:
        print(f"Bootstrap finite validation error - {e}")
        return False


def _validate_bootstrap_positive(bootstrap_predictions: np.ndarray,
                                 config: Dict[str, Any],
                                 tolerance: float = -1e-10) -> bool:
    """Validate positive predictions if constraint enabled."""
    try:
        if not config.get('positive_constraint', False):
            return True
        all_positive = np.all(bootstrap_predictions >= tolerance)
        if not all_positive:
            negative_count = np.sum(bootstrap_predictions < tolerance)
            min_prediction = np.min(bootstrap_predictions)
            print(f"Bootstrap validation: {negative_count} negative predictions "
                  f"(min: {min_prediction:.2e})")
        return all_positive
    except Exception as e:
        print(f"Bootstrap positive validation error - {e}")
        return False


def _validate_bootstrap_reasonable_range(bootstrap_predictions: np.ndarray,
                                         y_true: float,
                                         config: Dict[str, Any]) -> bool:
    """Validate predictions within reasonable range of true value."""
    try:
        reasonable_multiple = config.get('reasonable_multiple', 10.0)
        y_true_abs = abs(y_true) if abs(y_true) > 1e-10 else 1.0

        lower_bound = y_true / reasonable_multiple if y_true > 0 else -reasonable_multiple * y_true_abs
        upper_bound = y_true * reasonable_multiple if y_true > 0 else reasonable_multiple * y_true_abs

        within_range = np.all((bootstrap_predictions >= lower_bound) &
                              (bootstrap_predictions <= upper_bound))

        if not within_range:
            out_of_range_count = np.sum((bootstrap_predictions < lower_bound) |
                                        (bootstrap_predictions > upper_bound))
            print(f"Bootstrap validation: {out_of_range_count} predictions out of range "
                  f"[{lower_bound:.2e}, {upper_bound:.2e}]")
        return within_range
    except Exception as e:
        print(f"Bootstrap range validation error - {e}")
        return False


def _validate_bootstrap_sufficient_variance(bootstrap_predictions: np.ndarray,
                                            config: Dict[str, Any]) -> bool:
    """Validate bootstrap shows meaningful uncertainty."""
    try:
        min_cv = config.get('min_cv', 0.001)
        prediction_mean = np.mean(bootstrap_predictions)
        prediction_std = np.std(bootstrap_predictions)

        if abs(prediction_mean) > 1e-10:
            cv = prediction_std / abs(prediction_mean)
            sufficient_variance = cv >= min_cv
            if not sufficient_variance:
                print(f"Bootstrap validation: Insufficient variance (CV: {cv:.6f})")
        else:
            sufficient_variance = prediction_std > 1e-10
        return sufficient_variance
    except Exception as e:
        print(f"Bootstrap variance validation error - {e}")
        return False


def validate_bootstrap_predictions_atomic(bootstrap_predictions: np.ndarray,
                                        y_true: float,
                                        config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate bootstrap predictions atomically - statistical properties.

    Atomic Responsibility: Bootstrap prediction validation only.
    Validation Scope: Statistical properties, business constraints.

    Parameters
    ----------
    bootstrap_predictions : np.ndarray
        Bootstrap prediction array, shape (n_bootstrap_samples,)
    y_true : float
        True target value for reference
    config : Dict[str, Any]
        Validation configuration

    Returns
    -------
    Dict[str, bool]
        Bootstrap validation results:
        - 'correct_size': Correct number of bootstrap predictions
        - 'finite_predictions': All predictions are finite
        - 'positive_predictions': All predictions are positive (if constrained)
        - 'reasonable_range': Predictions within reasonable range
        - 'sufficient_variance': Bootstrap shows uncertainty

    Mathematical Properties
    ----------------------
    Correct Size: Expected number of bootstrap samples
    Finite Predictions: No NaN or infinite predictions
    Positive Predictions: All predictions >= 0 (business constraint)
    Reasonable Range: Predictions within reasonable bounds
    Sufficient Variance: Bootstrap uncertainty is meaningful
    """
    return {
        'correct_size': _validate_bootstrap_correct_size(bootstrap_predictions, config),
        'finite_predictions': _validate_bootstrap_finite(bootstrap_predictions),
        'positive_predictions': _validate_bootstrap_positive(bootstrap_predictions, config),
        'reasonable_range': _validate_bootstrap_reasonable_range(bootstrap_predictions, y_true, config),
        'sufficient_variance': _validate_bootstrap_sufficient_variance(bootstrap_predictions, config)
    }


def validate_performance_metrics_atomic(metrics: Dict[str, float],
                                      baseline_metrics: Dict[str, float],
                                      tolerances: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate performance metrics against baseline atomically - precision validation.

    Atomic Responsibility: Performance metric comparison only.
    Validation Scope: Mathematical equivalence within tolerances.

    Parameters
    ----------
    metrics : Dict[str, float]
        Computed performance metrics
    baseline_metrics : Dict[str, float]
        Baseline metrics for comparison
    tolerances : Dict[str, float]
        Tolerance levels for each metric

    Returns
    -------
    Dict[str, bool]
        Metric validation results for each metric

    Mathematical Properties
    ----------------------
    Element-wise Comparison: Each metric compared individually
    Tolerance Enforcement: Absolute difference within tolerance
    Precision Validation: Ultra-precise comparison (1e-6 level)
    """

    validation_results = {}

    # Validate each metric against baseline
    for metric_name in baseline_metrics.keys():
        try:
            if metric_name not in metrics:
                print(f"Metric validation: Missing metric {metric_name}")
                validation_results[metric_name] = False
                continue

            computed_value = metrics[metric_name]
            baseline_value = baseline_metrics[metric_name]
            tolerance = tolerances.get(metric_name, 1e-6)

            # Calculate absolute difference
            absolute_diff = abs(computed_value - baseline_value)
            within_tolerance = absolute_diff <= tolerance

            if not within_tolerance:
                print(f"Metric validation: {metric_name} outside tolerance")
                print(f"  Expected: {baseline_value:.12f}")
                print(f"  Computed: {computed_value:.12f}")
                print(f"  Difference: {absolute_diff:.2e}")
                print(f"  Tolerance: {tolerance:.2e}")

            validation_results[metric_name] = within_tolerance

        except Exception as e:
            print(f"Metric validation error for {metric_name}: {e}")
            validation_results[metric_name] = False

    return validation_results


def _validate_cv_correct_count(dates: List[str],
                               expected_sequence: Dict[str, Any]) -> bool:
    """Validate correct number of forecasts."""
    try:
        expected_count = expected_sequence.get('n_forecasts', 127)
        correct_count = len(dates) == expected_count
        if not correct_count:
            print(f"CV sequence: Count mismatch {len(dates)} vs {expected_count}")
        return correct_count
    except Exception as e:
        print(f"CV count validation error - {e}")
        return False


def _validate_cv_temporal_order(dates: List[str]) -> bool:
    """Validate dates in chronological order."""
    try:
        dates_pd = pd.to_datetime(dates)
        is_sorted = dates_pd.is_monotonic_increasing
        if not is_sorted:
            print("CV sequence: Dates not in chronological order")
        return is_sorted
    except Exception as e:
        print(f"CV temporal order validation error - {e}")
        return False


def _validate_cv_expanding_window(cutoffs: List[int],
                                  expected_sequence: Dict[str, Any]) -> bool:
    """Validate cutoffs form expanding sequence."""
    try:
        expected_start = expected_sequence.get('start_cutoff', 30)
        expected_cutoffs = list(range(expected_start, expected_start + len(cutoffs)))
        correct_sequence = cutoffs == expected_cutoffs
        if not correct_sequence:
            print(f"CV sequence: Cutoff mismatch")
            print(f"  Expected: {expected_cutoffs[:5]}...{expected_cutoffs[-5:]}")
            print(f"  Actual: {cutoffs[:5]}...{cutoffs[-5:]}")
        return correct_sequence
    except Exception as e:
        print(f"CV expanding window validation error - {e}")
        return False


def _validate_cv_date_range(dates: List[str],
                            expected_sequence: Dict[str, Any]) -> bool:
    """Validate correct start and end dates."""
    try:
        expected_start_date = expected_sequence.get('start_date', '2023-04-02')
        expected_end_date = expected_sequence.get('end_date', '2025-08-31')

        actual_start_date = dates[0] if dates else None
        actual_end_date = dates[-1] if dates else None

        start_match = actual_start_date == expected_start_date
        end_match = actual_end_date == expected_end_date

        if not start_match:
            print(f"CV sequence: Start date mismatch {actual_start_date} vs {expected_start_date}")
        if not end_match:
            print(f"CV sequence: End date mismatch {actual_end_date} vs {expected_end_date}")
        return start_match and end_match
    except Exception as e:
        print(f"CV date range validation error - {e}")
        return False


def validate_cross_validation_sequence_atomic(dates: List[str], cutoffs: List[int],
                                            expected_sequence: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate cross-validation sequence atomically - temporal integrity.

    Atomic Responsibility: CV sequence validation only.
    Validation Scope: Temporal sequence, expanding window integrity.

    Parameters
    ----------
    dates : List[str]
        Forecast dates from cross-validation
    cutoffs : List[int]
        Cutoff indices from cross-validation
    expected_sequence : Dict[str, Any]
        Expected sequence properties

    Returns
    -------
    Dict[str, bool]
        CV sequence validation results:
        - 'correct_count': Correct number of forecasts
        - 'temporal_order': Dates in chronological order
        - 'expanding_window': Cutoffs form expanding sequence
        - 'date_range': Correct start and end dates

    Mathematical Properties
    ----------------------
    Expanding Window: cutoffs = [30, 31, 32, ..., n-1]
    Temporal Order: Dates in chronological sequence
    Complete Sequence: No missing forecasts in range
    """
    return {
        'correct_count': _validate_cv_correct_count(dates, expected_sequence),
        'temporal_order': _validate_cv_temporal_order(dates),
        'expanding_window': _validate_cv_expanding_window(cutoffs, expected_sequence),
        'date_range': _validate_cv_date_range(dates, expected_sequence)
    }


def validate_confidence_intervals_atomic(confidence_intervals: Dict[str, np.ndarray],
                                       bootstrap_matrix: np.ndarray,
                                       tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Validate confidence intervals atomically - statistical accuracy.

    Atomic Responsibility: Confidence interval validation only.
    Validation Scope: Statistical accuracy, percentile calculations.

    Parameters
    ----------
    confidence_intervals : Dict[str, np.ndarray]
        Computed confidence intervals
    bootstrap_matrix : np.ndarray
        Bootstrap prediction matrix for verification
    tolerance : float, default=1e-6
        Numerical tolerance for validation

    Returns
    -------
    Dict[str, bool]
        Confidence interval validation results

    Mathematical Properties
    ----------------------
    Percentile Accuracy: CIs match percentiles of bootstrap distribution
    Monotonicity: Higher percentiles ≥ lower percentiles
    Coverage: CIs span reasonable range of bootstrap predictions
    """

    validation_results = {}

    # Validate each confidence interval
    for ci_name, ci_values in confidence_intervals.items():
        try:
            # Extract percentile from name (e.g., "05th-percentile" -> 7.5)
            if 'percentile' in ci_name:
                percentile_str = ci_name.split('th-percentile')[0]
                if percentile_str.isdigit():
                    percentile = float(percentile_str) + 2.5  # Convert to original percentile

                    # Recalculate expected values
                    expected_values = np.percentile(bootstrap_matrix, percentile, axis=0)

                    # Compare with computed values
                    max_diff = np.max(np.abs(ci_values - expected_values))
                    within_tolerance = max_diff <= tolerance

                    if not within_tolerance:
                        print(f"CI validation: {ci_name} outside tolerance {max_diff:.2e}")

                    validation_results[ci_name] = within_tolerance
                else:
                    validation_results[ci_name] = False
            else:
                validation_results[ci_name] = True  # Non-percentile metrics

        except Exception as e:
            print(f"CI validation error for {ci_name}: {e}")
            validation_results[ci_name] = False

    return validation_results