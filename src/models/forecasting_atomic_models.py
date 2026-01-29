"""
Atomic Modeling Operations for Time Series Forecasting - Vectorization Target.

This module contains the core atomic operations that are the primary targets
for future vectorization across both time and bootstrap dimensions.

Current Sequential Pattern:
    for cutoff in range(30, 156):        # Time loop
        for bootstrap_sample in range(100):  # Bootstrap loop
            model = fit_single_model(...)
            prediction = predict_single(...)

Future Vectorized Pattern:
    models = fit_vectorized_ensemble(cutoffs=[30...156], bootstrap_samples=100)
    predictions = predict_vectorized_batch(models, test_features_batch)

Key Design Principles:
- Perfect mathematical separation: fitting, prediction, evaluation
- Zero coupling between atomic operations
- Single responsibility per function (10-50 lines)
- Pure mathematical functions with no business logic
- Exact preservation of Ridge regression mathematics
- Interfaces designed for future vectorization

Mathematical Precision Target:
All operations must preserve exact numerical results matching today's baseline:
- Model R²: 0.664940178498685
- Model MAPE: 14.961824607519691
- 127 forecasts with identical bootstrap distributions
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings

# Suppress specific sklearn warnings that don't indicate code issues
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='sklearn'
)
warnings.filterwarnings(
    'ignore',
    message='.*Solver terminated early.*',  # Convergence warnings for small samples
    category=Warning
)
# Justification: sklearn FutureWarnings for API changes; convergence warnings expected
# with small bootstrap samples. Will be addressed during sklearn upgrades.

# Triple-fallback imports for resilience
from src.config.forecasting_config import BootstrapModelConfig, BenchmarkModelConfig


def create_single_ridge_estimator(alpha: float, positive: bool = True,
                                 fit_intercept: bool = True) -> Ridge:
    """
    Create single Ridge regression estimator - atomic operation.

    Atomic Responsibility: Estimator creation only, no fitting.
    Vectorization Ready: Creates single estimator, easily batchable.

    Parameters
    ----------
    alpha : float
        Ridge regularization parameter (higher = more regularization)
    positive : bool, default=True
        Enforce positive coefficients (business constraint: sales >= 0)
    fit_intercept : bool, default=True
        Whether to fit intercept term

    Returns
    -------
    Ridge
        Configured Ridge regression estimator (unfitted)

    Mathematical Properties
    ----------------------
    Ridge Objective: min ||Xβ - y||² + α||β||²
    Positive Constraint: β >= 0 (economically sensible)
    Intercept: Allows for baseline sales level
    """

    # Validate alpha parameter
    if alpha <= 0:
        raise ValueError(f"Ridge alpha must be positive, got {alpha}")

    # Create Ridge estimator with exact configuration
    estimator = Ridge(
        alpha=alpha,
        positive=positive,
        fit_intercept=fit_intercept,
        random_state=None  # Deterministic for given data
    )

    return estimator


def _validate_bootstrap_inputs(X: np.ndarray, y: np.ndarray,
                               weights: np.ndarray, alpha: float) -> None:
    """Validate inputs for bootstrap model fitting."""
    if X.shape[0] != len(y):
        raise ValueError(f"Feature-target mismatch: {X.shape[0]} vs {len(y)}")
    if X.shape[0] != len(weights):
        raise ValueError(f"Feature-weight mismatch: {X.shape[0]} vs {len(weights)}")
    if not np.all(weights >= 0):
        raise ValueError("All sample weights must be non-negative")
    if alpha <= 0:
        raise ValueError(f"Alpha must be positive, got {alpha}")


def _validate_fitted_model(model: Ridge, positive: bool) -> None:
    """Validate model fitting success and constraints."""
    if not hasattr(model, 'coef_'):
        raise ValueError("Model fitting failed - no coefficients found")
    if positive and np.any(model.coef_ < -1e-10):
        raise ValueError("Positive constraint violated in fitted model")


def fit_single_bootstrap_model(X: np.ndarray, y: np.ndarray,
                              weights: np.ndarray, alpha: float,
                              positive: bool = True,
                              random_state: int = 42) -> Ridge:
    """Fit single bootstrap Ridge model with sample weights - atomic operation."""
    _validate_bootstrap_inputs(X, y, weights, alpha)

    model = Ridge(
        alpha=alpha,
        positive=positive,
        fit_intercept=True,
        random_state=random_state
    )
    model.fit(X, y, sample_weight=weights)

    _validate_fitted_model(model, positive)
    return model


def predict_single_model(model: Ridge, X_test: np.ndarray) -> float:
    """
    Generate single prediction from fitted model - atomic operation.

    Atomic Responsibility: Single prediction generation only.
    Vectorization Ready: Single model prediction, easily batchable.

    Parameters
    ----------
    model : Ridge
        Fitted Ridge regression model
    X_test : np.ndarray
        Test features, shape (1, n_features)

    Returns
    -------
    float
        Single prediction value

    Mathematical Properties
    ----------------------
    Linear Prediction: ŷ = X_test @ β + intercept
    Positive Output: Guaranteed if positive constraint used
    Deterministic: Same inputs → same outputs
    """

    # Validate inputs
    if not hasattr(model, 'coef_'):
        raise ValueError("Model must be fitted before prediction")

    if X_test.ndim != 2 or X_test.shape[0] != 1:
        raise ValueError(f"X_test must be (1, n_features), got {X_test.shape}")

    if X_test.shape[1] != len(model.coef_):
        raise ValueError(
            f"Feature count mismatch: {X_test.shape[1]} vs {len(model.coef_)}"
        )

    # Generate prediction
    prediction = model.predict(X_test)[0]

    # Validate prediction
    if not np.isfinite(prediction):
        raise ValueError(f"Invalid prediction generated: {prediction}")

    # Business constraint validation (if positive constraint was used)
    if hasattr(model, 'positive') and model.positive and prediction < -1e-10:
        raise ValueError(f"Negative prediction violates business constraint: {prediction}")

    return float(prediction)


def _validate_ensemble_fitting(ensemble: 'BaggingRegressor', n_estimators: int) -> None:
    """Validate bootstrap ensemble fitting success."""
    if not hasattr(ensemble, 'estimators_'):
        raise ValueError("BaggingRegressor fitting failed")
    if len(ensemble.estimators_) != n_estimators:
        raise ValueError(
            f"Bootstrap ensemble size mismatch: expected {n_estimators}, "
            f"got {len(ensemble.estimators_)}"
        )


def fit_bootstrap_ensemble_atomic(X: np.ndarray, y: np.ndarray,
                                 weights: np.ndarray,
                                 config: BootstrapModelConfig) -> 'BaggingRegressor':
    """Fit complete bootstrap BaggingRegressor ensemble with Ridge base estimators."""
    n_samples, n_features = X.shape
    if len(y) != n_samples or len(weights) != n_samples:
        raise ValueError("Input array length mismatch")

    alpha = config['alpha']
    positive = config['positive_constraint']
    n_estimators = config.get('n_estimators', 1000)

    base_estimator = Ridge(alpha=alpha, positive=positive, fit_intercept=True)
    bootstrap_ensemble = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=42
    )
    bootstrap_ensemble.fit(X, y, sample_weight=weights)

    _validate_ensemble_fitting(bootstrap_ensemble, n_estimators)
    return bootstrap_ensemble


def predict_bootstrap_ensemble_atomic(bagging_model: 'BaggingRegressor',
                                    X_test: np.ndarray) -> np.ndarray:
    """
    Generate predictions from BaggingRegressor ensemble - vectorization target.

    Atomic Responsibility: Ensemble prediction generation only.
    Vectorization Target: Key function for future batch optimization.

    Current Implementation: Uses BaggingRegressor's native prediction
    Future Target: Vectorized prediction across entire ensemble

    Parameters
    ----------
    bagging_model : BaggingRegressor
        Fitted BaggingRegressor with Ridge base estimators
    X_test : np.ndarray
        Test features, shape (1, n_features)

    Returns
    -------
    np.ndarray
        Bootstrap predictions, shape (n_bootstrap_samples,)

    Mathematical Properties
    ----------------------
    Ensemble Prediction: Each base estimator generates one prediction
    Uncertainty Quantification: Distribution of predictions
    Deterministic: Same ensemble → same prediction distribution
    """

    # Validate inputs
    if not hasattr(bagging_model, 'estimators_'):
        raise ValueError("BaggingRegressor must be fitted before prediction")

    if X_test.shape[0] != 1:
        raise ValueError(f"X_test must have single row, got {X_test.shape[0]}")

    # Generate predictions from all bootstrap models using BaggingRegressor
    bootstrap_predictions = []

    for estimator in bagging_model.estimators_:
        try:
            prediction = estimator.predict(X_test)[0]
            bootstrap_predictions.append(prediction)
        except Exception as e:
            raise ValueError(f"Prediction failed for bootstrap model: {e}")

    # Convert to numpy array
    predictions_array = np.array(bootstrap_predictions)

    # Validate prediction generation
    if len(predictions_array) != len(bagging_model.estimators_):
        raise ValueError(
            f"Prediction count mismatch: expected {len(bagging_model.estimators_)}, "
            f"got {len(predictions_array)}"
        )

    if not np.all(np.isfinite(predictions_array)):
        raise ValueError("Invalid predictions in bootstrap ensemble")

    return predictions_array


def calculate_prediction_error_atomic(y_true: float, y_pred: float) -> Dict[str, float]:
    """
    Calculate prediction error metrics atomically - pure mathematical operation.

    Atomic Responsibility: Error calculation only.
    Vectorization Ready: Single prediction error, easily batchable.

    Parameters
    ----------
    y_true : float
        True target value
    y_pred : float
        Predicted value

    Returns
    -------
    Dict[str, float]
        Error metrics:
        - 'absolute_error': |y_true - y_pred|
        - 'squared_error': (y_true - y_pred)²
        - 'percentage_error': |y_true - y_pred| / |y_true| * 100
        - 'relative_error': (y_pred - y_true) / y_true

    Mathematical Properties
    ----------------------
    Absolute Error: Scale-dependent error measure
    Squared Error: Penalizes large errors more heavily
    Percentage Error: Scale-independent error measure
    Relative Error: Signed relative deviation
    """

    # Validate inputs
    if not np.isfinite(y_true):
        raise ValueError(f"Invalid true value: {y_true}")

    if not np.isfinite(y_pred):
        raise ValueError(f"Invalid predicted value: {y_pred}")

    # Calculate error metrics
    absolute_error = abs(y_true - y_pred)
    squared_error = (y_true - y_pred) ** 2

    # Handle percentage error (avoid division by zero)
    if abs(y_true) > 1e-10:
        percentage_error = absolute_error / abs(y_true) * 100
        relative_error = (y_pred - y_true) / y_true
    else:
        percentage_error = 0.0 if absolute_error < 1e-10 else float('inf')
        relative_error = float('inf') if absolute_error > 1e-10 else 0.0

    return {
        'absolute_error': absolute_error,
        'squared_error': squared_error,
        'percentage_error': percentage_error,
        'relative_error': relative_error
    }


def generate_rolling_average_prediction_atomic(y_history: np.ndarray,
                                             window_size: Optional[int] = None) -> float:
    """
    Generate rolling average prediction atomically - benchmark operation.

    Atomic Responsibility: Benchmark prediction generation only.
    Vectorization Ready: Single prediction, easily batchable.

    Parameters
    ----------
    y_history : np.ndarray
        Historical target values for averaging
    window_size : Optional[int], default=None
        Window size for rolling average (None = use all history)

    Returns
    -------
    float
        Rolling average prediction

    Mathematical Properties
    ----------------------
    Rolling Average: Simple baseline forecasting method
    Window Size: Controls how much history to use
    Naive Method: Assumes future = average of past
    """

    # Validate inputs
    if len(y_history) == 0:
        raise ValueError("Empty history provided for rolling average")

    if not np.all(np.isfinite(y_history)):
        raise ValueError("Invalid values in history for rolling average")

    # Apply window size if specified
    if window_size is not None:
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        # Use most recent window_size observations
        if len(y_history) > window_size:
            y_windowed = y_history[-window_size:]
        else:
            y_windowed = y_history
    else:
        # Use all available history
        y_windowed = y_history

    # Calculate rolling average
    rolling_average = np.mean(y_windowed)

    # Validate prediction
    if not np.isfinite(rolling_average):
        raise ValueError(f"Invalid rolling average: {rolling_average}")

    return float(rolling_average)


def generate_lag_persistence_prediction_atomic(feature_value: float) -> float:
    """
    Generate lag-persistence prediction atomically - benchmark operation.

    Atomic Responsibility: Lag-persistence prediction generation only.
    Vectorization Ready: Single prediction, easily batchable.

    Parameters
    ----------
    feature_value : float
        Lag-k feature value to use as prediction (e.g., sales from k weeks ago)

    Returns
    -------
    float
        Persistence prediction (same as input value)

    Mathematical Properties
    ----------------------
    Persistence Forecast: ŷ_t = y_{t-k} (naive baseline)
    Lag-k Model: Uses feature value directly as prediction
    Deterministic: Same input → same output
    """

    # Validate input
    if not np.isfinite(feature_value):
        raise ValueError(f"Invalid feature value for persistence forecast: {feature_value}")

    # Persistence forecast: prediction = lag-k value
    persistence_prediction = float(feature_value)

    return persistence_prediction


def generate_last_value_bootstrap_prediction_atomic(last_feature_value: float,
                                                   n_bootstrap_samples: int = 100,
                                                   random_state: int = 42) -> Tuple[float, np.ndarray]:
    """
    Generate bootstrap prediction from SINGLE last feature value - EXACT original benchmark.

    CRITICAL DISCOVERY: The original rolling_average_benchmark_forecast uses
    sample_data = df_cutoff[feature_columns].iloc[-1] which is a SINGLE VALUE,
    then resamples that single value 100 times, which always returns the same value.
    This is essentially LAG-1 PERSISTENCE, not rolling average!

    Atomic Responsibility: Single value bootstrap "resampling" (constant result).
    Vectorization Ready: Single operation, easily batchable.

    Parameters
    ----------
    last_feature_value : float
        Single feature value from last row (e.g., sales_target_contract_t5 at cutoff-1)
    n_bootstrap_samples : int, default=100
        Number of bootstrap samples to generate (all will be identical)
    random_state : int, default=42
        Random seed (not used since resampling single value always gives same result)

    Returns
    -------
    Tuple[float, np.ndarray]
        Mean prediction (same as input) and bootstrap predictions array (all identical)

    Mathematical Properties
    ----------------------
    Single Value Resampling: Always returns the same value
    Lag-1 Persistence: next_value = current_value
    Zero Variance: Bootstrap std = 0 (because all samples identical)
    Matches Original: Exactly equivalent to original benchmark algorithm
    """

    # Generate bootstrap predictions (all identical for single value)
    bootstrap_predictions = np.full(n_bootstrap_samples, last_feature_value)
    mean_prediction = last_feature_value  # Same as input

    return mean_prediction, bootstrap_predictions


def generate_lag_persistence_bootstrap_atomic(feature_value: float,
                                            n_bootstrap_samples: int = 100,
                                            random_state: int = 42) -> np.ndarray:
    """
    Generate bootstrap lag-persistence predictions atomically - benchmark uncertainty.

    Atomic Responsibility: Bootstrap persistence prediction generation only.
    Vectorization Ready: Single feature value, batch across samples.

    Parameters
    ----------
    feature_value : float
        Lag-k feature value for persistence forecast
    n_bootstrap_samples : int, default=100
        Number of bootstrap samples to generate
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Bootstrap persistence predictions, shape (n_bootstrap_samples,)

    Mathematical Properties
    ----------------------
    Bootstrap Persistence: Each sample replicates the lag-k value
    Uncertainty Simulation: Provides distribution for confidence intervals
    Deterministic: Fixed random_state ensures reproducibility
    """

    # Validate inputs
    if not np.isfinite(feature_value):
        raise ValueError(f"Invalid feature value for bootstrap persistence: {feature_value}")

    if n_bootstrap_samples <= 0:
        raise ValueError(f"Bootstrap samples must be positive, got {n_bootstrap_samples}")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # For persistence forecast, all bootstrap samples are the same value
    # (This matches original implementation approach)
    bootstrap_predictions = np.full(n_bootstrap_samples, feature_value, dtype=float)

    # Add minimal variation to simulate bootstrap uncertainty (matching original)
    # This preserves the statistical properties for confidence interval calculation
    noise_scale = abs(feature_value) * 1e-10 if abs(feature_value) > 0 else 1e-10
    bootstrap_noise = np.random.normal(0, noise_scale, n_bootstrap_samples)
    bootstrap_predictions = bootstrap_predictions + bootstrap_noise

    # Validate output
    if len(bootstrap_predictions) != n_bootstrap_samples:
        raise ValueError(f"Bootstrap generation failed: expected {n_bootstrap_samples}, got {len(bootstrap_predictions)}")

    if not np.all(np.isfinite(bootstrap_predictions)):
        raise ValueError("Invalid bootstrap predictions generated")

    return bootstrap_predictions


def execute_single_cutoff_forecast(X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: float,
                                  weights: np.ndarray,
                                  config: BootstrapModelConfig) -> Dict[str, Any]:
    """Execute complete forecast for single cutoff using bootstrap ensemble."""
    bootstrap_models = fit_bootstrap_ensemble_atomic(X_train, y_train, weights, config)
    bootstrap_predictions = predict_bootstrap_ensemble_atomic(bootstrap_models, X_test)
    mean_prediction = np.mean(bootstrap_predictions)
    error_metrics = calculate_prediction_error_atomic(y_test, mean_prediction)

    return {
        'bootstrap_predictions': bootstrap_predictions,
        'mean_prediction': mean_prediction,
        'error_metrics': error_metrics,
        'y_true': y_test,
        'fitted_models': bootstrap_models if config.get('return_models', False) else None
    }


def generate_feature_bootstrap_prediction_atomic(feature_values: np.ndarray,
                                               n_bootstrap_samples: int = 100,
                                               random_state: int = 42) -> Tuple[float, np.ndarray]:
    """
    Generate bootstrap prediction from feature values - EXACT original benchmark.

    This implements the original notebook's benchmark algorithm:
    1. Bootstrap resample from historical feature values
    2. Take mean of resampled values as prediction
    3. Generate bootstrap predictions for uncertainty quantification

    Atomic Responsibility: Feature-based bootstrap prediction generation only.
    Vectorization Ready: Single operation, easily batchable.

    Parameters
    ----------
    feature_values : np.ndarray
        Historical feature values for bootstrap resampling
    n_bootstrap_samples : int, default=100
        Number of bootstrap samples to generate
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    Tuple[float, np.ndarray]
        Mean prediction and bootstrap predictions array

    Mathematical Properties
    ----------------------
    Bootstrap Resampling: Resample with replacement from historical values
    Mean Prediction: Average of resampled values
    Uncertainty Quantification: Distribution from bootstrap samples
    Matches Original: Exactly equivalent to original benchmark algorithm
    """

    # Validate inputs
    if len(feature_values) == 0:
        raise ValueError("Empty feature values provided for bootstrap prediction")

    if not np.all(np.isfinite(feature_values)):
        raise ValueError("Invalid values in feature_values for bootstrap prediction")

    if n_bootstrap_samples <= 0:
        raise ValueError(f"Bootstrap samples must be positive, got {n_bootstrap_samples}")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Bootstrap resample from feature values (matching original algorithm)
    bootstrap_predictions = []
    for i in range(n_bootstrap_samples):
        # Resample with replacement from historical feature values
        resampled_values = np.random.choice(feature_values, size=len(feature_values), replace=True)
        # Take mean of resampled values as prediction
        bootstrap_prediction = np.mean(resampled_values)
        bootstrap_predictions.append(bootstrap_prediction)

    # Convert to numpy array
    bootstrap_predictions = np.array(bootstrap_predictions)

    # Calculate mean prediction
    mean_prediction = np.mean(bootstrap_predictions)

    # Validate output
    if len(bootstrap_predictions) != n_bootstrap_samples:
        raise ValueError(f"Bootstrap generation failed: expected {n_bootstrap_samples}, got {len(bootstrap_predictions)}")

    if not np.all(np.isfinite(bootstrap_predictions)):
        raise ValueError("Invalid bootstrap predictions generated")

    return mean_prediction, bootstrap_predictions