"""
Unit Tests for Forecasting Atomic Models
========================================

Tests for src/models/forecasting_atomic_models.py covering:
- Ridge estimator creation
- Single model fitting
- Single model prediction
- Input validation functions

Target: 80% coverage for forecasting_atomic_models.py

Test Pattern:
- Test atomic operations (single model fit/predict)
- Test validation functions
- Test edge cases (invalid inputs, constraints)
- Test mathematical correctness

Author: Claude Code
Date: 2026-01-29
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.models.forecasting_atomic_models import (
    create_single_ridge_estimator,
    fit_single_bootstrap_model,
    predict_single_model,
    _validate_bootstrap_inputs,
    _validate_fitted_model,
    _validate_ensemble_fitting,
    fit_bootstrap_ensemble_atomic,
    predict_bootstrap_ensemble_atomic,
    calculate_prediction_error_atomic,
    generate_rolling_average_prediction_atomic,
    generate_lag_persistence_prediction_atomic,
    generate_last_value_bootstrap_prediction_atomic,
    generate_lag_persistence_bootstrap_atomic,
    execute_single_cutoff_forecast,
    generate_feature_bootstrap_prediction_atomic,
)
from src.config.forecasting_config import BootstrapModelConfig
from sklearn.ensemble import BaggingRegressor


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_training_data():
    """Simple training data for model fitting."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([3, 7, 11, 15, 19])  # y = x1 + x2
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    return X, y, weights


@pytest.fixture
def simple_fitted_model(simple_training_data):
    """Pre-fitted Ridge model for testing predictions."""
    X, y, weights = simple_training_data
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y, sample_weight=weights)
    return model


# =============================================================================
# ESTIMATOR CREATION TESTS
# =============================================================================


def test_create_ridge_estimator_basic():
    """Test basic Ridge estimator creation."""
    estimator = create_single_ridge_estimator(alpha=1.0)

    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 1.0
    assert estimator.positive == True
    assert estimator.fit_intercept == True


def test_create_ridge_estimator_custom_params():
    """Test Ridge estimator with custom parameters."""
    estimator = create_single_ridge_estimator(
        alpha=2.5,
        positive=False,
        fit_intercept=False
    )

    assert estimator.alpha == 2.5
    assert estimator.positive == False
    assert estimator.fit_intercept == False


def test_create_ridge_estimator_invalid_alpha():
    """Test that zero or negative alpha raises error."""
    with pytest.raises(ValueError, match="Ridge alpha must be positive"):
        create_single_ridge_estimator(alpha=0.0)

    with pytest.raises(ValueError, match="Ridge alpha must be positive"):
        create_single_ridge_estimator(alpha=-1.0)


def test_create_ridge_estimator_very_small_alpha():
    """Test Ridge estimator with very small alpha."""
    estimator = create_single_ridge_estimator(alpha=1e-10)
    assert estimator.alpha == 1e-10


def test_create_ridge_estimator_very_large_alpha():
    """Test Ridge estimator with very large alpha."""
    estimator = create_single_ridge_estimator(alpha=1e6)
    assert estimator.alpha == 1e6


# =============================================================================
# BOOTSTRAP INPUT VALIDATION TESTS
# =============================================================================


def test_validate_bootstrap_inputs_valid(simple_training_data):
    """Test validation passes with valid inputs."""
    X, y, weights = simple_training_data
    # Should not raise
    _validate_bootstrap_inputs(X, y, weights, alpha=1.0)


def test_validate_bootstrap_inputs_feature_target_mismatch():
    """Test error when X and y have different lengths."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])  # Wrong length
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="Feature-target mismatch"):
        _validate_bootstrap_inputs(X, y, weights, alpha=1.0)


def test_validate_bootstrap_inputs_feature_weight_mismatch():
    """Test error when X and weights have different lengths."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    weights = np.array([1.0, 1.0, 1.0])  # Wrong length

    with pytest.raises(ValueError, match="Feature-weight mismatch"):
        _validate_bootstrap_inputs(X, y, weights, alpha=1.0)


def test_validate_bootstrap_inputs_negative_weights():
    """Test error when weights contain negative values."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    weights = np.array([1.0, -1.0])  # Negative weight

    with pytest.raises(ValueError, match="All sample weights must be non-negative"):
        _validate_bootstrap_inputs(X, y, weights, alpha=1.0)


def test_validate_bootstrap_inputs_zero_alpha():
    """Test error when alpha is zero or negative."""
    X = np.array([[1, 2]])
    y = np.array([1])
    weights = np.array([1.0])

    with pytest.raises(ValueError, match="Alpha must be positive"):
        _validate_bootstrap_inputs(X, y, weights, alpha=0.0)


def test_validate_bootstrap_inputs_accepts_zero_weights():
    """Test that zero weights are allowed (non-negative)."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    weights = np.array([0.0, 1.0])  # Zero weight is OK

    # Should not raise
    _validate_bootstrap_inputs(X, y, weights, alpha=1.0)


# =============================================================================
# FITTED MODEL VALIDATION TESTS
# =============================================================================


def test_validate_fitted_model_valid(simple_fitted_model):
    """Test validation passes with valid fitted model."""
    # Should not raise
    _validate_fitted_model(simple_fitted_model, positive=False)


def test_validate_fitted_model_no_coefficients():
    """Test error when model has no coefficients."""
    model = Ridge(alpha=1.0)
    # Not fitted, so no coef_

    with pytest.raises(ValueError, match="Model fitting failed - no coefficients"):
        _validate_fitted_model(model, positive=False)


def test_validate_fitted_model_positive_constraint_violated():
    """Test error when positive constraint is violated."""
    # Create model with negative coefficients
    model = Ridge(alpha=1.0, positive=False, fit_intercept=True)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([10, 5])  # Decreasing y might produce negative coef
    model.fit(X, y)

    # If coefficients are significantly negative, should fail
    if np.any(model.coef_ < -1e-10):
        with pytest.raises(ValueError, match="Positive constraint violated"):
            _validate_fitted_model(model, positive=True)


# =============================================================================
# SINGLE MODEL FITTING TESTS
# =============================================================================


def test_fit_single_bootstrap_model_basic(simple_training_data):
    """Test basic single bootstrap model fitting."""
    X, y, weights = simple_training_data

    model = fit_single_bootstrap_model(
        X, y, weights,
        alpha=1.0,
        positive=True,
        random_state=42
    )

    # Should be fitted Ridge model
    assert isinstance(model, Ridge)
    assert hasattr(model, 'coef_')
    assert hasattr(model, 'intercept_')


def test_fit_single_bootstrap_model_coefficients_shape(simple_training_data):
    """Test that fitted model has correct coefficient shape."""
    X, y, weights = simple_training_data

    model = fit_single_bootstrap_model(X, y, weights, alpha=1.0)

    # Should have coefficients for each feature
    assert len(model.coef_) == X.shape[1]


def test_fit_single_bootstrap_model_positive_constraint():
    """Test that positive constraint produces non-negative coefficients."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([2, 3, 5])  # Clearly positive relationship
    weights = np.array([1.0, 1.0, 1.0])

    model = fit_single_bootstrap_model(
        X, y, weights,
        alpha=0.1,
        positive=True
    )

    # All coefficients should be non-negative (within tolerance)
    assert np.all(model.coef_ >= -1e-10)


def test_fit_single_bootstrap_model_without_positive_constraint():
    """Test model fitting without positive constraint."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    weights = np.array([1.0, 1.0])

    model = fit_single_bootstrap_model(
        X, y, weights,
        alpha=1.0,
        positive=False
    )

    # Should fit successfully (coefficients can be negative)
    assert hasattr(model, 'coef_')


def test_fit_single_bootstrap_model_with_sample_weights():
    """Test that sample weights affect model fitting."""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 100])  # Third point is outlier

    # Weights giving more importance to outlier
    weights_heavy = np.array([0.1, 0.1, 10.0])
    model_heavy = fit_single_bootstrap_model(X, y, weights_heavy, alpha=0.1, positive=False)

    # Weights giving less importance to outlier
    weights_light = np.array([1.0, 1.0, 0.01])
    model_light = fit_single_bootstrap_model(X, y, weights_light, alpha=0.1, positive=False)

    # Models should differ due to different weighting
    # Heavy weight model should be more influenced by outlier
    pred_heavy_at_3 = model_heavy.predict([[3]])[0]
    pred_light_at_3 = model_light.predict([[3]])[0]

    # Heavy model should predict higher (closer to 100)
    assert pred_heavy_at_3 > pred_light_at_3


def test_fit_single_bootstrap_model_deterministic():
    """Test that fitting with same random_state is deterministic."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    weights = np.array([1.0, 1.0, 1.0])

    model1 = fit_single_bootstrap_model(X, y, weights, alpha=1.0, random_state=42)
    model2 = fit_single_bootstrap_model(X, y, weights, alpha=1.0, random_state=42)

    # Should produce identical models
    assert np.allclose(model1.coef_, model2.coef_)
    assert np.isclose(model1.intercept_, model2.intercept_)


# =============================================================================
# SINGLE MODEL PREDICTION TESTS
# =============================================================================


def test_predict_single_model_basic(simple_fitted_model):
    """Test basic single model prediction."""
    X_test = np.array([[2, 3]])

    prediction = predict_single_model(simple_fitted_model, X_test)

    # Should return a float
    assert isinstance(prediction, float)
    assert np.isfinite(prediction)


def test_predict_single_model_mathematical_correctness():
    """Test that predictions are mathematically correct."""
    # Create simple model: y = 2*x1 + 3*x2
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([2, 3, 5])
    weights = np.array([1.0, 1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=0.001, positive=True)

    # Predict on known point
    X_test = np.array([[1, 1]])
    prediction = predict_single_model(model, X_test)

    # Should be close to 5 (with small regularization)
    assert np.abs(prediction - 5.0) < 0.5


def test_predict_single_model_unfitted():
    """Test error when predicting with unfitted model."""
    model = Ridge(alpha=1.0)
    X_test = np.array([[1, 2]])

    with pytest.raises(ValueError, match="Model must be fitted before prediction"):
        predict_single_model(model, X_test)


def test_predict_single_model_wrong_shape():
    """Test error when X_test has wrong shape."""
    model = Ridge(alpha=1.0)
    model.coef_ = np.array([1.0, 2.0])  # Fake fitting
    model.intercept_ = 0.0

    # Wrong: 2D with multiple rows
    X_test_wrong = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="X_test must be \\(1, n_features\\)"):
        predict_single_model(model, X_test_wrong)

    # Wrong: 1D array
    X_test_1d = np.array([1, 2])
    with pytest.raises(ValueError, match="X_test must be \\(1, n_features\\)"):
        predict_single_model(model, X_test_1d)


def test_predict_single_model_feature_count_mismatch(simple_fitted_model):
    """Test error when X_test has wrong number of features."""
    # Model trained on 2 features, test on 3
    X_test = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="Feature count mismatch"):
        predict_single_model(simple_fitted_model, X_test)


def test_predict_single_model_invalid_prediction():
    """Test error when prediction is not finite."""
    model = Ridge(alpha=1.0)
    model.coef_ = np.array([1e308, 1e308])  # Will cause overflow
    model.intercept_ = 0.0

    X_test = np.array([[1e308, 1e308]])

    # This might produce inf or nan
    # Note: This test may not always trigger depending on numpy behavior
    try:
        prediction = predict_single_model(model, X_test)
        # If it doesn't raise, check if finite
        assert np.isfinite(prediction)
    except ValueError as e:
        # Expected if prediction is invalid
        assert "Invalid prediction" in str(e)


def test_predict_single_model_positive_output():
    """Test that model with positive constraint produces positive predictions."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([2, 3, 5])
    weights = np.array([1.0, 1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=0.1, positive=True)

    X_test = np.array([[1, 1]])
    prediction = predict_single_model(model, X_test)

    # Should be positive
    assert prediction > 0


def test_predict_single_model_multiple_predictions_via_loop():
    """Test multiple predictions by calling function multiple times."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    weights = np.array([1.0, 1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=1.0, positive=False)

    # Make multiple predictions
    test_points = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5, 6]])]
    predictions = [predict_single_model(model, X_test) for X_test in test_points]

    # All should be finite
    assert all(np.isfinite(p) for p in predictions)

    # Predictions should be in reasonable order (increasing)
    assert predictions[0] < predictions[1] < predictions[2]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_fit_predict_pipeline():
    """Test complete fit-predict pipeline."""
    # Create training data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_coef = np.array([1, 2, 3, 4, 5])
    y = X @ true_coef + np.random.randn(100) * 0.1
    weights = np.ones(100)

    # Fit model
    model = fit_single_bootstrap_model(
        X, y, weights,
        alpha=1.0,
        positive=False,
        random_state=42
    )

    # Make prediction
    X_test = np.array([[1, 1, 1, 1, 1]])
    prediction = predict_single_model(model, X_test)

    # Prediction should be reasonable
    expected = sum(true_coef)  # 1+2+3+4+5 = 15
    assert np.abs(prediction - expected) < 2.0  # Within 2 units


def test_ridge_regularization_effect():
    """Test that ridge regularization affects coefficients."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([2, 3, 5])
    weights = np.array([1.0, 1.0, 1.0])

    # Low regularization
    model_low = fit_single_bootstrap_model(X, y, weights, alpha=0.01, positive=False)

    # High regularization
    model_high = fit_single_bootstrap_model(X, y, weights, alpha=100.0, positive=False)

    # High regularization should produce smaller coefficients (closer to zero)
    coef_norm_low = np.linalg.norm(model_low.coef_)
    coef_norm_high = np.linalg.norm(model_high.coef_)

    assert coef_norm_high < coef_norm_low


def test_fit_predict_consistency():
    """Test that predictions are consistent with training data."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    weights = np.array([1.0, 1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=0.1, positive=False)

    # Predict on training data
    predictions = [predict_single_model(model, X[[i], :]) for i in range(len(X))]

    # Predictions should be close to actual y (with low regularization)
    for pred, actual in zip(predictions, y):
        assert np.abs(pred - actual) < 1.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_fit_with_single_sample():
    """Test fitting with single sample (edge case)."""
    X = np.array([[1, 2]])
    y = np.array([3])
    weights = np.array([1.0])

    # Should fit (though not meaningful)
    model = fit_single_bootstrap_model(X, y, weights, alpha=1.0, positive=False)

    assert hasattr(model, 'coef_')


def test_fit_with_all_zero_target():
    """Test fitting when all targets are zero."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 0])
    weights = np.array([1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=1.0, positive=True)

    # Should fit with near-zero coefficients
    assert np.allclose(model.coef_, 0, atol=1e-6)


def test_predict_with_all_zero_features():
    """Test prediction when all features are zero."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([3, 7])
    weights = np.array([1.0, 1.0])

    model = fit_single_bootstrap_model(X, y, weights, alpha=1.0, positive=False)

    X_test = np.array([[0, 0]])
    prediction = predict_single_model(model, X_test)

    # Should return intercept
    assert np.isclose(prediction, model.intercept_, atol=0.01)


def test_fit_with_uniform_weights():
    """Test that uniform weights equivalent to no weights."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    # Uniform weights
    weights = np.array([1.0, 1.0, 1.0])
    model_weighted = fit_single_bootstrap_model(X, y, weights, alpha=1.0, random_state=42, positive=False)

    # No sample_weight (sklearn default is uniform)
    model_unweighted = Ridge(alpha=1.0, positive=False, fit_intercept=True, random_state=42)
    model_unweighted.fit(X, y)

    # Should be approximately equivalent
    assert np.allclose(model_weighted.coef_, model_unweighted.coef_, atol=1e-10)
    assert np.isclose(model_weighted.intercept_, model_unweighted.intercept_, atol=1e-10)


# =============================================================================
# ENSEMBLE VALIDATION TESTS
# =============================================================================


def test_validate_ensemble_fitting_valid():
    """Test ensemble validation with valid fitted ensemble."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    base_estimator = Ridge(alpha=1.0)
    ensemble = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42)
    ensemble.fit(X, y)

    # Should not raise
    _validate_ensemble_fitting(ensemble, n_estimators=10)


def test_validate_ensemble_fitting_not_fitted():
    """Test error when ensemble is not fitted."""
    ensemble = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10)

    with pytest.raises(ValueError, match="BaggingRegressor fitting failed"):
        _validate_ensemble_fitting(ensemble, n_estimators=10)


def test_validate_ensemble_fitting_size_mismatch():
    """Test error when ensemble size doesn't match expected."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])

    ensemble = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=5, random_state=42)
    ensemble.fit(X, y)

    # Expect 10 but got 5
    with pytest.raises(ValueError, match="Bootstrap ensemble size mismatch"):
        _validate_ensemble_fitting(ensemble, n_estimators=10)


# =============================================================================
# BOOTSTRAP ENSEMBLE FITTING TESTS
# =============================================================================


def test_fit_bootstrap_ensemble_basic():
    """Test basic bootstrap ensemble fitting."""
    X = np.random.randn(50, 3)
    y = X @ np.array([1, 2, 3]) + np.random.randn(50) * 0.1
    weights = np.ones(50)

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': True,
        'n_estimators': 10
    }

    ensemble = fit_bootstrap_ensemble_atomic(X, y, weights, config)

    assert isinstance(ensemble, BaggingRegressor)
    assert hasattr(ensemble, 'estimators_')
    assert len(ensemble.estimators_) == 10


def test_fit_bootstrap_ensemble_default_n_estimators():
    """Test ensemble fitting with default n_estimators."""
    X = np.random.randn(20, 2)
    y = X @ np.array([1, 2]) + np.random.randn(20) * 0.1
    weights = np.ones(20)

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': True
        # n_estimators not specified, should default to 1000
    }

    ensemble = fit_bootstrap_ensemble_atomic(X, y, weights, config)

    # Should use default of 1000
    assert len(ensemble.estimators_) == 1000


def test_fit_bootstrap_ensemble_input_mismatch():
    """Test error when input arrays have mismatched lengths."""
    X = np.random.randn(10, 2)
    y = np.random.randn(8)  # Wrong length
    weights = np.ones(10)

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': True,
        'n_estimators': 10
    }

    with pytest.raises(ValueError, match="Input array length mismatch"):
        fit_bootstrap_ensemble_atomic(X, y, weights, config)


def test_fit_bootstrap_ensemble_positive_constraint():
    """Test ensemble fitting with positive constraint."""
    X = np.array([[1, 0], [0, 1], [1, 1]] * 10)  # Repeat for stability
    y = np.array([2, 3, 5] * 10)
    weights = np.ones(30)

    config: BootstrapModelConfig = {
        'alpha': 0.1,
        'positive_constraint': True,
        'n_estimators': 5
    }

    ensemble = fit_bootstrap_ensemble_atomic(X, y, weights, config)

    # Check that base estimators have positive constraint
    for estimator in ensemble.estimators_:
        assert np.all(estimator.coef_ >= -1e-10)


# =============================================================================
# BOOTSTRAP ENSEMBLE PREDICTION TESTS
# =============================================================================


def test_predict_bootstrap_ensemble_basic():
    """Test basic bootstrap ensemble prediction."""
    X = np.random.randn(30, 3)
    y = X @ np.array([1, 2, 3]) + np.random.randn(30) * 0.1
    weights = np.ones(30)

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': True,
        'n_estimators': 10
    }

    ensemble = fit_bootstrap_ensemble_atomic(X, y, weights, config)

    X_test = np.array([[1, 1, 1]])
    predictions = predict_bootstrap_ensemble_atomic(ensemble, X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 10
    assert np.all(np.isfinite(predictions))


def test_predict_bootstrap_ensemble_unfitted():
    """Test error when predicting with unfitted ensemble."""
    ensemble = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=10)
    X_test = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="BaggingRegressor must be fitted"):
        predict_bootstrap_ensemble_atomic(ensemble, X_test)


def test_predict_bootstrap_ensemble_multiple_rows():
    """Test error when X_test has multiple rows."""
    X = np.random.randn(20, 2)
    y = X @ np.array([1, 2])

    ensemble = BaggingRegressor(estimator=Ridge(alpha=1.0), n_estimators=5, random_state=42)
    ensemble.fit(X, y)

    X_test = np.array([[1, 2], [3, 4]])  # Multiple rows

    with pytest.raises(ValueError, match="X_test must have single row"):
        predict_bootstrap_ensemble_atomic(ensemble, X_test)


def test_predict_bootstrap_ensemble_distribution():
    """Test that bootstrap predictions form a distribution."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1, 2, 3]) + np.random.randn(100) * 0.5
    weights = np.ones(100)

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': False,
        'n_estimators': 50
    }

    ensemble = fit_bootstrap_ensemble_atomic(X, y, weights, config)

    X_test = np.array([[1, 1, 1]])
    predictions = predict_bootstrap_ensemble_atomic(ensemble, X_test)

    # Should have non-zero variance (bootstrapping creates variation)
    assert np.std(predictions) > 0

    # Mean should be close to expected value (1+2+3=6)
    assert np.abs(np.mean(predictions) - 6) < 1.0


# =============================================================================
# PREDICTION ERROR CALCULATION TESTS
# =============================================================================


def test_calculate_prediction_error_basic():
    """Test basic prediction error calculation."""
    errors = calculate_prediction_error_atomic(y_true=10.0, y_pred=8.0)

    assert 'absolute_error' in errors
    assert 'squared_error' in errors
    assert 'percentage_error' in errors
    assert 'relative_error' in errors

    assert errors['absolute_error'] == 2.0
    assert errors['squared_error'] == 4.0
    assert errors['percentage_error'] == 20.0
    assert errors['relative_error'] == -0.2


def test_calculate_prediction_error_perfect():
    """Test error calculation with perfect prediction."""
    errors = calculate_prediction_error_atomic(y_true=5.0, y_pred=5.0)

    assert errors['absolute_error'] == 0.0
    assert errors['squared_error'] == 0.0
    assert errors['percentage_error'] == 0.0
    assert errors['relative_error'] == 0.0


def test_calculate_prediction_error_division_by_zero():
    """Test error calculation when y_true is zero."""
    errors = calculate_prediction_error_atomic(y_true=0.0, y_pred=5.0)

    assert errors['absolute_error'] == 5.0
    assert errors['squared_error'] == 25.0
    assert errors['percentage_error'] == float('inf')
    assert errors['relative_error'] == float('inf')


def test_calculate_prediction_error_near_zero():
    """Test error calculation when y_true is near zero."""
    errors = calculate_prediction_error_atomic(y_true=1e-12, y_pred=1e-12)

    # Near-zero values should give zero error
    assert errors['percentage_error'] == 0.0
    assert errors['relative_error'] == 0.0


def test_calculate_prediction_error_invalid_inputs():
    """Test error with invalid inputs."""
    with pytest.raises(ValueError, match="Invalid true value"):
        calculate_prediction_error_atomic(y_true=np.nan, y_pred=5.0)

    with pytest.raises(ValueError, match="Invalid predicted value"):
        calculate_prediction_error_atomic(y_true=5.0, y_pred=np.inf)


# =============================================================================
# ROLLING AVERAGE PREDICTION TESTS
# =============================================================================


def test_rolling_average_basic():
    """Test basic rolling average prediction."""
    history = np.array([10, 20, 30, 40, 50])

    prediction = generate_rolling_average_prediction_atomic(history)

    assert prediction == 30.0  # Mean of [10,20,30,40,50]


def test_rolling_average_with_window():
    """Test rolling average with window size."""
    history = np.array([10, 20, 30, 40, 50])

    prediction = generate_rolling_average_prediction_atomic(history, window_size=3)

    # Should use last 3 values: [30, 40, 50]
    assert prediction == 40.0


def test_rolling_average_window_larger_than_history():
    """Test rolling average when window is larger than history."""
    history = np.array([10, 20])

    prediction = generate_rolling_average_prediction_atomic(history, window_size=10)

    # Should use all available history
    assert prediction == 15.0


def test_rolling_average_empty_history():
    """Test error with empty history."""
    with pytest.raises(ValueError, match="Empty history"):
        generate_rolling_average_prediction_atomic(np.array([]))


def test_rolling_average_invalid_history():
    """Test error with invalid values in history."""
    with pytest.raises(ValueError, match="Invalid values in history"):
        generate_rolling_average_prediction_atomic(np.array([10, np.nan, 30]))


def test_rolling_average_invalid_window():
    """Test error with invalid window size."""
    history = np.array([10, 20, 30])

    with pytest.raises(ValueError, match="Window size must be positive"):
        generate_rolling_average_prediction_atomic(history, window_size=0)


# =============================================================================
# LAG PERSISTENCE PREDICTION TESTS
# =============================================================================


def test_lag_persistence_basic():
    """Test basic lag persistence prediction."""
    prediction = generate_lag_persistence_prediction_atomic(feature_value=42.5)

    assert prediction == 42.5


def test_lag_persistence_zero():
    """Test lag persistence with zero value."""
    prediction = generate_lag_persistence_prediction_atomic(feature_value=0.0)

    assert prediction == 0.0


def test_lag_persistence_negative():
    """Test lag persistence with negative value."""
    prediction = generate_lag_persistence_prediction_atomic(feature_value=-10.5)

    assert prediction == -10.5


def test_lag_persistence_invalid():
    """Test error with invalid feature value."""
    with pytest.raises(ValueError, match="Invalid feature value"):
        generate_lag_persistence_prediction_atomic(feature_value=np.nan)


# =============================================================================
# LAST VALUE BOOTSTRAP PREDICTION TESTS
# =============================================================================


def test_last_value_bootstrap_basic():
    """Test basic last value bootstrap prediction."""
    mean_pred, bootstrap_preds = generate_last_value_bootstrap_prediction_atomic(
        last_feature_value=50.0,
        n_bootstrap_samples=100,
        random_state=42
    )

    assert mean_pred == 50.0
    assert len(bootstrap_preds) == 100
    assert np.all(bootstrap_preds == 50.0)  # All identical


def test_last_value_bootstrap_different_value():
    """Test last value bootstrap with different value."""
    mean_pred, bootstrap_preds = generate_last_value_bootstrap_prediction_atomic(
        last_feature_value=123.45,
        n_bootstrap_samples=50
    )

    assert mean_pred == 123.45
    assert len(bootstrap_preds) == 50
    assert np.all(bootstrap_preds == 123.45)


# =============================================================================
# LAG PERSISTENCE BOOTSTRAP TESTS
# =============================================================================


def test_lag_persistence_bootstrap_basic():
    """Test basic lag persistence bootstrap."""
    bootstrap_preds = generate_lag_persistence_bootstrap_atomic(
        feature_value=50.0,
        n_bootstrap_samples=100,
        random_state=42
    )

    assert len(bootstrap_preds) == 100
    assert np.all(np.isfinite(bootstrap_preds))

    # Should be very close to feature value (with tiny noise)
    assert np.abs(np.mean(bootstrap_preds) - 50.0) < 1e-8


def test_lag_persistence_bootstrap_invalid_samples():
    """Test error with invalid number of bootstrap samples."""
    with pytest.raises(ValueError, match="Bootstrap samples must be positive"):
        generate_lag_persistence_bootstrap_atomic(feature_value=50.0, n_bootstrap_samples=0)


def test_lag_persistence_bootstrap_invalid_feature():
    """Test error with invalid feature value."""
    with pytest.raises(ValueError, match="Invalid feature value"):
        generate_lag_persistence_bootstrap_atomic(feature_value=np.inf, n_bootstrap_samples=100)


# =============================================================================
# EXECUTE SINGLE CUTOFF FORECAST TESTS
# =============================================================================


def test_execute_single_cutoff_forecast_basic():
    """Test complete single cutoff forecast execution."""
    np.random.seed(42)
    X_train = np.random.randn(50, 3)
    y_train = X_train @ np.array([1, 2, 3]) + np.random.randn(50) * 0.1
    weights = np.ones(50)

    X_test = np.array([[1, 1, 1]])
    y_test = 6.0

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': False,
        'n_estimators': 20,
        'return_models': False
    }

    result = execute_single_cutoff_forecast(X_train, y_train, X_test, y_test, weights, config)

    assert 'bootstrap_predictions' in result
    assert 'mean_prediction' in result
    assert 'error_metrics' in result
    assert 'y_true' in result

    assert len(result['bootstrap_predictions']) == 20
    assert isinstance(result['mean_prediction'], float)
    assert result['y_true'] == 6.0


def test_execute_single_cutoff_forecast_with_models():
    """Test forecast execution with model return."""
    X_train = np.random.randn(30, 2)
    y_train = X_train @ np.array([1, 2])
    weights = np.ones(30)

    X_test = np.array([[1, 1]])
    y_test = 3.0

    config: BootstrapModelConfig = {
        'alpha': 1.0,
        'positive_constraint': True,
        'n_estimators': 10,
        'return_models': True
    }

    result = execute_single_cutoff_forecast(X_train, y_train, X_test, y_test, weights, config)

    assert result['fitted_models'] is not None
    assert isinstance(result['fitted_models'], BaggingRegressor)


# =============================================================================
# FEATURE BOOTSTRAP PREDICTION TESTS
# =============================================================================


def test_feature_bootstrap_prediction_basic():
    """Test basic feature bootstrap prediction."""
    feature_values = np.array([10, 20, 30, 40, 50])

    mean_pred, bootstrap_preds = generate_feature_bootstrap_prediction_atomic(
        feature_values,
        n_bootstrap_samples=100,
        random_state=42
    )

    assert isinstance(mean_pred, float)
    assert len(bootstrap_preds) == 100
    assert np.all(np.isfinite(bootstrap_preds))

    # Mean should be around the mean of original values
    assert np.abs(mean_pred - 30.0) < 10.0


def test_feature_bootstrap_prediction_single_value():
    """Test bootstrap with single feature value."""
    feature_values = np.array([42.0])

    mean_pred, bootstrap_preds = generate_feature_bootstrap_prediction_atomic(
        feature_values,
        n_bootstrap_samples=50,
        random_state=42
    )

    # All predictions should be the same value
    assert np.all(bootstrap_preds == 42.0)
    assert mean_pred == 42.0


def test_feature_bootstrap_prediction_empty():
    """Test error with empty feature values."""
    with pytest.raises(ValueError, match="Empty feature values"):
        generate_feature_bootstrap_prediction_atomic(np.array([]), n_bootstrap_samples=100)


def test_feature_bootstrap_prediction_invalid():
    """Test error with invalid feature values."""
    with pytest.raises(ValueError, match="Invalid values in feature_values"):
        generate_feature_bootstrap_prediction_atomic(
            np.array([10, np.nan, 30]),
            n_bootstrap_samples=100
        )


def test_feature_bootstrap_prediction_invalid_samples():
    """Test error with invalid number of bootstrap samples."""
    with pytest.raises(ValueError, match="Bootstrap samples must be positive"):
        generate_feature_bootstrap_prediction_atomic(
            np.array([10, 20, 30]),
            n_bootstrap_samples=0
        )


def test_feature_bootstrap_prediction_deterministic():
    """Test that bootstrap prediction is deterministic with same seed."""
    feature_values = np.array([10, 20, 30, 40, 50])

    mean1, preds1 = generate_feature_bootstrap_prediction_atomic(
        feature_values, n_bootstrap_samples=100, random_state=42
    )

    mean2, preds2 = generate_feature_bootstrap_prediction_atomic(
        feature_values, n_bootstrap_samples=100, random_state=42
    )

    assert mean1 == mean2
    assert np.allclose(preds1, preds2)
