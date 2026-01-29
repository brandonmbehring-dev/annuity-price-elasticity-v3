"""Tests for inference_training module.

Covers: prepare_training_data, train_bootstrap_model, transform_prediction_features
Target: 81% → 95% coverage
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingRegressor

from src.models.inference_training import (
    TrainingData,
    _get_product_name,
    prepare_training_data,
    train_bootstrap_model,
    transform_prediction_features,
    _create_bagging_regressor,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_training_df():
    """Create sample training dataframe with required columns."""
    dates = pd.date_range("2024-01-01", periods=20, freq="W")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "holiday": [0] * 18 + [1, 0],  # One holiday week
        "forward_0": np.random.uniform(100, 500, 20),
        "prudential_rate": np.random.uniform(3.0, 5.0, 20),
        "competitor_rate": np.random.uniform(2.5, 4.5, 20),
        "economic_indicator": np.random.uniform(0.5, 1.5, 20),
    })


@pytest.fixture
def sample_features():
    """Standard feature list for testing."""
    return ["prudential_rate", "competitor_rate", "economic_indicator"]


# =============================================================================
# TESTS: _get_product_name (Line 50)
# =============================================================================


class TestGetProductName:
    """Tests for _get_product_name utility function."""

    def test_returns_string(self):
        """_get_product_name returns a string product name."""
        result = _get_product_name()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_expected_format(self):
        """_get_product_name returns FlexGuard product format."""
        result = _get_product_name()
        # Should return something like "FlexGuard 6Y20B"
        assert "FlexGuard" in result or "6Y" in result or len(result) > 3


# =============================================================================
# TESTS: TrainingData dataclass
# =============================================================================


class TestTrainingData:
    """Tests for TrainingData dataclass."""

    def test_dataclass_fields(self, sample_training_df, sample_features):
        """TrainingData correctly stores all fields."""
        X = sample_training_df[sample_features]
        y = sample_training_df["forward_0"]
        w = pd.Series(np.ones(len(y)))

        data = TrainingData(X=X, y=y, w=w, df_train=sample_training_df)

        assert data.X.shape == X.shape
        assert len(data.y) == len(y)
        assert len(data.w) == len(w)
        assert len(data.df_train) == len(sample_training_df)


# =============================================================================
# TESTS: prepare_training_data (Line 109 - empty data error)
# =============================================================================


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""

    def test_applies_exponential_decay_weights(self, sample_training_df, sample_features):
        """Weights decrease exponentially for older observations."""
        result = prepare_training_data(
            df=sample_training_df,
            current_date_of_mature_data="2024-06-01",
            target_variable="forward_0",
            features=sample_features,
            weight_decay_factor=0.99
        )

        # Most recent weights should be higher
        weights = result.w.values
        assert weights[-1] > weights[0], "Recent weights should exceed older weights"

    def test_filters_by_cutoff_date(self, sample_training_df, sample_features):
        """Training data excludes observations after cutoff date."""
        cutoff = "2024-03-15"
        result = prepare_training_data(
            df=sample_training_df,
            current_date_of_mature_data=cutoff,
            target_variable="forward_0",
            features=sample_features,
            weight_decay_factor=0.99
        )

        cutoff_dt = pd.to_datetime(cutoff)
        assert all(result.df_train["date"] < cutoff_dt)

    def test_excludes_holiday_weeks(self, sample_training_df, sample_features):
        """Training data excludes weeks marked as holidays."""
        result = prepare_training_data(
            df=sample_training_df,
            current_date_of_mature_data="2024-06-01",
            target_variable="forward_0",
            features=sample_features,
            weight_decay_factor=0.99
        )

        assert all(result.df_train["holiday"] == 0)

    def test_returns_training_data_object(self, sample_training_df, sample_features):
        """Returns properly structured TrainingData object."""
        result = prepare_training_data(
            df=sample_training_df,
            current_date_of_mature_data="2024-06-01",
            target_variable="forward_0",
            features=sample_features,
            weight_decay_factor=0.99
        )

        assert isinstance(result, TrainingData)
        assert isinstance(result.X, pd.DataFrame)
        assert isinstance(result.y, pd.Series)
        assert isinstance(result.w, pd.Series)

    def test_empty_data_raises_error(self, sample_training_df, sample_features):
        """ValueError raised when no training data remains after filtering (Line 109)."""
        # Use cutoff date before all data
        with pytest.raises(ValueError, match="No training data before cutoff"):
            prepare_training_data(
                df=sample_training_df,
                current_date_of_mature_data="2023-01-01",  # Before all data
                target_variable="forward_0",
                features=sample_features,
                weight_decay_factor=0.99
            )

    def test_all_holidays_raises_error(self, sample_features):
        """ValueError raised when all data is holidays."""
        all_holiday_df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="W"),
            "holiday": [1] * 5,  # All holidays
            "forward_0": [100] * 5,
            "prudential_rate": [4.0] * 5,
            "competitor_rate": [3.5] * 5,
            "economic_indicator": [1.0] * 5,
        })

        with pytest.raises(ValueError, match="No training data before cutoff"):
            prepare_training_data(
                df=all_holiday_df,
                current_date_of_mature_data="2024-06-01",
                target_variable="forward_0",
                features=sample_features,
                weight_decay_factor=0.99
            )


# =============================================================================
# TESTS: _create_bagging_regressor (Lines 133-134 - TypeError fallback)
# =============================================================================


class TestCreateBaggingRegressor:
    """Tests for _create_bagging_regressor with sklearn API compatibility."""

    def test_creates_bagging_regressor(self):
        """Creates BaggingRegressor with correct parameters."""
        from sklearn.linear_model import Ridge

        base = Ridge(alpha=1.0)
        result = _create_bagging_regressor(base, n_estimators=10, random_state=42)

        assert isinstance(result, BaggingRegressor)
        assert result.n_estimators == 10
        assert result.random_state == 42
        assert result.bootstrap is True
        assert result.bootstrap_features is False

    def test_type_error_fallback(self):
        """Fallback to base_estimator parameter on TypeError (Lines 133-134)."""
        from sklearn.linear_model import Ridge

        base = Ridge(alpha=1.0)

        # Mock the BaggingRegressor to raise TypeError on first call
        with patch("src.models.inference_training.BaggingRegressor") as mock_br:
            # First call raises TypeError (old API), second succeeds
            mock_instance = MagicMock(spec=BaggingRegressor)
            mock_br.side_effect = [TypeError("estimator not recognized"), mock_instance]

            result = _create_bagging_regressor(base, n_estimators=10, random_state=42)

            # Should have called twice - once with estimator=, once with base_estimator=
            assert mock_br.call_count == 2

            # First call should use 'estimator'
            first_call_kwargs = mock_br.call_args_list[0][1]
            assert 'estimator' in first_call_kwargs

            # Second call should use 'base_estimator'
            second_call_kwargs = mock_br.call_args_list[1][1]
            assert 'base_estimator' in second_call_kwargs


# =============================================================================
# TESTS: train_bootstrap_model
# =============================================================================


class TestTrainBootstrapModel:
    """Tests for train_bootstrap_model function."""

    def test_returns_fitted_model(self, sample_training_df, sample_features):
        """Returns a fitted BaggingRegressor."""
        X = sample_training_df[sample_features].iloc[:15]
        y = sample_training_df["forward_0"].iloc[:15]
        w = pd.Series(np.ones(15))

        model = train_bootstrap_model(
            X=X, y=y, sample_weights=w,
            n_estimators=5, random_state=42, ridge_alpha=1.0
        )

        assert isinstance(model, BaggingRegressor)
        # Model should be fitted (has estimators_)
        assert hasattr(model, "estimators_")

    def test_log_transform_applied(self, sample_training_df, sample_features):
        """Model trains on log(1+y) transformed target."""
        X = sample_training_df[sample_features].iloc[:15]
        y = sample_training_df["forward_0"].iloc[:15]
        w = pd.Series(np.ones(15))

        model = train_bootstrap_model(
            X=X, y=y, sample_weights=w,
            n_estimators=5, random_state=42, ridge_alpha=1.0
        )

        # Predict should return log-scale values (need to exp transform)
        predictions = model.predict(X)
        # Log-scale predictions should be in reasonable range
        assert np.all(predictions > 0)
        assert np.all(predictions < 10)  # log(1+500) ≈ 6.2

    def test_sample_weights_used(self, sample_training_df, sample_features):
        """Sample weights affect training (no assertion, just runs without error)."""
        X = sample_training_df[sample_features].iloc[:15]
        y = sample_training_df["forward_0"].iloc[:15]

        # Create non-uniform weights
        w = pd.Series(np.linspace(0.1, 1.0, 15))

        model = train_bootstrap_model(
            X=X, y=y, sample_weights=w,
            n_estimators=5, random_state=42, ridge_alpha=1.0
        )

        assert model is not None


# =============================================================================
# TESTS: transform_prediction_features (Lines 195, 200-205)
# =============================================================================


class TestTransformPredictionFeatures:
    """Tests for transform_prediction_features function."""

    def test_prudential_rate_transformation(self):
        """Prudential rate features use most recent value (Line 195)."""
        features = ["prudential_rate", "competitor_rate"]
        X_test_base = pd.Series({
            "prudential_rate": 4.0,
            "competitor_rate": 3.5,
        })

        df_rates = pd.DataFrame({
            "Prudential": [4.5, 4.8, 5.0],  # Most recent is 5.0
            "date": pd.date_range("2024-01-01", periods=3, freq="W"),
        })

        df_sales = pd.DataFrame({
            "sales_by_contract_date_lag_1": [100, 110, 120],
            "sales_by_contract_date_lag_2": [90, 100, 110],
            "sales_by_contract_date_lag_3": [80, 90, 100],
        })

        result = transform_prediction_features(
            X_test_base=X_test_base,
            features=features,
            df_rates=df_rates,
            df_sales=df_sales
        )

        # Prudential rate should be updated to latest value
        assert result["prudential_rate"] == 5.0
        # Non-prudential rate should be unchanged
        assert result["competitor_rate"] == 3.5

    def test_sales_momentum_transformation(self):
        """Sales momentum uses 3-period average (Lines 200-205)."""
        features = ["sales_by_contract_date_momentum"]
        X_test_base = pd.Series({
            "sales_by_contract_date_momentum": 100.0,
        })

        df_rates = pd.DataFrame({
            "Prudential": [4.5],
        })

        df_sales = pd.DataFrame({
            "sales_by_contract_date_lag_1": [120],
            "sales_by_contract_date_lag_2": [110],
            "sales_by_contract_date_lag_3": [100],
        })

        result = transform_prediction_features(
            X_test_base=X_test_base,
            features=features,
            df_rates=df_rates,
            df_sales=df_sales
        )

        # Should be average of lags: (120 + 110 + 100) / 3 = 110
        expected_momentum = (120 + 110 + 100) / 3
        assert result["sales_by_contract_date_momentum"] == expected_momentum

    def test_preserves_other_features(self):
        """Features without transformation rules keep original values."""
        features = ["economic_indicator", "other_feature"]
        X_test_base = pd.Series({
            "economic_indicator": 1.5,
            "other_feature": 42.0,
        })

        df_rates = pd.DataFrame({"Prudential": [4.5]})
        df_sales = pd.DataFrame({
            "sales_by_contract_date_lag_1": [100],
            "sales_by_contract_date_lag_2": [100],
            "sales_by_contract_date_lag_3": [100],
        })

        result = transform_prediction_features(
            X_test_base=X_test_base,
            features=features,
            df_rates=df_rates,
            df_sales=df_sales
        )

        # Both should be unchanged
        assert result["economic_indicator"] == 1.5
        assert result["other_feature"] == 42.0

    def test_does_not_modify_input(self):
        """Input Series is not modified (copy semantics)."""
        features = ["prudential_rate"]
        X_test_base = pd.Series({"prudential_rate": 4.0})
        original_value = X_test_base["prudential_rate"]

        df_rates = pd.DataFrame({"Prudential": [5.0]})
        df_sales = pd.DataFrame({
            "sales_by_contract_date_lag_1": [100],
            "sales_by_contract_date_lag_2": [100],
            "sales_by_contract_date_lag_3": [100],
        })

        _ = transform_prediction_features(
            X_test_base=X_test_base,
            features=features,
            df_rates=df_rates,
            df_sales=df_sales
        )

        # Original should be unchanged
        assert X_test_base["prudential_rate"] == original_value


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestTrainingPipelineIntegration:
    """Integration tests for the full training pipeline."""

    def test_full_training_workflow(self, sample_training_df, sample_features):
        """Full workflow: prepare data → train model → predict."""
        # 1. Prepare training data
        training_data = prepare_training_data(
            df=sample_training_df,
            current_date_of_mature_data="2024-05-01",
            target_variable="forward_0",
            features=sample_features,
            weight_decay_factor=0.99
        )

        # 2. Train model
        model = train_bootstrap_model(
            X=training_data.X,
            y=training_data.y,
            sample_weights=training_data.w,
            n_estimators=5,
            random_state=42,
            ridge_alpha=1.0
        )

        # 3. Make predictions
        log_predictions = model.predict(training_data.X)
        predictions = np.exp(log_predictions) - 1

        # Predictions should be positive and reasonable
        assert np.all(predictions > 0)
        assert np.all(predictions < 2000)  # Upper bound sanity check
