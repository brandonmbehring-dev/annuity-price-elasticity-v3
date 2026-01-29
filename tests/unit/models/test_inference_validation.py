"""Tests for inference_validation module.

Covers all validation helper functions and main validation orchestrators.
Target: 28% → 60% coverage
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge

from src.models.inference_validation import (
    # Utilities
    _get_product_name,
    # Validation helpers
    _validate_dataframes,
    _validate_required_columns,
    _validate_date_format,
    _validate_model_parameters,
    _validate_data_quality,
    # Rate adjustment validators
    _validate_rate_adj_dataframes,
    _validate_rate_adj_arrays,
    _validate_rate_adj_model,
    _validate_rate_adj_features,
    _validate_rate_adj_parameters,
    # Main validators
    validate_center_baseline_inputs,
    validate_rate_adjustments_inputs,
    validate_confidence_interval_inputs,
    validate_melt_dataframe_inputs,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_sales_df():
    """Create sample sales dataframe with required columns."""
    dates = pd.date_range("2024-01-01", periods=20, freq="W")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "holiday": [0] * 20,
        "forward_0": np.random.uniform(100, 500, 20),
        "prudential_rate_current": np.random.uniform(3.0, 5.0, 20),
        "competitor_mid_current": np.random.uniform(2.5, 4.5, 20),
        "economic_indicator": np.random.uniform(0.5, 1.5, 20),
        "sales_by_contract_date_lag_1": np.random.uniform(100, 200, 20),
        "sales_by_contract_date_lag_2": np.random.uniform(90, 190, 20),
        "sales_by_contract_date_lag_3": np.random.uniform(80, 180, 20),
    })


@pytest.fixture
def sample_rates_df():
    """Create sample rates dataframe."""
    return pd.DataFrame({
        "Prudential": [4.5, 4.8, 5.0],
        "date": pd.date_range("2024-01-01", periods=3, freq="W"),
    })


@pytest.fixture
def sample_features():
    """Standard feature list for testing."""
    return ["prudential_rate_current", "competitor_mid_current", "economic_indicator"]


@pytest.fixture
def trained_model_fixture(sample_sales_df, sample_features):
    """Create a fitted BaggingRegressor."""
    X = sample_sales_df[sample_features].iloc[:15]
    y = sample_sales_df["forward_0"].iloc[:15]

    base_estimator = Ridge(alpha=1.0)
    model = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=5,
        random_state=42,
        bootstrap=True,
        bootstrap_features=False
    )
    model.fit(X, np.log(1 + y))
    return model


@pytest.fixture
def sample_ci_df():
    """Sample confidence interval DataFrame."""
    return pd.DataFrame({
        "rate_change_in_basis_points": [25, 50, 75, 100],
        "bottom": [100, 110, 120, 130],
        "median": [150, 160, 170, 180],
        "top": [200, 210, 220, 230]
    })


# =============================================================================
# TESTS: _get_product_name (Line 48)
# =============================================================================


class TestGetProductName:
    """Tests for _get_product_name utility."""

    def test_returns_string(self):
        """Returns a product name string."""
        result = _get_product_name()
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# TESTS: _validate_dataframes (Lines 74, 78)
# =============================================================================


class TestValidateDataframes:
    """Tests for _validate_dataframes helper."""

    def test_valid_dataframes_pass(self, sample_sales_df, sample_rates_df):
        """Valid DataFrames pass validation."""
        _validate_dataframes(sample_sales_df, sample_rates_df)  # No exception

    def test_none_sales_df_raises(self, sample_rates_df):
        """None sales DataFrame raises ValueError (Line 74)."""
        with pytest.raises(ValueError, match="RILA sales DataFrame cannot be None or empty"):
            _validate_dataframes(None, sample_rates_df)

    def test_empty_sales_df_raises(self, sample_rates_df):
        """Empty sales DataFrame raises ValueError (Line 74)."""
        with pytest.raises(ValueError, match="RILA sales DataFrame cannot be None or empty"):
            _validate_dataframes(pd.DataFrame(), sample_rates_df)

    def test_none_rates_df_raises(self, sample_sales_df):
        """None rates DataFrame raises ValueError (Line 78)."""
        with pytest.raises(ValueError, match="WINK competitive rates DataFrame cannot be None or empty"):
            _validate_dataframes(sample_sales_df, None)

    def test_empty_rates_df_raises(self, sample_sales_df):
        """Empty rates DataFrame raises ValueError (Line 78)."""
        with pytest.raises(ValueError, match="WINK competitive rates DataFrame cannot be None or empty"):
            _validate_dataframes(sample_sales_df, pd.DataFrame())


# =============================================================================
# TESTS: _validate_required_columns (Lines 103, 107)
# =============================================================================


class TestValidateRequiredColumns:
    """Tests for _validate_required_columns helper."""

    def test_valid_columns_pass(self, sample_sales_df, sample_features):
        """Valid columns pass validation."""
        _validate_required_columns(sample_sales_df, sample_features, "forward_0")

    def test_missing_target_raises(self, sample_sales_df, sample_features):
        """Missing target variable raises ValueError (Line 103)."""
        with pytest.raises(ValueError, match="Target variable 'nonexistent' not found"):
            _validate_required_columns(sample_sales_df, sample_features, "nonexistent")

    def test_missing_features_raises(self, sample_sales_df):
        """Missing features raise ValueError (Line 107)."""
        bad_features = ["prudential_rate_current", "missing_feature1", "missing_feature2"]
        with pytest.raises(ValueError, match="Missing features in sales DataFrame"):
            _validate_required_columns(sample_sales_df, bad_features, "forward_0")


# =============================================================================
# TESTS: _validate_date_format (Lines 125-126)
# =============================================================================


class TestValidateDateFormat:
    """Tests for _validate_date_format helper."""

    def test_valid_date_passes(self):
        """Valid ISO date passes."""
        _validate_date_format("2024-05-01")  # No exception

    def test_invalid_format_raises(self):
        """Invalid date format raises ValueError (Lines 125-126)."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _validate_date_format("05-01-2024")  # MM-DD-YYYY format

    def test_nonsense_raises(self):
        """Nonsense string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            _validate_date_format("not-a-date")

    def test_datetime_object_raises(self):
        """Non-string raises TypeError (strptime requires string)."""
        with pytest.raises(TypeError):
            _validate_date_format(20240501)


# =============================================================================
# TESTS: _validate_model_parameters (Lines 156, 158, 160, 162)
# =============================================================================


class TestValidateModelParameters:
    """Tests for _validate_model_parameters helper."""

    def test_valid_parameters_pass(self):
        """Valid parameters pass validation."""
        _validate_model_parameters(
            n_estimators=100,
            weight_decay_factor=0.99,
            random_state=42,
            ridge_alpha=1.0
        )

    def test_invalid_n_estimators_raises(self):
        """Invalid n_estimators raises ValueError (Line 156)."""
        with pytest.raises(ValueError, match="n_estimators must be positive integer"):
            _validate_model_parameters(0, 0.99, 42, 1.0)

        with pytest.raises(ValueError, match="n_estimators must be positive integer"):
            _validate_model_parameters(-5, 0.99, 42, 1.0)

        with pytest.raises(ValueError, match="n_estimators must be positive integer"):
            _validate_model_parameters(10.5, 0.99, 42, 1.0)

    def test_invalid_weight_decay_raises(self):
        """Invalid weight_decay_factor raises ValueError (Line 158)."""
        with pytest.raises(ValueError, match="weight_decay_factor must be in"):
            _validate_model_parameters(100, 0.0, 42, 1.0)

        with pytest.raises(ValueError, match="weight_decay_factor must be in"):
            _validate_model_parameters(100, 1.5, 42, 1.0)

        with pytest.raises(ValueError, match="weight_decay_factor must be in"):
            _validate_model_parameters(100, -0.5, 42, 1.0)

    def test_invalid_random_state_raises(self):
        """Invalid random_state raises ValueError (Line 160)."""
        with pytest.raises(ValueError, match="random_state must be non-negative integer"):
            _validate_model_parameters(100, 0.99, -1, 1.0)

        with pytest.raises(ValueError, match="random_state must be non-negative integer"):
            _validate_model_parameters(100, 0.99, 42.5, 1.0)

    def test_invalid_ridge_alpha_raises(self):
        """Invalid ridge_alpha raises ValueError (Line 162)."""
        with pytest.raises(ValueError, match="ridge_alpha must be non-negative"):
            _validate_model_parameters(100, 0.99, 42, -1.0)


# =============================================================================
# TESTS: _validate_data_quality (Lines 187, 191-192)
# =============================================================================


class TestValidateDataQuality:
    """Tests for _validate_data_quality helper."""

    def test_valid_data_passes(self, sample_sales_df, sample_features):
        """Data without missing values passes."""
        _validate_data_quality(sample_sales_df, sample_features, "forward_0")

    def test_target_na_raises(self, sample_sales_df, sample_features):
        """Missing values in target raises ValueError (Line 187)."""
        df = sample_sales_df.copy()
        df.loc[5, "forward_0"] = np.nan

        with pytest.raises(ValueError, match="Target variable.*contains missing values"):
            _validate_data_quality(df, sample_features, "forward_0")

    def test_feature_na_raises(self, sample_sales_df, sample_features):
        """Missing values in features raises ValueError (Lines 191-192)."""
        df = sample_sales_df.copy()
        df.loc[5, "prudential_rate_current"] = np.nan

        with pytest.raises(ValueError, match="Features contain missing values"):
            _validate_data_quality(df, sample_features, "forward_0")


# =============================================================================
# TESTS: _validate_rate_adj_dataframes (Lines 266-269)
# =============================================================================


class TestValidateRateAdjDataframes:
    """Tests for _validate_rate_adj_dataframes helper."""

    def test_valid_dfs_pass(self, sample_sales_df, sample_rates_df):
        """Valid DataFrames pass validation."""
        _validate_rate_adj_dataframes(sample_sales_df, sample_rates_df)

    def test_none_sales_raises(self, sample_rates_df):
        """None sales DataFrame raises ValueError (Line 267)."""
        with pytest.raises(ValueError, match="Sales DataFrame cannot be None or empty"):
            _validate_rate_adj_dataframes(None, sample_rates_df)

    def test_none_rates_raises(self, sample_sales_df):
        """None rates DataFrame raises ValueError (Line 269)."""
        with pytest.raises(ValueError, match="Rates DataFrame cannot be None or empty"):
            _validate_rate_adj_dataframes(sample_sales_df, None)


# =============================================================================
# TESTS: _validate_rate_adj_arrays (Lines 290-293)
# =============================================================================


class TestValidateRateAdjArrays:
    """Tests for _validate_rate_adj_arrays helper."""

    def test_valid_arrays_pass(self):
        """Valid arrays pass validation."""
        _validate_rate_adj_arrays(np.array([3.5, 4.0, 4.5]), np.array([100, 150, 200]))

    def test_none_scenarios_raises(self):
        """None rate_scenarios raises ValueError (Line 291)."""
        with pytest.raises(ValueError, match="Rate scenarios array cannot be None or empty"):
            _validate_rate_adj_arrays(None, np.array([100]))

    def test_empty_scenarios_raises(self):
        """Empty rate_scenarios raises ValueError (Line 291)."""
        with pytest.raises(ValueError, match="Rate scenarios array cannot be None or empty"):
            _validate_rate_adj_arrays(np.array([]), np.array([100]))

    def test_none_baseline_raises(self):
        """None baseline_predictions raises ValueError (Line 293)."""
        with pytest.raises(ValueError, match="Baseline predictions array cannot be None or empty"):
            _validate_rate_adj_arrays(np.array([4.0]), None)

    def test_empty_baseline_raises(self):
        """Empty baseline_predictions raises ValueError (Line 293)."""
        with pytest.raises(ValueError, match="Baseline predictions array cannot be None or empty"):
            _validate_rate_adj_arrays(np.array([4.0]), np.array([]))


# =============================================================================
# TESTS: _validate_rate_adj_model (Lines 309-312)
# =============================================================================


class TestValidateRateAdjModel:
    """Tests for _validate_rate_adj_model helper."""

    def test_valid_model_passes(self, trained_model_fixture):
        """Fitted model passes validation."""
        _validate_rate_adj_model(trained_model_fixture)

    def test_none_model_raises(self):
        """None model raises ValueError (Line 310)."""
        with pytest.raises(ValueError, match="Trained model cannot be None"):
            _validate_rate_adj_model(None)

    def test_model_without_estimators_raises(self):
        """Model without estimators_ raises ValueError (Line 312)."""
        unfitted_model = BaggingRegressor(n_estimators=5)
        with pytest.raises(ValueError, match="Trained model must have estimators_ attribute"):
            _validate_rate_adj_model(unfitted_model)


# =============================================================================
# TESTS: _validate_rate_adj_features (Lines 333-338)
# =============================================================================


class TestValidateRateAdjFeatures:
    """Tests for _validate_rate_adj_features helper."""

    def test_valid_features_pass(self, sample_sales_df, sample_features):
        """Valid features pass validation."""
        _validate_rate_adj_features(sample_features, sample_sales_df)

    def test_none_features_raises(self, sample_sales_df):
        """None features raises ValueError (Line 334)."""
        with pytest.raises(ValueError, match="Features list cannot be None or empty"):
            _validate_rate_adj_features(None, sample_sales_df)

    def test_empty_features_raises(self, sample_sales_df):
        """Empty features raises ValueError (Line 334)."""
        with pytest.raises(ValueError, match="Features list cannot be None or empty"):
            _validate_rate_adj_features([], sample_sales_df)

    def test_missing_features_raises(self, sample_sales_df):
        """Missing features raises ValueError (Lines 337-338)."""
        bad_features = ["prudential_rate_current", "nonexistent_feature"]
        with pytest.raises(ValueError, match="Missing features in sales DataFrame"):
            _validate_rate_adj_features(bad_features, sample_sales_df)


# =============================================================================
# TESTS: _validate_rate_adj_parameters (Lines 362-369)
# =============================================================================


class TestValidateRateAdjParameters:
    """Tests for _validate_rate_adj_parameters helper."""

    def test_valid_params_pass(self):
        """Valid parameters pass validation."""
        _validate_rate_adj_parameters(
            competitor_rate_adjustment=0.25,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

    def test_invalid_competitor_adj_raises(self):
        """Non-numeric competitor_rate_adjustment raises ValueError (Line 364)."""
        with pytest.raises(ValueError, match="competitor_rate_adjustment must be numeric"):
            _validate_rate_adj_parameters("invalid", 13.0, 3)

    def test_invalid_sales_multiplier_raises(self):
        """Invalid sales_multiplier raises ValueError (Line 367)."""
        with pytest.raises(ValueError, match="sales_multiplier must be positive"):
            _validate_rate_adj_parameters(0.25, 0, 3)

        with pytest.raises(ValueError, match="sales_multiplier must be positive"):
            _validate_rate_adj_parameters(0.25, -5.0, 3)

    def test_invalid_momentum_lookback_raises(self):
        """Invalid momentum_lookback_periods raises ValueError (Lines 369-370)."""
        with pytest.raises(ValueError, match="momentum_lookback_periods must be positive integer"):
            _validate_rate_adj_parameters(0.25, 13.0, 0)

        with pytest.raises(ValueError, match="momentum_lookback_periods must be positive integer"):
            _validate_rate_adj_parameters(0.25, 13.0, -1)

        with pytest.raises(ValueError, match="momentum_lookback_periods must be positive integer"):
            _validate_rate_adj_parameters(0.25, 13.0, 3.5)


# =============================================================================
# TESTS: validate_center_baseline_inputs
# =============================================================================


class TestValidateCenterBaselineInputs:
    """Tests for main validate_center_baseline_inputs orchestrator."""

    def test_valid_inputs_pass(self, sample_sales_df, sample_rates_df, sample_features):
        """Valid inputs pass all validation."""
        validate_center_baseline_inputs(
            df=sample_sales_df,
            df_rates=sample_rates_df,
            features=sample_features,
            target_variable="forward_0",
            current_date_of_mature_data="2024-05-01",
            n_estimators=100,
            weight_decay_factor=0.99,
            random_state=42,
            ridge_alpha=1.0
        )

    def test_cascades_dataframe_validation(self, sample_rates_df, sample_features):
        """Cascades to DataFrame validation."""
        with pytest.raises(ValueError, match="RILA sales DataFrame cannot be None"):
            validate_center_baseline_inputs(
                df=None,
                df_rates=sample_rates_df,
                features=sample_features,
                target_variable="forward_0",
                current_date_of_mature_data="2024-05-01",
                n_estimators=100,
                weight_decay_factor=0.99,
                random_state=42,
                ridge_alpha=1.0
            )


# =============================================================================
# TESTS: validate_rate_adjustments_inputs (Lines 415-419)
# =============================================================================


class TestValidateRateAdjustmentsInputs:
    """Tests for main validate_rate_adjustments_inputs orchestrator."""

    def test_valid_inputs_pass(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_fixture
    ):
        """Valid inputs pass all validation (Lines 415-421)."""
        validate_rate_adjustments_inputs(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=np.array([3.5, 4.0, 4.5]),
            baseline_predictions=np.array([150, 160, 170, 180, 190]),
            trained_model=trained_model_fixture,
            features=sample_features,
            competitor_rate_adjustment=0.25,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

    def test_cascades_dataframe_validation(
        self, sample_rates_df, sample_features, trained_model_fixture
    ):
        """Cascades to DataFrame validation (Line 415)."""
        with pytest.raises(ValueError, match="Sales DataFrame cannot be None"):
            validate_rate_adjustments_inputs(
                sales_df=None,
                rates_df=sample_rates_df,
                rate_scenarios=np.array([4.0]),
                baseline_predictions=np.array([150]),
                trained_model=trained_model_fixture,
                features=sample_features,
                competitor_rate_adjustment=0.25,
                sales_multiplier=13.0,
                momentum_lookback_periods=3
            )

    def test_cascades_array_validation(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_fixture
    ):
        """Cascades to array validation (Line 416)."""
        with pytest.raises(ValueError, match="Rate scenarios array cannot be None"):
            validate_rate_adjustments_inputs(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                rate_scenarios=None,
                baseline_predictions=np.array([150]),
                trained_model=trained_model_fixture,
                features=sample_features,
                competitor_rate_adjustment=0.25,
                sales_multiplier=13.0,
                momentum_lookback_periods=3
            )

    def test_cascades_model_validation(
        self, sample_sales_df, sample_rates_df, sample_features
    ):
        """Cascades to model validation (Line 417)."""
        with pytest.raises(ValueError, match="Trained model cannot be None"):
            validate_rate_adjustments_inputs(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                rate_scenarios=np.array([4.0]),
                baseline_predictions=np.array([150]),
                trained_model=None,
                features=sample_features,
                competitor_rate_adjustment=0.25,
                sales_multiplier=13.0,
                momentum_lookback_periods=3
            )

    def test_cascades_features_validation(
        self, sample_sales_df, sample_rates_df, trained_model_fixture
    ):
        """Cascades to features validation (Line 418)."""
        with pytest.raises(ValueError, match="Features list cannot be None"):
            validate_rate_adjustments_inputs(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                rate_scenarios=np.array([4.0]),
                baseline_predictions=np.array([150]),
                trained_model=trained_model_fixture,
                features=None,
                competitor_rate_adjustment=0.25,
                sales_multiplier=13.0,
                momentum_lookback_periods=3
            )

    def test_cascades_parameter_validation(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_fixture
    ):
        """Cascades to parameter validation (Lines 419-421)."""
        with pytest.raises(ValueError, match="sales_multiplier must be positive"):
            validate_rate_adjustments_inputs(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                rate_scenarios=np.array([4.0]),
                baseline_predictions=np.array([150]),
                trained_model=trained_model_fixture,
                features=sample_features,
                competitor_rate_adjustment=0.25,
                sales_multiplier=-1.0,  # Invalid
                momentum_lookback_periods=3
            )


# =============================================================================
# TESTS: validate_confidence_interval_inputs (Lines 451-468)
# =============================================================================


class TestValidateConfidenceIntervalInputs:
    """Tests for validate_confidence_interval_inputs function."""

    def test_valid_inputs_pass(self):
        """Valid inputs pass validation."""
        bootstrap_results = pd.DataFrame({
            3.5: [100, 110, 120],
            4.0: [150, 160, 170],
            4.5: [200, 210, 220],
        })
        rate_scenarios = np.array([3.5, 4.0, 4.5])

        validate_confidence_interval_inputs(
            bootstrap_results=bootstrap_results,
            rate_scenarios=rate_scenarios,
            confidence_level=0.95,
            rounding_precision=3,
            basis_points_multiplier=100
        )

    def test_empty_bootstrap_raises(self):
        """Empty bootstrap results raises ValueError (Line 452)."""
        with pytest.raises(ValueError, match="Bootstrap results DataFrame cannot be empty"):
            validate_confidence_interval_inputs(
                bootstrap_results=pd.DataFrame(),
                rate_scenarios=np.array([4.0]),
                confidence_level=0.95,
                rounding_precision=3,
                basis_points_multiplier=100
            )

    def test_empty_scenarios_raises(self):
        """Empty rate scenarios raises ValueError (Line 455)."""
        with pytest.raises(ValueError, match="Rate scenarios array cannot be empty"):
            validate_confidence_interval_inputs(
                bootstrap_results=pd.DataFrame({4.0: [100, 110]}),
                rate_scenarios=np.array([]),
                confidence_level=0.95,
                rounding_precision=3,
                basis_points_multiplier=100
            )

    def test_invalid_confidence_level_raises(self):
        """Invalid confidence level raises ValueError (Line 458)."""
        bootstrap_results = pd.DataFrame({4.0: [100, 110]})

        with pytest.raises(ValueError, match="Confidence level must be in"):
            validate_confidence_interval_inputs(
                bootstrap_results=bootstrap_results,
                rate_scenarios=np.array([4.0]),
                confidence_level=0.0,  # Invalid
                rounding_precision=3,
                basis_points_multiplier=100
            )

        with pytest.raises(ValueError, match="Confidence level must be in"):
            validate_confidence_interval_inputs(
                bootstrap_results=bootstrap_results,
                rate_scenarios=np.array([4.0]),
                confidence_level=1.5,  # Invalid
                rounding_precision=3,
                basis_points_multiplier=100
            )

    def test_negative_rounding_raises(self):
        """Negative rounding precision raises ValueError (Line 461)."""
        with pytest.raises(ValueError, match="Rounding precision must be non-negative"):
            validate_confidence_interval_inputs(
                bootstrap_results=pd.DataFrame({4.0: [100, 110]}),
                rate_scenarios=np.array([4.0]),
                confidence_level=0.95,
                rounding_precision=-1,  # Invalid
                basis_points_multiplier=100
            )

    def test_invalid_basis_points_raises(self):
        """Invalid basis points multiplier raises ValueError (Line 464)."""
        with pytest.raises(ValueError, match="Basis points multiplier must be positive"):
            validate_confidence_interval_inputs(
                bootstrap_results=pd.DataFrame({4.0: [100, 110]}),
                rate_scenarios=np.array([4.0]),
                confidence_level=0.95,
                rounding_precision=3,
                basis_points_multiplier=0  # Invalid
            )

    def test_dimension_mismatch_raises(self):
        """Dimension mismatch raises ValueError (Lines 467-470)."""
        bootstrap_results = pd.DataFrame({
            3.5: [100, 110],
            4.0: [150, 160],
            4.5: [200, 210],
        })
        # 2 scenarios but 3 columns
        rate_scenarios = np.array([3.5, 4.0])

        with pytest.raises(ValueError, match="Rate scenarios.*and bootstrap columns.*must match"):
            validate_confidence_interval_inputs(
                bootstrap_results=bootstrap_results,
                rate_scenarios=rate_scenarios,
                confidence_level=0.95,
                rounding_precision=3,
                basis_points_multiplier=100
            )


# =============================================================================
# TESTS: validate_melt_dataframe_inputs (Lines 507-532)
# =============================================================================


class TestValidateMeltDataframeInputs:
    """Tests for validate_melt_dataframe_inputs function."""

    def test_valid_inputs_pass(self, sample_ci_df, sample_sales_df, sample_features):
        """Valid inputs pass validation."""
        validate_melt_dataframe_inputs(
            df_ci=sample_ci_df,
            current_date_of_mature_data="2024-05-01",
            df=sample_sales_df,
            features=sample_features,
            scenarios_per_basis_point=5,
            scenarios_per_percent=50.0,
            baseline_rate=4.5
        )

    def test_none_ci_df_raises(self, sample_sales_df, sample_features):
        """None CI DataFrame raises ValueError (Line 508)."""
        with pytest.raises(ValueError, match="Confidence interval DataFrame cannot be None or empty"):
            validate_melt_dataframe_inputs(
                df_ci=None,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_empty_ci_df_raises(self, sample_sales_df, sample_features):
        """Empty CI DataFrame raises ValueError (Line 508)."""
        with pytest.raises(ValueError, match="Confidence interval DataFrame cannot be None or empty"):
            validate_melt_dataframe_inputs(
                df_ci=pd.DataFrame(),
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_missing_required_columns_raises(self, sample_sales_df, sample_features):
        """Missing required columns raises ValueError (Lines 512-513)."""
        bad_ci_df = pd.DataFrame({"rate_change_in_basis_points": [25, 50]})

        with pytest.raises(ValueError, match="Missing required columns in CI DataFrame"):
            validate_melt_dataframe_inputs(
                df_ci=bad_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_invalid_date_raises(self, sample_ci_df, sample_sales_df, sample_features):
        """Invalid date format raises ValueError (Line 515)."""
        with pytest.raises(ValueError, match="Invalid date format"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="05-01-2024",  # Wrong format
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_none_source_df_raises(self, sample_ci_df, sample_features):
        """None source DataFrame raises ValueError (Line 518)."""
        with pytest.raises(ValueError, match="Source sales DataFrame cannot be None or empty"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=None,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_empty_features_raises(self, sample_ci_df, sample_sales_df):
        """Empty features raises ValueError (Line 521)."""
        with pytest.raises(ValueError, match="Features list cannot be None or empty"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=[],
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_invalid_scenarios_per_bp_raises(self, sample_ci_df, sample_sales_df, sample_features):
        """Invalid scenarios_per_basis_point raises ValueError (Lines 524-526)."""
        with pytest.raises(ValueError, match="scenarios_per_basis_point must be positive integer"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=0,  # Invalid
                scenarios_per_percent=50.0,
                baseline_rate=4.5
            )

    def test_invalid_scenarios_per_pct_raises(self, sample_ci_df, sample_sales_df, sample_features):
        """Invalid scenarios_per_percent raises ValueError (Line 529)."""
        with pytest.raises(ValueError, match="scenarios_per_percent must be positive"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=-1.0,  # Invalid
                baseline_rate=4.5
            )

    def test_invalid_baseline_rate_raises(self, sample_ci_df, sample_sales_df, sample_features):
        """Non-numeric baseline_rate raises ValueError (Line 532)."""
        with pytest.raises(ValueError, match="baseline_rate must be numeric"):
            validate_melt_dataframe_inputs(
                df_ci=sample_ci_df,
                current_date_of_mature_data="2024-05-01",
                df=sample_sales_df,
                features=sample_features,
                scenarios_per_basis_point=5,
                scenarios_per_percent=50.0,
                baseline_rate="invalid"  # Non-numeric
            )
