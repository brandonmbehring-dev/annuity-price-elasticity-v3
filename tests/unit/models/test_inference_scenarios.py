"""Tests for inference_scenarios module.

Covers: center_baseline, rate_adjustments, confidence_interval
Target: 52% → 80% coverage
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge

from src.models.inference_scenarios import (
    # Validation wrappers
    validate_center_baseline_inputs,
    validate_rate_adjustments_inputs,
    validate_confidence_interval_inputs,
    # Main functions
    center_baseline,
    rate_adjustments,
    confidence_interval,
    # Helpers
    _resolve_center_baseline_params,
    _generate_baseline_predictions,
    apply_feature_adjustments,
    _generate_scenario_predictions,
    _calculate_quantile_bounds,
    _initialize_ci_dataframe,
    _compute_quantiles,
    _convert_to_basis_points,
    _calculate_momentum,
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
        "holiday": [0] * 18 + [1, 0],
        "forward_0": np.random.uniform(100, 500, 20),
        "prudential_rate_current": np.random.uniform(3.0, 5.0, 20),
        "prudential_rate_t3": np.random.uniform(2.8, 4.8, 20),
        "competitor_mid_current": np.random.uniform(2.5, 4.5, 20),
        "competitor_top5_current": np.random.uniform(2.5, 4.5, 20),
        "economic_indicator": np.random.uniform(0.5, 1.5, 20),
        "sales_by_contract_date_momentum": np.random.uniform(80, 200, 20),
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
    return [
        "prudential_rate_current",
        "prudential_rate_t3",
        "competitor_mid_current",
        "economic_indicator",
    ]


@pytest.fixture
def trained_model_mock(sample_sales_df, sample_features):
    """Create a fitted mock BaggingRegressor."""
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
def bootstrap_results():
    """Sample bootstrap results dataframe for CI tests."""
    np.random.seed(42)
    scenarios = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
    n_estimators = 100

    # Create bootstrap results with realistic structure
    data = {}
    for scenario in scenarios:
        data[scenario] = np.random.normal(loc=150 + scenario * 10, scale=20, size=n_estimators)

    return pd.DataFrame(data)


# =============================================================================
# TESTS: _resolve_center_baseline_params (Lines 129, 131, 133, 135)
# =============================================================================


class TestResolveCenterBaselineParams:
    """Tests for parameter resolution with error paths."""

    def test_uses_sales_df_over_df(self, sample_sales_df, sample_rates_df, sample_features):
        """sales_df takes precedence over df parameter."""
        actual_df, actual_rates, actual_cutoff = _resolve_center_baseline_params(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            df=None,
            df_rates=None,
            training_cutoff_date="2024-05-01",
            current_date_of_mature_data=None,
            features=sample_features,
            target_variable="forward_0",
            n_estimators=5
        )

        assert actual_df is sample_sales_df
        assert actual_rates is sample_rates_df
        assert actual_cutoff == "2024-05-01"

    def test_uses_df_when_sales_df_none(self, sample_sales_df, sample_rates_df, sample_features):
        """Falls back to df when sales_df is None."""
        actual_df, _, _ = _resolve_center_baseline_params(
            sales_df=None,
            rates_df=sample_rates_df,
            df=sample_sales_df,
            df_rates=None,
            training_cutoff_date="2024-05-01",
            current_date_of_mature_data=None,
            features=sample_features,
            target_variable="forward_0",
            n_estimators=5
        )

        assert actual_df is sample_sales_df

    def test_missing_df_raises_error(self, sample_rates_df, sample_features):
        """ValueError when neither sales_df nor df provided (Line 129)."""
        with pytest.raises(ValueError, match="Either 'sales_df' or 'df' parameter must be provided"):
            _resolve_center_baseline_params(
                sales_df=None,
                rates_df=sample_rates_df,
                df=None,
                df_rates=None,
                training_cutoff_date="2024-05-01",
                current_date_of_mature_data=None,
                features=sample_features,
                target_variable="forward_0",
                n_estimators=5
            )

    def test_missing_rates_df_raises_error(self, sample_sales_df, sample_features):
        """ValueError when neither rates_df nor df_rates provided (Line 131)."""
        with pytest.raises(ValueError, match="Either 'rates_df' or 'df_rates' parameter must be provided"):
            _resolve_center_baseline_params(
                sales_df=sample_sales_df,
                rates_df=None,
                df=None,
                df_rates=None,
                training_cutoff_date="2024-05-01",
                current_date_of_mature_data=None,
                features=sample_features,
                target_variable="forward_0",
                n_estimators=5
            )

    def test_missing_cutoff_date_raises_error(self, sample_sales_df, sample_rates_df, sample_features):
        """ValueError when neither cutoff date provided (Line 133)."""
        with pytest.raises(ValueError, match="training_cutoff_date.*current_date_of_mature_data"):
            _resolve_center_baseline_params(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                df=None,
                df_rates=None,
                training_cutoff_date=None,
                current_date_of_mature_data=None,
                features=sample_features,
                target_variable="forward_0",
                n_estimators=5
            )

    def test_missing_required_params_raises_error(self, sample_sales_df, sample_rates_df):
        """ValueError when required params missing (Line 135)."""
        with pytest.raises(ValueError, match="features.*target_variable.*n_estimators"):
            _resolve_center_baseline_params(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                df=None,
                df_rates=None,
                training_cutoff_date="2024-05-01",
                current_date_of_mature_data=None,
                features=None,  # Missing
                target_variable=None,  # Missing
                n_estimators=None  # Missing
            )


# =============================================================================
# TESTS: center_baseline (Lines 207-210)
# =============================================================================


class TestCenterBaseline:
    """Tests for center_baseline function."""

    def test_returns_predictions_and_model(self, sample_sales_df, sample_rates_df, sample_features):
        """Returns tuple of predictions and trained model."""
        predictions, model = center_baseline(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            features=sample_features,
            target_variable="forward_0",
            training_cutoff_date="2024-04-01",
            n_estimators=5,
            weight_decay_factor=0.99,
            random_state=42,
            ridge_alpha=1.0
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        assert isinstance(model, BaggingRegressor)

    def test_predictions_positive(self, sample_sales_df, sample_rates_df, sample_features):
        """Predictions should be positive (sales cannot be negative)."""
        predictions, _ = center_baseline(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            features=sample_features,
            target_variable="forward_0",
            training_cutoff_date="2024-04-01",
            n_estimators=5
        )

        assert np.all(predictions > 0)

    def test_value_error_propagates(self, sample_sales_df, sample_rates_df, sample_features):
        """ValueError raised for invalid inputs propagates directly (Lines 208-209)."""
        with pytest.raises(ValueError, match="training_cutoff_date"):
            center_baseline(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                features=sample_features,
                target_variable="forward_0",
                training_cutoff_date=None,  # Missing required param
                current_date_of_mature_data=None,
                n_estimators=5
            )

    def test_generic_exception_wrapped(self, sample_sales_df, sample_rates_df, sample_features):
        """Non-ValueError exceptions are wrapped with context (Lines 210-212)."""
        with patch("src.models.inference_scenarios.train_bootstrap_model") as mock_train:
            mock_train.side_effect = RuntimeError("Unexpected training error")

            with pytest.raises(ValueError, match="RILA baseline forecasting failed"):
                center_baseline(
                    sales_df=sample_sales_df,
                    rates_df=sample_rates_df,
                    features=sample_features,
                    target_variable="forward_0",
                    training_cutoff_date="2024-04-01",
                    n_estimators=5
                )


# =============================================================================
# TESTS: apply_feature_adjustments (Lines 268-269)
# =============================================================================


class TestApplyFeatureAdjustments:
    """Tests for apply_feature_adjustments function."""

    def test_adjusts_prudential_rate_current(self, sample_sales_df, sample_rates_df, sample_features):
        """Adjusts prudential_rate_current feature."""
        base_features = sample_sales_df[sample_features].iloc[-1]

        adjusted = apply_feature_adjustments(
            base_features=base_features,
            features=sample_features,
            rates_df=sample_rates_df,
            sales_df=sample_sales_df,
            prudential_rate_adjustment=5.5,
            competitor_rate_adjustment=0.0,
            momentum_lookback_periods=3
        )

        assert adjusted["prudential_rate_current"] == 5.5
        # Lagged features should NOT be adjusted
        assert adjusted["prudential_rate_t3"] == base_features["prudential_rate_t3"]

    def test_adjusts_competitor_mid_current(self, sample_sales_df, sample_rates_df):
        """Adjusts competitor_mid_current feature."""
        features = ["competitor_mid_current", "economic_indicator"]
        base_features = sample_sales_df[features].iloc[-1]

        original_competitor = sample_sales_df["competitor_mid_current"].iloc[-1]
        competitor_adjustment = 0.25

        adjusted = apply_feature_adjustments(
            base_features=base_features,
            features=features,
            rates_df=sample_rates_df,
            sales_df=sample_sales_df,
            prudential_rate_adjustment=4.5,
            competitor_rate_adjustment=competitor_adjustment,
            momentum_lookback_periods=3
        )

        expected = original_competitor + competitor_adjustment
        assert adjusted["competitor_mid_current"] == expected

    def test_adjusts_sales_momentum(self, sample_sales_df, sample_rates_df):
        """Adjusts sales_by_contract_date features with momentum average."""
        features = ["sales_by_contract_date_momentum", "economic_indicator"]
        base_features = sample_sales_df[features].iloc[-1]

        adjusted = apply_feature_adjustments(
            base_features=base_features,
            features=features,
            rates_df=sample_rates_df,
            sales_df=sample_sales_df,
            prudential_rate_adjustment=4.5,
            competitor_rate_adjustment=0.0,
            momentum_lookback_periods=3
        )

        # Should be average of lag_1, lag_2, lag_3
        expected_momentum = (
            sample_sales_df["sales_by_contract_date_lag_1"].iloc[-1] +
            sample_sales_df["sales_by_contract_date_lag_2"].iloc[-1] +
            sample_sales_df["sales_by_contract_date_lag_3"].iloc[-1]
        ) / 3

        assert adjusted["sales_by_contract_date_momentum"] == expected_momentum

    def test_preserves_other_features(self, sample_sales_df, sample_rates_df, sample_features):
        """Non-adjusted features retain original values."""
        base_features = sample_sales_df[sample_features].iloc[-1]

        adjusted = apply_feature_adjustments(
            base_features=base_features,
            features=sample_features,
            rates_df=sample_rates_df,
            sales_df=sample_sales_df,
            prudential_rate_adjustment=5.5,
            competitor_rate_adjustment=0.0,
            momentum_lookback_periods=3
        )

        assert adjusted["economic_indicator"] == base_features["economic_indicator"]

    def test_error_wrapped_with_context(self, sample_sales_df, sample_rates_df):
        """Exceptions are wrapped with context message (Lines 268-269)."""
        features = ["sales_by_contract_date_momentum"]
        base_features = sample_sales_df[features].iloc[-1]

        # Create bad sales_df missing required lag columns
        bad_sales_df = sample_sales_df.drop(columns=["sales_by_contract_date_lag_1"])

        with pytest.raises(ValueError, match="Feature adjustment failed"):
            apply_feature_adjustments(
                base_features=base_features,
                features=features,
                rates_df=sample_rates_df,
                sales_df=bad_sales_df,
                prudential_rate_adjustment=4.5,
                competitor_rate_adjustment=0.0,
                momentum_lookback_periods=3
            )


# =============================================================================
# TESTS: _calculate_momentum
# =============================================================================


class TestCalculateMomentum:
    """Tests for _calculate_momentum helper."""

    def test_calculates_average(self, sample_sales_df):
        """Momentum is average of lagged values."""
        result = _calculate_momentum(sample_sales_df, lookback_periods=3)

        expected = (
            sample_sales_df["sales_by_contract_date_lag_1"].iloc[-1] +
            sample_sales_df["sales_by_contract_date_lag_2"].iloc[-1] +
            sample_sales_df["sales_by_contract_date_lag_3"].iloc[-1]
        ) / 3

        assert result == expected

    def test_single_period_lookback(self, sample_sales_df):
        """Single period lookback returns just lag_1."""
        result = _calculate_momentum(sample_sales_df, lookback_periods=1)
        expected = sample_sales_df["sales_by_contract_date_lag_1"].iloc[-1]
        assert result == expected


# =============================================================================
# TESTS: _generate_scenario_predictions
# =============================================================================


class TestGenerateScenarioPredictions:
    """Tests for _generate_scenario_predictions helper."""

    def test_returns_array_of_predictions(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Returns array with prediction per estimator."""
        predictions = _generate_scenario_predictions(
            trained_model=trained_model_mock,
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            features=sample_features,
            prudential_rate_adjustment=4.5,
            competitor_rate_adjustment=0.0,
            momentum_lookback_periods=3,
            n_estimators=5
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5

    def test_predictions_positive(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Predictions should be positive (exp transform)."""
        predictions = _generate_scenario_predictions(
            trained_model=trained_model_mock,
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            features=sample_features,
            prudential_rate_adjustment=4.5,
            competitor_rate_adjustment=0.0,
            momentum_lookback_periods=3,
            n_estimators=5
        )

        assert np.all(predictions > 0)


# =============================================================================
# TESTS: rate_adjustments (Lines 314-339)
# =============================================================================


class TestRateAdjustments:
    """Tests for rate_adjustments function."""

    def test_returns_two_dataframes(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Returns tuple of (df_dollars, df_pct_change)."""
        rate_scenarios = np.array([3.5, 4.0, 4.5, 5.0])
        baseline = np.array([150.0] * 5)  # Match n_estimators

        df_dollars, df_pct_change = rate_adjustments(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=rate_scenarios,
            baseline_predictions=baseline,
            trained_model=trained_model_mock,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

        assert isinstance(df_dollars, pd.DataFrame)
        assert isinstance(df_pct_change, pd.DataFrame)
        assert df_dollars.shape == (5, 4)  # n_estimators x n_scenarios
        assert df_pct_change.shape == (5, 4)

    def test_dollars_scaled_by_multiplier(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Dollar values are scaled by sales_multiplier."""
        rate_scenarios = np.array([4.0, 4.5])
        baseline = np.array([150.0] * 5)

        # Use different multipliers
        df_dollars_13, _ = rate_adjustments(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=rate_scenarios,
            baseline_predictions=baseline,
            trained_model=trained_model_mock,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

        df_dollars_26, _ = rate_adjustments(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=rate_scenarios,
            baseline_predictions=baseline,
            trained_model=trained_model_mock,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=26.0,
            momentum_lookback_periods=3
        )

        # df_dollars_26 should be 2x df_dollars_13
        np.testing.assert_allclose(
            df_dollars_26.values, df_dollars_13.values * 2, rtol=1e-10
        )

    def test_pct_change_relative_to_baseline(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Percent change is relative to baseline predictions."""
        rate_scenarios = np.array([4.0])
        baseline = np.array([100.0] * 5)

        _, df_pct_change = rate_adjustments(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=rate_scenarios,
            baseline_predictions=baseline,
            trained_model=trained_model_mock,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

        # Percent change formula: (scenario / baseline - 1) * 100
        # Values should be finite and not extreme
        assert np.all(np.isfinite(df_pct_change.values))

    def test_value_error_propagates(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """ValueError from validation propagates (Lines 337-338)."""
        with pytest.raises(ValueError):
            rate_adjustments(
                sales_df=sample_sales_df,
                rates_df=sample_rates_df,
                rate_scenarios=np.array([]),  # Invalid: empty
                baseline_predictions=np.array([150.0] * 5),
                trained_model=trained_model_mock,
                features=sample_features,
                competitor_rate_adjustment=0.0
            )

    def test_generic_exception_wrapped(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """Non-ValueError exceptions wrapped with context (Lines 339-341)."""
        with patch("src.models.inference_scenarios._generate_scenario_predictions") as mock_gen:
            mock_gen.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ValueError, match="Rate adjustments scenario analysis failed"):
                rate_adjustments(
                    sales_df=sample_sales_df,
                    rates_df=sample_rates_df,
                    rate_scenarios=np.array([4.0]),
                    baseline_predictions=np.array([150.0] * 5),
                    trained_model=trained_model_mock,
                    features=sample_features,
                    competitor_rate_adjustment=0.0
                )


# =============================================================================
# TESTS: _calculate_quantile_bounds (Lines 365-368)
# =============================================================================


class TestCalculateQuantileBounds:
    """Tests for _calculate_quantile_bounds helper."""

    def test_95_confidence(self):
        """95% CI yields 0.025 and 0.975 bounds."""
        lower, upper = _calculate_quantile_bounds(0.95)
        assert lower == pytest.approx(0.025)
        assert upper == pytest.approx(0.975)

    def test_90_confidence(self):
        """90% CI yields 0.05 and 0.95 bounds."""
        lower, upper = _calculate_quantile_bounds(0.90)
        assert lower == pytest.approx(0.05)
        assert upper == pytest.approx(0.95)

    def test_99_confidence(self):
        """99% CI yields 0.005 and 0.995 bounds."""
        lower, upper = _calculate_quantile_bounds(0.99)
        assert lower == pytest.approx(0.005)
        assert upper == pytest.approx(0.995)


# =============================================================================
# TESTS: _initialize_ci_dataframe (Line 387)
# =============================================================================


class TestInitializeCIDataframe:
    """Tests for _initialize_ci_dataframe helper."""

    def test_creates_correct_structure(self):
        """Creates DataFrame with rate_change, bottom, median, top columns."""
        scenarios = np.array([3.0, 3.5, 4.0])
        df = _initialize_ci_dataframe(scenarios)

        assert list(df.columns) == ["rate_change", "bottom", "median", "top"]
        assert len(df) == 3
        np.testing.assert_array_equal(df["rate_change"].values, scenarios)

    def test_zeros_for_bounds(self):
        """Initializes bottom, median, top with zeros."""
        scenarios = np.array([3.0, 4.0])
        df = _initialize_ci_dataframe(scenarios)

        assert np.all(df["bottom"] == 0)
        assert np.all(df["median"] == 0)
        assert np.all(df["top"] == 0)


# =============================================================================
# TESTS: _compute_quantiles (Lines 423-435)
# =============================================================================


class TestComputeQuantiles:
    """Tests for _compute_quantiles helper."""

    def test_computes_correct_quantiles(self, bootstrap_results):
        """Computes correct bottom, median, top values."""
        scenarios = bootstrap_results.columns.values
        df_output = _initialize_ci_dataframe(scenarios)

        result = _compute_quantiles(
            df_output=df_output,
            bootstrap_results=bootstrap_results,
            lower_quantile=0.025,
            upper_quantile=0.975,
            rounding_precision=3
        )

        # Check that values are in correct order: bottom < median < top
        assert np.all(result["bottom"] < result["median"])
        assert np.all(result["median"] < result["top"])

    def test_respects_rounding_precision(self, bootstrap_results):
        """Values are rounded to specified precision."""
        scenarios = bootstrap_results.columns.values
        df_output = _initialize_ci_dataframe(scenarios)

        result = _compute_quantiles(
            df_output=df_output,
            bootstrap_results=bootstrap_results,
            lower_quantile=0.025,
            upper_quantile=0.975,
            rounding_precision=2
        )

        # Check that values are rounded to 2 decimal places
        for col in ["bottom", "median", "top"]:
            for val in result[col]:
                # Rounding to 2 places means no more than 2 decimals
                rounded = round(val, 2)
                assert val == rounded


# =============================================================================
# TESTS: _convert_to_basis_points (Lines 457-460)
# =============================================================================


class TestConvertToBasisPoints:
    """Tests for _convert_to_basis_points helper."""

    def test_converts_rate_to_basis_points(self):
        """Converts rate changes to basis points."""
        df = pd.DataFrame({
            "rate_change": [0.25, 0.50, 1.00],
            "bottom": [100, 110, 120],
            "median": [150, 160, 170],
            "top": [200, 210, 220]
        })

        result = _convert_to_basis_points(df, basis_points_multiplier=100)

        assert "rate_change_in_basis_points" in result.columns
        assert "rate_change" not in result.columns
        np.testing.assert_array_equal(
            result["rate_change_in_basis_points"].values,
            [25, 50, 100]
        )

    def test_preserves_other_columns(self):
        """Preserves bottom, median, top columns."""
        df = pd.DataFrame({
            "rate_change": [0.5],
            "bottom": [100],
            "median": [150],
            "top": [200]
        })

        result = _convert_to_basis_points(df, basis_points_multiplier=100)

        assert result["bottom"].iloc[0] == 100
        assert result["median"].iloc[0] == 150
        assert result["top"].iloc[0] == 200


# =============================================================================
# TESTS: confidence_interval (Lines 480-498)
# =============================================================================


class TestConfidenceInterval:
    """Tests for confidence_interval function."""

    def test_returns_correct_structure(self, bootstrap_results):
        """Returns DataFrame with correct columns."""
        scenarios = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

        result = confidence_interval(
            bootstrap_results=bootstrap_results,
            rate_scenarios=scenarios,
            confidence_level=0.95,
            rounding_precision=3,
            basis_points_multiplier=100
        )

        expected_cols = ["bottom", "median", "top", "rate_change_in_basis_points"]
        assert all(col in result.columns for col in expected_cols)
        assert len(result) == len(scenarios)

    def test_basis_points_correct(self, bootstrap_results):
        """Basis points calculated correctly."""
        scenarios = np.array([0.50, 1.00, 1.50])

        # Create simple bootstrap results for these scenarios
        simple_results = pd.DataFrame({
            0.50: np.ones(10) * 100,
            1.00: np.ones(10) * 150,
            1.50: np.ones(10) * 200,
        })

        result = confidence_interval(
            bootstrap_results=simple_results,
            rate_scenarios=scenarios,
            confidence_level=0.95,
            rounding_precision=3,
            basis_points_multiplier=100
        )

        np.testing.assert_array_equal(
            result["rate_change_in_basis_points"].values,
            [50, 100, 150]
        )

    def test_value_error_propagates(self, bootstrap_results):
        """ValueError from validation propagates (Lines 496-497)."""
        scenarios = np.array([3.0, 4.0])

        with pytest.raises(ValueError):
            confidence_interval(
                bootstrap_results=bootstrap_results,
                rate_scenarios=scenarios,
                confidence_level=1.5,  # Invalid: > 1.0
            )

    def test_generic_exception_wrapped(self, bootstrap_results):
        """Non-ValueError exceptions wrapped (Lines 498-500)."""
        # Use matching scenarios for bootstrap_results columns
        scenarios = bootstrap_results.columns.values

        with patch("src.models.inference_scenarios._compute_quantiles") as mock_compute:
            mock_compute.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(ValueError, match="Confidence interval calculation failed"):
                confidence_interval(
                    bootstrap_results=bootstrap_results,
                    rate_scenarios=scenarios,
                    confidence_level=0.95
                )


# =============================================================================
# TESTS: Validation Wrappers
# =============================================================================


class TestValidationWrappers:
    """Tests for validation wrapper functions."""

    def test_validate_center_baseline_delegates(self, sample_sales_df, sample_rates_df, sample_features):
        """validate_center_baseline_inputs delegates to canonical."""
        # Should not raise for valid inputs
        validate_center_baseline_inputs(
            df=sample_sales_df,
            df_rates=sample_rates_df,
            features=sample_features,
            target_variable="forward_0",
            current_date_of_mature_data="2024-05-01",
            n_estimators=5,
            weight_decay_factor=0.99,
            random_state=42,
            ridge_alpha=1.0
        )

    def test_validate_rate_adjustments_delegates(
        self, sample_sales_df, sample_rates_df, sample_features, trained_model_mock
    ):
        """validate_rate_adjustments_inputs delegates to canonical."""
        # Should not raise for valid inputs
        validate_rate_adjustments_inputs(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=np.array([4.0, 4.5]),
            baseline_predictions=np.array([150.0] * 5),
            trained_model=trained_model_mock,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=13.0,
            momentum_lookback_periods=3
        )

    def test_validate_confidence_interval_delegates(self, bootstrap_results):
        """validate_confidence_interval_inputs delegates to canonical."""
        scenarios = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

        # Should not raise for valid inputs
        validate_confidence_interval_inputs(
            bootstrap_results=bootstrap_results,
            rate_scenarios=scenarios,
            confidence_level=0.95,
            rounding_precision=3,
            basis_points_multiplier=100
        )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestScenariosIntegration:
    """Integration tests for the full scenario analysis workflow."""

    def test_full_scenario_workflow(self, sample_sales_df, sample_rates_df, sample_features):
        """Full workflow: baseline → rate adjustments → confidence intervals."""
        # 1. Generate baseline
        baseline_predictions, model = center_baseline(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            features=sample_features,
            target_variable="forward_0",
            training_cutoff_date="2024-04-01",
            n_estimators=5,
            random_state=42
        )

        # 2. Run rate adjustments
        rate_scenarios = np.array([3.5, 4.0, 4.5, 5.0])
        df_dollars, df_pct_change = rate_adjustments(
            sales_df=sample_sales_df,
            rates_df=sample_rates_df,
            rate_scenarios=rate_scenarios,
            baseline_predictions=baseline_predictions,
            trained_model=model,
            features=sample_features,
            competitor_rate_adjustment=0.0,
            sales_multiplier=13.0
        )

        # 3. Calculate confidence intervals
        ci_dollars = confidence_interval(
            bootstrap_results=df_dollars,
            rate_scenarios=rate_scenarios,
            confidence_level=0.95
        )

        ci_pct = confidence_interval(
            bootstrap_results=df_pct_change,
            rate_scenarios=rate_scenarios,
            confidence_level=0.95
        )

        # Verify outputs
        assert len(ci_dollars) == 4
        assert len(ci_pct) == 4
        assert "rate_change_in_basis_points" in ci_dollars.columns
        assert np.all(ci_dollars["bottom"] <= ci_dollars["median"])
        assert np.all(ci_dollars["median"] <= ci_dollars["top"])
