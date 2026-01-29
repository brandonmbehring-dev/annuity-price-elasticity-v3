"""
Unit tests for src/config/forecasting_config.py.

Tests validate TypedDict configurations for forecasting operations.
"""

import pytest
from typing import get_type_hints

from src.config.forecasting_config import (
    # Bootstrap Forecasting
    ForecastingConfig,
    CrossValidationConfig,
    BootstrapModelConfig,
    BenchmarkModelConfig,
    PerformanceMonitoringConfig,
    ValidationFrameworkConfig,
    # Inference Configuration
    ForecastingInferenceConfig,
    InferenceConfig,
    RateScenarioConfig,
    ConfidenceIntervalConfig,
    TableauFormattingConfig,
    ProductMetadataConfig,
)


class TestForecastingConfig:
    """Tests for ForecastingConfig TypedDict."""

    def test_can_create_forecasting_config(self):
        """ForecastingConfig should be creatable with all fields."""
        config: ForecastingConfig = {
            'n_bootstrap_samples': 100,
            'ridge_alpha': 1.0,
            'random_state': 42,
            'exclude_holidays': True,
            'mature_data_cutoff_days': 90,
            'min_training_cutoff': 30,
        }
        assert config['n_bootstrap_samples'] == 100
        assert config['random_state'] == 42

    def test_forecasting_config_defaults(self):
        """ForecastingConfig should work with expected default-like values."""
        config: ForecastingConfig = {
            'n_bootstrap_samples': 100,
            'ridge_alpha': 1.0,
            'random_state': 42,
            'exclude_holidays': True,
            'mature_data_cutoff_days': 90,
            'min_training_cutoff': 30,
        }
        # Verify sensible defaults
        assert config['n_bootstrap_samples'] >= 100
        assert config['ridge_alpha'] > 0


class TestCrossValidationConfig:
    """Tests for CrossValidationConfig TypedDict."""

    def test_can_create_cv_config(self):
        """CrossValidationConfig should be creatable."""
        config: CrossValidationConfig = {
            'start_cutoff': 30,
            'end_cutoff': None,
            'validation_method': 'expanding_window',
            'n_splits': 5,
        }
        assert config['validation_method'] == 'expanding_window'
        assert config['n_splits'] == 5


class TestBootstrapModelConfig:
    """Tests for BootstrapModelConfig TypedDict."""

    def test_can_create_bootstrap_model_config(self):
        """BootstrapModelConfig should be creatable."""
        config: BootstrapModelConfig = {
            'estimator_type': 'Ridge',
            'alpha': 1.0,
            'positive_constraint': True,
            'fit_intercept': True,
            'n_estimators': 100,
            'normalize': False,
        }
        assert config['estimator_type'] == 'Ridge'
        assert config['positive_constraint'] is True


class TestBenchmarkModelConfig:
    """Tests for BenchmarkModelConfig TypedDict."""

    def test_can_create_benchmark_config(self):
        """BenchmarkModelConfig should be creatable."""
        config: BenchmarkModelConfig = {
            'method': 'rolling_average',
            'window_size': 7,
            'seasonal_period': None,
        }
        assert config['method'] == 'rolling_average'


class TestValidationFrameworkConfig:
    """Tests for ValidationFrameworkConfig TypedDict."""

    def test_can_create_validation_framework_config(self):
        """ValidationFrameworkConfig should be creatable."""
        config: ValidationFrameworkConfig = {
            'tolerance_r2': 1e-10,
            'tolerance_mape': 1e-10,
            'tolerance_prediction': 1e-10,
            'enable_detailed_comparison': True,
            'reference_results_path': 'results/reference.parquet',
        }
        assert config['tolerance_r2'] == 1e-10


class TestForecastingInferenceConfig:
    """Tests for ForecastingInferenceConfig TypedDict."""

    def test_can_create_inference_config(self):
        """ForecastingInferenceConfig should be creatable."""
        config: ForecastingInferenceConfig = {
            'n_estimators': 100,
            'weight_decay_factor': 0.95,
            'random_state': 42,
            'ridge_alpha': 1.0,
            'sales_multiplier': 1.0,
            'momentum_lookback_periods': 4,
            'training_cutoff_days': 90,
            'target_variable': 'sales_target_current',
            'features': ['prudential_rate_current', 'competitor_mid_lag1'],
            'exclude_zero_sales': True,
            'date_filter_start': '2021-01-01',
        }
        assert config['n_estimators'] == 100
        assert config['target_variable'] == 'sales_target_current'

    def test_inference_config_is_alias(self):
        """InferenceConfig should be alias for ForecastingInferenceConfig."""
        assert InferenceConfig is ForecastingInferenceConfig


class TestRateScenarioConfig:
    """Tests for RateScenarioConfig TypedDict."""

    def test_can_create_rate_scenario_config(self):
        """RateScenarioConfig should be creatable."""
        config: RateScenarioConfig = {
            'rate_min': 3.0,
            'rate_max': 7.0,
            'rate_steps': 41,
            'competitor_rate_adjustment': 0.0,
        }
        assert config['rate_min'] == 3.0
        assert config['rate_steps'] == 41


class TestConfidenceIntervalConfig:
    """Tests for ConfidenceIntervalConfig TypedDict."""

    def test_can_create_ci_config(self):
        """ConfidenceIntervalConfig should be creatable."""
        config: ConfidenceIntervalConfig = {
            'confidence_level': 0.95,
            'rounding_precision': 4,
            'basis_points_multiplier': 100,
        }
        assert config['confidence_level'] == 0.95


class TestTableauFormattingConfig:
    """Tests for TableauFormattingConfig TypedDict."""

    def test_can_create_tableau_config(self):
        """TableauFormattingConfig should be creatable."""
        config: TableauFormattingConfig = {
            'prudential_rate_col': 'prudential_rate_current',
            'competitor_rate_col': 'competitor_mid_lag1',
            'sales_lag_cols': ['sales_target_lag1', 'sales_target_lag2'],
            'sales_rounding_power': 4,
        }
        assert config['prudential_rate_col'] == 'prudential_rate_current'


class TestProductMetadataConfig:
    """Tests for ProductMetadataConfig TypedDict."""

    def test_can_create_product_metadata_config(self):
        """ProductMetadataConfig should be creatable."""
        config: ProductMetadataConfig = {
            'product_name': 'FlexGuard_6Y20B',
            'version': 'v2_1',
            'flexguard_product_ids': {'Prudential': [2979]},
            'product_name_dict': {'Prudential': 'Prudential FlexGuard 6Y20B'},
        }
        assert config['product_name'] == 'FlexGuard_6Y20B'


class TestConfigTypeHints:
    """Tests verifying TypedDict configurations have proper type hints."""

    def test_forecasting_config_has_type_hints(self):
        """ForecastingConfig should have type hints for all fields."""
        hints = get_type_hints(ForecastingConfig)
        expected = ['n_bootstrap_samples', 'ridge_alpha', 'random_state']
        for field in expected:
            assert field in hints

    def test_forecasting_inference_config_has_type_hints(self):
        """ForecastingInferenceConfig should have type hints for all fields."""
        hints = get_type_hints(ForecastingInferenceConfig)
        expected = ['n_estimators', 'target_variable', 'features']
        for field in expected:
            assert field in hints
