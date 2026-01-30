"""
Unit tests for src/config/inference_builders.py

Tests inference configuration, rate scenarios, confidence intervals,
Tableau formatting, product metadata, and stage config builders.
"""

import pytest
from typing import Dict, List


class TestGetDefaultInferenceFeatures:
    """Tests for _get_default_inference_features function."""

    def test_returns_list(self):
        """Returns list of feature names."""
        from src.config.inference_builders import _get_default_inference_features

        features = _get_default_inference_features()
        assert isinstance(features, list)

    def test_contains_required_features(self):
        """Contains required feature names."""
        from src.config.inference_builders import _get_default_inference_features

        features = _get_default_inference_features()
        assert 'competitor_mid_t2' in features
        assert 'competitor_top5_t2' in features
        assert 'prudential_rate_current' in features

    def test_feature_count(self):
        """Returns expected number of features."""
        from src.config.inference_builders import _get_default_inference_features

        features = _get_default_inference_features()
        assert len(features) == 4


class TestBuildInferenceConfig:
    """Tests for build_inference_config function."""

    def test_returns_inference_config_type(self):
        """Returns InferenceConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert isinstance(config, dict)

    def test_default_n_estimators(self):
        """Default n_estimators is 10000."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert config['n_estimators'] == 10000

    def test_custom_n_estimators(self):
        """Custom n_estimators is respected."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config(n_estimators=5000)
        assert config['n_estimators'] == 5000

    def test_default_weight_decay_factor(self):
        """Default weight decay factor is 0.99."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert config['weight_decay_factor'] == 0.99

    def test_default_random_state(self):
        """Default random state is 42."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert config['random_state'] == 42

    def test_default_ridge_alpha(self):
        """Default ridge alpha is 1.0."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert config['ridge_alpha'] == 1.0

    def test_default_sales_multiplier(self):
        """Default sales multiplier is 13.0."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert config['sales_multiplier'] == 13.0

    def test_default_features(self):
        """Default features are set when None provided."""
        from src.config.inference_builders import build_inference_config

        config = build_inference_config()
        assert 'features' in config
        assert len(config['features']) == 4

    def test_custom_features(self):
        """Custom features are respected."""
        from src.config.inference_builders import build_inference_config

        custom = ['feature_a', 'feature_b']
        config = build_inference_config(features=custom)
        assert config['features'] == custom


class TestGetDefaultMetadataProductIds:
    """Tests for _get_default_metadata_product_ids function."""

    def test_returns_dict(self):
        """Returns dict mapping company to product IDs."""
        from src.config.inference_builders import _get_default_metadata_product_ids

        ids = _get_default_metadata_product_ids()
        assert isinstance(ids, dict)

    def test_contains_prudential(self):
        """Contains Prudential product IDs."""
        from src.config.inference_builders import _get_default_metadata_product_ids

        ids = _get_default_metadata_product_ids()
        assert 'Prudential' in ids

    def test_values_are_lists(self):
        """All values are lists of integers."""
        from src.config.inference_builders import _get_default_metadata_product_ids

        ids = _get_default_metadata_product_ids()
        for company, product_ids in ids.items():
            assert isinstance(product_ids, list), f"{company} should have list of IDs"


class TestGetDefaultProductNameDict:
    """Tests for _get_default_product_name_dict function."""

    def test_returns_dict(self):
        """Returns dict mapping company to full product name."""
        from src.config.inference_builders import _get_default_product_name_dict

        names = _get_default_product_name_dict()
        assert isinstance(names, dict)

    def test_contains_prudential(self):
        """Contains Prudential product name."""
        from src.config.inference_builders import _get_default_product_name_dict

        names = _get_default_product_name_dict()
        assert 'Prudential' in names
        assert 'FlexGuard' in names['Prudential'] or 'Flexguard' in names['Prudential']

    def test_contains_weighted_mean(self):
        """Contains weighted mean entry."""
        from src.config.inference_builders import _get_default_product_name_dict

        names = _get_default_product_name_dict()
        assert 'weighted_mean' in names


class TestBuildProductMetadataConfig:
    """Tests for build_product_metadata_config function."""

    def test_returns_product_metadata_config_type(self):
        """Returns ProductMetadataConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config()
        assert isinstance(config, dict)

    def test_default_product_name(self):
        """Default product name is 'FlexGuard_6Y20B'."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config()
        assert config['product_name'] == 'FlexGuard_6Y20B'

    def test_default_version(self):
        """Default version is 'v2_1'."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config()
        assert config['version'] == 'v2_1'

    def test_custom_product_name(self):
        """Custom product name is respected."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config(product_name='CustomProduct')
        assert config['product_name'] == 'CustomProduct'

    def test_includes_product_ids(self):
        """Config includes flexguard_product_ids."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config()
        assert 'flexguard_product_ids' in config

    def test_includes_product_name_dict(self):
        """Config includes product_name_dict."""
        from src.config.inference_builders import build_product_metadata_config

        config = build_product_metadata_config()
        assert 'product_name_dict' in config


class TestBuildRateScenarioConfig:
    """Tests for build_rate_scenario_config function."""

    def test_returns_rate_scenario_config_type(self):
        """Returns RateScenarioConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config()
        assert isinstance(config, dict)

    def test_default_rate_min(self):
        """Default rate min is 0.005 (50 bps)."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config()
        assert config['rate_min'] == 0.005

    def test_default_rate_max(self):
        """Default rate max is 4.5."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config()
        assert config['rate_max'] == 4.5

    def test_default_rate_steps(self):
        """Default rate steps is 19."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config()
        assert config['rate_steps'] == 19

    def test_default_competitor_adjustment(self):
        """Default competitor rate adjustment is 0.0."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config()
        assert config['competitor_rate_adjustment'] == 0.0

    def test_custom_rate_range(self):
        """Custom rate range is respected."""
        from src.config.inference_builders import build_rate_scenario_config

        config = build_rate_scenario_config(rate_min=0.01, rate_max=5.0, rate_steps=25)
        assert config['rate_min'] == 0.01
        assert config['rate_max'] == 5.0
        assert config['rate_steps'] == 25


class TestBuildConfidenceIntervalConfig:
    """Tests for build_confidence_interval_config function."""

    def test_returns_confidence_interval_config_type(self):
        """Returns ConfidenceIntervalConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_confidence_interval_config

        config = build_confidence_interval_config()
        assert isinstance(config, dict)

    def test_default_confidence_level(self):
        """Default confidence level is 0.95."""
        from src.config.inference_builders import build_confidence_interval_config

        config = build_confidence_interval_config()
        assert config['confidence_level'] == 0.95

    def test_default_rounding_precision(self):
        """Default rounding precision is 3."""
        from src.config.inference_builders import build_confidence_interval_config

        config = build_confidence_interval_config()
        assert config['rounding_precision'] == 3

    def test_default_basis_points_multiplier(self):
        """Default basis points multiplier is 100."""
        from src.config.inference_builders import build_confidence_interval_config

        config = build_confidence_interval_config()
        assert config['basis_points_multiplier'] == 100

    def test_custom_confidence_level(self):
        """Custom confidence level is respected."""
        from src.config.inference_builders import build_confidence_interval_config

        config = build_confidence_interval_config(confidence_level=0.99)
        assert config['confidence_level'] == 0.99


class TestBuildTableauFormattingConfig:
    """Tests for build_tableau_formatting_config function."""

    def test_returns_tableau_formatting_config_type(self):
        """Returns TableauFormattingConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_tableau_formatting_config

        config = build_tableau_formatting_config()
        assert isinstance(config, dict)

    def test_default_prudential_rate_col(self):
        """Default prudential rate column is correct."""
        from src.config.inference_builders import build_tableau_formatting_config

        config = build_tableau_formatting_config()
        assert config['prudential_rate_col'] == 'prudential_rate_current'

    def test_default_competitor_rate_col(self):
        """Default competitor rate column is correct."""
        from src.config.inference_builders import build_tableau_formatting_config

        config = build_tableau_formatting_config()
        assert config['competitor_rate_col'] == 'competitor_mid_current'

    def test_default_sales_rounding_power(self):
        """Default sales rounding power is -7 (10M)."""
        from src.config.inference_builders import build_tableau_formatting_config

        config = build_tableau_formatting_config()
        assert config['sales_rounding_power'] == -7

    def test_default_sales_lag_cols(self):
        """Default sales lag columns are set."""
        from src.config.inference_builders import build_tableau_formatting_config

        config = build_tableau_formatting_config()
        assert 'sales_lag_cols' in config
        assert 'sales_target_t2' in config['sales_lag_cols']


class TestBuildCoreInferenceConfigs:
    """Tests for _build_core_inference_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with inference_config and rate_scenario_config."""
        from src.config.inference_builders import _build_core_inference_configs

        configs = _build_core_inference_configs(
            n_estimators=1000,
            weight_decay_factor=0.99,
            random_state=42,
            ridge_alpha=1.0,
            sales_multiplier=13.0,
            momentum_lookback_periods=3,
            rate_min=0.005,
            rate_max=4.5,
            rate_steps=19,
            competitor_rate_adjustment=0.0
        )
        assert 'inference_config' in configs
        assert 'rate_scenario_config' in configs


class TestBuildOutputConfigs:
    """Tests for _build_output_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with CI, Tableau, metadata, and viz configs."""
        from src.config.inference_builders import _build_output_configs

        configs = _build_output_configs(
            confidence_level=0.95,
            rounding_precision=3,
            basis_points_multiplier=100,
            prudential_rate_col='prudential_rate_current',
            competitor_rate_col='competitor_mid_current',
            sales_lag_cols=None,
            output_directory='test_outputs'
        )
        assert 'confidence_interval_config' in configs
        assert 'tableau_formatting_config' in configs
        assert 'product_metadata_config' in configs
        assert 'visualization_config' in configs


class TestBuildInferenceStageConfig:
    """Tests for build_inference_stage_config function."""

    def test_returns_inference_stage_config_type(self):
        """Returns InferenceStageConfig TypedDict (dict at runtime)."""
        from src.config.inference_builders import build_inference_stage_config

        config = build_inference_stage_config()
        assert isinstance(config, dict)

    def test_contains_all_subconfigs(self):
        """Contains all required sub-configurations."""
        from src.config.inference_builders import build_inference_stage_config

        config = build_inference_stage_config()
        assert 'inference_config' in config
        assert 'rate_scenario_config' in config
        assert 'confidence_interval_config' in config
        assert 'tableau_formatting_config' in config
        assert 'product_metadata_config' in config
        assert 'visualization_config' in config

    def test_default_n_estimators_is_1000(self):
        """Stage config default n_estimators is 1000 (lower than inference config)."""
        from src.config.inference_builders import build_inference_stage_config

        config = build_inference_stage_config()
        # Stage config has lower default for faster execution
        assert config['inference_config']['n_estimators'] == 1000

    def test_custom_parameters_propagate(self):
        """Custom parameters propagate to sub-configs."""
        from src.config.inference_builders import build_inference_stage_config

        config = build_inference_stage_config(
            n_estimators=5000,
            confidence_level=0.99,
            rate_steps=25
        )
        assert config['inference_config']['n_estimators'] == 5000
        assert config['confidence_interval_config']['confidence_level'] == 0.99
        assert config['rate_scenario_config']['rate_steps'] == 25
