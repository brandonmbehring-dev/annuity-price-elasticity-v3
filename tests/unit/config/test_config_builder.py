"""
Unit tests for src/config/config_builder.py.

Tests validate configuration builder functions produce correct structures
with all expected keys and values.
"""

import pytest
from typing import Dict, Any

from src.config.config_builder import (
    # Pipeline configuration
    build_pipeline_configs,
    build_pipeline_configs_for_product,
    # Feature selection configuration
    build_aic_feature_selection_config,
    build_aic_analysis_config,
    build_feature_validation_config,
    build_model_selection_config,
    build_feature_selection_stage_config,
    # Internal helpers
    _get_default_base_features,
    _get_default_competitor_candidates,
)


class TestBuildPipelineConfigs:
    """Tests for build_pipeline_configs function."""

    def test_build_pipeline_configs_returns_all_keys(self, expected_pipeline_config_keys):
        """build_pipeline_configs should return dict with all expected keys."""
        configs = build_pipeline_configs(version=6)

        for key in expected_pipeline_config_keys:
            assert key in configs, f"Missing key: {key}"

    def test_build_pipeline_configs_version_6(self):
        """build_pipeline_configs version 6 should be valid."""
        configs = build_pipeline_configs(version=6)

        assert isinstance(configs, dict)
        assert 'product_filter' in configs
        assert 'sales_cleanup' in configs

    def test_build_pipeline_configs_default_product(self):
        """build_pipeline_configs should use default FlexGuard product."""
        configs = build_pipeline_configs(
            version=6,
            product_name="FlexGuard indexed variable annuity"
        )

        assert configs['product_filter']['product_name'] == "FlexGuard indexed variable annuity"

    def test_build_pipeline_configs_with_term_filter(self):
        """build_pipeline_configs should apply term filter."""
        configs = build_pipeline_configs(
            version=6,
            term_filter="6Y"
        )

        # Config uses 'term' not 'term_filter'
        assert configs['product_filter']['term'] == "6Y"

    def test_build_pipeline_configs_with_buffer_rate_filter(self):
        """build_pipeline_configs should apply buffer rate filter."""
        configs = build_pipeline_configs(
            version=6,
            buffer_rate_filter="20%"
        )

        # Config uses 'buffer_rate' not 'buffer_rate_filter'
        assert configs['product_filter']['buffer_rate'] == "20%"


class TestBuildPipelineConfigsForProduct:
    """Tests for build_pipeline_configs_for_product function."""

    def test_build_pipeline_configs_for_product_6y20b(self):
        """build_pipeline_configs_for_product should work for 6Y20B."""
        configs = build_pipeline_configs_for_product("6Y20B")

        assert 'product' in configs
        assert configs['product'].product_code == "6Y20B"
        assert configs['product'].buffer_level == 0.20

    def test_build_pipeline_configs_for_product_6y10b(self):
        """build_pipeline_configs_for_product should work for 6Y10B."""
        configs = build_pipeline_configs_for_product("6Y10B")

        assert configs['product'].product_code == "6Y10B"
        assert configs['product'].buffer_level == 0.10

    def test_build_pipeline_configs_for_product_contains_pipeline_keys(self, expected_pipeline_config_keys):
        """build_pipeline_configs_for_product should contain pipeline config keys."""
        configs = build_pipeline_configs_for_product("6Y20B")

        for key in expected_pipeline_config_keys:
            assert key in configs, f"Missing key: {key}"

    def test_build_pipeline_configs_for_product_invalid_raises(self, invalid_product_codes):
        """build_pipeline_configs_for_product should raise for invalid products."""
        for code in invalid_product_codes:
            if code:  # Skip empty string
                with pytest.raises(KeyError):
                    build_pipeline_configs_for_product(code)


class TestProductFilterConfig:
    """Tests for product_filter configuration structure."""

    def test_product_filter_config_structure(self):
        """product_filter config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        product_filter = configs['product_filter']

        # Actual config uses 'term' and 'buffer_rate' keys
        expected_keys = ['product_name', 'term', 'buffer_rate']
        for key in expected_keys:
            assert key in product_filter, f"Missing product_filter key: {key}"


class TestSalesCleanupConfig:
    """Tests for sales_cleanup configuration structure."""

    def test_sales_cleanup_config_structure(self):
        """sales_cleanup config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        sales_cleanup = configs['sales_cleanup']

        assert isinstance(sales_cleanup, dict)
        # Should have date validation settings
        assert 'contract_vs_application_max_days' in sales_cleanup or len(sales_cleanup) > 0


class TestWinkProcessingConfig:
    """Tests for wink_processing configuration structure."""

    def test_wink_processing_config_structure(self):
        """wink_processing config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        wink_processing = configs['wink_processing']

        assert isinstance(wink_processing, dict)


class TestCompetitiveConfig:
    """Tests for competitive configuration structure."""

    def test_competitive_config_structure(self):
        """competitive config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        competitive = configs['competitive']

        assert isinstance(competitive, dict)


class TestLagFeaturesConfig:
    """Tests for lag_features configuration structure."""

    def test_lag_features_config_structure(self):
        """lag_features config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        lag_features = configs['lag_features']

        assert isinstance(lag_features, (dict, list))


class TestFinalFeaturesConfig:
    """Tests for final_features configuration structure."""

    def test_final_features_config_structure(self):
        """final_features config should have expected structure."""
        configs = build_pipeline_configs(version=6)
        final_features = configs['final_features']

        assert isinstance(final_features, dict)


class TestConfigImmutability:
    """Tests for configuration immutability."""

    def test_configs_are_independent(self):
        """Multiple calls should return independent configs."""
        configs1 = build_pipeline_configs(version=6)
        configs2 = build_pipeline_configs(version=6)

        # Modifying one should not affect the other
        if isinstance(configs1['product_filter'], dict):
            configs1['product_filter']['test_key'] = 'test_value'
            assert 'test_key' not in configs2.get('product_filter', {})


class TestBuildAicFeatureSelectionConfig:
    """Tests for build_aic_feature_selection_config function."""

    def test_default_config(self):
        """Default config should have expected values.

        Feature Naming Unification (2026-01-26): Uses _t0 naming.
        """
        config = build_aic_feature_selection_config()

        assert config['target_variable'] == "sales_target_t0"
        assert config['max_candidate_features'] == 3
        assert config['economic_constraints'] is True

    def test_custom_target_variable(self):
        """Should accept custom target variable."""
        config = build_aic_feature_selection_config(target_variable="sales_log")

        assert config['target_variable'] == "sales_log"

    def test_custom_max_features(self):
        """Should accept custom max_candidate_features."""
        config = build_aic_feature_selection_config(max_candidate_features=5)

        assert config['max_candidate_features'] == 5


class TestBuildAicAnalysisConfig:
    """Tests for build_aic_analysis_config function."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = build_aic_analysis_config()

        assert config['max_models_to_test'] == 1000
        assert config['convergence_tolerance'] == 1e-6
        assert config['r_squared_threshold'] == 0.5
        assert config['p_value_threshold'] == 0.05

    def test_custom_max_models(self):
        """Should accept custom max_models_to_test."""
        config = build_aic_analysis_config(max_models_to_test=500)

        assert config['max_models_to_test'] == 500

    def test_coefficient_constraints_included(self):
        """Should include coefficient constraints."""
        config = build_aic_analysis_config()

        assert 'coefficient_constraints' in config
        assert 'competitor_' in config['coefficient_constraints']


class TestBuildFeatureValidationConfig:
    """Tests for build_feature_validation_config function."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = build_feature_validation_config()

        assert config['require_competitor_features'] is True
        assert config['require_prudential_features'] is True
        assert config['max_correlation_threshold'] == 0.9
        assert config['cross_validation_folds'] == 5

    def test_custom_cv_folds(self):
        """Should accept custom cross_validation_folds."""
        config = build_feature_validation_config(cross_validation_folds=10)

        assert config['cross_validation_folds'] == 10


class TestBuildModelSelectionConfig:
    """Tests for build_model_selection_config function."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = build_model_selection_config()

        assert config['selection_criteria'] == 'aic'
        assert config['ensemble_size'] == 5
        assert config['bootstrap_iterations'] == 100
        assert config['confidence_level'] == 0.95

    def test_custom_ensemble_size(self):
        """Should accept custom ensemble_size."""
        config = build_model_selection_config(ensemble_size=3)

        assert config['ensemble_size'] == 3

    def test_performance_metrics_included(self):
        """Should include performance metrics list."""
        config = build_model_selection_config()

        assert 'performance_metrics' in config
        assert 'aic' in config['performance_metrics']


class TestBuildFeatureSelectionStageConfig:
    """Tests for build_feature_selection_stage_config function."""

    def test_default_config_structure(self):
        """Default config should have all sub-configs."""
        config = build_feature_selection_stage_config()

        assert 'selection_config' in config
        assert 'analysis_config' in config
        assert 'validation_config' in config
        assert 'model_selection_config' in config

    def test_custom_max_candidate_features(self):
        """Should pass max_candidate_features to selection_config."""
        config = build_feature_selection_stage_config(max_candidate_features=4)

        assert config['selection_config']['max_candidate_features'] == 4


class TestDefaultFeatureHelpers:
    """Tests for internal default feature helper functions."""

    def test_get_default_base_features(self, expected_base_features):
        """_get_default_base_features should return expected features."""
        features = _get_default_base_features()

        for feature in expected_base_features:
            assert feature in features

    def test_get_default_competitor_candidates(self):
        """_get_default_competitor_candidates should return competitor features."""
        candidates = _get_default_competitor_candidates()

        assert len(candidates) > 0
        # All should contain 'competitor' in name
        for feature in candidates:
            assert 'competitor' in feature.lower()
