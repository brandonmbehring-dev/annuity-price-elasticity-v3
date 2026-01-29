"""
Unit tests for src/config/forecasting_builders.py

Tests forecasting core config, cross-validation, bootstrap models,
validation framework, sign correction, and stage config builders.
"""

import pytest
import numpy as np


class TestBuildForecastingCoreConfig:
    """Tests for _build_forecasting_core_config function."""

    def test_returns_forecasting_config_type(self):
        """Returns ForecastingConfig TypedDict (dict at runtime)."""
        from src.config.forecasting_builders import _build_forecasting_core_config

        config = _build_forecasting_core_config(
            bootstrap_samples=1000,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert isinstance(config, dict)

    def test_n_bootstrap_samples_set(self):
        """Bootstrap samples parameter is set."""
        from src.config.forecasting_builders import _build_forecasting_core_config

        config = _build_forecasting_core_config(
            bootstrap_samples=500,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert config['n_bootstrap_samples'] == 500

    def test_exclude_holidays_default_true(self):
        """Exclude holidays is True by default."""
        from src.config.forecasting_builders import _build_forecasting_core_config

        config = _build_forecasting_core_config(
            bootstrap_samples=1000,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert config['exclude_holidays'] is True

    def test_mature_data_cutoff_set(self):
        """Mature data cutoff days is set (business rule: 50 days)."""
        from src.config.forecasting_builders import _build_forecasting_core_config

        config = _build_forecasting_core_config(
            bootstrap_samples=1000,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert config['mature_data_cutoff_days'] == 50


class TestBuildCvConfig:
    """Tests for _build_cv_config function."""

    def test_returns_cross_validation_config_type(self):
        """Returns CrossValidationConfig TypedDict (dict at runtime)."""
        from src.config.forecasting_builders import _build_cv_config

        config = _build_cv_config(
            start_cutoff=30,
            end_cutoff=None,
            validation_method='expanding_window'
        )
        assert isinstance(config, dict)

    def test_start_cutoff_set(self):
        """Start cutoff parameter is set."""
        from src.config.forecasting_builders import _build_cv_config

        config = _build_cv_config(
            start_cutoff=50,
            end_cutoff=None,
            validation_method='expanding_window'
        )
        assert config['start_cutoff'] == 50

    def test_end_cutoff_none_allowed(self):
        """End cutoff can be None (use full dataset)."""
        from src.config.forecasting_builders import _build_cv_config

        config = _build_cv_config(
            start_cutoff=30,
            end_cutoff=None,
            validation_method='expanding_window'
        )
        assert config['end_cutoff'] is None

    def test_n_splits_zero_for_expanding(self):
        """n_splits is 0 for expanding window (continuous)."""
        from src.config.forecasting_builders import _build_cv_config

        config = _build_cv_config(
            start_cutoff=30,
            end_cutoff=None,
            validation_method='expanding_window'
        )
        assert config['n_splits'] == 0


class TestBuildBootstrapModelConfig:
    """Tests for _build_bootstrap_model_config function."""

    def test_returns_bootstrap_model_config_type(self):
        """Returns BootstrapModelConfig TypedDict (dict at runtime)."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=1.0, bootstrap_samples=1000)
        assert isinstance(config, dict)

    def test_estimator_type_is_ridge(self):
        """Estimator type is 'Ridge'."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=1.0, bootstrap_samples=1000)
        assert config['estimator_type'] == 'Ridge'

    def test_positive_constraint_true(self):
        """Positive constraint is True (sales >= 0)."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=1.0, bootstrap_samples=1000)
        assert config['positive_constraint'] is True

    def test_normalize_false(self):
        """Normalize is False (features already preprocessed)."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=1.0, bootstrap_samples=1000)
        assert config['normalize'] is False

    def test_alpha_parameter_set(self):
        """Alpha parameter is set from ridge_alpha."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=2.5, bootstrap_samples=1000)
        assert config['alpha'] == 2.5

    def test_n_estimators_set(self):
        """n_estimators is set from bootstrap_samples."""
        from src.config.forecasting_builders import _build_bootstrap_model_config

        config = _build_bootstrap_model_config(ridge_alpha=1.0, bootstrap_samples=500)
        assert config['n_estimators'] == 500


class TestBuildValidationFrameworkConfig:
    """Tests for _build_validation_framework_config function."""

    def test_returns_validation_framework_config_type(self):
        """Returns ValidationFrameworkConfig TypedDict (dict at runtime)."""
        from src.config.forecasting_builders import _build_validation_framework_config

        config = _build_validation_framework_config(enable_detailed_validation=True)
        assert isinstance(config, dict)

    def test_tolerance_r2_set(self):
        """R2 tolerance is set (1e-6)."""
        from src.config.forecasting_builders import _build_validation_framework_config

        config = _build_validation_framework_config(enable_detailed_validation=True)
        assert config['tolerance_r2'] == 1e-6

    def test_tolerance_mape_set(self):
        """MAPE tolerance is set (1e-4)."""
        from src.config.forecasting_builders import _build_validation_framework_config

        config = _build_validation_framework_config(enable_detailed_validation=True)
        assert config['tolerance_mape'] == 1e-4

    def test_enable_detailed_comparison_set(self):
        """Enable detailed comparison flag is set."""
        from src.config.forecasting_builders import _build_validation_framework_config

        config = _build_validation_framework_config(enable_detailed_validation=False)
        assert config['enable_detailed_comparison'] is False


class TestGetCompetitiveFeatures:
    """Tests for _get_competitive_features function."""

    def test_returns_list(self):
        """Returns list of feature names."""
        from src.config.forecasting_builders import _get_competitive_features

        features = _get_competitive_features()
        assert isinstance(features, list)

    def test_all_features_contain_competitor(self):
        """All returned features contain 'competitor' in name."""
        from src.config.forecasting_builders import _get_competitive_features

        features = _get_competitive_features()
        for f in features:
            assert 'competitor' in f.lower()


class TestBuildSignCorrectionConfigs:
    """Tests for _build_sign_correction_configs function."""

    def test_returns_tuple_of_two_dicts(self):
        """Returns tuple of (model_config, benchmark_config)."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        model_cfg, benchmark_cfg = _build_sign_correction_configs(
            model_features=['prudential_rate', 'competitor_mid'],
            benchmark_features=['sales_target_t5'],
            competitive_features=['competitor_mid'],
            decay_rate=0.98
        )
        assert isinstance(model_cfg, dict)
        assert isinstance(benchmark_cfg, dict)

    def test_model_sign_correction_mask_type(self):
        """Model sign correction mask is numpy array."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        model_cfg, _ = _build_sign_correction_configs(
            model_features=['prudential_rate', 'competitor_mid'],
            benchmark_features=['sales_target_t5'],
            competitive_features=['competitor_mid'],
            decay_rate=0.98
        )
        assert isinstance(model_cfg['sign_correction_mask'], np.ndarray)

    def test_competitive_features_marked_for_correction(self):
        """Competitive features are marked True in sign correction mask."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        model_cfg, _ = _build_sign_correction_configs(
            model_features=['prudential_rate', 'competitor_mid'],
            benchmark_features=['sales_target_t5'],
            competitive_features=['competitor_mid'],
            decay_rate=0.98
        )
        # competitor_mid is at index 1
        assert model_cfg['sign_correction_mask'][1] is True or model_cfg['sign_correction_mask'][1] == True

    def test_non_competitive_features_not_marked(self):
        """Non-competitive features are marked False."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        model_cfg, _ = _build_sign_correction_configs(
            model_features=['prudential_rate', 'competitor_mid'],
            benchmark_features=['sales_target_t5'],
            competitive_features=['competitor_mid'],
            decay_rate=0.98
        )
        # prudential_rate is at index 0
        assert model_cfg['sign_correction_mask'][0] is False or model_cfg['sign_correction_mask'][0] == False

    def test_benchmark_has_no_sign_correction(self):
        """Benchmark features have all False sign correction mask."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        _, benchmark_cfg = _build_sign_correction_configs(
            model_features=['prudential_rate', 'competitor_mid'],
            benchmark_features=['sales_target_t5'],
            competitive_features=['competitor_mid'],
            decay_rate=0.98
        )
        assert all(v == False for v in benchmark_cfg['sign_correction_mask'])

    def test_decay_rate_included(self):
        """Decay rate is included in both configs."""
        from src.config.forecasting_builders import _build_sign_correction_configs

        model_cfg, benchmark_cfg = _build_sign_correction_configs(
            model_features=['prudential_rate'],
            benchmark_features=['sales_target_t5'],
            competitive_features=[],
            decay_rate=0.95
        )
        assert model_cfg['decay_rate'] == 0.95
        assert benchmark_cfg['decay_rate'] == 0.95


class TestBuildModelConfigs:
    """Tests for _build_model_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with forecasting, bootstrap, benchmark configs."""
        from src.config.forecasting_builders import _build_model_configs

        configs = _build_model_configs(
            bootstrap_samples=1000,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert 'forecasting_config' in configs
        assert 'bootstrap_model_config' in configs
        assert 'benchmark_model_config' in configs

    def test_benchmark_method_is_rolling_average(self):
        """Benchmark model method is 'rolling_average'."""
        from src.config.forecasting_builders import _build_model_configs

        configs = _build_model_configs(
            bootstrap_samples=1000,
            ridge_alpha=1.0,
            random_state=42,
            min_training_cutoff=30
        )
        assert configs['benchmark_model_config']['method'] == 'rolling_average'


class TestBuildValidationConfigs:
    """Tests for _build_validation_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with cv_config and validation_config."""
        from src.config.forecasting_builders import _build_validation_configs

        configs = _build_validation_configs(
            start_cutoff=30,
            end_cutoff=None,
            validation_method='expanding_window',
            enable_detailed_validation=True
        )
        assert 'cv_config' in configs
        assert 'validation_config' in configs


class TestBuildFeatureAndSignConfigs:
    """Tests for _build_feature_and_sign_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with feature and sign correction configs."""
        from src.config.forecasting_builders import _build_feature_and_sign_configs

        configs = _build_feature_and_sign_configs()
        assert 'business_filter_config' in configs
        assert 'competitive_features' in configs
        assert 'model_features' in configs
        assert 'benchmark_features' in configs
        assert 'target_variable' in configs
        assert 'model_sign_correction_config' in configs
        assert 'benchmark_sign_correction_config' in configs

    def test_analysis_start_date_set(self):
        """Business filter includes analysis start date."""
        from src.config.forecasting_builders import _build_feature_and_sign_configs

        configs = _build_feature_and_sign_configs()
        assert configs['business_filter_config']['analysis_start_date'] == '2022-04-01'

    def test_target_variable_is_sales_target_current(self):
        """Target variable is 'sales_target_current'."""
        from src.config.forecasting_builders import _build_feature_and_sign_configs

        configs = _build_feature_and_sign_configs()
        assert configs['target_variable'] == 'sales_target_current'


class TestBuildForecastingStageConfig:
    """Tests for build_forecasting_stage_config function."""

    def test_returns_forecasting_stage_config_type(self):
        """Returns ForecastingStageConfig TypedDict (dict at runtime)."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert isinstance(config, dict)

    def test_contains_all_subconfigs(self):
        """Contains all required sub-configurations."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert 'forecasting_config' in config
        assert 'bootstrap_model_config' in config
        assert 'benchmark_model_config' in config
        assert 'cv_config' in config
        assert 'validation_config' in config
        assert 'business_filter_config' in config
        assert 'model_sign_correction_config' in config

    def test_default_bootstrap_samples(self):
        """Default bootstrap samples is 1000."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert config['forecasting_config']['n_bootstrap_samples'] == 1000

    def test_default_ridge_alpha(self):
        """Default ridge alpha is 1.0."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert config['bootstrap_model_config']['alpha'] == 1.0

    def test_default_random_state(self):
        """Default random state is 42."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert config['forecasting_config']['random_state'] == 42

    def test_default_start_cutoff(self):
        """Default start cutoff is 30."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config()
        assert config['cv_config']['start_cutoff'] == 30

    def test_custom_bootstrap_samples(self):
        """Custom bootstrap samples propagates."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config(bootstrap_samples=500)
        assert config['forecasting_config']['n_bootstrap_samples'] == 500
        assert config['bootstrap_model_config']['n_estimators'] == 500

    def test_custom_validation_method(self):
        """Custom validation method propagates."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config(validation_method='rolling_window')
        assert config['cv_config']['validation_method'] == 'rolling_window'

    def test_custom_enable_detailed_validation(self):
        """Custom enable_detailed_validation propagates."""
        from src.config.forecasting_builders import build_forecasting_stage_config

        config = build_forecasting_stage_config(enable_detailed_validation=False)
        assert config['validation_config']['enable_detailed_comparison'] is False
