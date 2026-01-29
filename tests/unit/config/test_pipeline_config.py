"""
Unit tests for src/config/pipeline_config.py.

Tests validate TypedDict configurations for pipeline operations.
"""

import pytest
from typing import get_type_hints

from src.config.pipeline_config import (
    # AWS & Infrastructure
    PipelineAWSConfig,
    AWSConfig,
    S3Config,
    # Data Loading
    DataLoadingConfig,
    TDELoadingConfig,
    WINKLoadingConfig,
    # Preprocessing
    PreprocessingTimeSeriesConfig,
    PreprocessingWINKConfig,
    PreprocessingTDEConfig,
    PreprocessingValidationConfig,
    # Processing
    ProcessingMathematicalConfig,
    ProcessingTimeSeriesConfig,
    # Feature Engineering
    FeatureSelectionConfig,
    FeatureValidationConfig,
    ModelSelectionConfig,
    FeatureSelectionStageConfig,
    # Pipeline Stage Configs
    ProductFilterConfig,
    SalesCleanupConfig,
    TimeSeriesConfig,
    WinkProcessingConfig,
    DataIntegrationConfig,
    WeeklyAggregationConfig,
    CompetitiveConfig,
    LagFeatureConfig,
    FeatureConfig,
)


class TestPipelineAWSConfig:
    """Tests for PipelineAWSConfig TypedDict."""

    def test_can_create_pipeline_aws_config(self):
        """PipelineAWSConfig should be creatable with all fields."""
        config: PipelineAWSConfig = {
            'xid': 'test_user',
            'role_arn': 'arn:aws:iam::123:role/TestRole',
            'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com',
            'source_bucket_name': 'source-bucket',
            'output_bucket_name': 'output-bucket',
            'output_base_path': 'outputs/',
        }
        assert config['xid'] == 'test_user'
        assert config['source_bucket_name'] == 'source-bucket'

    def test_aws_config_is_alias_for_pipeline_aws_config(self):
        """AWSConfig should be an alias for PipelineAWSConfig."""
        assert AWSConfig is PipelineAWSConfig


class TestS3Config:
    """Tests for S3Config TypedDict."""

    def test_can_create_s3_config(self):
        """S3Config should be creatable with all fields."""
        config: S3Config = {
            'bucket_name': 'test-bucket',
            'base_path': 'data/',
            'current_date': '2024-01-15',
            'date_path': '2024/01/15',
            'export_enabled': True,
        }
        assert config['bucket_name'] == 'test-bucket'
        assert config['export_enabled'] is True


class TestDataLoadingConfig:
    """Tests for DataLoadingConfig TypedDict."""

    def test_can_create_data_loading_config(self):
        """DataLoadingConfig should be creatable with all fields."""
        config: DataLoadingConfig = {
            'purpose': 'validation',
            'version': 6,
            'date_filter_start': '2021-01-01',
            'days_before_mature': 90,
            'exclude_holidays': True,
        }
        assert config['purpose'] == 'validation'
        assert config['version'] == 6


class TestPreprocessingConfigs:
    """Tests for preprocessing configuration TypedDicts."""

    def test_can_create_time_series_config(self):
        """PreprocessingTimeSeriesConfig should be creatable."""
        config: PreprocessingTimeSeriesConfig = {
            'rolling_window_days': 7,
            'groupby_frequency': 'd',
            'analysis_start_date': '2022-01-01',
            'interpolation_method': 'linear',
        }
        assert config['rolling_window_days'] == 7

    def test_can_create_tde_config(self):
        """PreprocessingTDEConfig should be creatable."""
        config: PreprocessingTDEConfig = {
            'premium_column': 'contract_initial_premium_amount',
            'min_premium': 0.0,
            'max_premium': 1_000_000.0,
            'business_rules': {'filter_zeros': True},
            'numeric_columns': ['premium', 'age'],
            'categorical_columns': ['product_name'],
        }
        assert config['premium_column'] == 'contract_initial_premium_amount'


class TestFeatureSelectionConfig:
    """Tests for feature selection configuration TypedDicts."""

    def test_can_create_feature_selection_config(self):
        """FeatureSelectionConfig should be creatable."""
        config: FeatureSelectionConfig = {
            'target_variable': 'sales_target_current',
            'base_features': ['prudential_rate_current'],
            'candidate_features': ['competitor_mid_lag1'],
            'max_candidate_features': 3,
            'economic_constraints': True,
            'min_observations': 30,
            'validation_split': 0.3,
        }
        assert config['target_variable'] == 'sales_target_current'
        assert config['max_candidate_features'] == 3

    def test_can_create_feature_validation_config(self):
        """FeatureValidationConfig should be creatable."""
        config: FeatureValidationConfig = {
            'require_competitor_features': True,
            'require_prudential_features': True,
            'max_correlation_threshold': 0.9,
            'min_feature_importance': 0.01,
            'cross_validation_folds': 5,
        }
        assert config['cross_validation_folds'] == 5

    def test_can_create_model_selection_config(self):
        """ModelSelectionConfig should be creatable."""
        config: ModelSelectionConfig = {
            'selection_criteria': 'aic',
            'ensemble_size': 5,
            'bootstrap_iterations': 100,
            'confidence_level': 0.95,
            'performance_metrics': ['aic', 'r_squared'],
        }
        assert config['selection_criteria'] == 'aic'


class TestPipelineStageConfigs:
    """Tests for pipeline stage configuration TypedDicts."""

    def test_can_create_product_filter_config(self):
        """ProductFilterConfig should be creatable."""
        config: ProductFilterConfig = {
            'product_name': 'FlexGuard indexed variable annuity',
            'buffer_rate': '20%',
            'term': '6Y',
        }
        assert config['product_name'] == 'FlexGuard indexed variable annuity'

    def test_can_create_sales_cleanup_config(self):
        """SalesCleanupConfig should be creatable."""
        config: SalesCleanupConfig = {
            'min_premium': 0.0,
            'max_premium': 1_000_000.0,
            'quantile_threshold': 0.99,
            'start_date_col': 'application_signed_date',
            'end_date_col': 'contract_issue_date',
            'processing_days_col': 'processing_days',
            'premium_column': 'contract_initial_premium_amount',
            'sales_alias_col': 'sales_amount',
        }
        assert config['quantile_threshold'] == 0.99

    def test_can_create_time_series_config(self):
        """TimeSeriesConfig should be creatable."""
        config: TimeSeriesConfig = {
            'date_column': 'date',
            'value_column': 'premium',
            'alias_date_col': 'date',
            'alias_value_col': 'sales',
            'groupby_frequency': 'd',
            'rolling_window_days': 7,
        }
        assert config['groupby_frequency'] == 'd'

    def test_can_create_lag_feature_config(self):
        """LagFeatureConfig should be creatable."""
        config: LagFeatureConfig = {
            'max_lag': 4,
            'min_lag': 1,
            'lag_direction': 'backward',
        }
        assert config['max_lag'] == 4


class TestConfigTypeHints:
    """Tests verifying TypedDict configurations have proper type hints."""

    def test_pipeline_aws_config_has_type_hints(self):
        """PipelineAWSConfig should have type hints for all fields."""
        hints = get_type_hints(PipelineAWSConfig)
        expected_fields = ['xid', 'role_arn', 'sts_endpoint_url',
                          'source_bucket_name', 'output_bucket_name', 'output_base_path']
        for field in expected_fields:
            assert field in hints

    def test_feature_selection_config_has_type_hints(self):
        """FeatureSelectionConfig should have type hints for all fields."""
        hints = get_type_hints(FeatureSelectionConfig)
        expected_fields = ['target_variable', 'base_features', 'candidate_features']
        for field in expected_fields:
            assert field in hints
