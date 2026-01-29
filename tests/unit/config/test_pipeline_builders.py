"""
Unit tests for src/config/pipeline_builders.py

Tests lag column configs, weekly aggregation, pipeline stage builders,
and the main pipeline config builder functions.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List


class TestGetLagColumnConfigs:
    """Tests for get_lag_column_configs function."""

    def test_returns_list(self):
        """Returns list of configuration dictionaries."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        assert isinstance(configs, list)

    def test_returns_13_configs(self):
        """Returns 13 lag column configurations."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        assert len(configs) == 13

    def test_each_config_has_required_keys(self):
        """Each config has source_col, prefix, and lag_direction."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        for config in configs:
            assert 'source_col' in config
            assert 'prefix' in config
            assert 'lag_direction' in config

    def test_sales_target_prefix_exists(self):
        """sales_target prefix exists for sales features."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        prefixes = [c['prefix'] for c in configs]
        assert 'sales_target' in prefixes

    def test_competitor_top5_prefix_exists(self):
        """competitor_top5 prefix exists."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        prefixes = [c['prefix'] for c in configs]
        assert 'competitor_top5' in prefixes

    def test_prudential_rate_prefix_exists(self):
        """prudential_rate prefix exists."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        prefixes = [c['prefix'] for c in configs]
        assert 'prudential_rate' in prefixes

    def test_lag_directions_are_valid(self):
        """All lag directions are 'backward' or 'both'."""
        from src.config.pipeline_builders import get_lag_column_configs

        configs = get_lag_column_configs()
        valid_directions = {'backward', 'both'}
        for config in configs:
            assert config['lag_direction'] in valid_directions


class TestGetWeeklyAggregationDict:
    """Tests for get_weekly_aggregation_dict function."""

    def test_returns_dict(self):
        """Returns dictionary mapping columns to aggregation methods."""
        from src.config.pipeline_builders import get_weekly_aggregation_dict

        agg_dict = get_weekly_aggregation_dict()
        assert isinstance(agg_dict, dict)

    def test_competitive_features_use_mean(self):
        """Competitive features use 'mean' aggregation."""
        from src.config.pipeline_builders import get_weekly_aggregation_dict

        agg_dict = get_weekly_aggregation_dict()
        assert agg_dict['C_weighted_mean'] == 'mean'
        assert agg_dict['C_core'] == 'mean'
        assert agg_dict['Prudential'] == 'mean'

    def test_sales_uses_sum(self):
        """Sales columns use 'sum' aggregation."""
        from src.config.pipeline_builders import get_weekly_aggregation_dict

        agg_dict = get_weekly_aggregation_dict()
        assert agg_dict['sales'] == 'sum'
        assert agg_dict['sales_by_contract_date'] == 'sum'

    def test_economic_indicators_use_last(self):
        """Economic indicators use 'last' aggregation."""
        from src.config.pipeline_builders import get_weekly_aggregation_dict

        agg_dict = get_weekly_aggregation_dict()
        assert agg_dict['DGS5'] == 'last'
        assert agg_dict['VIXCLS'] == 'last'

    def test_company_columns_use_mean(self):
        """Individual company rate columns use 'mean'."""
        from src.config.pipeline_builders import get_weekly_aggregation_dict

        agg_dict = get_weekly_aggregation_dict()
        assert agg_dict['Allianz'] == 'mean'
        assert agg_dict['Lincoln'] == 'mean'


class TestBuildProductFilterConfig:
    """Tests for _build_product_filter_config function."""

    def test_returns_product_filter_config_type(self):
        """Returns ProductFilterConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_product_filter_config

        config = _build_product_filter_config(
            product_name='FlexGuard',
            buffer_rate_filter='20%',
            term_filter='6Y'
        )
        assert isinstance(config, dict)

    def test_product_name_set(self):
        """Product name is set correctly."""
        from src.config.pipeline_builders import _build_product_filter_config

        config = _build_product_filter_config(
            product_name='TestProduct',
            buffer_rate_filter='20%',
            term_filter='6Y'
        )
        assert config['product_name'] == 'TestProduct'

    def test_buffer_rate_set(self):
        """Buffer rate is set correctly."""
        from src.config.pipeline_builders import _build_product_filter_config

        config = _build_product_filter_config(
            product_name='FlexGuard',
            buffer_rate_filter='10%',
            term_filter='6Y'
        )
        assert config['buffer_rate'] == '10%'


class TestBuildSalesCleanupConfig:
    """Tests for _build_sales_cleanup_config function."""

    def test_returns_sales_cleanup_config_type(self):
        """Returns SalesCleanupConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_sales_cleanup_config

        config = _build_sales_cleanup_config(
            min_premium=1000.0,
            max_premium=1000000.0,
            quantile_threshold=0.99
        )
        assert isinstance(config, dict)

    def test_premium_range_set(self):
        """Premium range is set correctly."""
        from src.config.pipeline_builders import _build_sales_cleanup_config

        config = _build_sales_cleanup_config(
            min_premium=5000.0,
            max_premium=500000.0,
            quantile_threshold=0.99
        )
        assert config['min_premium'] == 5000.0
        assert config['max_premium'] == 500000.0

    def test_column_names_set(self):
        """Standard column names are set."""
        from src.config.pipeline_builders import _build_sales_cleanup_config

        config = _build_sales_cleanup_config(
            min_premium=1000.0,
            max_premium=1000000.0,
            quantile_threshold=0.99
        )
        assert config['start_date_col'] == 'application_signed_date'
        assert config['end_date_col'] == 'contract_issue_date'
        assert config['premium_column'] == 'contract_initial_premium_amount'


class TestBuildTimeSeriesConfig:
    """Tests for _build_time_series_config function."""

    def test_returns_time_series_config_type(self):
        """Returns TimeSeriesConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_time_series_config

        config = _build_time_series_config(
            groupby_frequency='d',
            rolling_window_days=14
        )
        assert isinstance(config, dict)

    def test_frequency_set(self):
        """Groupby frequency is set correctly."""
        from src.config.pipeline_builders import _build_time_series_config

        config = _build_time_series_config(
            groupby_frequency='W',
            rolling_window_days=14
        )
        assert config['groupby_frequency'] == 'W'

    def test_rolling_window_set(self):
        """Rolling window days is set correctly."""
        from src.config.pipeline_builders import _build_time_series_config

        config = _build_time_series_config(
            groupby_frequency='d',
            rolling_window_days=7
        )
        assert config['rolling_window_days'] == 7


class TestBuildWinkProcessingConfig:
    """Tests for _build_wink_processing_config function."""

    def test_returns_wink_processing_config_type(self):
        """Returns WinkProcessingConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_wink_processing_config

        config = _build_wink_processing_config(
            flexguard_product_ids={'Prudential': [123]},
            rate_analysis_start_date='2018-06-21',
            rolling_window_days=14,
            data_filter_start_date='2018-01-01'
        )
        assert isinstance(config, dict)

    def test_product_ids_set(self):
        """Product IDs are set correctly."""
        from src.config.pipeline_builders import _build_wink_processing_config

        ids = {'Prudential': [123, 456]}
        config = _build_wink_processing_config(
            flexguard_product_ids=ids,
            rate_analysis_start_date='2018-06-21',
            rolling_window_days=14,
            data_filter_start_date='2018-01-01'
        )
        assert config['product_ids'] == ids

    def test_business_filters_set(self):
        """Business filter values are set."""
        from src.config.pipeline_builders import _build_wink_processing_config

        config = _build_wink_processing_config(
            flexguard_product_ids={'Prudential': [123]},
            rate_analysis_start_date='2018-06-21',
            rolling_window_days=14,
            data_filter_start_date='2018-01-01'
        )
        assert config['buffer_modifier_filter'] == 'Losses Covered Up To'
        assert config['product_type_filter'] == 'Structured'
        assert config['index_name_filter'] == 'S&P 500'


class TestBuildDataIntegrationConfig:
    """Tests for _build_data_integration_config function."""

    def test_returns_data_integration_config_type(self):
        """Returns DataIntegrationConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_data_integration_config

        config = _build_data_integration_config(
            analysis_start_date='2021-01-01',
            current_date='2024-01-01',
            economic_indicator_rolling_window=7
        )
        assert isinstance(config, dict)

    def test_date_range_set(self):
        """Date range is set correctly."""
        from src.config.pipeline_builders import _build_data_integration_config

        config = _build_data_integration_config(
            analysis_start_date='2022-01-01',
            current_date='2024-06-01',
            economic_indicator_rolling_window=7
        )
        assert config['start_date'] == '2022-01-01'
        assert config['end_date'] == '2024-06-01'


class TestBuildLagFeaturesConfig:
    """Tests for _build_lag_features_config function."""

    def test_returns_lag_feature_config_type(self):
        """Returns LagFeatureConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_lag_features_config

        config = _build_lag_features_config(
            max_lag_periods=18,
            allow_inplace_operations=True
        )
        assert isinstance(config, dict)

    def test_max_lag_periods_set(self):
        """Max lag periods is set correctly."""
        from src.config.pipeline_builders import _build_lag_features_config

        config = _build_lag_features_config(
            max_lag_periods=12,
            allow_inplace_operations=True
        )
        assert config['max_lag_periods'] == 12

    def test_lag_column_configs_included(self):
        """Lag column configs are included."""
        from src.config.pipeline_builders import _build_lag_features_config

        config = _build_lag_features_config(
            max_lag_periods=18,
            allow_inplace_operations=True
        )
        assert 'lag_column_configs' in config
        assert len(config['lag_column_configs']) == 13


class TestBuildFinalFeaturesConfig:
    """Tests for _build_final_features_config function."""

    def test_returns_feature_config_type(self):
        """Returns FeatureConfig TypedDict (dict at runtime)."""
        from src.config.pipeline_builders import _build_final_features_config

        config = _build_final_features_config(
            feature_analysis_start_date='2022-01-01',
            holiday_start_day=13,
            holiday_end_day=359,
            current_date_of_mature_data='2024-01-01'
        )
        assert isinstance(config, dict)

    def test_holiday_range_set(self):
        """Holiday day range is set correctly."""
        from src.config.pipeline_builders import _build_final_features_config

        config = _build_final_features_config(
            feature_analysis_start_date='2022-01-01',
            holiday_start_day=1,
            holiday_end_day=365,
            current_date_of_mature_data='2024-01-01'
        )
        assert config['holiday_start_day'] == 1
        assert config['holiday_end_day'] == 365


class TestGetDefaultFlexguardProductIds:
    """Tests for _get_default_flexguard_product_ids function."""

    def test_returns_dict(self):
        """Returns dictionary mapping companies to product IDs."""
        from src.config.pipeline_builders import _get_default_flexguard_product_ids

        ids = _get_default_flexguard_product_ids()
        assert isinstance(ids, dict)

    def test_contains_prudential(self):
        """Contains Prudential product IDs."""
        from src.config.pipeline_builders import _get_default_flexguard_product_ids

        ids = _get_default_flexguard_product_ids()
        assert 'Prudential' in ids

    def test_values_are_lists(self):
        """All values are lists."""
        from src.config.pipeline_builders import _get_default_flexguard_product_ids

        ids = _get_default_flexguard_product_ids()
        for company, product_ids in ids.items():
            assert isinstance(product_ids, list)


class TestBuildExtractionConfigs:
    """Tests for _build_extraction_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with product_filter and sales_cleanup."""
        from src.config.pipeline_builders import _build_extraction_configs

        configs = _build_extraction_configs(
            product_name='FlexGuard',
            buffer_rate_filter='20%',
            term_filter='6Y',
            min_premium=1000.0,
            max_premium=1000000.0,
            quantile_threshold=0.99
        )
        assert 'product_filter' in configs
        assert 'sales_cleanup' in configs


class TestBuildProcessingConfigs:
    """Tests for _build_processing_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with time_series, wink, weekly_agg, competitive."""
        from src.config.pipeline_builders import _build_processing_configs

        configs = _build_processing_configs(
            groupby_frequency='d',
            rolling_window_days=14,
            flexguard_product_ids={'Prudential': [123]},
            rate_analysis_start_date='2018-06-21',
            data_filter_start_date='2018-01-01'
        )
        assert 'time_series' in configs
        assert 'wink_processing' in configs
        assert 'weekly_aggregation' in configs
        assert 'competitive' in configs


class TestBuildIntegrationConfigs:
    """Tests for _build_integration_configs function."""

    def test_returns_dict_with_required_keys(self):
        """Returns dict with data_integration, lag_features, final_features."""
        from src.config.pipeline_builders import _build_integration_configs

        configs = _build_integration_configs(
            analysis_start_date='2021-01-01',
            current_date='2024-01-01',
            economic_indicator_rolling_window=7,
            max_lag_periods=18,
            allow_inplace_operations=True,
            feature_analysis_start_date='2022-01-01',
            holiday_start_day=13,
            holiday_end_day=359,
            current_date_of_mature_data='2024-01-01'
        )
        assert 'data_integration' in configs
        assert 'lag_features' in configs
        assert 'final_features' in configs


class TestBuildPipelineConfigs:
    """Tests for build_pipeline_configs function."""

    def test_returns_dict(self):
        """Returns dictionary with all configurations."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs()
        assert isinstance(configs, dict)

    def test_contains_9_configs(self):
        """Returns 9 pipeline configurations."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs()
        expected_keys = [
            'product_filter', 'sales_cleanup', 'time_series',
            'wink_processing', 'weekly_aggregation', 'competitive',
            'data_integration', 'lag_features', 'final_features'
        ]
        for key in expected_keys:
            assert key in configs, f"Missing key: {key}"

    def test_default_product_name(self):
        """Default product name is FlexGuard."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs()
        assert 'FlexGuard' in configs['product_filter']['product_name']

    def test_default_term_filter(self):
        """Default term filter is 6Y."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs()
        assert configs['product_filter']['term'] == '6Y'

    def test_default_buffer_rate_filter(self):
        """Default buffer rate filter is 20%."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs()
        assert configs['product_filter']['buffer_rate'] == '20%'

    def test_custom_term_filter(self):
        """Custom term filter is respected."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs(term_filter='10Y')
        assert configs['product_filter']['term'] == '10Y'

    def test_custom_max_lag_periods(self):
        """Custom max lag periods is respected."""
        from src.config.pipeline_builders import build_pipeline_configs

        configs = build_pipeline_configs(max_lag_periods=12)
        assert configs['lag_features']['max_lag_periods'] == 12


class TestConvertProductToFilters:
    """Tests for _convert_product_to_filters function."""

    def test_returns_dict_with_filter_keys(self):
        """Returns dict with buffer_rate_filter and term_filter."""
        from src.config.pipeline_builders import _convert_product_to_filters
        from src.config.product_config import get_product_config

        product = get_product_config('6Y20B')
        filters = _convert_product_to_filters(product)
        assert 'buffer_rate_filter' in filters
        assert 'term_filter' in filters

    def test_6y20b_filters(self):
        """6Y20B product produces correct filters."""
        from src.config.pipeline_builders import _convert_product_to_filters
        from src.config.product_config import get_product_config

        product = get_product_config('6Y20B')
        filters = _convert_product_to_filters(product)
        assert filters['buffer_rate_filter'] == '20%'
        assert filters['term_filter'] == '6Y'


class TestBuildPipelineConfigsForProduct:
    """Tests for build_pipeline_configs_for_product function."""

    def test_returns_dict_with_product(self):
        """Returns dict including 'product' key."""
        from src.config.pipeline_builders import build_pipeline_configs_for_product

        configs = build_pipeline_configs_for_product('6Y20B')
        assert 'product' in configs

    def test_6y20b_product_config_included(self):
        """6Y20B product config is included."""
        from src.config.pipeline_builders import build_pipeline_configs_for_product

        configs = build_pipeline_configs_for_product('6Y20B')
        assert configs['product'].buffer_level == 0.20
        assert configs['product'].term_years == 6

    def test_product_derives_filters(self):
        """Product config derives correct filters."""
        from src.config.pipeline_builders import build_pipeline_configs_for_product

        configs = build_pipeline_configs_for_product('6Y10B')
        assert configs['product_filter']['buffer_rate'] == '10%'
        assert configs['product_filter']['term'] == '6Y'

    def test_custom_kwargs_override(self):
        """Custom kwargs can override product-derived values."""
        from src.config.pipeline_builders import build_pipeline_configs_for_product

        configs = build_pipeline_configs_for_product('6Y20B', max_lag_periods=10)
        assert configs['lag_features']['max_lag_periods'] == 10

    def test_10y20b_product(self):
        """10Y20B product config works correctly."""
        from src.config.pipeline_builders import build_pipeline_configs_for_product

        configs = build_pipeline_configs_for_product('10Y20B')
        assert configs['product'].term_years == 10
        assert configs['product_filter']['term'] == '10Y'
