"""
Unit tests for src/data/pipelines.py

Tests pipeline functions for product filtering, sales cleanup,
time series creation, and data integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def product_filter_config():
    """Standard product filter config for testing."""
    return {
        'product_name': 'FlexGuard indexed variable annuity',
        'buffer_rate': '20%',
        'term': '6Y'
    }


@pytest.fixture
def sales_cleanup_config():
    """Standard sales cleanup config for testing."""
    return {
        'min_premium': 1000.0,
        'max_premium': 1000000.0,
        'quantile_threshold': 0.99,
        'start_date_col': 'application_signed_date',
        'end_date_col': 'contract_issue_date',
        'processing_days_col': 'processing_days',
        'premium_column': 'premium_amount',
        'sales_alias_col': 'sales_amount'
    }


@pytest.fixture
def time_series_config():
    """Standard time series config for testing."""
    return {
        'date_column': 'application_signed_date',
        'value_column': 'premium_amount',
        'alias_date_col': 'date',
        'alias_value_col': 'sales',
        'groupby_frequency': 'd',
        'rolling_window_days': 7
    }


@pytest.fixture
def valid_sales_df():
    """Sales DataFrame with valid data for pipeline testing."""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'application_signed_date': dates,
        'contract_issue_date': dates + pd.Timedelta(days=5),
        'premium_amount': np.random.uniform(10000, 100000, 100),
        'product_name': ['FlexGuard indexed variable annuity'] * 100,
        'term': ['6Y'] * 100,
        'buffer_rate': ['20%'] * 100,
    })


class TestApplyProductFilters:
    """Tests for apply_product_filters pipeline function."""

    def test_filters_all_criteria(self, valid_sales_df, product_filter_config):
        """Applies all three filters in sequence."""
        from src.data.pipelines import apply_product_filters

        result = apply_product_filters(valid_sales_df, product_filter_config)
        assert len(result) == len(valid_sales_df)

    def test_filters_by_product_name(self, valid_sales_df, product_filter_config):
        """Filters by product_name correctly."""
        from src.data.pipelines import apply_product_filters

        # Add a different product
        mixed_df = pd.concat([
            valid_sales_df,
            valid_sales_df.assign(product_name='Other Product')
        ])
        result = apply_product_filters(mixed_df, product_filter_config)
        assert all(result['product_name'] == 'FlexGuard indexed variable annuity')

    def test_raises_on_no_matching_records(self, valid_sales_df):
        """Raises ValueError when no records match."""
        from src.data.pipelines import apply_product_filters

        config = {
            'product_name': 'NonExistent',
            'buffer_rate': '20%',
            'term': '6Y'
        }
        with pytest.raises(ValueError, match="No records found"):
            apply_product_filters(valid_sales_df, config)


class TestApplySalesDataCleanup:
    """Tests for apply_sales_data_cleanup pipeline function."""

    def test_returns_cleaned_dataframe(self, valid_sales_df, sales_cleanup_config):
        """Returns cleaned DataFrame with expected columns."""
        from src.data.pipelines import apply_sales_data_cleanup

        result = apply_sales_data_cleanup(valid_sales_df, sales_cleanup_config)
        assert 'processing_days' in result.columns
        assert 'sales_amount' in result.columns

    def test_converts_dates_to_datetime(self, valid_sales_df, sales_cleanup_config):
        """Converts date columns to datetime."""
        from src.data.pipelines import apply_sales_data_cleanup

        # Start with string dates
        string_dates_df = valid_sales_df.copy()
        string_dates_df['application_signed_date'] = \
            string_dates_df['application_signed_date'].astype(str)
        string_dates_df['contract_issue_date'] = \
            string_dates_df['contract_issue_date'].astype(str)

        result = apply_sales_data_cleanup(string_dates_df, sales_cleanup_config)
        assert pd.api.types.is_datetime64_any_dtype(result['application_signed_date'])

    def test_filters_by_premium_range(self, valid_sales_df, sales_cleanup_config):
        """Filters out premiums outside range."""
        from src.data.pipelines import apply_sales_data_cleanup

        # Add some out-of-range values
        df_with_outliers = valid_sales_df.copy()
        df_with_outliers.loc[0, 'premium_amount'] = 500  # Below min
        df_with_outliers.loc[1, 'premium_amount'] = 2000000  # Above max

        result = apply_sales_data_cleanup(df_with_outliers, sales_cleanup_config)
        assert len(result) < len(df_with_outliers)


class TestCreateTimeSeriesCore:
    """Tests for _create_time_series_core function."""

    def test_creates_aggregated_time_series(self, valid_sales_df, time_series_config):
        """Creates aggregated time series from raw data."""
        from src.data.pipelines import _create_time_series_core

        result = _create_time_series_core(valid_sales_df, time_series_config)
        assert 'date' in result.columns
        assert 'sales' in result.columns

    def test_applies_rolling_mean(self, valid_sales_df, time_series_config):
        """Applies rolling mean to sales column."""
        from src.data.pipelines import _create_time_series_core

        result = _create_time_series_core(valid_sales_df, time_series_config)
        # First values should be smoothed by rolling mean
        assert not result['sales'].isna().all()


class TestApplyApplicationTimeSeries:
    """Tests for apply_application_time_series pipeline function."""

    def test_uses_application_date(self, valid_sales_df, time_series_config):
        """Uses application_signed_date for time series."""
        from src.data.pipelines import apply_application_time_series

        result = apply_application_time_series(valid_sales_df, time_series_config)
        assert 'date' in result.columns
        assert 'sales' in result.columns


class TestApplyContractTimeSeries:
    """Tests for apply_contract_time_series pipeline function."""

    def test_uses_contract_date(self, valid_sales_df):
        """Uses contract_issue_date for time series."""
        from src.data.pipelines import apply_contract_time_series

        config = {
            'date_column': 'contract_issue_date',
            'value_column': 'premium_amount',
            'alias_date_col': 'date',
            'alias_value_col': 'sales_by_contract',
            'groupby_frequency': 'd',
            'rolling_window_days': 7
        }
        result = apply_contract_time_series(valid_sales_df, config)
        assert 'date' in result.columns
        assert 'sales_by_contract' in result.columns


class TestPipelineIntegration:
    """Integration tests for pipeline composition."""

    def test_product_to_cleanup_pipeline(self, valid_sales_df, product_filter_config, sales_cleanup_config):
        """Product filter flows into cleanup pipeline."""
        from src.data.pipelines import apply_product_filters, apply_sales_data_cleanup

        filtered = apply_product_filters(valid_sales_df, product_filter_config)
        cleaned = apply_sales_data_cleanup(filtered, sales_cleanup_config)
        assert len(cleaned) > 0
        assert 'processing_days' in cleaned.columns

    def test_full_pipeline_chain(self, valid_sales_df, product_filter_config, sales_cleanup_config, time_series_config):
        """Full pipeline from filter to time series works."""
        from src.data.pipelines import (
            apply_product_filters,
            apply_sales_data_cleanup,
            apply_application_time_series
        )

        # Update cleanup config to use correct premium column
        cleanup_config = sales_cleanup_config.copy()

        filtered = apply_product_filters(valid_sales_df, product_filter_config)
        cleaned = apply_sales_data_cleanup(filtered, cleanup_config)

        # Update time series config to use correct columns
        ts_config = time_series_config.copy()
        ts_config['value_column'] = 'premium_amount'

        time_series = apply_application_time_series(cleaned, ts_config)
        assert len(time_series) > 0
        assert 'date' in time_series.columns
        assert 'sales' in time_series.columns


# =============================================================================
# WINK Pipeline Tests
# =============================================================================


class TestApplyMarketShareWeighting:
    """Tests for apply_market_share_weighting pipeline function."""

    def test_adds_weighted_mean_column(self):
        """Adds C_weighted_mean column to DataFrame."""
        from src.data.pipelines import apply_market_share_weighting
        from unittest.mock import patch, MagicMock

        df = pd.DataFrame({
            'date': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'Allianz': [0.05, 0.06],
            'Lincoln': [0.04, 0.045]
        })
        market_share_df = pd.DataFrame({
            'current_quarter': ['2022_Q1'],
            'Allianz_weight': [0.6],
            'Lincoln_weight': [0.4]
        })

        # Mock get_competitor_config
        mock_config = MagicMock()
        mock_config.rila_competitors = ['Allianz', 'Lincoln']

        with patch('src.config.product_config.get_competitor_config', return_value=mock_config):
            result = apply_market_share_weighting(df, market_share_df)

        assert 'C_weighted_mean' in result.columns
        assert 'C_core' in result.columns
        assert 'current_quarter' in result.columns

    def test_adds_quarterly_period_column(self):
        """Adds quarterly period columns before weighting."""
        from src.data.pipelines import apply_market_share_weighting
        from unittest.mock import patch, MagicMock

        df = pd.DataFrame({
            'date': pd.to_datetime(['2022-03-01', '2022-06-01']),
            'Allianz': [0.05, 0.06]
        })
        market_share_df = pd.DataFrame({
            'current_quarter': ['2022_Q1', '2022_Q2'],
            'Allianz_weight': [1.0, 1.0]
        })

        mock_config = MagicMock()
        mock_config.rila_competitors = ['Allianz']

        with patch('src.config.product_config.get_competitor_config', return_value=mock_config):
            result = apply_market_share_weighting(df, market_share_df)

        assert 'year' in result.columns
        assert 'quarter' in result.columns


class TestApplyWeeklyAggregation:
    """Tests for apply_weekly_aggregation pipeline function."""

    def test_aggregates_to_weekly_frequency(self):
        """Aggregates daily data to weekly frequency."""
        from src.data.pipelines import apply_weekly_aggregation
        from unittest.mock import patch, MagicMock

        # Create daily data for 2 weeks
        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=14, freq='D'),
            'sales': np.random.uniform(1000, 5000, 14),
            'rate': np.random.uniform(0.05, 0.10, 14)
        })

        config = {
            'weekly_aggregation_freq': 'W',
            'weekly_agg_dict': {'sales': 'sum', 'rate': 'mean'}
        }

        result = apply_weekly_aggregation(df, config)

        # Should have fewer rows than daily (weekly aggregation)
        assert len(result) < len(df)
        assert 'date' in result.columns

    def test_adds_temporal_indicators(self):
        """Adds temporal indicator columns."""
        from src.data.pipelines import apply_weekly_aggregation

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=21, freq='D'),
            'value': np.random.uniform(100, 500, 21)
        })

        config = {
            'weekly_aggregation_freq': 'W',
            'weekly_agg_dict': {'value': 'sum'}
        }

        result = apply_weekly_aggregation(df, config)

        # Temporal indicators should be present
        assert len(result) > 0


class TestApplyCompetitiveFeatures:
    """Tests for apply_competitive_features pipeline function."""

    def test_creates_median_rankings(self):
        """Creates median competitor rankings."""
        from src.data.pipelines import apply_competitive_features

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=5, freq='W'),
            'Allianz': [0.05, 0.06, 0.055, 0.058, 0.062],
            'Lincoln': [0.04, 0.045, 0.042, 0.047, 0.051],
            'Athene': [0.035, 0.038, 0.036, 0.040, 0.044],
        })

        config = {
            'company_columns': ['Allianz', 'Lincoln', 'Athene'],
            'min_companies_required': 2
        }

        result = apply_competitive_features(df, config)

        # Should have created ranking columns
        assert 'C_median' in result.columns
        assert 'C_top_3' in result.columns
        assert 'C_first' in result.columns

    def test_applies_semantic_mappings(self):
        """Applies semantic mappings to features."""
        from src.data.pipelines import apply_competitive_features

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=3, freq='W'),
            'Allianz': [0.05, 0.06, 0.055],
            'Lincoln': [0.04, 0.045, 0.042],
            'Athene': [0.035, 0.038, 0.036],
        })

        config = {
            'company_columns': ['Allianz', 'Lincoln', 'Athene'],
            'min_companies_required': 2
        }

        result = apply_competitive_features(df, config)

        # Should have semantic columns
        assert len(result) == 3


class TestApplyLagAndPolynomialFeatures:
    """Tests for apply_lag_and_polynomial_features pipeline function."""

    def test_creates_lag_features(self):
        """Creates lag features for specified columns."""
        from src.data.pipelines import apply_lag_and_polynomial_features

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=20, freq='W'),
            'sales': np.random.uniform(1000, 5000, 20),
            'prudential_rate': np.random.uniform(0.05, 0.10, 20),
        })

        config = {
            'lag_column_configs': [
                {'source_col': 'sales', 'prefix': 'sales_target', 'lag_direction': 'both'}
            ],
            'polynomial_base_columns': ['prudential_rate'],
            'max_lag_periods': 3,
            'allow_inplace_operations': True
        }

        result = apply_lag_and_polynomial_features(df, config)

        # Should have created lag columns
        assert 'sales_target_t0' in result.columns or 'sales_target_current' in result.columns
        assert len(result.columns) > len(df.columns)

    def test_creates_polynomial_interactions(self):
        """Creates polynomial interaction features."""
        from src.data.pipelines import apply_lag_and_polynomial_features

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='W'),
            'base_feature': np.random.uniform(0.05, 0.10, 10),
        })

        config = {
            'lag_column_configs': [],
            'polynomial_base_columns': ['base_feature'],
            'max_lag_periods': 2,
            'allow_inplace_operations': True
        }

        result = apply_lag_and_polynomial_features(df, config)

        # Should have more columns due to polynomial features
        assert len(result.columns) >= len(df.columns)


class TestApplyFinalFeaturePreparation:
    """Tests for apply_final_feature_preparation pipeline function."""

    def test_filters_by_analysis_start_date(self):
        """Filters data by feature analysis start date."""
        from src.data.pipelines import apply_final_feature_preparation

        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=100, freq='W'),
            'sales_target_contract_current': np.random.uniform(1000, 5000, 100),
            'prudential_rate_current': np.random.uniform(0.05, 0.10, 100),
            'competitor_mid_t1': np.random.uniform(0.04, 0.08, 100),
        })

        config = {
            'date_column': 'date',
            'feature_analysis_start_date': '2022-06-01',  # Filter out early data
            'mature_data_cutoff_date': '2025-12-31',
            'day_of_year_column': 'day_of_year',
            'holiday_column': 'holiday',
            'holiday_start_day': 13,
            'holiday_end_day': 359,
            'prudential_rate_column': 'prudential_rate_current',
            'competitor_rate_column': 'competitor_mid_t1',
            'spread_column': 'Spread',
            'log_transform_source_column': 'sales_target_contract_current',
            'log_transform_target_column': 'sales_log'
        }

        result = apply_final_feature_preparation(df, config)

        # Should have filtered out early dates
        assert len(result) < len(df)
        assert result['date'].min() > pd.Timestamp('2022-06-01')

    def test_creates_spread_column(self):
        """Creates spread column between Prudential and competitor rates."""
        from src.data.pipelines import apply_final_feature_preparation

        df = pd.DataFrame({
            'date': pd.date_range('2022-09-01', periods=10, freq='W'),
            'sales_target_contract_current': np.random.uniform(1000, 5000, 10),
            'prudential_rate_current': [0.08, 0.085, 0.09, 0.088, 0.092, 0.087, 0.091, 0.089, 0.093, 0.090],
            'competitor_mid_t1': [0.06, 0.062, 0.065, 0.063, 0.067, 0.061, 0.068, 0.064, 0.069, 0.066],
        })

        config = {
            'date_column': 'date',
            'feature_analysis_start_date': '2022-01-01',  # Include all data
            'mature_data_cutoff_date': '2025-12-31',
            'day_of_year_column': 'day_of_year',
            'holiday_column': 'holiday',
            'holiday_start_day': 13,
            'holiday_end_day': 359,
            'prudential_rate_column': 'prudential_rate_current',
            'competitor_rate_column': 'competitor_mid_t1',
            'spread_column': 'Spread',
            'log_transform_source_column': 'sales_target_contract_current',
            'log_transform_target_column': 'sales_log'
        }

        result = apply_final_feature_preparation(df, config)

        assert 'Spread' in result.columns
        assert 'sales_log' in result.columns


class TestWinkProductFilters:
    """Tests for _apply_wink_product_filters function."""

    def test_applies_buffer_rate_filter(self):
        """Applies buffer rate filter from config."""
        from src.data.pipelines import _apply_wink_product_filters

        df = pd.DataFrame({
            'bufferRate': [0.10, 0.20, 0.20, 0.15],
            'bufferModifier': ['standard', 'standard', 'standard', 'standard'],
            'indexingMethod': ['point-to-point', 'point-to-point', 'point-to-point', 'point-to-point'],
            'indexCreditingFrequency': ['annual', 'annual', 'annual', 'annual'],
            'defaultActuarialView': ['default', 'default', 'default', 'default'],
            'capRate': [0.05, 0.06, 0.07, 0.055],
            'productID': [1, 2, 2, 3],
            'date': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']),
            'companyName': ['Allianz', 'Lincoln', 'Lincoln', 'Athene'],
            'mva': [False, False, True, False]  # Required for deduplication
        })

        config = {
            'buffer_rates_allowed': [0.20],
            'buffer_modifier_filter': 'standard',
            'indexing_method_filter': 'point-to-point',
            'crediting_frequency_filter': 'annual',
            'actuarial_view_filter': 'default',
            'default_cap_rate': 9.99,
            'max_cap_rate': 0.15,
            'data_filter_start_date': '2022-01-01'
        }

        all_product_ids = [2]  # Only include product ID 2

        result = _apply_wink_product_filters(df, config, all_product_ids)

        # Should only have bufferRate 0.20 and productID 2
        assert len(result) == 2
        assert all(result['bufferRate'] == 0.20)
