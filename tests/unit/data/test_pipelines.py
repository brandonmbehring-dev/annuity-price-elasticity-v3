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
