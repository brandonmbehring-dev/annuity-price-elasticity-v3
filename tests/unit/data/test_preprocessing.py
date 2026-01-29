"""
Unit tests for src/data/preprocessing.py

Tests data filtering, datetime conversion, data cleaning, and
WINK data processing functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestFilterDataframeByProductName:
    """Tests for filter_dataframe_by_product_name function."""

    def test_filters_by_exact_product_name(self, sample_sales_df):
        """Filters DataFrame to exact product name match."""
        from src.data.preprocessing import filter_dataframe_by_product_name

        result = filter_dataframe_by_product_name(
            sample_sales_df, 'FlexGuard indexed variable annuity'
        )
        assert len(result) == len(sample_sales_df)

    def test_raises_on_missing_column(self):
        """Raises ValueError if product_name column missing."""
        from src.data.preprocessing import filter_dataframe_by_product_name

        df = pd.DataFrame({'other_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="product_name.*not found"):
            filter_dataframe_by_product_name(df, 'Product')

    def test_raises_on_empty_product_name(self, sample_sales_df):
        """Raises ValueError for empty product name."""
        from src.data.preprocessing import filter_dataframe_by_product_name

        with pytest.raises(ValueError, match="cannot be empty"):
            filter_dataframe_by_product_name(sample_sales_df, '')

    def test_raises_on_no_matching_records(self, sample_sales_df):
        """Raises ValueError when no records match."""
        from src.data.preprocessing import filter_dataframe_by_product_name

        with pytest.raises(ValueError, match="No records found"):
            filter_dataframe_by_product_name(sample_sales_df, 'NonExistentProduct')

    def test_returns_copy(self, sample_sales_df):
        """Returns a copy, not a view."""
        from src.data.preprocessing import filter_dataframe_by_product_name

        result = filter_dataframe_by_product_name(
            sample_sales_df, 'FlexGuard indexed variable annuity'
        )
        assert result is not sample_sales_df


class TestFilterByBufferRate:
    """Tests for filter_by_buffer_rate function."""

    def test_filters_by_buffer_rate(self, sample_sales_df):
        """Filters DataFrame by buffer rate value."""
        from src.data.preprocessing import filter_by_buffer_rate

        result = filter_by_buffer_rate(sample_sales_df, '20%')
        assert len(result) == len(sample_sales_df)

    def test_raises_on_missing_column(self):
        """Raises ValueError if buffer_rate column missing."""
        from src.data.preprocessing import filter_by_buffer_rate

        df = pd.DataFrame({'other_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="buffer_rate.*not found"):
            filter_by_buffer_rate(df, '20%')

    def test_raises_on_no_matching_records(self, sample_sales_df):
        """Raises ValueError when no records match."""
        from src.data.preprocessing import filter_by_buffer_rate

        with pytest.raises(ValueError, match="No records found"):
            filter_by_buffer_rate(sample_sales_df, '99%')


class TestFilterByTerm:
    """Tests for filter_by_term function."""

    def test_filters_by_term(self, sample_sales_df):
        """Filters DataFrame by term value."""
        from src.data.preprocessing import filter_by_term

        result = filter_by_term(sample_sales_df, '6Y')
        assert len(result) == len(sample_sales_df)

    def test_raises_on_missing_column(self):
        """Raises ValueError if term column missing."""
        from src.data.preprocessing import filter_by_term

        df = pd.DataFrame({'other_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="term.*not found"):
            filter_by_term(df, '6Y')

    def test_raises_on_no_matching_records(self, sample_sales_df):
        """Raises ValueError when no records match."""
        from src.data.preprocessing import filter_by_term

        with pytest.raises(ValueError, match="No records found"):
            filter_by_term(sample_sales_df, '99Y')


class TestConvertColumnToDatetime:
    """Tests for convert_column_to_datetime function."""

    def test_converts_string_dates(self):
        """Converts string date column to datetime."""
        from src.data.preprocessing import convert_column_to_datetime

        df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02']})
        result = convert_column_to_datetime(df, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])

    def test_raises_on_missing_column(self):
        """Raises ValueError if column not found."""
        from src.data.preprocessing import convert_column_to_datetime

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            convert_column_to_datetime(df, 'date')

    def test_raises_on_invalid_date_format(self):
        """Raises ValueError on unparseable dates."""
        from src.data.preprocessing import convert_column_to_datetime

        df = pd.DataFrame({'date': ['not-a-date', 'also-not']})
        with pytest.raises(ValueError, match="Failed to convert"):
            convert_column_to_datetime(df, 'date')

    def test_returns_copy(self):
        """Returns copy without modifying original."""
        from src.data.preprocessing import convert_column_to_datetime

        df = pd.DataFrame({'date': ['2022-01-01']})
        result = convert_column_to_datetime(df, 'date')
        assert df['date'].dtype == object  # Original unchanged


class TestCalculateDaysBetweenDates:
    """Tests for calculate_days_between_dates function."""

    def test_calculates_correct_days(self):
        """Calculates days between two date columns."""
        from src.data.preprocessing import calculate_days_between_dates

        df = pd.DataFrame({
            'start': pd.to_datetime(['2022-01-01', '2022-01-10']),
            'end': pd.to_datetime(['2022-01-05', '2022-01-20'])
        })
        result = calculate_days_between_dates(df, 'start', 'end', 'days')
        assert result['days'].tolist() == [4, 10]

    def test_raises_on_missing_columns(self):
        """Raises ValueError if date columns missing."""
        from src.data.preprocessing import calculate_days_between_dates

        df = pd.DataFrame({'other': [1]})
        with pytest.raises(ValueError, match="Missing date columns"):
            calculate_days_between_dates(df, 'start', 'end', 'days')

    def test_raises_on_non_datetime_columns(self):
        """Raises ValueError if columns not datetime type."""
        from src.data.preprocessing import calculate_days_between_dates

        df = pd.DataFrame({'start': ['2022-01-01'], 'end': ['2022-01-05']})
        with pytest.raises(ValueError, match="must be datetime"):
            calculate_days_between_dates(df, 'start', 'end', 'days')


class TestRemoveNullValuesFromColumn:
    """Tests for remove_null_values_from_column function."""

    def test_removes_null_rows(self):
        """Removes rows with null values in specified column."""
        from src.data.preprocessing import remove_null_values_from_column

        df = pd.DataFrame({'col': [1, None, 3, None, 5]})
        result = remove_null_values_from_column(df, 'col')
        assert len(result) == 3

    def test_raises_on_missing_column(self):
        """Raises ValueError if column not found."""
        from src.data.preprocessing import remove_null_values_from_column

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            remove_null_values_from_column(df, 'col')

    def test_raises_on_all_null(self):
        """Raises ValueError if all values are null."""
        from src.data.preprocessing import remove_null_values_from_column

        df = pd.DataFrame({'col': [None, None, None]})
        with pytest.raises(ValueError, match="All values.*are null"):
            remove_null_values_from_column(df, 'col')


class TestFilterColumnByRange:
    """Tests for filter_column_by_range function."""

    def test_filters_by_numeric_range(self):
        """Filters DataFrame by numeric range."""
        from src.data.preprocessing import filter_column_by_range

        df = pd.DataFrame({'value': [10, 50, 100, 150, 200]})
        result = filter_column_by_range(df, 'value', 25, 175)
        assert len(result) == 3
        assert result['value'].tolist() == [50, 100, 150]

    def test_raises_on_missing_column(self):
        """Raises ValueError if column not found."""
        from src.data.preprocessing import filter_column_by_range

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            filter_column_by_range(df, 'value', 0, 100)

    def test_raises_on_invalid_range(self):
        """Raises ValueError if min >= max."""
        from src.data.preprocessing import filter_column_by_range

        df = pd.DataFrame({'value': [50]})
        with pytest.raises(ValueError, match="must be less than"):
            filter_column_by_range(df, 'value', 100, 50)

    def test_raises_on_no_values_in_range(self):
        """Raises ValueError if no values in range."""
        from src.data.preprocessing import filter_column_by_range

        df = pd.DataFrame({'value': [10, 20, 30]})
        with pytest.raises(ValueError, match="No values found"):
            filter_column_by_range(df, 'value', 100, 200)


class TestCreateColumnAlias:
    """Tests for create_column_alias function."""

    def test_creates_alias_column(self):
        """Creates alias column from source."""
        from src.data.preprocessing import create_column_alias

        df = pd.DataFrame({'source': [1, 2, 3]})
        result = create_column_alias(df, 'source', 'target')
        assert 'target' in result.columns
        assert result['target'].tolist() == [1, 2, 3]

    def test_raises_on_missing_source(self):
        """Raises ValueError if source column missing."""
        from src.data.preprocessing import create_column_alias

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="Source column.*not found"):
            create_column_alias(df, 'source', 'target')

    def test_preserves_existing_target(self):
        """Does not overwrite existing target column."""
        from src.data.preprocessing import create_column_alias

        df = pd.DataFrame({'source': [1, 2], 'target': [10, 20]})
        result = create_column_alias(df, 'source', 'target')
        assert result['target'].tolist() == [10, 20]


class TestApplyQuantileThresholdFilter:
    """Tests for apply_quantile_threshold_filter function."""

    def test_filters_by_quantile(self):
        """Filters values below quantile threshold."""
        from src.data.preprocessing import apply_quantile_threshold_filter

        df = pd.DataFrame({'value': list(range(1, 101))})  # 1-100
        result = apply_quantile_threshold_filter(df, 'value', 0.95)
        # 95th percentile of 1-100 is ~95, so values < 95 remain
        assert len(result) < len(df)

    def test_raises_on_missing_column(self):
        """Raises ValueError if column missing."""
        from src.data.preprocessing import apply_quantile_threshold_filter

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            apply_quantile_threshold_filter(df, 'value', 0.95)

    def test_raises_on_invalid_quantile(self):
        """Raises ValueError if quantile not in [0, 1]."""
        from src.data.preprocessing import apply_quantile_threshold_filter

        df = pd.DataFrame({'value': [1, 2, 3]})
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            apply_quantile_threshold_filter(df, 'value', 1.5)


class TestSortDataframeByColumn:
    """Tests for sort_dataframe_by_column function."""

    def test_sorts_ascending(self):
        """Sorts DataFrame ascending by column."""
        from src.data.preprocessing import sort_dataframe_by_column

        df = pd.DataFrame({'value': [3, 1, 2]})
        result = sort_dataframe_by_column(df, 'value')
        assert result['value'].tolist() == [1, 2, 3]

    def test_raises_on_missing_column(self):
        """Raises ValueError if column missing."""
        from src.data.preprocessing import sort_dataframe_by_column

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            sort_dataframe_by_column(df, 'value')


class TestRenameColumnsWithMapping:
    """Tests for rename_columns_with_mapping function."""

    def test_renames_columns(self):
        """Renames columns using mapping."""
        from src.data.preprocessing import rename_columns_with_mapping

        df = pd.DataFrame({'old_name': [1, 2]})
        result = rename_columns_with_mapping(df, {'old_name': 'new_name'})
        assert 'new_name' in result.columns
        assert 'old_name' not in result.columns

    def test_raises_on_missing_source_columns(self):
        """Raises ValueError if source columns missing."""
        from src.data.preprocessing import rename_columns_with_mapping

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="Cannot rename missing"):
            rename_columns_with_mapping(df, {'missing': 'new'})


class TestApplyRollingMean:
    """Tests for apply_rolling_mean function."""

    def test_applies_rolling_mean(self):
        """Applies rolling mean to column."""
        from src.data.preprocessing import apply_rolling_mean

        df = pd.DataFrame({'value': [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = apply_rolling_mean(df, 'value', 3)
        # Rolling mean with window=3 and min_periods=1
        assert result['value'].iloc[2] == 20.0  # (10+20+30)/3

    def test_raises_on_missing_column(self):
        """Raises ValueError if column missing."""
        from src.data.preprocessing import apply_rolling_mean

        df = pd.DataFrame({'other': [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            apply_rolling_mean(df, 'value', 3)

    def test_raises_on_invalid_window(self):
        """Raises ValueError if window <= 0."""
        from src.data.preprocessing import apply_rolling_mean

        df = pd.DataFrame({'value': [1, 2, 3]})
        with pytest.raises(ValueError, match="must be positive"):
            apply_rolling_mean(df, 'value', 0)


class TestStandardizeWinkCompanyNames:
    """Tests for standardize_wink_company_names function."""

    def test_standardizes_prudential(self):
        """Standardizes 'pruco life' to 'Prudential'."""
        from src.data.preprocessing import standardize_wink_company_names

        df = pd.DataFrame({'companyName': ['pruco life', 'PRUCO LIFE']})
        result = standardize_wink_company_names(df)
        assert result['companyName'].iloc[0] == 'Prudential'

    def test_standardizes_multiple_companies(self):
        """Standardizes multiple company name variants."""
        from src.data.preprocessing import standardize_wink_company_names

        df = pd.DataFrame({'companyName': ['allianz', 'athene annuity and life']})
        result = standardize_wink_company_names(df)
        assert result['companyName'].tolist() == ['Allianz', 'Athene']


class TestRoundRateColumns:
    """Tests for round_rate_columns function."""

    def test_rounds_specified_columns(self):
        """Rounds specified rate columns."""
        from src.data.preprocessing import round_rate_columns

        df = pd.DataFrame({'rate1': [1.23456], 'rate2': [2.34567]})
        result = round_rate_columns(df, ['rate1', 'rate2'], 2)
        assert result['rate1'].iloc[0] == 1.23
        assert result['rate2'].iloc[0] == 2.35

    def test_ignores_missing_columns(self):
        """Ignores columns not in DataFrame."""
        from src.data.preprocessing import round_rate_columns

        df = pd.DataFrame({'rate1': [1.23456]})
        result = round_rate_columns(df, ['rate1', 'missing'], 2)
        assert result['rate1'].iloc[0] == 1.23


class TestHandleNullCapRates:
    """Tests for handle_null_cap_rates function."""

    def test_fills_null_cap_rates(self):
        """Fills null capRate values with default."""
        from src.data.preprocessing import handle_null_cap_rates

        df = pd.DataFrame({'capRate': [1.0, None, 3.0]})
        result = handle_null_cap_rates(df, 9.99)
        assert result['capRate'].tolist() == [1.0, 9.99, 3.0]


class TestApplyCapRateCeiling:
    """Tests for apply_cap_rate_ceiling function."""

    def test_clips_cap_rates(self):
        """Clips capRate values above ceiling."""
        from src.data.preprocessing import apply_cap_rate_ceiling

        df = pd.DataFrame({'capRate': [1.0, 5.0, 10.0]})
        result = apply_cap_rate_ceiling(df, 4.5)
        assert result['capRate'].tolist() == [1.0, 4.5, 4.5]


class TestAddQuarterlyPeriodColumn:
    """Tests for add_quarterly_period_column function."""

    def test_adds_quarterly_columns(self):
        """Adds year, quarter, and current_quarter columns."""
        from src.data.preprocessing import add_quarterly_period_column

        df = pd.DataFrame({'date': pd.to_datetime(['2022-03-15', '2022-07-20'])})
        result = add_quarterly_period_column(df, 'date')
        assert 'year' in result.columns
        assert 'quarter' in result.columns
        assert 'current_quarter' in result.columns
        assert result['current_quarter'].tolist() == ['2022_Q1', '2022_Q3']
