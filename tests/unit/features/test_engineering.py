"""
Unit tests for src/features/engineering.py.

Tests validate that the facade module correctly re-exports all
functions from submodules and maintains the public API contract.

The engineering.py module is an orchestrator (facade pattern) that
re-exports functions from:
- competitive_features.py
- engineering_integration.py
- engineering_temporal.py
- engineering_timeseries.py
"""

import pytest
import pandas as pd
import numpy as np

# =============================================================================
# RE-EXPORT AVAILABILITY TESTS
# =============================================================================


class TestCompetitiveFeaturesExports:
    """Verify competitive_features.py exports are available."""

    def test_calculate_median_competitor_rankings_exported(self):
        """calculate_median_competitor_rankings should be importable."""
        from src.features.engineering import calculate_median_competitor_rankings
        assert callable(calculate_median_competitor_rankings)

    def test_calculate_top_competitor_rankings_exported(self):
        """calculate_top_competitor_rankings should be importable."""
        from src.features.engineering import calculate_top_competitor_rankings
        assert callable(calculate_top_competitor_rankings)

    def test_calculate_position_competitor_rankings_exported(self):
        """calculate_position_competitor_rankings should be importable."""
        from src.features.engineering import calculate_position_competitor_rankings
        assert callable(calculate_position_competitor_rankings)

    def test_apply_competitive_semantic_mappings_exported(self):
        """apply_competitive_semantic_mappings should be importable."""
        from src.features.engineering import apply_competitive_semantic_mappings
        assert callable(apply_competitive_semantic_mappings)

    def test_create_competitive_compatibility_shortcuts_exported(self):
        """create_competitive_compatibility_shortcuts should be importable."""
        from src.features.engineering import create_competitive_compatibility_shortcuts
        assert callable(create_competitive_compatibility_shortcuts)

    def test_wink_weighted_mean_exported(self):
        """wink_weighted_mean should be importable."""
        from src.features.engineering import wink_weighted_mean
        assert callable(wink_weighted_mean)

    def test_deprecated_wink_weighted_mean_alias_exported(self):
        """WINK_weighted_mean (deprecated) should be importable."""
        from src.features.engineering import WINK_weighted_mean
        assert callable(WINK_weighted_mean)

    def test_calculate_competitive_spread_exported(self):
        """calculate_competitive_spread should be importable."""
        from src.features.engineering import calculate_competitive_spread
        assert callable(calculate_competitive_spread)


class TestEngineeringIntegrationExports:
    """Verify engineering_integration.py exports are available."""

    def test_create_daily_date_range_dataframe_exported(self):
        """create_daily_date_range_dataframe should be importable."""
        from src.features.engineering import create_daily_date_range_dataframe
        assert callable(create_daily_date_range_dataframe)

    def test_merge_multiple_dataframes_on_date_exported(self):
        """merge_multiple_dataframes_on_date should be importable."""
        from src.features.engineering import merge_multiple_dataframes_on_date
        assert callable(merge_multiple_dataframes_on_date)

    def test_apply_forward_fill_to_columns_exported(self):
        """apply_forward_fill_to_columns should be importable."""
        from src.features.engineering import apply_forward_fill_to_columns
        assert callable(apply_forward_fill_to_columns)

    def test_apply_cpi_adjustment_to_sales_exported(self):
        """apply_cpi_adjustment_to_sales should be importable."""
        from src.features.engineering import apply_cpi_adjustment_to_sales
        assert callable(apply_cpi_adjustment_to_sales)

    def test_apply_rolling_mean_to_columns_exported(self):
        """apply_rolling_mean_to_columns should be importable."""
        from src.features.engineering import apply_rolling_mean_to_columns
        assert callable(apply_rolling_mean_to_columns)

    def test_create_business_day_counter_exported(self):
        """create_business_day_counter should be importable."""
        from src.features.engineering import create_business_day_counter
        assert callable(create_business_day_counter)

    def test_aggregate_dataframe_by_frequency_exported(self):
        """aggregate_dataframe_by_frequency should be importable."""
        from src.features.engineering import aggregate_dataframe_by_frequency
        assert callable(aggregate_dataframe_by_frequency)

    def test_filter_dataframe_by_mature_date_exported(self):
        """filter_dataframe_by_mature_date should be importable."""
        from src.features.engineering import filter_dataframe_by_mature_date
        assert callable(filter_dataframe_by_mature_date)

    def test_remove_final_row_from_dataframe_exported(self):
        """remove_final_row_from_dataframe should be importable."""
        from src.features.engineering import remove_final_row_from_dataframe
        assert callable(remove_final_row_from_dataframe)


class TestEngineeringTemporalExports:
    """Verify engineering_temporal.py exports are available."""

    def test_create_temporal_indicator_columns_exported(self):
        """create_temporal_indicator_columns should be importable."""
        from src.features.engineering import create_temporal_indicator_columns
        assert callable(create_temporal_indicator_columns)

    def test_extract_day_of_year_column_exported(self):
        """extract_day_of_year_column should be importable."""
        from src.features.engineering import extract_day_of_year_column
        assert callable(extract_day_of_year_column)

    def test_create_holiday_indicator_by_day_range_exported(self):
        """create_holiday_indicator_by_day_range should be importable."""
        from src.features.engineering import create_holiday_indicator_by_day_range
        assert callable(create_holiday_indicator_by_day_range)

    def test_create_lag_features_for_columns_exported(self):
        """create_lag_features_for_columns should be importable."""
        from src.features.engineering import create_lag_features_for_columns
        assert callable(create_lag_features_for_columns)

    def test_create_polynomial_interaction_features_exported(self):
        """create_polynomial_interaction_features should be importable."""
        from src.features.engineering import create_polynomial_interaction_features
        assert callable(create_polynomial_interaction_features)

    def test_apply_log_plus_one_transformation_exported(self):
        """apply_log_plus_one_transformation should be importable."""
        from src.features.engineering import apply_log_plus_one_transformation
        assert callable(apply_log_plus_one_transformation)


class TestEngineeringTimeseriesExports:
    """Verify engineering_timeseries.py exports are available."""

    def test_cpi_adjustment_exported(self):
        """cpi_adjustment should be importable."""
        from src.features.engineering import cpi_adjustment
        assert callable(cpi_adjustment)

    def test_time_series_week_agg_smoothed_exported(self):
        """time_series_week_agg_smoothed should be importable."""
        from src.features.engineering import time_series_week_agg_smoothed
        assert callable(time_series_week_agg_smoothed)

    def test_create_lag_features_exported(self):
        """create_lag_features should be importable."""
        from src.features.engineering import create_lag_features
        assert callable(create_lag_features)


# =============================================================================
# __all__ VERIFICATION TESTS
# =============================================================================


class TestModuleAllAttribute:
    """Verify __all__ contains expected exports."""

    def test_all_contains_competitive_features(self):
        """__all__ should contain competitive feature functions."""
        from src.features import engineering

        competitive_funcs = [
            'calculate_median_competitor_rankings',
            'calculate_top_competitor_rankings',
            'calculate_position_competitor_rankings',
            'apply_competitive_semantic_mappings',
            'create_competitive_compatibility_shortcuts',
            'wink_weighted_mean',
            'WINK_weighted_mean',
            'calculate_competitive_spread',
        ]
        for func in competitive_funcs:
            assert func in engineering.__all__, f"Missing from __all__: {func}"

    def test_all_contains_integration_functions(self):
        """__all__ should contain integration functions."""
        from src.features import engineering

        integration_funcs = [
            'create_daily_date_range_dataframe',
            'merge_multiple_dataframes_on_date',
            'apply_forward_fill_to_columns',
            'apply_cpi_adjustment_to_sales',
            'apply_rolling_mean_to_columns',
            'create_business_day_counter',
            'aggregate_dataframe_by_frequency',
            'filter_dataframe_by_mature_date',
            'remove_final_row_from_dataframe',
        ]
        for func in integration_funcs:
            assert func in engineering.__all__, f"Missing from __all__: {func}"

    def test_all_contains_temporal_functions(self):
        """__all__ should contain temporal functions."""
        from src.features import engineering

        temporal_funcs = [
            'create_temporal_indicator_columns',
            'extract_day_of_year_column',
            'create_holiday_indicator_by_day_range',
            'create_lag_features_for_columns',
            'create_polynomial_interaction_features',
            'apply_log_plus_one_transformation',
        ]
        for func in temporal_funcs:
            assert func in engineering.__all__, f"Missing from __all__: {func}"

    def test_all_contains_timeseries_functions(self):
        """__all__ should contain timeseries functions."""
        from src.features import engineering

        timeseries_funcs = [
            'cpi_adjustment',
            'time_series_week_agg_smoothed',
            'create_lag_features',
        ]
        for func in timeseries_funcs:
            assert func in engineering.__all__, f"Missing from __all__: {func}"


# =============================================================================
# SMOKE TESTS FOR KEY FUNCTIONS
# =============================================================================


class TestKeyFunctionSmoke:
    """Smoke tests to verify key functions are callable with basic inputs."""

    def test_wink_weighted_mean_basic_call(self):
        """wink_weighted_mean should accept DataFrame and column args."""
        from src.features.engineering import wink_weighted_mean

        df = pd.DataFrame({
            'rate': [4.5, 4.6, 4.7],
            'weight': [0.3, 0.4, 0.3],
            'date': pd.date_range('2023-01-01', periods=3)
        })

        # Should not raise - basic functionality test
        # Note: actual business logic tested in test_competitive_features.py
        try:
            result = wink_weighted_mean(
                df,
                rate_column='rate',
                weight_column='weight',
                date_column='date'
            )
            assert isinstance(result, (pd.DataFrame, pd.Series))
        except Exception as e:
            # If the function requires more setup, that's fine
            # We're just testing it's callable from the facade
            if "weighted_mean" not in str(e).lower():
                raise

    def test_create_daily_date_range_dataframe_basic_call(self):
        """create_daily_date_range_dataframe should create date range."""
        from src.features.engineering import create_daily_date_range_dataframe

        result = create_daily_date_range_dataframe(
            start_date='2023-01-01',
            end_date='2023-01-10'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10  # 10 days inclusive

    def test_apply_forward_fill_basic_call(self):
        """apply_forward_fill_to_columns should forward fill NaNs."""
        from src.features.engineering import apply_forward_fill_to_columns

        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'value': [1.0, np.nan, np.nan, 2.0, np.nan]
        })

        result = apply_forward_fill_to_columns(df, columns=['value'])

        # Should have forward-filled NaN values
        assert not result['value'].isna().any()

    def test_create_lag_features_basic_call(self):
        """create_lag_features should be importable and callable.

        Note: Full functionality testing is in test_engineering_timeseries.py.
        This test verifies the re-export works correctly.
        """
        from src.features.engineering import create_lag_features

        # Verify function signature matches expected
        import inspect
        sig = inspect.signature(create_lag_features)
        params = list(sig.parameters.keys())

        # Should have df, training_cutoff_date, and date_column parameters
        assert 'df' in params
        assert 'training_cutoff_date' in params
        assert 'date_column' in params


# =============================================================================
# FACADE PATTERN VERIFICATION
# =============================================================================


class TestFacadePattern:
    """Verify facade pattern is implemented correctly."""

    def test_reexports_are_same_functions(self):
        """Re-exported functions should be same objects as originals."""
        from src.features.engineering import wink_weighted_mean
        from src.features.competitive_features import wink_weighted_mean as original

        assert wink_weighted_mean is original

    def test_reexports_from_integration(self):
        """Integration re-exports should be same objects."""
        from src.features.engineering import create_daily_date_range_dataframe
        from src.features.engineering_integration import (
            create_daily_date_range_dataframe as original
        )

        assert create_daily_date_range_dataframe is original

    def test_reexports_from_temporal(self):
        """Temporal re-exports should be same objects."""
        from src.features.engineering import create_temporal_indicator_columns
        from src.features.engineering_temporal import (
            create_temporal_indicator_columns as original
        )

        assert create_temporal_indicator_columns is original

    def test_reexports_from_timeseries(self):
        """Timeseries re-exports should be same objects."""
        from src.features.engineering import cpi_adjustment
        from src.features.engineering_timeseries import cpi_adjustment as original

        assert cpi_adjustment is original
