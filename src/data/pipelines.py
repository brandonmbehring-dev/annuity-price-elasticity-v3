"""
Convenience pipeline functions for clean_v0 RILA price elasticity analysis.

This module provides 11 pipeline functions that eliminate feature duplication
and capture the complete clean_v0 workflow. Each pipeline function groups
3-5 related atomic functions while maintaining DRY principles.

Critical fixes from initial implementation:
- Eliminates C_weighted_mean duplication between pipelines
- Adds missing market share weighting pipeline
- Adds missing lag feature engineering pipeline
- Adds missing data integration and weekly aggregation pipeline
- Maintains separation between processing and I/O operations

Following CODING_STANDARDS.md principles:
- Single responsibility pipeline functions
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
- Professional documentation with NumPy-style docstrings
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import modules - Fail fast with clear error if imports fail
from . import extraction as ext
from . import preprocessing as prep
from ..features import engineering as eng
from ..visualization import visualization as viz
from ..config import config_builder

# Import pipeline-specific configuration types
from ..config.pipeline_config import (
    ProductFilterConfig, SalesCleanupConfig, TimeSeriesConfig,
    WinkProcessingConfig, CompetitiveConfig,
    FeatureConfig, LagFeatureConfig, FeatureSelectionStageConfig
)


# =============================================================================
# Pipeline 1: Product Filtering
# =============================================================================

def apply_product_filters(df: pd.DataFrame, config: ProductFilterConfig) -> pd.DataFrame:
    """Apply standard annuity product filtering pipeline.

    Combines 3 atomic filtering functions for consistent product selection
    across different annuity products. Filters by product name, buffer rate,
    and term in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw product data containing product_name, buffer_rate, and term columns
    config : ProductFilterConfig
        Configuration containing product_name, buffer_rate, and term filter values

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records matching all filter criteria

    Raises
    ------
    ValueError
        If required columns are missing or no records match filter criteria

    Examples
    --------
    >>> config = {'product_name': 'FlexGuard indexed variable annuity',
    ...           'buffer_rate': '20%', 'term': '6Y'}
    >>> filtered_df = apply_product_filters(raw_df, config)
    >>> assert len(filtered_df) > 0
    """
    return (df
            .pipe(prep.filter_dataframe_by_product_name, config['product_name'])
            .pipe(prep.filter_by_buffer_rate, config['buffer_rate'])
            .pipe(prep.filter_by_term, config['term']))


# =============================================================================
# Pipeline 2: Sales Data Cleanup
# =============================================================================

def apply_sales_data_cleanup(df: pd.DataFrame, config: SalesCleanupConfig) -> pd.DataFrame:
    """Apply standard sales data cleaning and validation pipeline.

    Combines 7 atomic functions for comprehensive sales data preparation:
    datetime conversions, processing day calculations, null handling,
    range filtering, column aliasing, and quantile-based outlier removal.

    Includes production validation checkpoint with fail-fast error handling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales data with date and premium amount columns
    config : SalesCleanupConfig
        Configuration containing cleanup parameters and column names

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for time series creation

    Raises
    ------
    ValueError
        If required columns are missing, all data is filtered out, or validation fails

    Examples
    --------
    >>> config = {'min_premium': 1000.0, 'max_premium': 1000000.0,
    ...           'quantile_threshold': 0.99, 'start_date_col': 'application_signed_date',
    ...           'end_date_col': 'contract_issue_date', 'processing_days_col': 'processing_days',
    ...           'premium_column': 'contract_initial_premium_amount', 'sales_alias_col': 'sales_amount'}
    >>> clean_df = apply_sales_data_cleanup(raw_df, config)
    >>> assert config['sales_alias_col'] in clean_df.columns
    """
    from src.validation.pipeline_validation_helpers import validate_preprocessing_output

    df_cleaned = (df
            .pipe(prep.convert_column_to_datetime, config['start_date_col'])
            .pipe(prep.convert_column_to_datetime, config['end_date_col'])
            .pipe(prep.calculate_days_between_dates, config['start_date_col'],
                  config['end_date_col'], config['processing_days_col'])
            .pipe(prep.remove_null_values_from_column, config['premium_column'])
            .pipe(prep.filter_column_by_range, config['premium_column'],
                  config['min_premium'], config['max_premium'])
            .pipe(prep.create_column_alias, config['premium_column'], config['sales_alias_col'])
            .pipe(prep.apply_quantile_threshold_filter, config['premium_column'],
                  config['quantile_threshold']))

    # Production validation checkpoint
    return validate_preprocessing_output(
        df=df_cleaned,
        stage_name="sales_data_cleanup",
        date_column=config['start_date_col'],
        allow_shrinkage=True  # Filtering and cleanup may legitimately reduce row counts
    )


# =============================================================================
# Pipeline 3: Time Series Creation (DRY Shared Core)
# =============================================================================

def _create_time_series_core(df: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Shared core function for time series creation (DRY principle)."""
    return (df[[config['date_column'], config['value_column']]]
            .pipe(prep.sort_dataframe_by_column, config['date_column'])
            .pipe(prep.rename_columns_with_mapping, {
                config['value_column']: config['alias_value_col'],
                config['date_column']: config['alias_date_col']
            })
            .groupby(pd.Grouper(key=config['alias_date_col'], freq=config['groupby_frequency']))
            .sum()
            .reset_index()
            .pipe(prep.apply_rolling_mean, config['alias_value_col'], config['rolling_window_days']))


def apply_application_time_series(df: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Create time series aggregation using application date column.

    Uses shared core time series functionality for DRY compliance.
    Aggregates sales by application signed date with rolling averages.

    Parameters
    ----------
    df : pd.DataFrame
        Clean sales data with application date and premium columns
    config : TimeSeriesConfig
        Configuration with date_column='application_signed_date' and other parameters

    Returns
    -------
    pd.DataFrame
        Time series DataFrame with date and sales columns

    Examples
    --------
    >>> config = {'date_column': 'application_signed_date', 'value_column': 'contract_initial_premium_amount',
    ...           'alias_date_col': 'date', 'alias_value_col': 'sales', 'groupby_frequency': 'd',
    ...           'rolling_window_days': 14}
    >>> ts_df = apply_application_time_series(clean_df, config)
    >>> assert 'date' in ts_df.columns and 'sales' in ts_df.columns
    """
    return _create_time_series_core(df, config)


def apply_contract_time_series(df: pd.DataFrame, config: TimeSeriesConfig) -> pd.DataFrame:
    """Create time series aggregation using contract issue date column.

    Uses shared core time series functionality for DRY compliance.
    Aggregates sales by contract issue date with rolling averages.

    Parameters
    ----------
    df : pd.DataFrame
        Clean sales data with contract date and premium columns
    config : TimeSeriesConfig
        Configuration with date_column='contract_issue_date' and other parameters

    Returns
    -------
    pd.DataFrame
        Time series DataFrame with date and sales columns

    Examples
    --------
    >>> config = {'date_column': 'contract_issue_date', 'value_column': 'contract_initial_premium_amount',
    ...           'alias_date_col': 'date', 'alias_value_col': 'sales_by_contract_date',
    ...           'groupby_frequency': 'd', 'rolling_window_days': 14}
    >>> ts_df = apply_contract_time_series(clean_df, config)
    >>> assert 'sales_by_contract_date' in ts_df.columns
    """
    return _create_time_series_core(df, config)


# =============================================================================
# Pipeline 4: WINK Competitive Rate Processing (NO C_weighted_mean creation)
# =============================================================================

def _apply_wink_preprocessing_chain(df: pd.DataFrame, config: WinkProcessingConfig) -> pd.DataFrame:
    """Apply WINK preprocessing filters chain.

    Applies standardization, rate rounding, and product type filtering.
    """
    return (df
            .pipe(prep.convert_column_to_datetime, "effectiveDate")
            .assign(date=lambda x: x["effectiveDate"])
            .pipe(prep.standardize_wink_company_names)
            .pipe(prep.standardize_wink_product_names)
            .pipe(prep.round_rate_columns, ["bufferRate", "participationRate", "spreadRate", "performanceTriggeredRate"], 3)
            .pipe(prep.filter_by_product_type_name, config['product_type_filter'])
            .pipe(prep.filter_by_participation_rate, config['participation_rate_target'])
            .pipe(prep.filter_by_index_used, config['index_name_filter'])
            .pipe(lambda df: df[df["annualFeeForIndexingMethod"].isna()]))


def _apply_wink_product_filters(df: pd.DataFrame, config: WinkProcessingConfig, all_product_ids: List[Any]) -> pd.DataFrame:
    """Apply WINK product-specific filtering and deduplication."""
    return (df
            .pipe(prep.filter_by_buffer_rates_list, config['buffer_rates_allowed'])
            .pipe(prep.filter_by_buffer_modifier, config['buffer_modifier_filter'])
            .pipe(lambda df: df[df["indexingMethod"] == config['indexing_method_filter']])
            .pipe(lambda df: df[df["indexCreditingFrequency"] == config['crediting_frequency_filter']])
            .pipe(lambda df: df[df["defaultActuarialView"] == config['actuarial_view_filter']])
            .pipe(prep.handle_null_cap_rates, config['default_cap_rate'])
            .pipe(prep.apply_cap_rate_ceiling, config['max_cap_rate'])
            .pipe(lambda df: df[df["productID"].isin(all_product_ids)])
            .pipe(prep.sort_dataframe_by_column, "date")
            .pipe(lambda df: df[df["date"] > config['data_filter_start_date']])
            .pipe(lambda df: df.sort_values(by=["date", "capRate"], ascending=[True, False]))
            .drop_duplicates(subset=["date", "companyName", "mva"])
            .reset_index(drop=True))


def apply_wink_rate_processing(df: pd.DataFrame, config: WinkProcessingConfig) -> pd.DataFrame:
    """
    Apply comprehensive WINK competitive rate processing pipeline.

    Orchestrates atomic functions for competitive rate standardization,
    filtering, validation, and company pivoting. Does NOT create C_weighted_mean.

    Includes production validation checkpoint with fail-fast error handling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw WINK competitive rate data
    config : WinkProcessingConfig
        Configuration for all WINK processing parameters

    Returns
    -------
    pd.DataFrame
        Processed competitive rates with company columns

    Raises
    ------
    ValueError
        If validation fails (data quality issues, schema violations)
    """
    from src.validation.pipeline_validation_helpers import validate_preprocessing_output

    # Extract all product IDs for filtering
    all_product_ids = []
    for company_ids in config['product_ids'].values():
        all_product_ids.extend(company_ids)

    # Step 1: Apply preprocessing chain
    df_preprocessed = _apply_wink_preprocessing_chain(df, config)

    # Step 2: Apply product filters
    df_processed = _apply_wink_product_filters(df_preprocessed, config, all_product_ids)

    # Step 3: Create competitive rate time series
    df_rates, _ = prep.pivot_wink_rates_by_company(
        df_processed, config['product_ids'], config['start_date'],
        datetime.now(), config['rolling_days']
    )

    # Production validation checkpoint
    return validate_preprocessing_output(
        df=df_rates,
        stage_name="wink_rate_processing",
        date_column="date",
        allow_shrinkage=True  # Filtering may legitimately reduce row counts
    )


# =============================================================================
# Pipeline 5: Market Share Weighting (NEW - eliminates duplication)
# =============================================================================

def apply_market_share_weighting(df: pd.DataFrame, market_share_df: pd.DataFrame) -> pd.DataFrame:
    """Apply quarterly market share weighting to competitive rates.

    Creates C_weighted_mean and C_core using actual market share data.
    This is separated from WINK processing to eliminate feature duplication.

    Parameters
    ----------
    df : pd.DataFrame
        Rate matrix with company columns from apply_wink_rate_processing
    market_share_df : pd.DataFrame
        Market share weights by company and quarter

    Returns
    -------
    pd.DataFrame
        Rate matrix with C_weighted_mean and C_core columns added

    Examples
    --------
    >>> rates_weighted = apply_market_share_weighting(rates_matrix, market_share_df)
    >>> assert 'C_weighted_mean' in rates_weighted.columns
    >>> assert 'C_core' in rates_weighted.columns
    """
    return (df
            .pipe(prep.add_quarterly_period_column, "date")
            .pipe(prep.calculate_weighted_competitive_mean, market_share_df))


# =============================================================================
# Pipeline 6: Daily Data Integration (Split from original Pipeline 6)
# =============================================================================

def _merge_all_data_sources(
    config: 'DataIntegrationConfig', data_sources: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Create daily date range and merge all data sources."""
    df_daily_range = eng.create_daily_date_range_dataframe(config['start_date'], config['end_date'])
    return eng.merge_multiple_dataframes_on_date(
        df_daily_range,
        [data_sources['sales'], data_sources['sales_contract'],
         data_sources['dgs5'], data_sources['vixcls'], data_sources['cpi']]
    )


def _apply_cpi_and_smoothing(df: pd.DataFrame, config: 'DataIntegrationConfig') -> pd.DataFrame:
    """Apply CPI adjustment and economic indicator smoothing."""
    return (df
            .pipe(eng.apply_forward_fill_to_columns, [config['cpi_column']] + config['economic_columns'])
            .pipe(eng.apply_cpi_adjustment_to_sales, config['cpi_column'], config['sales_columns'])
            .pipe(eng.apply_rolling_mean_to_columns, config['economic_columns'], config['rolling_window']))


def apply_data_integration(
    rates_df: pd.DataFrame,
    data_sources: Dict[str, pd.DataFrame],
    config: 'DataIntegrationConfig'
) -> pd.DataFrame:
    """Apply daily data integration without weekly aggregation.

    Combines daily date range creation, multi-dataframe merging, CPI adjustment,
    and business counter creation.

    Includes production validation checkpoint with fail-fast error handling.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Competitive rates DataFrame with C_weighted_mean and C_core
    data_sources : Dict[str, pd.DataFrame]
        Dictionary containing 'sales', 'sales_contract', 'dgs5', 'vixcls', 'cpi' DataFrames
    config : DataIntegrationConfig
        Configuration for daily integration parameters

    Returns
    -------
    pd.DataFrame
        Daily integrated DataFrame ready for competitive feature engineering

    Raises
    ------
    ValueError
        If validation fails (data quality issues, missing data)
    """
    from src.validation.pipeline_validation_helpers import validate_preprocessing_output

    # Step 1: Merge all data sources
    df_merged = _merge_all_data_sources(config, data_sources)

    # Step 2: Apply CPI adjustment and economic indicator smoothing
    df_processed = _apply_cpi_and_smoothing(df_merged, config)

    # Step 3: Merge with competitive rates and create business counter
    df_combined = rates_df.merge(df_processed, on="date").drop_duplicates()
    df_with_counter = eng.create_business_day_counter(
        df_combined, config['sales_reference_column'], config['business_counter_column']
    )

    # Production validation checkpoint
    return validate_preprocessing_output(
        df=df_with_counter,
        stage_name="data_integration",
        date_column="date",
        allow_shrinkage=False  # Integration should preserve/increase data
    )


# =============================================================================
# Pipeline 8: Weekly Aggregation (Split from original Pipeline 6)
# =============================================================================

def apply_weekly_aggregation(
    df: pd.DataFrame,
    config: 'WeeklyAggregationConfig'
) -> pd.DataFrame:
    """Apply weekly aggregation with competitive features included.

    Performs weekly frequency aggregation and adds temporal indicators.
    This function is called AFTER competitive features are created,
    so the aggregation dictionary can include all C_* columns.

    Parameters
    ----------
    df : pd.DataFrame
        Daily DataFrame with competitive features (C_median, C_first, etc.)
    config : WeeklyAggregationConfig
        Configuration for weekly aggregation with complete feature dictionary

    Returns
    -------
    pd.DataFrame
        Weekly aggregated DataFrame with temporal indicators

    Examples
    --------
    >>> weekly_df = apply_weekly_aggregation(competitive_df, config)
    >>> assert weekly_df.index.freq is None  # reset_index applied
    """
    # Step 1: Weekly aggregation with comprehensive aggregation dictionary
    df_weekly = eng.aggregate_dataframe_by_frequency(
        df, 'date', config['weekly_aggregation_freq'], config['weekly_agg_dict']
    )

    # Step 2: Add temporal indicators
    df_ts_weekly = eng.create_temporal_indicator_columns(df_weekly, 'date')

    return df_ts_weekly


# =============================================================================
# Pipeline 7: Competitive Feature Engineering (Fixed - no duplication)
# =============================================================================

def apply_competitive_features(df: pd.DataFrame, config: CompetitiveConfig) -> pd.DataFrame:
    """Apply 4 atomic competitive ranking and semantic mapping functions.

    Executes competitive feature engineering using clean_v0 atomic functions:
    median rankings, top rankings, position rankings, and semantic mappings.
    No backward compatibility shortcuts - uses clean semantic naming only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with company rate columns and existing C_weighted_mean
    config : CompetitiveConfig
        Configuration for company columns and minimum companies required

    Returns
    -------
    pd.DataFrame
        DataFrame with semantic competitive features (C_median, C_top_3, C_first, etc.)

    Raises
    ------
    ValueError
        If insufficient company data for competitive analysis

    Examples
    --------
    >>> config = {'company_columns': ['Allianz', 'Athene', 'Brighthouse', ...],
    ...           'min_companies_required': 3}
    >>> competitive_df = apply_competitive_features(weekly_df, config)
    >>> assert 'C_median' in competitive_df.columns
    >>> assert 'C_top_3' in competitive_df.columns
    """
    return (df
            .pipe(eng.calculate_median_competitor_rankings,
                  config['company_columns'], config['min_companies_required'])
            .pipe(eng.calculate_top_competitor_rankings,
                  config['company_columns'], config['min_companies_required'])
            .pipe(eng.calculate_position_competitor_rankings,
                  config['company_columns'], config['min_companies_required'])
            .pipe(eng.apply_competitive_semantic_mappings,
                  ['raw_median', 'top_3', 'top_5', 'first_highest_benefit',
                   'second_highest_benefit', 'third_highest_benefit']))


# =============================================================================
# Pipeline 9: Lag Feature Engineering (MAJOR missing pipeline)
# =============================================================================

def apply_lag_and_polynomial_features(df: pd.DataFrame, config: LagFeatureConfig) -> pd.DataFrame:
    """Create comprehensive lag features and polynomial interactions.

    This was the most critical missing pipeline! Creates 13 lag column
    configurations with forward/backward lags, then polynomial interactions
    for enhanced competitive features.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly DataFrame with competitive features ready for lag creation
    config : LagFeatureConfig
        Configuration with lag_column_configs, polynomial_base_columns, max_lag_periods

    Returns
    -------
    pd.DataFrame
        DataFrame with comprehensive lag features and polynomial interactions

    Examples
    --------
    >>> config = {'lag_column_configs': [{'source_col': 'sales', 'prefix': 'sales_target', 'lag_direction': 'both'}, ...],
    ...           'polynomial_base_columns': ['prudential_rate', 'competitor_mid', ...],
    ...           'max_lag_periods': 18, 'allow_inplace_operations': True}
    >>> lagged_df = apply_lag_and_polynomial_features(competitive_df, config)
    >>> assert 'sales_target_current' in lagged_df.columns
    >>> assert 'sales_target_t1' in lagged_df.columns
    """
    # Create comprehensive lag features using 13 lag configurations
    df_with_lags = eng.create_lag_features_for_columns(
        df,
        config['lag_column_configs'],
        config['max_lag_periods'],
        config['allow_inplace_operations']
    )

    # Create polynomial interaction features for competitive analysis
    df_with_polynomials = eng.create_polynomial_interaction_features(
        df_with_lags,
        config['polynomial_base_columns'],
        config['max_lag_periods'],
        config['allow_inplace_operations']
    )

    return df_with_polynomials


# =============================================================================
# Pipeline 10: Final Feature Preparation (Fixed - add feature analysis filtering)
# =============================================================================

def apply_final_feature_preparation(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Apply feature analysis filtering, temporal features, spreads, and cleanup.

    Combines feature analysis date filtering plus 5 atomic functions for final
    dataset preparation: temporal indicators, holiday flags, competitive spreads,
    log transformations, and row cleanup.

    Parameters
    ----------
    df : pd.DataFrame
        Lag-engineered DataFrame ready for final preparation
    config : FeatureConfig
        Configuration for all final feature preparation parameters

    Returns
    -------
    pd.DataFrame
        Final dataset ready for model training with all features prepared

    Examples
    --------
    >>> config = {'feature_analysis_start_date': '2022-09-01', 'date_column': 'date',
    ...           'day_of_year_column': 'day_of_year', 'holiday_column': 'holiday',
    ...           'holiday_start_day': 13, 'holiday_end_day': 359,
    ...           'prudential_rate_column': 'prudential_rate_current',
    ...           'competitor_rate_column': 'competitor_mid_t1', 'spread_column': 'Spread',
    ...           'log_transform_source_column': 'sales_target_contract_current',
    ...           'log_transform_target_column': 'sales_log',
    ...           'mature_data_cutoff_date': '2025-08-17'}
    >>> final_df = apply_final_feature_preparation(lagged_df, config)
    >>> assert config['spread_column'] in final_df.columns
    """
    # Missing step from original: Filter by feature analysis start date
    df_filtered = df[df[config['date_column']] > config['feature_analysis_start_date']]

    return (df_filtered
            .pipe(eng.filter_dataframe_by_mature_date, config['date_column'],
                  config['mature_data_cutoff_date'])
            .pipe(eng.extract_day_of_year_column, config['date_column'],
                  config['day_of_year_column'])
            .pipe(eng.create_holiday_indicator_by_day_range, config['day_of_year_column'],
                  config['holiday_start_day'], config['holiday_end_day'], config['holiday_column'])
            .pipe(eng.calculate_competitive_spread, config['prudential_rate_column'],
                  config['competitor_rate_column'], config['spread_column'])
            .pipe(eng.apply_log_plus_one_transformation, config['log_transform_source_column'],
                  config['log_transform_target_column'])
            .pipe(eng.remove_final_row_from_dataframe))


# Note: apply_feature_selection removed - not used by refactored notebooks
# Feature selection is now handled by features.selection.notebook_interface module


# =============================================================================
# Module Access to Atomic Functions (for flexibility)
# =============================================================================

# Expose 10 pipeline functions and config builder (apply_feature_selection removed)
__all__ = [
    # Core 10 pipeline functions used by refactored notebooks
    'apply_product_filters',                    # Pipeline 1: Product filtering
    'apply_sales_data_cleanup',                 # Pipeline 2: Sales cleanup
    'apply_application_time_series',            # Pipeline 3a: App time series
    'apply_contract_time_series',               # Pipeline 3b: Contract time series
    'apply_wink_rate_processing',               # Pipeline 4: WINK processing
    'apply_market_share_weighting',             # Pipeline 5: Market share weighting
    'apply_data_integration',                   # Pipeline 6: Daily data integration
    'apply_competitive_features',               # Pipeline 7: Competitive features
    'apply_weekly_aggregation',                 # Pipeline 8: Weekly aggregation
    'apply_lag_and_polynomial_features',        # Pipeline 9: Lag features
    'apply_final_feature_preparation',          # Pipeline 10: Final prep
    # Configuration builder for easy setup
    'config_builder',
    # Module access for atomic functions (proc removed - not used)
    'ext', 'prep', 'eng', 'viz'
]