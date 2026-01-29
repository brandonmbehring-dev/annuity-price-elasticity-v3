"""
Pipeline Stage Configuration Builders

Extracted from config_builder.py for single-responsibility compliance.
Handles all pipeline stage configurations: product filtering, sales cleanup,
time series, WINK processing, data integration, and feature engineering.

Usage:
    from src.config.pipeline_builders import build_pipeline_configs
    configs = build_pipeline_configs(version=6)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.config.types.product_config import (
    ProductConfig, get_product_config, get_pipeline_product_ids_as_lists
)
from src.config.types.pipeline_config import (
    ProductFilterConfig, SalesCleanupConfig, TimeSeriesConfig,
    WinkProcessingConfig, DataIntegrationConfig, WeeklyAggregationConfig,
    CompetitiveConfig, LagFeatureConfig, FeatureConfig
)


# Module exports (including private functions for testing)
__all__ = [
    # Public API
    'get_lag_column_configs',
    'get_weekly_aggregation_dict',
    'build_pipeline_configs',
    'build_pipeline_configs_for_product',
    # Private builders (for unit testing)
    '_build_product_filter_config',
    '_build_sales_cleanup_config',
    '_build_time_series_config',
    '_build_wink_processing_config',
    '_build_data_integration_config',
    '_build_lag_features_config',
    '_build_final_features_config',
    '_get_default_flexguard_product_ids',
    '_build_extraction_configs',
    '_build_processing_configs',
    '_build_integration_configs',
    '_convert_product_to_filters',
    '_merge_product_kwargs',
]


# =============================================================================
# LAG COLUMN AND AGGREGATION CONFIGURATIONS
# =============================================================================


def get_lag_column_configs() -> List[Dict[str, Any]]:
    """Return 13 lag column configurations with clean semantic naming.

    Creates comprehensive lag feature configurations for sales, competitive,
    and economic indicators using descriptive prefixes for clear feature naming.

    Feature Naming Unification (2026-01-26):
    - competitor_mid renamed to competitor_weighted for semantic clarity

    Returns
    -------
    List[Dict[str, Any]]
        List of 13 lag column configuration dictionaries

    Examples
    --------
    >>> configs = get_lag_column_configs()
    >>> assert len(configs) == 13
    >>> assert configs[0]['prefix'] == 'sales_target'
    >>> assert 'competitor_weighted' in [c['prefix'] for c in configs]
    """
    return [
        # Sales targets (backward only for causal identification)
        # Forward-looking features violate causal identification - see Audit Issue #2
        {'source_col': 'sales', 'prefix': 'sales_target', 'lag_direction': 'backward'},
        {'source_col': 'sales_by_contract_date', 'prefix': 'sales_target_contract', 'lag_direction': 'backward'},

        # Core competitive features (backward looking for prediction)
        {'source_col': 'Prudential', 'prefix': 'prudential_rate', 'lag_direction': 'backward'},
        {'source_col': 'C_weighted_mean', 'prefix': 'competitor_weighted', 'lag_direction': 'backward'},
        {'source_col': 'C_core', 'prefix': 'competitor_core', 'lag_direction': 'backward'},

        # Enhanced competitive features from atomic functions (clean semantic names)
        {'source_col': 'C_median', 'prefix': 'competitor_median', 'lag_direction': 'backward'},
        {'source_col': 'C_first', 'prefix': 'competitor_1st', 'lag_direction': 'backward'},
        {'source_col': 'C_second', 'prefix': 'competitor_2nd', 'lag_direction': 'backward'},
        {'source_col': 'C_third', 'prefix': 'competitor_3rd', 'lag_direction': 'backward'},
        {'source_col': 'C_top_3', 'prefix': 'competitor_top3', 'lag_direction': 'backward'},
        {'source_col': 'C_top_5', 'prefix': 'competitor_top5', 'lag_direction': 'backward'},

        # Economic indicators
        {'source_col': 'DGS5', 'prefix': 'econ_treasury_5y', 'lag_direction': 'backward'},
        {'source_col': 'VIXCLS', 'prefix': 'market_volatility', 'lag_direction': 'backward'}
    ]


def get_weekly_aggregation_dict() -> Dict[str, str]:
    """Return weekly aggregation dictionary with clean semantic naming.

    Creates aggregation mapping for weekly frequency conversion.
    No backward compatibility shortcuts - semantic names only.

    Returns
    -------
    Dict[str, str]
        Aggregation method mapping for each feature

    Examples
    --------
    >>> agg_dict = get_weekly_aggregation_dict()
    >>> assert agg_dict['C_weighted_mean'] == 'mean'
    >>> assert agg_dict['sales'] == 'sum'
    >>> assert 'C' not in agg_dict  # No backward compatibility shortcuts
    """
    return {
        # Competitive features (semantic names only)
        'C_weighted_mean': 'mean',
        'C_core': 'mean',
        'C_median': 'mean',
        'C_first': 'mean',
        'C_second': 'mean',
        'C_third': 'mean',
        'C_top_3': 'mean',
        'C_top_5': 'mean',
        'Prudential': 'mean',  # Direct reference, no shortcut

        # Individual company rates
        'Allianz': 'mean',
        'Athene': 'mean',
        'Brighthouse': 'mean',
        'Equitable': 'mean',
        'Jackson': 'mean',
        'Lincoln': 'mean',
        'Symetra': 'mean',
        'Trans': 'mean',

        # Sales data
        'sales': 'sum',
        'sales_by_contract_date': 'sum',
        'counter': 'sum',

        # Economic indicators
        'DGS5': 'last',
        'VIXCLS': 'last'
    }


# =============================================================================
# PIPELINE STAGE BUILDER HELPERS
# =============================================================================


def _build_product_filter_config(
    product_name: str,
    buffer_rate_filter: str,
    term_filter: str
) -> ProductFilterConfig:
    """Build product filter configuration.

    Parameters
    ----------
    product_name : str
        Product name for filtering
    buffer_rate_filter : str
        Buffer rate filter string (e.g., "20%")
    term_filter : str
        Term filter string (e.g., "6Y")

    Returns
    -------
    ProductFilterConfig
        Product filter configuration
    """
    return ProductFilterConfig({
        'product_name': product_name,
        'buffer_rate': buffer_rate_filter,
        'term': term_filter
    })


def _build_sales_cleanup_config(
    min_premium: float,
    max_premium: float,
    quantile_threshold: float
) -> SalesCleanupConfig:
    """Build sales cleanup configuration with premium filters.

    Parameters
    ----------
    min_premium : float
        Minimum premium threshold
    max_premium : float
        Maximum premium threshold
    quantile_threshold : float
        Quantile threshold for outlier removal

    Returns
    -------
    SalesCleanupConfig
        Sales cleanup configuration
    """
    return SalesCleanupConfig({
        'min_premium': min_premium,
        'max_premium': max_premium,
        'quantile_threshold': quantile_threshold,
        'start_date_col': 'application_signed_date',
        'end_date_col': 'contract_issue_date',
        'processing_days_col': 'processing_days',
        'premium_column': 'contract_initial_premium_amount',
        'sales_alias_col': 'sales_amount'
    })


def _build_time_series_config(
    groupby_frequency: str,
    rolling_window_days: int
) -> TimeSeriesConfig:
    """Build time series configuration.

    Parameters
    ----------
    groupby_frequency : str
        Frequency for grouping (e.g., "d" for daily)
    rolling_window_days : int
        Rolling window size in days

    Returns
    -------
    TimeSeriesConfig
        Time series configuration
    """
    return TimeSeriesConfig({
        'date_column': 'date',
        'value_column': 'contract_initial_premium_amount',
        'alias_date_col': 'date',
        'alias_value_col': 'sales',
        'groupby_frequency': groupby_frequency,
        'rolling_window_days': rolling_window_days
    })


def _build_wink_processing_config(
    flexguard_product_ids: Dict[str, List[int]],
    rate_analysis_start_date: str,
    rolling_window_days: int,
    data_filter_start_date: str
) -> WinkProcessingConfig:
    """Build WINK competitive rate processing configuration.

    Parameters
    ----------
    flexguard_product_ids : Dict[str, List[int]]
        Product ID mapping by company
    rate_analysis_start_date : str
        Start date for rate analysis
    rolling_window_days : int
        Rolling window size in days
    data_filter_start_date : str
        Start date for data filtering

    Returns
    -------
    WinkProcessingConfig
        WINK processing configuration
    """
    return WinkProcessingConfig({
        'product_ids': flexguard_product_ids,
        'start_date': rate_analysis_start_date,
        'rolling_days': rolling_window_days,
        'buffer_modifier_filter': 'Losses Covered Up To',
        'indexing_method_filter': 'Term End Point',
        'crediting_frequency_filter': '6 years',
        'actuarial_view_filter': 'Term End Point: Term End Point, No Premium Bonus',
        'product_type_filter': 'Structured',
        'participation_rate_target': 1.0,
        'index_name_filter': 'S&P 500',
        'buffer_rates_allowed': [0.20, 0.25],
        'default_cap_rate': 9.99,
        'max_cap_rate': 4.5,
        'data_filter_start_date': data_filter_start_date
    })


def _build_data_integration_config(
    analysis_start_date: str,
    current_date: str,
    economic_indicator_rolling_window: int
) -> DataIntegrationConfig:
    """Build data integration configuration.

    Parameters
    ----------
    analysis_start_date : str
        Start date for analysis
    current_date : str
        Current date string
    economic_indicator_rolling_window : int
        Rolling window for economic indicators

    Returns
    -------
    DataIntegrationConfig
        Data integration configuration
    """
    return DataIntegrationConfig({
        'start_date': analysis_start_date,
        'end_date': current_date,
        'cpi_column': 'CPILFESL_inv',
        'sales_columns': ['sales', 'sales_by_contract_date'],
        'economic_columns': ['DGS5', 'VIXCLS'],
        'rolling_window': economic_indicator_rolling_window,
        'sales_reference_column': 'sales',
        'business_counter_column': 'counter'
    })


def _build_lag_features_config(
    max_lag_periods: int,
    allow_inplace_operations: bool
) -> LagFeatureConfig:
    """Build lag feature engineering configuration.

    Parameters
    ----------
    max_lag_periods : int
        Maximum lag periods for feature creation
    allow_inplace_operations : bool
        Whether to allow in-place DataFrame operations

    Returns
    -------
    LagFeatureConfig
        Lag feature configuration
    """
    return LagFeatureConfig({
        'lag_column_configs': get_lag_column_configs(),
        'polynomial_base_columns': [
            'prudential_rate', 'competitor_weighted', 'competitor_median',
            'competitor_top3', 'competitor_top5'
        ],
        'max_lag_periods': max_lag_periods,
        'allow_inplace_operations': allow_inplace_operations
    })


def _build_final_features_config(
    feature_analysis_start_date: str,
    holiday_start_day: int,
    holiday_end_day: int,
    current_date_of_mature_data: str
) -> FeatureConfig:
    """Build final feature preparation configuration.

    Feature Naming Unification (2026-01-26):
    - _current → _t0 (prudential_rate_t0, sales_target_contract_t0)
    - competitor_mid → competitor_weighted

    Parameters
    ----------
    feature_analysis_start_date : str
        Start date for feature analysis
    holiday_start_day : int
        Day of year holiday period starts
    holiday_end_day : int
        Day of year holiday period ends
    current_date_of_mature_data : str
        Cutoff date for mature data

    Returns
    -------
    FeatureConfig
        Final feature configuration
    """
    return FeatureConfig({
        'feature_analysis_start_date': feature_analysis_start_date,
        'date_column': 'date',
        'day_of_year_column': 'day_of_year',
        'holiday_column': 'holiday',
        'holiday_start_day': holiday_start_day,
        'holiday_end_day': holiday_end_day,
        'prudential_rate_column': 'prudential_rate_t0',
        'competitor_rate_column': 'competitor_weighted_t1',
        'spread_column': 'Spread',
        'log_transform_source_column': 'sales_target_contract_t0',
        'log_transform_target_column': 'sales_log',
        'mature_data_cutoff_date': current_date_of_mature_data
    })


def _get_default_flexguard_product_ids() -> Dict[str, List[int]]:
    """Return default FlexGuard product IDs mapping.

    DEPRECATED: Use get_pipeline_product_ids_as_lists() from product_config
    for the canonical source. This function is maintained for backward
    compatibility and delegates to the canonical source.

    Returns
    -------
    Dict[str, List[int]]
        Pipeline product ID mapping (current WINK data)
    """
    return get_pipeline_product_ids_as_lists()


# =============================================================================
# STAGE-BASED CONFIG BUILDERS
# =============================================================================


def _build_extraction_configs(
    product_name: str,
    buffer_rate_filter: str,
    term_filter: str,
    min_premium: float,
    max_premium: float,
    quantile_threshold: float
) -> Dict[str, Any]:
    """Build extraction stage configs: product_filter, sales_cleanup.

    Parameters
    ----------
    product_name : str
        Product name for filtering
    buffer_rate_filter : str
        Buffer rate filter (e.g., "20%")
    term_filter : str
        Term filter (e.g., "6Y")
    min_premium : float
        Minimum premium threshold
    max_premium : float
        Maximum premium threshold
    quantile_threshold : float
        Quantile threshold for outlier removal

    Returns
    -------
    Dict[str, Any]
        Extraction stage configuration dict
    """
    return {
        'product_filter': _build_product_filter_config(product_name, buffer_rate_filter, term_filter),
        'sales_cleanup': _build_sales_cleanup_config(min_premium, max_premium, quantile_threshold)
    }


def _build_processing_configs(
    groupby_frequency: str,
    rolling_window_days: int,
    flexguard_product_ids: Dict[str, List[int]],
    rate_analysis_start_date: str,
    data_filter_start_date: str
) -> Dict[str, Any]:
    """Build processing stage configs: time_series, wink, weekly_agg, competitive.

    Parameters
    ----------
    groupby_frequency : str
        Frequency for grouping (e.g., "d" for daily)
    rolling_window_days : int
        Rolling window size in days
    flexguard_product_ids : Dict[str, List[int]]
        Product ID mapping by company
    rate_analysis_start_date : str
        Start date for rate analysis
    data_filter_start_date : str
        Start date for data filtering

    Returns
    -------
    Dict[str, Any]
        Processing stage configuration dict
    """
    return {
        'time_series': _build_time_series_config(groupby_frequency, rolling_window_days),
        'wink_processing': _build_wink_processing_config(
            flexguard_product_ids, rate_analysis_start_date,
            rolling_window_days, data_filter_start_date
        ),
        'weekly_aggregation': WeeklyAggregationConfig({
            'weekly_aggregation_freq': 'W',
            'weekly_agg_dict': get_weekly_aggregation_dict()
        }),
        'competitive': CompetitiveConfig({
            'company_columns': ['Allianz', 'Athene', 'Brighthouse', 'Equitable',
                              'Jackson', 'Lincoln', 'Symetra', 'Trans'],
            'min_companies_required': 3
        })
    }


def _build_integration_configs(
    analysis_start_date: str,
    current_date: str,
    economic_indicator_rolling_window: int,
    max_lag_periods: int,
    allow_inplace_operations: bool,
    feature_analysis_start_date: str,
    holiday_start_day: int,
    holiday_end_day: int,
    current_date_of_mature_data: str
) -> Dict[str, Any]:
    """Build integration stage configs: data_integration, lag_features, final_features.

    Parameters
    ----------
    analysis_start_date : str
        Start date for analysis
    current_date : str
        Current date string
    economic_indicator_rolling_window : int
        Rolling window for economic indicators
    max_lag_periods : int
        Maximum lag periods for feature creation
    allow_inplace_operations : bool
        Whether to allow in-place DataFrame operations
    feature_analysis_start_date : str
        Start date for feature analysis
    holiday_start_day : int
        Day of year holiday period starts
    holiday_end_day : int
        Day of year holiday period ends
    current_date_of_mature_data : str
        Cutoff date for mature data

    Returns
    -------
    Dict[str, Any]
        Integration stage configuration dict
    """
    return {
        'data_integration': _build_data_integration_config(
            analysis_start_date, current_date, economic_indicator_rolling_window
        ),
        'lag_features': _build_lag_features_config(max_lag_periods, allow_inplace_operations),
        'final_features': _build_final_features_config(
            feature_analysis_start_date, holiday_start_day,
            holiday_end_day, current_date_of_mature_data
        )
    }


# =============================================================================
# MAIN PIPELINE CONFIG BUILDER
# =============================================================================


def build_pipeline_configs(
    version: int = 6,
    product_name: str = "FlexGuard indexed variable annuity",
    term_filter: str = "6Y",
    buffer_rate_filter: str = "20%",
    min_premium: float = 1000.0,
    max_premium: float = 1000000.0,
    quantile_threshold: float = 0.99,
    groupby_frequency: str = "d",
    rolling_window_days: int = 14,
    flexguard_product_ids: Dict[str, List[int]] = None,
    rate_analysis_start_date: str = "2018-06-21",
    data_filter_start_date: str = "2018-01-01",
    analysis_start_date: str = "2021-01-01",
    feature_analysis_start_date: str = "2022-01-01",
    mature_data_offset_days: int = 50,
    max_lag_periods: int = 18,
    allow_inplace_operations: bool = True,
    economic_indicator_rolling_window: int = 7,
    holiday_start_day: int = 13,
    holiday_end_day: int = 359
) -> Dict[str, Any]:
    """Build all 9 pipeline configurations from business parameters.

    Orchestrates extraction, processing, and integration stage builders
    to create comprehensive configuration for the data pipeline.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: product_filter, sales_cleanup, time_series,
        wink_processing, weekly_aggregation, competitive, data_integration,
        lag_features, final_features
    """
    # Calculate derived parameters
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    mature_date = (current_time - timedelta(days=mature_data_offset_days)).strftime("%Y-%m-%d")

    if flexguard_product_ids is None:
        flexguard_product_ids = _get_default_flexguard_product_ids()

    # Assemble configs from stage builders
    configs: Dict[str, Any] = {}
    configs.update(_build_extraction_configs(
        product_name, buffer_rate_filter, term_filter,
        min_premium, max_premium, quantile_threshold
    ))
    configs.update(_build_processing_configs(
        groupby_frequency, rolling_window_days, flexguard_product_ids,
        rate_analysis_start_date, data_filter_start_date
    ))
    configs.update(_build_integration_configs(
        analysis_start_date, current_date, economic_indicator_rolling_window,
        max_lag_periods, allow_inplace_operations, feature_analysis_start_date,
        holiday_start_day, holiday_end_day, mature_date
    ))
    return configs


# =============================================================================
# PRODUCT-AWARE PIPELINE CONFIG BUILDER
# =============================================================================


def _convert_product_to_filters(product: ProductConfig) -> Dict[str, str]:
    """Convert ProductConfig attributes to pipeline filter strings.

    Parameters
    ----------
    product : ProductConfig
        Product configuration instance

    Returns
    -------
    Dict[str, str]
        Dictionary with 'buffer_rate_filter' and 'term_filter' keys
    """
    return {
        'buffer_rate_filter': f"{int(product.buffer_level * 100)}%",
        'term_filter': f"{product.term_years}Y"
    }


def _merge_product_kwargs(
    product: ProductConfig,
    filters: Dict[str, str],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge product-derived parameters with user-provided kwargs.

    Returns merged kwargs with product defaults applied where not overridden.

    Parameters
    ----------
    product : ProductConfig
        Product configuration instance
    filters : Dict[str, str]
        Filter strings derived from product
    kwargs : Dict[str, Any]
        User-provided keyword arguments

    Returns
    -------
    Dict[str, Any]
        Merged keyword arguments
    """
    return {
        'buffer_rate_filter': kwargs.pop('buffer_rate_filter', filters['buffer_rate_filter']),
        'term_filter': kwargs.pop('term_filter', filters['term_filter']),
        'max_lag_periods': kwargs.pop('max_lag_periods', product.max_lag),
        **kwargs
    }


def build_pipeline_configs_for_product(
    product_code: str = "6Y20B",
    **kwargs
) -> Dict[str, Any]:
    """Build pipeline configurations for a specific product.

    This is the RECOMMENDED entry point for product-aware configuration.
    It combines ProductConfig with existing pipeline configs using composition.

    Parameters
    ----------
    product_code : str, default="6Y20B"
        Product identifier (e.g., "6Y20B", "6Y10B", "10Y20B")
    **kwargs : Any
        Additional parameters passed to build_pipeline_configs()

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary including 'product' (ProductConfig instance)
        and all standard pipeline configs from build_pipeline_configs()

    Examples
    --------
    >>> configs = build_pipeline_configs_for_product("6Y20B")
    >>> assert 'product' in configs
    >>> assert configs['product'].buffer_level == 0.20
    """
    product = get_product_config(product_code)
    filters = _convert_product_to_filters(product)
    merged_kwargs = _merge_product_kwargs(product, filters, kwargs)

    pipeline_configs = build_pipeline_configs(**merged_kwargs)
    pipeline_configs['product'] = product

    return pipeline_configs
