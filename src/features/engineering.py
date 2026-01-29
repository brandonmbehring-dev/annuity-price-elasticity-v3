"""
Feature engineering and ML preparation for clean_v0 pipeline.

This module handles feature engineering operations used in 00_clean_v0.ipynb:
- Data integration and merging
- Time series feature creation
- Lag features and polynomial interactions
- Business transformations and final dataset preparation

Module Architecture (Phase 6.2 Split):
- competitive_features.py: Competitive rankings + WINK weighted mean
- engineering_integration.py: Data integration and merging
- engineering_temporal.py: Time-based feature engineering
- engineering_timeseries.py: Time series aggregation and creation
- engineering.py: Public API orchestrator (this file)

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

# Import config types - Fail fast with clear error if imports fail
from ..config.pipeline_config import EngineeringMathematicalConfig, EngineeringTimeSeriesConfig


# =============================================================================
# RE-EXPORTS FROM COMPETITIVE FEATURES MODULE (public API only)
# =============================================================================

from src.features.competitive_features import (
    # Ranking functions (public)
    calculate_median_competitor_rankings,
    calculate_top_competitor_rankings,
    calculate_position_competitor_rankings,
    # Semantic mappings (public)
    apply_competitive_semantic_mappings,
    # Compatibility (public)
    create_competitive_compatibility_shortcuts,
    # WINK weighted mean (public)
    wink_weighted_mean,
    WINK_weighted_mean,  # Deprecated alias
    # Spread calculation (public)
    calculate_competitive_spread,
)


# =============================================================================
# RE-EXPORTS FROM DATA INTEGRATION MODULE
# =============================================================================

from src.features.engineering_integration import (
    # Date range creation
    create_daily_date_range_dataframe,
    # DataFrame merging
    merge_multiple_dataframes_on_date,
    # Forward fill
    apply_forward_fill_to_columns,
    # CPI adjustments
    apply_cpi_adjustment_to_sales,
    # Rolling mean
    apply_rolling_mean_to_columns,
    # Business day counting
    create_business_day_counter,
    # Frequency aggregation
    aggregate_dataframe_by_frequency,
    # Data filtering
    filter_dataframe_by_mature_date,
    remove_final_row_from_dataframe,
)


# =============================================================================
# RE-EXPORTS FROM TEMPORAL FEATURES MODULE
# =============================================================================

from src.features.engineering_temporal import (
    # Temporal indicators
    create_temporal_indicator_columns,
    # Day of year
    extract_day_of_year_column,
    # Holiday indicators
    create_holiday_indicator_by_day_range,
    # Lag features
    create_lag_features_for_columns,
    # Polynomial interactions
    create_polynomial_interaction_features,
    # Log transformation
    apply_log_plus_one_transformation,
)


# =============================================================================
# RE-EXPORTS FROM TIME SERIES MODULE (public API only)
# =============================================================================

from src.features.engineering_timeseries import (
    # CPI adjustment (public)
    cpi_adjustment,
    # Weekly aggregation (public)
    time_series_week_agg_smoothed,
    # Lag features (public)
    create_lag_features,
)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Public API from competitive_features.py
    'calculate_median_competitor_rankings',
    'calculate_top_competitor_rankings',
    'calculate_position_competitor_rankings',
    'apply_competitive_semantic_mappings',
    'create_competitive_compatibility_shortcuts',
    'wink_weighted_mean',
    'WINK_weighted_mean',  # Deprecated alias for backward compatibility
    'calculate_competitive_spread',

    # Public API from engineering_integration.py
    'create_daily_date_range_dataframe',
    'merge_multiple_dataframes_on_date',
    'apply_forward_fill_to_columns',
    'apply_cpi_adjustment_to_sales',
    'apply_rolling_mean_to_columns',
    'create_business_day_counter',
    'aggregate_dataframe_by_frequency',
    'filter_dataframe_by_mature_date',
    'remove_final_row_from_dataframe',

    # Public API from engineering_temporal.py
    'create_temporal_indicator_columns',
    'extract_day_of_year_column',
    'create_holiday_indicator_by_day_range',
    'create_lag_features_for_columns',
    'create_polynomial_interaction_features',
    'apply_log_plus_one_transformation',

    # Public API from engineering_timeseries.py
    'cpi_adjustment',
    'time_series_week_agg_smoothed',
    'create_lag_features',
]
