"""
Time Series Aggregation Module for Feature Engineering.

This module handles time series aggregation and creation operations
from helpers/feature_engineering_functions_RILA.py:
- CPI adjustment pipeline
- Weekly aggregation with smoothing
- Semantic feature creation
- Lag features for time series

Part of Phase 6.2 module split.

Module Architecture:
- competitive_features.py: Competitive rankings + WINK weighted mean
- engineering_integration.py: Data integration and merging
- engineering_temporal.py: Time-based feature engineering
- engineering_timeseries.py: Time series aggregation and creation (this file)
- engineering.py: Public API orchestrator

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

from typing import Any, Dict

import pandas as pd


# =============================================================================
# CPI ADJUSTMENT HELPERS
# =============================================================================


def _merge_economic_data(
    df_ts: pd.DataFrame,
    data: pd.DataFrame,
    df_contract: pd.DataFrame,
    df_DGS5: pd.DataFrame,
    df_VIXCLS: pd.DataFrame,
    cpi_data: pd.DataFrame
) -> pd.DataFrame:
    """Merge all economic data sources into time series DataFrame.

    Parameters
    ----------
    df_ts : pd.DataFrame
        Base time series DataFrame with date column
    data : pd.DataFrame
        Sales data
    df_contract : pd.DataFrame
        Contract date sales data
    df_DGS5 : pd.DataFrame
        5-Year Treasury Rate data
    df_VIXCLS : pd.DataFrame
        VIX volatility index data
    cpi_data : pd.DataFrame
        CPI data

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all economic indicators
    """
    return df_ts.merge(
        data, on='date', how='left').merge(
        df_contract, on='date', how='left').merge(
        df_DGS5, on='date', how='left').merge(
        df_VIXCLS, on='date', how='left').merge(
        cpi_data, on='date', how='left').drop_duplicates()


def _apply_cpi_transformations(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Apply forward fill and CPI transformations to economic data.

    Parameters
    ----------
    df_ts : pd.DataFrame
        DataFrame with economic indicator columns

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with CPI-adjusted sales and smoothed indicators
    """
    df_ts['CPILFESL_inv'] = df_ts.CPILFESL_inv.fillna(method='ffill')
    df_ts['DGS5'] = df_ts.DGS5.fillna(method='ffill')
    df_ts['VIXCLS'] = df_ts.VIXCLS.fillna(method='ffill')

    df_ts['sales'] = df_ts['CPILFESL_inv'] * df_ts['sales']
    df_ts['sales_by_contract_date'] = df_ts['CPILFESL_inv'] * df_ts['sales_by_contract_date']
    df_ts['DGS5'] = df_ts['DGS5'].rolling(7).mean()
    df_ts['VIXCLS'] = df_ts['VIXCLS'].rolling(7).mean()

    return df_ts


def cpi_adjustment(
    data: pd.DataFrame,
    cpi_data: pd.DataFrame,
    df_DGS5: pd.DataFrame,
    df_VIXCLS: pd.DataFrame,
    df_contract: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Apply CPI adjustment to sales data and merge economic indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Base sales data with date column
    cpi_data : pd.DataFrame
        CPI data with CPILFESL_inv column
    df_DGS5 : pd.DataFrame
        5-Year Treasury Rate data
    df_VIXCLS : pd.DataFrame
        VIX volatility index data
    df_contract : pd.DataFrame
        Contract date sales data
    start_date : str
        Start date for time series (YYYY-MM-DD format)
    end_date : str
        End date for time series (YYYY-MM-DD format)

    Returns
    -------
    pd.DataFrame
        CPI-adjusted data with economic indicators
    """
    data['date'] = pd.to_datetime(data['date'])
    time_index = pd.date_range(start=start_date, end=end_date, freq='d')
    df_ts = pd.DataFrame(time_index, columns=['date'])

    df_ts = _merge_economic_data(df_ts, data, df_contract, df_DGS5, df_VIXCLS, cpi_data)
    df_ts = _apply_cpi_transformations(df_ts)

    return df_ts[['date', 'sales', 'DGS5', 'VIXCLS', 'sales_by_contract_date']]


# =============================================================================
# WEEKLY AGGREGATION HELPERS
# =============================================================================


def _create_aggregation_dict(how: str = 'last') -> Dict[str, str]:
    """Create aggregation dictionary for weekly groupby operation."""
    from src.config.product_config import get_competitor_config

    competitor_config = get_competitor_config()
    own_company = competitor_config.own_company

    comp_list = ['raw_mean', 'raw_median', 'first_highest_benefit',
                 'second_highest_benefit', 'third_highest_benefit', 'top_5', 'top_3']
    agg_dict = {key: how for key in ['C_weighted_mean', 'C_core'] + comp_list}
    agg_dict[own_company] = how
    agg_dict['sales'] = 'sum'
    agg_dict['counter'] = 'sum'
    agg_dict['DGS5'] = 'last'
    agg_dict['VIXCLS'] = 'last'
    agg_dict['sales_by_contract_date'] = 'sum'
    return agg_dict


def _add_semantic_features(df: pd.DataFrame, rolling: int) -> pd.DataFrame:
    """Add base semantic features with smoothing."""
    from src.config.product_config import get_competitor_config

    competitor_config = get_competitor_config()
    own_company = competitor_config.own_company

    df['sales_target'] = df['sales'].rolling(rolling).mean()
    df['sales_by_contract_date'] = df['sales_by_contract_date'].rolling(rolling).mean()
    df['competitor_core.aggregate'] = df['C_core'].rolling(rolling).mean()
    df['competitor_weighted.aggregate'] = df['C_weighted_mean'].rolling(rolling).mean()
    df['C_core'] = df['competitor_core.aggregate']
    df['C'] = df['competitor_weighted.aggregate']
    df['P'] = df[own_company].rolling(rolling).mean()
    df['competitive.spread_base.aggregate'] = df['P'] - df['C']
    df['mean_diff'] = df['competitive.spread_base.aggregate']
    df['date'] = pd.to_datetime(df['date'])
    df['C_median'] = df['raw_median']
    df['C_first'] = df['first_highest_benefit']
    df['C_second'] = df['second_highest_benefit']
    df['C_third'] = df['third_highest_benefit']
    df['C_top_3'] = df['top_3']
    df['C_top_5'] = df['top_5']
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal indicator features.

    Naming Convention (Feature Naming Unification 2026-01-26):
    - Uses _t0 suffix instead of _current for consistency
    """
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    for m in range(1, 13):
        df[f'seasonal.month_{m}.t0'] = (df['month'] == m)
    for q in range(1, 5):
        df[f'seasonal.q{q}.t0'] = (df['quarter'] == q)
        df[f'Q{q}'] = df[f'seasonal.q{q}.t0']
    return df


def time_series_week_agg_smoothed(
    df_ts: pd.DataFrame,
    rolling: int,
    freq: str = 'W',
    how: str = 'last'
) -> pd.DataFrame:
    """
    Create time series aggregated features using semantic notation.

    Parameters
    ----------
    df_ts : pd.DataFrame
        Daily time series data
    rolling : int
        Rolling window size for smoothing
    freq : str, default='W'
        Frequency for aggregation (W=weekly)
    how : str, default='last'
        Aggregation method for most columns

    Returns
    -------
    pd.DataFrame
        Weekly aggregated data with semantic feature names
    """
    agg_dict = _create_aggregation_dict(how)

    business_mask = df_ts['sales'].notna()
    df_ts.loc[business_mask, 'counter'] = 1
    df = df_ts.groupby(pd.Grouper(key="date", freq=freq)).agg(agg_dict).reset_index()

    df = _add_semantic_features(df, rolling)
    df = _add_temporal_features(df)

    return df


# =============================================================================
# LAG FEATURE HELPERS
# =============================================================================


def _add_rate_features(df: pd.DataFrame, M: int, temporal: str) -> None:
    """Add rate and competitor features at given lag."""
    df[f'prudential_rate.{temporal}'] = df.P.shift(M)
    df[f'competitor_weighted.{temporal}'] = df.C.shift(M)
    df[f'competitor_core.{temporal}'] = df['C_core'].shift(M)
    df[f'competitor_median.{temporal}'] = df['C_median'].shift(M)
    df[f'competitor_1st.{temporal}'] = df['C_first'].shift(M)
    df[f'competitor_2nd.{temporal}'] = df['C_second'].shift(M)
    df[f'competitor_3rd.{temporal}'] = df['C_third'].shift(M)
    df[f'competitor_top3.{temporal}'] = df['C_top_3'].shift(M)
    df[f'competitor_top5.{temporal}'] = df['C_top_5'].shift(M)


def _add_derived_features(df: pd.DataFrame, M: int, temporal: str) -> None:
    """Add derived polynomial and interaction features at given lag."""
    df[f'derived.comp_2nd_3rd_avg.{temporal}'] = (
        df[f'competitor_3rd.{temporal}'] + df[f'competitor_2nd.{temporal}']) / 2
    df[f'derived.comp_1st_2nd_avg.{temporal}'] = (
        df[f'competitor_1st.{temporal}'] + df[f'competitor_2nd.{temporal}']) / 2
    df[f'competitive.momentum.{temporal}'] = df['C'].shift(M).diff(2)
    df[f'competitive.top5_momentum.{temporal}'] = df['C_top_5'].shift(M).diff(2)
    df[f'derived.pru_squared.{temporal}'] = df[f'prudential_rate.{temporal}'] ** 2
    df[f'derived.comp_squared.{temporal}'] = df[f'competitor_weighted.{temporal}'] ** 2
    df[f'derived.pru_times_comp.{temporal}'] = (
        df[f'competitor_weighted.{temporal}'] * df[f'prudential_rate.{temporal}'])
    df[f'derived.pru_cubed.{temporal}'] = df[f'prudential_rate.{temporal}'] ** 3
    df[f'derived.pru_sq_times_comp.{temporal}'] = (
        df[f'prudential_rate.{temporal}'] ** 2) * df[f'competitor_weighted.{temporal}']
    df[f'derived.pru_times_comp_sq.{temporal}'] = (
        df[f'prudential_rate.{temporal}'] * (df[f'competitor_weighted.{temporal}'] ** 2))
    df[f'derived.comp_cubed.{temporal}'] = df[f'competitor_weighted.{temporal}'] ** 3


def _create_lag_features_impl(df: pd.DataFrame) -> pd.DataFrame:
    """Internal implementation of lag feature creation (no split awareness).

    This is the core lag feature logic extracted for use by the split-aware
    wrapper function. Should NOT be called directly in production code.

    Naming Convention (Feature Naming Unification 2026-01-26):
    - All temporal suffixes use _t{N} format: _t0, _t1, _t2, etc.
    - Previous _current suffix normalized to _t0
    - Enables consistent regex: (.+)_t(\\d+)$
    """
    for k in range(0, 18, 1):
        M = k
        # Unified temporal suffix: always use t{k} (no more 'current')
        temporal = f't{k}'
        lead_temporal = f'lead{k}' if k > 0 else 'lead0'

        # Sales features - backward looking (safe for causal identification)
        df[f'sales_volume.{temporal}'] = df['sales_target'].shift(M)
        df[f'sales_target.contract_{temporal}'] = df['sales_by_contract_date'].shift(M)

        # Forward-looking features: Only create for k <= 1 (t0 and lead1)
        # Lead1 is preserved for forecast evaluation; lead2-17 violate causal ID
        # See Audit Issue #2: Forward-Looking Features Exist in Codebase
        if k <= 1:
            df[f'sales_target.{lead_temporal}'] = df['sales_target'].shift(-M)
            df[f'sales_target.contract_{lead_temporal}'] = df['sales_by_contract_date'].shift(-M)

        _add_rate_features(df, M, temporal)
        _add_derived_features(df, M, temporal)

        # Economic indicators
        df[f'econ.treasury_5y_momentum.{temporal}'] = df['DGS5'].shift(M).diff(2)
        df[f'econ.treasury_5y.{temporal}'] = df['DGS5'].shift(M)
        df[f'market.volatility.{temporal}'] = df['VIXCLS'].shift(M)
        df[f'competitive.spread_mean.{temporal}'] = df['mean_diff'].shift(M)

        # Seasonal indicators
        for q in range(1, 5):
            df[f'seasonal.q{q}.{temporal}'] = df[f'Q{q}'].shift(M)

    return df


def create_lag_features(
    df: pd.DataFrame,
    training_cutoff_date: str,
    date_column: str = "date"
) -> pd.DataFrame:
    """Create lag features with explicit split awareness.

    REQUIRED: training_cutoff_date must be explicitly provided.
    Philosophy: Fail-fast, explicit parameters, no silent defaults.

    Lag features are computed separately for train (before cutoff) and
    test (after cutoff) to prevent information leakage from test to train.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data with base features
    training_cutoff_date : str
        REQUIRED. ISO format date (YYYY-MM-DD) marking train/test boundary.
        Lag features computed separately for train (before) and test (after).
        This prevents information leakage from test to train.
    date_column : str, default="date"
        Column containing dates

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features computed split-aware

    Raises
    ------
    ValueError
        If training_cutoff_date is None or empty

    Examples
    --------
    >>> df = create_lag_features(df, "2024-01-01")  # OK
    >>> df = create_lag_features(df)  # Raises TypeError - required arg
    """
    if not training_cutoff_date:
        raise ValueError(
            "training_cutoff_date is REQUIRED. No default assumed. "
            "Pass explicit date: '2024-01-01'. "
            "This prevents information leakage from test to train splits."
        )

    # Convert cutoff to datetime
    cutoff = pd.to_datetime(training_cutoff_date)

    # Split data
    train_mask = df[date_column] < cutoff
    test_mask = ~train_mask

    # Compute lags separately to prevent leakage
    train_df = _create_lag_features_impl(df[train_mask].copy())
    test_df = _create_lag_features_impl(df[test_mask].copy())

    # Recombine and sort
    result = pd.concat([train_df, test_df], ignore_index=True)
    result = result.sort_values(date_column).reset_index(drop=True)

    return result


# =============================================================================
# MODULE EXPORTS (public API only)
# =============================================================================

__all__ = [
    # CPI adjustment (public)
    'cpi_adjustment',
    # Weekly aggregation (public)
    'time_series_week_agg_smoothed',
    # Lag features (public)
    'create_lag_features',
]
