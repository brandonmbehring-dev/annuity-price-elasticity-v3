"""
Competitive Feature Engineering for RILA Price Elasticity.

This module handles competitive rate calculations and semantic mappings:
- Median, top-N, and positional competitor rankings
- WINK weighted mean calculation with market share weights
- Semantic C_* notation mappings
- Backward compatibility shortcuts

Module Architecture (Phase 6.3e Split):
- competitive_features.py: Competitive rankings + WINK (this file)
- engineering.py: Core engineering + time series (thin wrapper)

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

import numpy as np
import pandas as pd
from typing import Dict, List


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def _validate_company_columns_config(company_columns: List[str], min_companies: int) -> None:
    """Validate company columns configuration parameters.

    Parameters
    ----------
    company_columns : List[str]
        List of company column names
    min_companies : int
        Minimum number of companies required

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if not company_columns:
        raise ValueError(
            "CRITICAL: company_columns parameter cannot be empty. "
            "Business impact: Cannot calculate competitor rates without company data. "
            "Required action: Verify company_columns list is properly configured in pipeline settings."
        )

    if min_companies <= 0:
        raise ValueError(
            f"CRITICAL: min_companies must be positive, got {min_companies}. "
            f"Business impact: Invalid configuration will prevent competitive analysis. "
            f"Required action: Set min_companies to a positive integer (recommended: 3 or more)."
        )


def _get_available_companies(
    df: pd.DataFrame, company_columns: List[str], min_companies: int, context: str
) -> List[str]:
    """Find available company columns and validate minimum requirement.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for columns
    company_columns : List[str]
        List of company column names to look for
    min_companies : int
        Minimum number of companies required
    context : str
        Description of the calling context for error messages

    Returns
    -------
    List[str]
        List of available company column names

    Raises
    ------
    ValueError
        If insufficient companies available
    """
    available_companies = [col for col in company_columns if col in df.columns]

    if len(available_companies) < min_companies:
        raise ValueError(
            f"Insufficient company data for {context}. "
            f"Found {len(available_companies)} companies, need minimum {min_companies}. "
            f"Available: {available_companies}. "
            f"Missing: {set(company_columns) - set(df.columns)}. "
            f"Check data loading and competitive rate filtering."
        )

    return available_companies


# =============================================================================
# MEDIAN COMPETITOR RANKINGS
# =============================================================================


def calculate_median_competitor_rankings(
    df: pd.DataFrame, company_columns: List[str], min_companies: int
) -> pd.DataFrame:
    """Calculate median competitor rankings from company rate columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing company rate columns
    company_columns : List[str]
        List of company column names to use for ranking calculations
    min_companies : int
        Minimum number of companies required for valid calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'raw_median' column containing median competitor rates

    Raises
    ------
    ValueError
        If insufficient company data available or no valid columns found
    """
    _validate_company_columns_config(company_columns, min_companies)
    available_companies = _get_available_companies(
        df, company_columns, min_companies, "median competitor rankings"
    )

    result = df.copy()
    company_data = result[available_companies].fillna(0)
    result['raw_median'] = company_data.median(axis=1)

    return result


# =============================================================================
# TOP-N COMPETITOR RANKINGS
# =============================================================================


def _compute_top_n_averages(
    sorted_values: np.ndarray, num_companies: int, company_data: pd.DataFrame
) -> tuple:
    """Compute top N averages from sorted values.

    Parameters
    ----------
    sorted_values : np.ndarray
        Values sorted in descending order per row
    num_companies : int
        Number of available companies
    company_data : pd.DataFrame
        Original company data for fallback calculations

    Returns
    -------
    tuple
        (top_3_values, top_5_values) as numpy arrays
    """
    if num_companies >= 5:
        top_5 = np.mean(sorted_values[:, :5], axis=1)
        top_3 = np.mean(sorted_values[:, :3], axis=1)
    elif num_companies >= 3:
        top_3 = np.mean(sorted_values[:, :3], axis=1)
        top_5 = np.mean(sorted_values, axis=1)  # Use all available companies
    else:
        # Fallback for limited data scenarios
        top_3 = company_data.mean(axis=1).values
        top_5 = company_data.mean(axis=1).values

    return top_3, top_5


def calculate_top_competitor_rankings(
    df: pd.DataFrame, company_columns: List[str], min_companies: int
) -> pd.DataFrame:
    """Calculate top N competitor rankings using highest rates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing company rate columns
    company_columns : List[str]
        List of company column names to use for ranking calculations
    min_companies : int
        Minimum number of companies required for valid calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'top_3' and 'top_5' columns containing average of top N rates

    Raises
    ------
    ValueError
        If insufficient company data available for top ranking calculations
    """
    available_companies = _get_available_companies(
        df, company_columns, min_companies, "top competitor rankings"
    )

    result = df.copy()
    company_data = result[available_companies].fillna(0)
    sorted_values = np.sort(company_data.values, axis=1)[:, ::-1]  # Sort descending

    top_3, top_5 = _compute_top_n_averages(
        sorted_values, len(available_companies), company_data
    )
    result['top_3'] = top_3
    result['top_5'] = top_5

    return result


# =============================================================================
# POSITIONAL COMPETITOR RANKINGS
# =============================================================================


def _extract_positional_rankings(sorted_values: np.ndarray) -> tuple:
    """Extract positional rankings from sorted values with edge case handling.

    Parameters
    ----------
    sorted_values : np.ndarray
        Values sorted in descending order per row

    Returns
    -------
    tuple
        (first, second, third) highest benefit values as numpy arrays
    """
    first = sorted_values[:, 0]
    second = sorted_values[:, 1] if sorted_values.shape[1] > 1 else sorted_values[:, 0]
    third = sorted_values[:, 2] if sorted_values.shape[1] > 2 else sorted_values[:, 0]

    return first, second, third


def calculate_position_competitor_rankings(
    df: pd.DataFrame, company_columns: List[str], min_companies: int
) -> pd.DataFrame:
    """Calculate positional competitor rankings (1st, 2nd, 3rd highest rates).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing company rate columns
    company_columns : List[str]
        List of company column names to use for ranking calculations
    min_companies : int
        Minimum number of companies required for valid calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with added positional ranking columns: first_highest_benefit,
        second_highest_benefit, third_highest_benefit

    Raises
    ------
    ValueError
        If insufficient company data available for positional rankings
    """
    available_companies = _get_available_companies(
        df, company_columns, min_companies, "positional competitor rankings"
    )

    result = df.copy()
    company_data = result[available_companies].fillna(0)
    sorted_values = np.sort(company_data.values, axis=1)[:, ::-1]  # Sort descending

    first, second, third = _extract_positional_rankings(sorted_values)
    result['first_highest_benefit'] = first
    result['second_highest_benefit'] = second
    result['third_highest_benefit'] = third

    return result


# =============================================================================
# SEMANTIC MAPPINGS
# =============================================================================


def _get_semantic_mappings() -> Dict[str, str]:
    """Return the semantic mapping dictionary for competitive features.

    Returns
    -------
    Dict[str, str]
        Mapping from raw column names to semantic C_* notation
    """
    return {
        'raw_median': 'C_median',
        'top_3': 'C_top_3',
        'top_5': 'C_top_5',
        'first_highest_benefit': 'C_first',
        'second_highest_benefit': 'C_second',
        'third_highest_benefit': 'C_third'
    }


def _validate_ranking_columns(
    df: pd.DataFrame, ranking_columns: List[str], semantic_mappings: Dict[str, str]
) -> None:
    """Validate that required ranking columns exist in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    ranking_columns : List[str]
        List of required ranking column names
    semantic_mappings : Dict[str, str]
        Mapping dictionary for reference

    Raises
    ------
    ValueError
        If required columns are missing
    """
    missing_columns = [
        col for col in ranking_columns
        if col in semantic_mappings and col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required ranking columns for semantic mapping: {missing_columns}. "
            f"Available columns: {sorted(df.columns)}. "
            f"Ensure ranking calculation functions have been executed first."
        )


def apply_competitive_semantic_mappings(
    df: pd.DataFrame, ranking_columns: List[str]
) -> pd.DataFrame:
    """Apply semantic mappings from raw rankings to C_* notation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw ranking columns
    ranking_columns : List[str]
        List of raw ranking column names to map to semantic notation

    Returns
    -------
    pd.DataFrame
        DataFrame with added semantic C_* columns mapped from raw rankings

    Raises
    ------
    ValueError
        If required ranking columns are missing from DataFrame
    """
    semantic_mappings = _get_semantic_mappings()
    _validate_ranking_columns(df, ranking_columns, semantic_mappings)

    result = df.copy()
    for raw_col, semantic_col in semantic_mappings.items():
        if raw_col in result.columns:
            result[semantic_col] = result[raw_col]

    return result


# =============================================================================
# COMPATIBILITY SHORTCUTS
# =============================================================================


def create_competitive_compatibility_shortcuts(
    df: pd.DataFrame, weighted_mean_col: str, prudential_col: str
) -> pd.DataFrame:
    """Create backward compatibility shorthand variables for competitive analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing competitive feature columns
    weighted_mean_col : str
        Column name for weighted mean competitor rate (maps to 'C')
    prudential_col : str
        Column name for Prudential rate (maps to 'P')

    Returns
    -------
    pd.DataFrame
        DataFrame with added shorthand compatibility variables: C, P

    Raises
    ------
    ValueError
        If required source columns for shortcuts are missing

    Examples
    --------
    >>> df = pd.DataFrame({'C_weighted_mean': [4.0, 4.2], 'Prudential': [4.5, 4.7]})
    >>> result = create_competitive_compatibility_shortcuts(df, 'C_weighted_mean', 'Prudential')
    >>> assert 'C' in result.columns and 'P' in result.columns
    """
    # Validate required source columns exist
    missing_columns = []
    if weighted_mean_col not in df.columns:
        missing_columns.append(weighted_mean_col)
    if prudential_col not in df.columns:
        missing_columns.append(prudential_col)

    if missing_columns:
        raise ValueError(
            f"Missing required columns for compatibility shortcuts: {missing_columns}. "
            f"Available columns: {sorted([col for col in df.columns if 'competitor' in col.lower() or 'prudential' in col.lower()])}. "
            f"Check that competitive rate processing has completed successfully."
        )

    result = df.copy()

    # Create backward compatibility shorthand variables
    result['C'] = result[weighted_mean_col]  # Competitor weighted mean
    result['P'] = result[prudential_col]     # Prudential rate

    return result


# =============================================================================
# WINK WEIGHTED MEAN
# =============================================================================


def wink_weighted_mean(df: pd.DataFrame, df_quarter_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate WINK weighted mean competitor rates using quarterly market share weights.

    Migrated from helpers/feature_engineering_functions_RILA.py for DRY compliance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing competitor rate data with date column
    df_quarter_weights : pd.DataFrame
        Quarterly market share weights data

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with C_weighted_mean and C_core columns added
    """
    from src.config.product_config import get_competitor_config

    competitor_config = get_competitor_config()

    # Competitors used in weighted mean (excludes Allianz, Trans from full list)
    # Full list: Allianz, Athene, Brighthouse, Equitable, Jackson, Lincoln, Symetra, Trans
    # Weighted mean excludes: Allianz, Trans
    excluded_from_weighted = {'Allianz', 'Trans'}
    weighted_competitors = [c for c in competitor_config.rila_competitors
                           if c not in excluded_from_weighted]
    competitors_weight = [f'{c}_weight' for c in weighted_competitors]

    df['current_quarter'] = df['date'].dt.year.astype(
        'string') + '_Q' + df['date'].dt.quarter.astype('string')
    df = df.merge(df_quarter_weights,
                  on='current_quarter').drop_duplicates().reset_index(drop=True)

    weights = (df[weighted_competitors].fillna(0).values *
               df[competitors_weight].fillna(0).values)

    df['C_weighted_mean'] = weights.sum(axis=1)/(1-df['Allianz_weight']-df['Trans_weight'])

    # Core competitors from config
    core_competitors = list(competitor_config.core_competitors)
    df['C_core'] = df[core_competitors].sum(axis=1) / len(core_competitors)

    return df


# Backward compatibility alias with deprecation warning
def WINK_weighted_mean(df: pd.DataFrame, df_quarter_weights: pd.DataFrame) -> pd.DataFrame:
    """Deprecated: Use wink_weighted_mean instead."""
    import warnings
    warnings.warn(
        "WINK_weighted_mean is deprecated; use wink_weighted_mean instead",
        DeprecationWarning,
        stacklevel=2
    )
    return wink_weighted_mean(df, df_quarter_weights)


# =============================================================================
# COMPETITIVE SPREAD
# =============================================================================


def calculate_competitive_spread(
    df: pd.DataFrame,
    prudential_column: str,
    competitor_column: str,
    spread_column: str
) -> pd.DataFrame:
    """Calculate competitive spread (Prudential rate - Competitor rate).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing rate columns
    prudential_column : str
        Column name for Prudential rate
    competitor_column : str
        Column name for competitor rate
    spread_column : str
        Output column name for spread

    Returns
    -------
    pd.DataFrame
        DataFrame with added spread column

    Raises
    ------
    ValueError
        If required columns are missing
    """
    result = df.copy()

    if prudential_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Prudential column '{prudential_column}' not found in DataFrame. "
            f"Business impact: Cannot calculate competitive spread, pricing analysis will fail. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify Prudential rate data was successfully loaded and transformed."
        )
    if competitor_column not in result.columns:
        raise ValueError(
            f"CRITICAL: Competitor column '{competitor_column}' not found in DataFrame. "
            f"Business impact: Cannot calculate competitive spread, pricing analysis will fail. "
            f"Available columns: {list(result.columns)}. "
            f"Required action: Verify competitor rate data was successfully aggregated."
        )

    result[spread_column] = result[prudential_column] - result[competitor_column]

    return result


# =============================================================================
# MODULE EXPORTS (public API only)
# =============================================================================

__all__ = [
    # Ranking functions (public)
    'calculate_median_competitor_rankings',
    'calculate_top_competitor_rankings',
    'calculate_position_competitor_rankings',

    # Semantic mappings (public)
    'apply_competitive_semantic_mappings',

    # Compatibility (public)
    'create_competitive_compatibility_shortcuts',

    # WINK weighted mean (public)
    'wink_weighted_mean',
    'WINK_weighted_mean',  # Deprecated alias for backward compatibility

    # Spread calculation (public)
    'calculate_competitive_spread',
]
