"""
Data cleaning and standardization for clean_v0 pipeline.

This module handles data preprocessing operations used in 00_clean_v0.ipynb:
- TDE data filtering and cleaning
- WINK data standardization and processing
- Data validation and quality checks

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import config types - Fail fast with clear error if imports fail
from ..config.pipeline_config import PreprocessingWINKConfig, PreprocessingTDEConfig


def filter_dataframe_by_product_name(df: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """Filter DataFrame by product name.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing product data with 'product_name' column
    product_name : str
        Product name to filter for (exact match)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records matching the product name

    Raises
    ------
    ValueError
        If 'product_name' column is missing, product_name is empty, or no records found

    Examples
    --------
    >>> df = pd.DataFrame({'product_name': ['A', 'B', 'A'], 'sales': [100, 200, 150]})
    >>> result = filter_dataframe_by_product_name(df, 'A')
    >>> assert len(result) == 2
    """
    if 'product_name' not in df.columns:
        raise ValueError("Column 'product_name' not found in DataFrame")

    if not product_name:
        raise ValueError("product_name cannot be empty")

    result = df[df['product_name'] == product_name].copy()
    if len(result) == 0:
        raise ValueError(f"No records found for product: {product_name}")

    return result


def filter_by_buffer_rate(df: pd.DataFrame, buffer_rate: str) -> pd.DataFrame:
    """Filter DataFrame by buffer_rate column."""
    if 'buffer_rate' not in df.columns:
        raise ValueError("Column 'buffer_rate' not found")

    result = df[(df['buffer_rate'].notna()) & (df['buffer_rate'] == buffer_rate)]

    if len(result) == 0:
        raise ValueError(f"No records found with buffer_rate: {buffer_rate}")

    return result


def filter_by_term(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Filter DataFrame by term column."""
    if 'term' not in df.columns:
        raise ValueError("Column 'term' not found")

    result = df[(df['term'].notna()) & (df['term'] == term)]

    if len(result) == 0:
        raise ValueError(f"No records found with term: {term}")

    return result


def convert_column_to_datetime(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Convert column to datetime with validation."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    result = df.copy()
    try:
        result[column_name] = pd.to_datetime(result[column_name])
        return result
    except Exception as e:
        raise ValueError(f"Failed to convert column '{column_name}' to datetime: {e}") from e


def calculate_days_between_dates(df: pd.DataFrame, start_col: str, end_col: str, result_col: str) -> pd.DataFrame:
    """Calculate days between two date columns."""
    missing_cols = [col for col in [start_col, end_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing date columns: {missing_cols}")

    for col in [start_col, end_col]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be datetime type")

    result = df.copy()
    result[result_col] = (result[end_col] - result[start_col]).dt.days
    return result


def remove_null_values_from_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Remove rows with null values in specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found. Available: {sorted(df.columns)}")

    result = df[df[column_name].notna()]

    if len(result) == 0:
        raise ValueError(f"All values in column '{column_name}' are null")

    return result


def filter_column_by_range(df: pd.DataFrame, column_name: str, min_val: float, max_val: float) -> pd.DataFrame:
    """Filter DataFrame by numeric range on specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found. Available: {sorted(df.columns)}")

    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")

    mask = (df[column_name] >= min_val) & (df[column_name] <= max_val)
    result = df[mask]

    if len(result) == 0:
        raise ValueError(f"No values found in range [{min_val}, {max_val}] for column '{column_name}'")

    return result


def create_column_alias(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Create alias column from existing column."""
    if source_col not in df.columns:
        raise ValueError(f"Source column '{source_col}' not found. Available: {sorted(df.columns)}")

    result = df.copy()

    if target_col not in result.columns:
        result[target_col] = result[source_col]

    return result


def apply_quantile_threshold_filter(df: pd.DataFrame, column: str, quantile: float) -> pd.DataFrame:
    """Filter DataFrame by quantile threshold on specified column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {sorted(df.columns)}")

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be between 0.0 and 1.0, got {quantile}")

    threshold = np.quantile(df[column], quantile)
    result = df[df[column] < threshold]

    if len(result) == 0:
        raise ValueError(f"All values filtered out at quantile {quantile}")

    return result


def sort_dataframe_by_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Sort DataFrame by specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    return df.sort_values(by=column_name).reset_index(drop=True)


def rename_columns_with_mapping(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns using mapping dictionary."""
    missing_cols = set(column_mapping.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Cannot rename missing columns: {missing_cols}")

    return df.rename(columns=column_mapping)


def apply_rolling_mean(df: pd.DataFrame, column_name: str, window_size: int) -> pd.DataFrame:
    """Apply rolling mean to specified column."""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    if window_size <= 0:
        raise ValueError("window_size must be positive")

    result = df.copy()
    result[column_name] = result[column_name].rolling(window=window_size, min_periods=1).mean()
    return result


# =============================================================================
# WINK Data Processing Functions
# =============================================================================

def standardize_wink_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize WINK company names for consistent analysis."""
    df = df.copy()

    company_mapping = {
        'pruco life': 'Prudential',
        'allianz': 'Allianz',
        'athene annuity and life': 'Athene',
        'brighthouse': 'Brighthouse',
        'equitable financial life insurance company': 'Equitable',
        'equitable financial life insurance company of america': 'Equitable',
        'jackson national': 'Jackson',
        'lincoln national': 'Lincoln',
        'symetra': 'Symetra',
        'transamerica life': 'Trans'
    }

    df['companyName'] = df['companyName'].str.lower()
    for old_name, new_name in company_mapping.items():
        df.loc[df['companyName'] == old_name, 'companyName'] = new_name

    return df


def standardize_wink_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize WINK product names for analysis."""
    df = df.copy()
    df['productName'] = df['productName'].str.strip()
    return df


def round_rate_columns(df: pd.DataFrame, columns: List[str], decimal_places: int) -> pd.DataFrame:
    """Round specified rate columns to given decimal places."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(decimal_places)

    return df


def filter_by_product_type_name(df: pd.DataFrame, product_type: str) -> pd.DataFrame:
    """Filter DataFrame by product type name."""
    if 'productTypeName' not in df.columns:
        raise ValueError("Column 'productTypeName' not found")

    result = df[df['productTypeName'] == product_type]
    if len(result) == 0:
        raise ValueError(f"No records found with productTypeName: {product_type}")

    return result


def filter_by_participation_rate(df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
    """Filter DataFrame by participation rate."""
    if 'participationRate' not in df.columns:
        raise ValueError("Column 'participationRate' not found")

    result = df[df['participationRate'] == target_rate]
    if len(result) == 0:
        raise ValueError(f"No records found with participationRate: {target_rate}")

    return result


def filter_by_index_used(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Filter DataFrame by index used."""
    if 'indexUsed' not in df.columns:
        raise ValueError("Column 'indexUsed' not found")

    result = df[df['indexUsed'] == index_name]
    if len(result) == 0:
        raise ValueError(f"No records found with indexUsed: {index_name}")

    return result


def filter_by_buffer_rates_list(df: pd.DataFrame, buffer_rates: List[float]) -> pd.DataFrame:
    """Filter DataFrame by list of allowed buffer rates."""
    if 'bufferRate' not in df.columns:
        raise ValueError("Column 'bufferRate' not found")

    result = df[df['bufferRate'].isin(buffer_rates)]
    if len(result) == 0:
        raise ValueError(f"No records found with bufferRate in: {buffer_rates}")

    return result


def filter_by_buffer_modifier(df: pd.DataFrame, modifier: str) -> pd.DataFrame:
    """Filter DataFrame by buffer modifier."""
    if 'bufferModifier' not in df.columns:
        raise ValueError("Column 'bufferModifier' not found")

    result = df[df['bufferModifier'] == modifier]
    if len(result) == 0:
        raise ValueError(f"No records found with bufferModifier: {modifier}")

    return result


def handle_null_cap_rates(df: pd.DataFrame, default_rate: float) -> pd.DataFrame:
    """Handle null cap rates by filling with default value."""
    df = df.copy()

    if 'capRate' in df.columns:
        df['capRate'] = df['capRate'].fillna(default_rate)

    return df


def apply_cap_rate_ceiling(df: pd.DataFrame, max_rate: float) -> pd.DataFrame:
    """Apply ceiling to cap rates."""
    df = df.copy()

    if 'capRate' in df.columns:
        df['capRate'] = df['capRate'].clip(upper=max_rate)

    return df


def pivot_wink_rates_by_company(
    df: pd.DataFrame,
    product_ids: Dict[str, List[int]],
    start_date: str,
    current_time: datetime,
    rolling_days: int
) -> Tuple[pd.DataFrame, List[str]]:
    """Pivot WINK rates by company with rolling averages."""

    date_range = pd.date_range(start=start_date, end=current_time, freq='D')
    result_df = pd.DataFrame({'date': date_range})

    company_progress = []

    for company, ids in product_ids.items():
        company_data = df[df['productID'].isin(ids)].copy()

        if len(company_data) > 0:
            latest_record = company_data.loc[company_data['date'].idxmax()]

            progress_info = f"""-------------
{company}
{latest_record['date'].strftime('%Y-%m-%d')}
{latest_record['companyName']}
{latest_record['productName']}
{latest_record['capRate']*100:.2f}%
-------------"""
            company_progress.append(progress_info)

            daily_rates = company_data.groupby('date')['capRate'].mean().reset_index()
            daily_rates = daily_rates.set_index('date').reindex(date_range).ffill()

            daily_rates[company] = daily_rates['capRate'].rolling(window=rolling_days, min_periods=1).mean()

            result_df = result_df.merge(
                daily_rates[company].reset_index().rename(columns={'index': 'date'}),
                on='date',
                how='left'
            )

    return result_df, company_progress


def add_quarterly_period_column(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Add quarterly period columns."""
    df = df.copy()

    df['year'] = df[date_column].dt.year
    df['quarter'] = df[date_column].dt.quarter
    df['current_quarter'] = df['year'].astype(str) + '_Q' + df['quarter'].astype(str)

    return df


def calculate_weighted_competitive_mean(df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted competitive mean using market share weights."""
    from src.config.product_config import get_competitor_config

    df = df.copy()

    df = df.merge(weights_df, on='current_quarter', how='left')

    competitor_config = get_competitor_config()
    competitors = list(competitor_config.rila_competitors)

    weighted_sum = 0
    total_weight = 0

    for comp in competitors:
        if comp in df.columns and f'{comp}_weight' in df.columns:
            weight_col = f'{comp}_weight'
            df[weight_col] = df[weight_col].fillna(0)
            df[comp] = df[comp].fillna(0)

            weighted_sum += df[comp] * df[weight_col]
            total_weight += df[weight_col]

    df['C_weighted_mean'] = weighted_sum / (total_weight + 1e-8)

    core_competitors = [comp for comp in competitors if comp in df.columns]
    df['C_core'] = df[core_competitors].mean(axis=1, skipna=True)

    return df