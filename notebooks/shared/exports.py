"""
Shared BI Export Utilities for Notebooks.

Provides standardized export functions for Tableau-ready datasets
and competitor rate formatting that appear identically across product notebooks.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


# =============================================================================
# PRODUCT NAME MAPPINGS
# =============================================================================


ProductNameDict: Dict[str, str] = {
    "Prudential": "Prudential Flexguard Indexed Variable Annuity",
    "Allianz": "Allianz Index Advantage+ NF Variable Annuity",
    "Athene": "Athene Amplify 2.0 NF",
    "Brighthouse": "Brighthouse Shield Level Select 6-year",
    "Equitable": "Structured Capital Strategies Plus 21",
    "Jackson": "Jackson Market Link Pro Advisory Single Premium Deferred Index-Linked Annuity",
    "Lincoln": "Lincoln Level Advantage Design Advisory Share Individual Variable and Indexed-Linked Annuity",
    "Symetra": "Symetra Trek Frontier Index-Linked Annuity",
    "Trans": "Transamerica Structured Index Advantage Annuity",
    "weighted_mean": "Weighted Mean By Market Share of Competitors Cap Rate",
}

COMPETITOR_LIST = [
    "Allianz",
    "Athene",
    "Brighthouse",
    "Equitable",
    "Jackson",
    "Lincoln",
    "Symetra",
    "Trans",
    "Prudential",
]


# =============================================================================
# BI EXPORT FUNCTIONS
# =============================================================================


def prepare_bi_export(
    df_output_dollar: pd.DataFrame,
    df_output_pct: pd.DataFrame,
    df: pd.DataFrame,
    df_rates: pd.DataFrame,
    current_date: str,
    product_config: Dict[str, Any],
    tableau_config: Dict[str, Any],
    melt_function: callable,
) -> pd.DataFrame:
    """
    Prepare Tableau-ready BI export dataset.

    Combines dollar and percentage confidence intervals with business metadata.

    Parameters
    ----------
    df_output_dollar : pd.DataFrame
        Dollar impact confidence intervals.
    df_output_pct : pd.DataFrame
        Percentage change confidence intervals.
    df : pd.DataFrame
        Sales data for rate context.
    df_rates : pd.DataFrame
        WINK competitive rates data.
    current_date : str
        Prediction date string (YYYY-MM-DD).
    product_config : dict
        Product metadata configuration.
    tableau_config : dict
        Tableau formatting configuration.
    melt_function : callable
        The melt_dataframe_for_tableau function from src.models.inference.

    Returns
    -------
    pd.DataFrame
        Tableau-ready BI export dataset.
    """
    # Combine confidence intervals into unified BI format
    df_num = df_output_dollar.melt(id_vars=["rate_change_in_basis_points"]).rename(
        columns={"variable": "range", "value": "dollar"}
    )
    df_pct = df_output_pct.melt(id_vars=["rate_change_in_basis_points"]).rename(
        columns={"variable": "range", "value": "pct_change"}
    )

    # Merge dollar and percentage data
    df_to_bi = df_num.merge(
        df_pct[["rate_change_in_basis_points", "range", "pct_change"]],
        on=["rate_change_in_basis_points", "range"],
    )

    # Generate Tableau-formatted export with business metadata
    df_to_bi_melt = melt_function(
        confidence_intervals=df_to_bi,
        sales_data=df,
        prediction_date=current_date,
        prudential_rate_col=tableau_config["prudential_rate_col"],
        competitor_rate_col=tableau_config["competitor_rate_col"],
        sales_lag_cols=tableau_config["sales_lag_cols"],
        sales_rounding_power=tableau_config["sales_rounding_power"],
    )

    # Add product and competitive metadata
    df_to_bi_melt["product"] = product_config["product_name"]
    df_to_bi_melt["Synthetic Competitor"] = df_rates["C_weighted_mean"].iloc[-1]
    df_to_bi_melt["Weighted Mean By Market Share of Competitors Cap Rate"] = df_rates[
        "C_weighted_mean"
    ].iloc[-1]

    # Apply business rounding standards
    mask_dollar = df_to_bi_melt["output_type"] == "dollar"
    mask_pct_change = df_to_bi_melt["output_type"] == "pct_change"

    df_to_bi_melt.loc[mask_dollar, "value"] = np.round(
        df_to_bi_melt.loc[mask_dollar, "value"], tableau_config["sales_rounding_power"]
    )
    df_to_bi_melt.loc[mask_pct_change, "value"] = np.round(
        df_to_bi_melt.loc[mask_pct_change, "value"], 2
    )

    return df_to_bi_melt


def add_product_metadata(
    df_merge: pd.DataFrame,
    df: pd.DataFrame,
    df_rates: pd.DataFrame,
    version: str = "v2_0",
) -> pd.DataFrame:
    """
    Add product and competitor metadata to BI export.

    This is the final formatting step before CSV export.

    Parameters
    ----------
    df_merge : pd.DataFrame
        Melted BI export DataFrame.
    df : pd.DataFrame
        Sales DataFrame for rate context.
    df_rates : pd.DataFrame
        WINK rates DataFrame for competitor rates.
    version : str
        Version string (e.g., "v2_0").

    Returns
    -------
    pd.DataFrame
        DataFrame with full product metadata.
    """
    prudential_current = df["prudential_rate_current"].iloc[-1]
    competitor_current = df["competitor_mid_current"].iloc[-1]

    # Format version
    df_merge["version"] = version.replace("_", ".").replace("v", "")

    # Add competitor rates
    for competitor in COMPETITOR_LIST:
        if competitor in df_rates.columns:
            df_merge[ProductNameDict[competitor]] = np.round(df_rates[competitor].iloc[-1] * 100)

    # Add synthetic competitor and weighted mean
    df_merge["Synthetic Competitor"] = np.round(100 * competitor_current)
    df_merge["Weighted Mean By Market Share of Competitors Cap Rate"] = np.round(
        100 * competitor_current
    )
    df_merge["Prudential Cap Rate"] = np.round(100 * prudential_current)

    # Convert prediction date to datetime
    df_merge["prediction_date"] = pd.to_datetime(df_merge["prediction_date"])

    return df_merge


def format_competitor_rates(
    df_rates: pd.DataFrame,
    as_percentage: bool = True,
) -> Dict[str, float]:
    """
    Extract latest competitor rates as dictionary.

    Parameters
    ----------
    df_rates : pd.DataFrame
        WINK rates DataFrame.
    as_percentage : bool
        If True, multiply by 100 and round.

    Returns
    -------
    Dict[str, float]
        Competitor name to rate mapping.
    """
    rates = {}
    for competitor in COMPETITOR_LIST:
        if competitor in df_rates.columns:
            rate = df_rates[competitor].iloc[-1]
            if as_percentage:
                rate = np.round(rate * 100)
            rates[ProductNameDict[competitor]] = rate
    return rates


def print_bi_export_summary(
    df_to_bi_melt: pd.DataFrame,
    product_config: Dict[str, Any],
    current_date: str,
) -> None:
    """
    Print BI export summary to console.

    Parameters
    ----------
    df_to_bi_melt : pd.DataFrame
        Tableau-ready BI export.
    product_config : dict
        Product metadata configuration.
    current_date : str
        Prediction date string.
    """
    print(f"Tableau-ready BI export completed:")
    print(f"  Product: {product_config['product_name']}")
    print(f"  Version: {product_config['version']}")
    print(f"  Prediction Date: {current_date}")
    print(f"  Total Records: {len(df_to_bi_melt):,}")
    print(f"  Rate Scenarios: {df_to_bi_melt['rate_change_in_basis_points'].nunique()}")
    print(f"  Output Types: {df_to_bi_melt['output_type'].unique()}")

    if "Prudential Cap Rate" in df_to_bi_melt.columns:
        print(f"  Latest Prudential Rate: {df_to_bi_melt['Prudential Cap Rate'].iloc[0]:.1f}%")

    if "Synthetic Competitor" in df_to_bi_melt.columns:
        print(f"  Latest Competitive Rate: {df_to_bi_melt['Synthetic Competitor'].iloc[0]:.2f}%")
