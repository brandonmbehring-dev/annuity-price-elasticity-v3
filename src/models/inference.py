"""
RILA Price Elasticity Inference Functions - Canonical Implementation.

This module contains the canonical implementation of price elasticity inference
functions for RILA product analysis. All functions maintain mathematical
equivalence with the original helper implementations while following clean
architecture patterns.

Business Context:
- RILA product price elasticity analysis (supports multiple products via ProductConfig)
- Bootstrap Ridge Regression ensemble for weekly sales forecasting
- Competitive intelligence integration for strategic pricing decisions
- Economic theory constraint validation for production deployment

Architecture Pattern:
- Configuration-driven using src.config.config_builder functions
- Product-specific behavior via src.config.product_config
- Canonical imports: from src.models.inference import function_name
- Mathematical equivalence maintained with 1e-12 precision tolerance
- HARD GATE validation enforced for all migrated functions

Module Architecture (Phase 6.3d Split):
- inference_training.py: Training pipeline (TrainingData, prepare_training_data, train_bootstrap_model)
- inference_scenarios.py: Baseline + scenario predictions (center_baseline, rate_adjustments, confidence_interval)
- inference.py: Thin wrapper + Tableau export + public API (this file)

Author: Claude Code Refactoring
Date: 2024-11-18
Standards: CODING_STANDARDS.md v2.0
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

from src.config.product_config import get_default_product


# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES
# =============================================================================

# Training module
from src.models.inference_training import (
    TrainingData,
    prepare_training_data,
    train_bootstrap_model,
    transform_prediction_features,
    _get_product_name,
)

# Scenarios module
from src.models.inference_scenarios import (
    validate_center_baseline_inputs,
    validate_rate_adjustments_inputs,
    validate_confidence_interval_inputs,
    center_baseline,
    rate_adjustments,
    confidence_interval,
    apply_feature_adjustments,
    _resolve_center_baseline_params,
    _generate_baseline_predictions,
    _generate_scenario_predictions,
    _calculate_quantile_bounds,
    _initialize_ci_dataframe,
    _compute_quantiles,
    _convert_to_basis_points,
)


# =============================================================================
# TABLEAU EXPORT VALIDATION
# =============================================================================


def validate_melt_dataframe_inputs(
    confidence_intervals: pd.DataFrame,
    sales_data: pd.DataFrame,
    prediction_date: str,
    prudential_rate_col: str,
    competitor_rate_col: str,
    sales_lag_cols: List[str],
    sales_rounding_power: int
) -> None:
    """
    Validate inputs for melt_dataframe_for_tableau function.

    Parameters
    ----------
    confidence_intervals : pd.DataFrame
        Confidence interval data to melt
    sales_data : pd.DataFrame
        Sales data with rate and lag columns
    prediction_date : str
        Date string for prediction timestamp
    prudential_rate_col : str
        Column name for Prudential rate data
    competitor_rate_col : str
        Column name for competitor rate data
    sales_lag_cols : List[str]
        Column names for sales lag data
    sales_rounding_power : int
        Power for sales rounding (e.g., -7 for nearest 10M)

    Raises
    ------
    ValueError
        If validation fails with business context
    """
    try:
        if confidence_intervals.empty:
            raise ValueError("Confidence intervals DataFrame cannot be empty")

        if sales_data.empty:
            raise ValueError("Sales data DataFrame cannot be empty")

        # Check required confidence interval columns
        required_ci_cols = ["rate_change_in_basis_points", "range"]
        missing_ci_cols = [col for col in required_ci_cols if col not in confidence_intervals.columns]
        if missing_ci_cols:
            raise ValueError(f"Missing confidence interval columns: {missing_ci_cols}")

        # Check required sales columns
        required_sales_cols = [prudential_rate_col, competitor_rate_col] + sales_lag_cols
        missing_sales_cols = [col for col in required_sales_cols if col not in sales_data.columns]
        if missing_sales_cols:
            raise ValueError(f"Missing sales data columns: {missing_sales_cols}")

        # Validate date format
        pd.to_datetime(prediction_date)

        # Validate rounding power
        if sales_rounding_power > 0:
            raise ValueError(f"Sales rounding power should be negative or zero, got {sales_rounding_power}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Melt dataframe input validation failed: {e}") from e


# =============================================================================
# TABLEAU EXPORT HELPERS
# =============================================================================


def calculate_sales_momentum(
    sales_data: pd.DataFrame,
    sales_lag_cols: List[str],
    sales_rounding_power: int
) -> int:
    """
    Calculate sales momentum metric for business intelligence.

    Parameters
    ----------
    sales_data : pd.DataFrame
        Sales data with lag columns
    sales_lag_cols : List[str]
        Column names for sales lag calculations
    sales_rounding_power : int
        Power for rounding (e.g., -7 for nearest 10M)

    Returns
    -------
    int
        Rounded sales momentum metric
    """
    try:
        # Sum sales lag values (original: sales_lag_2 + sales_lag_1)
        total_sales = sum([
            sales_data[col].iloc[-1]
            for col in sales_lag_cols
        ])

        # Apply business rounding (original: -7 for nearest 10M)
        return int(np.round(total_sales, sales_rounding_power))

    except Exception as e:
        raise ValueError(f"Sales momentum calculation failed: {e}") from e


def _melt_ci_to_long_format(
    confidence_intervals: pd.DataFrame
) -> pd.DataFrame:
    """
    Melt confidence intervals DataFrame to long format.

    Parameters
    ----------
    confidence_intervals : pd.DataFrame
        Wide-format confidence intervals with rate_change_in_basis_points and range

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with output_type column
    """
    return confidence_intervals.melt(
        id_vars=["rate_change_in_basis_points", "range"]
    ).rename(columns={"variable": "output_type"})


def _add_business_context_columns(
    df_melted: pd.DataFrame,
    sales_data: pd.DataFrame,
    prediction_date: str,
    prudential_rate_col: str,
    competitor_rate_col: str
) -> pd.DataFrame:
    """
    Add business context columns to melted DataFrame.

    Parameters
    ----------
    df_melted : pd.DataFrame
        Melted DataFrame to augment
    sales_data : pd.DataFrame
        Sales data with rate columns
    prediction_date : str
        Prediction timestamp
    prudential_rate_col : str
        Column name for Prudential rate
    competitor_rate_col : str
        Column name for competitor rate

    Returns
    -------
    pd.DataFrame
        DataFrame with business context columns added
    """
    df_melted["prediction_date"] = prediction_date
    df_melted["Prudential Cap Rate"] = sales_data[prudential_rate_col].iloc[-1]
    df_melted["Weighted Mean By Market Share of Competitors Cap Rate"] = (
        sales_data[competitor_rate_col].iloc[-1]
    )
    return df_melted


def _add_sales_momentum_column(
    df_melted: pd.DataFrame,
    sales_data: pd.DataFrame,
    sales_lag_cols: List[str],
    sales_rounding_power: int
) -> pd.DataFrame:
    """
    Add sales momentum metric column to melted DataFrame.

    Parameters
    ----------
    df_melted : pd.DataFrame
        Melted DataFrame to augment
    sales_data : pd.DataFrame
        Sales data with lag columns
    sales_lag_cols : List[str]
        Column names for sales lag calculations
    sales_rounding_power : int
        Power for rounding (e.g., -7 for nearest 10M)

    Returns
    -------
    pd.DataFrame
        DataFrame with Previous Two Week Sales column added
    """
    df_melted["Previous Two Week Sales"] = calculate_sales_momentum(
        sales_data, sales_lag_cols, sales_rounding_power
    )
    return df_melted


# =============================================================================
# TABLEAU EXPORT
# =============================================================================


def melt_dataframe_for_tableau(
    confidence_intervals: pd.DataFrame,
    sales_data: pd.DataFrame,
    prediction_date: str,
    prudential_rate_col: str = "prudential_rate_current",
    competitor_rate_col: str = "competitor_mid_current",
    sales_lag_cols: List[str] = None,
    sales_rounding_power: int = -7
) -> pd.DataFrame:
    """
    Format confidence intervals for Tableau dashboards in long format with business context.

    Returns melted DataFrame with rate scenarios, CI bounds, and market intelligence columns.
    """
    try:
        if sales_lag_cols is None:
            sales_lag_cols = ['sales_target_t2', 'sales_target_t3']

        validate_melt_dataframe_inputs(
            confidence_intervals, sales_data, prediction_date,
            prudential_rate_col, competitor_rate_col,
            sales_lag_cols, sales_rounding_power
        )

        df_melted = _melt_ci_to_long_format(confidence_intervals)
        df_melted = _add_business_context_columns(
            df_melted, sales_data, prediction_date, prudential_rate_col, competitor_rate_col
        )
        df_melted = _add_sales_momentum_column(
            df_melted, sales_data, sales_lag_cols, sales_rounding_power
        )

        return df_melted

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Tableau formatting failed for {_get_product_name()} business intelligence "
            f"with {len(confidence_intervals) if 'confidence_intervals' in locals() else 'unknown'} records: {e}"
        ) from e


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    'TrainingData',

    # Training pipeline (from inference_training.py)
    'prepare_training_data',
    'train_bootstrap_model',
    'transform_prediction_features',

    # Validation (from inference_scenarios.py)
    'validate_center_baseline_inputs',
    'validate_rate_adjustments_inputs',
    'validate_confidence_interval_inputs',

    # Main entry points (from inference_scenarios.py)
    'center_baseline',
    'rate_adjustments',
    'confidence_interval',

    # Tableau export (this file)
    'validate_melt_dataframe_inputs',
    'calculate_sales_momentum',
    'melt_dataframe_for_tableau',

    # Public helpers
    'apply_feature_adjustments',
]
