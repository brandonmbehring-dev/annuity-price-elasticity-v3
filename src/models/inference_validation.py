"""
Inference Validation Module for RILA Price Elasticity.

This module contains all input validation functions for the inference pipeline.
Extracted from inference.py to improve maintainability and reduce module size.

Design Principles:
- Fail-fast validation with comprehensive error messages
- Business context included in all error messages
- Single responsibility per validation function
- Type hints for all parameters

Usage:
    from src.models.inference_validation import (
        validate_center_baseline_inputs,
        validate_rate_adjustments_inputs,
        validate_confidence_interval_inputs,
        validate_melt_dataframe_inputs
    )
"""

from datetime import datetime
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# Import product configuration for parametrized error messages
from src.config.product_config import get_default_product

# Conditional import for model types
try:
    from sklearn.ensemble import BaggingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaggingRegressor = None  # type: ignore


def _get_product_name() -> str:
    """Get current product name for error messages.

    Returns
    -------
    str
        Product name (e.g., "FlexGuard 6Y20B")
    """
    return get_default_product().name


# =============================================================================
# Helper Validation Functions
# =============================================================================

def _validate_dataframes(
    df: pd.DataFrame,
    df_rates: pd.DataFrame
) -> None:
    """Validate that required DataFrames are not None or empty.

    Parameters
    ----------
    df : pd.DataFrame
        RILA sales time series DataFrame
    df_rates : pd.DataFrame
        Competitive rates DataFrame

    Raises
    ------
    ValueError
        If either DataFrame is None or empty
    """
    if df is None or df.empty:
        raise ValueError(
            f"[ERROR] RILA sales DataFrame cannot be None or empty for {_get_product_name()} analysis"
        )
    if df_rates is None or df_rates.empty:
        raise ValueError("[ERROR] WINK competitive rates DataFrame cannot be None or empty")


def _validate_required_columns(
    df: pd.DataFrame,
    features: List[str],
    target_variable: str
) -> None:
    """Validate that target variable and features exist in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate columns against
    features : List[str]
        Feature column names required
    target_variable : str
        Target variable column name

    Raises
    ------
    ValueError
        If target or features are missing from DataFrame
    """
    if target_variable not in df.columns:
        raise ValueError(f"[ERROR] Target variable '{target_variable}' not found in sales DataFrame")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"[ERROR] Missing features in sales DataFrame: {missing_features}")


def _validate_date_format(current_date_of_mature_data: str) -> None:
    """Validate date string is in ISO format YYYY-MM-DD.

    Parameters
    ----------
    current_date_of_mature_data : str
        Date string to validate

    Raises
    ------
    ValueError
        If date format is invalid
    """
    try:
        datetime.strptime(current_date_of_mature_data, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"[ERROR] Invalid date format '{current_date_of_mature_data}'. Expected YYYY-MM-DD"
        )


def _validate_model_parameters(
    n_estimators: int,
    weight_decay_factor: float,
    random_state: int,
    ridge_alpha: float
) -> None:
    """Validate model hyperparameters for bootstrap ensemble.

    Parameters
    ----------
    n_estimators : int
        Number of bootstrap estimators (must be positive integer)
    weight_decay_factor : float
        Exponential decay factor (must be in (0,1])
    random_state : int
        Random seed (must be non-negative integer)
    ridge_alpha : float
        Ridge regularization (must be non-negative)

    Raises
    ------
    ValueError
        If any parameter is invalid
    """
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError(f"[ERROR] n_estimators must be positive integer, got {n_estimators}")
    if not (0.0 < weight_decay_factor <= 1.0):
        raise ValueError(f"[ERROR] weight_decay_factor must be in (0,1], got {weight_decay_factor}")
    if not isinstance(random_state, int) or random_state < 0:
        raise ValueError(f"[ERROR] random_state must be non-negative integer, got {random_state}")
    if not isinstance(ridge_alpha, (int, float)) or ridge_alpha < 0:
        raise ValueError(f"[ERROR] ridge_alpha must be non-negative, got {ridge_alpha}")


def _validate_data_quality(
    df: pd.DataFrame,
    features: List[str],
    target_variable: str
) -> None:
    """Validate data quality - no missing values in target or features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for missing values
    features : List[str]
        Feature columns to check
    target_variable : str
        Target column to check

    Raises
    ------
    ValueError
        If target or features contain missing values
    """
    if df[target_variable].isna().sum() > 0:
        raise ValueError(f"[ERROR] Target variable '{target_variable}' contains missing values")

    feature_na_counts = df[features].isna().sum()
    if feature_na_counts.sum() > 0:
        na_features = feature_na_counts[feature_na_counts > 0]
        raise ValueError(f"[ERROR] Features contain missing values: {na_features.to_dict()}")


# =============================================================================
# Main Validation Functions
# =============================================================================

def validate_center_baseline_inputs(
    df: pd.DataFrame,
    df_rates: pd.DataFrame,
    features: List[str],
    target_variable: str,
    current_date_of_mature_data: str,
    n_estimators: int,
    weight_decay_factor: float,
    random_state: int,
    ridge_alpha: float
) -> None:
    """Validate inputs for RILA center_baseline function with business context.

    Performs comprehensive validation of all parameters ensuring data integrity
    for RILA price elasticity analysis.

    Parameters
    ----------
    df : pd.DataFrame
        RILA sales time series with features and target variables
    df_rates : pd.DataFrame
        Competitive rate data from WINK
    features : List[str]
        Feature names for model training
    target_variable : str
        Target variable name
    current_date_of_mature_data : str
        Cutoff date for training data (ISO format YYYY-MM-DD)
    n_estimators : int
        Number of bootstrap estimators
    weight_decay_factor : float
        Exponential decay factor for sample weighting
    random_state : int
        Random seed for reproducibility
    ridge_alpha : float
        Ridge regression regularization parameter

    Raises
    ------
    ValueError
        If any input validation fails
    """
    _validate_dataframes(df, df_rates)
    _validate_required_columns(df, features, target_variable)
    _validate_date_format(current_date_of_mature_data)
    _validate_model_parameters(n_estimators, weight_decay_factor, random_state, ridge_alpha)
    _validate_data_quality(df, features, target_variable)


def _validate_rate_adj_dataframes(
    sales_df: pd.DataFrame,
    rates_df: pd.DataFrame
) -> None:
    """Validate DataFrames for rate adjustments.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales data with feature columns
    rates_df : pd.DataFrame
        Rate data with Prudential and competitor columns

    Raises
    ------
    ValueError
        If DataFrames are None or empty
    """
    if sales_df is None or sales_df.empty:
        raise ValueError("[ERROR] Sales DataFrame cannot be None or empty")
    if rates_df is None or rates_df.empty:
        raise ValueError("[ERROR] Rates DataFrame cannot be None or empty")


def _validate_rate_adj_arrays(
    rate_scenarios: np.ndarray,
    baseline_predictions: np.ndarray
) -> None:
    """Validate arrays for rate adjustments.

    Parameters
    ----------
    rate_scenarios : np.ndarray
        Array of rate adjustment scenarios
    baseline_predictions : np.ndarray
        Baseline forecast predictions

    Raises
    ------
    ValueError
        If arrays are None or empty
    """
    if rate_scenarios is None or len(rate_scenarios) == 0:
        raise ValueError("[ERROR] Rate scenarios array cannot be None or empty")
    if baseline_predictions is None or len(baseline_predictions) == 0:
        raise ValueError("[ERROR] Baseline predictions array cannot be None or empty")


def _validate_rate_adj_model(trained_model: Any) -> None:
    """Validate trained model for rate adjustments.

    Parameters
    ----------
    trained_model : BaggingRegressor
        Trained bootstrap ensemble model

    Raises
    ------
    ValueError
        If model is None or missing estimators
    """
    if trained_model is None:
        raise ValueError("[ERROR] Trained model cannot be None")
    if not hasattr(trained_model, 'estimators_'):
        raise ValueError("[ERROR] Trained model must have estimators_ attribute")


def _validate_rate_adj_features(
    features: List[str],
    sales_df: pd.DataFrame
) -> None:
    """Validate features for rate adjustments.

    Parameters
    ----------
    features : List[str]
        Feature column names
    sales_df : pd.DataFrame
        Sales data to check features against

    Raises
    ------
    ValueError
        If features missing or not in DataFrame
    """
    if features is None or len(features) == 0:
        raise ValueError("[ERROR] Features list cannot be None or empty")

    missing_features = [f for f in features if f not in sales_df.columns]
    if missing_features:
        raise ValueError(f"[ERROR] Missing features in sales DataFrame: {missing_features}")


def _validate_rate_adj_parameters(
    competitor_rate_adjustment: float,
    sales_multiplier: float,
    momentum_lookback_periods: int
) -> None:
    """Validate parameters for rate adjustments.

    Parameters
    ----------
    competitor_rate_adjustment : float
        Adjustment to competitor rates
    sales_multiplier : float
        Sales multiplier for business scaling
    momentum_lookback_periods : int
        Periods for momentum calculations

    Raises
    ------
    ValueError
        If parameters are invalid types or values
    """
    if not isinstance(competitor_rate_adjustment, (int, float)):
        raise ValueError(
            f"[ERROR] competitor_rate_adjustment must be numeric, got {type(competitor_rate_adjustment)}"
        )
    if not isinstance(sales_multiplier, (int, float)) or sales_multiplier <= 0:
        raise ValueError(f"[ERROR] sales_multiplier must be positive, got {sales_multiplier}")
    if not isinstance(momentum_lookback_periods, int) or momentum_lookback_periods <= 0:
        raise ValueError(
            f"[ERROR] momentum_lookback_periods must be positive integer, got {momentum_lookback_periods}"
        )


def validate_rate_adjustments_inputs(
    sales_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    rate_scenarios: np.ndarray,
    baseline_predictions: np.ndarray,
    trained_model: Any,  # BaggingRegressor
    features: List[str],
    competitor_rate_adjustment: float,
    sales_multiplier: float,
    momentum_lookback_periods: int
) -> None:
    """Validate inputs for RILA rate_adjustments function.

    Orchestrates atomic validation functions for comprehensive input checking.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Sales data with feature columns
    rates_df : pd.DataFrame
        Rate data with Prudential and competitor columns
    rate_scenarios : np.ndarray
        Array of rate adjustment scenarios
    baseline_predictions : np.ndarray
        Baseline forecast predictions
    trained_model : BaggingRegressor
        Trained bootstrap ensemble model
    features : List[str]
        Feature column names
    competitor_rate_adjustment : float
        Adjustment to competitor rates
    sales_multiplier : float
        Sales multiplier for business scaling
    momentum_lookback_periods : int
        Periods for momentum calculations

    Raises
    ------
    ValueError
        If any validation fails
    """
    _validate_rate_adj_dataframes(sales_df, rates_df)
    _validate_rate_adj_arrays(rate_scenarios, baseline_predictions)
    _validate_rate_adj_model(trained_model)
    _validate_rate_adj_features(features, sales_df)
    _validate_rate_adj_parameters(
        competitor_rate_adjustment, sales_multiplier, momentum_lookback_periods
    )


def validate_confidence_interval_inputs(
    bootstrap_results: pd.DataFrame,
    rate_scenarios: np.ndarray,
    confidence_level: float,
    rounding_precision: int,
    basis_points_multiplier: int
) -> None:
    """Validate inputs for confidence_interval function.

    Parameters
    ----------
    bootstrap_results : pd.DataFrame
        Bootstrap simulation results
    rate_scenarios : np.ndarray
        Rate scenarios tested
    confidence_level : float
        Confidence level (0.95 for 95%)
    rounding_precision : int
        Decimal places for rounding
    basis_points_multiplier : int
        Multiplier for basis points conversion

    Raises
    ------
    ValueError
        If validation fails
    """
    if bootstrap_results.empty:
        raise ValueError("[ERROR] Bootstrap results DataFrame cannot be empty")

    if len(rate_scenarios) == 0:
        raise ValueError("[ERROR] Rate scenarios array cannot be empty")

    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"[ERROR] Confidence level must be in (0,1), got {confidence_level}")

    if rounding_precision < 0:
        raise ValueError(f"[ERROR] Rounding precision must be non-negative, got {rounding_precision}")

    if basis_points_multiplier <= 0:
        raise ValueError(f"[ERROR] Basis points multiplier must be positive, got {basis_points_multiplier}")

    # Check dimensions match
    if len(rate_scenarios) != bootstrap_results.shape[1]:
        raise ValueError(
            f"[ERROR] Rate scenarios ({len(rate_scenarios)}) and bootstrap columns "
            f"({bootstrap_results.shape[1]}) must match"
        )


def validate_melt_dataframe_inputs(
    df_ci: pd.DataFrame,
    current_date_of_mature_data: str,
    df: pd.DataFrame,
    features: List[str],
    scenarios_per_basis_point: int,
    scenarios_per_percent: float,
    baseline_rate: float
) -> None:
    """Validate inputs for melt_dataframe_for_tableau function.

    Parameters
    ----------
    df_ci : pd.DataFrame
        Confidence interval DataFrame
    current_date_of_mature_data : str
        Date of mature data
    df : pd.DataFrame
        Source sales DataFrame
    features : List[str]
        Model features
    scenarios_per_basis_point : int
        Number of scenarios per basis point
    scenarios_per_percent : float
        Number of scenarios per percent
    baseline_rate : float
        Baseline rate for calculations

    Raises
    ------
    ValueError
        If validation fails
    """
    if df_ci is None or df_ci.empty:
        raise ValueError("[ERROR] Confidence interval DataFrame cannot be None or empty")

    required_cols = ['rate_change_in_basis_points', 'bottom', 'median', 'top']
    missing_cols = [col for col in required_cols if col not in df_ci.columns]
    if missing_cols:
        raise ValueError(f"[ERROR] Missing required columns in CI DataFrame: {missing_cols}")

    _validate_date_format(current_date_of_mature_data)

    if df is None or df.empty:
        raise ValueError("[ERROR] Source sales DataFrame cannot be None or empty")

    if features is None or len(features) == 0:
        raise ValueError("[ERROR] Features list cannot be None or empty")

    if not isinstance(scenarios_per_basis_point, int) or scenarios_per_basis_point <= 0:
        raise ValueError(
            f"[ERROR] scenarios_per_basis_point must be positive integer, got {scenarios_per_basis_point}"
        )

    if not isinstance(scenarios_per_percent, (int, float)) or scenarios_per_percent <= 0:
        raise ValueError(f"[ERROR] scenarios_per_percent must be positive, got {scenarios_per_percent}")

    if not isinstance(baseline_rate, (int, float)):
        raise ValueError(f"[ERROR] baseline_rate must be numeric, got {type(baseline_rate)}")
