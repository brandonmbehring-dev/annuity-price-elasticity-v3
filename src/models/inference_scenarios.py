"""
Inference Scenarios Module for RILA Price Elasticity.

This module contains baseline forecasting, scenario analysis, and confidence
interval computation functions. Extracted from inference.py.

Functions:
- center_baseline: Main entry point for baseline forecast
- rate_adjustments: Main entry point for rate sensitivity analysis
- confidence_interval: Main entry point for CI computation

Design Principles:
- Core ML inference logic separated from training and export
- Public API functions for key business operations
- Configuration-driven using src.config.product_config

Module Architecture (Phase 6.3d Split):
- inference_training.py: Training pipeline
- inference_scenarios.py: Baseline + scenario predictions (this file)
- inference.py: Thin wrapper + Tableau export + public API
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

from src.config.product_config import get_default_product
from src.models.inference_training import (
    TrainingData,
    prepare_training_data,
    train_bootstrap_model,
    transform_prediction_features,
    _get_product_name,
)
from src.models.inference_validation import (
    validate_center_baseline_inputs as _validate_center_baseline_inputs_canonical,
    validate_rate_adjustments_inputs as _validate_rate_adjustments_inputs_canonical,
    validate_confidence_interval_inputs as _validate_confidence_interval_inputs_canonical,
)


# =============================================================================
# VALIDATION WRAPPERS
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
    """
    Validate inputs for RILA center_baseline function with business context.

    Delegates to canonical implementation in inference_validation.py.
    """
    _validate_center_baseline_inputs_canonical(
        df, df_rates, features, target_variable, current_date_of_mature_data,
        n_estimators, weight_decay_factor, random_state, ridge_alpha
    )


def validate_rate_adjustments_inputs(
    sales_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    rate_scenarios: np.ndarray,
    baseline_predictions: np.ndarray,
    trained_model: BaggingRegressor,
    features: List[str],
    competitor_rate_adjustment: float,
    sales_multiplier: float,
    momentum_lookback_periods: int
) -> None:
    """
    Validate inputs for RILA rate_adjustments function.

    Delegates to canonical implementation in inference_validation.py.
    """
    _validate_rate_adjustments_inputs_canonical(
        sales_df, rates_df, rate_scenarios, baseline_predictions,
        trained_model, features, competitor_rate_adjustment,
        sales_multiplier, momentum_lookback_periods
    )


def validate_confidence_interval_inputs(
    bootstrap_results: pd.DataFrame,
    rate_scenarios: np.ndarray,
    confidence_level: float,
    rounding_precision: int,
    basis_points_multiplier: int
) -> None:
    """
    Validate inputs for confidence_interval function.

    Delegates to canonical implementation in inference_validation.py.
    """
    _validate_confidence_interval_inputs_canonical(
        bootstrap_results, rate_scenarios, confidence_level,
        rounding_precision, basis_points_multiplier
    )


# =============================================================================
# BASELINE FORECASTING HELPERS
# =============================================================================


def _resolve_center_baseline_params(
    sales_df: pd.DataFrame, rates_df: pd.DataFrame,
    df: pd.DataFrame, df_rates: pd.DataFrame,
    training_cutoff_date: str, current_date_of_mature_data: str,
    features: List[str], target_variable: str, n_estimators: int
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Resolve parameter compatibility and validate required params."""
    actual_df = sales_df if sales_df is not None else df
    actual_df_rates = rates_df if rates_df is not None else df_rates
    actual_cutoff_date = training_cutoff_date if training_cutoff_date is not None else current_date_of_mature_data

    if actual_df is None:
        raise ValueError("Either 'sales_df' or 'df' parameter must be provided")
    if actual_df_rates is None:
        raise ValueError("Either 'rates_df' or 'df_rates' parameter must be provided")
    if actual_cutoff_date is None:
        raise ValueError("Either 'training_cutoff_date' or 'current_date_of_mature_data' must be provided")
    if features is None or target_variable is None or n_estimators is None:
        raise ValueError("Required parameters 'features', 'target_variable', 'n_estimators' must be provided")

    return actual_df, actual_df_rates, actual_cutoff_date


def _generate_baseline_predictions(
    trained_model: BaggingRegressor,
    actual_df: pd.DataFrame,
    actual_df_rates: pd.DataFrame,
    features: List[str],
    n_estimators: int
) -> np.ndarray:
    """Generate baseline predictions from bootstrap ensemble."""
    baseline_prediction = np.zeros(n_estimators)
    for index, estimator in enumerate(trained_model.estimators_):
        X_test_base = actual_df[features].iloc[-1]
        X_test_transformed = transform_prediction_features(
            X_test_base, features, actual_df_rates, actual_df
        )
        baseline_prediction[index] = estimator.predict(
            X_test_transformed.values.reshape(1, -1)
        )[0]
    # Invert log(1+y) transformation: exp(pred) - 1
    return np.exp(baseline_prediction) - 1


# =============================================================================
# BASELINE FORECASTING
# =============================================================================


def center_baseline(
    sales_df: pd.DataFrame = None,
    rates_df: pd.DataFrame = None,
    df: pd.DataFrame = None,
    df_rates: pd.DataFrame = None,
    features: List[str] = None,
    target_variable: str = None,
    training_cutoff_date: str = None,
    current_date_of_mature_data: str = None,
    n_estimators: int = None,
    weight_decay_factor: float = 0.99,
    random_state: int = 42,
    ridge_alpha: float = 1.0
) -> Tuple[np.ndarray, BaggingRegressor]:
    """Generate baseline forecast for RILA price elasticity using bootstrap ensemble."""
    try:
        actual_df, actual_df_rates, actual_cutoff_date = _resolve_center_baseline_params(
            sales_df, rates_df, df, df_rates, training_cutoff_date,
            current_date_of_mature_data, features, target_variable, n_estimators
        )

        validate_center_baseline_inputs(
            actual_df, actual_df_rates, features, target_variable, actual_cutoff_date,
            n_estimators, weight_decay_factor, random_state, ridge_alpha
        )

        training_data = prepare_training_data(
            actual_df, actual_cutoff_date, target_variable, features, weight_decay_factor
        )

        trained_model = train_bootstrap_model(
            training_data.X, training_data.y, training_data.w,
            n_estimators, random_state, ridge_alpha
        )

        baseline_prediction = _generate_baseline_predictions(
            trained_model, actual_df, actual_df_rates, features, n_estimators
        )

        return baseline_prediction, trained_model

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"RILA baseline forecasting failed for {_get_product_name()} analysis: {e}"
        ) from e


# =============================================================================
# SCENARIO ANALYSIS HELPERS
# =============================================================================


def _calculate_momentum(sales_df: pd.DataFrame, lookback_periods: int) -> float:
    """Calculate sales momentum as average of lagged values."""
    momentum_sum = sum([
        sales_df[f"sales_by_contract_date_lag_{i+1}"].iloc[-1]
        for i in range(lookback_periods)
    ])
    return momentum_sum / lookback_periods


def apply_feature_adjustments(
    base_features: pd.Series,
    features: List[str],
    rates_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    prudential_rate_adjustment: float,
    competitor_rate_adjustment: float,
    momentum_lookback_periods: int
) -> pd.Series:
    """
    Apply rate and momentum adjustments to feature vector for scenario analysis.

    IMPORTANT: Only adjusts CURRENT period features (suffix '_current'), not lagged features.
    This preserves temporal structure - lagged features (e.g., prudential_rate_t3) remain
    at their historical values while current features are adjusted for scenario analysis.
    """
    try:
        adjusted_features = base_features.copy()

        for feature_name in features:
            # Only adjust current period prudential rates, not lagged features (e.g., _t3)
            if "prudential_rate" in feature_name and feature_name.endswith("_current"):
                adjusted_features.loc[feature_name] = prudential_rate_adjustment
            # Only adjust current period competitor rates
            elif "competitor_mid" in feature_name and feature_name.endswith("_current"):
                adjusted_features.loc[feature_name] = (
                    sales_df["competitor_mid_current"].iloc[-1] + competitor_rate_adjustment
                )
            elif "competitor_top5" in feature_name and feature_name.endswith("_current"):
                adjusted_features.loc[feature_name] = (
                    sales_df["competitor_top5_current"].iloc[-1] + competitor_rate_adjustment
                )
            elif "sales_by_contract_date" in feature_name:
                adjusted_features.loc[feature_name] = _calculate_momentum(
                    sales_df, momentum_lookback_periods
                )

        return adjusted_features

    except Exception as e:
        raise ValueError(f"Feature adjustment failed: {e}") from e


def _generate_scenario_predictions(
    trained_model: BaggingRegressor,
    sales_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    features: List[str],
    prudential_rate_adjustment: float,
    competitor_rate_adjustment: float,
    momentum_lookback_periods: int,
    n_estimators: int
) -> np.ndarray:
    """Generate predictions for a single rate scenario across bootstrap ensemble."""
    scenario_predictions = np.zeros(n_estimators)
    for estimator_idx, estimator in enumerate(trained_model.estimators_):
        base_features = sales_df[features].iloc[-1]
        adjusted_features = apply_feature_adjustments(
            base_features, features, rates_df, sales_df,
            prudential_rate_adjustment, competitor_rate_adjustment,
            momentum_lookback_periods
        )
        # Invert log(1+y) transformation: exp(pred) - 1
        prediction = np.exp(estimator.predict(adjusted_features.values.reshape(1, -1))[0]) - 1
        scenario_predictions[estimator_idx] = prediction
    return scenario_predictions


# =============================================================================
# SCENARIO ANALYSIS
# =============================================================================


def rate_adjustments(
    sales_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    rate_scenarios: np.ndarray,
    baseline_predictions: np.ndarray,
    trained_model: BaggingRegressor,
    features: List[str],
    competitor_rate_adjustment: float,
    sales_multiplier: float = 13.0,
    momentum_lookback_periods: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate rate adjustment scenario analysis for RILA price elasticity."""
    try:
        validate_rate_adjustments_inputs(
            sales_df, rates_df, rate_scenarios, baseline_predictions,
            trained_model, features, competitor_rate_adjustment,
            sales_multiplier, momentum_lookback_periods
        )

        n_estimators = len(baseline_predictions)
        df_dollars = pd.DataFrame(np.zeros((n_estimators, len(rate_scenarios))), columns=rate_scenarios)
        df_pct_change = df_dollars.copy()

        for scenario_idx, prudential_rate_adjustment in enumerate(rate_scenarios):
            scenario_predictions = _generate_scenario_predictions(
                trained_model, sales_df, rates_df, features,
                prudential_rate_adjustment, competitor_rate_adjustment,
                momentum_lookback_periods, n_estimators
            )
            df_dollars.iloc[:, scenario_idx] = sales_multiplier * scenario_predictions
            df_pct_change.iloc[:, scenario_idx] = (scenario_predictions / baseline_predictions - 1) * 100

        return df_dollars, df_pct_change

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Rate adjustments scenario analysis failed for {_get_product_name()}: {e}"
        ) from e


# =============================================================================
# CONFIDENCE INTERVAL HELPERS
# =============================================================================


def _calculate_quantile_bounds(
    confidence_level: float
) -> Tuple[float, float]:
    """
    Calculate lower and upper quantile bounds from confidence level.

    Parameters
    ----------
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)

    Returns
    -------
    Tuple[float, float]
        (lower_quantile, upper_quantile) bounds
    """
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2  # e.g., 0.025 for 95% CI
    upper_quantile = 1 - alpha / 2  # e.g., 0.975 for 95% CI
    return lower_quantile, upper_quantile


def _initialize_ci_dataframe(
    rate_scenarios: np.ndarray
) -> pd.DataFrame:
    """
    Initialize confidence interval DataFrame structure.

    Parameters
    ----------
    rate_scenarios : np.ndarray
        Array of rate adjustment scenarios

    Returns
    -------
    pd.DataFrame
        Initialized DataFrame with rate_change, bottom, median, top columns
    """
    return pd.DataFrame({
        "rate_change": rate_scenarios,
        "bottom": np.zeros(len(rate_scenarios)),
        "median": np.zeros(len(rate_scenarios)),
        "top": np.zeros(len(rate_scenarios))
    })


def _compute_quantiles(
    df_output: pd.DataFrame,
    bootstrap_results: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
    rounding_precision: int
) -> pd.DataFrame:
    """
    Compute quantile values for bootstrap results.

    Parameters
    ----------
    df_output : pd.DataFrame
        DataFrame to populate with quantile values
    bootstrap_results : pd.DataFrame
        Bootstrap prediction results with scenarios as columns
    lower_quantile : float
        Lower quantile bound (e.g., 0.025)
    upper_quantile : float
        Upper quantile bound (e.g., 0.975)
    rounding_precision : int
        Decimal places for rounding

    Returns
    -------
    pd.DataFrame
        DataFrame with computed bottom, median, top values
    """
    df_output["bottom"] = np.round(
        np.quantile(bootstrap_results.values, lower_quantile, axis=0),
        rounding_precision
    )
    df_output["median"] = np.round(
        np.quantile(bootstrap_results.values, 0.50, axis=0),
        rounding_precision
    )
    df_output["top"] = np.round(
        np.quantile(bootstrap_results.values, upper_quantile, axis=0),
        rounding_precision
    )
    return df_output


def _convert_to_basis_points(
    df_output: pd.DataFrame,
    basis_points_multiplier: int
) -> pd.DataFrame:
    """
    Convert rate changes to basis points and finalize DataFrame.

    Parameters
    ----------
    df_output : pd.DataFrame
        DataFrame with rate_change column
    basis_points_multiplier : int
        Multiplier for basis points conversion (typically 100)

    Returns
    -------
    pd.DataFrame
        DataFrame with rate_change_in_basis_points, rate_change column removed
    """
    df_output["rate_change_in_basis_points"] = (
        df_output["rate_change"] * basis_points_multiplier
    ).astype("int")
    return df_output.drop(columns="rate_change")


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================


def confidence_interval(
    bootstrap_results: pd.DataFrame,
    rate_scenarios: np.ndarray,
    confidence_level: float = 0.95,
    rounding_precision: int = 3,
    basis_points_multiplier: int = 100
) -> pd.DataFrame:
    """
    Calculate confidence intervals from bootstrap results for RILA price elasticity.

    Returns DataFrame with columns: rate_change_in_basis_points, bottom, median, top.
    """
    try:
        validate_confidence_interval_inputs(
            bootstrap_results, rate_scenarios, confidence_level,
            rounding_precision, basis_points_multiplier
        )

        lower_quantile, upper_quantile = _calculate_quantile_bounds(confidence_level)
        df_output = _initialize_ci_dataframe(rate_scenarios)
        df_output = _compute_quantiles(
            df_output, bootstrap_results, lower_quantile, upper_quantile, rounding_precision
        )
        df_output = _convert_to_basis_points(df_output, basis_points_multiplier)

        return df_output

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Confidence interval calculation failed for {_get_product_name()} "
            f"with {len(rate_scenarios) if 'rate_scenarios' in locals() else 'unknown'} scenarios: {e}"
        ) from e


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Validation wrappers
    'validate_center_baseline_inputs',
    'validate_rate_adjustments_inputs',
    'validate_confidence_interval_inputs',

    # Main entry points
    'center_baseline',
    'rate_adjustments',
    'confidence_interval',

    # Helpers (for backward compatibility)
    '_resolve_center_baseline_params',
    '_generate_baseline_predictions',
    'apply_feature_adjustments',
    '_generate_scenario_predictions',
    '_calculate_quantile_bounds',
    '_initialize_ci_dataframe',
    '_compute_quantiles',
    '_convert_to_basis_points',
]
