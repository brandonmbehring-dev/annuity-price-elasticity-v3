"""
Inference Training Module for RILA Price Elasticity.

This module contains training pipeline and data preparation functions
for bootstrap ensemble model training. Extracted from inference.py.

Functions:
- prepare_training_data: Data preprocessing with exponential decay weighting
- train_bootstrap_model: Bootstrap Ridge ensemble training
- transform_prediction_features: Feature engineering for predictions

Design Principles:
- Isolated training logic, reusable for retraining workflows
- Configuration-driven using src.config.product_config
- Mathematical equivalence maintained with 1e-12 precision

Module Architecture (Phase 6.3d Split):
- inference_training.py: Training pipeline (this file)
- inference_scenarios.py: Baseline + scenario predictions
- inference.py: Thin wrapper + Tableau export + public API
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge

from src.config.product_config import get_default_product


# =============================================================================
# UTILITIES
# =============================================================================


def _get_product_name() -> str:
    """Get current product name for error messages.

    Returns default product name from ProductConfig. This centralizes
    the product name reference for all error messages.

    Returns
    -------
    str
        Product name (e.g., "FlexGuard 6Y20B")
    """
    return get_default_product().name


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrainingData:
    """
    Training data container for RILA price elasticity modeling.

    Encapsulates the processed training dataset with features, targets,
    and sample weights for bootstrap ensemble training.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix with competitive rates, economic indicators, and sales lags
    y : pd.Series
        Target variable (weekly sales forward_0)
    w : pd.Series
        Sample weights with exponential decay (0.99^(n-k))
    df_train : pd.DataFrame
        Complete filtered training dataset for reference
    """
    X: pd.DataFrame
    y: pd.Series
    w: pd.Series
    df_train: pd.DataFrame


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_training_data(
    df: pd.DataFrame,
    current_date_of_mature_data: str,
    target_variable: str,
    features: List[str],
    weight_decay_factor: float
) -> TrainingData:
    """Prepare training data with exponential decay weighting and business filters."""
    df_work = df.copy()

    # Apply exponential decay weighting on FULL dataset
    df_work["weight"] = np.array([
        weight_decay_factor ** (len(df_work) - k) for k in range(len(df_work))
    ])

    # Apply business logic filters: date cutoff and holiday exclusion
    cutoff_date = pd.to_datetime(current_date_of_mature_data)
    df_train = df_work[df_work["date"] < cutoff_date]
    df_train = df_train[df_train["holiday"] == 0]

    if df_train.empty:
        raise ValueError(f"No training data before cutoff {current_date_of_mature_data}")

    return TrainingData(
        X=df_train[features],
        y=df_train[target_variable],
        w=df_train["weight"],
        df_train=df_train
    )


# =============================================================================
# MODEL TRAINING
# =============================================================================


def _create_bagging_regressor(
    base_estimator: Ridge, n_estimators: int, random_state: int
) -> BaggingRegressor:
    """Create BaggingRegressor with sklearn API compatibility."""
    try:
        return BaggingRegressor(
            estimator=base_estimator, n_estimators=n_estimators,
            random_state=random_state, bootstrap=True, bootstrap_features=False
        )
    except TypeError:
        return BaggingRegressor(
            base_estimator=base_estimator, n_estimators=n_estimators,
            random_state=random_state, bootstrap=True, bootstrap_features=False
        )


def train_bootstrap_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: pd.Series,
    n_estimators: int,
    random_state: int,
    ridge_alpha: float
) -> BaggingRegressor:
    """Train bootstrap Ridge ensemble model for RILA sales forecasting."""
    base_estimator = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model = _create_bagging_regressor(base_estimator, n_estimators, random_state)
    model.fit(X, np.log(1 + y), sample_weight=sample_weights)
    return model


# =============================================================================
# FEATURE TRANSFORMATION
# =============================================================================


def transform_prediction_features(
    X_test_base: pd.Series,
    features: List[str],
    df_rates: pd.DataFrame,
    df_sales: pd.DataFrame,
    own_rate_column: str = "Prudential",
) -> pd.Series:
    """
    Transform base features for current market condition predictions.

    Matches original helper business logic exactly for mathematical equivalence.
    Uses configurable column mappings and sales momentum calculations.

    Parameters
    ----------
    X_test_base : pd.Series
        Base feature values from most recent sales observation
    features : List[str]
        Feature names for prediction
    df_rates : pd.DataFrame
        Current competitive rate data from WINK
    df_sales : pd.DataFrame
        Sales time series for competitor and momentum calculations
    own_rate_column : str
        Column name for own company rate in df_rates (default: "Prudential").
        Use config.own_rate_column for multi-product support.

    Returns
    -------
    pd.Series
        Transformed feature vector for prediction
    """
    X_test = X_test_base.copy()

    # Apply feature transformations matching original helper business logic
    for feature_name in features:

        # Own company rate features: use config-driven column name
        if "prudential_rate" in feature_name:
            X_test[feature_name] = df_rates[own_rate_column].iloc[-1]

        # Sales momentum features: 3-period averaging (matching original)
        elif "sales_by_contract_date" in feature_name:
            # Note: competitor features are commented out in original - they keep historical values
            momentum_avg = (
                df_sales["sales_by_contract_date_lag_1"].iloc[-1]
                + df_sales["sales_by_contract_date_lag_2"].iloc[-1]
                + df_sales["sales_by_contract_date_lag_3"].iloc[-1]
            ) / 3
            X_test[feature_name] = momentum_avg

    return X_test


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'TrainingData',
    'prepare_training_data',
    'train_bootstrap_model',
    'transform_prediction_features',
    '_get_product_name',
]
