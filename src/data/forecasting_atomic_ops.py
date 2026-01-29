"""
Atomic Data Operations for Time Series Forecasting - Vectorization Ready.

This module provides atomic data operations with single responsibility and perfect
separation of concerns, optimized for future vectorization across both time and
bootstrap dimensions.

Key Design Principles:
- Atomic operations (10-50 lines each)
- Single mathematical responsibility per function
- Vectorization-ready interfaces (operate on arrays where possible)
- Zero business logic mixing with mathematical operations
- Perfect DRY compliance with configuration-driven parameters

Future Vectorization Target:
Current: for cutoff in range(30, 156): for bootstrap in range(100): ...
Future:  vectorized_operation(cutoffs=[30...156], bootstrap_samples=100)

Mathematical Precision:
All operations designed to preserve exact numerical results within 1e-6 tolerance
for complete computational lineage validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings

# Suppress specific pandas/numpy warnings that don't indicate code issues
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='pandas'
)
# Justification: Pandas FutureWarnings for API changes in future versions
# These don't affect current functionality and will be addressed during pandas upgrades

# Triple-fallback imports for resilience
from src.config.forecasting_config import ForecastingConfig


def _validate_cutoff_bounds(cutoff: int, df_length: int) -> None:
    """Validate cutoff is within valid bounds.

    Raises:
        ValueError: If cutoff is invalid
    """
    if cutoff <= 0:
        raise ValueError(f"FlexGuard forecasting: cutoff must be positive, got {cutoff}")
    if cutoff > df_length:
        raise ValueError(f"FlexGuard forecasting: cutoff {cutoff} exceeds dataset length {df_length}")


def _validate_feature_columns_exist(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """Validate all feature columns exist in DataFrame.

    Raises:
        ValueError: If any feature columns are missing
    """
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"FlexGuard forecasting: missing features {missing}")


def extract_features_at_cutoff(df: pd.DataFrame, cutoff: int,
                              feature_columns: List[str]) -> np.ndarray:
    """
    Extract feature matrix for single cutoff point - atomic operation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with temporal structure
    cutoff : int
        Single cutoff index for extraction (0-based)
    feature_columns : List[str]
        Column names for features to extract

    Returns
    -------
    np.ndarray
        Feature matrix for cutoff point, shape (cutoff, n_features)

    Raises
    ------
    ValueError
        If cutoff exceeds dataset bounds or features missing
    """
    # Validate inputs
    _validate_cutoff_bounds(cutoff, len(df))
    _validate_feature_columns_exist(df, feature_columns)

    # Extract training features (strict cutoff - no data leakage)
    feature_matrix = df.iloc[:cutoff][feature_columns].values

    # Validate extraction success
    if feature_matrix.shape != (cutoff, len(feature_columns)):
        raise ValueError(
            f"Feature extraction failed: expected ({cutoff}, {len(feature_columns)}), "
            f"got {feature_matrix.shape}"
        )

    return feature_matrix


def _validate_target_column_exists(df: pd.DataFrame, target_column: str) -> None:
    """Validate target column exists in DataFrame.

    Raises:
        ValueError: If target column is missing
    """
    if target_column not in df.columns:
        raise ValueError(f"FlexGuard forecasting: target column '{target_column}' not found")


def extract_target_at_cutoff(df: pd.DataFrame, cutoff: int,
                           target_column: str) -> np.ndarray:
    """
    Extract target values for single cutoff point - atomic operation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with temporal structure
    cutoff : int
        Single cutoff index for extraction (0-based)
    target_column : str
        Target variable column name

    Returns
    -------
    np.ndarray
        Target values for training, shape (cutoff,)
    """
    # Validate inputs
    _validate_cutoff_bounds(cutoff, len(df))
    _validate_target_column_exists(df, target_column)

    # Extract training targets (strict cutoff enforcement)
    target_values = df.iloc[:cutoff][target_column].values

    # Validate extraction
    if len(target_values) != cutoff:
        raise ValueError(f"Target extraction failed: expected {cutoff} values, got {len(target_values)}")

    return target_values


def extract_test_features_at_cutoff(df: pd.DataFrame, cutoff: int,
                                   feature_columns: List[str]) -> np.ndarray:
    """
    Extract test features for prediction at single cutoff - atomic operation.

    Atomic Responsibility: Test feature extraction for prediction point.
    Vectorization Ready: Single cutoff, easily extended to batch cutoffs.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with temporal structure
    cutoff : int
        Cutoff index (test point is at cutoff index)
    feature_columns : List[str]
        Feature columns for prediction

    Returns
    -------
    np.ndarray
        Test features for prediction, shape (1, n_features)

    Mathematical Properties
    ----------------------
    - Extracts exactly one test observation
    - Maintains feature consistency with training
    - No future data leakage
    """

    # Validate inputs
    if cutoff >= len(df):
        raise ValueError(
            f"FlexGuard forecasting: test cutoff {cutoff} exceeds dataset bounds {len(df)}"
        )

    # Extract test point features
    test_features = df.iloc[cutoff][feature_columns].values.reshape(1, -1)

    # Validate test extraction
    if test_features.shape != (1, len(feature_columns)):
        raise ValueError(
            f"Test extraction failed: expected (1, {len(feature_columns)}), "
            f"got {test_features.shape}"
        )

    return test_features


def extract_test_target_at_cutoff(df: pd.DataFrame, cutoff: int,
                                 target_column: str) -> float:
    """Extract true target value at cutoff using sales_by_contract_date for validation."""

    # Validate inputs
    if cutoff >= len(df):
        raise ValueError(
            f"FlexGuard forecasting: test cutoff {cutoff} exceeds dataset bounds"
        )

    # CRITICAL: Use sales_by_contract_date for validation (matching original)
    validation_target_column = 'sales_by_contract_date'
    if validation_target_column not in df.columns:
        raise ValueError(
            f"FlexGuard forecasting: validation target column '{validation_target_column}' not found"
        )

    # Extract true target value using contract date column
    true_target = df.iloc[cutoff][validation_target_column]

    # Validate extraction (must be numeric)
    if not np.isfinite(true_target):
        raise ValueError(
            f"Invalid target value at cutoff {cutoff}: {true_target}"
        )

    return float(true_target)


def extract_test_target_contract_date_atomic(df: pd.DataFrame, cutoff: int) -> float:
    """
    Extract contract-date test target for benchmark validation - atomic operation.

    Atomic Responsibility: Contract-date target extraction for benchmark validation.
    Vectorization Ready: Single cutoff operation, designed for batching.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with temporal structure
    cutoff : int
        Cutoff index for test data (0-based)

    Returns
    -------
    float
        Contract-date target value for benchmark validation

    Mathematical Properties
    ----------------------
    Business Logic: Uses sales_by_contract_date for benchmark validation
    Temporal Alignment: Matches original notebook validation approach
    Zero Data Leakage: Uses only data at cutoff point
    """

    # Validate inputs
    if cutoff >= len(df):
        raise ValueError(
            f"Benchmark validation: cutoff {cutoff} exceeds dataset bounds"
        )

    # Use contract date target for benchmark validation (matching original)
    contract_date_column = 'sales_by_contract_date'
    if contract_date_column not in df.columns:
        raise ValueError(
            f"Benchmark validation: '{contract_date_column}' column not found in dataset"
        )

    # Extract contract-date target value
    contract_target = df.iloc[cutoff][contract_date_column]

    # Validate extraction
    if not np.isfinite(contract_target):
        raise ValueError(
            f"Invalid contract-date target at cutoff {cutoff}: {contract_target}"
        )

    return float(contract_target)


def apply_business_filters_atomic(df: pd.DataFrame,
                                 config: Dict[str, Any]) -> pd.DataFrame:
    """Apply business logic filters atomically (zero sales, temporal, incomplete obs)."""

    # Start with copy to avoid mutations
    filtered_df = df.copy()

    # Business Rule 1: Remove zero sales periods
    if 'sales' in filtered_df.columns:
        initial_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['sales'] != 0].copy()
        removed_count = initial_count - len(filtered_df)

        if removed_count > 0:
            print(f"Business filter: removed {removed_count} zero sales observations")

    # Business Rule 2: Temporal filtering for mature data
    if 'analysis_start_date' in config and 'date' in filtered_df.columns:
        start_date = pd.to_datetime(config['analysis_start_date'])
        filtered_df = filtered_df[filtered_df['date'] > start_date].copy()

    # Business Rule 3: Remove incomplete observations (last row)
    if config.get('remove_incomplete_final_obs', True):
        if len(filtered_df) > 0:
            filtered_df = filtered_df.iloc[:-1].copy()

    # Validate filtering preserved data integrity
    if len(filtered_df) == 0:
        raise ValueError("Business filtering removed all data - check filter criteria")

    return filtered_df


def apply_sign_corrections_atomic(features: np.ndarray,
                                 correction_mask: np.ndarray) -> np.ndarray:
    """
    Apply economic theory sign corrections atomically - pure math operation.

    Atomic Responsibility: Mathematical sign negation only.
    Vectorization Ready: Pure array operation, fully vectorizable.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix to apply corrections to
    correction_mask : np.ndarray
        Boolean mask indicating which columns to negate

    Returns
    -------
    np.ndarray
        Features with sign corrections applied

    Mathematical Properties
    ----------------------
    Economic Theory: Higher competitor rates → Higher FlexGuard sales
    Implementation: Negate competitive features to align with theory
    Precision: Exact negation preserves all numerical properties
    """

    # Input validation
    if features.shape[1] != len(correction_mask):
        raise ValueError(
            f"Feature count {features.shape[1]} doesn't match "
            f"correction mask length {len(correction_mask)}"
        )

    # Create corrected copy (avoid mutation)
    corrected_features = features.copy()

    # Apply sign corrections where mask is True
    corrected_features[:, correction_mask] = -corrected_features[:, correction_mask]

    # Validate correction preserved array structure
    if corrected_features.shape != features.shape:
        raise ValueError(
            f"Sign correction changed array shape: {features.shape} -> {corrected_features.shape}"
        )

    return corrected_features


def calculate_temporal_weights_atomic(n_observations: int,
                                    decay_rate: float = 0.98) -> np.ndarray:
    """
    Calculate exponential temporal weights atomically - pure mathematical function.

    Atomic Responsibility: Mathematical weight calculation only.
    Vectorization Ready: Pure computation, easily vectorizable.

    Parameters
    ----------
    n_observations : int
        Number of observations for weight calculation
    decay_rate : float, default=0.98
        Exponential decay rate (0.98 = recent observations weighted more)

    Returns
    -------
    np.ndarray
        Temporal weights, shape (n_observations,)

    Mathematical Properties
    ----------------------
    Formula: weight[k] = decay_rate^(n_observations - k - 1)
    Effect: More recent observations have higher weights
    Range: (0, 1] with max weight = 1.0 for most recent
    """

    # Validate inputs
    if n_observations <= 0:
        raise ValueError(f"Number of observations must be positive, got {n_observations}")

    if not (0 < decay_rate <= 1):
        raise ValueError(f"Decay rate must be in (0, 1], got {decay_rate}")

    # Calculate exponential weights (vectorized)
    k_values = np.arange(n_observations)
    weights = decay_rate ** (n_observations - k_values - 1)

    # Validate weight calculation
    if len(weights) != n_observations:
        raise ValueError(
            f"Weight calculation failed: expected {n_observations} weights, got {len(weights)}"
        )

    if not np.all(weights > 0):
        raise ValueError("All weights must be positive")

    if not np.all(weights <= 1.0):
        raise ValueError("All weights must be <= 1.0")

    return weights


def validate_cutoff_data_atomic(X: np.ndarray, y: np.ndarray, cutoff: int) -> bool:
    """Validate data consistency at cutoff - atomic validation operation."""

    try:
        # Check feature-target alignment
        if X.shape[0] != len(y):
            print(f"Cutoff {cutoff}: Feature-target misalignment {X.shape[0]} vs {len(y)}")
            return False

        # Check for invalid values
        if np.any(~np.isfinite(X)):
            print(f"Cutoff {cutoff}: Invalid values in features")
            return False

        if np.any(~np.isfinite(y)):
            print(f"Cutoff {cutoff}: Invalid values in targets")
            return False

        # Check sufficient observations
        if X.shape[0] < 10:  # Minimum for meaningful modeling
            print(f"Cutoff {cutoff}: Insufficient observations {X.shape[0]}")
            return False

        # Check feature dimensionality
        if X.shape[1] == 0:
            print(f"Cutoff {cutoff}: No features available")
            return False

        return True

    except Exception as e:
        print(f"Cutoff {cutoff}: Validation error - {e}")
        return False


def extract_weights_at_cutoff(df: pd.DataFrame, cutoff: int) -> np.ndarray:
    """
    Extract precomputed weights for single cutoff point - atomic operation.

    Atomic Responsibility: Weight extraction only, no calculations.
    Vectorization Ready: Single cutoff operation, easily batchable.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with precomputed 'weight' column
    cutoff : int
        Single cutoff index for extraction (0-based)

    Returns
    -------
    np.ndarray
        Weight values for training, shape (cutoff,)

    Mathematical Properties
    ----------------------
    - Extracts exact precomputed weights
    - Maintains temporal alignment with features/targets
    - Zero data leakage enforcement
    """
    # Validate inputs
    if cutoff <= 0:
        raise ValueError(f"FlexGuard forecasting: cutoff must be positive, got {cutoff}")

    if cutoff > len(df):
        raise ValueError(
            f"FlexGuard forecasting: cutoff {cutoff} exceeds dataset length {len(df)}"
        )

    if 'weight' not in df.columns:
        # Fall back to calculating weights if not precomputed
        return calculate_temporal_weights_atomic(cutoff, 0.98)

    # Extract training weights (strict cutoff enforcement)
    training_data = df.iloc[:cutoff]
    weight_values = training_data['weight'].values

    # Validate extraction
    if len(weight_values) != cutoff:
        raise ValueError(
            f"Weight extraction failed: expected {cutoff} values, got {len(weight_values)}"
        )

    return weight_values


def apply_original_training_filters_atomic(df: pd.DataFrame, cutoff: int,
                                         target_column: str,
                                         config: Dict[str, Any]) -> pd.DataFrame:
    """Apply training filters: cutoff, dropna, holiday exclusion, mature data cutoff."""
    from datetime import datetime, timedelta

    # Apply cutoff and remove missing target values (matching original)
    df_filtered = df[:cutoff].dropna(subset=[target_column]).copy()

    # Apply holiday filter if configured (matching original)
    exclude_holidays = config.get('exclude_holidays', False)
    if exclude_holidays and 'holiday' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['holiday'] == 0]

    # Apply mature data cutoff if configured (matching original)
    mature_data_cutoff_days = config.get('mature_data_cutoff_days', 0)
    if mature_data_cutoff_days > 0:
        cutoff_date = datetime.now() - timedelta(days=mature_data_cutoff_days)
        cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')

        if 'date' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['date'] < cutoff_date_str]

    # Validate filtering didn't remove all data
    if len(df_filtered) == 0:
        raise ValueError("No training data remaining after filters")

    return df_filtered


def _extract_training_data(
    df_filtered: pd.DataFrame, feature_columns: List[str], target_column: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract training features, targets, and weights from filtered DataFrame.

    Returns:
        Tuple of (X_train, y_train, weights)
    """
    X_train = df_filtered[feature_columns].values
    y_train = df_filtered[target_column].values
    weights = df_filtered['weight'].values if 'weight' in df_filtered.columns else np.ones(len(df_filtered))
    return X_train, y_train, weights


def _apply_sign_corrections_if_configured(
    X_train: np.ndarray, X_test: np.ndarray, config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sign corrections if configured.

    Returns:
        Tuple of (corrected_X_train, corrected_X_test)
    """
    if 'sign_correction_mask' in config:
        correction_mask = config['sign_correction_mask']
        X_train = apply_sign_corrections_atomic(X_train, correction_mask)
        X_test = apply_sign_corrections_atomic(X_test, correction_mask)
    return X_train, X_test


def prepare_cutoff_data_complete(df: pd.DataFrame, cutoff: int,
                               feature_columns: List[str], target_column: str,
                               config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Complete data preparation for single cutoff - orchestration of atomic operations.

    CRITICAL: Matches original prepare_training_cutoff_data filtering exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    cutoff : int
        Cutoff point for data preparation
    feature_columns : List[str]
        Feature column names
    target_column : str
        Target column name
    config : Dict[str, Any]
        Configuration for data preparation

    Returns
    -------
    Dict[str, np.ndarray]
        Complete prepared data for cutoff with keys:
        X_train, y_train, X_test, y_test, weights, cutoff
    """
    # Step 1: Apply original training data filters
    df_train_filtered = apply_original_training_filters_atomic(df, cutoff, target_column, config)

    # Step 2: Extract training data
    X_train, y_train, weights = _extract_training_data(df_train_filtered, feature_columns, target_column)

    # Step 3: Extract test data from unfiltered dataset
    X_test = extract_test_features_at_cutoff(df, cutoff, feature_columns)
    y_test = extract_test_target_at_cutoff(df, cutoff, target_column)

    # Step 4: Apply sign corrections if configured
    X_train, X_test = _apply_sign_corrections_if_configured(X_train, X_test, config)

    # Step 5: Validate prepared data
    if not validate_cutoff_data_atomic(X_train, y_train, cutoff):
        raise ValueError(f"Data validation failed at cutoff {cutoff}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'weights': weights, 'cutoff': cutoff
    }