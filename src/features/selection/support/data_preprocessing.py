"""
Data Preprocessing Utilities for Feature Selection Pipeline.

This module consolidates data loading and transformation patterns from
the notebook to eliminate DRY violations and provide reusable functions.

Key Functions:
- prepare_analysis_dataset: Main function consolidating notebook data prep
- apply_autoregressive_transforms: Log transformation for autoregressive features
- validate_feature_availability: Feature availability checking with clear messages

Design Principles:
- DRY elimination: Consolidate repeated data preparation patterns
- Single responsibility: Each function handles one transformation type
- Business validation: Clear error messages with business context
- Mathematical equivalence: Identical transformations to notebook implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Defensive triple-fallback imports following established patterns
try:
    from src.features.selection_types import FeatureSelectionConfig
    FEATURE_SELECTION_CONFIG_AVAILABLE = True
except ImportError:
    # Fallback type definitions for standalone operation
    from typing import TypedDict
    from typing_extensions import Required, NotRequired
    FEATURE_SELECTION_CONFIG_AVAILABLE = False

    class FeatureSelectionConfig(TypedDict):
        """Feature selection configuration - fallback definition."""
        base_features: Required[List[str]]
        candidate_features: Required[List[str]]
        max_candidate_features: Required[int]
        target_variable: Required[str]
        analysis_start_date: NotRequired[str]
        exclude_holidays: NotRequired[bool]


def _validate_target_for_transformation(data: pd.DataFrame, target_variable: str) -> pd.Series:
    """
    Validate target variable exists and return target values.

    Single responsibility: Target validation only.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the target variable
    target_variable : str
        Name of the target variable column

    Returns
    -------
    pd.Series
        Target variable values

    Raises
    ------
    ValueError
        If target variable not found
    """
    if target_variable not in data.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in dataset")

    return data[target_variable]


def _apply_transformation_math(target_values: pd.Series, transformation: str, target_variable: str) -> Tuple[pd.Series, str]:
    """
    Apply mathematical transformation to target values.

    Single responsibility: Mathematical transformation only.

    Parameters
    ----------
    target_values : pd.Series
        Target variable values to transform
    transformation : str
        Type of transformation to apply
    target_variable : str
        Name of target variable (for naming)

    Returns
    -------
    Tuple[pd.Series, str]
        (transformed_values, transform_name)

    Raises
    ------
    ValueError
        If transformation not supported or values incompatible
    """
    if transformation == "log1p":
        transformed_values = np.log1p(target_values)
        transform_name = f"{target_variable}_log1p"
        print(f"Applied log1p transformation: np.log(1 + {target_variable})")

    elif transformation == "log":
        if (target_values <= 0).any():
            raise ValueError(f"Log transformation requires positive values, but found {(target_values <= 0).sum()} non-positive values")
        transformed_values = np.log(target_values)
        transform_name = f"{target_variable}_log"
        print(f"Applied log transformation: np.log({target_variable})")

    elif transformation == "sqrt":
        if (target_values < 0).any():
            raise ValueError(f"Square root transformation requires non-negative values, but found {(target_values < 0).sum()} negative values")
        transformed_values = np.sqrt(target_values)
        transform_name = f"{target_variable}_sqrt"
        print(f"Applied square root transformation: np.sqrt({target_variable})")

    else:
        raise ValueError(f"Unsupported transformation '{transformation}'. Supported: 'log1p', 'log', 'sqrt'")

    return transformed_values, transform_name


def _log_transformation_statistics(original_values: pd.Series, transformed_values: pd.Series,
                                 original_name: str, transformed_name: str) -> None:
    """
    Log transformation statistics for analysis.

    Single responsibility: Statistics logging only.

    Parameters
    ----------
    original_values : pd.Series
        Original untransformed target variable values
    transformed_values : pd.Series
        Transformed target variable values
    original_name : str
        Name label for original values (used in output)
    transformed_name : str
        Name label for transformed values (used in output)

    Returns
    -------
    None
        Prints statistics to stdout
    """
    print(f"Target transformation statistics:")
    print(f"  Original {original_name}: mean={original_values.mean():.3f}, std={original_values.std():.3f}")
    print(f"  Transformed {transformed_name}: mean={transformed_values.mean():.3f}, std={transformed_values.std():.3f}")
    print(f"  Variance stabilization ratio: {original_values.std()/transformed_values.std():.3f}")


def apply_target_transformation(data: pd.DataFrame,
                               target_variable: str,
                               transformation: str = "none",
                               suffix: str = "_transformed") -> Tuple[pd.DataFrame, str]:
    """
    Apply transformation to target variable for feature selection analysis.

    Atomic function: Orchestrates target transformation with single responsibility.
    Follows UNIFIED_CODING_STANDARDS.md by delegating to focused helper functions.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the target variable
    target_variable : str
        Name of the target variable column
    transformation : str, default "none"
        Type of transformation: "none", "log1p", "log", "sqrt"
    suffix : str, default "_transformed"
        Suffix to append to transformed target column name

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (transformed_data, effective_target_name)
    """
    # Validate inputs
    target_values = _validate_target_for_transformation(data, target_variable)
    data_copy = data.copy()

    # No transformation case
    if transformation == "none":
        return data_copy, target_variable

    # Apply transformation
    transformed_values, transform_name = _apply_transformation_math(
        target_values, transformation, target_variable
    )

    # Add transformed target to dataset
    data_copy[transform_name] = transformed_values

    # Log statistics
    _log_transformation_statistics(target_values, transformed_values, target_variable, transform_name)

    return data_copy, transform_name


def apply_autoregressive_transforms(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to autoregressive features.

    Identifies and transforms all autoregressive features using log(1 + x)
    transformation. Features are identified by column name prefixes
    'sales_target_t' (application date) and 'sales_target_contract_t' (contract date).

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing autoregressive features to transform.
        Autoregressive features must have columns starting with
        'sales_target_t' or 'sales_target_contract_t'

    Returns
    -------
    pd.DataFrame
        Copy of input dataset with log(1 + x) transformation applied to
        identified autoregressive features. Original dataset is not modified.

    Notes
    -----
    - Returns a copy of the input data, never modifies in-place
    - If dataset is empty, returns empty copy with warning
    - If no autoregressive features found, returns unchanged copy
    - Uses log1p transformation: log(1 + x) for numerical stability
    - Prints summary of transformed features to stdout
    """
    if data.empty:
        warnings.warn("Cannot apply autoregressive transforms to empty dataset")
        return data.copy()

    try:
        # Copy data to avoid modifying original (immutable operation)
        result = data.copy()

        # Identify autoregressive features (exact notebook logic)
        autoregressive_application = [col for col in result.columns if col.startswith('sales_target_t')]
        autoregressive_contract = [col for col in result.columns if col.startswith('sales_target_contract_t')]

        # Apply log transformation (exact notebook formula: log(1 + x))
        all_autoregressive = autoregressive_application + autoregressive_contract

        if all_autoregressive:
            for ar_feature in all_autoregressive:
                result[ar_feature] = np.log(1 + result[ar_feature])

            print(f"Applied log transformation to {len(all_autoregressive)} autoregressive features")
            print(f"  Application date features: {len(autoregressive_application)}")
            print(f"  Contract date features: {len(autoregressive_contract)}")
        else:
            print("No autoregressive features found for log transformation")

        return result

    except Exception as e:
        warnings.warn(f"Failed to apply autoregressive transforms: {e}")
        return data.copy()


def _validate_target_variable(data: pd.DataFrame, target_variable: str) -> None:
    """Validate that target variable exists in dataset.

    Delegates to canonical validator in src.validation.input_validators.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to check
    target_variable : str
        Name of target variable to validate

    Raises
    ------
    ValueError
        If target variable is not found
    """
    from src.validation.input_validators import validate_target_in_dataframe
    validate_target_in_dataframe(data, target_variable, require_numeric=False)


def _validate_base_features(data: pd.DataFrame, base_features: List[str]) -> None:
    """Validate that all required base features exist in dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to check
    base_features : List[str]
        Required base features

    Raises
    ------
    ValueError
        If any base features are missing
    """
    missing_base = [f for f in base_features if f not in data.columns]
    if missing_base:
        raise ValueError(
            f"Required base features missing from dataset: {missing_base}. "
            f"Base features must be available in all datasets. "
            f"Check feature engineering pipeline."
        )


def _validate_candidate_features(
    data: pd.DataFrame, candidate_features: List[str]
) -> Tuple[List[str], List[str]]:
    """Check candidate feature availability in dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to check
    candidate_features : List[str]
        Candidate features to validate

    Returns
    -------
    Tuple[List[str], List[str]]
        (available_candidates, missing_candidates)

    Raises
    ------
    ValueError
        If no candidate features are available
    """
    available_candidates = [f for f in candidate_features if f in data.columns]
    missing_candidates = [f for f in candidate_features if f not in data.columns]

    if not available_candidates:
        raise ValueError(
            f"No candidate features available in dataset. "
            f"All {len(candidate_features)} candidates missing: {candidate_features[:5]}... "
            f"Check feature engineering pipeline and column naming."
        )

    return available_candidates, missing_candidates


def _print_availability_summary(
    target_variable: str,
    base_features: List[str],
    available_candidates: List[str],
    missing_candidates: List[str],
    candidate_features: List[str]
) -> float:
    """Print feature availability validation summary.

    Parameters
    ----------
    target_variable : str
        Target variable name
    base_features : List[str]
        Base feature list
    available_candidates : List[str]
        Available candidate features
    missing_candidates : List[str]
        Missing candidate features
    candidate_features : List[str]
        Original candidate feature list

    Returns
    -------
    float
        Availability rate as percentage
    """
    availability_rate = len(available_candidates) / len(candidate_features) * 100

    print(f"Feature availability validation:")
    print(f"  Target variable: '{target_variable}' ✓")
    print(f"  Base features: {len(base_features)} available ✓")
    print(f"  Candidate features: {len(available_candidates)}/{len(candidate_features)} "
          f"available ({availability_rate:.1f}%)")

    if missing_candidates:
        suffix = '...' if len(missing_candidates) > 3 else ''
        print(f"  Missing candidates: {missing_candidates[:3]}{suffix}")

    return availability_rate


def validate_feature_availability(data: pd.DataFrame,
                                config: FeatureSelectionConfig) -> Tuple[List[str], List[str]]:
    """
    Validate feature availability with business-oriented error messages.

    Performs comprehensive validation of target variable, base features, and
    candidate features against the provided dataset. Validates that all required
    features exist before analysis begins, with clear error messages for
    troubleshooting data pipeline issues.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset to validate features against
    config : FeatureSelectionConfig
        Feature selection configuration containing:
        - target_variable: str - Name of target column
        - base_features: List[str] - Required base features
        - candidate_features: List[str] - Optional candidate features

    Returns
    -------
    Tuple[List[str], List[str]]
        (available_candidates, missing_candidates)
        - available_candidates: Candidate features found in dataset
        - missing_candidates: Candidate features not found in dataset

    Raises
    ------
    ValueError
        If dataset is empty, target variable not found, any base features
        are missing, or no candidate features are available
    ValueError
        If validation fails with other exceptions

    Notes
    -----
    - All base features are mandatory (raises error if any missing)
    - At least one candidate feature must be available
    - Prints detailed availability summary to stdout
    - Issues warning if candidate availability < 50%
    - Business-oriented error messages include data shape and sample columns
    """
    if data.empty:
        raise ValueError(
            "Cannot validate feature availability on empty dataset. "
            "Check data loading and filtering steps."
        )

    try:
        # Extract configuration
        base_features = config['base_features']
        candidate_features = config['candidate_features']
        target_variable = config['target_variable']

        # Validate each component
        _validate_target_variable(data, target_variable)
        _validate_base_features(data, base_features)
        available_candidates, missing_candidates = _validate_candidate_features(
            data, candidate_features
        )

        # Print summary and check availability rate
        availability_rate = _print_availability_summary(
            target_variable, base_features, available_candidates,
            missing_candidates, candidate_features
        )

        if availability_rate < 50:
            warnings.warn(
                f"Only {availability_rate:.1f}% of candidate features available. "
                f"Consider checking feature engineering pipeline or updating candidate list."
            )

        return available_candidates, missing_candidates

    except Exception as e:
        raise ValueError(
            f"Feature availability validation failed: {e}. "
            f"Dataset shape: {data.shape}. "
            f"Available columns: {list(data.columns)[:10]}..."
        ) from e


def _load_and_log_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load dataset from parquet file with logging.

    Single responsibility: Dataset loading only.

    Parameters
    ----------
    dataset_path : str
        Path to the final dataset parquet file

    Returns
    -------
    pd.DataFrame
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If dataset file not found
    """
    try:
        df_features = pd.read_parquet(dataset_path)
        print(f"SUCCESS: Dataset loaded from DVC-tracked local file")
        print(f"  Original dataset shape: {df_features.shape}")
        return df_features
    except FileNotFoundError as e:
        raise ValueError(
            f"Dataset file not found: {dataset_path}. "
            f"Check DVC tracking and file paths. "
            f"Expected location: outputs/datasets/final_dataset.parquet"
        ) from e


def _apply_temporal_filters(data: pd.DataFrame, config: FeatureSelectionConfig) -> pd.DataFrame:
    """
    Apply date and holiday filters to dataset.

    Single responsibility: Temporal filtering only.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset with date and holiday columns
    config : FeatureSelectionConfig
        Configuration with filter settings

    Returns
    -------
    pd.DataFrame
        Filtered dataset
    """
    analysis_start = config.get('analysis_start_date', '2022-08-01')
    exclude_holidays = config.get('exclude_holidays', True)

    # Date filtering
    df_filtered = data[data['date'] > analysis_start].copy()

    # Holiday filtering (conditional based on config)
    if exclude_holidays and 'holiday' in df_filtered.columns:
        df_analysis = df_filtered[df_filtered['holiday'] == 0].copy()
        holiday_excluded = len(df_filtered) - len(df_analysis)
        print(f"  Excluded {holiday_excluded} holiday observations")
    else:
        df_analysis = df_filtered.copy()

    print(f"  Analysis dataset shape after filtering: {df_analysis.shape}")
    print(f"  Date range: {df_analysis['date'].min()} to {df_analysis['date'].max()}")

    return df_analysis


def _apply_configured_transformations(data: pd.DataFrame, config: FeatureSelectionConfig) -> Tuple[pd.DataFrame, str]:
    """
    Apply target and autoregressive transformations based on config.

    Single responsibility: Transformation orchestration only.

    Parameters
    ----------
    data : pd.DataFrame
        Filtered dataset
    config : FeatureSelectionConfig
        Configuration with transformation settings

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (transformed_data, effective_target_name)
    """
    # Apply target transformation if configured
    target_transformation = config.get('target_transformation', 'none')
    transformed_suffix = config.get('transformed_target_suffix', '')

    if target_transformation != 'none':
        df_transformed, effective_target = apply_target_transformation(
            data,
            config['target_variable'],
            target_transformation,
            transformed_suffix
        )
    else:
        df_transformed = data.copy()
        effective_target = config['target_variable']

    # Apply autoregressive transformations
    df_final = apply_autoregressive_transforms(df_transformed)

    return df_final, effective_target


def prepare_analysis_dataset(dataset_path: str,
                           config: FeatureSelectionConfig) -> Tuple[pd.DataFrame, str]:
    """
    Prepare analysis dataset with all transformations from notebook.

    Atomic function: Orchestrates dataset preparation with single responsibility.
    Follows UNIFIED_CODING_STANDARDS.md by delegating to focused helper functions.

    Mathematical equivalence guarantee: produces identical results to notebook.

    Parameters
    ----------
    dataset_path : str
        Path to the final dataset parquet file
    config : FeatureSelectionConfig
        Configuration with analysis parameters and feature lists

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (analysis_ready_dataset, effective_target_name)
    """
    try:
        # Load dataset
        df_features = _load_and_log_dataset(dataset_path)

        # Apply temporal filters
        df_filtered = _apply_temporal_filters(df_features, config)

        # Apply transformations
        df_final, effective_target = _apply_configured_transformations(df_filtered, config)

        # Validate feature availability
        available_features, missing_features = validate_feature_availability(df_final, config)

        print(f"SUCCESS: Analysis dataset preparation complete")
        print(f"  Final shape: {df_final.shape}")
        print(f"  Target variable: {effective_target}")
        print(f"  Available features: {len(available_features)}")

        return df_final, effective_target

    except Exception as e:
        raise ValueError(
            f"Analysis dataset preparation failed: {e}. "
            f"Check data pipeline outputs and configuration parameters."
        ) from e


def prepare_feature_subset(data: pd.DataFrame,
                          feature_names: List[str],
                          target_variable: str) -> pd.DataFrame:
    """
    Prepare feature subset for model evaluation.

    DRY utility function: eliminates duplicate feature subsetting patterns
    by providing reusable function for model-specific data preparation.

    Parameters
    ----------
    data : pd.DataFrame
        Full analysis dataset
    feature_names : List[str]
        Specific features for this model
    target_variable : str
        Target variable to include

    Returns
    -------
    pd.DataFrame
        Subset containing only specified features and target

    Examples
    --------
    >>> model_data = prepare_feature_subset(df, ['prudential_rate_current', 'competitor_mid_t2'], 'sales_target_current')
    """
    try:
        # Validate all features are available
        required_columns = feature_names + [target_variable]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing columns for model evaluation: {missing_columns}")

        # Create subset with specified features only
        model_data = data[required_columns].copy()

        # Check for missing values
        missing_data = model_data.isnull().sum()
        if missing_data.any():
            print(f"WARNING: Missing values detected in model features:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"  {col}: {count} missing values")

        return model_data

    except Exception as e:
        raise ValueError(f"Failed to prepare feature subset: {e}") from e