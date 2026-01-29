"""
AIC Evaluation Engine for Feature Selection.

This module provides atomic functions for calculating AIC scores across
feature combinations, following the established pipeline architecture.

Key Functions:
- evaluate_aic_combinations: Main atomic function for AIC evaluation
- calculate_aic_for_features: Single feature combination AIC calculation
- generate_feature_combinations: Systematic combination generation

Design Principles:
- Single responsibility: AIC calculation only
- Immutable operations: (data, config) -> results
- Business-context error handling
- Mathematical equivalence to existing notebook implementation
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import combinations

# Import types - Fail fast with clear error if imports fail
from src.features.selection_types import FeatureSelectionConfig, AICResult


def _create_error_aic_result(features: List[str], data_length: int, error_msg: str) -> AICResult:
    """
    Create standardized error AIC result.

    Single responsibility: Error result generation only.

    Parameters
    ----------
    features : List[str]
        Feature list for error context
    data_length : int
        Dataset length for n_obs
    error_msg : str
        Error message to include

    Returns
    -------
    AICResult
        Standardized error result
    """
    return AICResult(
        features=' + '.join(features) if features else "",
        n_features=len(features),
        aic=np.inf,
        bic=np.inf,
        r_squared=0.0,
        r_squared_adj=0.0,
        coefficients={},
        converged=False,
        n_obs=data_length,
        standard_errors={},
        error=error_msg
    )


def _validate_aic_inputs(data: pd.DataFrame, features: List[str], target: str) -> Optional[AICResult]:
    """
    Validate inputs for AIC calculation.

    Single responsibility: Input validation only.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    features : List[str]
        Feature list to validate
    target : str
        Target variable to validate

    Returns
    -------
    Optional[AICResult]
        Error result if validation fails, None if validation passes
    """
    if data.empty:
        return _create_error_aic_result(
            features, 0, "Cannot calculate metrics on empty dataset"
        )

    if not features:
        return _create_error_aic_result(
            features, len(data), "No features provided for model calculation"
        )

    # Validate feature availability
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        return _create_error_aic_result(
            features, len(data),
            f"Missing features in dataset: {missing_features}. Available columns: {list(data.columns)[:10]}..."
        )

    # Validate target availability
    if target not in data.columns:
        return _create_error_aic_result(
            features, len(data),
            f"Target variable '{target}' not found in dataset. Available columns: {list(data.columns)[:10]}..."
        )

    return None  # Validation passed


def _fit_ols_model(data: pd.DataFrame, features: List[str], target: str) -> AICResult:
    """
    Fit OLS model and extract metrics.

    Single responsibility: Model fitting and metric extraction only.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    features : List[str]
        Feature list for regression
    target : str
        Target variable

    Returns
    -------
    AICResult
        Model results or error result (includes standard errors for CI-based constraints)
    """
    try:
        # Create regression formula (exact notebook logic)
        formula = f"{target} ~ {' + '.join(features)}"

        # Fit OLS model using statsmodels (same as notebook)
        model = smf.ols(formula, data=data).fit()

        # Calculate BIC (exact same formula as original)
        # BIC = -2 * log-likelihood + k * log(n)
        # where k is number of parameters (including intercept), n is number of observations
        n_params = len(model.params)  # includes intercept
        bic_score = model.aic + (n_params * (np.log(model.nobs) - 2))

        # Extract standard errors for CI-based constraint validation
        # (Addresses Issue #6: constraints using point estimates only)
        standard_errors = dict(model.bse)

        return AICResult(
            features=' + '.join(features),
            n_features=len(features),
            aic=model.aic,
            bic=bic_score,
            r_squared=model.rsquared,
            r_squared_adj=model.rsquared_adj,
            coefficients=dict(model.params),
            converged=True,
            n_obs=int(model.nobs),
            standard_errors=standard_errors
        )

    except Exception as e:
        return _create_error_aic_result(
            features, len(data),
            f"Model calculation failed for features {features} with target '{target}': {str(e)}"
        )


def calculate_aic_for_features(data: pd.DataFrame,
                             features: List[str],
                             target: str) -> AICResult:
    """
    Calculate comprehensive model metrics for specific feature combination.

    Atomic function: Orchestrates AIC calculation with single responsibility.
    Follows UNIFIED_CODING_STANDARDS.md by delegating to focused helper functions.

    Mathematical equivalence guarantee: produces identical results to notebook.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target variable
    features : List[str]
        List of feature column names for regression model
    target : str
        Target variable column name for regression

    Returns
    -------
    AICResult
        Structured result with AIC, R², coefficients, and metadata
    """
    # Validate inputs
    validation_error = _validate_aic_inputs(data, features, target)
    if validation_error is not None:
        return validation_error

    # Fit model and extract metrics
    return _fit_ols_model(data, features, target)


def _validate_combination_generation_inputs(
    candidate_features: List[str],
    max_candidates: int
) -> int:
    """
    Validate and adjust inputs for combination generation.

    Parameters
    ----------
    candidate_features : List[str]
        Features to evaluate in combinations
    max_candidates : int
        Maximum number of candidate features per combination

    Returns
    -------
    int
        Adjusted max_candidates value

    Raises
    ------
    ValueError
        If parameters are invalid with business context explanation
    """
    if max_candidates < 1:
        raise ValueError(
            f"max_candidates must be >= 1 for meaningful feature selection. "
            f"Got: {max_candidates}"
        )

    if not candidate_features:
        raise ValueError(
            "No candidate features provided for combination generation. "
            "Feature selection requires at least one candidate feature to evaluate."
        )

    if max_candidates > len(candidate_features):
        adjusted = len(candidate_features)
        print(f"WARNING: max_candidates reduced from {max_candidates} to {adjusted}")
        return adjusted

    return max_candidates


def generate_feature_combinations(
    base_features: List[str],
    candidate_features: List[str],
    max_candidates: int
) -> List[List[str]]:
    """
    Generate systematic feature combinations for AIC evaluation.

    Parameters
    ----------
    base_features : List[str]
        Features required in all models (always included)
    candidate_features : List[str]
        Features to evaluate in combinations
    max_candidates : int
        Maximum number of candidate features per combination

    Returns
    -------
    List[List[str]]
        All valid feature combinations (base + candidate combinations)
    """
    max_candidates = _validate_combination_generation_inputs(
        candidate_features, max_candidates
    )

    try:
        feature_combinations = []
        for k in range(1, max_candidates + 1):
            for candidate_combo in combinations(candidate_features, k):
                all_features = base_features + list(candidate_combo)
                feature_combinations.append(all_features)
        return feature_combinations

    except Exception as e:
        raise ValueError(
            f"Failed to generate feature combinations with {len(base_features)} base "
            f"features and {len(candidate_features)} candidates: {e}"
        ) from e


def _validate_evaluation_inputs(data: pd.DataFrame, config: FeatureSelectionConfig, target: str) -> List[str]:
    """
    Validate inputs for AIC evaluation across combinations.

    Single responsibility: Input validation only.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    config : FeatureSelectionConfig
        Configuration to validate
    target : str
        Target variable to validate

    Returns
    -------
    List[str]
        Available candidate features

    Raises
    ------
    ValueError
        If validation fails
    """
    if data.empty:
        raise ValueError(
            "Cannot perform AIC evaluation on empty dataset. "
            "Check data loading and filtering steps in the pipeline."
        )

    if target not in data.columns:
        available_targets = [col for col in data.columns if 'sales' in col.lower()]
        raise ValueError(
            f"Target variable '{target}' not found in dataset. "
            f"Available sales-related columns: {available_targets}. "
            f"Check target_variable parameter or configuration."
        )

    available_candidates = [f for f in config['candidate_features'] if f in data.columns]
    if not available_candidates:
        missing = config['candidate_features']
        raise ValueError(
            f"No candidate features found in dataset. "
            f"Missing: {missing}. Available columns: {list(data.columns)[:10]}... "
            f"Check feature engineering pipeline and column names."
        )

    return available_candidates


def _evaluate_all_combinations(data: pd.DataFrame, feature_combinations: List[List[str]], target: str) -> pd.DataFrame:
    """
    Evaluate AIC for all feature combinations.

    Single responsibility: Combination evaluation only.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    feature_combinations : List[List[str]]
        All feature combinations to evaluate
    target : str
        Target variable

    Returns
    -------
    pd.DataFrame
        Results DataFrame with all combinations evaluated
    """
    print(f"Evaluating {len(feature_combinations)} feature combinations...")

    # Calculate AIC for each combination (exact notebook replication)
    results = []
    for features in feature_combinations:
        aic_result = calculate_aic_for_features(data, features, target)
        results.append(asdict(aic_result))

    # Convert to DataFrame (same structure as notebook)
    results_df = pd.DataFrame(results)

    print(f"AIC evaluation complete: {len(results_df)} combinations evaluated")
    print(f"Converged models: {results_df['converged'].sum()}")

    return results_df


def evaluate_aic_combinations(data: pd.DataFrame,
                            config: FeatureSelectionConfig,
                            target_variable: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate comprehensive model metrics across all feature combinations.

    Atomic function: Orchestrates AIC evaluation with single responsibility.
    Follows UNIFIED_CODING_STANDARDS.md by delegating to focused helper functions.

    Mathematical equivalence guarantee: produces identical results to notebook.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing all features and target variable
    config : FeatureSelectionConfig
        Type-safe configuration with feature lists and parameters
    target_variable : Optional[str], default None
        Override target variable name (for transformed targets)

    Returns
    -------
    pd.DataFrame
        Results with AIC, R², coefficients, and metadata columns
    """
    try:
        # Determine target variable
        target = target_variable if target_variable is not None else config['target_variable']

        # Validate inputs
        available_candidates = _validate_evaluation_inputs(data, config, target)

        # Generate feature combinations
        feature_combinations = generate_feature_combinations(
            config['base_features'],
            available_candidates,
            config['max_candidate_features']
        )

        # Evaluate all combinations
        return _evaluate_all_combinations(data, feature_combinations, target)

    except Exception as e:
        raise ValueError(
            f"AIC evaluation failed: {e}"
        ) from e