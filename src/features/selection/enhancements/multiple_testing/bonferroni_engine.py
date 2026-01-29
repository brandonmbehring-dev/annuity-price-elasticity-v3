"""
Bonferroni Correction Engine for Multiple Testing.

This module implements Bonferroni correction for family-wise error rate (FWER)
control in feature selection. Extracted from multiple_testing_correction.py
for maintainability.

Mathematical Foundation:
- Bonferroni: α_corrected = α / m (where m = number of tests)
- For 793 models: α_corrected = 0.05 / 793 = 0.000063

Key Function:
- apply_bonferroni_correction: Conservative FWER control

Design Principles:
- Single correction method per module
- Clear validation → calculation → significance → build pattern
- Comprehensive business context in errors
"""

import logging
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.features.selection.enhancements.multiple_testing.multiple_testing_types import MultipleTestingResults

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================


def _validate_bonferroni_inputs(model_results: pd.DataFrame,
                                aic_column: str) -> None:
    """
    Validate inputs for Bonferroni correction.

    Parameters
    ----------
    model_results : pd.DataFrame
        AIC evaluation results with model performance metrics
    aic_column : str
        Column containing AIC values for comparison

    Raises
    ------
    ValueError
        If model_results is empty or aic_column is missing
    """
    if len(model_results) == 0:
        raise ValueError(
            f"CRITICAL: No model results provided for Bonferroni correction. "
            f"Business impact: Cannot control false discovery rate. "
            f"Required action: Ensure AIC evaluation completed successfully."
        )

    if aic_column not in model_results.columns:
        raise ValueError(
            f"CRITICAL: AIC column '{aic_column}' not found in results. "
            f"Available columns: {list(model_results.columns)}. "
            f"Business impact: Cannot apply statistical correction. "
            f"Required action: Verify AIC evaluation output format."
        )


# =============================================================================
# CALCULATION
# =============================================================================


def _calculate_bonferroni_threshold(n_tests: int, alpha: float) -> Tuple[float, float]:
    """
    Calculate Bonferroni corrected alpha and critical value.

    Parameters
    ----------
    n_tests : int
        Number of statistical tests performed
    alpha : float
        Original family-wise error rate

    Returns
    -------
    Tuple[float, float]
        Corrected alpha and chi-squared critical value
    """
    from scipy.stats import chi2

    corrected_alpha = alpha / n_tests
    critical_value = chi2.ppf(1 - corrected_alpha, df=1)

    logger.info(f"Applying Bonferroni correction: {n_tests} tests, "
                f"corrected α = {corrected_alpha:.6f}")

    return corrected_alpha, critical_value


# =============================================================================
# SIGNIFICANCE TEST
# =============================================================================


def _apply_bonferroni_significance(model_results: pd.DataFrame,
                                   aic_column: str,
                                   critical_value: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Bonferroni significance test and split results.

    Parameters
    ----------
    model_results : pd.DataFrame
        AIC evaluation results
    aic_column : str
        Column containing AIC values
    critical_value : float
        Chi-squared critical value for significance

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Significant models and rejected models
    """
    min_aic = model_results[aic_column].min()

    model_results = model_results.copy()
    model_results['aic_diff'] = model_results[aic_column] - min_aic
    model_results['bonferroni_significant'] = model_results['aic_diff'] <= critical_value

    significant_models = model_results[model_results['bonferroni_significant']].copy()
    rejected_models = model_results[~model_results['bonferroni_significant']].copy()

    return significant_models, rejected_models


# =============================================================================
# IMPACT ASSESSMENT (Internal helpers)
# =============================================================================


def _calculate_correction_impact(original_count: int,
                                significant_count: int,
                                method: str) -> Dict[str, Any]:
    """
    Calculate impact of multiple testing correction.

    Parameters
    ----------
    original_count : int
        Original number of models
    significant_count : int
        Models remaining after correction
    method : str
        Correction method used

    Returns
    -------
    Dict[str, Any]
        Impact assessment with business interpretation
    """
    rejection_rate = (original_count - significant_count) / original_count
    retention_rate = significant_count / original_count

    return {
        'models_rejected': original_count - significant_count,
        'models_retained': significant_count,
        'rejection_rate_percent': rejection_rate * 100,
        'retention_rate_percent': retention_rate * 100,
        'correction_stringency': 'HIGH' if rejection_rate > 0.8 else 'MODERATE' if rejection_rate > 0.5 else 'LOW',
        'business_impact': _interpret_rejection_rate(rejection_rate, method)
    }


def _interpret_rejection_rate(rejection_rate: float, method: str) -> str:
    """
    Interpret rejection rate for business stakeholders.

    Parameters
    ----------
    rejection_rate : float
        Proportion of models rejected
    method : str
        Correction method

    Returns
    -------
    str
        Business interpretation
    """
    if rejection_rate > 0.9:
        return f"{method} correction is very stringent - may eliminate valid models"
    elif rejection_rate > 0.7:
        return f"{method} correction is moderately stringent - good balance of rigor and discovery"
    elif rejection_rate > 0.3:
        return f"{method} correction is conservative but allows reasonable discovery"
    else:
        return f"{method} correction has minimal impact - may not adequately control false positives"


def _estimate_statistical_power(n_tests: int,
                               alpha: float,
                               effect_size: float = 0.1) -> Dict[str, float]:
    """
    Estimate statistical power for multiple testing scenario.

    Parameters
    ----------
    n_tests : int
        Number of statistical tests
    alpha : float
        Significance level
    effect_size : float, default 0.1
        Assumed effect size for power calculation

    Returns
    -------
    Dict[str, float]
        Power analysis results
    """
    from scipy.stats import norm

    # Z-score for given alpha (two-tailed)
    z_alpha = norm.ppf(1 - alpha / 2)

    # Approximate power calculation
    z_beta = z_alpha - effect_size / np.sqrt(1 / n_tests)
    power = 1 - norm.cdf(z_beta)

    return {
        'statistical_power': max(0.0, min(1.0, power)),
        'type_ii_error_rate': 1 - power,
        'power_interpretation': 'HIGH' if power > 0.8 else 'MODERATE' if power > 0.6 else 'LOW',
        'effect_size_assumed': effect_size
    }


# =============================================================================
# RESULT BUILDER
# =============================================================================


def _build_bonferroni_results(alpha: float,
                              corrected_alpha: float,
                              n_tests: int,
                              significant_models: pd.DataFrame,
                              rejected_models: pd.DataFrame,
                              min_significant_models: int) -> MultipleTestingResults:
    """Build final Bonferroni correction results with impact assessment."""
    if len(significant_models) < min_significant_models:
        warnings.warn(
            f"Bonferroni correction rejected all but {len(significant_models)} models. "
            f"This may be overly conservative for exploratory analysis."
        )

    correction_impact = _calculate_correction_impact(
        original_count=n_tests,
        significant_count=len(significant_models),
        method='bonferroni'
    )

    power_analysis = _estimate_statistical_power(
        n_tests=n_tests,
        alpha=corrected_alpha,
        effect_size=0.1
    )

    return MultipleTestingResults(
        method='bonferroni',
        original_alpha=alpha,
        corrected_alpha=corrected_alpha,
        n_tests=n_tests,
        significant_models=significant_models,
        rejected_models=rejected_models,
        correction_impact=correction_impact,
        statistical_power=power_analysis
    )


# =============================================================================
# PUBLIC API
# =============================================================================


def apply_bonferroni_correction(model_results: pd.DataFrame,
                               aic_column: str = 'aic',
                               alpha: float = 0.05,
                               min_significant_models: int = 1) -> MultipleTestingResults:
    """Apply Bonferroni correction for family-wise error rate control."""
    # Step 1: Validate inputs
    _validate_bonferroni_inputs(model_results, aic_column)

    # Step 2: Calculate corrected threshold
    n_tests = len(model_results)
    corrected_alpha, critical_value = _calculate_bonferroni_threshold(n_tests, alpha)

    # Step 3: Apply significance test
    significant_models, rejected_models = _apply_bonferroni_significance(
        model_results, aic_column, critical_value
    )

    # Step 4: Build and return results
    return _build_bonferroni_results(
        alpha=alpha,
        corrected_alpha=corrected_alpha,
        n_tests=n_tests,
        significant_models=significant_models,
        rejected_models=rejected_models,
        min_significant_models=min_significant_models
    )
