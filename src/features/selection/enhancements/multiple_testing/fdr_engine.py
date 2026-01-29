"""
FDR Correction Engine for Multiple Testing.

This module implements False Discovery Rate (FDR) correction using the
Benjamini-Hochberg procedure. Extracted from multiple_testing_correction.py
for maintainability.

Mathematical Foundation:
- FDR (Benjamini-Hochberg): Controls expected proportion of false discoveries
- Less conservative than Bonferroni, more power for discovery

Key Function:
- apply_fdr_correction: FDR control using Benjamini-Hochberg

Design Principles:
- Single correction method per module
- Clear validation → calculation → significance → build pattern
- Comprehensive business context in errors
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.features.selection.enhancements.multiple_testing.multiple_testing_types import MultipleTestingResults

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION
# =============================================================================


def _validate_fdr_inputs(model_results: pd.DataFrame, aic_column: str) -> None:
    """
    Validate inputs for FDR correction.

    Parameters
    ----------
    model_results : pd.DataFrame
        AIC evaluation results
    aic_column : str
        Column containing AIC values

    Raises
    ------
    ValueError
        If model_results is empty or aic_column missing
    """
    if len(model_results) == 0:
        raise ValueError(
            "CRITICAL: No model results provided for FDR correction. "
            "Business impact: Cannot control false discovery rate. "
            "Required action: Ensure AIC evaluation completed successfully."
        )

    if aic_column not in model_results.columns:
        raise ValueError(
            f"CRITICAL: AIC column '{aic_column}' not found in results. "
            f"Available columns: {list(model_results.columns)}. "
            f"Business impact: Cannot apply FDR correction. "
            f"Required action: Verify AIC evaluation output format."
        )


# =============================================================================
# CALCULATION
# =============================================================================


def _convert_aic_to_pvalues(model_results: pd.DataFrame, aic_column: str) -> pd.DataFrame:
    """
    Convert AIC differences to approximate p-values using chi-squared approximation.

    Parameters
    ----------
    model_results : pd.DataFrame
        AIC evaluation results
    aic_column : str
        Column containing AIC values

    Returns
    -------
    pd.DataFrame
        Model results with aic_diff and p_value columns added
    """
    from scipy.stats import chi2

    results = model_results.copy()
    min_aic = results[aic_column].min()
    results['aic_diff'] = results[aic_column] - min_aic

    # Chi-squared approximation (df=1 for nested model comparison)
    results['p_value'] = 1 - chi2.cdf(results['aic_diff'], df=1)

    # Handle numerical issues (very small p-values)
    results['p_value'] = np.clip(results['p_value'], 1e-16, 1.0)

    return results


def _apply_fdr_to_pvalues(
    model_results: pd.DataFrame, alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply FDR correction and split into significant/rejected models.

    Parameters
    ----------
    model_results : pd.DataFrame
        Model results with p_value column
    alpha : float
        False discovery rate threshold

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (significant_models, rejected_models)
    """
    from statsmodels.stats.multitest import fdrcorrection

    reject_fdr, p_corrected = fdrcorrection(
        model_results['p_value'].values, alpha=alpha
    )

    model_results['fdr_corrected_p'] = p_corrected
    model_results['fdr_significant'] = reject_fdr

    significant_models = model_results[model_results['fdr_significant']].copy()
    rejected_models = model_results[~model_results['fdr_significant']].copy()

    return significant_models, rejected_models


def _compute_effective_alpha(
    significant_models: pd.DataFrame, alpha: float, n_models: int
) -> float:
    """
    Compute effective alpha threshold from FDR correction.

    Parameters
    ----------
    significant_models : pd.DataFrame
        Models that passed FDR correction
    alpha : float
        Original alpha level
    n_models : int
        Total number of models tested

    Returns
    -------
    float
        Effective alpha threshold
    """
    if len(significant_models) > 0:
        return significant_models['p_value'].max()
    return alpha / n_models  # Fallback to Bonferroni-like


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


def _build_fdr_results(alpha: float,
                       effective_alpha: float,
                       n_tests: int,
                       significant_models: pd.DataFrame,
                       rejected_models: pd.DataFrame) -> MultipleTestingResults:
    """
    Build final FDR correction results with impact assessment.

    Parameters
    ----------
    alpha : float
        Original false discovery rate
    effective_alpha : float
        Effective alpha threshold after correction
    n_tests : int
        Number of statistical tests
    significant_models : pd.DataFrame
        Models passing FDR threshold
    rejected_models : pd.DataFrame
        Models rejected by FDR correction

    Returns
    -------
    MultipleTestingResults
        Complete FDR correction results
    """
    correction_impact = _calculate_correction_impact(
        original_count=n_tests,
        significant_count=len(significant_models),
        method='fdr_bh'
    )
    power_analysis = _estimate_statistical_power(
        n_tests=n_tests,
        alpha=effective_alpha,
        effect_size=0.1
    )

    return MultipleTestingResults(
        method='fdr_bh',
        original_alpha=alpha,
        corrected_alpha=effective_alpha,
        n_tests=n_tests,
        significant_models=significant_models,
        rejected_models=rejected_models,
        correction_impact=correction_impact,
        statistical_power=power_analysis
    )


# =============================================================================
# PUBLIC API
# =============================================================================


def apply_fdr_correction(model_results: pd.DataFrame,
                        aic_column: str = 'aic',
                        alpha: float = 0.05,
                        method: str = 'fdr_bh') -> MultipleTestingResults:
    """Apply FDR correction using Benjamini-Hochberg procedure."""
    try:
        # Step 1: Validate inputs
        _validate_fdr_inputs(model_results, aic_column)

        # Step 2: Convert AIC to p-values and apply FDR correction
        results_with_pvalues = _convert_aic_to_pvalues(model_results, aic_column)
        significant_models, rejected_models = _apply_fdr_to_pvalues(
            results_with_pvalues, alpha
        )

        # Step 3: Calculate effective alpha
        effective_alpha = _compute_effective_alpha(
            significant_models, alpha, len(model_results)
        )

        # Step 4: Build and return results
        return _build_fdr_results(
            alpha=alpha,
            effective_alpha=effective_alpha,
            n_tests=len(model_results),
            significant_models=significant_models,
            rejected_models=rejected_models
        )

    except ImportError as e:
        raise ValueError(
            f"CRITICAL: Required statistical library unavailable. "
            f"Business impact: Cannot apply FDR correction. "
            f"Required action: Install statsmodels. "
            f"Original error: {e}"
        ) from e
