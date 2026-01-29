"""
Multiple Testing Correction Engine for Feature Selection.

This module addresses Issue #1 from the mathematical analysis report:
the critical multiple testing problem where 793 models are tested without
family-wise error rate (FWER) correction, leading to ~40 expected false discoveries.

Key Functions:
- apply_bonferroni_correction: Conservative FWER control
- apply_fdr_correction: False Discovery Rate control (Benjamini-Hochberg)
- create_reduced_search_space: Domain-guided combination reduction
- compare_correction_methods: Comprehensive method comparison

Critical Statistical Issues Addressed:
- Issue #1: Multiple Testing Without Correction (SEVERITY: CRITICAL)
- Expected False Discoveries: 793 × α = 793 × 0.05 = 39.65 spurious models
- Family-Wise Error Rate: P(≥1 false discovery) ≈ 100%
- Invalid statistical inference with meaningless p-values and confidence intervals

Module Architecture (Phase 6.4 Split):
- multiple_testing_types.py: Shared dataclass
- bonferroni_engine.py: Bonferroni correction
- fdr_engine.py: FDR correction
- search_space_reduction.py: Search space reduction
- multiple_testing_correction.py: Orchestrator + method comparison (this file)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES
# =============================================================================

# Shared types
from src.features.selection.enhancements.multiple_testing.multiple_testing_types import MultipleTestingResults

# Bonferroni engine
from src.features.selection.enhancements.multiple_testing.bonferroni_engine import (
    apply_bonferroni_correction,
    _validate_bonferroni_inputs,
    _calculate_bonferroni_threshold,
    _apply_bonferroni_significance,
    _build_bonferroni_results,
    _calculate_correction_impact,
    _interpret_rejection_rate,
    _estimate_statistical_power,
)

# FDR engine
from src.features.selection.enhancements.multiple_testing.fdr_engine import (
    apply_fdr_correction,
    _validate_fdr_inputs,
    _convert_aic_to_pvalues,
    _apply_fdr_to_pvalues,
    _compute_effective_alpha,
    _build_fdr_results,
)

# Search space reduction
from src.features.selection.enhancements.multiple_testing.search_space_reduction import (
    create_reduced_search_space,
    _validate_search_space_inputs,
    _generate_priority_combinations,
    _add_non_priority_combinations,
    _limit_and_sort_combinations,
    _build_search_space_result,
)


# =============================================================================
# METHOD COMPARISON HELPERS
# =============================================================================


def _build_bonferroni_method_summary(
    results: MultipleTestingResults, n_models: int
) -> Dict[str, Any]:
    """Build summary dict for Bonferroni correction results."""
    return {
        'significant_models': len(results.significant_models),
        'corrected_alpha': results.corrected_alpha,
        'rejection_rate': len(results.rejected_models) / n_models,
        'statistical_power': results.statistical_power,
        'interpretation': 'Most conservative - controls family-wise error rate'
    }


def _build_fdr_method_summary(
    results: MultipleTestingResults, n_models: int
) -> Dict[str, Any]:
    """Build summary dict for FDR correction results."""
    return {
        'significant_models': len(results.significant_models),
        'corrected_alpha': results.corrected_alpha,
        'rejection_rate': len(results.rejected_models) / n_models,
        'statistical_power': results.statistical_power,
        'interpretation': 'Moderate - controls false discovery rate'
    }


def _build_reduced_space_method_summary(
    reduced_space: Dict[str, Any]
) -> Dict[str, Any]:
    """Build summary dict for reduced search space results."""
    return {
        'combinations_tested': reduced_space['n_combinations'],
        'reduction_factor': reduced_space['reduction_factor'],
        'multiple_testing_eliminated': reduced_space['statistical_properties']['no_multiple_testing_correction_needed'],
        'effective_alpha': reduced_space['statistical_properties']['effective_alpha'],
        'interpretation': 'Most liberal - uses domain knowledge to avoid multiple testing'
    }


# =============================================================================
# RECOMMENDATION GENERATORS
# =============================================================================


def _recommend_conservative_approach(
    bonf_results: MultipleTestingResults
) -> str:
    """Generate conservative (Bonferroni) recommendation."""
    n_significant = len(bonf_results.significant_models)
    if n_significant >= 3:
        return (
            "Use Bonferroni correction - provides strongest statistical rigor "
            f"with {n_significant} significant models"
        )
    return f"Bonferroni too stringent - only {n_significant} models remain"


def _recommend_liberal_approach(reduced_space: Dict[str, Any]) -> str:
    """Generate liberal (reduced search space) recommendation."""
    n_combinations = reduced_space['n_combinations']
    if n_combinations <= 50:
        return (
            f"Use reduced search space - {n_combinations} "
            "combinations eliminate multiple testing concerns"
        )
    return (
        f"Reduced search space still requires correction "
        f"({n_combinations} combinations)"
    )


def _recommend_primary_approach(
    fdr_results: MultipleTestingResults, reduced_space: Dict[str, Any]
) -> str:
    """Generate primary recommendation based on all methods."""
    if len(fdr_results.significant_models) >= 5:
        return "FDR correction recommended for optimal balance"
    if reduced_space['statistical_properties']['no_multiple_testing_correction_needed']:
        return "Reduced search space recommended for simplicity"
    return "Bonferroni correction recommended for maximum rigor"


def _generate_method_recommendations(
    bonf_results: MultipleTestingResults,
    fdr_results: MultipleTestingResults,
    reduced_space: Dict[str, Any],
    alpha: float
) -> Dict[str, str]:
    """
    Generate business recommendations for method selection.

    Parameters
    ----------
    bonf_results : MultipleTestingResults
        Bonferroni correction results
    fdr_results : MultipleTestingResults
        FDR correction results
    reduced_space : Dict[str, Any]
        Reduced search space results
    alpha : float
        Original significance level

    Returns
    -------
    Dict[str, str]
        Method recommendations with rationale
    """
    return {
        'conservative': _recommend_conservative_approach(bonf_results),
        'balanced': (
            f"Use FDR correction - good balance with "
            f"{len(fdr_results.significant_models)} significant models "
            f"(FDR = {alpha})"
        ),
        'liberal': _recommend_liberal_approach(reduced_space),
        'primary': _recommend_primary_approach(fdr_results, reduced_space)
    }


# =============================================================================
# METHOD COMPARISON
# =============================================================================


def compare_correction_methods(
    model_results: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Compare all three multiple testing correction approaches (Bonferroni, FDR, reduced space)."""
    comparison_results: Dict[str, Any] = {
        'comparison_timestamp': datetime.now().isoformat(),
        'original_model_count': len(model_results),
        'methods': {}
    }

    try:
        # Apply all three methods and build summaries
        bonf_results = apply_bonferroni_correction(model_results, alpha=alpha)
        comparison_results['methods']['bonferroni'] = _build_bonferroni_method_summary(
            bonf_results, len(model_results)
        )

        fdr_results = apply_fdr_correction(model_results, alpha=alpha)
        comparison_results['methods']['fdr_bh'] = _build_fdr_method_summary(
            fdr_results, len(model_results)
        )

        reduced_space = create_reduced_search_space(
            candidate_features=list(range(12)), max_combinations=100
        )
        comparison_results['methods']['reduced_space'] = _build_reduced_space_method_summary(
            reduced_space
        )

        # Business recommendations
        comparison_results['recommendations'] = _generate_method_recommendations(
            bonf_results, fdr_results, reduced_space, alpha
        )

    except Exception as e:
        comparison_results['error'] = str(e)
        comparison_results['comparison_failed'] = True

    return comparison_results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Shared types
    'MultipleTestingResults',

    # Bonferroni correction
    'apply_bonferroni_correction',

    # FDR correction
    'apply_fdr_correction',

    # Search space reduction
    'create_reduced_search_space',

    # Method comparison
    'compare_correction_methods',
]
