"""
Feature Selection Validation and Convenience Module.

This module handles validation, comparison, and convenience functions for
feature selection notebooks. Part of Phase 6.1 module split.

Responsibilities:
- Dual validation stability analysis
- Side-by-side result comparison
- Quick and production convenience functions

Module Architecture (Phase 6.1 Split):
- interface_environment.py: Environment and import setup
- interface_config.py: Configuration building and feature flags
- interface_execution.py: Pipeline execution and error handling
- interface_validation.py: Validation, comparison, convenience functions (this file)
- interface_dashboard.py: Dashboard/stability/visualization functions
- interface_display.py: Display/formatting/HTML report functions
- interface_export.py: Export/MLflow/DVC functions
- notebook_interface.py: Public API orchestrator
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.features.selection_types import FeatureSelectionResults
from .interface_config import FEATURE_FLAGS
from .interface_display import _display_comparison_results
from .interface_export import save_dual_validation_results


# =============================================================================
# DEFENSIVE IMPORTS
# =============================================================================

try:
    from src.features.selection.stability.bootstrap_stability_analysis import (
        run_advanced_stability_analysis,
    )
    from .interface_display import display_dual_validation_results as _display_dual_validation_results

    ATOMIC_FUNCTIONS_AVAILABLE = True
except ImportError:
    ATOMIC_FUNCTIONS_AVAILABLE = False
    warnings.warn(
        "Atomic functions not available - falling back to legacy implementation"
    )


# =============================================================================
# DUAL VALIDATION HELPERS
# =============================================================================


def _validate_dual_analysis_inputs(bootstrap_results: List[Any]) -> None:
    """Validate inputs for dual validation analysis."""
    if not ATOMIC_FUNCTIONS_AVAILABLE:
        raise ImportError(
            "Dual validation analysis requires atomic functions to be available"
        )
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for dual validation analysis")


def _print_dual_analysis_header(n_models: int) -> None:
    """Print header for dual validation stability analysis."""
    print("=== DUAL VALIDATION STABILITY ANALYSIS ===")
    print("6-Metric System: AIC (in/out) + Adjusted R-squared (in/out) + MAPE (in/out)")
    print(f"Analyzing {n_models} models with dual validation framework\n")


# =============================================================================
# DUAL VALIDATION ANALYSIS
# =============================================================================


def run_dual_validation_stability_analysis(
    bootstrap_results: List[Any],
    display_results: bool = True,
    save_results: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run dual validation stability analysis.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results
    display_results : bool, default=True
        Whether to display results
    save_results : bool, default=False
        Whether to save results to file
    output_path : Optional[str]
        Path to save results if save_results is True

    Returns
    -------
    Dict[str, Any]
        Analysis results dictionary
    """
    _validate_dual_analysis_inputs(bootstrap_results)
    _print_dual_analysis_header(len(bootstrap_results))

    try:
        summary = run_advanced_stability_analysis(bootstrap_results)

        if display_results:
            _display_dual_validation_results(summary)
        if save_results and output_path:
            save_dual_validation_results(summary, output_path)

        return summary

    except Exception as e:
        print(f"ERROR: Dual validation stability analysis failed: {e}")
        return {"error": str(e), "success": False}


# =============================================================================
# COMPARISON HELPERS
# =============================================================================


def _run_new_implementation(
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    kwargs: Dict[str, Any],
) -> FeatureSelectionResults:
    """Run the new feature selection implementation for comparison."""
    # Import here to avoid circular imports
    from .interface_execution import run_feature_selection

    return run_feature_selection(
        data=data,
        candidate_features=candidate_features,
        target_variable=target_variable,
        display_results=False,
        return_detailed=True,
        **kwargs,
    )


def _compare_model_counts(
    original_results: pd.DataFrame,
    new_results: FeatureSelectionResults,
    comparison: Dict[str, Any],
) -> None:
    """Compare model counts between original and new implementations."""
    if len(original_results) != len(new_results.all_results):
        comparison["differences"].append(
            f"Model count differs: Original={len(original_results)}, "
            f"New={len(new_results.all_results)}"
        )


def _compute_aic_differences(
    original_results: pd.DataFrame, new_results: FeatureSelectionResults
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute AIC differences for matching models."""
    if (
        "features" not in original_results.columns
        or "aic" not in original_results.columns
    ):
        return [], 0.0

    aic_differences = []
    for _, orig_row in original_results.iterrows():
        orig_features = orig_row["features"]
        matching_new = new_results.all_results[
            new_results.all_results["features"] == orig_features
        ]

        if len(matching_new) > 0:
            orig_aic = orig_row["aic"]
            new_aic = matching_new.iloc[0]["aic"]
            aic_diff = abs(orig_aic - new_aic)

            if aic_diff > 1e-6:
                aic_differences.append(
                    {
                        "features": orig_features,
                        "original_aic": orig_aic,
                        "new_aic": new_aic,
                        "difference": aic_diff,
                    }
                )

    max_diff = max([d["difference"] for d in aic_differences]) if aic_differences else 0.0
    return aic_differences, max_diff


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================


def compare_with_original(
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    original_results: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compare new feature selection results with original implementation."""
    if not FEATURE_FLAGS["ENABLE_VALIDATION"]:
        return {
            "validation_enabled": False,
            "message": "Validation disabled by feature flag",
        }

    print("Running Side-by-Side Validation...")

    new_results = _run_new_implementation(
        data, candidate_features, target_variable, kwargs
    )

    comparison: Dict[str, Any] = {
        "validation_enabled": True,
        "original_models": len(original_results),
        "new_models": len(new_results.all_results),
        "differences": [],
    }

    _compare_model_counts(original_results, new_results, comparison)

    aic_differences, max_aic_diff = _compute_aic_differences(
        original_results, new_results
    )
    comparison["aic_differences"] = aic_differences
    comparison["max_aic_difference"] = max_aic_diff

    comparison["validation_passed"] = (
        len(comparison["differences"]) == 0
        and comparison.get("max_aic_difference", 0) < 1e-6
    )

    _display_comparison_results(comparison)

    return comparison


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_feature_selection(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    max_features: int = 2,
) -> pd.DataFrame:
    """
    Quick feature selection for exploratory analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    target : str
        Target variable
    features : List[str]
        Candidate features
    max_features : int, default=2
        Maximum features per model

    Returns
    -------
    pd.DataFrame
        Results DataFrame
    """
    # Import here to avoid circular imports
    from .interface_execution import run_feature_selection

    return run_feature_selection(
        data=data,
        candidate_features=features,
        target_variable=target,
        max_features=max_features,
        enable_bootstrap=False,
        display_results=True,
        return_detailed=False,
    )


def production_feature_selection(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    max_features: int = 3,
) -> FeatureSelectionResults:
    """
    Production-ready feature selection with full validation.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    target : str
        Target variable
    features : List[str]
        Candidate features
    max_features : int, default=3
        Maximum features per model

    Returns
    -------
    FeatureSelectionResults
        Complete results object
    """
    # Import here to avoid circular imports
    from .interface_execution import run_feature_selection

    return run_feature_selection(
        data=data,
        candidate_features=features,
        target_variable=target,
        max_features=max_features,
        enable_bootstrap=True,
        bootstrap_samples=200,
        enable_constraints=True,
        display_results=True,
        return_detailed=True,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dual validation helpers
    "_validate_dual_analysis_inputs",
    "_print_dual_analysis_header",
    # Dual validation analysis
    "run_dual_validation_stability_analysis",
    # Comparison helpers
    "_run_new_implementation",
    "_compare_model_counts",
    "_compute_aic_differences",
    # Comparison function
    "compare_with_original",
    # Convenience functions
    "quick_feature_selection",
    "production_feature_selection",
]
