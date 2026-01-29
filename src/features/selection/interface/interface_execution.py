"""
Feature Selection Pipeline Execution Module.

This module handles pipeline execution and error handling for
feature selection notebooks. Part of Phase 6.1 module split.

Responsibilities:
- Pipeline parameter preparation
- Atomic pipeline execution
- Result formatting
- Error handling and fallback creation

Module Architecture (Phase 6.1 Split):
- interface_environment.py: Environment and import setup
- interface_config.py: Configuration building and feature flags
- interface_execution.py: Pipeline execution and error handling (this file)
- interface_validation.py: Validation, comparison, convenience functions
- interface_dashboard.py: Dashboard/stability/visualization functions
- interface_display.py: Display/formatting/HTML report functions
- interface_export.py: Export/MLflow/DVC functions
- notebook_interface.py: Public API orchestrator
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.features.selection_types import (
    AICResult,
    BootstrapAnalysisConfig,
    EconomicConstraintConfig,
    FeatureSelectionConfig,
    FeatureSelectionResults,
)
from .interface_config import (
    FEATURE_FLAGS,
    create_feature_selection_config,
)
from .interface_display import display_results_summary


# =============================================================================
# DEFENSIVE IMPORTS
# =============================================================================

try:
    from src.features.selection.pipeline_orchestrator import (
        create_pipeline_summary,
        run_feature_selection_pipeline,
    )

    ATOMIC_FUNCTIONS_AVAILABLE = True
except ImportError:
    ATOMIC_FUNCTIONS_AVAILABLE = False
    warnings.warn(
        "Atomic functions not available - falling back to legacy implementation"
    )


# =============================================================================
# PARAMETER PREPARATION
# =============================================================================


def _prepare_feature_selection_parameters(
    display_results: Optional[bool], enable_bootstrap: Optional[bool]
) -> Tuple[bool, bool]:
    """
    Prepare and resolve feature selection parameters using feature flags.

    Returns
    -------
    Tuple[bool, bool]
        (resolved_display_results, resolved_enable_bootstrap)
    """
    resolved_display_results = (
        display_results
        if display_results is not None
        else FEATURE_FLAGS["AUTO_DISPLAY_RESULTS"]
    )
    resolved_enable_bootstrap = (
        enable_bootstrap
        if enable_bootstrap is not None
        else FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"]
    )

    return resolved_display_results, resolved_enable_bootstrap


def _log_pipeline_status() -> None:
    """Log feature selection pipeline status message."""
    pipeline_type = (
        "Atomic Functions Pipeline"
        if FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] and ATOMIC_FUNCTIONS_AVAILABLE
        else "Legacy Implementation"
    )
    print("Starting Feature Selection Analysis...")
    print(f"Using {pipeline_type}")


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================


def _execute_feature_selection_pipeline(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    bootstrap_config: Optional[BootstrapAnalysisConfig],
    resolved_display_results: bool,
) -> FeatureSelectionResults:
    """Execute the atomic function feature selection pipeline."""
    results = run_feature_selection_pipeline(
        data=data,
        feature_config=feature_config,
        constraint_config=constraint_config,
        bootstrap_config=bootstrap_config,
    )

    if resolved_display_results:
        display_results_summary(results, feature_flags=FEATURE_FLAGS)

    return results


def _format_pipeline_results(
    results: FeatureSelectionResults, return_detailed: bool
) -> Union[pd.DataFrame, FeatureSelectionResults]:
    """Format pipeline results according to user preferences."""
    if return_detailed:
        return results
    else:
        return (
            results.valid_results
            if len(results.valid_results) > 0
            else results.all_results
        )


def _run_atomic_pipeline(
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    max_features: int,
    base_features: Optional[List[str]],
    enable_constraints: bool,
    resolved_enable_bootstrap: bool,
    bootstrap_samples: int,
    random_seed: Optional[int],
    resolved_display_results: bool,
    return_detailed: bool,
) -> Union[pd.DataFrame, FeatureSelectionResults]:
    """Execute atomic pipeline."""
    feature_config, constraint_config, bootstrap_config = create_feature_selection_config(
        candidate_features=candidate_features,
        target_variable=target_variable,
        max_features=max_features,
        base_features=base_features,
        enable_constraints=enable_constraints,
        enable_bootstrap=resolved_enable_bootstrap,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )

    results = _execute_feature_selection_pipeline(
        data=data,
        feature_config=feature_config,
        constraint_config=constraint_config,
        bootstrap_config=bootstrap_config,
        resolved_display_results=resolved_display_results,
    )

    return _format_pipeline_results(results, return_detailed)


# =============================================================================
# ERROR HANDLING
# =============================================================================


def _create_error_aic_result(n_obs: int, error: Exception) -> AICResult:
    """Create an AICResult representing a pipeline error."""
    return AICResult(
        features="ERROR",
        n_features=0,
        aic=np.inf,
        bic=np.inf,
        r_squared=0.0,
        r_squared_adj=0.0,
        coefficients={},
        converged=False,
        n_obs=n_obs,
        error=str(error),
    )


def _create_error_feature_selection_results(
    error_aic: AICResult,
    candidate_features: List[str],
    target_variable: str,
    max_features: int,
    base_features: Optional[List[str]],
    enable_constraints: bool,
) -> FeatureSelectionResults:
    """Create FeatureSelectionResults representing a pipeline error."""
    return FeatureSelectionResults(
        best_model=error_aic,
        all_results=pd.DataFrame(),
        valid_results=pd.DataFrame(),
        total_combinations=0,
        converged_models=0,
        economically_valid_models=0,
        constraint_violations=[],
        feature_config=FeatureSelectionConfig(
            base_features=base_features or [],
            candidate_features=candidate_features,
            max_candidate_features=max_features,
            target_variable=target_variable,
        ),
        constraint_config=EconomicConstraintConfig(enabled=enable_constraints),
    )


def _create_error_fallback_results(
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    max_features: int,
    base_features: Optional[List[str]],
    enable_constraints: bool,
    error: Exception,
    return_detailed: bool,
) -> Union[pd.DataFrame, FeatureSelectionResults]:
    """Create fallback error results for backward compatibility."""
    if not return_detailed:
        return pd.DataFrame()

    error_aic = _create_error_aic_result(len(data), error)
    return _create_error_feature_selection_results(
        error_aic,
        candidate_features,
        target_variable,
        max_features,
        base_features,
        enable_constraints,
    )


def _handle_pipeline_error(
    error: Exception,
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    max_features: int,
    base_features: Optional[List[str]],
    enable_constraints: bool,
    return_detailed: bool,
) -> Union[pd.DataFrame, FeatureSelectionResults]:
    """Log error and create fallback results for backward compatibility."""
    print(f"ERROR: Feature selection failed: {str(error)}")

    return _create_error_fallback_results(
        data=data,
        candidate_features=candidate_features,
        target_variable=target_variable,
        max_features=max_features,
        base_features=base_features,
        enable_constraints=enable_constraints,
        error=error,
        return_detailed=return_detailed,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def run_feature_selection(
    data: pd.DataFrame,
    candidate_features: List[str],
    target_variable: str,
    max_features: int = 3,
    base_features: Optional[List[str]] = None,
    enable_constraints: bool = True,
    enable_bootstrap: Optional[bool] = None,
    bootstrap_samples: int = 100,
    random_seed: Optional[int] = 42,
    display_results: Optional[bool] = None,
    return_detailed: bool = False,
) -> Union[pd.DataFrame, FeatureSelectionResults]:
    """Run feature selection with notebook interface and optional bootstrap analysis."""
    resolved_display_results, resolved_enable_bootstrap = (
        _prepare_feature_selection_parameters(display_results, enable_bootstrap)
    )
    _log_pipeline_status()

    try:
        if FEATURE_FLAGS["USE_ATOMIC_FUNCTIONS"] and ATOMIC_FUNCTIONS_AVAILABLE:
            return _run_atomic_pipeline(
                data=data,
                candidate_features=candidate_features,
                target_variable=target_variable,
                max_features=max_features,
                base_features=base_features,
                enable_constraints=enable_constraints,
                resolved_enable_bootstrap=resolved_enable_bootstrap,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
                resolved_display_results=resolved_display_results,
                return_detailed=return_detailed,
            )
        else:
            raise NotImplementedError(
                "Legacy implementation fallback not available. "
                "Enable atomic functions with set_feature_flag('USE_ATOMIC_FUNCTIONS', True)"
            )
    except Exception as e:
        return _handle_pipeline_error(
            error=e,
            data=data,
            candidate_features=candidate_features,
            target_variable=target_variable,
            max_features=max_features,
            base_features=base_features,
            enable_constraints=enable_constraints,
            return_detailed=return_detailed,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Status check
    "ATOMIC_FUNCTIONS_AVAILABLE",
    # Parameter preparation
    "_prepare_feature_selection_parameters",
    "_log_pipeline_status",
    # Pipeline execution
    "_execute_feature_selection_pipeline",
    "_format_pipeline_results",
    "_run_atomic_pipeline",
    # Error handling
    "_create_error_aic_result",
    "_create_error_feature_selection_results",
    "_create_error_fallback_results",
    "_handle_pipeline_error",
    # Main entry point
    "run_feature_selection",
]
