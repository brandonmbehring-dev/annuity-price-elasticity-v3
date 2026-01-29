"""
Feature Selection Environment Setup Module.

This module handles environment initialization and import setup for
feature selection notebooks. Part of Phase 6.1 module split.

Responsibilities:
- Core library imports for notebooks
- Atomic function imports with error handling
- MLflow integration setup
- Visualization environment configuration

Module Architecture (Phase 6.1 Split):
- interface_environment.py: Environment and import setup (this file)
- interface_config.py: Configuration building and feature flags
- interface_execution.py: Pipeline execution and error handling
- interface_validation.py: Validation, comparison, convenience functions
- interface_dashboard.py: Dashboard/stability/visualization functions
- interface_display.py: Display/formatting/HTML report functions
- interface_export.py: Export/MLflow/DVC functions
- notebook_interface.py: Public API orchestrator
"""

import warnings
from typing import Any, Dict, Tuple


# =============================================================================
# STATUS CONSTANTS
# =============================================================================


def _get_status_constants() -> Dict[str, str]:
    """Return status message constants for notebook output."""
    return {"SUCCESS": "SUCCESS:", "WARNING": "WARNING:", "ERROR": "ERROR:"}


# =============================================================================
# CORE LIBRARY IMPORTS
# =============================================================================


def _build_core_imports_dict() -> Dict[str, Any]:
    """Build dictionary of core library imports for notebook use."""
    import json
    import os
    import sys
    from datetime import datetime
    from itertools import combinations

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.formula.api as smf
    from scipy import stats

    return {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "stats": stats,
        "warnings": warnings,
        "datetime": datetime,
        "combinations": combinations,
        "smf": smf,
        "sys": sys,
        "os": os,
        "json": json,
    }


def _import_core_libraries() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Import core Python libraries for feature selection notebooks.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (imports_dict, constants_dict)
    """
    import sys

    imports = _build_core_imports_dict()
    constants = _get_status_constants()

    # Add path for atomic functions
    if "../src" not in sys.path:
        sys.path.insert(0, "../src")

    return imports, constants


# =============================================================================
# ATOMIC FUNCTION IMPORTS
# =============================================================================


def _import_atomic_functions(imports: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Import feature selection atomic functions with error handling.

    Parameters
    ----------
    imports : Dict[str, Any]
        Existing imports dictionary to update

    Returns
    -------
    Tuple[Dict[str, Any], bool]
        (updated_imports_dict, import_success_status)
    """
    try:
        from src.features.selection.engines.aic_engine import (
            calculate_aic_for_features,
            evaluate_aic_combinations,
        )
        from src.features.selection.engines.bootstrap_engine import run_bootstrap_stability
        from src.features.selection.engines.constraints_engine import apply_economic_constraints
        from src.features.selection.support.data_preprocessing import (
            prepare_analysis_dataset,
            validate_feature_availability,
        )
        from src.features.selection_types import build_feature_selection_configs

        imports.update(
            {
                "evaluate_aic_combinations": evaluate_aic_combinations,
                "calculate_aic_for_features": calculate_aic_for_features,
                "apply_economic_constraints": apply_economic_constraints,
                "run_bootstrap_stability": run_bootstrap_stability,
                "prepare_analysis_dataset": prepare_analysis_dataset,
                "validate_feature_availability": validate_feature_availability,
                "build_feature_selection_configs": build_feature_selection_configs,
            }
        )
        return imports, True

    except ImportError:
        warnings.warn(
            "Atomic functions not available - falling back to legacy implementation"
        )
        return imports, False


# =============================================================================
# MLFLOW INTEGRATION
# =============================================================================


def _import_mlflow_integration(imports: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Import MLflow integration functions with error handling.

    Parameters
    ----------
    imports : Dict[str, Any]
        Existing imports dictionary to update

    Returns
    -------
    Tuple[Dict[str, Any], bool]
        (updated_imports_dict, import_success_status)
    """
    try:
        from src.config.mlflow_config import (
            end_mlflow_experiment,
            safe_mlflow_log_metric,
            safe_mlflow_log_param,
            setup_environment_for_notebooks,
            setup_mlflow_experiment,
        )

        imports.update(
            {
                "setup_environment_for_notebooks": setup_environment_for_notebooks,
                "setup_mlflow_experiment": setup_mlflow_experiment,
                "safe_mlflow_log_param": safe_mlflow_log_param,
                "safe_mlflow_log_metric": safe_mlflow_log_metric,
                "end_mlflow_experiment": end_mlflow_experiment,
            }
        )
        return imports, True

    except ImportError as e:
        warnings.warn(f"MLflow import failed: {e}")
        return imports, False


# =============================================================================
# VISUALIZATION ENVIRONMENT
# =============================================================================


def _configure_visualization_environment() -> Tuple[Dict[str, Any], bool, bool]:
    """
    Configure visualization environment (matplotlib, seaborn).

    Returns
    -------
    Tuple[Dict[str, Any], bool, bool]
        (fig_config_dict, seaborn_success, mlflow_init_success)
    """
    import seaborn as sns

    # Suppress specific seaborn configuration warnings
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        module='seaborn'
    )
    # Justification: seaborn UserWarnings for theme configuration are cosmetic

    fig_config = {"fig_width": 10, "fig_height": 8}

    seaborn_success = True
    try:
        sns.set_theme(style="whitegrid", palette="deep")
    except Exception as e:
        seaborn_success = False
        warnings.warn(f"Seaborn configuration failed: {e}")

    return fig_config, seaborn_success, False


def _initialize_mlflow_environment(
    imports: Dict[str, Any], mlflow_available: bool
) -> bool:
    """
    Initialize MLflow environment if available.

    Parameters
    ----------
    imports : Dict[str, Any]
        Imports dictionary containing MLflow functions
    mlflow_available : bool
        Whether MLflow imports were successful

    Returns
    -------
    bool
        MLflow initialization success status
    """
    if not mlflow_available:
        return False

    try:
        imports["setup_environment_for_notebooks"]()
        return True
    except Exception as e:
        warnings.warn(f"MLflow environment setup failed: {e}")
        return False


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def setup_feature_selection_environment() -> Dict[str, Any]:
    """
    Setup feature selection environment with all imports and constants.

    Atomic function: Orchestrates environment setup with single responsibility.
    Mathematical equivalence guarantee: produces identical environment to original function.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing imports, constants, config, and status information
    """
    imports, constants = _import_core_libraries()
    imports, atomic_functions_available = _import_atomic_functions(imports)
    imports, mlflow_available = _import_mlflow_integration(imports)
    fig_config, seaborn_configured, _ = _configure_visualization_environment()
    mlflow_initialized = _initialize_mlflow_environment(imports, mlflow_available)

    setup_data = {
        "imports": imports,
        "constants": constants,
        "fig_config": fig_config,
        "status": {
            "atomic_functions_available": atomic_functions_available,
            "mlflow_available": mlflow_available,
            "mlflow_initialized": mlflow_initialized,
            "seaborn_configured": seaborn_configured,
            "setup_complete": True,
        },
    }

    return setup_data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Status constants
    "_get_status_constants",
    # Core imports
    "_build_core_imports_dict",
    "_import_core_libraries",
    # Atomic function imports
    "_import_atomic_functions",
    # MLflow integration
    "_import_mlflow_integration",
    # Visualization
    "_configure_visualization_environment",
    "_initialize_mlflow_environment",
    # Main orchestrator
    "setup_feature_selection_environment",
]
