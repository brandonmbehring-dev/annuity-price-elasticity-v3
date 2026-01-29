"""
Environment Setup Module for Feature Selection Pipeline.

This module provides atomic functions for environment initialization, import management,
and configuration setup following CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Decompose notebook_interface.py environment setup functions (150-200 lines)
Status: MANDATORY (decomposition of 2,274-line module)
Priority: HIGH (code organization and maintainability)

Key Functions:
- setup_feature_selection_environment(): Main environment setup orchestration
- import_core_libraries(): Core Python library imports
- import_atomic_functions(): Feature selection function imports
- import_mlflow_integration(): MLflow integration setup
- configure_visualization_environment(): Plotting configuration

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
import sys
import os


def import_core_libraries() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Import core Python libraries for feature selection notebooks."""
    try:
        import pandas as pd
        import numpy as np
        import warnings
        from datetime import datetime
        from itertools import combinations
        import statsmodels.formula.api as smf
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        import sys
        import os
        import json

        # Status constants for consistent messaging
        constants = {
            'SUCCESS': "SUCCESS:",
            'WARNING': "WARNING:",
            'ERROR': "ERROR:"
        }

        # Core library imports for notebook use
        imports = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'stats': stats,
            'warnings': warnings,
            'datetime': datetime,
            'combinations': combinations,
            'smf': smf,
            'sys': sys,
            'os': os,
            'json': json
        }

        # Add path for atomic functions
        if '../src' not in sys.path:
            sys.path.insert(0, '../src')

        return imports, constants

    except Exception as e:
        raise ImportError(
            f"CRITICAL: Core library import failed: {e}. "
            f"Business impact: Feature selection environment cannot be initialized. "
            f"Required action: Check Python environment and library installations."
        ) from e


def import_atomic_functions(imports: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Import feature selection atomic functions. Returns (enhanced_imports, success_status)."""
    try:
        # Defensive imports for atomic functions
        from features.selection.aic_engine import evaluate_aic_combinations
        from features.selection.constraints_engine import apply_economic_constraints
        from features.selection.bootstrap_engine import run_bootstrap_stability
        from features.selection.data_preprocessing import prepare_analysis_dataset, validate_feature_availability
        from features.selection.selection_types import build_feature_selection_configs

        # Add atomic functions to imports
        imports.update({
            'evaluate_aic_combinations': evaluate_aic_combinations,
            'apply_economic_constraints': apply_economic_constraints,
            'run_bootstrap_stability': run_bootstrap_stability,
            'prepare_analysis_dataset': prepare_analysis_dataset,
            'validate_feature_availability': validate_feature_availability,
            'build_feature_selection_configs': build_feature_selection_configs
        })

        return imports, True

    except ImportError as e:
        warnings.warn(f"Atomic functions import failed: {e}")
        return imports, False

    except Exception as e:
        raise ImportError(
            f"CRITICAL: Atomic function import failed: {e}. "
            f"Business impact: Advanced feature selection capabilities unavailable. "
            f"Required action: Check atomic function module structure."
        ) from e


def import_mlflow_integration(imports: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Import MLflow integration with error handling.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: MLflow integration imports only.

    Parameters
    ----------
    imports : Dict[str, Any]
        Base imports dictionary to enhance

    Returns
    -------
    Tuple[Dict[str, Any], bool]
        (enhanced_imports_dict, mlflow_available_status)
    """
    try:
        # Defensive imports for MLflow integration
        from src.config.mlflow_config import (
            setup_environment_for_notebooks,
            setup_mlflow_experiment,
            safe_mlflow_log_param,
            safe_mlflow_log_metric,
            end_mlflow_experiment
        )

        # Add MLflow functions to imports
        imports.update({
            'setup_environment_for_notebooks': setup_environment_for_notebooks,
            'setup_mlflow_experiment': setup_mlflow_experiment,
            'safe_mlflow_log_param': safe_mlflow_log_param,
            'safe_mlflow_log_metric': safe_mlflow_log_metric,
            'end_mlflow_experiment': end_mlflow_experiment
        })

        return imports, True

    except ImportError:
        # MLflow not available - continue without tracking
        warnings.warn("MLflow integration not available - continuing without experiment tracking")
        return imports, False

    except Exception as e:
        warnings.warn(f"MLflow setup failed: {e}")
        return imports, False


def configure_visualization_environment() -> Tuple[Dict[str, Any], bool, bool]:
    """Configure visualization environment with professional styling.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Visualization configuration only.

    Returns
    -------
    Tuple[Dict[str, Any], bool, bool]
        (figure_config, jupyter_available, matplotlib_available)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Professional figure configuration
        fig_config = {
            'fig_width': 12,
            'fig_height': 8
        }

        # Apply professional styling
        plt.style.use('default')
        sns.set_theme(style="whitegrid", palette="deep")

        # Check for Jupyter display capabilities
        try:
            from IPython.display import display, HTML, Markdown
            jupyter_available = True
        except ImportError:
            jupyter_available = False
            warnings.warn("Jupyter display functions not available")

        return fig_config, jupyter_available, True

    except ImportError:
        warnings.warn("Matplotlib/Seaborn not available")
        return {}, False, False

    except Exception as e:
        warnings.warn(f"Visualization setup failed: {e}")
        return {}, False, False


def initialize_mlflow_environment(imports: Dict[str, Any], mlflow_available: bool) -> bool:
    """Initialize MLflow environment for experiment tracking.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: MLflow environment initialization only.

    Parameters
    ----------
    imports : Dict[str, Any]
        Imports dictionary containing MLflow functions
    mlflow_available : bool
        Whether MLflow integration was successfully imported

    Returns
    -------
    bool
        MLflow initialization success status
    """
    if not mlflow_available:
        return False

    try:
        # Initialize MLflow environment if available
        setup_environment_for_notebooks = imports.get('setup_environment_for_notebooks')
        if setup_environment_for_notebooks:
            setup_environment_for_notebooks()
            return True
        else:
            warnings.warn("MLflow setup function not available")
            return False

    except Exception as e:
        warnings.warn(f"MLflow environment initialization failed: {e}")
        return False


def setup_feature_selection_environment() -> Dict[str, Any]:
    """Orchestrate comprehensive environment setup for feature selection."""
    try:
        # Step 1: Import core libraries
        imports, constants = import_core_libraries()

        # Step 2: Import atomic functions
        imports, atomic_success = import_atomic_functions(imports)

        # Step 3: Import MLflow integration
        imports, mlflow_available = import_mlflow_integration(imports)

        # Step 4: Configure visualization
        fig_config, jupyter_available, matplotlib_available = configure_visualization_environment()

        # Step 5: Initialize MLflow if available
        mlflow_initialized = initialize_mlflow_environment(imports, mlflow_available)

        # Compile comprehensive environment status
        environment_status = {
            'atomic_functions': atomic_success,
            'mlflow_available': mlflow_available,
            'mlflow_initialized': mlflow_initialized,
            'jupyter_available': jupyter_available,
            'matplotlib_available': matplotlib_available
        }

        return {
            'constants': constants,
            'imports': imports,
            'status': environment_status,
            'fig_config': fig_config
        }

    except Exception as e:
        raise RuntimeError(
            f"CRITICAL: Feature selection environment setup failed: {e}. "
            f"Business impact: Cannot proceed with feature selection analysis. "
            f"Required action: Fix environment configuration and retry setup."
        ) from e