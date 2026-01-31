"""
Dual Validation Module for Feature Selection Pipeline.

This module provides dual validation stability analysis using a 6-metric system:
AIC (in/out) + Adjusted R² (in/out) + MAPE (in/out).

Purpose: Extract dual validation functionality from notebook_interface.py
Status: MANDATORY (decomposition of 2,356-line module)
Priority: HIGH (code organization and maintainability)

Key Functions:
- run_dual_validation_stability_analysis(): Main dual validation entry point
- create_dual_validation_config(): Create dual validation configuration
- Display helpers for notebook output
- Results export functionality

Mathematical Equivalence: All functions maintain identical results to original
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Defensive imports for notebook display
try:
    from IPython.display import display, HTML
    DISPLAY_AVAILABLE = True
except ImportError:
    DISPLAY_AVAILABLE = False
    def display(x: Any) -> None:
        """Fallback display function when IPython unavailable."""
        print(x)
    def HTML(x: str) -> str:
        """Fallback HTML wrapper when IPython unavailable."""
        return x

# Import advanced stability analysis
try:
    from src.features.selection.stability.bootstrap_stability_analysis import (
        run_advanced_stability_analysis
    )
    STABILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    STABILITY_ANALYSIS_AVAILABLE = False
    def run_advanced_stability_analysis(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Raise ImportError when stability analysis module unavailable."""
        raise ImportError("Bootstrap stability analysis not available")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _validate_dual_analysis_inputs(bootstrap_results: List[Any]) -> None:
    """Validate inputs for dual validation analysis.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap results to validate

    Raises
    ------
    ImportError
        If stability analysis module is not available
    ValueError
        If bootstrap_results is empty
    """
    if not STABILITY_ANALYSIS_AVAILABLE:
        raise ImportError("Dual validation analysis requires bootstrap stability analysis module")
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for dual validation analysis")


def _print_dual_analysis_header(n_models: int) -> None:
    """Print header for dual validation stability analysis.

    Parameters
    ----------
    n_models : int
        Number of models being analyzed
    """
    print("=== DUAL VALIDATION STABILITY ANALYSIS ===")
    print("6-Metric System: AIC (in/out) + Adjusted R² (in/out) + MAPE (in/out)")
    print(f"Analyzing {n_models} models with dual validation framework\n")


# =============================================================================
# DISPLAY FUNCTIONS (Imported from canonical location: interface_display.py)
# =============================================================================

# Import display functions from canonical location to eliminate duplication
from src.features.selection.interface.interface_display import (
    _display_dual_validation_header,
    _display_dual_validation_metadata,
    _display_dual_validation_best_model,
    _display_dual_validation_grade_distribution,
    _display_dual_validation_top_models_table,
    _display_dual_validation_recommendations,
    display_dual_validation_results as _display_dual_validation_results,
)


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def _convert_for_json(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable types.

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        JSON-serializable object
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    else:
        return obj


def _save_dual_validation_results(summary: Dict[str, Any], output_path: str) -> None:
    """Save dual validation results to file.

    Parameters
    ----------
    summary : Dict[str, Any]
        Analysis summary dictionary
    output_path : str
        Path to save results

    Raises
    ------
    RuntimeError
        If save operation fails
    """
    try:
        json_ready_summary = _convert_for_json(summary)

        with open(output_path, 'w') as f:
            json.dump(json_ready_summary, f, indent=2)

        print(f"SUCCESS: Dual validation results saved to {output_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save dual validation results: {e}") from e


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_dual_validation_defaults() -> Dict[str, Any]:
    """Return default configuration for dual validation analysis.

    Returns
    -------
    Dict[str, Any]
        Default configuration dictionary
    """
    return {
        'enable_dual_validation': True,
        'n_bootstrap_samples': 100,
        'win_rate_weight': 0.5,
        'information_ratio_weight': 0.5,
        'min_stability_grade': 'B',
        'out_of_sample_split': 0.3
    }


def _update_configs_for_dual_validation(
    configs: Dict[str, Any],
    dual_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Update pipeline configs with dual validation settings.

    Parameters
    ----------
    configs : Dict[str, Any]
        Existing pipeline configuration
    dual_config : Dict[str, Any]
        Dual validation specific configuration

    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    configs['dual_validation'] = dual_config
    configs['bootstrap_config']['n_bootstrap_samples'] = dual_config.get('n_bootstrap_samples', 100)
    return configs


def create_dual_validation_config(
    n_bootstrap_samples: int = 100,
    win_rate_weight: float = 0.5,
    information_ratio_weight: float = 0.5,
    min_stability_grade: str = 'B',
    out_of_sample_split: float = 0.3
) -> Dict[str, Any]:
    """Create configuration for dual validation analysis.

    Parameters
    ----------
    n_bootstrap_samples : int, default 100
        Number of bootstrap samples per model
    win_rate_weight : float, default 0.5
        Weight for win rate in composite score
    information_ratio_weight : float, default 0.5
        Weight for information ratio in composite score
    min_stability_grade : str, default 'B'
        Minimum acceptable stability grade
    out_of_sample_split : float, default 0.3
        Fraction of data for out-of-sample validation

    Returns
    -------
    Dict[str, Any]
        Dual validation configuration dictionary
    """
    # Validate weights sum to 1.0
    if abs(win_rate_weight + information_ratio_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0. Got win_rate_weight={win_rate_weight}, "
            f"information_ratio_weight={information_ratio_weight}"
        )

    return {
        'enable_dual_validation': True,
        'n_bootstrap_samples': n_bootstrap_samples,
        'win_rate_weight': win_rate_weight,
        'information_ratio_weight': information_ratio_weight,
        'min_stability_grade': min_stability_grade,
        'out_of_sample_split': out_of_sample_split
    }


# =============================================================================
# MAIN API
# =============================================================================

def run_dual_validation_stability_analysis(
    bootstrap_results: List[Any],
    display_results: bool = True,
    save_results: bool = False,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run dual validation stability analysis using a 6-metric system."""
    _validate_dual_analysis_inputs(bootstrap_results)
    _print_dual_analysis_header(len(bootstrap_results))

    try:
        summary = run_advanced_stability_analysis(bootstrap_results)

        if display_results:
            _display_dual_validation_results(summary)
        if save_results and output_path:
            _save_dual_validation_results(summary, output_path)

        return summary

    except Exception as e:
        print(f"ERROR: Dual validation stability analysis failed: {e}")
        return {"error": str(e), "success": False}
