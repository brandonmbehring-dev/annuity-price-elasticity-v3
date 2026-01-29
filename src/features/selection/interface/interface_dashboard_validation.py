"""
Dashboard Input Validation and Configuration Extraction.

This module contains validation and configuration extraction functions
for the comprehensive stability dashboard.

Module Responsibilities:
- Input validation with fail-fast error handling
- Module import validation
- Configuration extraction and logging
- Stability result processing

Used by: interface_dashboard.py (orchestrator)
"""

from typing import Dict, List, Any, Tuple


def validate_dashboard_inputs(
    bootstrap_results: List[Any], config: Dict[str, Any]
) -> None:
    """
    Validate inputs for comprehensive stability dashboard with fail-fast approach.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results to validate
    config : Dict[str, Any]
        Configuration dictionary to validate

    Raises
    ------
    ValueError
        If bootstrap_results is empty or config is not a dictionary
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results available for comprehensive dashboard")
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")


def import_advanced_stability_analysis() -> Any:
    """
    Import and return the advanced stability analysis function.

    Returns
    -------
    Callable
        The run_advanced_stability_analysis function

    Raises
    ------
    RuntimeError
        If the stability analysis module cannot be imported
    """
    try:
        from src.features.selection.stability.bootstrap_stability_analysis import (
            run_advanced_stability_analysis,
        )

        return run_advanced_stability_analysis
    except ImportError:
        raise RuntimeError(
            "Advanced stability analysis functions not available - "
            "ensure bootstrap_stability_analysis.py is properly installed"
        )


def extract_dashboard_config(
    config: Dict[str, Any], bootstrap_results: List[Any]
) -> Tuple[int, bool, Dict[str, float]]:
    """
    Extract and log dashboard configuration parameters.

    Parameters
    ----------
    config : Dict[str, Any]
        Dashboard configuration
    bootstrap_results : List[Any]
        Bootstrap results for determining model count

    Returns
    -------
    Tuple[int, bool, Dict[str, float]]
        (n_models, create_visualizations, integration_weights)
    """
    n_models = min(config.get("models_to_analyze", 15), len(bootstrap_results))
    create_visualizations = config.get("create_visualizations", True)
    integration_weights = config.get(
        "integration_weights",
        {"win_rate_weight": 0.5, "information_ratio_weight": 0.5},
    )

    print("Generating comprehensive stability dashboard...")
    print(f"  Models to analyze: {n_models}")
    print(f"  Win Rate Weight: {integration_weights['win_rate_weight']*100:.0f}%")
    print(
        f"  Information Ratio Weight: {integration_weights['information_ratio_weight']*100:.0f}%"
    )

    return n_models, create_visualizations, integration_weights


def process_advanced_stability_results(
    advanced_results: Dict[str, Any], results: Dict[str, Any]
) -> None:
    """
    Process win rate and information ratio results from advanced analysis.

    Parameters
    ----------
    advanced_results : Dict[str, Any]
        Results from run_advanced_stability_analysis
    results : Dict[str, Any]
        Results dictionary to update in-place

    Raises
    ------
    RuntimeError
        If win rate or information ratio analysis is missing
    """
    if "win_rate_results" in advanced_results:
        results["win_rate_results"] = advanced_results["win_rate_results"]
        print(
            f"  Win Rate Analysis: {len(advanced_results['win_rate_results'])} models analyzed"
        )
    else:
        raise RuntimeError(
            "Win Rate Analysis not completed in advanced stability analysis"
        )

    if "information_ratio_results" in advanced_results:
        results["information_ratio_results"] = advanced_results[
            "information_ratio_results"
        ]
        print(
            f"  Information Ratio Analysis: {len(advanced_results['information_ratio_results'])} models analyzed"
        )
    else:
        raise RuntimeError(
            "Information Ratio Analysis not completed in advanced stability analysis"
        )
