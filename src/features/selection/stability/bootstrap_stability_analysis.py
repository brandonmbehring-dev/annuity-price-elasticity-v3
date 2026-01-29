"""
Advanced Bootstrap Stability Analysis Module for Feature Selection.

Consolidates advanced stability analysis functions (Win Rate + Information Ratio)
from cells 25-28 of the refactored feature selection notebook.

Module Architecture (Phase 6.3f Split):
- stability_win_rate.py: Win rate analysis
- stability_ir.py: Information ratio analysis
- stability_visualizations.py: Plotting functions
- bootstrap_stability_analysis.py: Orchestrator + public API (this file)

Key Functions:
- run_advanced_stability_analysis: Main entry point for advanced analysis
- calculate_bootstrap_win_rates: Bootstrap Win Rate Analysis
- calculate_information_ratio_analysis: Information Ratio Analysis
- create_advanced_visualizations: Unified visualization creation
"""

from typing import Dict, List, Any, Optional

# Defensive imports for bootstrap types
try:
    from src.features.selection_types import BootstrapResult
    BOOTSTRAP_TYPES_AVAILABLE = True
except ImportError:
    from dataclasses import dataclass
    BOOTSTRAP_TYPES_AVAILABLE = False

    @dataclass
    class BootstrapResult:
        model_name: str
        model_features: str
        bootstrap_aics: List[float]
        bootstrap_r2_values: List[float]
        original_aic: float
        original_r2: float
        aic_stability_coefficient: float
        r2_stability_coefficient: float
        confidence_intervals: Dict[str, Dict[str, float]]
        successful_fits: int
        total_attempts: int
        stability_assessment: str


# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES
# =============================================================================

# Win rate analysis module
from src.features.selection.stability.stability_win_rate import (
    _build_bootstrap_aic_matrix,
    _calculate_win_counts,
    _build_win_rate_results,
    _print_win_rate_table,
    _print_win_rate_insights,
    calculate_bootstrap_win_rates,
)

# Information ratio analysis module
from src.features.selection.stability.stability_ir import (
    _prepare_ir_benchmark_stats,
    _compute_ratio_safe,
    _compute_sortino_ratio,
    _calculate_single_model_ir_metrics,
    _classify_risk_adjusted,
    _print_ir_results_table,
    _print_ir_insights,
    _print_best_ir_model,
    calculate_information_ratio_analysis,
)

# Visualization module
from src.features.selection.stability.stability_visualizations import (
    _plot_win_rate_bar_chart,
    _plot_win_rate_distribution,
    _plot_ir_rankings,
    _plot_risk_return_scatter,
    _plot_consistency_scatter,
    _plot_excess_aic_distributions,
    _create_win_rate_figure,
    _create_ir_figure,
    create_advanced_visualizations,
)


# =============================================================================
# ORCHESTRATION UTILITIES
# =============================================================================


def _validate_analysis_inputs(bootstrap_results: List[BootstrapResult], config: Dict[str, Any]) -> None:
    """
    Validate inputs for advanced stability analysis.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results from core analysis
    config : Dict[str, Any]
        Analysis configuration

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results available for advanced stability analysis")
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")


def _extract_analysis_config(config: Dict[str, Any], n_results: int) -> tuple:
    """
    Extract and parse analysis configuration parameters.

    Parameters
    ----------
    config : Dict[str, Any]
        Analysis configuration
    n_results : int
        Number of bootstrap results available

    Returns
    -------
    tuple
        (enable_win_rate, enable_ir, create_visualizations, n_models)
    """
    enable_win_rate = config.get('enable_win_rate_analysis', True)
    enable_ir = config.get('enable_information_ratio', True)
    create_visualizations = config.get('create_visualizations', True)
    n_models = min(config.get('models_to_analyze', 15), n_results)
    return enable_win_rate, enable_ir, create_visualizations, n_models


def _print_analysis_header(n_models: int, enable_win_rate: bool, enable_ir: bool) -> None:
    """
    Print analysis header with configuration summary.

    Parameters
    ----------
    n_models : int
        Number of models to analyze
    enable_win_rate : bool
        Whether win rate analysis is enabled
    enable_ir : bool
        Whether IR analysis is enabled
    """
    print(f"Starting advanced stability analysis...")
    print(f"  Models to analyze: {n_models}")
    print(f"  Win Rate Analysis: {'Enabled' if enable_win_rate else 'Disabled'}")
    print(f"  Information Ratio Analysis: {'Enabled' if enable_ir else 'Disabled'}")


def _package_analysis_results(
    win_rate_results: Optional[List[Dict[str, Any]]],
    ir_results: Optional[List[Dict[str, Any]]],
    visualizations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Package analysis results into return dictionary.

    Parameters
    ----------
    win_rate_results : Optional[List[Dict[str, Any]]]
        Win rate analysis results
    ir_results : Optional[List[Dict[str, Any]]]
        Information ratio results
    visualizations : Dict[str, Any]
        Visualization figures

    Returns
    -------
    Dict[str, Any]
        Packaged results dictionary
    """
    results = {}
    if win_rate_results:
        results['win_rate_results'] = win_rate_results
    if ir_results:
        results['information_ratio_results'] = ir_results
    results['visualizations'] = visualizations
    return results


# =============================================================================
# ANALYSIS COMPONENT RUNNERS
# =============================================================================


def _run_win_rate_analysis(
    bootstrap_results: List[BootstrapResult], n_models: int, enable_win_rate: bool
) -> Optional[List[Dict[str, Any]]]:
    """
    Run Win Rate Analysis component.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results
    n_models : int
        Number of models to analyze
    enable_win_rate : bool
        Whether to run win rate analysis

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        Win rate results or None

    Raises
    ------
    RuntimeError
        If analysis fails
    """
    if not enable_win_rate:
        return None
    try:
        results = calculate_bootstrap_win_rates(bootstrap_results[:n_models])
        print(f"  Win Rate Analysis complete: {len(results)} models analyzed")
        return results
    except Exception as e:
        raise RuntimeError(f"Win Rate Analysis failed: {str(e)}") from e


def _run_information_ratio_analysis(
    bootstrap_results: List[BootstrapResult], n_models: int, enable_ir: bool
) -> Optional[List[Dict[str, Any]]]:
    """
    Run Information Ratio Analysis component.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results
    n_models : int
        Number of models to analyze
    enable_ir : bool
        Whether to run IR analysis

    Returns
    -------
    Optional[List[Dict[str, Any]]]
        IR results or None

    Raises
    ------
    RuntimeError
        If analysis fails
    """
    if not enable_ir:
        return None
    try:
        results = calculate_information_ratio_analysis(bootstrap_results[:n_models])
        print(f"  Information Ratio Analysis complete: {len(results)} models analyzed")
        return results
    except Exception as e:
        raise RuntimeError(f"Information Ratio Analysis failed: {str(e)}") from e


def _create_visualizations_if_enabled(
    win_rate_results: Optional[List[Dict[str, Any]]],
    ir_results: Optional[List[Dict[str, Any]]],
    create_vis: bool,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create visualizations if enabled and results available.

    Parameters
    ----------
    win_rate_results : Optional[List[Dict[str, Any]]]
        Win rate analysis results
    ir_results : Optional[List[Dict[str, Any]]]
        Information ratio results
    create_vis : bool
        Whether to create visualizations
    config : Dict[str, Any]
        Configuration parameters

    Returns
    -------
    Dict[str, Any]
        Visualizations dictionary
    """
    if not create_vis or not (win_rate_results or ir_results):
        return {}

    try:
        visualizations = create_advanced_visualizations(
            win_rate_results=win_rate_results,
            ir_results=ir_results,
            config=config
        )
        print(f"  Advanced visualizations created: {len(visualizations) if visualizations else 0} figures")
        return visualizations
    except Exception as e:
        print(f"WARNING: Advanced visualization creation failed: {str(e)}")
        return {}


# =============================================================================
# PUBLIC API
# =============================================================================


def run_advanced_stability_analysis(bootstrap_results: List[BootstrapResult],
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run advanced stability analysis: Win Rate + Information Ratio.

    Consolidates cells 25-28 into unified analysis function following DRY principles.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results from core analysis
    config : Dict[str, Any]
        Analysis configuration with advanced analysis parameters

    Returns
    -------
    Dict[str, Any]
        Comprehensive results containing:
        - win_rate_results: Win Rate Analysis results
        - information_ratio_results: Information Ratio Analysis results
        - visualizations: Dict[str, matplotlib.Figure] (if enabled)

    Raises
    ------
    ValueError
        If bootstrap results are empty or configuration is invalid
    RuntimeError
        If advanced analysis fails unexpectedly
    """
    _validate_analysis_inputs(bootstrap_results, config)

    # Extract configuration and print header
    enable_win_rate, enable_ir, create_vis, n_models = _extract_analysis_config(
        config, len(bootstrap_results)
    )
    _print_analysis_header(n_models, enable_win_rate, enable_ir)

    # Run analyses
    win_rate_results = _run_win_rate_analysis(bootstrap_results, n_models, enable_win_rate)
    ir_results = _run_information_ratio_analysis(bootstrap_results, n_models, enable_ir)
    visualizations = _create_visualizations_if_enabled(
        win_rate_results, ir_results, create_vis, config
    )

    # Package and return results
    results = _package_analysis_results(win_rate_results, ir_results, visualizations)
    print(f"SUCCESS: Advanced stability analysis complete")
    return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Re-exported from stability_win_rate.py
    '_build_bootstrap_aic_matrix',
    '_calculate_win_counts',
    '_build_win_rate_results',
    '_print_win_rate_table',
    '_print_win_rate_insights',
    'calculate_bootstrap_win_rates',

    # Re-exported from stability_ir.py
    '_prepare_ir_benchmark_stats',
    '_compute_ratio_safe',
    '_compute_sortino_ratio',
    '_calculate_single_model_ir_metrics',
    '_classify_risk_adjusted',
    '_print_ir_results_table',
    '_print_ir_insights',
    '_print_best_ir_model',
    'calculate_information_ratio_analysis',

    # Re-exported from stability_visualizations.py
    '_plot_win_rate_bar_chart',
    '_plot_win_rate_distribution',
    '_plot_ir_rankings',
    '_plot_risk_return_scatter',
    '_plot_consistency_scatter',
    '_plot_excess_aic_distributions',
    '_create_win_rate_figure',
    '_create_ir_figure',
    'create_advanced_visualizations',

    # Orchestration (this file)
    '_validate_analysis_inputs',
    '_extract_analysis_config',
    '_print_analysis_header',
    '_package_analysis_results',
    '_run_win_rate_analysis',
    '_run_information_ratio_analysis',
    '_create_visualizations_if_enabled',

    # Main public API
    'run_advanced_stability_analysis',
]
