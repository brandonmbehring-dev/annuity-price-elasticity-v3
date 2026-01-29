"""
Bootstrap Win Rate Analysis for Feature Selection.

This module handles win rate calculations and reporting for bootstrap stability analysis:
- AIC matrix construction from bootstrap results
- Win count calculations across bootstrap samples
- Formatted output and insights

Module Architecture (Phase 6.3f Split):
- stability_win_rate.py: Win rate analysis (this file)
- stability_ir.py: Information ratio analysis
- stability_visualizations.py: Plotting functions
- bootstrap_stability_analysis.py: Orchestrator + public API
"""

import numpy as np
from typing import Dict, List, Any, Tuple

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
# WIN RATE MATRIX CONSTRUCTION
# =============================================================================


def _build_bootstrap_aic_matrix(bootstrap_results: List[BootstrapResult]) -> Tuple[np.ndarray, List[str]]:
    """
    Build AIC matrix for win rate calculation.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (bootstrap_aic_matrix, model_names)
    """
    n_models = len(bootstrap_results)
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)

    bootstrap_aic_matrix = np.zeros((n_bootstrap_samples, n_models))
    model_names = []

    for i in range(n_models):
        bootstrap_aic_matrix[:, i] = bootstrap_results[i].bootstrap_aics
        model_names.append(f"Model {i+1}")

    return bootstrap_aic_matrix, model_names


def _calculate_win_counts(bootstrap_aic_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate win counts for each model.

    Parameters
    ----------
    bootstrap_aic_matrix : np.ndarray
        Matrix of bootstrap AICs (rows=samples, cols=models)

    Returns
    -------
    np.ndarray
        Win counts for each model
    """
    n_bootstrap_samples, n_models = bootstrap_aic_matrix.shape
    win_counts = np.zeros(n_models)

    for sample_idx in range(n_bootstrap_samples):
        sample_aics = bootstrap_aic_matrix[sample_idx, :]
        winner_idx = np.argmin(sample_aics)
        win_counts[winner_idx] += 1

    return win_counts


# =============================================================================
# WIN RATE RESULTS CONSTRUCTION
# =============================================================================


def _build_win_rate_results(
    bootstrap_results: List[BootstrapResult],
    model_names: List[str],
    win_rates: np.ndarray,
    win_counts: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Build comprehensive win rate results.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results
    model_names : List[str]
        Model names
    win_rates : np.ndarray
        Win rates for each model
    win_counts : np.ndarray
        Win counts for each model

    Returns
    -------
    List[Dict[str, Any]]
        Win rate results sorted by performance
    """
    n_models = len(bootstrap_results)
    win_rate_results = []

    for i in range(n_models):
        win_rate_results.append({
            'model': model_names[i],
            'features': bootstrap_results[i].model_features,
            'win_rate_pct': win_rates[i],
            'win_count': int(win_counts[i]),
            'original_aic': bootstrap_results[i].original_aic,
            'median_bootstrap_aic': np.median(bootstrap_results[i].bootstrap_aics)
        })

    win_rate_results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
    return win_rate_results


# =============================================================================
# WIN RATE DISPLAY FUNCTIONS
# =============================================================================


def _print_win_rate_table(
    win_rate_results: List[Dict[str, Any]], n_models: int, n_bootstrap_samples: int
) -> None:
    """
    Print formatted win rate results table.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate results sorted by performance
    n_models : int
        Number of models analyzed
    n_bootstrap_samples : int
        Number of bootstrap samples
    """
    print(f"\nBootstrap Win Rate Analysis ({n_models} Models, {n_bootstrap_samples} samples):")
    print(f"{'Rank':<5} {'Model':<8} {'Win Rate':<10} {'Wins':<6} {'Original AIC':<12} {'Bootstrap Med':<13}")
    print("-" * 65)

    for rank, result in enumerate(win_rate_results, 1):
        print(f"{rank:<5} {result['model']:<8} {result['win_rate_pct']:<9.1f}% {result['win_count']:<6} "
              f"{result['original_aic']:<12.1f} {result['median_bootstrap_aic']:<13.1f}")


def _print_win_rate_insights(win_rate_results: List[Dict[str, Any]], n_models: int) -> None:
    """
    Print win rate insights and summary statistics.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate results sorted by performance
    n_models : int
        Total number of models analyzed
    """
    top_performer = win_rate_results[0]
    competitive_models = [r for r in win_rate_results if r['win_rate_pct'] > 10]

    print(f"\n=== WIN RATE INSIGHTS ===")
    print(f"Top Performer: {top_performer['model']} ({top_performer['win_rate_pct']:.1f}% win rate)")
    print(f"Competitive Models (>10%): {len(competitive_models)} out of {n_models}")

    confidence = 'High' if top_performer['win_rate_pct'] > 30 else 'Moderate' if top_performer['win_rate_pct'] > 20 else 'Low'
    print(f"Selection Confidence: {confidence}")


# =============================================================================
# MAIN WIN RATE ANALYSIS FUNCTION
# =============================================================================


def calculate_bootstrap_win_rates(bootstrap_results: List[BootstrapResult]) -> List[Dict[str, Any]]:
    """
    Calculate Bootstrap Win Rate Analysis - competitive performance assessment.

    Consolidates win rate calculation from cells 25-26 with enhanced error handling.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results

    Returns
    -------
    List[Dict[str, Any]]
        Win rate results sorted by performance
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results available for win rate analysis")

    print("=== BOOTSTRAP WIN RATE ANALYSIS ===")

    # Build data structures
    bootstrap_aic_matrix, model_names = _build_bootstrap_aic_matrix(bootstrap_results)
    n_models = len(bootstrap_results)
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)

    # Calculate win statistics
    win_counts = _calculate_win_counts(bootstrap_aic_matrix)
    win_rates = (win_counts / n_bootstrap_samples) * 100

    # Build and display results
    win_rate_results = _build_win_rate_results(bootstrap_results, model_names, win_rates, win_counts)
    _print_win_rate_table(win_rate_results, n_models, n_bootstrap_samples)
    _print_win_rate_insights(win_rate_results, n_models)

    return win_rate_results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Matrix construction
    '_build_bootstrap_aic_matrix',
    '_calculate_win_counts',

    # Results construction
    '_build_win_rate_results',

    # Display functions
    '_print_win_rate_table',
    '_print_win_rate_insights',

    # Main function
    'calculate_bootstrap_win_rates',
]
