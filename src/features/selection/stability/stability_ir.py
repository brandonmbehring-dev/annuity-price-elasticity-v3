"""
Information Ratio Analysis for Feature Selection.

This module handles Information Ratio calculations and reporting:
- Benchmark statistics preparation
- Sortino ratio and risk-adjusted metrics
- IR classification and insights

Module Architecture (Phase 6.3f Split):
- stability_win_rate.py: Win rate analysis
- stability_ir.py: Information ratio analysis (this file)
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
        """Fallback BootstrapResult when types module unavailable."""
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
# BENCHMARK STATISTICS
# =============================================================================


def _prepare_ir_benchmark_stats(bootstrap_results: List[BootstrapResult]) -> Tuple[float, float, float]:
    """
    Prepare benchmark statistics for Information Ratio analysis.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results

    Returns
    -------
    Tuple[float, float, float]
        (benchmark_aic, benchmark_mean, benchmark_std)
    """
    n_models = len(bootstrap_results)
    all_bootstrap_aics = []
    for i in range(n_models):
        all_bootstrap_aics.extend(bootstrap_results[i].bootstrap_aics)

    benchmark_aic = np.median(all_bootstrap_aics)
    benchmark_mean = np.mean(all_bootstrap_aics)
    benchmark_std = np.std(all_bootstrap_aics)

    return benchmark_aic, benchmark_mean, benchmark_std


# =============================================================================
# RATIO CALCULATIONS
# =============================================================================


def _compute_ratio_safe(numerator: float, denominator: float) -> float:
    """
    Compute ratio safely, handling zero/small denominator cases.

    Parameters
    ----------
    numerator : float
        Ratio numerator
    denominator : float
        Ratio denominator

    Returns
    -------
    float
        Computed ratio with edge cases handled
    """
    if denominator > 1e-10:
        return numerator / denominator
    else:
        # Zero variance: model performance is perfectly consistent
        if abs(numerator) > 1e-10:
            return 999.0 if numerator > 0 else -999.0  # Cap for numerical stability
        else:
            return 0.0  # No excess return and no variability


def _compute_sortino_ratio(excess_aics: np.ndarray, mean_excess: float) -> float:
    """
    Compute Sortino ratio - penalize only downside volatility.

    Parameters
    ----------
    excess_aics : np.ndarray
        Excess AIC values (positive = better than benchmark)
    mean_excess : float
        Mean excess AIC

    Returns
    -------
    float
        Sortino ratio (capped at 999 for display)
    """
    downside_deviations = excess_aics[excess_aics < 0]  # Negative = worse than benchmark
    downside_std = np.std(downside_deviations) if len(downside_deviations) > 0 else 0
    if downside_std > 0:
        return min(mean_excess / downside_std, 999)
    return 999.0 if mean_excess > 0 else 0.0


# =============================================================================
# SINGLE MODEL METRICS
# =============================================================================


def _calculate_single_model_ir_metrics(
    model_idx: int, bootstrap_result: BootstrapResult, benchmark_aic: float
) -> Dict[str, Any]:
    """Calculate Information Ratio metrics for a single model."""
    bootstrap_aics = np.array(bootstrap_result.bootstrap_aics)
    original_aic = bootstrap_result.original_aic

    # Excess performance relative to benchmark (positive = better)
    excess_aics = benchmark_aic - bootstrap_aics
    mean_excess = np.mean(excess_aics)
    std_excess = np.std(excess_aics)

    # Core ratios
    information_ratio = _compute_ratio_safe(mean_excess, std_excess)
    original_excess = benchmark_aic - original_aic
    sharpe_like = _compute_ratio_safe(original_excess, std_excess)
    sortino_ratio = _compute_sortino_ratio(excess_aics, mean_excess)

    return {
        'model_idx': model_idx,
        'model_name': f'Model {model_idx + 1}',
        'features': bootstrap_result.model_features,
        'original_aic': original_aic,
        'mean_bootstrap_aic': np.mean(bootstrap_aics),
        'benchmark_aic': benchmark_aic,
        'mean_excess': mean_excess,
        'std_excess': std_excess,
        'information_ratio': information_ratio,
        'sharpe_like': sharpe_like,
        'sortino_ratio': sortino_ratio,
        'success_rate': np.mean(excess_aics > 0) * 100,
        'positive_excess_count': np.sum(excess_aics > 0),
        'excess_aics': excess_aics
    }


# =============================================================================
# CLASSIFICATION
# =============================================================================


def _classify_risk_adjusted(ir: float) -> str:
    """
    Classify model by Information Ratio into risk-adjusted category.

    Parameters
    ----------
    ir : float
        Information Ratio value

    Returns
    -------
    str
        Risk-adjusted classification
    """
    if ir > 0.5:
        return "High"
    elif ir > 0.2:
        return "Moderate"
    elif ir > 0:
        return "Low"
    else:
        return "Negative"


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def _print_ir_results_table(model_ir_metrics_sorted: List[Dict[str, Any]]) -> None:
    """
    Print formatted Information Ratio results table.

    Parameters
    ----------
    model_ir_metrics_sorted : List[Dict[str, Any]]
        IR metrics sorted by performance
    """
    print(f"\nInformation Ratio Analysis Results (Sorted by IR):")
    print(f"{'Model':<8} {'Features':<30} {'IR':<6} {'Success%':<9} {'Sharpe':<7} {'Risk-Adj':<9}")
    print("-" * 75)

    for metrics in model_ir_metrics_sorted:
        features_short = metrics['features'][:27] + "..." if len(metrics['features']) > 27 else metrics['features']
        risk_adj_class = _classify_risk_adjusted(metrics['information_ratio'])

        print(f"{metrics['model_name']:<8} {features_short:<30} {metrics['information_ratio']:<6.3f} "
              f"{metrics['success_rate']:<8.1f}% {metrics['sharpe_like']:<7.3f} {risk_adj_class:<9}")


def _print_ir_insights(model_ir_metrics: List[Dict[str, Any]], n_models: int) -> None:
    """
    Print Information Ratio insights and summary statistics.

    Parameters
    ----------
    model_ir_metrics : List[Dict[str, Any]]
        All model IR metrics (unsorted, for counting)
    n_models : int
        Total number of models analyzed
    """
    high_ir_count = sum(1 for m in model_ir_metrics if m['information_ratio'] > 0.5)
    moderate_ir_count = sum(1 for m in model_ir_metrics if 0.2 <= m['information_ratio'] <= 0.5)
    low_ir_count = sum(1 for m in model_ir_metrics if 0 < m['information_ratio'] < 0.2)
    negative_ir_count = sum(1 for m in model_ir_metrics if m['information_ratio'] < 0)

    print(f"\n=== INFORMATION RATIO INSIGHTS ===")
    print(f"Risk-Adjusted Performance Classification:")
    print(f"  High IR (> 0.5): {high_ir_count}/{n_models} models ({high_ir_count/n_models:.1%})")
    print(f"  Moderate IR (0.2-0.5): {moderate_ir_count}/{n_models} models ({moderate_ir_count/n_models:.1%})")
    print(f"  Low IR (0-0.2): {low_ir_count}/{n_models} models ({low_ir_count/n_models:.1%})")
    print(f"  Negative IR (< 0): {negative_ir_count}/{n_models} models ({negative_ir_count/n_models:.1%})")


def _print_best_ir_model(best_ir_model: Dict[str, Any]) -> None:
    """
    Print details about the best risk-adjusted model.

    Parameters
    ----------
    best_ir_model : Dict[str, Any]
        Metrics for the best IR model
    """
    print(f"\nBest Risk-Adjusted Model: {best_ir_model['model_name']}")
    print(f"  Information Ratio: {best_ir_model['information_ratio']:.3f}")
    print(f"  Success Rate: {best_ir_model['success_rate']:.1f}% (beats benchmark)")
    print(f"  Mean Excess AIC: {best_ir_model['mean_excess']:.3f}")
    print(f"  Risk (Std Excess): {best_ir_model['std_excess']:.3f}")


# =============================================================================
# MAIN IR ANALYSIS FUNCTION
# =============================================================================


def calculate_information_ratio_analysis(bootstrap_results: List[BootstrapResult]) -> List[Dict[str, Any]]:
    """
    Calculate Information Ratio Analysis - risk-adjusted performance assessment.

    Consolidates information ratio calculation from cells 27-28 with enhanced metrics.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap analysis results

    Returns
    -------
    List[Dict[str, Any]]
        Information ratio results sorted by performance
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results available for information ratio analysis")

    print("=== INFORMATION RATIO ANALYSIS ===")

    # Prepare benchmark statistics
    n_models = len(bootstrap_results)
    benchmark_aic, benchmark_mean, benchmark_std = _prepare_ir_benchmark_stats(bootstrap_results)

    print(f"\nBenchmark Statistics:")
    print(f"  Population Median AIC: {benchmark_aic:.3f}")
    print(f"  Population Mean AIC: {benchmark_mean:.3f}")
    print(f"  Population Std AIC: {benchmark_std:.3f}")

    # Calculate IR metrics for each model
    model_ir_metrics = [
        _calculate_single_model_ir_metrics(i, bootstrap_results[i], benchmark_aic)
        for i in range(n_models)
    ]

    # Sort by Information Ratio (higher is better)
    model_ir_metrics_sorted = sorted(model_ir_metrics, key=lambda x: x['information_ratio'], reverse=True)

    # Display results and insights
    _print_ir_results_table(model_ir_metrics_sorted)
    _print_ir_insights(model_ir_metrics, n_models)
    _print_best_ir_model(model_ir_metrics_sorted[0])

    return model_ir_metrics_sorted


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Benchmark statistics
    '_prepare_ir_benchmark_stats',

    # Ratio calculations
    '_compute_ratio_safe',
    '_compute_sortino_ratio',

    # Single model metrics
    '_calculate_single_model_ir_metrics',

    # Classification
    '_classify_risk_adjusted',

    # Display functions
    '_print_ir_results_table',
    '_print_ir_insights',
    '_print_best_ir_model',

    # Main function
    'calculate_information_ratio_analysis',
]
