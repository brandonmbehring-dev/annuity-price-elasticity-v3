"""
Bootstrap Information Ratio Analysis Engine for Feature Selection.

Enhanced module for comprehensive information ratio analysis with integrated
visualization and risk-adjusted performance evaluation.

Key Functions:
- run_information_ratio_analysis: Main atomic function for risk-adjusted analysis
- calculate_bootstrap_information_ratios: Core information ratio computation engine
- create_information_ratio_visualizations: Focused visualization generation
- generate_information_ratio_insights: Business insights and risk-adjusted recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Defensive imports for bootstrap types
try:
    from src.features.selection_types import BootstrapResult
    BOOTSTRAP_TYPES_AVAILABLE = True
except ImportError:
    # Fallback for standalone operation
    from typing import TypedDict
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


def _calculate_benchmark_aic(bootstrap_results: List[BootstrapResult],
                            n_models: int) -> float:
    """Calculate benchmark AIC from all bootstrap samples across models."""
    all_bootstrap_aics = []
    for i in range(n_models):
        all_bootstrap_aics.extend(bootstrap_results[i].bootstrap_aics)
    return np.median(all_bootstrap_aics)


def _calculate_risk_adjusted_ratios(excess_aics: np.ndarray,
                                    mean_excess: float,
                                    std_excess: float,
                                    original_aic: float,
                                    benchmark_aic: float) -> Dict[str, float]:
    """Calculate Sharpe-like, Sortino, and Calmar ratios."""
    # Sharpe-like ratio using original AIC as reference
    original_excess = benchmark_aic - original_aic
    sharpe_like = original_excess / std_excess if std_excess > 0 else 0

    # Sortino ratio - penalize only downside volatility
    downside_deviations = excess_aics[excess_aics < 0]
    downside_std = np.std(downside_deviations) if len(downside_deviations) > 0 else 0
    sortino_ratio = mean_excess / downside_std if downside_std > 0 else (np.inf if mean_excess > 0 else 0)

    # Calmar-like ratio: return per unit of maximum drawdown
    running_max_excess = np.maximum.accumulate(np.concatenate([[0], excess_aics]))
    drawdowns = running_max_excess[:-1] - excess_aics
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    calmar_ratio = mean_excess / max_drawdown if max_drawdown > 0 else (np.inf if mean_excess > 0 else 0)

    return {
        'sharpe_like': sharpe_like,
        'sortino_ratio': min(sortino_ratio, 999),
        'calmar_ratio': min(calmar_ratio, 999),
        'max_drawdown': max_drawdown
    }


def _calculate_consistency_metrics(excess_aics: np.ndarray) -> Dict[str, Any]:
    """Calculate success rate and consecutive win metrics."""
    success_rate = np.mean(excess_aics > 0) * 100
    positive_excess_count = np.sum(excess_aics > 0)
    consecutive_wins = 0
    current_streak = 0

    for excess in excess_aics:
        if excess > 0:
            current_streak += 1
            consecutive_wins = max(consecutive_wins, current_streak)
        else:
            current_streak = 0

    return {
        'success_rate': success_rate,
        'positive_excess_count': int(positive_excess_count),
        'consecutive_wins': consecutive_wins
    }


def _compute_single_model_ir(bootstrap_result: BootstrapResult,
                             model_idx: int,
                             benchmark_aic: float) -> Dict[str, Any]:
    """Compute information ratio metrics for a single model."""
    bootstrap_aics = np.array(bootstrap_result.bootstrap_aics)
    original_aic = bootstrap_result.original_aic
    excess_aics = benchmark_aic - bootstrap_aics

    mean_excess = np.mean(excess_aics)
    std_excess = np.std(excess_aics)
    information_ratio = mean_excess / std_excess if std_excess > 0 else 0

    risk_ratios = _calculate_risk_adjusted_ratios(
        excess_aics, mean_excess, std_excess, original_aic, benchmark_aic
    )
    consistency = _calculate_consistency_metrics(excess_aics)

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
        'excess_aics': excess_aics,
        **risk_ratios,
        **consistency
    }


def calculate_bootstrap_information_ratios(bootstrap_results: List[BootstrapResult],
                                         max_models: int = 15) -> Tuple[List[Dict[str, Any]], float]:
    """
    Calculate bootstrap information ratios for risk-adjusted model evaluation.

    Orchestrator function delegating to focused helper functions.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap stability results with AIC distributions
    max_models : int, default 15
        Maximum number of models to analyze

    Returns
    -------
    Tuple[List[Dict[str, Any]], float]
        Information ratio results sorted by performance and benchmark AIC
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for information ratio analysis")

    n_models = min(len(bootstrap_results), max_models)
    benchmark_aic = _calculate_benchmark_aic(bootstrap_results, n_models)

    model_ir_metrics = [
        _compute_single_model_ir(bootstrap_results[i], i, benchmark_aic)
        for i in range(n_models)
    ]

    model_ir_metrics_sorted = sorted(
        model_ir_metrics, key=lambda x: x['information_ratio'], reverse=True
    )

    return model_ir_metrics_sorted, benchmark_aic


def _get_ir_color(ir_value: float) -> str:
    """Return color based on information ratio threshold."""
    if ir_value > 0.5:
        return '#2E8B57'  # Green
    elif ir_value > 0.2:
        return '#FFD700'  # Gold
    elif ir_value > 0:
        return '#FFA500'  # Orange
    return '#DC143C'  # Crimson


def _plot_ir_ranking(ax: plt.Axes, model_ir_metrics: List[Dict[str, Any]]) -> None:
    """Plot information ratio ranking bar chart."""
    ir_values = [m['information_ratio'] for m in model_ir_metrics[:15]]
    model_names = [m['model_name'] for m in model_ir_metrics[:15]]
    colors = [_get_ir_color(ir) for ir in ir_values]

    ax.barh(range(len(model_names)), ir_values, color=colors, alpha=0.7)
    ax.set_xlabel('Information Ratio (Higher = Better Risk-Adjusted Performance)')
    ax.set_ylabel('Models (Best Risk-Adjusted First)')
    ax.set_title('Information Ratio Rankings', fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='High IR Threshold')
    ax.axvline(0.2, color='orange', linestyle='--', alpha=0.7, label='Moderate IR Threshold')
    ax.axvline(0, color='red', linestyle='-', alpha=0.7, label='Break-even')
    ax.legend()


def _plot_risk_return_scatter(ax: plt.Axes, model_ir_metrics: List[Dict[str, Any]]) -> None:
    """Plot risk vs return scatter with colorbar."""
    mean_excesses = [m['mean_excess'] for m in model_ir_metrics]
    std_excesses = [m['std_excess'] for m in model_ir_metrics]
    ir_colors = [m['information_ratio'] for m in model_ir_metrics]

    scatter = ax.scatter(std_excesses, mean_excesses, c=ir_colors, cmap='RdYlGn',
                        s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Risk (Std of Excess AIC)')
    ax.set_ylabel('Return (Mean Excess AIC)')
    ax.set_title('Risk-Return Profile of Models', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='-', alpha=0.5, label='Benchmark Performance')
    ax.axvline(np.mean(std_excesses), color='blue', linestyle='--', alpha=0.5, label='Average Risk')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Information Ratio')
    ax.legend()


def _plot_success_rate_scatter(ax: plt.Axes, model_ir_metrics: List[Dict[str, Any]]) -> None:
    """Plot success rate vs information ratio scatter."""
    success_rates = [m['success_rate'] for m in model_ir_metrics]
    information_ratios = [m['information_ratio'] for m in model_ir_metrics]
    colors = [_get_ir_color(ir) for ir in information_ratios]

    ax.scatter(success_rates, information_ratios, c=colors, s=80, alpha=0.7)
    ax.set_xlabel('Success Rate (% Better than Benchmark)')
    ax.set_ylabel('Information Ratio')
    ax.set_title('Consistency vs Risk-Adjusted Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)
    ax.axvline(50, color='blue', linestyle='--', alpha=0.5, label='50% Success Rate')
    ax.legend()


def _plot_excess_distributions(ax: plt.Axes, model_ir_metrics: List[Dict[str, Any]]) -> None:
    """Plot excess AIC distributions for top 3 models."""
    for model in model_ir_metrics[:3]:
        ax.hist(model['excess_aics'], bins=20, alpha=0.6,
               label=f"{model['model_name']} (IR: {model['information_ratio']:.3f})",
               density=True)

    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Benchmark')
    ax.set_xlabel('Excess AIC (Positive = Better than Benchmark)')
    ax.set_ylabel('Density')
    ax.set_title('Excess Performance Distributions: Top 3 Models', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_information_ratio_visualizations(model_ir_metrics: List[Dict[str, Any]],
                                          benchmark_aic: float,
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive information ratio visualizations.

    Orchestrator function delegating to focused plotting helpers.

    Parameters
    ----------
    model_ir_metrics : List[Dict[str, Any]]
        Information ratio analysis results
    benchmark_aic : float
        Benchmark AIC for reference
    config : Optional[Dict[str, Any]], default None
        Configuration for figure sizing

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing information_ratio_analysis figure
    """
    if not model_ir_metrics:
        raise ValueError("No information ratio metrics provided for visualization")

    fig_width = config.get('fig_width', 16) if config else 16
    fig_height = config.get('fig_height', 12) if config else 12

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    _plot_ir_ranking(ax1, model_ir_metrics)
    _plot_risk_return_scatter(ax2, model_ir_metrics)
    _plot_success_rate_scatter(ax3, model_ir_metrics)
    _plot_excess_distributions(ax4, model_ir_metrics)

    plt.tight_layout()

    return {'information_ratio_analysis': fig}


def _classify_ir_models(model_ir_metrics: List[Dict[str, Any]]) -> Dict[str, int]:
    """Classify models by information ratio thresholds."""
    return {
        'high_ir_models': sum(1 for m in model_ir_metrics if m['information_ratio'] > 0.5),
        'moderate_ir_models': sum(1 for m in model_ir_metrics if 0.2 <= m['information_ratio'] <= 0.5),
        'low_ir_models': sum(1 for m in model_ir_metrics if 0 < m['information_ratio'] < 0.2),
        'negative_ir_models': sum(1 for m in model_ir_metrics if m['information_ratio'] < 0),
        'total_models': len(model_ir_metrics)
    }


def _get_ir_recommendation(risk_classification: Dict[str, int]) -> Tuple[str, str]:
    """Get recommendation category and detail based on IR classification."""
    if risk_classification['high_ir_models'] > 0:
        return "RECOMMENDED", f"{risk_classification['high_ir_models']} models show strong risk-adjusted performance"
    elif risk_classification['moderate_ir_models'] > 0:
        return "ACCEPTABLE", f"{risk_classification['moderate_ir_models']} models offer reasonable risk-adjusted value"
    return "CAUTION", "No models show strong risk-adjusted performance over benchmark"


def generate_information_ratio_insights(model_ir_metrics: List[Dict[str, Any]],
                                       n_bootstrap_samples: int) -> Dict[str, Any]:
    """Generate business insights from information ratio analysis results."""
    if not model_ir_metrics:
        return {
            'best_risk_adjusted_model': None,
            'risk_classification': 'No Data',
            'insights_summary': 'No information ratio results available for analysis'
        }

    best_ir_model = model_ir_metrics[0]
    risk_classification = _classify_ir_models(model_ir_metrics)
    recommendation_category, recommendation_detail = _get_ir_recommendation(risk_classification)

    insights_summary = (
        f"Information Ratio Analysis reveals:\n"
        f"- Best Risk-Adjusted Model: {best_ir_model['model_name']} (IR: {best_ir_model['information_ratio']:.3f})\n"
        f"- Success Rate: {best_ir_model['success_rate']:.1f}% beats benchmark\n"
        f"- Risk-Adjusted Performance: {recommendation_category} - {recommendation_detail}\n"
        f"- Model Distribution: High IR ({risk_classification['high_ir_models']}), "
        f"Moderate IR ({risk_classification['moderate_ir_models']}), "
        f"Low IR ({risk_classification['low_ir_models']}), "
        f"Negative IR ({risk_classification['negative_ir_models']})"
    )

    return {
        'best_risk_adjusted_model': best_ir_model,
        'risk_classification': risk_classification,
        'recommendation_category': recommendation_category,
        'recommendation_detail': recommendation_detail,
        'insights_summary': insights_summary,
        'benchmark_statistics': {
            'benchmark_aic': float(best_ir_model['benchmark_aic']),
            'n_bootstrap_samples': n_bootstrap_samples,
            'analysis_type': 'information_ratio'
        }
    }


def _get_ir_class_label(ir_value: float) -> str:
    """Get risk-adjusted classification label for IR value."""
    if ir_value > 0.5:
        return "High"
    elif ir_value > 0.2:
        return "Moderate"
    elif ir_value > 0:
        return "Low"
    return "Negative"


def _display_benchmark_stats(bootstrap_results: List[BootstrapResult],
                            max_models: int, benchmark_aic: float) -> None:
    """Display benchmark statistics."""
    all_bootstrap_aics = []
    for i in range(min(len(bootstrap_results), max_models)):
        all_bootstrap_aics.extend(bootstrap_results[i].bootstrap_aics)

    print(f"\nBenchmark Statistics:")
    print(f"  Population Median AIC: {benchmark_aic:.3f}")
    print(f"  Population Mean AIC: {np.mean(all_bootstrap_aics):.3f}")
    print(f"  Population Std AIC: {np.std(all_bootstrap_aics):.3f}")


def _display_ir_results_table(model_ir_metrics_sorted: List[Dict[str, Any]]) -> None:
    """Display information ratio results table."""
    print(f"\nInformation Ratio Analysis Results (Sorted by IR):")
    print(f"{'Model':<8} {'Features':<30} {'IR':<6} {'Success%':<9} {'Sharpe':<7} {'Risk-Adj':<9}")
    print("-" * 75)

    for metrics in model_ir_metrics_sorted[:15]:
        features_short = metrics['features'][:27] + "..." if len(metrics['features']) > 27 else metrics['features']
        risk_adj_class = _get_ir_class_label(metrics['information_ratio'])
        print(f"{metrics['model_name']:<8} {features_short:<30} {metrics['information_ratio']:<6.3f} "
              f"{metrics['success_rate']:<8.1f}% {metrics['sharpe_like']:<7.3f} {risk_adj_class:<9}")


def _display_ir_insights(insights: Dict[str, Any],
                        model_ir_metrics_sorted: List[Dict[str, Any]]) -> None:
    """Display information ratio insights."""
    print(f"\n=== INFORMATION RATIO INSIGHTS ===")
    risk_class = insights['risk_classification']
    total = risk_class['total_models']
    print(f"Risk-Adjusted Performance Classification:")
    print(f"  High IR (> 0.5): {risk_class['high_ir_models']}/{total} models ({risk_class['high_ir_models']/total:.1%})")
    print(f"  Moderate IR (0.2-0.5): {risk_class['moderate_ir_models']}/{total} models ({risk_class['moderate_ir_models']/total:.1%})")
    print(f"  Low IR (0-0.2): {risk_class['low_ir_models']}/{total} models ({risk_class['low_ir_models']/total:.1%})")
    print(f"  Negative IR (< 0): {risk_class['negative_ir_models']}/{total} models ({risk_class['negative_ir_models']/total:.1%})")

    best_ir_model = insights['best_risk_adjusted_model']
    print(f"\nBest Risk-Adjusted Model: {best_ir_model['model_name']}")
    print(f"  Information Ratio: {best_ir_model['information_ratio']:.3f}")
    print(f"  Success Rate: {best_ir_model['success_rate']:.1f}% (beats benchmark)")
    print(f"  Mean Excess AIC: {best_ir_model['mean_excess']:.3f}")
    print(f"  Risk (Std Excess): {best_ir_model['std_excess']:.3f}")
    print(f"  Interpretation: Provides {best_ir_model['mean_excess']:.1f} AIC improvement per unit of risk")

    print(f"\n=== RISK-ADJUSTED RECOMMENDATIONS ===")
    print(f"{insights['recommendation_category']}: {insights['recommendation_detail']}")
    if risk_class['high_ir_models'] > 0:
        high_ir_models = [m for m in model_ir_metrics_sorted if m['information_ratio'] > 0.5]
        for model in high_ir_models[:3]:
            print(f"  {model['model_name']}: IR={model['information_ratio']:.3f}, Success={model['success_rate']:.1f}%")


def _compute_ir_metrics(bootstrap_results: List[BootstrapResult],
                        max_models: int) -> Tuple[List[Dict[str, Any]], float]:
    """Compute information ratio metrics for all models."""
    model_ir_metrics_sorted, benchmark_aic = calculate_bootstrap_information_ratios(
        bootstrap_results, max_models
    )
    return model_ir_metrics_sorted, benchmark_aic


def _handle_results_display(bootstrap_results: List[BootstrapResult],
                            max_models: int,
                            benchmark_aic: float,
                            model_ir_metrics_sorted: List[Dict[str, Any]],
                            insights: Dict[str, Any]) -> None:
    """Display all analysis results."""
    _display_benchmark_stats(bootstrap_results, max_models, benchmark_aic)
    _display_ir_results_table(model_ir_metrics_sorted)
    _display_ir_insights(insights, model_ir_metrics_sorted)


def _handle_visualization_creation(model_ir_metrics_sorted: List[Dict[str, Any]],
                                   benchmark_aic: float,
                                   config: Optional[Dict[str, Any]]) -> Dict[str, plt.Figure]:
    """Create and display information ratio visualizations."""
    visualizations = {}
    try:
        visualizations = create_information_ratio_visualizations(
            model_ir_metrics_sorted, benchmark_aic, config
        )
        for fig in visualizations.values():
            plt.figure(fig.number)
            plt.show()
        print(f"SUCCESS: Information ratio visualizations displayed")
    except Exception as e:
        print(f"WARNING: Information ratio visualization creation failed: {str(e)}")
    return visualizations


def _build_detailed_results(model_ir_metrics_sorted: List[Dict[str, Any]],
                           insights: Dict[str, Any],
                           visualizations: Dict[str, plt.Figure],
                           n_bootstrap_samples: int,
                           benchmark_aic: float,
                           max_models: int) -> Dict[str, Any]:
    """Build detailed analysis results dictionary."""
    return {
        'ir_results': model_ir_metrics_sorted,
        'insights': insights,
        'visualizations': visualizations,
        'analysis_metadata': {
            'n_models_analyzed': len(model_ir_metrics_sorted),
            'n_bootstrap_samples': n_bootstrap_samples,
            'benchmark_aic': benchmark_aic,
            'max_models_configured': max_models,
            'analysis_complete': True
        }
    }


def run_information_ratio_analysis(bootstrap_results: List[BootstrapResult],
                                 config: Optional[Dict[str, Any]] = None,
                                 display_results: bool = True,
                                 create_visualizations: bool = True,
                                 return_detailed: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive bootstrap information ratio analysis.

    Orchestrator function delegating to focused helper functions.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap stability results with AIC distributions
    config : Optional[Dict[str, Any]], default None
        Configuration for analysis parameters and visualization
    display_results : bool, default True
        Whether to print formatted results to console
    create_visualizations : bool, default True
        Whether to generate and display information ratio charts
    return_detailed : bool, default True
        Whether to return detailed analysis results

    Returns
    -------
    Dict[str, Any]
        Comprehensive information ratio analysis results
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for information ratio analysis")

    max_models = config.get('models_to_analyze', 15) if config else 15
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)
    print("=== INFORMATION RATIO ANALYSIS ===")

    model_ir_metrics_sorted, benchmark_aic = _compute_ir_metrics(bootstrap_results, max_models)
    insights = generate_information_ratio_insights(model_ir_metrics_sorted, n_bootstrap_samples)

    if display_results:
        _handle_results_display(bootstrap_results, max_models, benchmark_aic,
                               model_ir_metrics_sorted, insights)

    visualizations = {}
    if create_visualizations:
        visualizations = _handle_visualization_creation(model_ir_metrics_sorted,
                                                       benchmark_aic, config)

    print(f"SUCCESS: Information Ratio Analysis complete")

    if return_detailed:
        return _build_detailed_results(model_ir_metrics_sorted, insights, visualizations,
                                       n_bootstrap_samples, benchmark_aic, max_models)
    return {'ir_results': model_ir_metrics_sorted}


# Convenience function for notebook integration
def run_notebook_information_ratio_analysis(bootstrap_results: List[BootstrapResult]) -> List[Dict[str, Any]]:
    """
    Notebook-friendly information ratio analysis with automatic display.

    Simplified interface for backward compatibility with existing notebook cells.
    """
    if not bootstrap_results:
        print("No bootstrap results available for information ratio analysis")
        return []

    analysis_results = run_information_ratio_analysis(
        bootstrap_results=bootstrap_results,
        display_results=True,
        create_visualizations=True,
        return_detailed=False
    )

    return analysis_results.get('ir_results', [])