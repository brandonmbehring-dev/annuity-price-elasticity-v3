"""
Bootstrap Win Rate Analysis Engine for Feature Selection.

Enhanced module for comprehensive bootstrap win rate analysis with integrated
visualization and business insights generation.

Key Functions:
- run_bootstrap_win_rate_analysis: Main atomic function for win rate calculation
- calculate_model_win_rates: Core win rate computation engine
- create_win_rate_visualizations: Focused visualization generation
- generate_win_rate_insights: Business insights and recommendations
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


def _build_bootstrap_matrix(bootstrap_results: List[BootstrapResult],
                           n_models: int) -> Tuple[np.ndarray, List[str]]:
    """Build bootstrap AIC matrix and model names list."""
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)
    bootstrap_aic_matrix = np.zeros((n_bootstrap_samples, n_models))
    model_names = []

    for i in range(n_models):
        bootstrap_aic_matrix[:, i] = bootstrap_results[i].bootstrap_aics
        model_names.append(f"Model {i+1}")

    return bootstrap_aic_matrix, model_names


def _compute_win_counts(bootstrap_aic_matrix: np.ndarray) -> np.ndarray:
    """Compute win counts for each model across bootstrap samples."""
    n_bootstrap_samples, n_models = bootstrap_aic_matrix.shape
    win_counts = np.zeros(n_models)

    for sample_idx in range(n_bootstrap_samples):
        winner_idx = np.argmin(bootstrap_aic_matrix[sample_idx, :])
        win_counts[winner_idx] += 1

    return win_counts


def calculate_model_win_rates(bootstrap_results: List[BootstrapResult],
                            max_models: int = 15) -> List[Dict[str, Any]]:
    """
    Calculate bootstrap win rates for models using AIC competition.

    Parameters
    ----------
    bootstrap_results : List[BootstrapResult]
        Bootstrap stability results with AIC distributions
    max_models : int, default 15
        Maximum number of models to analyze

    Returns
    -------
    List[Dict[str, Any]]
        Win rate results sorted by performance
    """
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for win rate analysis")

    n_models = min(len(bootstrap_results), max_models)
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)

    bootstrap_aic_matrix, model_names = _build_bootstrap_matrix(bootstrap_results, n_models)
    win_counts = _compute_win_counts(bootstrap_aic_matrix)
    win_rates = (win_counts / n_bootstrap_samples) * 100

    win_rate_results = [
        {
            'model': model_names[i],
            'features': bootstrap_results[i].model_features,
            'win_rate_pct': win_rates[i],
            'win_count': int(win_counts[i]),
            'original_aic': bootstrap_results[i].original_aic,
            'median_bootstrap_aic': np.median(bootstrap_results[i].bootstrap_aics)
        }
        for i in range(n_models)
    ]

    win_rate_results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
    return win_rate_results


def _get_win_rate_color(rate: float) -> str:
    """Return color based on win rate threshold."""
    if rate > 20:
        return '#2E8B57'  # Green
    elif rate > 10:
        return '#FF6B35'  # Orange
    return '#6B73FF'  # Blue


def _plot_win_rate_bars(ax: plt.Axes, win_rate_results: List[Dict[str, Any]]) -> None:
    """Plot win rate bar chart."""
    models = [r['model'] for r in win_rate_results]
    rates = [r['win_rate_pct'] for r in win_rate_results]
    colors = [_get_win_rate_color(rate) for rate in rates]

    ax.bar(range(len(models)), rates, color=colors, alpha=0.7)
    ax.set_xlabel('Models (Ranked by Win Rate)')
    ax.set_ylabel('Bootstrap Win Rate (%)')
    ax.set_title('Bootstrap Win Rate Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, axis='y', alpha=0.3)


def _plot_win_rate_distribution(ax: plt.Axes, win_rate_results: List[Dict[str, Any]]) -> None:
    """Plot win rate distribution histogram."""
    rates = [r['win_rate_pct'] for r in win_rate_results]
    ax.hist(rates, bins=8, color='#4CAF50', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rates), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(rates):.1f}%')
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Number of Models')
    ax.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)


def create_win_rate_visualizations(win_rate_results: List[Dict[str, Any]],
                                 n_bootstrap_samples: int,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
    """
    Create comprehensive win rate visualizations.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    n_bootstrap_samples : int
        Total number of bootstrap samples for context
    config : Optional[Dict[str, Any]], default None
        Configuration for figure sizing

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing win_rate_chart figure
    """
    if not win_rate_results:
        raise ValueError("No win rate results provided for visualization")

    fig_width = config.get('fig_width', 16) if config else 16
    fig_height = config.get('fig_height', 6) if config else 6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    _plot_win_rate_bars(ax1, win_rate_results)
    _plot_win_rate_distribution(ax2, win_rate_results)

    plt.tight_layout()
    return {'win_rate_chart': fig}


def _assess_selection_confidence(win_rate_pct: float) -> Tuple[str, str]:
    """Assess selection confidence based on win rate percentage."""
    if win_rate_pct > 30:
        return 'High', f"Top model wins {win_rate_pct:.1f}% of bootstrap samples"
    elif win_rate_pct > 20:
        return 'Moderate', f"Top model wins {win_rate_pct:.1f}% of bootstrap samples - reasonable consistency"
    return 'Low', f"Top model wins only {win_rate_pct:.1f}% of bootstrap samples - high uncertainty"


def generate_win_rate_insights(win_rate_results: List[Dict[str, Any]],
                             n_bootstrap_samples: int) -> Dict[str, Any]:
    """Generate business insights from win rate analysis results."""
    if not win_rate_results:
        return {
            'top_performer': None,
            'competitive_models': [],
            'selection_confidence': 'No Data',
            'insights_summary': 'No win rate results available for analysis'
        }

    top_performer = win_rate_results[0]
    competitive_models = [r for r in win_rate_results if r['win_rate_pct'] > 10]
    n_models = len(win_rate_results)
    selection_confidence, confidence_rationale = _assess_selection_confidence(top_performer['win_rate_pct'])

    competition_status = 'Highly competitive field' if len(competitive_models) > 5 else 'Clear leaders emerged'
    insights_summary = (
        f"Bootstrap Win Rate Analysis reveals:\n"
        f"- Top Performer: {top_performer['model']} ({top_performer['win_rate_pct']:.1f}% win rate)\n"
        f"- Competitive Models: {len(competitive_models)}/{n_models} models show >10% win rate\n"
        f"- Selection Confidence: {selection_confidence} - {confidence_rationale}\n"
        f"- Model Competition: {competition_status}"
    )

    return {
        'top_performer': top_performer,
        'competitive_models': competitive_models,
        'selection_confidence': selection_confidence,
        'confidence_rationale': confidence_rationale,
        'insights_summary': insights_summary,
        'analysis_metadata': {
            'n_models_analyzed': n_models,
            'n_bootstrap_samples': n_bootstrap_samples,
            'competitive_threshold': 10.0,
            'analysis_type': 'bootstrap_win_rate'
        }
    }


def _display_win_rate_table(win_rate_results: List[Dict[str, Any]],
                           n_bootstrap_samples: int) -> None:
    """Display win rate results table."""
    n_models = len(win_rate_results)
    print(f"\nBootstrap Win Rate Analysis (Top {n_models} Models, {n_bootstrap_samples} samples):")
    print(f"{'Rank':<5} {'Model':<8} {'Win Rate':<10} {'Wins':<6} {'Original AIC':<12} {'Bootstrap Med':<13}")
    print("-" * 65)

    for rank, result in enumerate(win_rate_results, 1):
        print(f"{rank:<5} {result['model']:<8} {result['win_rate_pct']:<9.1f}% {result['win_count']:<6} "
              f"{result['original_aic']:<12.1f} {result['median_bootstrap_aic']:<13.1f}")


def _validate_bootstrap_inputs(bootstrap_results: List[BootstrapResult]) -> None:
    """Validate bootstrap results for win rate analysis."""
    if not bootstrap_results:
        raise ValueError("No bootstrap results provided for win rate analysis")
    if len(bootstrap_results[0].bootstrap_aics) == 0:
        raise ValueError("Bootstrap results contain no AIC values")


def _display_insights_output(insights: Dict[str, Any]) -> None:
    """Display insights summary to console."""
    print(f"\n=== WIN RATE INSIGHTS ===")
    print(insights['insights_summary'])


def _create_and_display_visualizations(
    win_rate_results: List[Dict[str, Any]],
    n_bootstrap_samples: int,
    config: Optional[Dict[str, Any]]
) -> Dict[str, plt.Figure]:
    """Create visualizations and display them, handling errors gracefully."""
    visualizations: Dict[str, plt.Figure] = {}
    try:
        visualizations = create_win_rate_visualizations(
            win_rate_results, n_bootstrap_samples, config
        )
        for fig in visualizations.values():
            plt.figure(fig.number)
            plt.show()
        print(f"SUCCESS: Win rate visualizations displayed")
    except Exception as e:
        print(f"WARNING: Win rate visualization creation failed: {str(e)}")
    return visualizations


def _build_detailed_results(
    win_rate_results: List[Dict[str, Any]],
    insights: Dict[str, Any],
    visualizations: Dict[str, plt.Figure],
    n_bootstrap_samples: int,
    max_models: int
) -> Dict[str, Any]:
    """Build detailed analysis results dictionary."""
    return {
        'win_rate_results': win_rate_results,
        'insights': insights,
        'visualizations': visualizations,
        'analysis_metadata': {
            'n_models_analyzed': len(win_rate_results),
            'n_bootstrap_samples': n_bootstrap_samples,
            'max_models_configured': max_models,
            'analysis_complete': True
        }
    }


def run_bootstrap_win_rate_analysis(bootstrap_results: List[BootstrapResult],
                                  config: Optional[Dict[str, Any]] = None,
                                  display_results: bool = True,
                                  create_visualizations: bool = True,
                                  return_detailed: bool = True) -> Dict[str, Any]:
    """Run comprehensive bootstrap win rate analysis with visualizations and insights."""
    _validate_bootstrap_inputs(bootstrap_results)

    max_models = config.get('models_to_analyze', 15) if config else 15
    n_bootstrap_samples = len(bootstrap_results[0].bootstrap_aics)
    print("=== BOOTSTRAP WIN RATE ANALYSIS ===")

    win_rate_results = calculate_model_win_rates(bootstrap_results, max_models)
    if display_results:
        _display_win_rate_table(win_rate_results, n_bootstrap_samples)

    insights = generate_win_rate_insights(win_rate_results, n_bootstrap_samples)
    if display_results:
        _display_insights_output(insights)

    visualizations: Dict[str, plt.Figure] = {}
    if create_visualizations:
        visualizations = _create_and_display_visualizations(
            win_rate_results, n_bootstrap_samples, config
        )

    print(f"SUCCESS: Bootstrap Win Rate Analysis complete")

    if return_detailed:
        return _build_detailed_results(
            win_rate_results, insights, visualizations, n_bootstrap_samples, max_models
        )
    return {'win_rate_results': win_rate_results}


# Convenience function for notebook integration
def run_notebook_win_rate_analysis(bootstrap_results: List[BootstrapResult]) -> List[Dict[str, Any]]:
    """
    Notebook-friendly win rate analysis with automatic display.

    Simplified interface for backward compatibility with existing notebook cells.
    """
    if not bootstrap_results:
        print("No bootstrap results available for win rate analysis")
        return []

    analysis_results = run_bootstrap_win_rate_analysis(
        bootstrap_results=bootstrap_results,
        display_results=True,
        create_visualizations=True,
        return_detailed=False
    )

    return analysis_results.get('win_rate_results', [])