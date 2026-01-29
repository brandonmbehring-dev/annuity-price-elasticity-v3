"""
Stability Analysis Visualizations for Feature Selection.

This module handles all visualization functions for bootstrap stability analysis:
- Win rate bar charts and distributions
- Information ratio rankings and scatter plots
- Risk-return profiles and consistency plots

Module Architecture (Phase 6.3f Split):
- stability_win_rate.py: Win rate analysis
- stability_ir.py: Information ratio analysis
- stability_visualizations.py: Plotting functions (this file)
- bootstrap_stability_analysis.py: Orchestrator + public API
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


# =============================================================================
# WIN RATE VISUALIZATION FUNCTIONS
# =============================================================================


def _plot_win_rate_bar_chart(ax: plt.Axes, win_rate_results: List[Dict[str, Any]]) -> None:
    """
    Plot bar chart of win rates by model.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    """
    models = [r['model'] for r in win_rate_results]
    rates = [r['win_rate_pct'] for r in win_rate_results]
    colors = ['#2E8B57' if rate > 20 else '#FF6B35' if rate > 10 else '#6B73FF' for rate in rates]

    ax.bar(range(len(models)), rates, color=colors, alpha=0.7)
    ax.set_xlabel('Models (Ranked by Win Rate)')
    ax.set_ylabel('Bootstrap Win Rate (%)')
    ax.set_title('Bootstrap Win Rate Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, axis='y', alpha=0.3)


def _plot_win_rate_distribution(ax: plt.Axes, win_rate_results: List[Dict[str, Any]]) -> None:
    """
    Plot histogram distribution of win rates.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    """
    rates = [r['win_rate_pct'] for r in win_rate_results]

    ax.hist(rates, bins=8, color='#4CAF50', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rates), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rates):.1f}%')
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Number of Models')
    ax.set_title('Win Rate Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)


# =============================================================================
# INFORMATION RATIO VISUALIZATION FUNCTIONS
# =============================================================================


def _plot_ir_rankings(ax: plt.Axes, ir_results: List[Dict[str, Any]]) -> None:
    """
    Plot horizontal bar chart of Information Ratio rankings.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    """
    ir_values = [m['information_ratio'] for m in ir_results[:15]]
    model_names = [m['model_name'] for m in ir_results[:15]]

    # Color coding based on IR thresholds
    colors = ['#2E8B57' if ir > 0.5 else '#FFD700' if ir > 0.2 else '#FFA500' if ir > 0 else '#DC143C'
              for ir in ir_values]

    ax.barh(range(len(model_names)), ir_values, color=colors, alpha=0.7)
    ax.set_xlabel('Information Ratio (Higher = Better Risk-Adjusted Performance)')
    ax.set_ylabel('Models (Best Risk-Adjusted First)')
    ax.set_title('Information Ratio Rankings', fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.grid(True, axis='x', alpha=0.3)

    # Add threshold lines
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='High IR Threshold')
    ax.axvline(0.2, color='orange', linestyle='--', alpha=0.7, label='Moderate IR Threshold')
    ax.axvline(0, color='red', linestyle='-', alpha=0.7, label='Break-even')
    ax.legend()


def _plot_risk_return_scatter(ax: plt.Axes, ir_results: List[Dict[str, Any]]) -> None:
    """
    Plot risk vs return scatter plot with colorbar.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    """
    mean_excesses = [m['mean_excess'] for m in ir_results]
    std_excesses = [m['std_excess'] for m in ir_results]
    ir_colors_val = [m['information_ratio'] for m in ir_results]

    scatter = ax.scatter(std_excesses, mean_excesses, c=ir_colors_val, cmap='RdYlGn',
                         s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Risk (Std of Excess AIC)')
    ax.set_ylabel('Return (Mean Excess AIC)')
    ax.set_title('Risk-Return Profile of Models', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(0, color='red', linestyle='-', alpha=0.5, label='Benchmark Performance')
    ax.axvline(np.mean(std_excesses), color='blue', linestyle='--', alpha=0.5, label='Average Risk')

    # Add colorbar for Information Ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Information Ratio')
    ax.legend()


def _plot_consistency_scatter(ax: plt.Axes, ir_results: List[Dict[str, Any]]) -> None:
    """
    Plot success rate vs Information Ratio scatter.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    """
    success_rates = [m['success_rate'] for m in ir_results]
    information_ratios = [m['information_ratio'] for m in ir_results]

    # Color coding based on IR thresholds
    colors = ['#2E8B57' if ir > 0.5 else '#FFD700' if ir > 0.2 else '#FFA500' if ir > 0 else '#DC143C'
              for ir in information_ratios]

    ax.scatter(success_rates, information_ratios, c=colors, s=80, alpha=0.7)
    ax.set_xlabel('Success Rate (% Better than Benchmark)')
    ax.set_ylabel('Information Ratio')
    ax.set_title('Consistency vs Risk-Adjusted Performance', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(0, color='red', linestyle='-', alpha=0.5)
    ax.axvline(50, color='blue', linestyle='--', alpha=0.5, label='50% Success Rate')
    ax.legend()


def _plot_excess_aic_distributions(ax: plt.Axes, ir_results: List[Dict[str, Any]]) -> None:
    """
    Plot excess AIC distributions for top 3 models.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    """
    top_3_models = ir_results[:3]

    for model in top_3_models:
        ax.hist(model['excess_aics'], bins=20, alpha=0.6,
                label=f"{model['model_name']} (IR: {model['information_ratio']:.3f})",
                density=True)

    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Benchmark')
    ax.set_xlabel('Excess AIC (Positive = Better than Benchmark)')
    ax.set_ylabel('Density')
    ax.set_title('Excess Performance Distributions: Top 3 Models', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


# =============================================================================
# FIGURE CREATION FUNCTIONS
# =============================================================================


def _create_win_rate_figure(
    win_rate_results: List[Dict[str, Any]],
    fig_width: int,
    fig_height: int
) -> Optional[plt.Figure]:
    """
    Create win rate visualization figure.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    fig_width : int
        Figure width in inches
    fig_height : int
        Figure height in inches

    Returns
    -------
    Optional[plt.Figure]
        Created figure or None if creation fails
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
        _plot_win_rate_bar_chart(ax1, win_rate_results)
        _plot_win_rate_distribution(ax2, win_rate_results)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"WARNING: Win Rate visualization creation failed: {str(e)}")
        return None


def _create_ir_figure(
    ir_results: List[Dict[str, Any]],
    fig_width: int,
    fig_height: int
) -> Optional[plt.Figure]:
    """
    Create Information Ratio visualization figure.

    Parameters
    ----------
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    fig_width : int
        Figure width in inches
    fig_height : int
        Figure height in inches

    Returns
    -------
    Optional[plt.Figure]
        Created figure or None if creation fails
    """
    try:
        fig, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(fig_width, fig_height * 2))
        _plot_ir_rankings(ax3, ir_results)
        _plot_risk_return_scatter(ax4, ir_results)
        _plot_consistency_scatter(ax5, ir_results)
        _plot_excess_aic_distributions(ax6, ir_results)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"WARNING: Information Ratio visualization creation failed: {str(e)}")
        return None


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================


def create_advanced_visualizations(
    win_rate_results: Optional[List[Dict[str, Any]]] = None,
    ir_results: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, plt.Figure]:
    """
    Create unified visualizations for advanced stability analysis.

    Consolidates visualization creation from cells 25-28 with enhanced design.

    Parameters
    ----------
    win_rate_results : Optional[List[Dict[str, Any]]]
        Win rate analysis results
    ir_results : Optional[List[Dict[str, Any]]]
        Information ratio analysis results
    config : Optional[Dict[str, Any]]
        Configuration for visualization parameters

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary containing visualization figures
    """
    if not win_rate_results and not ir_results:
        raise ValueError("No analysis results available for visualization")

    config = config or {}
    fig_width = config.get('fig_width', 16)
    fig_height = config.get('fig_height', 6)

    visualizations = {}

    # Win Rate Visualization
    if win_rate_results:
        fig = _create_win_rate_figure(win_rate_results, fig_width, fig_height)
        if fig is not None:
            visualizations['win_rate_analysis'] = fig

    # Information Ratio Visualization
    if ir_results:
        fig = _create_ir_figure(ir_results, fig_width, fig_height)
        if fig is not None:
            visualizations['information_ratio_analysis'] = fig

    return visualizations


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Win rate plots
    '_plot_win_rate_bar_chart',
    '_plot_win_rate_distribution',

    # IR plots
    '_plot_ir_rankings',
    '_plot_risk_return_scatter',
    '_plot_consistency_scatter',
    '_plot_excess_aic_distributions',

    # Figure creation
    '_create_win_rate_figure',
    '_create_ir_figure',

    # Main function
    'create_advanced_visualizations',
]
