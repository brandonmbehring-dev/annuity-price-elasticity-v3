"""
Dashboard Visualization Functions (6 Plot Types).

This module contains all visualization functions for the comprehensive
stability dashboard, including scatter plots, distributions, rankings,
and recommendation summaries.

Module Responsibilities:
- Win Rate vs IR scatter plot with colorbar
- Composite score distribution histogram
- Win rate rankings (top 10 bar chart)
- Information ratio rankings (top 10 bar chart)
- AIC vs stability trade-off scatter
- Recommendation summary text box
- Main dashboard orchestration

Used by: interface_dashboard.py (orchestrator)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def plot_winrate_vs_ir_scatter(
    ax: plt.Axes, comprehensive_scores: List[Dict[str, Any]]
) -> List[float]:
    """
    Plot Win Rate vs Information Ratio scatter with colorbar.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive score results

    Returns
    -------
    List[float]
        Composite score values for use in other plots
    """
    win_rates = [result["win_rate_score"] for result in comprehensive_scores]
    ir_scores = [result["ir_score"] for result in comprehensive_scores]
    composite_scores_vals = [result["composite_score"] for result in comprehensive_scores]

    scatter = ax.scatter(
        win_rates,
        ir_scores,
        c=composite_scores_vals,
        cmap="RdYlGn",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    ax.set_xlabel("Bootstrap Win Rate (%)")
    ax.set_ylabel("Information Ratio Score")
    ax.set_title("Win Rate vs Information Ratio Performance", fontweight="bold")
    ax.grid(True, alpha=0.3)

    for i, result in enumerate(comprehensive_scores[:5]):
        ax.annotate(
            result["model_name"],
            (result["win_rate_score"], result["ir_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.colorbar(scatter, ax=ax, label="Composite Score")
    return composite_scores_vals


def plot_composite_score_distribution(
    ax: plt.Axes, composite_scores_vals: List[float]
) -> None:
    """
    Plot composite score distribution histogram.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    composite_scores_vals : List[float]
        Composite score values
    """
    ax.hist(composite_scores_vals, bins=8, alpha=0.7, color="skyblue", edgecolor="black")
    ax.axvline(
        np.mean(composite_scores_vals),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(composite_scores_vals):.1f}",
    )
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Count")
    ax.set_title("Composite Score Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_winrate_rankings(
    ax: plt.Axes, comprehensive_scores: List[Dict[str, Any]]
) -> None:
    """
    Plot bootstrap win rate rankings (top 10).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive score results
    """
    top_10_wr = comprehensive_scores[:10]
    wr_values = [model["win_rate_score"] for model in top_10_wr]
    wr_names = [model["model_name"] for model in top_10_wr]

    ax.barh(
        range(len(wr_names)),
        wr_values,
        alpha=0.7,
        color=plt.cm.viridis(np.linspace(0, 1, len(wr_names))),
    )
    ax.set_yticks(range(len(wr_names)))
    ax.set_yticklabels(wr_names)
    ax.set_xlabel("Bootstrap Win Rate (%)")
    ax.set_title("Bootstrap Win Rate Rankings (Top 10)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)


def plot_ir_rankings(ax: plt.Axes, comprehensive_scores: List[Dict[str, Any]]) -> None:
    """
    Plot information ratio rankings (top 10).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive score results
    """
    ir_sorted = sorted(comprehensive_scores, key=lambda x: x["ir_score"], reverse=True)
    top_10_ir = ir_sorted[:10]
    ir_values = [model["ir_score"] for model in top_10_ir]
    ir_names = [model["model_name"] for model in top_10_ir]

    ax.barh(
        range(len(ir_names)),
        ir_values,
        alpha=0.7,
        color=plt.cm.plasma(np.linspace(0, 1, len(ir_names))),
    )
    ax.set_yticks(range(len(ir_names)))
    ax.set_yticklabels(ir_names)
    ax.set_xlabel("IR Score")
    ax.set_title("Information Ratio Rankings (Top 10)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)


def plot_aic_vs_stability(
    ax: plt.Axes,
    comprehensive_scores: List[Dict[str, Any]],
    composite_scores_vals: List[float],
) -> None:
    """
    Plot AIC vs stability performance trade-off.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive score results
    composite_scores_vals : List[float]
        Composite score values
    """
    original_aics = [result["original_aic"] for result in comprehensive_scores]

    scatter2 = ax.scatter(
        original_aics,
        composite_scores_vals,
        c=composite_scores_vals,
        cmap="RdYlGn",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )
    ax.set_xlabel("Original AIC Score (Lower is Better)")
    ax.set_ylabel("Composite Stability Score (Higher is Better)")
    ax.set_title("AIC Performance vs Stability Trade-off", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax, shrink=0.8, label="Stability Score")


def plot_recommendation_summary(
    ax: plt.Axes, comprehensive_scores: List[Dict[str, Any]]
) -> None:
    """
    Plot final recommendation summary text box.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive score results
    """
    ax.axis("off")
    best_overall = comprehensive_scores[0]
    best_aic_idx = np.argmin([result["original_aic"] for result in comprehensive_scores])
    best_aic_model = comprehensive_scores[best_aic_idx]

    recommendation_text = f"""STABILITY RECOMMENDATION

Best Overall Stability:
{best_overall['model_name']}
* Score: {best_overall['composite_score']:.1f}/100
* Grade: {best_overall['stability_grade']}
* Win Rate: {best_overall['win_rate_score']:.1f}%
* IR Score: {best_overall['ir_score']:.1f}

Best AIC Model:
{best_aic_model['model_name']}
* AIC: {best_aic_model['original_aic']:.1f}
* Stability: {best_aic_model['composite_score']:.1f}/100

Excellent Models:
{sum(1 for r in comprehensive_scores if r['composite_score'] >= 80)}/{len(comprehensive_scores)}"""

    ax.text(
        0.05,
        0.95,
        recommendation_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
    )


def create_comprehensive_dashboard_visualizations(
    win_rate_results: List[Dict[str, Any]],
    ir_results: List[Dict[str, Any]],
    comprehensive_scores: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive dashboard visualizations consolidating 6 plot types.

    Orchestrator function: delegates to focused plot helpers following CODING_STANDARDS.md.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    ir_results : List[Dict[str, Any]]
        Information ratio results
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive scoring results
    config : Dict[str, Any]
        Dashboard configuration

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of matplotlib figures
    """
    visualizations = {}
    fig_width = config.get("fig_width", 16)
    fig_height = config.get("fig_height", 12)

    try:
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Win Rate vs IR scatter (row 0, cols 0-1)
        ax1 = fig.add_subplot(gs[0, 0:2])
        composite_scores_vals = plot_winrate_vs_ir_scatter(ax1, comprehensive_scores)

        # Plot 2: Composite score distribution (row 0, col 2)
        ax2 = fig.add_subplot(gs[0, 2])
        plot_composite_score_distribution(ax2, composite_scores_vals)

        # Plot 3: Win rate rankings (row 1, cols 0-1)
        ax3 = fig.add_subplot(gs[1, 0:2])
        plot_winrate_rankings(ax3, comprehensive_scores)

        # Plot 4: IR rankings (row 1, col 2)
        ax4 = fig.add_subplot(gs[1, 2])
        plot_ir_rankings(ax4, comprehensive_scores)

        # Plot 5: AIC vs stability (row 2, cols 0-1)
        ax5 = fig.add_subplot(gs[2, 0:2])
        plot_aic_vs_stability(ax5, comprehensive_scores, composite_scores_vals)

        # Plot 6: Recommendation summary (row 2, col 2)
        ax6 = fig.add_subplot(gs[2, 2])
        plot_recommendation_summary(ax6, comprehensive_scores)

        plt.suptitle(
            "Comprehensive Stability Analysis Dashboard (Win Rate + Information Ratio)",
            fontsize=14,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()
        visualizations["comprehensive_dashboard"] = fig

    except Exception as e:
        print(f"WARNING: Comprehensive dashboard visualization creation failed: {str(e)}")

    return visualizations


def create_dashboard_visualizations_safe(
    results: Dict[str, Any],
    comprehensive_scores: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """
    Safely create dashboard visualizations with error handling.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary to update in-place
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive scoring results
    config : Dict[str, Any]
        Dashboard configuration
    """
    try:
        dashboard_visualizations = create_comprehensive_dashboard_visualizations(
            results["win_rate_results"],
            results["information_ratio_results"],
            comprehensive_scores,
            config,
        )
        results["visualizations"] = dashboard_visualizations
        print(
            f"  Comprehensive dashboard visualizations created: {len(dashboard_visualizations)} figures"
        )
    except Exception as e:
        print(f"WARNING: Dashboard visualization creation failed: {str(e)}")
        results["visualizations"] = {}
