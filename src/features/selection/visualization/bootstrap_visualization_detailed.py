"""
Bootstrap Visualization Detailed Module for Feature Selection Pipeline.

This module provides atomic functions for detailed bootstrap visualization including
DataFrame creation, violin plots, boxplots, and ranking statistics following
CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Extract Cell 24 visualization code (249 lines) to atomic functions
Status: NEW (decomposition from notebook inline code)
Priority: HIGH (DRY compliance and maintainability)

Key Functions:
- create_bootstrap_dataframe(): Bootstrap data preparation (≤50 lines)
- generate_violin_plot_visualization(): Violin plot creation (≤50 lines)
- generate_boxplot_visualization(): Boxplot creation (≤50 lines)
- calculate_ranking_statistics(): Ranking statistics calculation (≤50 lines)

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Optional


def _build_bootstrap_row(result: Any, model_idx: int, sample_idx: int, aic_val: float) -> Dict[str, Any]:
    """Build a single row for the bootstrap DataFrame."""
    return {
        'model': f'Model {model_idx + 1}',
        'model_features': result.model_features,
        'bootstrap_aic': aic_val,
        'bootstrap_r2': result.bootstrap_r2_values[sample_idx] if sample_idx < len(result.bootstrap_r2_values) else np.nan,
        'original_aic': result.original_aic,
        'original_r2': result.original_r2,
        'stability_assessment': result.stability_assessment
    }


def create_bootstrap_dataframe(bootstrap_results: List[Any]) -> pd.DataFrame:
    """Create bootstrap DataFrame from bootstrap analysis results.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results with AIC distributions

    Returns
    -------
    pd.DataFrame
        Bootstrap visualization DataFrame with model data
    """
    if not bootstrap_results:
        raise ValueError(
            "CRITICAL: No bootstrap results for DataFrame creation. "
            "Business impact: Cannot prepare visualization data. "
            "Required action: Complete bootstrap analysis first."
        )

    bootstrap_viz_data = [
        _build_bootstrap_row(result, i, j, aic_val)
        for i, result in enumerate(bootstrap_results)
        for j, aic_val in enumerate(result.bootstrap_aics)
    ]

    bootstrap_df = pd.DataFrame(bootstrap_viz_data)
    print(f"Bootstrap DataFrame created: {bootstrap_df.shape[0]} samples for {len(bootstrap_results)} models")
    return bootstrap_df


def _get_stability_color(stability: str) -> tuple:
    """Get color based on stability assessment."""
    if stability == 'STABLE':
        return sns.color_palette("deep")[2]  # Green
    elif stability == 'MODERATE':
        return sns.color_palette("deep")[1]  # Orange
    return sns.color_palette("deep")[3]  # Red


def _plot_single_violin(ax: plt.Axes, model_data: np.ndarray,
                       row_idx: int, color: tuple, stability: str) -> None:
    """Plot a single half-violin with KDE for one model."""
    p1, p99 = np.percentile(model_data, [2.5, 97.5])
    x_range = np.linspace(p1, p99, 100)

    kde = stats.gaussian_kde(model_data)
    density = kde(x_range)
    density_scaled = density / density.max() * 0.35

    ax.fill_between(x_range, row_idx, row_idx + density_scaled,
                   color=color, alpha=0.7, label=stability if row_idx == 0 else "")
    ax.plot(x_range, row_idx + density_scaled, color=color, linewidth=1.5)
    ax.axhline(y=row_idx, color='#CCCCCC', linewidth=0.5, alpha=0.8)


def _add_violin_markers(ax: plt.Axes, model_data: np.ndarray,
                       original_aic: float, row_idx: int) -> None:
    """Add original AIC point and median line to violin plot."""
    ax.scatter(original_aic, row_idx, color='black', s=60, zorder=10,
              edgecolors='white', linewidth=1.5)
    median_val = np.median(model_data)
    ax.plot([median_val, median_val], [row_idx - 0.1, row_idx + 0.3],
           color='darkblue', linewidth=2.5, zorder=5)


def _format_violin_axes(ax: plt.Axes, bootstrap_df: pd.DataFrame,
                       models_order: List[str]) -> None:
    """Apply formatting to violin plot axes."""
    ax.set_xlabel('AIC Score (Lower is Better)')
    ax.set_ylabel('Model Ranking')
    ax.set_title('Bootstrap AIC Distribution Analysis - Top Models', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(models_order)))
    ax.set_yticklabels([
        f'{model}\n({bootstrap_df[bootstrap_df["model"]==model]["stability_assessment"].iloc[0]})'
        for model in models_order
    ])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(False, axis='y')


def generate_violin_plot_visualization(bootstrap_df: pd.DataFrame,
                                     bootstrap_results: List[Any],
                                     config: Dict[str, Any],
                                     display_plot: bool = True) -> Dict[str, Any]:
    """Generate violin plot visualization for bootstrap AIC distributions."""
    if bootstrap_df.empty:
        raise ValueError(
            "CRITICAL: Empty DataFrame for violin plot generation. "
            "Required action: Ensure bootstrap DataFrame creation succeeds."
        )

    n_models = min(len(bootstrap_results), config.get('models_to_analyze', 15))
    models_order = [f'Model {i+1}' for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(config.get('fig_width', 12), config.get('fig_height', 10)))

    for i, model in enumerate(models_order):
        model_data = bootstrap_df[bootstrap_df['model'] == model]['bootstrap_aic'].values
        stability = bootstrap_df[bootstrap_df['model'] == model]['stability_assessment'].iloc[0]

        if len(model_data) > 5:
            color = _get_stability_color(stability)
            _plot_single_violin(ax, model_data, i, color, stability)
            original_aic = bootstrap_df[bootstrap_df['model'] == model]['original_aic'].iloc[0]
            _add_violin_markers(ax, model_data, original_aic, i)

    _format_violin_axes(ax, bootstrap_df, models_order)

    plt.tight_layout()
    if display_plot:
        plt.show()

    return {
        'violin_plot_created': True,
        'models_visualized': len(models_order),
        'plot_config': config
    }


def _prepare_boxplot_data(bootstrap_df: pd.DataFrame,
                         models_order: List[str]) -> tuple:
    """Prepare data, labels, and colors for boxplot visualization."""
    boxplot_data = []
    model_labels = []
    colors = []

    for model in models_order:
        model_data = bootstrap_df[bootstrap_df['model'] == model]['bootstrap_aic'].values
        stability = bootstrap_df[bootstrap_df['model'] == model]['stability_assessment'].iloc[0]

        if len(model_data) > 5:
            boxplot_data.append(model_data)
            features = bootstrap_df[bootstrap_df['model'] == model]['model_features'].iloc[0]
            short_features = features[:35] + "..." if len(features) > 35 else features
            model_labels.append(f'{model}\n{short_features}\n({stability})')
            colors.append(_get_stability_color(stability))

    return boxplot_data, model_labels, colors


def _create_styled_boxplot(ax: plt.Axes, boxplot_data: List[np.ndarray],
                          model_labels: List[str], colors: List[tuple]) -> None:
    """Create and style the boxplot on the given axes."""
    box_plot = ax.boxplot(boxplot_data, labels=model_labels, patch_artist=True,
                          vert=True,
                          boxprops=dict(linewidth=1.2),
                          medianprops=dict(linewidth=2, color='darkblue'),
                          whiskerprops=dict(linewidth=1.2),
                          capprops=dict(linewidth=1.2))

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Models (Ranked by Original AIC)')
    ax.set_ylabel('Bootstrap AIC Scores (Lower is Better)')
    ax.set_title(f'Bootstrap AIC Distribution - Top {len(boxplot_data)} Models',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_boxplot_visualization(bootstrap_df: pd.DataFrame,
                                 bootstrap_results: List[Any],
                                 config: Dict[str, Any],
                                 display_plot: bool = True) -> Dict[str, Any]:
    """Generate boxplot visualization for bootstrap AIC distributions."""
    if bootstrap_df.empty:
        raise ValueError(
            "CRITICAL: Empty DataFrame for boxplot generation. "
            "Required action: Ensure bootstrap DataFrame creation succeeds."
        )

    print(f"\n=== BOOTSTRAP AIC BOXPLOT ANALYSIS (TOP MODELS) ===")

    n_models = min(len(bootstrap_results), config.get('models_to_analyze', 15))
    models_order = [f'Model {i+1}' for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(config.get('fig_width', 16), config.get('fig_height', 10)))
    boxplot_data, model_labels, colors = _prepare_boxplot_data(bootstrap_df, models_order)

    if not boxplot_data:
        print("No boxplot data available for models")
        return {'boxplot_created': False, 'models_visualized': 0}

    _create_styled_boxplot(ax, boxplot_data, model_labels, colors)

    plt.tight_layout()
    if display_plot:
        plt.show()

    print(f"SUCCESS: Displayed bootstrap distributions for top {len(boxplot_data)} models")
    return {
        'boxplot_created': True,
        'models_visualized': len(boxplot_data),
        'plot_config': config
    }


def _compute_summary_statistics(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by model."""
    summary_stats = bootstrap_df.groupby('model')['bootstrap_aic'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('q25', lambda x: np.percentile(x, 25)),
        ('q75', lambda x: np.percentile(x, 75))
    ]).round(3)

    stability_map = bootstrap_df.groupby('model')['stability_assessment'].first()
    summary_stats['stability'] = stability_map
    return summary_stats.sort_values('median')


def _display_ranking_comparison(summary_stats_sorted: pd.DataFrame,
                               models_to_analyze: int) -> None:
    """Display ranking comparison table."""
    print(f"\n=== BOOTSTRAP vs ORIGINAL AIC RANKING COMPARISON (TOP {models_to_analyze}) ===")
    print(f"{'Model':<8} {'Original Rank':<13} {'Bootstrap Median':<17} {'Median Rank':<12} {'Stability':<10}")
    print("-" * 80)

    for i, (model_key, stats) in enumerate(summary_stats_sorted.head(models_to_analyze).iterrows()):
        original_rank = int(model_key.replace('Model ', ''))
        print(f"{model_key:<8} {original_rank:<13} {stats['median']:<17.1f} {i + 1:<12} {stats['stability']:<10}")


def calculate_ranking_statistics(bootstrap_df: pd.DataFrame,
                               models_to_analyze: int = 15,
                               display_results: bool = True) -> Dict[str, Any]:
    """Calculate ranking statistics and median AIC comparisons.

    Parameters
    ----------
    bootstrap_df : pd.DataFrame
        Bootstrap visualization DataFrame
    models_to_analyze : int, default 15
        Number of top models to analyze
    display_results : bool, default True
        Whether to display results

    Returns
    -------
    Dict[str, Any]
        Ranking statistics with median AIC rankings
    """
    if bootstrap_df.empty:
        raise ValueError(
            "CRITICAL: Empty DataFrame for ranking statistics. "
            "Required action: Ensure bootstrap DataFrame creation succeeds."
        )

    if display_results:
        print("\nBootstrap Stability Summary Statistics:")

    summary_stats_sorted = _compute_summary_statistics(bootstrap_df)

    if display_results:
        print("Sorted by Median Bootstrap AIC (Model Selection Criterion):")
        print(summary_stats_sorted.head(models_to_analyze)[['mean', 'median', 'std', 'stability']])
        _display_ranking_comparison(summary_stats_sorted, models_to_analyze)

    return {
        'summary_statistics': summary_stats_sorted,
        'median_aic_ranking': summary_stats_sorted.index.tolist(),
        'models_analyzed': min(models_to_analyze, len(summary_stats_sorted))
    }