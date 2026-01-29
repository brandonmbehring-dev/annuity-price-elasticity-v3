"""
Model Comparison Scatter Plot Functions.

This module provides scatter plot visualizations for model comparison:
- AIC vs R² scatter plot
- Bootstrap stability scatter
- Model complexity distribution
- Feature frequency analysis
- Ranking correlation matrix

Part of Phase 6.3 module split.

Module Architecture:
- comparison_helpers.py: Utility functions
- comparison_scatter_plots.py: Scatter plot functions (this file)
- comparison_coefficient_analysis.py: Coefficient analysis functions
- comparison_bootstrap_plots.py: Bootstrap distribution functions
- model_comparison.py: Orchestrator class
"""

from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .comparison_helpers import (
    compute_stability_ranking,
    extract_bootstrap_metrics,
    find_pareto_frontier,
)


# =============================================================================
# AIC VS R² SCATTER
# =============================================================================


def plot_aic_vs_r2_scatter(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot AIC vs R² scatter with model complexity coloring."""
    scatter = ax.scatter(
        top_models['aic'], top_models['r_squared'],
        c=top_models['n_features'], cmap='viridis',
        s=80, alpha=0.7, edgecolors='black', linewidth=0.5
    )

    ax.set_xlabel('AIC Score (lower is better)')
    ax.set_ylabel('R-squared (higher is better)')
    ax.set_title('Performance vs Model Quality Trade-off')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Features')

    # Annotate Pareto frontier models
    pareto_models = find_pareto_frontier(
        top_models['aic'].values, top_models['r_squared'].values
    )
    pareto_indices = top_models.index[pareto_models]

    ax.scatter(
        top_models.loc[pareto_indices, 'aic'],
        top_models.loc[pareto_indices, 'r_squared'],
        s=150, facecolors='none', edgecolors=colors['highlight'],
        linewidth=3, label='Pareto Optimal'
    )
    ax.legend()


# =============================================================================
# INFORMATION CRITERIA HEATMAP
# =============================================================================


def plot_information_criteria_heatmap(
    ax: plt.Axes,
    information_criteria_results: List[Dict[str, Any]],
    colors: Dict[str, str]
) -> None:
    """Plot information criteria comparison heatmap."""
    if not information_criteria_results:
        ax.text(
            0.5, 0.5, 'Information Criteria\nData Not Available',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
        )
        return

    criteria_data = []
    model_labels = []

    for result in information_criteria_results[:10]:
        if hasattr(result, 'model_features'):
            model_labels.append(result.model_features)
            criteria_values = result.criteria_values

            normalized_values = [
                criteria_values.get(criterion, 0)
                for criterion in ['aic', 'bic', 'hqic', 'caic']
            ]
            criteria_data.append(normalized_values)

    if not criteria_data:
        return

    criteria_matrix = np.array(criteria_data)

    # Normalize each column separately
    for j in range(criteria_matrix.shape[1]):
        col = criteria_matrix[:, j]
        if col.max() > col.min():
            criteria_matrix[:, j] = (col - col.min()) / (col.max() - col.min())

    im = ax.imshow(criteria_matrix, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['AIC', 'BIC', 'HQIC', 'CAIC'])
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(
        [label[:20] + '...' if len(label) > 20 else label for label in model_labels],
        fontsize=8
    )
    ax.set_title('Information Criteria Comparison\n(Red=Worse, Blue=Better)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Score')


# =============================================================================
# BOOTSTRAP STABILITY SCATTER
# =============================================================================


def plot_bootstrap_stability_scatter(
    ax: plt.Axes,
    bootstrap_results: List[Dict[str, Any]],
    colors: Dict[str, str]
) -> None:
    """Plot bootstrap stability as CV vs success rate scatter."""
    if not bootstrap_results:
        ax.text(
            0.5, 0.5, 'Bootstrap Stability\nData Not Available',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
        )
        return

    model_names, stability_cvs, success_rates = extract_bootstrap_metrics(
        bootstrap_results[:10]
    )

    if not model_names:
        return

    bubble_sizes = [100 + i * 20 for i in range(len(model_names))]

    ax.scatter(
        stability_cvs, success_rates, s=bubble_sizes,
        alpha=0.6, c=range(len(model_names)), cmap='plasma',
        edgecolors='black', linewidth=1
    )

    ax.set_xlabel('AIC Coefficient of Variation (lower is better)')
    ax.set_ylabel('Bootstrap Success Rate (higher is better)')
    ax.set_title('Model Stability: CV vs Success Rate')
    ax.grid(True, alpha=0.3)

    # Annotate top 3 most stable models
    if stability_cvs:
        stable_indices = np.argsort(stability_cvs)[:3]
        for idx in stable_indices:
            ax.annotate(
                model_names[idx][:15],
                (stability_cvs[idx], success_rates[idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )


# =============================================================================
# COMPLEXITY DISTRIBUTION
# =============================================================================


def plot_complexity_distribution(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot model complexity vs average performance."""
    complexity_groups = top_models.groupby('n_features').agg({
        'aic': ['count', 'mean', 'std', 'min'],
        'r_squared': ['mean', 'std', 'max']
    }).round(4)
    complexity_groups.columns = ['_'.join(col).strip() for col in complexity_groups.columns]

    if len(complexity_groups) <= 1:
        return

    x_pos = complexity_groups.index

    bars = ax.bar(
        x_pos, complexity_groups['aic_mean'],
        yerr=complexity_groups['aic_std'],
        alpha=0.7, color=colors['primary'], capsize=5
    )

    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Average AIC Score')
    ax.set_title('Model Complexity vs Average Performance')
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels on bars
    for bar, count in zip(bars, complexity_groups['aic_count']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + complexity_groups['aic_std'].iloc[0] + 10,
            f'n={int(count)}', ha='center', va='bottom', fontsize=8
        )

    # Twin axis for R²
    ax_twin = ax.twinx()
    ax_twin.plot(x_pos, complexity_groups['r_squared_mean'], 'ro-', linewidth=2, markersize=8)
    ax_twin.set_ylabel('Average R-squared', color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')


# =============================================================================
# RANKING CORRELATION MATRIX
# =============================================================================


def plot_ranking_correlation_matrix(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    bootstrap_results: List[Dict[str, Any]]
) -> None:
    """Plot correlation matrix of different ranking criteria."""
    ranking_data = []
    ranking_labels = []

    # AIC ranking
    ranking_data.append(top_models['aic'].rank().values)
    ranking_labels.append('AIC')

    # R² ranking (inverse - higher is better)
    ranking_data.append((-top_models['r_squared']).rank().values)
    ranking_labels.append('R²')

    # Bootstrap stability ranking
    if bootstrap_results:
        stability_ranking = compute_stability_ranking(top_models, bootstrap_results)
        if stability_ranking is not None:
            ranking_data.append(stability_ranking)
            ranking_labels.append('Bootstrap CV')

    if len(ranking_data) < 2:
        return

    ranking_matrix = np.array(ranking_data).T
    corr_matrix = np.corrcoef(ranking_matrix.T)

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(ranking_labels)))
    ax.set_xticklabels(ranking_labels)
    ax.set_yticks(range(len(ranking_labels)))
    ax.set_yticklabels(ranking_labels)
    ax.set_title('Ranking Correlation Matrix')

    # Add correlation values as text
    for i in range(len(ranking_labels)):
        for j in range(len(ranking_labels)):
            ax.text(
                j, i, f'{corr_matrix[i, j]:.2f}',
                ha="center", va="center", color="black", fontsize=10
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')


# =============================================================================
# FEATURE FREQUENCY ANALYSIS
# =============================================================================


def plot_feature_frequency(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot feature frequency analysis in top models."""
    feature_freq = {}
    for _, model in top_models.iterrows():
        features = model['features'].split('+')
        for feature in features:
            feature = feature.strip()
            feature_freq[feature] = feature_freq.get(feature, 0) + 1

    if not feature_freq:
        return

    sorted_pairs = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
    features_sorted = [pair[0] for pair in sorted_pairs]
    frequencies_sorted = [pair[1] for pair in sorted_pairs]

    y_pos = np.arange(len(features_sorted))
    bars = ax.barh(y_pos, frequencies_sorted, color=colors['tertiary'], alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f[:20] + '...' if len(f) > 20 else f for f in features_sorted],
        fontsize=9
    )
    ax.set_xlabel('Frequency in Top Models')
    ax.set_title('Feature Popularity Analysis')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, freq in zip(bars, frequencies_sorted):
        ax.text(freq + 0.1, bar.get_y() + bar.get_height() / 2, f'{freq}', va='center', fontsize=9)


# =============================================================================
# DECISION SUMMARY
# =============================================================================


def plot_decision_summary(
    ax: plt.Axes,
    aic_results: pd.DataFrame,
    information_criteria_results: List[Dict[str, Any]],
    bootstrap_results: List[Dict[str, Any]],
    colors: Dict[str, str],
    create_summary_func: Callable[..., str]
) -> None:
    """Plot model selection decision summary text."""
    ax.axis('off')
    decision_summary = create_summary_func(
        aic_results, information_criteria_results, bootstrap_results
    )
    ax.text(
        0.02, 0.95, decision_summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor=colors['light_gray'], alpha=0.9)
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # AIC vs R²
    'plot_aic_vs_r2_scatter',
    # Information criteria
    'plot_information_criteria_heatmap',
    # Bootstrap stability
    'plot_bootstrap_stability_scatter',
    # Complexity distribution
    'plot_complexity_distribution',
    # Ranking correlation
    'plot_ranking_correlation_matrix',
    # Feature frequency
    'plot_feature_frequency',
    # Decision summary
    'plot_decision_summary',
]
