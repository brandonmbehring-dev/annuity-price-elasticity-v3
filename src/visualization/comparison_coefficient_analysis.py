"""
Model Comparison Coefficient Analysis Functions.

This module provides coefficient analysis visualizations:
- Coefficient heatmap
- Sign consistency scatter
- Economic constraint validation
- Coefficient uncertainty quantification

Part of Phase 6.3 module split.

Module Architecture:
- comparison_helpers.py: Utility functions
- comparison_scatter_plots.py: Scatter plot functions
- comparison_coefficient_analysis.py: Coefficient analysis functions (this file)
- comparison_bootstrap_plots.py: Bootstrap distribution functions
- model_comparison.py: Orchestrator class
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .comparison_helpers import (
    compute_constraint_summary,
    extract_sign_consistency_data,
    extract_uncertainty_data,
)


# =============================================================================
# COEFFICIENT HEATMAP
# =============================================================================


def plot_coefficient_heatmap(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot coefficient values heatmap across models."""
    if 'coefficients' not in top_models.columns:
        ax.text(
            0.5, 0.5, 'Coefficient Data\nNot Available',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
        )
        return

    # Extract coefficient information
    model_labels = []
    feature_names = set()

    for _, model in top_models.iterrows():
        model_labels.append(model['features'][:25])
        coefficients = model.get('coefficients', {})
        for coef_name in coefficients.keys():
            if coef_name != 'const':
                feature_names.add(coef_name)

    feature_names = sorted(list(feature_names))

    # Build coefficient matrix
    coef_matrix = []
    for _, model in top_models.iterrows():
        coefficients = model.get('coefficients', {})
        coef_row = [coefficients.get(feature, 0) for feature in feature_names]
        coef_matrix.append(coef_row)

    if not coef_matrix or not feature_names:
        return

    coef_array = np.array(coef_matrix)

    im = ax.imshow(coef_array, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(
        [name[:15] + '...' if len(name) > 15 else name for name in feature_names],
        rotation=45, ha='right'
    )
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title('Coefficient Values Across Models')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coefficient Value')

    # Add text annotations for small matrices
    if len(feature_names) <= 4 and len(model_labels) <= 6:
        for i in range(len(model_labels)):
            for j in range(len(feature_names)):
                color = "white" if abs(coef_array[i, j]) > np.std(coef_array) else "black"
                ax.text(j, i, f'{coef_array[i, j]:.3f}', ha="center", va="center", color=color, fontsize=8)


# =============================================================================
# SIGN CONSISTENCY SCATTER
# =============================================================================


def plot_sign_consistency_scatter(
    ax: plt.Axes,
    coefficient_stability: Dict[str, Any],
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot coefficient sign consistency scatter."""
    if not coefficient_stability:
        ax.text(
            0.5, 0.5, 'Coefficient Stability\nData Not Available',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
        )
        return

    features_analyzed, sign_consistency, mean_values = extract_sign_consistency_data(
        coefficient_stability, top_models
    )

    if not features_analyzed:
        return

    ax.scatter(
        sign_consistency, mean_values,
        c=range(len(features_analyzed)), cmap='plasma',
        s=80, alpha=0.7, edgecolors='black'
    )

    ax.set_xlabel('Sign Consistency (0-1)')
    ax.set_ylabel('Coefficient Magnitude')
    ax.set_title('Coefficient Stability vs Magnitude')
    ax.grid(True, alpha=0.3)

    # Annotate points
    if len(features_analyzed) <= 8:
        for x, y, label in zip(sign_consistency, mean_values, features_analyzed):
            ax.annotate(
                label, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=7, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )

    # Threshold lines
    ax.axvline(0.8, color=colors['success'], linestyle='--', alpha=0.7, label='Good Consistency (0.8)')
    ax.axvline(0.6, color=colors['secondary'], linestyle='--', alpha=0.7, label='Moderate Consistency (0.6)')
    ax.legend(fontsize=8)


# =============================================================================
# ECONOMIC CONSTRAINT VALIDATION
# =============================================================================


def plot_economic_constraint_validation(
    ax: plt.Axes,
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot economic constraint validation bar chart."""
    constraint_summary = compute_constraint_summary(top_models)

    if not constraint_summary:
        return

    models_short = [item['model'] for item in constraint_summary]
    pass_rates = [item['pass_rate'] for item in constraint_summary]

    colors_constraint = [
        colors['success'] if rate >= 0.8 else
        colors['secondary'] if rate >= 0.6 else
        colors['highlight']
        for rate in pass_rates
    ]

    bars = ax.barh(range(len(models_short)), pass_rates, color=colors_constraint, alpha=0.7)
    ax.set_yticks(range(len(models_short)))
    ax.set_yticklabels(models_short, fontsize=8)
    ax.set_xlabel('Economic Constraint Pass Rate')
    ax.set_title('Economic Logic Validation')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')

    for bar, rate in zip(bars, pass_rates):
        ax.text(rate + 0.02, bar.get_y() + bar.get_height() / 2, f'{rate:.1%}', va='center', fontsize=8)


# =============================================================================
# COEFFICIENT UNCERTAINTY
# =============================================================================


def plot_coefficient_uncertainty(
    ax: plt.Axes,
    coefficient_stability: Dict[str, Any],
    top_models: pd.DataFrame,
    colors: Dict[str, str]
) -> None:
    """Plot coefficient uncertainty quantification."""
    if not coefficient_stability:
        ax.text(
            0.5, 0.5, 'Coefficient Uncertainty\nData Not Available',
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
        )
        return

    uncertainty_data = extract_uncertainty_data(coefficient_stability, top_models)

    if not uncertainty_data:
        return

    df_uncertainty = pd.DataFrame(uncertainty_data)
    unique_features = df_uncertainty['feature'].unique()[:6]

    if len(unique_features) == 0:
        return

    x_pos = np.arange(len(unique_features))
    models = df_uncertainty['model'].unique()
    width = 0.8 / len(models)

    color_keys = list(colors.keys())
    for i, model in enumerate(models):
        model_data = df_uncertainty[df_uncertainty['model'] == model]
        cvs = [
            model_data[model_data['feature'] == f]['uncertainty_score'].iloc[0]
            if len(model_data[model_data['feature'] == f]) > 0 else 0
            for f in unique_features
        ]

        ax.bar(
            x_pos + i * width, cvs, width, label=model, alpha=0.7,
            color=colors[color_keys[i % len(color_keys)]]
        )

    ax.set_xlabel('Features')
    ax.set_ylabel('Coefficient Uncertainty (CV)')
    ax.set_title('Coefficient Uncertainty by Feature')
    ax.set_xticks(x_pos + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in unique_features], rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.1, color=colors['success'], linestyle='--', alpha=0.7)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Coefficient heatmap
    'plot_coefficient_heatmap',
    # Sign consistency
    'plot_sign_consistency_scatter',
    # Economic constraint
    'plot_economic_constraint_validation',
    # Coefficient uncertainty
    'plot_coefficient_uncertainty',
]
