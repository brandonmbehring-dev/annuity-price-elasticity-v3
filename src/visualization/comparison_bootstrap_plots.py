"""
Model Comparison Bootstrap Distribution Functions.

This module provides bootstrap distribution visualizations:
- Dict-format bootstrap placeholder
- NamedTuple bootstrap histogram

Part of Phase 6.3 module split.

Module Architecture:
- comparison_helpers.py: Utility functions
- comparison_scatter_plots.py: Scatter plot functions
- comparison_coefficient_analysis.py: Coefficient analysis functions
- comparison_bootstrap_plots.py: Bootstrap distribution functions (this file)
- model_comparison.py: Orchestrator class
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# DICT FORMAT BOOTSTRAP PLOT
# =============================================================================


def plot_dict_format_bootstrap(
    ax: plt.Axes,
    result: Dict[str, Any],
    cv: float,
    colors: Dict[str, str]
) -> None:
    """Plot placeholder for dict-format bootstrap result (no samples)."""
    model_name = result.get('features', 'Unknown')
    ax.text(
        0.5, 0.5,
        f'Model: {model_name}\nCV: {cv:.4f}\n(Distribution data\nnot available)',
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor=colors['light_gray'])
    )


# =============================================================================
# NAMEDTUPLE BOOTSTRAP HISTOGRAM
# =============================================================================


def plot_namedtuple_bootstrap_histogram(
    ax: plt.Axes,
    result: Any,
    rank: int,
    colors: Dict[str, str]
) -> None:
    """Plot histogram for NamedTuple-format bootstrap result with samples."""
    model_name = result.model_features
    bootstrap_samples = result.bootstrap_samples
    stability_metrics = result.stability_metrics
    confidence_intervals = result.confidence_intervals

    if not bootstrap_samples or len(bootstrap_samples) == 0:
        ax.text(
            0.5, 0.5, f'Model: {model_name}\n(No bootstrap samples)',
            ha='center', va='center', transform=ax.transAxes
        )
        return

    aic_values = [
        sample.get('aic', 0) for sample in bootstrap_samples
        if isinstance(sample, dict) and 'aic' in sample
    ]

    if not aic_values:
        ax.text(
            0.5, 0.5, f'Model: {model_name}\n(No valid AIC samples)',
            ha='center', va='center', transform=ax.transAxes
        )
        return

    # Plot histogram
    ax.hist(
        aic_values, bins=min(15, len(aic_values) // 2 + 1),
        alpha=0.7, color=colors['primary'],
        density=True, edgecolor='black', linewidth=0.5
    )

    # Original AIC line
    ax.axvline(
        result.original_aic, color=colors['highlight'],
        linewidth=2, label=f'Original: {result.original_aic:.2f}'
    )

    # 95% CI shading
    if 95 in confidence_intervals:
        ci_95 = confidence_intervals[95]
        ax.axvspan(ci_95[0], ci_95[1], alpha=0.2, color=colors['tertiary'], label='95% CI')

    # Statistics text
    mean_aic = np.mean(aic_values)
    std_aic = np.std(aic_values)
    cv_aic = stability_metrics.get('aic_cv', 0)

    ax.text(
        0.98, 0.98, f'Mean: {mean_aic:.2f}\nStd: {std_aic:.3f}\nCV: {cv_aic:.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('AIC Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name[:20]}...\nStability Rank #{rank}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Dict format
    'plot_dict_format_bootstrap',
    # NamedTuple format
    'plot_namedtuple_bootstrap_histogram',
]
