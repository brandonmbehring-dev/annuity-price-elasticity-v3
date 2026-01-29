"""
Model Comparison Visualization Module for Feature Selection V2.

This module provides comprehensive model comparison visualizations for
detailed technical analysis and model selection validation, including:
1. Multi-dimensional model performance comparisons
2. Feature importance and coefficient analysis
3. Bootstrap stability comparisons across models
4. Information criteria trade-off analysis
5. Model selection decision trees and rationale

Module Architecture (Phase 6.3 Split):
- comparison_helpers.py: Utility functions
- comparison_scatter_plots.py: Scatter plot functions
- comparison_coefficient_analysis.py: Coefficient analysis functions
- comparison_bootstrap_plots.py: Bootstrap distribution functions
- model_comparison.py: Orchestrator class (this file)

Design Principles:
- Technical depth for data science and analytics teams
- Comprehensive comparison across multiple evaluation dimensions
- Interactive-style plots for detailed model exploration
- Statistical rigor with uncertainty quantification
- Clear visualization of model selection rationale
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# Import helper functions
from .comparison_helpers import (
    create_model_selection_summary,
    sort_bootstrap_by_stability,
)

# Import scatter plot functions
from .comparison_scatter_plots import (
    plot_aic_vs_r2_scatter,
    plot_bootstrap_stability_scatter,
    plot_complexity_distribution,
    plot_decision_summary,
    plot_feature_frequency,
    plot_information_criteria_heatmap,
    plot_ranking_correlation_matrix,
)

# Import coefficient analysis functions
from .comparison_coefficient_analysis import (
    plot_coefficient_heatmap,
    plot_coefficient_uncertainty,
    plot_economic_constraint_validation,
    plot_sign_consistency_scatter,
)

# Import bootstrap plot functions
from .comparison_bootstrap_plots import (
    plot_dict_format_bootstrap,
    plot_namedtuple_bootstrap_histogram,
)


# Configure matplotlib for technical analysis
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

logger = logging.getLogger(__name__)


class ModelComparisonPlots:
    """
    Advanced model comparison visualization suite.

    Provides technical visualizations for comprehensive model evaluation
    and comparison across multiple statistical dimensions.

    Uses delegated functions from split modules for single responsibility.
    """

    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize model comparison plotting framework.

        Args:
            style: Matplotlib style for technical plots
        """
        self.style = style

        # Technical color scheme
        self.colors = {
            'primary': '#1f77b4',      # Blue
            'secondary': '#ff7f0e',    # Orange
            'tertiary': '#2ca02c',     # Green
            'quaternary': '#d62728',   # Red
            'quinary': '#9467bd',      # Purple
            'senary': '#8c564b',       # Brown
            'septenary': '#e377c2',    # Pink
            'octonary': '#7f7f7f',     # Gray
            'nonary': '#bcbd22',       # Olive
            'denary': '#17becf',       # Cyan
            'light_gray': '#f0f0f0',
            'dark_gray': '#404040',
            'highlight': '#ff4444',
            'success': '#00aa00'
        }

        # Plot configuration
        self.plot_config = {
            'figure_size_large': (16, 12),
            'figure_size_wide': (14, 8),
            'figure_size_standard': (12, 9),
            'figure_size_compact': (10, 6),
            'subplot_spacing': 0.25,
            'title_fontsize': 14,
            'label_fontsize': 11
        }

    # =========================================================================
    # MULTI-CRITERIA COMPARISON MATRIX
    # =========================================================================

    def _populate_comparison_matrix_subplots(self, fig: plt.Figure, gs: GridSpec,
                                             top_models: pd.DataFrame, aic_results: pd.DataFrame,
                                             information_criteria_results: List[Dict[str, Any]],
                                             bootstrap_results: List[Dict[str, Any]]) -> None:
        """Populate all subplots in the multi-criteria comparison matrix."""
        plot_aic_vs_r2_scatter(fig.add_subplot(gs[0, :2]), top_models, self.colors)
        plot_information_criteria_heatmap(fig.add_subplot(gs[0, 2:]), information_criteria_results, self.colors)
        plot_bootstrap_stability_scatter(fig.add_subplot(gs[1, :2]), bootstrap_results, self.colors)
        plot_complexity_distribution(fig.add_subplot(gs[1, 2:]), top_models, self.colors)
        plot_ranking_correlation_matrix(fig.add_subplot(gs[2, :2]), top_models, bootstrap_results)
        plot_feature_frequency(fig.add_subplot(gs[2, 2:]), top_models, self.colors)
        plot_decision_summary(fig.add_subplot(gs[3, :]), aic_results, information_criteria_results,
                              bootstrap_results, self.colors, create_model_selection_summary)

    def create_multi_criteria_comparison_matrix(
        self, aic_results: pd.DataFrame, information_criteria_results: List[Dict[str, Any]],
        bootstrap_results: List[Dict[str, Any]], save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create comprehensive multi-criteria model comparison matrix."""
        fig = plt.figure(figsize=self.plot_config['figure_size_large'])
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Multi-Criteria Model Comparison Matrix', fontsize=16, fontweight='bold', y=0.95)

        top_models = aic_results.head(15).copy()
        self._populate_comparison_matrix_subplots(fig, gs, top_models, aic_results,
                                                   information_criteria_results, bootstrap_results)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Multi-criteria comparison matrix saved to {save_path}")
        return fig

    # =========================================================================
    # BOOTSTRAP DISTRIBUTION COMPARISON
    # =========================================================================

    def create_bootstrap_distribution_comparison(
        self,
        bootstrap_results: List[Dict[str, Any]],
        top_n_models: int = 6,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create detailed bootstrap distribution comparison for top models."""
        fig, axes = plt.subplots(2, 3, figsize=self.plot_config['figure_size_wide'])
        fig.suptitle(
            'Bootstrap Distribution Comparison: Top Model Stability Analysis',
            fontsize=14, fontweight='bold', y=0.95
        )
        axes = axes.flatten()

        if not bootstrap_results:
            for ax in axes:
                ax.text(
                    0.5, 0.5, 'No Bootstrap\nData Available',
                    ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=self.colors['light_gray'])
                )
            return fig

        sorted_results = sort_bootstrap_by_stability(bootstrap_results, top_n_models)

        for i, (cv, result) in enumerate(sorted_results[:6]):
            ax = axes[i]

            if isinstance(result, dict):
                plot_dict_format_bootstrap(ax, result, cv, self.colors)
            else:
                plot_namedtuple_bootstrap_histogram(ax, result, rank=i + 1, colors=self.colors)

        # Hide unused subplots
        for j in range(len(sorted_results), 6):
            axes[j].axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Bootstrap distribution comparison saved to {save_path}")

        return fig

    # =========================================================================
    # FEATURE COEFFICIENT ANALYSIS
    # =========================================================================

    def create_feature_coefficient_analysis(
        self,
        aic_results: pd.DataFrame,
        coefficient_stability: Dict[str, Any],
        top_n_models: int = 8,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create feature coefficient analysis and stability visualization.

        Shows coefficient values, signs, and stability across models.

        Args:
            aic_results: Model evaluation results
            coefficient_stability: Coefficient stability analysis from bootstrap
            top_n_models: Number of models to analyze
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure with coefficient analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.plot_config['figure_size_standard']
        )
        fig.suptitle(
            'Feature Coefficient Analysis: Values and Stability',
            fontsize=14, fontweight='bold', y=0.95
        )

        top_models = aic_results.head(top_n_models).copy()

        # 1. Coefficient Values Heatmap
        plot_coefficient_heatmap(ax1, top_models, self.colors)

        # 2. Coefficient Sign Consistency
        plot_sign_consistency_scatter(ax2, coefficient_stability, top_models, self.colors)

        # 3. Economic Constraint Validation
        plot_economic_constraint_validation(ax3, top_models, self.colors)

        # 4. Coefficient Uncertainty Quantification
        plot_coefficient_uncertainty(ax4, coefficient_stability, top_models, self.colors)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Feature coefficient analysis saved to {save_path}")

        return fig

    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS (delegate to module functions)
    # =========================================================================

    def _find_pareto_frontier(self, x_values: Any, y_values: Any) -> Any:
        """Backward compatibility wrapper."""
        from .comparison_helpers import find_pareto_frontier
        return find_pareto_frontier(x_values, y_values)

    def _create_model_selection_summary(self, aic_results: Any, info_criteria: Any, bootstrap: Any) -> str:
        """Backward compatibility wrapper."""
        return create_model_selection_summary(aic_results, info_criteria, bootstrap)

    def _extract_bootstrap_metrics(self, bootstrap_results: Any) -> Any:
        """Backward compatibility wrapper."""
        from .comparison_helpers import extract_bootstrap_metrics
        return extract_bootstrap_metrics(bootstrap_results)


def create_model_comparison_report(
    analysis_results: Dict[str, Any],
    output_dir: Path,
    file_prefix: str = "model_comparison"
) -> Dict[str, Path]:
    """
    Generate complete model comparison report with all visualizations.

    Args:
        analysis_results: Complete analysis results from Part 1
        output_dir: Directory to save visualization files
        file_prefix: Prefix for output files

    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model comparison framework
    plotter = ModelComparisonPlots()

    # Generate all model comparison visualizations
    report_files = {}

    try:
        # Extract data for visualizations
        aic_results = analysis_results.get('aic_results', pd.DataFrame())
        enhanced_metrics = analysis_results.get('enhanced_metrics', {})
        bootstrap_results = analysis_results.get('bootstrap_results', {})

        information_criteria_results = enhanced_metrics.get('information_criteria_analysis', [])
        bootstrap_data = bootstrap_results.get('block_bootstrap_results', [])

        # 1. Multi-Criteria Comparison Matrix
        if not aic_results.empty:
            matrix_path = output_dir / f"{file_prefix}_multi_criteria_matrix.png"
            fig_matrix = plotter.create_multi_criteria_comparison_matrix(
                aic_results, information_criteria_results, bootstrap_data,
                save_path=matrix_path
            )
            plt.close(fig_matrix)
            report_files['multi_criteria_matrix'] = matrix_path

        # 2. Bootstrap Distribution Comparison
        if bootstrap_data:
            bootstrap_path = output_dir / f"{file_prefix}_bootstrap_distributions.png"
            fig_bootstrap = plotter.create_bootstrap_distribution_comparison(
                bootstrap_data, save_path=bootstrap_path
            )
            plt.close(fig_bootstrap)
            report_files['bootstrap_distributions'] = bootstrap_path

        # 3. Feature Coefficient Analysis
        if not aic_results.empty:
            coeff_path = output_dir / f"{file_prefix}_coefficient_analysis.png"

            # Extract coefficient stability if available
            coefficient_stability = {}
            if 'coefficient_stability_analysis' in bootstrap_results:
                coefficient_stability = bootstrap_results['coefficient_stability_analysis']

            fig_coeff = plotter.create_feature_coefficient_analysis(
                aic_results, coefficient_stability, save_path=coeff_path
            )
            plt.close(fig_coeff)
            report_files['coefficient_analysis'] = coeff_path

        logger.info(f"Model comparison report generated with {len(report_files)} visualizations")
        return report_files

    except Exception as e:
        logger.error(f"Error generating model comparison report: {e}")
        return report_files


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'ModelComparisonPlots',
    'create_model_comparison_report',
]
