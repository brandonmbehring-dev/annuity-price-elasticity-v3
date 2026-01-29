"""
Feature Selection Visualization Subpackage.

Provides visualization tools for feature selection analysis:
- bootstrap_visualization_detailed: Detailed bootstrap result plots (violin, boxplot)
- bootstrap_visualization_analysis: Bootstrap analysis dashboards
- dual_validation: Dual validation stability visualization
"""

# Detailed bootstrap visualizations
from src.features.selection.visualization.bootstrap_visualization_detailed import (
    create_bootstrap_dataframe,
    generate_violin_plot_visualization,
    generate_boxplot_visualization,
    calculate_ranking_statistics,
)

# Bootstrap analysis visualizations
from src.features.selection.visualization.bootstrap_visualization_analysis import (
    prepare_bootstrap_visualization_data,
    create_aic_distribution_visualizations,
)

# Dual validation
from src.features.selection.visualization.dual_validation import (
    run_dual_validation_stability_analysis,
    create_dual_validation_config,
)

__all__ = [
    # Detailed bootstrap
    "create_bootstrap_dataframe",
    "generate_violin_plot_visualization",
    "generate_boxplot_visualization",
    "calculate_ranking_statistics",
    # Analysis
    "prepare_bootstrap_visualization_data",
    "create_aic_distribution_visualizations",
    # Dual validation
    "run_dual_validation_stability_analysis",
    "create_dual_validation_config",
]
