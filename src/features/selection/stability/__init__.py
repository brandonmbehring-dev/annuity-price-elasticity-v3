"""
Feature Selection Stability Analysis Subpackage.

Provides stability analysis tools for feature selection:
- stability_analysis: Core stability analysis functions
- bootstrap_stability_analysis: Advanced bootstrap stability
- stability_win_rate / bootstrap_win_rate_analysis: Win rate metrics
- stability_ir / information_ratio_analysis: Information ratio metrics
- stability_visualizations: Stability visualizations
"""

# Core stability analysis
from src.features.selection.stability.stability_analysis import (
    run_bootstrap_stability_analysis,
    run_stability_analysis,  # Backward-compat alias (TODO: remove after v2.1)
    calculate_win_rates,
    analyze_information_ratios,
    evaluate_feature_consistency,
    generate_stability_metrics,
    validate_bootstrap_results,
    aggregate_stability_insights,
    format_stability_outputs,
)

# Bootstrap stability
from src.features.selection.stability.bootstrap_stability_analysis import (
    run_advanced_stability_analysis,
)

# Win rate analysis
from src.features.selection.stability.stability_win_rate import (
    calculate_bootstrap_win_rates,
)
from src.features.selection.stability.bootstrap_win_rate_analysis import (
    run_bootstrap_win_rate_analysis,
)

# Information ratio analysis
from src.features.selection.stability.stability_ir import (
    calculate_information_ratio_analysis,
)
from src.features.selection.stability.information_ratio_analysis import (
    run_information_ratio_analysis,
)

# Visualizations
from src.features.selection.stability.stability_visualizations import (
    create_advanced_visualizations,
)

__all__ = [
    # Core stability analysis (from stability_analysis.py)
    "run_bootstrap_stability_analysis",
    "run_stability_analysis",  # Backward-compat alias
    "calculate_win_rates",
    "analyze_information_ratios",
    "evaluate_feature_consistency",
    "generate_stability_metrics",
    "validate_bootstrap_results",
    "aggregate_stability_insights",
    "format_stability_outputs",
    # Bootstrap stability
    "run_advanced_stability_analysis",
    # Win rate analysis
    "calculate_bootstrap_win_rates",
    "run_bootstrap_win_rate_analysis",
    # Information ratio analysis
    "calculate_information_ratio_analysis",
    "run_information_ratio_analysis",
    # Visualizations
    "create_advanced_visualizations",
]
