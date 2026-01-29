"""
Feature Selection Modules for RILA Price Elasticity Analysis.

This package provides atomic functions for AIC-based feature selection
following the established pipeline architecture patterns.

Atomic Functions:
- evaluate_aic_combinations: Pure AIC calculation across feature combinations
- apply_economic_constraints: Pure constraint filtering with business rules
- run_bootstrap_stability: Pure bootstrap stability analysis
- run_feature_selection_pipeline: Orchestration function combining all steps

Design Principles:
- Single responsibility per function (30-50 lines maximum)
- Immutable transformations (data, config) -> result
- Comprehensive error handling with business context
- Full type safety with TypedDict configurations
- Zero regression from existing notebook implementation
"""

__version__ = "1.0.0"

# Import atomic functions from engines subpackage - Fail fast if any import fails
from src.features.selection.engines import (
    # AIC Engine
    evaluate_aic_combinations,
    calculate_aic_for_features,
    generate_feature_combinations,
    # Constraints Engine
    apply_economic_constraints,
    validate_constraint_rule,
    generate_constraint_violations,
    # Bootstrap Engine
    run_bootstrap_stability,
    calculate_bootstrap_metrics,
    assess_model_stability,
)
from src.features.selection.pipeline_orchestrator import run_feature_selection_pipeline, create_pipeline_summary, validate_pipeline_inputs

# Type system
from src.features.selection_types import (
    FeatureSelectionConfig,
    EconomicConstraintConfig,
    BootstrapAnalysisConfig,
    ExperimentConfig,
    FeatureSelectionResults,
    AICResult,
    ConstraintViolation,
    BootstrapResult,
    ConstraintRule,
    ConstraintType,
    create_default_constraint_rules
)

# Notebook interface (main module + split components)
from src.features.selection.notebook_interface import (
    run_feature_selection,
    quick_feature_selection,
    production_feature_selection,
    create_feature_selection_config,
    set_feature_flag,
    get_feature_flags,
    compare_with_original,
    create_feature_selection_report,
    setup_feature_selection_environment,
    configure_analysis_pipeline,
    generate_comprehensive_stability_dashboard,
    export_final_model_selection,
    display_results_summary,
)

# Interface split modules (Phase 6.1, now in interface/)
from src.features.selection.interface.interface_dashboard import (
    generate_comprehensive_stability_dashboard as dashboard_generate,
)
from src.features.selection.interface.interface_display import (
    display_results_summary as display_summary,
    create_feature_selection_report as create_report,
)
from src.features.selection.interface.interface_export import (
    export_final_model_selection as export_selection,
)

# Dual validation (extracted module, now in visualization/)
from src.features.selection.visualization.dual_validation import (
    run_dual_validation_stability_analysis,
    create_dual_validation_config
)

__all__ = [
    # Main orchestration
    "run_feature_selection_pipeline",

    # Atomic functions
    "evaluate_aic_combinations",
    "apply_economic_constraints",
    "run_bootstrap_stability",

    # Supporting functions
    "calculate_aic_for_features",
    "generate_feature_combinations",
    "validate_constraint_rule",
    "generate_constraint_violations",
    "calculate_bootstrap_metrics",
    "assess_model_stability",
    "create_pipeline_summary",
    "validate_pipeline_inputs",

    # Type system
    "FeatureSelectionConfig",
    "EconomicConstraintConfig",
    "BootstrapAnalysisConfig",
    "ExperimentConfig",
    "FeatureSelectionResults",
    "AICResult",
    "ConstraintViolation",
    "BootstrapResult",
    "ConstraintRule",
    "ConstraintType",
    "create_default_constraint_rules",

    # Notebook interface
    "run_feature_selection",
    "quick_feature_selection",
    "production_feature_selection",
    "create_feature_selection_config",
    "set_feature_flag",
    "get_feature_flags",
    "compare_with_original",
    "create_feature_selection_report",
    "setup_feature_selection_environment",
    "configure_analysis_pipeline",
    "generate_comprehensive_stability_dashboard",
    "export_final_model_selection",
    "display_results_summary",

    # Dual validation
    "run_dual_validation_stability_analysis",
    "create_dual_validation_config",
]