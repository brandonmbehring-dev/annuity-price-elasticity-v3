"""
CANONICAL Feature Selection Interface - THE PRIMARY PATTERN TO FOLLOW

This is the authoritative interface used by notebooks/01_feature_selection_refactored.ipynb.
When working with feature selection, ALWAYS use this module as your starting point.

CANONICAL FUNCTIONS (use these):
- run_feature_selection: Main interface function
- setup_feature_selection_environment: Environment setup
- export_final_model_selection: Results export
- generate_comprehensive_stability_dashboard: Analysis dashboard

AVOID: Any "enhanced_*" modules (all removed to eliminate choice paralysis)

Usage Pattern:
    from src.features.selection.notebook_interface import setup_feature_selection_environment

This module orchestrates calls to src.features.selection.pipeline_orchestrator for core logic.

Module Split (Phase 6.1):
- interface_environment.py: Environment and import setup
- interface_config.py: Configuration building and feature flags
- interface_execution.py: Pipeline execution and error handling
- interface_validation.py: Validation, comparison, convenience functions
- interface_dashboard.py: Dashboard/stability/visualization functions
- interface_display.py: Display/formatting/HTML report functions
- interface_export.py: Export/MLflow/DVC functions
- notebook_interface.py: Public API orchestrator (this file)
"""

# =============================================================================
# CONTEXT ANCHOR: FEATURE SELECTION INTERFACE OBJECTIVES
# =============================================================================
# PURPOSE: Primary entry point for ALL feature selection operations (AIC, constraints, bootstrap)
# USED BY: notebooks/01_feature_selection_refactored.ipynb (only notebook interface)
# DEPENDENCIES: pipeline_orchestrator (core logic), config_builder (configurations)
# LAST VALIDATED: 2025-01-20 (v5.0 - Phase 6.1 complete module split)
# PATTERN STATUS: CANONICAL (single interface, no competing implementations)
#
# ARCHITECTURAL FLOW: notebook -> interface -> orchestrator -> engines -> results
# SUCCESS CRITERIA: Mathematical equivalence maintained, no interface bypass violations
# INTEGRATION: Works seamlessly with config_builder and pipeline_orchestrator
# MAINTENANCE: Never import engines directly - always use orchestrator for business logic

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - ENVIRONMENT
# =============================================================================

from .interface.interface_environment import (
    setup_feature_selection_environment,
    _get_status_constants,
    _build_core_imports_dict,
    _import_core_libraries,
    _import_atomic_functions,
    _import_mlflow_integration,
    _configure_visualization_environment,
    _initialize_mlflow_environment,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - CONFIGURATION
# =============================================================================

from .interface.interface_config import (
    FEATURE_FLAGS,
    set_feature_flag,
    get_feature_flags,
    configure_analysis_pipeline,
    create_dual_validation_config,
    create_feature_selection_config,
    _update_pipeline_bootstrap_config,
    _update_pipeline_experiment_config,
    _add_pipeline_visualization_config,
    _add_pipeline_status_indicators,
    _create_pipeline_fallback_config,
    _get_dual_validation_defaults,
    _update_configs_for_dual_validation,
    _build_feature_config,
    _build_constraint_config_wrapper as _build_constraint_config,
    _build_bootstrap_config_wrapper as _build_bootstrap_config,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - EXECUTION
# =============================================================================

from .interface.interface_execution import (
    run_feature_selection,
    ATOMIC_FUNCTIONS_AVAILABLE,
    _prepare_feature_selection_parameters,
    _log_pipeline_status,
    _execute_feature_selection_pipeline,
    _format_pipeline_results,
    _run_atomic_pipeline,
    _create_error_aic_result,
    _create_error_feature_selection_results,
    _create_error_fallback_results,
    _handle_pipeline_error,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - VALIDATION
# =============================================================================

from .interface.interface_validation import (
    run_dual_validation_stability_analysis,
    compare_with_original,
    quick_feature_selection,
    production_feature_selection,
    _validate_dual_analysis_inputs,
    _print_dual_analysis_header,
    _run_new_implementation,
    _compare_model_counts,
    _compute_aic_differences,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - DASHBOARD
# =============================================================================

from .interface.interface_dashboard import (
    generate_comprehensive_stability_dashboard,
    create_comprehensive_scoring_system,
    generate_final_recommendations,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - DISPLAY
# =============================================================================

from .interface.interface_display import (
    display_results_summary,
    display_dual_validation_results as _display_dual_validation_results,
    create_feature_selection_report,
    _display_comparison_results,
)

# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES - EXPORT
# =============================================================================

from .interface.interface_export import (
    export_final_model_selection,
    save_dual_validation_results,
    _convert_numpy_types,
    _export_model_metadata,
    _export_bootstrap_results,
)
# Alias for backward compatibility
_save_dual_validation_results = save_dual_validation_results


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Display current feature flags on import
print("Feature Selection Notebook Interface Loaded")
print("Current Feature Flags:")
for flag, value in FEATURE_FLAGS.items():
    status = "ENABLED" if value else "DISABLED"
    print(f"  {status}: {flag}")
print("\nUse set_feature_flag() to modify settings")
print("Use get_feature_flags() to view current settings")


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # === PRIMARY PUBLIC API ===
    # Environment setup
    "setup_feature_selection_environment",
    # Configuration
    "configure_analysis_pipeline",
    "create_dual_validation_config",
    "create_feature_selection_config",
    # Feature flags
    "FEATURE_FLAGS",
    "set_feature_flag",
    "get_feature_flags",
    # Execution
    "run_feature_selection",
    "ATOMIC_FUNCTIONS_AVAILABLE",
    # Validation
    "run_dual_validation_stability_analysis",
    "compare_with_original",
    # Convenience
    "quick_feature_selection",
    "production_feature_selection",
    # Dashboard
    "generate_comprehensive_stability_dashboard",
    # Display
    "display_results_summary",
    "create_feature_selection_report",
    # Export
    "export_final_model_selection",
    # === INTERNAL HELPERS (for backward compatibility) ===
    # Environment helpers
    "_get_status_constants",
    "_build_core_imports_dict",
    "_import_core_libraries",
    "_import_atomic_functions",
    "_import_mlflow_integration",
    "_configure_visualization_environment",
    "_initialize_mlflow_environment",
    # Config helpers
    "_update_pipeline_bootstrap_config",
    "_update_pipeline_experiment_config",
    "_add_pipeline_visualization_config",
    "_add_pipeline_status_indicators",
    "_create_pipeline_fallback_config",
    "_get_dual_validation_defaults",
    "_update_configs_for_dual_validation",
    "_build_feature_config",
    "_build_constraint_config",
    "_build_bootstrap_config",
    # Execution helpers
    "_prepare_feature_selection_parameters",
    "_log_pipeline_status",
    "_execute_feature_selection_pipeline",
    "_format_pipeline_results",
    "_run_atomic_pipeline",
    "_create_error_aic_result",
    "_create_error_feature_selection_results",
    "_create_error_fallback_results",
    "_handle_pipeline_error",
    # Validation helpers
    "_validate_dual_analysis_inputs",
    "_print_dual_analysis_header",
    "_run_new_implementation",
    "_compare_model_counts",
    "_compute_aic_differences",
    # Dashboard helpers
    "create_comprehensive_scoring_system",
    "generate_final_recommendations",
    # Display helpers
    "_display_dual_validation_results",
    "_display_comparison_results",
    # Export helpers
    "_save_dual_validation_results",
    "_convert_numpy_types",
    "_export_model_metadata",
    "_export_bootstrap_results",
]
