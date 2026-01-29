"""
Feature Selection Interface Subpackage.

Provides the notebook interface API surface for feature selection:
- interface_config: Configuration creation and management
- interface_dashboard: Comprehensive stability dashboards
- interface_display: Results display and reporting
- interface_environment: Environment setup and validation
- interface_execution: Pipeline execution coordination
- interface_export: Results export and persistence
- interface_validation: Input/output validation
"""

# Configuration functions
from src.features.selection.interface.interface_config import (
    set_feature_flag,
    get_feature_flags,
    configure_analysis_pipeline,
    create_feature_selection_config,
    create_dual_validation_config,
)

# Dashboard functions
from src.features.selection.interface.interface_dashboard import (
    generate_comprehensive_stability_dashboard,
)

# Display functions
from src.features.selection.interface.interface_display import (
    display_results_summary,
    create_feature_selection_report,
    display_dual_validation_results,
    _display_dual_validation_header,
    _display_dual_validation_metadata,
    _display_dual_validation_best_model,
    _display_dual_validation_grade_distribution,
    _display_dual_validation_top_models_table,
    _display_dual_validation_recommendations,
)

# Environment setup
from src.features.selection.interface.interface_environment import (
    setup_feature_selection_environment,
)

# Execution functions
from src.features.selection.interface.interface_execution import (
    run_feature_selection,
)

# Export functions
from src.features.selection.interface.interface_export import (
    export_final_model_selection,
    save_dual_validation_results,
)

# Validation/convenience functions
from src.features.selection.interface.interface_validation import (
    quick_feature_selection,
    production_feature_selection,
    compare_with_original,
)

__all__ = [
    # Config
    "set_feature_flag",
    "get_feature_flags",
    "configure_analysis_pipeline",
    "create_feature_selection_config",
    "create_dual_validation_config",
    # Dashboard
    "generate_comprehensive_stability_dashboard",
    # Display
    "display_results_summary",
    "create_feature_selection_report",
    "display_dual_validation_results",
    # Environment
    "setup_feature_selection_environment",
    # Execution
    "run_feature_selection",
    "quick_feature_selection",
    "production_feature_selection",
    "compare_with_original",
    # Export
    "export_final_model_selection",
    "save_dual_validation_results",
    # Validation
    "validate_feature_selection_inputs",
]
