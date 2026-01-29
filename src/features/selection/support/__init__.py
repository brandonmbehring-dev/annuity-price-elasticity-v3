"""
Feature Selection Support Subpackage.

Provides utility modules for feature selection:
- data_preprocessing: Data preparation and validation
- configuration_management: Configuration utilities
- visualization_config: Visualization settings
- notebook_helpers: Notebook utility functions
- results_export: Results persistence
- environment_setup: Environment configuration
- regression_diagnostics: Model diagnostics
"""

# Data preprocessing
from src.features.selection.support.data_preprocessing import (
    prepare_analysis_dataset,
    validate_feature_availability,
)

# Configuration management
from src.features.selection.support.configuration_management import (
    build_comprehensive_feature_config,
    validate_configuration_integrity,
    manage_parameter_inheritance,
    create_experiment_configurations,
    handle_configuration_updates,
)

# Notebook helpers
from src.features.selection.support.notebook_helpers import (
    create_feature_flags,
    get_feature_flags,
    display_results_summary,
    format_feature_selection_results,
    create_progress_tracker,
    generate_diagnostic_information,
    provide_notebook_utilities,
)

# Results export
from src.features.selection.support.results_export import (
    export_feature_selection_results,
    manage_dvc_checkpoints,
    finalize_mlflow_experiment,
    generate_analysis_summary,
)

# Environment setup
from src.features.selection.support.environment_setup import (
    setup_feature_selection_environment,
)

# Regression diagnostics
from src.features.selection.support.regression_diagnostics import (
    comprehensive_diagnostic_suite,
    check_autocorrelation,
    check_heteroscedasticity,
    check_multicollinearity,
    check_normality,
)

__all__ = [
    # Data preprocessing
    "prepare_analysis_dataset",
    "validate_feature_availability",
    # Configuration
    "build_comprehensive_feature_config",
    "validate_configuration_integrity",
    "manage_parameter_inheritance",
    "create_experiment_configurations",
    "handle_configuration_updates",
    # Notebook helpers
    "create_feature_flags",
    "get_feature_flags",
    "display_results_summary",
    "format_feature_selection_results",
    "create_progress_tracker",
    "generate_diagnostic_information",
    "provide_notebook_utilities",
    # Results
    "export_feature_selection_results",
    "manage_dvc_checkpoints",
    "finalize_mlflow_experiment",
    "generate_analysis_summary",
    # Environment
    "setup_feature_selection_environment",
    # Diagnostics
    "comprehensive_diagnostic_suite",
    "check_autocorrelation",
    "check_heteroscedasticity",
    "check_multicollinearity",
    "check_normality",
]
