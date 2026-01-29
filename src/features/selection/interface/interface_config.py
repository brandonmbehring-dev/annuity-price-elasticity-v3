"""
Feature Selection Configuration Module.

This module handles configuration building and feature flags for
feature selection notebooks. Part of Phase 6.1 module split.

Responsibilities:
- Pipeline configuration building
- Dual validation configuration
- Feature selection config creation
- Feature flag management

Module Architecture (Phase 6.1 Split):
- interface_environment.py: Environment and import setup
- interface_config.py: Configuration building and feature flags (this file)
- interface_execution.py: Pipeline execution and error handling
- interface_validation.py: Validation, comparison, convenience functions
- interface_dashboard.py: Dashboard/stability/visualization functions
- interface_display.py: Display/formatting/HTML report functions
- interface_export.py: Export/MLflow/DVC functions
- notebook_interface.py: Public API orchestrator
"""

from typing import Any, Dict, List, Optional, Tuple

from src.features.selection_types import (
    BootstrapAnalysisConfig,
    EconomicConstraintConfig,
    FeatureSelectionConfig,
    build_bootstrap_config,
    build_constraint_config,
)


# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Feature flag system for gradual migration
FEATURE_FLAGS: Dict[str, bool] = {
    "USE_ATOMIC_FUNCTIONS": True,  # Enable atomic function pipeline
    "ENABLE_VALIDATION": True,  # Enable side-by-side validation
    "SHOW_DETAILED_OUTPUT": True,  # Show detailed analysis output
    "ENABLE_BOOTSTRAP_DEFAULT": False,  # Default bootstrap analysis
    "STRICT_CONSTRAINTS_DEFAULT": True,  # Default constraint validation
    "AUTO_DISPLAY_RESULTS": True,  # Auto-display formatted results
    # Enhancement module flags (Phase 1: Methodological Integration)
    "ENABLE_MULTIPLE_TESTING": False,  # Enable FWER/FDR corrections
    "ENABLE_BLOCK_BOOTSTRAP": False,  # Replace standard with block bootstrap
    "ENABLE_OOS_VALIDATION": False,  # Enable out-of-sample validation
    "ENABLE_REGRESSION_DIAGNOSTICS": False,  # Enable assumption checks
    "ENABLE_STATISTICAL_CONSTRAINTS": False,  # Enable CI-based validation
    "ENABLE_SEARCH_SPACE_REDUCTION": False,  # Pre-AIC search space reduction
}


def set_feature_flag(flag_name: str, value: bool) -> None:
    """
    Set feature flag for gradual migration control.

    Parameters
    ----------
    flag_name : str
        Name of the feature flag to set
    value : bool
        Flag value (True/False)
    """
    if flag_name in FEATURE_FLAGS:
        FEATURE_FLAGS[flag_name] = value
        print(f"SUCCESS: Feature flag '{flag_name}' set to {value}")
    else:
        available_flags = list(FEATURE_FLAGS.keys())
        print(
            f"WARNING: Unknown feature flag '{flag_name}'. Available flags: {available_flags}"
        )


def get_feature_flags() -> Dict[str, bool]:
    """Get current feature flag settings."""
    return FEATURE_FLAGS.copy()


# =============================================================================
# PIPELINE CONFIGURATION HELPERS
# =============================================================================


def _update_pipeline_bootstrap_config(
    configs: Dict[str, Any],
    n_bootstrap_samples: int,
    models_to_analyze: int,
    enable_bootstrap_analysis: bool,
) -> None:
    """Update bootstrap configuration in place."""
    configs["bootstrap_config"]["n_samples"] = n_bootstrap_samples
    configs["bootstrap_config"]["models_to_analyze"] = models_to_analyze
    configs["bootstrap_config"]["enabled"] = enable_bootstrap_analysis


def _update_pipeline_experiment_config(
    configs: Dict[str, Any],
    enable_economic_constraints: bool,
    enable_mlflow_tracking: bool,
    experiment_name: str,
) -> None:
    """Update constraint and experiment configuration in place."""
    configs["constraint_config"]["enabled"] = enable_economic_constraints
    configs["experiment_config"]["enabled"] = enable_mlflow_tracking
    configs["experiment_config"]["experiment_name"] = experiment_name


def _add_pipeline_visualization_config(configs: Dict[str, Any]) -> None:
    """Add visualization parameters to configs."""
    configs["visualization_config"] = {
        "fig_width": 10,
        "fig_height": 8,
        "style": "whitegrid",
        "palette": "deep",
    }


def _add_pipeline_status_indicators(configs: Dict[str, Any]) -> None:
    """Add status indicators to configs."""
    configs["status_indicators"] = {
        "SUCCESS": "SUCCESS:",
        "WARNING": "WARNING:",
        "ERROR": "ERROR:",
    }


def _create_pipeline_fallback_config(
    max_candidate_features: int,
    target_variable: str,
    analysis_start_date: str,
    enable_bootstrap_analysis: bool,
    enable_economic_constraints: bool,
    enable_mlflow_tracking: bool,
    experiment_name: str,
) -> Dict[str, Any]:
    """Create fallback configuration when builder is unavailable."""
    print("WARNING: Could not import configuration builder - using fallback config")
    return {
        "error": "Configuration builder not available",
        "fallback_config": {
            "max_candidate_features": max_candidate_features,
            "target_variable": target_variable,
            "analysis_start_date": analysis_start_date,
            "bootstrap_enabled": enable_bootstrap_analysis,
            "constraints_enabled": enable_economic_constraints,
            "mlflow_enabled": enable_mlflow_tracking,
            "experiment_name": experiment_name,
        },
    }


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================


def configure_analysis_pipeline(
    max_candidate_features: int = 5,
    target_variable: str = "sales_target_current",
    analysis_start_date: str = "2022-08-01",
    n_bootstrap_samples: int = 1000,
    models_to_analyze: int = 15,
    enable_economic_constraints: bool = True,
    enable_bootstrap_analysis: bool = True,
    enable_mlflow_tracking: bool = True,
    target_transformation: str = "none",
    experiment_name: str = "rila_feature_selection",
) -> Dict[str, Any]:
    """Configure analysis pipeline. Returns complete configuration dictionary."""
    try:
        from src.features.selection_types import build_feature_selection_configs

        configs = build_feature_selection_configs(
            max_candidate_features=max_candidate_features,
            target_variable=target_variable,
            analysis_start_date=analysis_start_date,
            target_transformation=target_transformation,
        )

        _update_pipeline_bootstrap_config(
            configs, n_bootstrap_samples, models_to_analyze, enable_bootstrap_analysis
        )
        _update_pipeline_experiment_config(
            configs, enable_economic_constraints, enable_mlflow_tracking, experiment_name
        )
        _add_pipeline_visualization_config(configs)
        _add_pipeline_status_indicators(configs)

        return configs

    except ImportError:
        return _create_pipeline_fallback_config(
            max_candidate_features,
            target_variable,
            analysis_start_date,
            enable_bootstrap_analysis,
            enable_economic_constraints,
            enable_mlflow_tracking,
            experiment_name,
        )


# =============================================================================
# DUAL VALIDATION CONFIGURATION
# =============================================================================


def _get_dual_validation_defaults() -> Dict[str, Any]:
    """Return default dual validation configuration settings."""
    return {
        "enabled": True,
        "core_metrics": ["AIC", "Adjusted_R2", "MAPE"],
        "validation_types": ["in_sample", "out_of_sample"],
        "win_rate_weight": 0.5,
        "information_ratio_weight": 0.5,
    }


def _update_configs_for_dual_validation(
    configs: Dict[str, Any], n_bootstrap_samples: int, models_to_analyze: int
) -> Dict[str, Any]:
    """Update configs with dual validation bootstrap and metric settings."""
    configs["bootstrap_config"]["n_samples"] = n_bootstrap_samples
    configs["bootstrap_config"]["models_to_analyze"] = models_to_analyze
    configs["bootstrap_config"]["enabled"] = True
    configs["dual_validation_config"] = _get_dual_validation_defaults()
    return configs


def create_dual_validation_config(
    max_candidate_features: int = 4,
    target_variable: str = "sales_target_current",
    analysis_start_date: str = "2022-08-01",
    n_bootstrap_samples: int = 1000,
    models_to_analyze: int = 15,
    target_transformation: str = "none",
) -> Dict[str, Any]:
    """
    Create configuration optimized for dual validation analysis.

    Returns
    -------
    Dict[str, Any]
        Complete configuration for dual validation
    """
    try:
        from src.features.selection_types import build_feature_selection_configs

        configs = build_feature_selection_configs(
            max_candidate_features=max_candidate_features,
            target_variable=target_variable,
            analysis_start_date=analysis_start_date,
            target_transformation=target_transformation,
        )
        return _update_configs_for_dual_validation(
            configs, n_bootstrap_samples, models_to_analyze
        )

    except ImportError:
        print("WARNING: Could not import configuration builder - using minimal config")
        return {"dual_validation_config": _get_dual_validation_defaults()}


# =============================================================================
# FEATURE SELECTION CONFIG BUILDERS
# =============================================================================


def _build_feature_config(
    candidate_features: List[str],
    target_variable: str,
    max_features: int,
    base_features: Optional[List[str]],
) -> FeatureSelectionConfig:
    """Build feature selection configuration object."""
    return FeatureSelectionConfig(
        base_features=base_features or [],
        candidate_features=candidate_features,
        max_candidate_features=max_features,
        target_variable=target_variable,
    )


def _build_constraint_config_wrapper(enable_constraints: bool) -> EconomicConstraintConfig:
    """Build economic constraint configuration object.

    Delegates to canonical builder in selection_types.py.
    """
    return build_constraint_config(
        enabled=enable_constraints,
        strict_validation=FEATURE_FLAGS["STRICT_CONSTRAINTS_DEFAULT"],
    )


def _build_bootstrap_config_wrapper(
    enable_bootstrap: Optional[bool],
    bootstrap_samples: int,
    candidate_features: List[str],
    random_seed: Optional[int],
) -> Optional[BootstrapAnalysisConfig]:
    """Build bootstrap configuration if enabled.

    Delegates to canonical builder in selection_types.py.
    """
    bootstrap_enabled = (
        enable_bootstrap
        if enable_bootstrap is not None
        else FEATURE_FLAGS["ENABLE_BOOTSTRAP_DEFAULT"]
    )
    if not bootstrap_enabled:
        return None

    return build_bootstrap_config(
        enabled=True,
        n_samples=bootstrap_samples,
        models_to_analyze=min(5, len(candidate_features)),
        confidence_intervals=[90, 95, 99],
        random_seed=random_seed,
    )


def create_feature_selection_config(
    candidate_features: List[str],
    target_variable: str,
    max_features: int = 3,
    base_features: Optional[List[str]] = None,
    enable_constraints: bool = True,
    enable_bootstrap: Optional[bool] = None,
    bootstrap_samples: int = 100,
    random_seed: Optional[int] = 42,
) -> Tuple[FeatureSelectionConfig, EconomicConstraintConfig, Optional[BootstrapAnalysisConfig]]:
    """
    Create feature selection configuration with business-friendly parameters.

    Thin orchestrator: delegates to focused helper functions.

    Parameters
    ----------
    candidate_features : List[str]
        Features to evaluate in combinations
    target_variable : str
        Target variable for regression analysis
    max_features : int, default=3
        Maximum number of candidate features per model
    base_features : Optional[List[str]], default=None
        Features required in all models
    enable_constraints : bool, default=True
        Enable economic constraint validation
    enable_bootstrap : Optional[bool], default=None
        Enable bootstrap stability analysis
    bootstrap_samples : int, default=100
        Number of bootstrap samples
    random_seed : Optional[int], default=42
        Random seed for reproducibility

    Returns
    -------
    Tuple[FeatureSelectionConfig, EconomicConstraintConfig, Optional[BootstrapAnalysisConfig]]
        Complete configuration tuple
    """
    feature_config = _build_feature_config(
        candidate_features, target_variable, max_features, base_features
    )
    constraint_config = _build_constraint_config_wrapper(enable_constraints)
    bootstrap_config = _build_bootstrap_config_wrapper(
        enable_bootstrap, bootstrap_samples, candidate_features, random_seed
    )

    return feature_config, constraint_config, bootstrap_config


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Feature flags
    "FEATURE_FLAGS",
    "set_feature_flag",
    "get_feature_flags",
    # Pipeline configuration helpers
    "_update_pipeline_bootstrap_config",
    "_update_pipeline_experiment_config",
    "_add_pipeline_visualization_config",
    "_add_pipeline_status_indicators",
    "_create_pipeline_fallback_config",
    # Pipeline configuration
    "configure_analysis_pipeline",
    # Dual validation configuration
    "_get_dual_validation_defaults",
    "_update_configs_for_dual_validation",
    "create_dual_validation_config",
    # Feature selection config builders
    "_build_feature_config",
    "_build_constraint_config_wrapper",
    "_build_bootstrap_config_wrapper",
    "create_feature_selection_config",
]
