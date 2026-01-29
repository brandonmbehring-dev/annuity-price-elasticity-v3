"""
Configuration Management Module for Feature Selection Pipeline.

This module provides atomic functions for configuration management, parameter validation,
and dynamic configuration updates following CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Decompose notebook_interface.py configuration functions (200-300 lines)
Status: MANDATORY (decomposition of 2,274-line module)
Priority: HIGH (code organization and maintainability)

Key Functions:
- build_comprehensive_feature_config(): Configuration builder with TypedDict validation
- validate_configuration_integrity(): Configuration validation with business rules
- manage_parameter_inheritance(): Parameter inheritance logic
- create_experiment_configurations(): Experiment config creation
- handle_configuration_updates(): Dynamic configuration updates

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

# Canonical feature configuration (Single Source of Truth)
from src.config.product_config import get_default_feature_config, ProductFeatureConfig

# Import canonical config builders from selection_types
from src.features.selection_types import (
    EconomicConstraintConfig,
    BootstrapAnalysisConfig,
    build_constraint_config,
    build_bootstrap_config,
)


def _validate_feature_count(max_candidate_features: int) -> None:
    """Validate max_candidate_features parameter.

    Parameters
    ----------
    max_candidate_features : int
        Maximum number of candidate features to select

    Raises
    ------
    ValueError
        If max_candidate_features is outside valid range (1-10)
    """
    if max_candidate_features <= 0 or max_candidate_features > 10:
        raise ValueError(
            f"CRITICAL: max_candidate_features must be 1-10, got {max_candidate_features}. "
            f"Business impact: Model complexity constraint violated. "
            f"Required action: Use reasonable feature count for model parsimony."
        )


def _validate_target_variable(target_variable: str) -> None:
    """Validate target_variable parameter.

    Delegates to canonical validator in src.validation.input_validators.

    Parameters
    ----------
    target_variable : str
        Target variable name for regression analysis

    Raises
    ------
    ValueError
        If target_variable is empty or not a string
    """
    from src.validation.input_validators import validate_target_variable_string
    validate_target_variable_string(target_variable)


def _get_default_candidate_features() -> List[str]:
    """Return default candidate features (business-validated list).

    Delegates to ProductFeatureConfig (Single Source of Truth in product_config.py).

    Returns
    -------
    List[str]
        Default list of candidate features for feature selection
    """
    config = get_default_feature_config()
    return list(config.candidate_features)


def build_comprehensive_feature_config(max_candidate_features: int = 4,
                                     target_variable: str = "sales_target_current",
                                     analysis_start_date: str = "2022-08-01",
                                     candidate_features: Optional[List[str]] = None,
                                     base_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build comprehensive feature selection configuration with TypedDict validation.

    Orchestrates configuration building by delegating to helper functions.

    Parameters
    ----------
    max_candidate_features : int, default 4
        Maximum number of candidate features to select
    target_variable : str, default "sales_target_current"
        Target variable name for regression analysis
    analysis_start_date : str, default "2022-08-01"
        Analysis start date in YYYY-MM-DD format
    candidate_features : List[str], optional
        List of candidate features (uses default if None)
    base_features : List[str], optional
        List of base features (uses empty list if None)

    Returns
    -------
    Dict[str, Any]
        Comprehensive configuration dictionary with validated parameters

    Raises
    ------
    ValueError
        If configuration parameters are invalid
    """
    _validate_feature_count(max_candidate_features)
    _validate_target_variable(target_variable)

    return {
        'max_candidate_features': max_candidate_features,
        'target_variable': target_variable,
        'analysis_start_date': analysis_start_date,
        'candidate_features': candidate_features if candidate_features is not None else _get_default_candidate_features(),
        'base_features': base_features if base_features is not None else [],
        'target_transformation': 'np.log(1 + target)',  # Standard log transformation
        'holiday_exclusion': True,  # Exclude holiday observations
        'autoregressive_transform': True  # Apply log transform to AR features
    }


def _get_required_config_keys() -> Dict[str, str]:
    """Return required configuration keys with business rationale.

    Returns
    -------
    Dict[str, str]
        Mapping of required key names to their business purpose
    """
    return {
        'max_candidate_features': 'Model complexity constraint',
        'target_variable': 'Dependent variable specification',
        'candidate_features': 'Feature selection scope',
        'base_features': 'Required features specification'
    }


def _validate_required_keys(config: Dict[str, Any], errors: List[str]) -> None:
    """Validate that all required keys are present in configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    errors : List[str]
        List to append validation errors to (mutated in place)
    """
    for key, purpose in _get_required_config_keys().items():
        if key not in config:
            errors.append(f"Missing required key '{key}' ({purpose})")


def _validate_config_data_types(config: Dict[str, Any], errors: List[str]) -> None:
    """Validate data types of configuration values.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate
    errors : List[str]
        List to append validation errors to (mutated in place)
    """
    if 'max_candidate_features' in config:
        if not isinstance(config['max_candidate_features'], int) or config['max_candidate_features'] <= 0:
            errors.append("max_candidate_features must be positive integer")

    if 'candidate_features' in config:
        if not isinstance(config['candidate_features'], list):
            errors.append("candidate_features must be list")
        elif len(config['candidate_features']) < config.get('max_candidate_features', 1):
            errors.append("Not enough candidate features for selection constraint")

    if 'target_variable' in config:
        if not isinstance(config['target_variable'], str) or not config['target_variable'].strip():
            errors.append("target_variable must be non-empty string")

    if len(config.get('candidate_features', [])) == 0:
        errors.append("At least one candidate feature required for selection")


def validate_configuration_integrity(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration integrity with business rules.

    Orchestrates validation by delegating to helper functions.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (validation_passed, validation_errors)

    Raises
    ------
    ValueError
        If configuration structure is fundamentally invalid
    """
    if not isinstance(config, dict):
        raise ValueError(
            "CRITICAL: Configuration must be dictionary. "
            "Business impact: Cannot validate configuration structure. "
            "Required action: Provide valid configuration dictionary."
        )

    validation_errors: List[str] = []
    _validate_required_keys(config, validation_errors)
    _validate_config_data_types(config, validation_errors)

    return len(validation_errors) == 0, validation_errors


def _validate_inheritance_inputs(base_config: Dict[str, Any], user_overrides: Dict[str, Any]) -> None:
    """Validate inputs for parameter inheritance.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration with default values
    user_overrides : Dict[str, Any]
        User-provided overrides for specific parameters

    Raises
    ------
    ValueError
        If inputs are not dictionaries
    """
    if not isinstance(base_config, dict):
        raise ValueError(
            "CRITICAL: Base configuration must be dictionary. "
            "Business impact: Cannot apply parameter inheritance. "
            "Required action: Provide valid base configuration."
        )

    if not isinstance(user_overrides, dict):
        raise ValueError(
            "CRITICAL: User overrides must be dictionary. "
            "Business impact: Cannot apply user customizations. "
            "Required action: Provide valid overrides dictionary."
        )


def _apply_overrides(merged_config: Dict[str, Any], user_overrides: Dict[str, Any]) -> None:
    """Apply user overrides to merged configuration (mutates in place).

    Parameters
    ----------
    merged_config : Dict[str, Any]
        Configuration to apply overrides to (mutated)
    user_overrides : Dict[str, Any]
        User-provided overrides
    """
    for key, value in user_overrides.items():
        if key in merged_config:
            original_type = type(merged_config[key])
            override_type = type(value)

            if original_type != override_type and value is not None:
                warnings.warn(
                    f"Type mismatch for '{key}': expected {original_type.__name__}, "
                    f"got {override_type.__name__}. Applying override anyway."
                )
        else:
            warnings.warn(f"Adding new parameter '{key}' not in base configuration")

        merged_config[key] = value


def manage_parameter_inheritance(base_config: Dict[str, Any],
                               user_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Manage parameter inheritance with user override support.

    Orchestrates parameter inheritance by delegating to helper functions.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration with default values
    user_overrides : Dict[str, Any]
        User-provided overrides for specific parameters

    Returns
    -------
    Dict[str, Any]
        Merged configuration with user overrides applied

    Raises
    ------
    ValueError
        If parameter inheritance creates invalid configuration
    """
    _validate_inheritance_inputs(base_config, user_overrides)

    merged_config = base_config.copy()
    _apply_overrides(merged_config, user_overrides)

    is_valid, errors = validate_configuration_integrity(merged_config)
    if not is_valid:
        raise ValueError(
            f"CRITICAL: Parameter inheritance created invalid configuration: {errors}. "
            f"Business impact: Configuration not suitable for feature selection. "
            f"Required action: Fix parameter conflicts and retry."
        )

    return merged_config


def _build_constraint_config() -> EconomicConstraintConfig:
    """Build economic constraint configuration.

    Delegates to canonical builder in selection_types.py.

    Returns
    -------
    EconomicConstraintConfig
        Economic constraint configuration with business rules
    """
    return build_constraint_config(
        enabled=True,
        strict_validation=True,
        constraint_rules={
            'competitor_negative': True,  # Competitor rates should be negative
            'prudential_positive': True,  # Prudential rates should be positive
            'autoregressive_positive': True  # Autoregressive features should be positive
        },
        violation_tolerance=0.0,  # Zero tolerance for constraint violations
        business_validation=True
    )


def _build_bootstrap_config() -> BootstrapAnalysisConfig:
    """Build bootstrap analysis configuration.

    Delegates to canonical builder in selection_types.py.

    Returns
    -------
    BootstrapAnalysisConfig
        Bootstrap analysis configuration with stability thresholds
    """
    return build_bootstrap_config(
        enabled=True,
        n_samples=100,  # Standard bootstrap sample size
        models_to_analyze=15,  # Top models for stability analysis
        confidence_intervals=[50, 70, 90],  # Multiple confidence levels
        random_seed=42,  # Reproducible bootstrap sampling
    )


def _build_mlflow_experiment_config() -> Dict[str, Any]:
    """Build MLflow experiment configuration.

    Returns
    -------
    Dict[str, Any]
        MLflow experiment tracking configuration
    """
    return {
        'enabled': True,
        'experiment_name': 'rila_feature_selection',
        'auto_tagging': True,
        'track_parameters': True,
        'track_metrics': True,
        'track_artifacts': True
    }


def create_experiment_configurations(feature_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create experiment configurations for all analysis components.

    Orchestrates experiment configuration creation by delegating to helpers.

    Parameters
    ----------
    feature_config : Dict[str, Any]
        Feature selection configuration

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing all experiment configurations:
        - constraint_config: Economic constraint configuration
        - bootstrap_config: Bootstrap analysis configuration
        - experiment_config: MLflow experiment configuration

    Raises
    ------
    ValueError
        If feature_config is invalid for experiment creation
    """
    if not feature_config:
        raise ValueError(
            "CRITICAL: Feature configuration required for experiment setup. "
            "Business impact: Cannot create analysis configurations. "
            "Required action: Provide valid feature configuration."
        )

    return {
        'constraint_config': _build_constraint_config(),
        'bootstrap_config': _build_bootstrap_config(),
        'experiment_config': _build_mlflow_experiment_config()
    }


def _validate_update_inputs(current_config: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Validate inputs for configuration updates.

    Parameters
    ----------
    current_config : Dict[str, Any]
        Current configuration to update
    updates : Dict[str, Any]
        Configuration updates to apply

    Raises
    ------
    ValueError
        If inputs are not dictionaries
    """
    if not isinstance(current_config, dict):
        raise ValueError(
            "CRITICAL: Current configuration must be dictionary. "
            "Business impact: Cannot apply configuration updates. "
            "Required action: Provide valid current configuration."
        )

    if not isinstance(updates, dict):
        raise ValueError(
            "CRITICAL: Updates must be dictionary. "
            "Business impact: Cannot process configuration changes. "
            "Required action: Provide valid updates dictionary."
        )


def _apply_updates_with_logging(updated_config: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Apply updates to configuration with logging (mutates in place).

    Parameters
    ----------
    updated_config : Dict[str, Any]
        Configuration to update (mutated)
    updates : Dict[str, Any]
        Updates to apply
    """
    for key, value in updates.items():
        old_value = updated_config.get(key, "NOT_SET")
        updated_config[key] = value
        print(f"Configuration update: {key}: {old_value} -> {value}")


def handle_configuration_updates(current_config: Dict[str, Any],
                                updates: Dict[str, Any]) -> Dict[str, Any]:
    """Handle dynamic configuration updates with validation.

    Orchestrates configuration updates by delegating to helper functions.

    Parameters
    ----------
    current_config : Dict[str, Any]
        Current configuration to update
    updates : Dict[str, Any]
        Configuration updates to apply

    Returns
    -------
    Dict[str, Any]
        Updated configuration with changes applied

    Raises
    ------
    ValueError
        If configuration updates create invalid configuration
    """
    _validate_update_inputs(current_config, updates)

    updated_config = current_config.copy()
    _apply_updates_with_logging(updated_config, updates)

    is_valid, errors = validate_configuration_integrity(updated_config)
    if not is_valid:
        raise ValueError(
            f"CRITICAL: Configuration updates created invalid state: {errors}. "
            f"Business impact: Updated configuration not suitable for analysis. "
            f"Required action: Fix configuration conflicts and retry updates."
        )

    return updated_config