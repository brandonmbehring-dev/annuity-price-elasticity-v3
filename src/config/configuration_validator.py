"""
Configuration Validator for Feature Selection Systems.

This module provides validation and error checking to prevent common
configuration mistakes and TypeError issues.
"""

import inspect
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings


def validate_function_parameters(func: Callable[..., Any], provided_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that provided parameters match function signature.

    Parameters
    ----------
    func : callable
        The function to validate against
    provided_params : Dict[str, Any]
        Parameters being provided to the function

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, list_of_errors)
    """
    errors = []

    # Get function signature
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    provided_params_set = set(provided_params.keys())

    # Check for invalid parameters
    invalid_params = provided_params_set - valid_params
    if invalid_params:
        func_name = func.__name__
        errors.append(
            f"Function '{func_name}' does not accept parameters: {list(invalid_params)}"
        )
        errors.append(f"Valid parameters are: {list(valid_params)}")

    return len(errors) == 0, errors


def check_common_parameter_mistakes(provided_params: Dict[str, Any]) -> List[str]:
    """
    Check for common parameter naming mistakes.

    Parameters
    ----------
    provided_params : Dict[str, Any]
        Parameters being provided

    Returns
    -------
    List[str]
        List of warnings about parameter mistakes
    """
    warnings_list = []

    # Common parameter mistakes mapping
    common_mistakes = {
        'min_feature_combinations': 'This parameter does not exist in any configuration function',
        'bootstrap_enabled': 'Use "enable_block_bootstrap" for enhanced config',
        'enhanced_metrics_enabled': 'Enhanced metrics are enabled by default in enhanced config',
        'constraints_enabled': 'Use "economic_constraints" for basic config or "enable_statistical_constraints" for enhanced',
        'mlflow_enabled': 'MLflow tracking is handled separately'
    }

    for param_name, param_value in provided_params.items():
        if param_name in common_mistakes:
            warnings_list.append(f"Parameter '{param_name}': {common_mistakes[param_name]}")

    return warnings_list


def _map_enhanced_parameters(provided_params: Dict[str, Any]) -> Dict[str, Any]:
    """Map provided parameters to enhanced configuration parameters.

    Parameters
    ----------
    provided_params : Dict[str, Any]
        Parameters user provided

    Returns
    -------
    Dict[str, Any]
        Corrected parameters for enhanced configuration
    """
    param_mapping = {
        'max_candidate_features': 'max_candidate_features',
        'target_variable': 'target_variable',
        'bootstrap_enabled': 'enable_block_bootstrap',
        'enhanced_metrics_enabled': 'enable_multiple_testing_correction',
        'min_feature_combinations': None  # This doesn't exist
    }

    corrected = {}
    for param, value in provided_params.items():
        if param in param_mapping:
            mapped_param = param_mapping[param]
            if mapped_param:
                corrected[mapped_param] = value
        else:
            corrected[param] = value

    return corrected


def _map_basic_parameters(provided_params: Dict[str, Any]) -> Dict[str, Any]:
    """Map provided parameters to basic configuration parameters.

    Parameters
    ----------
    provided_params : Dict[str, Any]
        Parameters user provided

    Returns
    -------
    Dict[str, Any]
        Corrected parameters for basic configuration with defaults
    """
    basic_params = ['max_candidate_features', 'target_variable', 'base_features', 'economic_constraints']
    corrected = {k: v for k, v in provided_params.items() if k in basic_params}

    # Set defaults for missing basic parameters
    if 'target_variable' not in corrected:
        corrected['target_variable'] = '"sales_target_current"'
    if 'economic_constraints' not in corrected:
        corrected['economic_constraints'] = True

    return corrected


def suggest_correct_configuration(provided_params: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest the correct configuration based on provided parameters.

    Parameters
    ----------
    provided_params : Dict[str, Any]
        Parameters user is trying to use

    Returns
    -------
    Dict[str, Any]
        Suggested configuration approach
    """
    suggestion = {
        'recommended_system': 'basic',
        'reason': 'Simple parameters suggest basic configuration',
        'corrected_params': {},
        'import_statement': '',
        'function_call': ''
    }

    # Check if user wants enhanced features
    enhanced_indicators = [
        'bootstrap_enabled', 'enhanced_metrics_enabled', 'min_feature_combinations',
        'temporal_validation', 'multiple_testing', 'statistical_constraints'
    ]
    wants_enhanced = any(param in str(provided_params.keys()) for param in enhanced_indicators)

    if wants_enhanced:
        suggestion['recommended_system'] = 'enhanced'
        suggestion['reason'] = 'Parameters suggest you want enhanced statistical features'
        suggestion['import_statement'] = (
            'from src.features.selection.enhanced_configuration import build_enhanced_feature_selection_config'
        )
        corrected = _map_enhanced_parameters(provided_params)
        suggestion['corrected_params'] = corrected
        suggestion['function_call'] = (
            f"build_enhanced_feature_selection_config({', '.join(f'{k}={repr(v)}' for k, v in corrected.items())})"
        )
    else:
        suggestion['recommended_system'] = 'basic'
        suggestion['import_statement'] = 'from src.config.config_builder import build_feature_selection_stage_config'
        corrected = _map_basic_parameters(provided_params)
        suggestion['corrected_params'] = corrected
        suggestion['function_call'] = (
            f"build_feature_selection_stage_config({', '.join(f'{k}={repr(v)}' for k, v in corrected.items())})"
        )

    return suggestion


def generate_error_report(func_name: str, provided_params: Dict[str, Any], error: Exception) -> str:
    """Generate comprehensive error report with suggested fixes."""
    report = []
    report.append("=" * 70)
    report.append("CONFIGURATION ERROR ANALYSIS")
    report.append("=" * 70)
    report.append(f"Function called: {func_name}")
    report.append(f"Error: {error}")
    report.append(f"Parameters provided: {list(provided_params.keys())}")
    report.append("")

    # Check for common mistakes
    warnings_list = check_common_parameter_mistakes(provided_params)
    if warnings_list:
        report.append("COMMON PARAMETER ISSUES:")
        for warning in warnings_list:
            report.append(f"  ⚠ {warning}")
        report.append("")

    # Suggest correct configuration
    suggestion = suggest_correct_configuration(provided_params)
    report.append(f"RECOMMENDED SOLUTION:")
    report.append(f"  System: {suggestion['recommended_system']} configuration")
    report.append(f"  Reason: {suggestion['reason']}")
    report.append("")
    report.append(f"CORRECT CODE:")
    report.append(f"  {suggestion['import_statement']}")
    report.append("")
    report.append(f"  config = {suggestion['function_call']}")
    report.append("")

    # Add reference to documentation
    report.append("For more information, see:")
    report.append("  - CONFIGURATION_SYSTEMS_GUIDE.md")
    report.append("  - CONFIGURATION_FIX_EXAMPLES.py")
    report.append("=" * 70)

    return "\n".join(report)


def validate_configuration_call(func_name: str, **kwargs) -> None:
    """
    Validate a configuration function call before execution.

    Parameters
    ----------
    func_name : str
        Name of the function being called
    **kwargs
        Parameters being passed to the function

    Raises
    ------
    ValueError
        If configuration is invalid with detailed error message
    """
    # Import functions for validation (defensive imports for cross-context compatibility)
    try:
        from src.config.config_builder import build_feature_selection_stage_config
        from src.features.selection.enhanced_configuration import build_enhanced_feature_selection_config

        functions = {
            'build_feature_selection_stage_config': build_feature_selection_stage_config,
            'build_enhanced_feature_selection_config': build_enhanced_feature_selection_config
        }

        if func_name not in functions:
            raise ValueError(f"Unknown configuration function: {func_name}")

        func = functions[func_name]
        is_valid, errors = validate_function_parameters(func, kwargs)

        if not is_valid:
            # Create a mock TypeError to generate report
            error_msg = f"{func_name}() got an unexpected keyword argument"
            mock_error = TypeError(error_msg)

            report = generate_error_report(func_name, kwargs, mock_error)
            raise ValueError(f"Configuration validation failed:\n\n{report}")

        # Check for common mistakes
        warnings_list = check_common_parameter_mistakes(kwargs)
        if warnings_list:
            for warning in warnings_list:
                warnings.warn(f"Configuration warning: {warning}", UserWarning)

    except ImportError:
        warnings.warn("Configuration validation modules not available", UserWarning)


# Convenience functions for specific validations
def validate_basic_config(**kwargs) -> None:
    """Validate basic configuration parameters."""
    validate_configuration_call('build_feature_selection_stage_config', **kwargs)


def validate_enhanced_config(**kwargs) -> None:
    """Validate enhanced configuration parameters."""
    validate_configuration_call('build_enhanced_feature_selection_config', **kwargs)


def create_safe_config_wrapper(
    func: Callable[..., Any],
    func_name: str
) -> Callable[..., Any]:
    """
    Create a wrapper that validates parameters before calling config function.

    Parameters
    ----------
    func : Callable[..., Any]
        Configuration function to wrap
    func_name : str
        Name of the function for error reporting

    Returns
    -------
    Callable[..., Any]
        Wrapped function with validation
    """
    def wrapped_config_function(**kwargs: Any) -> Any:
        try:
            # Validate before calling
            validate_configuration_call(func_name, **kwargs)

            # Call original function
            return func(**kwargs)

        except Exception as e:
            # Generate helpful error report
            if "unexpected keyword argument" in str(e):
                report = generate_error_report(func_name, kwargs, e)
                raise TypeError(f"\n{report}") from e
            else:
                raise

    return wrapped_config_function


# Create safe wrappers for export
from src.config.config_builder import build_feature_selection_stage_config

safe_build_feature_selection_stage_config = create_safe_config_wrapper(
    build_feature_selection_stage_config,
    'build_feature_selection_stage_config'
)

# Enhanced configuration is optional (module may not exist)
try:
    from src.features.selection.enhanced_configuration import build_enhanced_feature_selection_config
    safe_build_enhanced_feature_selection_config = create_safe_config_wrapper(
        build_enhanced_feature_selection_config,
        'build_enhanced_feature_selection_config'
    )
except ImportError:
    # Enhanced configuration not implemented yet
    safe_build_enhanced_feature_selection_config = None