"""
Notebook Helpers Module for Feature Selection Pipeline.

This module provides atomic functions for notebook display, formatting utilities,
convenience wrappers, and progress tracking following CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Decompose notebook_interface.py helper functions (~800 lines)
Status: MANDATORY (decomposition of 2,274-line module)
Priority: HIGH (code organization and maintainability)

Key Functions:
- display_results_summary(): Notebook-friendly results display
- format_feature_selection_results(): Professional result formatting
- create_progress_tracker(): Progress tracking utilities
- generate_diagnostic_information(): Diagnostic reporting
- handle_feature_flags(): Feature flag management
- create_configuration_helpers(): Configuration assistance
- provide_notebook_utilities(): Notebook convenience functions

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import warnings
from IPython.display import display, HTML, Markdown
import json


# =============================================================================
# DEPENDENCY INJECTION PATTERN - ARCHITECTURAL DECISION
# =============================================================================
# PATTERN: Dual-mode global for notebook convenience + factory for testability
# NOTEBOOK USAGE: Import and use FEATURE_FLAGS directly (simple access)
# TEST USAGE: Factory functions for isolated testing (no shared state)
# INTERNAL MODULES: Should avoid globals, use explicit parameters
#
# RATIONALE: Global flags provide ergonomic API for notebooks while factory
# functions enable comprehensive testing. This is an intentional architectural
# choice, not technical debt. Notebooks are a presentation layer where
# convenience matters.
#
# USAGE GUIDANCE:
# - Notebooks: Import FEATURE_FLAGS directly or use handle_feature_flags()
# - Tests: Use create_feature_flags() for test isolation
# - Internal src/: Pass feature flags explicitly when needed

# Feature flag system for notebook convenience
FEATURE_FLAGS = {
    'USE_ATOMIC_FUNCTIONS': True,        # Enable atomic function pipeline
    'ENABLE_VALIDATION': True,           # Enable side-by-side validation
    'SHOW_DETAILED_OUTPUT': True,       # Show detailed analysis output
    'ENABLE_BOOTSTRAP_DEFAULT': False,   # Default bootstrap analysis
    'STRICT_CONSTRAINTS_DEFAULT': True,  # Default constraint validation
    'AUTO_DISPLAY_RESULTS': True        # Auto-display formatted results
}

def create_feature_flags() -> Dict[str, bool]:
    """Create a new feature flags dictionary (DI pattern).

    This is the preferred pattern for dependency injection.
    Use this when you need isolated feature flags (e.g., in tests).

    Returns:
        New feature flags dictionary with defaults

    Example:
        flags = create_feature_flags()
        flags['USE_ATOMIC_FUNCTIONS'] = False
        display_results_summary(results, feature_flags=flags)
    """
    return {
        'USE_ATOMIC_FUNCTIONS': True,
        'ENABLE_VALIDATION': True,
        'SHOW_DETAILED_OUTPUT': True,
        'ENABLE_BOOTSTRAP_DEFAULT': False,
        'STRICT_CONSTRAINTS_DEFAULT': True,
        'AUTO_DISPLAY_RESULTS': True
    }

def get_feature_flags() -> Dict[str, bool]:
    """Get global feature flags (backward compatibility).

    MIGRATION NOTE: This function returns the global singleton for backward compatibility.
    New code should create flags explicitly:
        flags = create_feature_flags()

    Returns:
        Global feature flags dictionary (singleton for backward compatibility)
    """
    return FEATURE_FLAGS


def display_results_summary(results: Any,
                          display_format: str = "notebook",
                          show_coefficients: bool = True) -> None:
    """Display notebook-friendly results summary with professional formatting.

    Renders feature selection results in a formatted tabular display suitable
    for Jupyter notebooks. Handles both Series (single model) and DataFrame
    (multiple models) result formats.

    Parameters
    ----------
    results : Any
        Feature selection results, typically pd.Series (single model) or
        pd.DataFrame (multiple ranked models)
    display_format : str, default="notebook"
        Display format specification (currently supports "notebook")
    show_coefficients : bool, default=True
        Whether to display model coefficients in the output

    Returns
    -------
    None
        Prints formatted results directly to notebook output
    """
    if results is None:
        print("No results available for display")
        return

    try:
        print("=== FEATURE SELECTION RESULTS SUMMARY ===")

        if isinstance(results, pd.Series):
            # Single model display
            print(f"Selected Model:")
            print(f"  Features: {results.get('features', 'Not available')}")
            print(f"  AIC Score: {results.get('aic', 'Not available'):.3f}")
            print(f"  R-squared: {results.get('r_squared', 'Not available'):.4f}")
            print(f"  Feature Count: {results.get('n_features', 'Not available')}")

            if show_coefficients and 'coefficients' in results:
                print(f"\nModel Coefficients:")
                coefficients = results['coefficients']
                for feature, coef in coefficients.items():
                    if feature == 'Intercept':
                        print(f"  {feature}: {coef:.4f}")
                    else:
                        print(f"  {feature}: {coef:.4f}")

        elif isinstance(results, pd.DataFrame):
            # Multiple models display
            print(f"Model Rankings (Top 10):")
            print(f"{'Rank':<5} {'Features':<40} {'AIC':<8} {'R²':<8}")
            print("-" * 65)

            for idx, (_, row) in enumerate(results.head(10).iterrows(), 1):
                features_short = str(row.get('features', ''))[:37] + "..." if len(str(row.get('features', ''))) > 37 else str(row.get('features', ''))
                aic_val = row.get('aic', 0)
                r2_val = row.get('r_squared', 0)
                print(f"{idx:<5} {features_short:<40} {aic_val:<8.1f} {r2_val:<8.4f}")

        else:
            print(f"Results type: {type(results)}")
            print(f"Results summary: {str(results)[:200]}...")

    except Exception as e:
        print(f"Results display failed: {e}")
        print(f"Results type: {type(results)}")


def _format_executive_summary(features: Any, aic_score: float,
                             r_squared: float, n_features: int) -> str:
    """Format executive summary section of results.

    Private helper for format_feature_selection_results.
    Single responsibility: Executive summary formatting.

    Parameters
    ----------
    features : Any
        Selected model features
    aic_score : float
        Model AIC score
    r_squared : float
        Model R-squared value
    n_features : int
        Number of features in model

    Returns
    -------
    str
        Formatted executive summary
    """
    return f"""
FINAL FEATURE SELECTION RESULTS

Selected Model:
  Features: {features}
  AIC Score: {aic_score:.3f} (lower is better)
  R-squared: {r_squared:.4f} (explained variance)
  Feature Count: {n_features} (parsimony achieved)
"""


def _format_coefficients_summary(coefficients: Dict[str, float]) -> str:
    """Format coefficients with business interpretation.

    Private helper for format_feature_selection_results.
    Single responsibility: Coefficient formatting with economic validation.

    Parameters
    ----------
    coefficients : Dict[str, float]
        Model coefficients dictionary

    Returns
    -------
    str
        Formatted coefficients summary
    """
    summary = "\nModel Coefficients & Economic Validation:\n"

    for feature, coef in coefficients.items():
        if feature == 'Intercept':
            summary += f"  {feature}: {coef:.4f} (baseline level)\n"
        elif 'competitor_' in feature:
            status = "[PASS]" if coef < 0 else "[WARN]"
            summary += f"  {feature}: {coef:.4f} {status} (competitive advantage)\n"
        elif 'prudential_rate' in feature:
            status = "[PASS]" if coef > 0 else "[WARN]"
            summary += f"  {feature}: {coef:.4f} {status} (pricing power)\n"
        else:
            summary += f"  {feature}: {coef:.4f}\n"

    return summary


def _format_metadata_summary(analysis_metadata: Dict[str, Any]) -> str:
    """Format analysis metadata section.

    Private helper for format_feature_selection_results.
    Single responsibility: Metadata formatting.

    Parameters
    ----------
    analysis_metadata : Dict[str, Any]
        Analysis metadata dictionary

    Returns
    -------
    str
        Formatted metadata summary
    """
    return f"""
Analysis Metadata:
  Selection Method: {analysis_metadata.get('selection_method', 'Not specified')}
  Models Evaluated: {analysis_metadata.get('total_models', 'Unknown')}
  Economic Constraints: {'Applied' if analysis_metadata.get('constraints_applied', False) else 'Not Applied'}
  Bootstrap Analysis: {'Completed' if analysis_metadata.get('bootstrap_completed', False) else 'Not Performed'}
"""


def format_feature_selection_results(final_model: pd.Series,
                                   analysis_metadata: Dict[str, Any]) -> Dict[str, str]:
    """Format feature selection results with professional presentation.

    Generates comprehensive formatted result sections including executive summary,
    coefficient analysis with economic validation, and analysis metadata. Combines
    all sections into a complete professional report.

    Parameters
    ----------
    final_model : pd.Series
        Selected model Series containing features, AIC, R-squared, n_features,
        and coefficients dictionary
    analysis_metadata : Dict[str, Any]
        Metadata about the analysis including selection_method, total_models,
        constraints_applied, and bootstrap_completed flags

    Returns
    -------
    Dict[str, str]
        Dictionary with keys: executive_summary, coefficients_summary,
        metadata_summary, and full_report (all formatted as strings)

    Raises
    ------
    ValueError
        If final_model is None or empty
    ValueError
        If result formatting fails due to invalid structure or metadata format
    """
    if final_model is None or final_model.empty:
        raise ValueError(
            "CRITICAL: No model provided for formatting. "
            "Business impact: Cannot present feature selection results. "
            "Required action: Ensure feature selection produces valid model."
        )

    try:
        # Extract model information
        features = final_model.get('features', 'Not available')
        aic_score = final_model.get('aic', 0)
        r_squared = final_model.get('r_squared', 0)
        n_features = final_model.get('n_features', 0)
        coefficients = final_model.get('coefficients', {})

        # Generate formatted sections using helper functions
        executive_summary = _format_executive_summary(features, aic_score, r_squared, n_features)
        coefficients_summary = _format_coefficients_summary(coefficients)
        metadata_summary = _format_metadata_summary(analysis_metadata)

        return {
            'executive_summary': executive_summary.strip(),
            'coefficients_summary': coefficients_summary.strip(),
            'metadata_summary': metadata_summary.strip(),
            'full_report': (executive_summary + coefficients_summary + metadata_summary).strip()
        }

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Result formatting failed: {e}. "
            f"Business impact: Cannot present professional analysis results. "
            f"Required action: Check model structure and metadata format."
        ) from e


def create_progress_tracker(total_stages: int, stage_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create progress tracking utilities for pipeline execution.

    Initializes a progress tracker dictionary for monitoring multi-stage pipeline
    execution with timing and status information for each stage.

    Parameters
    ----------
    total_stages : int
        Total number of pipeline stages to track (must be > 0)
    stage_names : List[str], optional
        Human-readable names for each stage. If None, generates default names
        (Stage 1, Stage 2, etc.). Must match total_stages if provided.

    Returns
    -------
    Dict[str, Any]
        Progress tracker dictionary with keys:
        - total_stages: int, number of stages
        - stage_names: list of stage names
        - current_stage: int, current stage index (0-based)
        - completed_stages: list of completed stage names
        - stage_status: dict mapping stage names to status ('pending'/'completed')
        - start_time: timestamp or None
        - stage_times: dict mapping stage names to execution times

    Raises
    ------
    ValueError
        If total_stages <= 0
    ValueError
        If progress tracker creation fails

    Warnings
    --------
    UserWarning
        If stage_names count doesn't match total_stages
    """
    if total_stages <= 0:
        raise ValueError(
            "CRITICAL: Total stages must be positive. "
            "Business impact: Cannot track pipeline progress. "
            "Required action: Specify valid number of pipeline stages."
        )

    try:
        # Default stage names if not provided
        if stage_names is None:
            stage_names = [f"Stage {i+1}" for i in range(total_stages)]

        if len(stage_names) != total_stages:
            warnings.warn(f"Stage name count ({len(stage_names)}) doesn't match total stages ({total_stages})")

        # Initialize progress tracker
        progress_tracker = {
            'total_stages': total_stages,
            'stage_names': stage_names,
            'current_stage': 0,
            'completed_stages': [],
            'stage_status': {name: 'pending' for name in stage_names},
            'start_time': None,
            'stage_times': {}
        }

        return progress_tracker

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Progress tracker creation failed: {e}. "
            f"Business impact: Cannot monitor pipeline execution progress. "
            f"Required action: Check stage configuration parameters."
        ) from e


def _collect_pipeline_diagnostics(pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Collect pipeline execution diagnostics.

    Private helper for generate_diagnostic_information.
    Single responsibility: Pipeline state diagnostics collection.

    Parameters
    ----------
    pipeline_state : Dict[str, Any]
        Current pipeline state

    Returns
    -------
    Dict[str, Any]
        Pipeline diagnostics information
    """
    return {
        'stages_completed': pipeline_state.get('pipeline_progress', []),
        'last_completed': pipeline_state.get('last_completed_stage', 'None'),
        'current_state_keys': list(pipeline_state.keys()),
        'pipeline_status': 'active' if pipeline_state else 'empty'
    }


def _analyze_data_diagnostics(pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data quality and availability diagnostics.

    Private helper for generate_diagnostic_information.
    Single responsibility: Data diagnostics analysis.

    Parameters
    ----------
    pipeline_state : Dict[str, Any]
        Current pipeline state

    Returns
    -------
    Dict[str, Any]
        Data diagnostics information
    """
    data_diagnostics = {
        'datasets_available': [],
        'dataset_shapes': {},
        'missing_data_flags': []
    }

    for key, value in pipeline_state.items():
        if isinstance(value, pd.DataFrame):
            data_diagnostics['datasets_available'].append(key)
            data_diagnostics['dataset_shapes'][key] = value.shape

            # Check for common data issues
            if value.empty:
                data_diagnostics['missing_data_flags'].append(f"{key}: empty dataset")
            elif value.isnull().any().any():
                data_diagnostics['missing_data_flags'].append(f"{key}: contains null values")

    return data_diagnostics


def _generate_diagnostic_recommendations(pipeline_diagnostics: Dict[str, Any],
                                        data_diagnostics: Dict[str, Any]) -> List[str]:
    """Generate actionable diagnostic recommendations.

    Private helper for generate_diagnostic_information.
    Single responsibility: Recommendation generation.

    Parameters
    ----------
    pipeline_diagnostics : Dict[str, Any]
        Pipeline diagnostics information
    data_diagnostics : Dict[str, Any]
        Data diagnostics information

    Returns
    -------
    List[str]
        List of actionable recommendations
    """
    recommendations = []

    if not pipeline_diagnostics['stages_completed']:
        recommendations.append("Pipeline not started - check initial configuration")
    if data_diagnostics['missing_data_flags']:
        recommendations.append("Data quality issues detected - review data preparation")
    if not data_diagnostics['datasets_available']:
        recommendations.append("No datasets found - ensure data loading completed")

    return recommendations


def generate_diagnostic_information(pipeline_state: Dict[str, Any],
                                  error_context: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive diagnostic information for troubleshooting.

    Collects pipeline state, data quality, configuration, and provides actionable
    diagnostic recommendations. Handles errors gracefully by returning diagnostic
    error information.

    Parameters
    ----------
    pipeline_state : Dict[str, Any]
        Current pipeline state containing execution history, datasets, and
        configuration information
    error_context : str, optional
        Description of error context if diagnostics are being generated in
        response to an error. If None, uses default message.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - pipeline_diagnostics: stages_completed, last_completed, current_state_keys
        - data_diagnostics: datasets_available, dataset_shapes, missing_data_flags
        - configuration_diagnostics: feature_flags, config_keys
        - error_context: error context string
        - recommendations: list of actionable recommendations

        On failure, returns dict with diagnostic_error, pipeline_state_type,
        and recommendations keys

    Notes
    -----
    If diagnostic generation fails, returns graceful error dict with
    recommendations for manual review
    """
    try:
        # Collect diagnostics using helper functions
        pipeline_diagnostics = _collect_pipeline_diagnostics(pipeline_state)
        data_diagnostics = _analyze_data_diagnostics(pipeline_state)

        # Configuration diagnostics
        config_keys = [key for key in pipeline_state.keys() if 'config' in key.lower()]
        configuration_diagnostics = {
            'feature_flags': FEATURE_FLAGS,
            'config_keys': config_keys
        }

        # Generate recommendations
        recommendations = _generate_diagnostic_recommendations(pipeline_diagnostics, data_diagnostics)

        return {
            'pipeline_diagnostics': pipeline_diagnostics,
            'data_diagnostics': data_diagnostics,
            'configuration_diagnostics': configuration_diagnostics,
            'error_context': error_context or 'No error context provided',
            'recommendations': recommendations
        }

    except Exception as e:
        return {
            'diagnostic_error': str(e),
            'pipeline_state_type': type(pipeline_state).__name__,
            'recommendations': ['Diagnostic generation failed - manual review required']
        }


def handle_feature_flags(
    flag_updates: Optional[Dict[str, bool]] = None,
    feature_flags: Optional[Dict[str, bool]] = None
) -> Dict[str, bool]:
    """Handle feature flag management with validation.

    Atomic function following CODING_STANDARDS.md Section 3.1 (25-35 lines).
    Single responsibility: Feature flag management with validation.

    Parameters
    ----------
    flag_updates : Dict[str, bool], optional
        Feature flag updates to apply
    feature_flags : Dict[str, bool], optional
        Feature flags dictionary (DI pattern). If None, uses global FEATURE_FLAGS.

    Returns
    -------
    Dict[str, bool]
        Current feature flag status
    """
    global FEATURE_FLAGS

    # Use provided flags or fallback to global
    if feature_flags is None:
        feature_flags = FEATURE_FLAGS
        update_global = True
    else:
        update_global = False

    if flag_updates:
        for flag_name, value in flag_updates.items():
            if flag_name in feature_flags:
                old_value = feature_flags[flag_name]
                feature_flags[flag_name] = value
                print(f"Feature flag updated: {flag_name}: {old_value} -> {value}")

                # Update global if using global flags
                if update_global:
                    FEATURE_FLAGS[flag_name] = value
            else:
                warnings.warn(f"Unknown feature flag: {flag_name}")

    return feature_flags.copy()


def create_configuration_helpers() -> Dict[str, Any]:
    """Create configuration helper utilities for notebook use.

    Provides configuration templates and validation helpers for feature selection,
    constraint, and bootstrap configurations. Includes a built-in validation
    function for config type checking.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - templates: Dict with config templates for feature_config, constraint_config,
          and bootstrap_config
        - validation: Dict with required_keys and default_values for each config type
        - helper_functions: Dict with validate_config function

        On failure, returns dict with error key containing error message

    Notes
    -----
    Configuration templates include default values for max_candidate_features (4),
    n_samples (100), and models_to_analyze (15). Validation checks required keys
    for each configuration type.

    Warnings
    --------
    UserWarning
        If configuration helper creation fails
    """
    try:
        # Default configuration templates
        config_templates = {
            'feature_config_template': {
                'max_candidate_features': 4,
                'target_variable': 'sales_target_current',
                'analysis_start_date': '2022-08-01',
                'candidate_features': [],
                'base_features': []
            },
            'constraint_config_template': {
                'enabled': True,
                'strict_validation': True,
                'constraint_rules': {
                    'competitor_negative': True,
                    'prudential_positive': True,
                    'autoregressive_positive': True
                }
            },
            'bootstrap_config_template': {
                'enabled': True,
                'n_samples': 100,
                'confidence_intervals': [50, 70, 90],
                'models_to_analyze': 15
            }
        }

        # Configuration validation helpers
        validation_helpers = {
            'required_keys': {
                'feature_config': ['max_candidate_features', 'target_variable', 'candidate_features'],
                'constraint_config': ['enabled', 'strict_validation'],
                'bootstrap_config': ['enabled', 'n_samples']
            },
            'default_values': {
                'max_candidate_features': 4,
                'n_samples': 100,
                'models_to_analyze': 15
            }
        }

        return {
            'templates': config_templates,
            'validation': validation_helpers,
            'helper_functions': {
                'validate_config': lambda config, config_type: all(
                    key in config for key in validation_helpers['required_keys'].get(config_type, [])
                )
            }
        }

    except Exception as e:
        warnings.warn(f"Configuration helper creation failed: {e}")
        return {'error': str(e)}


def _build_utilities() -> Dict[str, Any]:
    """Build core utility functions for notebook use.

    Private helper for provide_notebook_utilities(). Creates convenience functions
    for data exploration including shape info, missing data checks, numeric summaries,
    number formatting, and feature summarization.

    Returns
    -------
    Dict[str, Any]
        Dictionary with callable utility functions:
        - display_dataframe_info: Print shape, columns, and memory usage
        - check_missing_data: Identify columns with null values
        - summarize_numeric_columns: Generate describe() summary of numeric columns
        - format_large_numbers: Format numbers with commas (1000+) or 2 decimals
        - create_feature_summary: Count total, competitor, prudential, and autoregressive features
    """
    return {
        'display_dataframe_info': lambda df: print(f"Shape: {df.shape}, Columns: {len(df.columns)}, Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"),
        'check_missing_data': lambda df: df.isnull().sum()[df.isnull().sum() > 0] if not df.empty else "Empty DataFrame",
        'summarize_numeric_columns': lambda df: df.select_dtypes(include=[np.number]).describe() if not df.empty else "No numeric columns",
        'format_large_numbers': lambda x: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}",
        'create_feature_summary': lambda features: {
            'total_features': len(features), 'competitor_features': len([f for f in features if 'competitor_' in f]),
            'prudential_features': len([f for f in features if 'prudential_' in f]),
            'autoregressive_features': len([f for f in features if 'sales_target_t' in f])}
    }


def _build_display_helpers() -> Dict[str, Any]:
    """Build display helper functions for notebook use.

    Private helper for provide_notebook_utilities(). Creates convenience functions
    for progress display, configuration visualization, and result table formatting.

    Returns
    -------
    Dict[str, Any]
        Dictionary with callable display helper functions:
        - show_progress: Print progress bar with percentage (current/total)
        - display_config_summary: Pretty-print configuration dictionary as JSON
        - create_results_table: Format results DataFrame as string table
    """
    return {
        'show_progress': lambda current, total: print(f"Progress: {current}/{total} ({current/total*100:.1f}%)"),
        'display_config_summary': lambda config: print(json.dumps(config, indent=2, default=str)),
        'create_results_table': lambda results: results.to_string(index=False) if isinstance(results, pd.DataFrame) else str(results)
    }


def _build_validation_utilities() -> Dict[str, Any]:
    """Build validation utility functions for notebook use.

    Private helper for provide_notebook_utilities(). Creates convenience functions
    for data quality checks, target variable validation, and feature availability
    verification.

    Returns
    -------
    Dict[str, Any]
        Dictionary with callable validation utility functions:
        - check_data_quality: Check if empty, calculate null percentage, count duplicates
        - validate_target_variable: Check if target column exists in DataFrame
        - check_feature_availability: Return list of features that exist in DataFrame
    """
    return {
        'check_data_quality': lambda df: {
            'empty': df.empty, 'null_percentage': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100 if not df.empty else 0,
            'duplicate_rows': df.duplicated().sum() if not df.empty else 0},
        'validate_target_variable': lambda df, target: target in df.columns if not df.empty else False,
        'check_feature_availability': lambda df, features: [f for f in features if f in df.columns] if not df.empty else []
    }


def provide_notebook_utilities(feature_flags: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """Provide notebook convenience utilities and functions.

    Aggregates all notebook utility functions (data utilities, display helpers,
    validation utilities) with optional feature flags using dependency injection.
    Uses global FEATURE_FLAGS as fallback if flags not provided.

    Parameters
    ----------
    feature_flags : Dict[str, bool], optional
        Feature flags dictionary for dependency injection. If None, uses global
        FEATURE_FLAGS singleton for backward compatibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - utilities: Data exploration and analysis functions
        - display_helpers: Display and formatting functions
        - validation_utilities: Data quality and validation functions
        - feature_flags: Feature flags used (provided or global)

        On failure, returns dict with error key and empty utilities dict

    Warnings
    --------
    UserWarning
        If notebook utilities creation fails
    """
    if feature_flags is None:
        feature_flags = FEATURE_FLAGS  # Fallback to global
    try:
        return {'utilities': _build_utilities(), 'display_helpers': _build_display_helpers(),
                'validation_utilities': _build_validation_utilities(), 'feature_flags': feature_flags}
    except Exception as e:
        warnings.warn(f"Notebook utilities creation failed: {e}")
        return {'error': str(e), 'utilities': {}}


# Initialize notebook helpers on import
print("Notebook Helpers Module Loaded")
print("Available utilities: display_results_summary, format_feature_selection_results")
print("Progress tracking: create_progress_tracker, generate_diagnostic_information")
print(f"Feature flags: {len(FEATURE_FLAGS)} flags available")
print("Use handle_feature_flags() to modify settings")