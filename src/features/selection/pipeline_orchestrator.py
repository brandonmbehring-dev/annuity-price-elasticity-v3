"""
Feature Selection Pipeline Orchestrator.

This module provides the main orchestration function that combines all
atomic feature selection functions into a complete pipeline following
established architecture patterns.

Key Functions:
- run_feature_selection_pipeline: Main orchestration function
- create_pipeline_summary: Generate comprehensive analysis summary
- validate_pipeline_inputs: Input validation with business context

Design Principles:
- Single responsibility: Pipeline orchestration only
- Immutable composition: Combines atomic functions without side effects
- Comprehensive error handling and business context
- Zero regression from existing notebook implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import time

# =============================================================================
# CONTEXT ANCHOR: PIPELINE ORCHESTRATION OBJECTIVES
# =============================================================================
# PURPOSE: Coordinate AIC, constraints, and bootstrap engines in proper sequence for feature selection
# USED BY: notebook_interface.py (single caller, no direct notebook usage)
# DEPENDENCIES: aic_engine, constraints_engine, bootstrap_engine (atomic functions only)
# LAST VALIDATED: 2025-11-12 (v3.0 cleanup - import patterns standardized)
# PATTERN STATUS: CANONICAL (single orchestration approach, no competing workflows)
#
# ARCHITECTURAL FLOW: interface → orchestrator → (aic + constraints + bootstrap) → results
# SUCCESS CRITERIA: Mathematical equivalence maintained, complete pipeline execution
# INTEGRATION: Called only from notebook_interface, never directly from notebooks
# MAINTENANCE: Engines should remain atomic functions, no business logic in orchestrator

# Clear, single import pattern for consistent context
from src.features.selection_types import (
    FeatureSelectionConfig,
    EconomicConstraintConfig,
    BootstrapAnalysisConfig,
    ExperimentConfig,
    FeatureSelectionResults,
    ConstraintViolation,
    AICResult
)
from src.features.selection.engines.aic_engine import evaluate_aic_combinations
from src.features.selection.engines.constraints_engine import apply_economic_constraints
from src.features.selection.engines.bootstrap_engine import run_bootstrap_stability

# Enhancement modules (conditionally used based on feature flags)
from src.features.selection.enhancements import (
    run_block_bootstrap_stability,
    evaluate_temporal_generalization,
)
from src.features.selection.enhancements.multiple_testing import (
    apply_multiple_testing_correction,
    reduce_search_space,
)
from src.features.selection.enhancements.statistical_constraints import (
    apply_statistical_constraints,
)
from src.features.selection.support.regression_diagnostics import (
    comprehensive_diagnostic_suite,
)

# Feature flags for enhancement control
from src.features.selection.interface.interface_config import FEATURE_FLAGS


def _validate_target_variable(
    data: pd.DataFrame,
    target: str
) -> List[str]:
    """Validate target variable exists and is numeric.

    Delegates to canonical validator in src.validation.input_validators.
    """
    from src.validation.input_validators import validate_target_with_warnings
    return validate_target_with_warnings(data, target, require_numeric=True)


def _validate_candidate_features(
    data: pd.DataFrame,
    candidate_features: List[str]
) -> tuple[List[str], List[str]]:
    """Validate candidate features availability. Returns (warnings, available_candidates)."""
    warnings = []
    available_candidates = [f for f in candidate_features if f in data.columns]
    missing_candidates = [f for f in candidate_features if f not in data.columns]

    if not available_candidates:
        warnings.append(
            f"CRITICAL: No candidate features found in dataset. "
            f"Missing: {missing_candidates[:5]}... "
            f"Check feature engineering pipeline and column naming."
        )
    elif missing_candidates:
        warnings.append(
            f"WARNING: {len(missing_candidates)} candidate features missing from dataset: "
            f"{missing_candidates[:3]}... Available: {len(available_candidates)} features."
        )
    return warnings, available_candidates


def _validate_base_features(
    data: pd.DataFrame,
    base_features: List[str]
) -> List[str]:
    """Validate that all base features exist in dataset."""
    warnings = []
    missing_base = [f for f in base_features if f not in data.columns]
    if missing_base:
        warnings.append(
            f"CRITICAL: Base features missing from dataset: {missing_base}. "
            f"Base features are required in all models - check feature engineering pipeline."
        )
    return warnings


def _validate_config_settings(
    max_candidates: int,
    available_candidates_count: int
) -> List[str]:
    """Validate configuration settings are sensible."""
    warnings = []
    if max_candidates < 1:
        warnings.append(
            f"WARNING: max_candidate_features ({max_candidates}) should be >= 1 for meaningful analysis."
        )
    elif max_candidates > available_candidates_count:
        warnings.append(
            f"WARNING: max_candidate_features ({max_candidates}) exceeds available candidates "
            f"({available_candidates_count}). Will be reduced to {available_candidates_count}."
        )
    return warnings


def _validate_data_quality(data: pd.DataFrame) -> List[str]:
    """Validate data quality (row count, etc.)."""
    warnings = []
    if len(data) < 50:
        warnings.append(
            f"WARNING: Small dataset ({len(data)} rows). "
            f"Consider at least 50+ observations for reliable statistical analysis."
        )
    return warnings


def validate_pipeline_inputs(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig
) -> List[str]:
    """Validate pipeline inputs with business context. Returns list of warnings (empty if valid)."""
    # Early exit for empty dataset
    if data.empty:
        return ["CRITICAL: Empty dataset provided. Check data loading and filtering steps."]

    warnings = []
    target = feature_config['target_variable']

    # Validate target variable
    warnings.extend(_validate_target_variable(data, target))

    # Validate candidate features
    candidate_warnings, available_candidates = _validate_candidate_features(
        data, feature_config['candidate_features']
    )
    warnings.extend(candidate_warnings)

    # Validate base features
    warnings.extend(_validate_base_features(data, feature_config['base_features']))

    # Validate configuration settings
    warnings.extend(_validate_config_settings(
        feature_config['max_candidate_features'],
        len(available_candidates)
    ))

    # Validate data quality
    warnings.extend(_validate_data_quality(data))

    return warnings


def _build_execution_metrics(results: FeatureSelectionResults) -> Dict[str, Any]:
    """Build pipeline execution metrics section."""
    success_rate = (
        f"{(results.converged_models / results.total_combinations * 100):.1f}%"
        if results.total_combinations > 0 else "0%"
    )
    return {
        'total_combinations_evaluated': results.total_combinations,
        'models_converged': results.converged_models,
        'economically_valid_models': results.economically_valid_models,
        'execution_time_seconds': results.execution_time_seconds,
        'success_rate': success_rate
    }


def _build_best_model_metrics(results: FeatureSelectionResults) -> Dict[str, Any]:
    """Build best model metrics section."""
    r_squared = results.best_model.r_squared
    if r_squared > 0.7:
        quality = 'Excellent'
    elif r_squared > 0.5:
        quality = 'Good'
    else:
        quality = 'Moderate'

    return {
        'features': results.best_model.features,
        'n_features': results.best_model.n_features,
        'aic_score': results.best_model.aic,
        'r_squared': r_squared,
        'model_fit_quality': quality
    }


def _build_constraint_metrics(results: FeatureSelectionResults) -> Dict[str, Any]:
    """Build economic constraints metrics section."""
    compliance_rate = (
        (results.economically_valid_models / results.converged_models * 100)
        if results.converged_models > 0 else 0
    )
    return {
        'constraints_enabled': results.constraint_config['enabled'],
        'total_violations': len(results.constraint_violations),
        'constraint_compliance_rate': f"{compliance_rate:.1f}%"
    }


def _build_bootstrap_metrics(results: FeatureSelectionResults) -> Optional[Dict[str, Any]]:
    """Build bootstrap analysis metrics section if available."""
    if not results.bootstrap_results:
        return None

    stable_models = sum(1 for r in results.bootstrap_results if r.stability_assessment == "STABLE")
    total_analyzed = len(results.bootstrap_results)
    stability_rate = (stable_models / total_analyzed * 100) if total_analyzed > 0 else 0

    return {
        'models_analyzed': total_analyzed,
        'stable_models': stable_models,
        'stability_rate': f"{stability_rate:.1f}%",
        'top_model_stability': results.bootstrap_results[0].stability_assessment
    }


def _build_business_interpretation(
    results: FeatureSelectionResults,
    bootstrap_stable_count: int
) -> List[str]:
    """Build business interpretation section."""
    interpretation = []

    # Model fit quality interpretation
    if results.best_model.r_squared > 0.7:
        interpretation.append("Strong predictive model identified with excellent fit quality.")
    elif results.best_model.r_squared > 0.5:
        interpretation.append("Good predictive model identified with reasonable fit quality.")
    else:
        interpretation.append("Moderate model fit - consider additional feature engineering.")

    # Economic constraint compliance interpretation
    compliance_ratio = (
        results.economically_valid_models / results.converged_models
        if results.converged_models > 0 else 0
    )
    if compliance_ratio > 0.8:
        interpretation.append("High economic constraint compliance - models align with business expectations.")
    else:
        interpretation.append("Some models violate economic constraints - review business rules or feature engineering.")

    # Bootstrap stability interpretation
    if results.bootstrap_results and bootstrap_stable_count > 0:
        interpretation.append("Bootstrap analysis confirms model stability for production use.")
    elif results.bootstrap_results:
        interpretation.append("Models show bootstrap instability - consider larger dataset or different features.")

    return interpretation


def create_pipeline_summary(results: FeatureSelectionResults) -> Dict[str, Any]:
    """
    Create comprehensive pipeline execution summary.

    Atomic function for results summarization:
    - Single responsibility: Summary generation
    - Business-oriented metrics and interpretation
    - Executive-level insights with technical details

    Parameters
    ----------
    results : FeatureSelectionResults
        Complete pipeline results

    Returns
    -------
    Dict[str, Any]
        Comprehensive execution summary with business insights
    """
    try:
        summary = {
            'pipeline_execution': _build_execution_metrics(results),
            'best_model': _build_best_model_metrics(results),
            'economic_constraints': _build_constraint_metrics(results)
        }

        # Add bootstrap analysis if available
        bootstrap_metrics = _build_bootstrap_metrics(results)
        bootstrap_stable_count = 0
        if bootstrap_metrics:
            summary['bootstrap_analysis'] = bootstrap_metrics
            bootstrap_stable_count = bootstrap_metrics['stable_models']

        # Add business interpretation
        summary['business_interpretation'] = _build_business_interpretation(
            results, bootstrap_stable_count
        )

        return summary

    except Exception as e:
        return {
            'error': f"Failed to generate pipeline summary: {str(e)}",
            'pipeline_execution': {
                'total_combinations_evaluated': getattr(results, 'total_combinations', 0),
                'success': False
            }
        }


def _create_error_result(
    error_type: str,
    error_msg: str,
    data_len: int,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    execution_time: float,
    all_results: Optional[pd.DataFrame] = None
) -> FeatureSelectionResults:
    """Create standardized error result for pipeline failures."""
    return FeatureSelectionResults(
        best_model=AICResult(
            features=error_type,
            n_features=0,
            aic=np.inf,
            bic=np.inf,
            r_squared=0.0,
            r_squared_adj=0.0,
            coefficients={},
            converged=False,
            n_obs=data_len,
            error=error_msg
        ),
        all_results=all_results if all_results is not None else pd.DataFrame(),
        valid_results=pd.DataFrame(),
        total_combinations=len(all_results) if all_results is not None else 0,
        converged_models=0,
        economically_valid_models=0,
        constraint_violations=[],
        feature_config=feature_config,
        constraint_config=constraint_config,
        execution_time_seconds=execution_time
    )


def _select_best_model(
    valid_results: pd.DataFrame,
    converged_results: pd.DataFrame
) -> AICResult:
    """Select best model from results (lowest AIC)."""
    if len(valid_results) > 0:
        best_row = valid_results.loc[valid_results['aic'].idxmin()]
        error = None
    else:
        print("WARNING: No economically valid models - using best converged model")
        best_row = converged_results.loc[converged_results['aic'].idxmin()]
        error = "Best model violates economic constraints"

    return AICResult(
        features=best_row['features'],
        n_features=best_row['n_features'],
        aic=best_row['aic'],
        bic=best_row.get('bic', best_row['aic']),
        r_squared=best_row['r_squared'],
        r_squared_adj=best_row.get('r_squared_adj', best_row['r_squared']),
        coefficients=best_row['coefficients'],
        converged=best_row['converged'],
        n_obs=best_row['n_obs'],
        error=error
    )


def _print_pipeline_summary(
    summary: Dict[str, Any],
    execution_time: float,
    has_bootstrap: bool
) -> None:
    """Print formatted pipeline execution summary."""
    print("\nPipeline Execution Summary")
    print(f"   Total combinations: {summary['pipeline_execution']['total_combinations_evaluated']}")
    print(f"   Converged models: {summary['pipeline_execution']['models_converged']} ({summary['pipeline_execution']['success_rate']})")
    print(f"   Economically valid: {summary['pipeline_execution']['economically_valid_models']}")
    print(f"   Execution time: {execution_time:.1f}s")
    print(f"\nBest Model: {summary['best_model']['features']}")
    print(f"   AIC: {summary['best_model']['aic_score']:.1f}")
    print(f"   R²: {summary['best_model']['r_squared']:.3f} ({summary['best_model']['model_fit_quality']})")
    if has_bootstrap:
        print(f"   Stability: {summary['bootstrap_analysis']['top_model_stability']}")
    print(f"\nBusiness Insights:")
    for insight in summary['business_interpretation']:
        print(f"   {insight}")


# =============================================================================
# ENHANCEMENT PHASE FUNCTIONS (Optional, controlled by feature flags)
# =============================================================================


def _run_search_space_reduction(
    feature_config: FeatureSelectionConfig,
    data: pd.DataFrame
) -> FeatureSelectionConfig:
    """Optionally reduce search space before AIC evaluation.

    Uses statistical pre-filtering to reduce the number of feature combinations
    evaluated, improving computational efficiency while maintaining validity.

    Parameters
    ----------
    feature_config : FeatureSelectionConfig
        Original feature configuration
    data : pd.DataFrame
        Dataset for statistical filtering

    Returns
    -------
    FeatureSelectionConfig
        Modified config with reduced candidate features (or original if disabled)
    """
    if not FEATURE_FLAGS.get("ENABLE_SEARCH_SPACE_REDUCTION", False):
        return feature_config

    print("\nPhase 1.5: Search Space Reduction (Pre-AIC Filtering)")
    try:
        reduced_features = reduce_search_space(
            data=data,
            candidate_features=feature_config['candidate_features'],
            target_variable=feature_config['target_variable'],
        )
        original_count = len(feature_config['candidate_features'])
        reduced_count = len(reduced_features)
        print(f"SUCCESS: Reduced candidates from {original_count} to {reduced_count}")

        # Return modified config with reduced candidates
        return FeatureSelectionConfig(
            base_features=feature_config['base_features'],
            candidate_features=reduced_features,
            max_candidate_features=feature_config['max_candidate_features'],
            target_variable=feature_config['target_variable'],
        )
    except Exception as e:
        print(f"WARNING: Search space reduction failed ({e}), using original candidates")
        return feature_config


def _run_multiple_testing_correction(
    converged_results: pd.DataFrame
) -> pd.DataFrame:
    """Apply multiple testing corrections to p-values.

    Addresses Issue #1 (Multiple Testing Problem) from the mathematical analysis.
    Applies Bonferroni or FDR corrections to control false discovery rates.

    Parameters
    ----------
    converged_results : pd.DataFrame
        Converged model results with p-values

    Returns
    -------
    pd.DataFrame
        Results with corrected p-values (or unchanged if disabled)
    """
    if not FEATURE_FLAGS.get("ENABLE_MULTIPLE_TESTING", False):
        return converged_results

    print("\nPhase 2.5: Multiple Testing Correction (FWER/FDR)")
    try:
        corrected_results = apply_multiple_testing_correction(converged_results)
        print("SUCCESS: Applied Bonferroni/FDR corrections")
        return corrected_results
    except Exception as e:
        print(f"WARNING: Multiple testing correction failed ({e}), using original p-values")
        return converged_results


def _run_oos_validation(
    data: pd.DataFrame,
    valid_results: pd.DataFrame,
    target_variable: str
) -> Optional[Dict[str, Any]]:
    """Run out-of-sample validation for top models.

    Addresses Issue #2 (Missing OOS Validation) from the mathematical analysis.
    Provides evidence of model generalization to unseen data.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    valid_results : pd.DataFrame
        Economically valid model results
    target_variable : str
        Target variable name

    Returns
    -------
    Optional[Dict[str, Any]]
        OOS validation results (or None if disabled)
    """
    if not FEATURE_FLAGS.get("ENABLE_OOS_VALIDATION", False):
        return None

    print("\nPhase 3.5: Out-of-Sample Validation")
    try:
        oos_results = evaluate_temporal_generalization(
            model_results=valid_results,
            data=data,
            target_variable=target_variable,
        )
        print("SUCCESS: Completed OOS validation")
        return oos_results
    except Exception as e:
        print(f"WARNING: OOS validation failed ({e})")
        return None


def _run_block_bootstrap(
    data: pd.DataFrame,
    valid_results: pd.DataFrame,
    bootstrap_config: Optional[BootstrapAnalysisConfig],
    target_variable: str
) -> Optional[List[Any]]:
    """Run block bootstrap for time series data.

    Addresses Issue #4 (Time Series Bootstrap Violations) from the mathematical
    analysis. Uses contiguous blocks to preserve temporal dependencies.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset with time series structure
    valid_results : pd.DataFrame
        Economically valid results
    bootstrap_config : Optional[BootstrapAnalysisConfig]
        Bootstrap configuration
    target_variable : str
        Target variable name

    Returns
    -------
    Optional[List[Any]]
        Block bootstrap results (or None if disabled/not configured)
    """
    if not bootstrap_config or not bootstrap_config.get('enabled', False):
        return None

    # Check if block bootstrap is requested via feature flag or config
    use_block = (
        FEATURE_FLAGS.get("ENABLE_BLOCK_BOOTSTRAP", False) or
        bootstrap_config.get('use_block_bootstrap', False)
    )

    if not use_block:
        return None  # Will fall through to standard bootstrap

    print("\nPhase 4: Block Bootstrap Stability Analysis (Time Series)")
    try:
        block_size = bootstrap_config.get('block_size', 4)  # Default: 4 weeks
        n_samples = bootstrap_config.get('n_samples', 1000)
        results = run_block_bootstrap_stability(
            model_results=valid_results,
            data=data,
            target_variable=target_variable,
            block_size=block_size,
            n_bootstrap_samples=n_samples,
        )
        print(f"SUCCESS: Block bootstrap ({block_size}-week blocks, {n_samples} samples)")
        return results
    except Exception as e:
        print(f"WARNING: Block bootstrap failed ({e}), falling back to standard")
        return None


def _run_regression_diagnostics(
    data: pd.DataFrame,
    best_model: Any,
    target_variable: str
) -> Optional[Dict[str, Any]]:
    """Run comprehensive regression diagnostics on best model.

    Addresses Issue #8 (Missing Regression Diagnostics) from the mathematical
    analysis. Checks autocorrelation, heteroscedasticity, multicollinearity.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset for diagnostics
    best_model : Any
        Selected best model result
    target_variable : str
        Target variable name

    Returns
    -------
    Optional[Dict[str, Any]]
        Diagnostic results (or None if disabled)
    """
    if not FEATURE_FLAGS.get("ENABLE_REGRESSION_DIAGNOSTICS", False):
        return None

    print("\nPhase 5: Regression Diagnostics")
    try:
        # Need to fit a model to run diagnostics - comprehensive_diagnostic_suite takes
        # a fitted model, so we need to refit or pass the model object
        # For now, we'll skip if features aren't available as a list
        features = best_model.features if hasattr(best_model, 'features') else []
        if isinstance(features, str):
            # Features stored as comma-separated string
            features = [f.strip() for f in features.split(',')]

        if not features:
            print("WARNING: No features available for diagnostics")
            return None

        # Build formula and fit model for diagnostics
        import statsmodels.formula.api as smf
        formula = f"{target_variable} ~ " + " + ".join(features)
        model = smf.ols(formula, data=data).fit()

        diagnostics = comprehensive_diagnostic_suite(
            model=model,
            data=data,
            features=features,
            target_variable=target_variable,
        )
        print("SUCCESS: Completed assumption validation")
        return diagnostics
    except Exception as e:
        print(f"WARNING: Regression diagnostics failed ({e})")
        return None


def _run_statistical_constraints(
    valid_results: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Apply CI-based statistical constraints.

    Provides additional validation beyond economic sign constraints,
    checking confidence interval-based significance.

    Parameters
    ----------
    valid_results : pd.DataFrame
        Economically valid model results

    Returns
    -------
    Optional[Dict[str, Any]]
        Statistical constraint results (or None if disabled)
    """
    if not FEATURE_FLAGS.get("ENABLE_STATISTICAL_CONSTRAINTS", False):
        return None

    print("\nPhase 3.7: Statistical Constraints (CI-based)")
    try:
        stat_results = apply_statistical_constraints(valid_results)
        print("SUCCESS: Applied statistical constraints")
        return stat_results
    except Exception as e:
        print(f"WARNING: Statistical constraints failed ({e})")
        return None


# =============================================================================
# CORE PIPELINE PHASE FUNCTIONS
# =============================================================================


def _run_aic_evaluation(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run AIC evaluation phase. Returns (all_results, converged_results)."""
    print("\nPhase 2: AIC Evaluation Across Feature Combinations")
    all_results = evaluate_aic_combinations(data, feature_config)
    converged_results = all_results[all_results['converged']].copy()
    print(f"SUCCESS: {len(converged_results)}/{len(all_results)} models converged")
    return all_results, converged_results


def _run_constraint_validation(
    converged_results: pd.DataFrame,
    constraint_config: EconomicConstraintConfig
) -> tuple[pd.DataFrame, List[ConstraintViolation]]:
    """Run economic constraint validation phase."""
    print("\nPhase 3: Economic Constraint Validation")
    return apply_economic_constraints(converged_results, constraint_config)


def _run_bootstrap_analysis(
    data: pd.DataFrame,
    valid_results: pd.DataFrame,
    bootstrap_config: Optional[BootstrapAnalysisConfig],
    target_variable: str
) -> Optional[List[Any]]:
    """Run optional bootstrap stability analysis."""
    if not bootstrap_config or not bootstrap_config.get('enabled', False):
        return None
    print("\nPhase 4: Bootstrap Stability Analysis")
    return run_bootstrap_stability(data, valid_results, bootstrap_config, target_variable)


def _compile_final_results(
    best_model: AICResult,
    all_results: pd.DataFrame,
    valid_results: pd.DataFrame,
    converged_results: pd.DataFrame,
    constraint_violations: List[ConstraintViolation],
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    bootstrap_results: Optional[List[Any]],
    bootstrap_config: Optional[BootstrapAnalysisConfig],
    experiment_config: Optional[ExperimentConfig],
    execution_time: float
) -> FeatureSelectionResults:
    """Compile all results into final FeatureSelectionResults."""
    return FeatureSelectionResults(
        best_model=best_model,
        all_results=all_results,
        valid_results=valid_results,
        total_combinations=len(all_results),
        converged_models=len(converged_results),
        economically_valid_models=len(valid_results),
        constraint_violations=constraint_violations,
        feature_config=feature_config,
        constraint_config=constraint_config,
        bootstrap_results=bootstrap_results,
        bootstrap_config=bootstrap_config,
        experiment_config=experiment_config,
        execution_time_seconds=execution_time
    )


def _check_critical_validation_failures(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    start_time: float
) -> Optional[FeatureSelectionResults]:
    """Check for critical validation failures. Returns error result if critical, None otherwise."""
    print("\nPhase 1: Input Validation")
    validation_warnings = validate_pipeline_inputs(data, feature_config, constraint_config)
    critical_warnings = [w for w in validation_warnings if w.startswith("CRITICAL")]
    if critical_warnings:
        return _create_error_result(
            "VALIDATION_FAILED", f"Critical failures: {len(critical_warnings)}",
            len(data), feature_config, constraint_config, time.time() - start_time
        )
    return None


def _finalize_and_report(
    results: FeatureSelectionResults,
    execution_time: float,
    has_bootstrap: bool
) -> FeatureSelectionResults:
    """Generate summary, print report, and return final results."""
    summary = create_pipeline_summary(results)
    _print_pipeline_summary(summary, execution_time, has_bootstrap)
    print("SUCCESS: Feature Selection Pipeline Complete!")
    return results


def _execute_pipeline_phases(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    bootstrap_config: Optional[BootstrapAnalysisConfig],
    experiment_config: Optional[ExperimentConfig],
    start_time: float
) -> FeatureSelectionResults:
    """Execute all pipeline phases and return results.

    Pipeline Flow (with optional enhancements):
    1. Input Validation (already done before this function)
    1.5. [Optional] Search Space Reduction
    2. AIC Evaluation
    2.5. [Optional] Multiple Testing Correction
    3. Economic Constraints
    3.5. [Optional] Out-of-Sample Validation
    3.7. [Optional] Statistical Constraints
    4. Bootstrap Analysis (or Block Bootstrap)
    5. [Optional] Regression Diagnostics
    """
    # Phase 1.5: Optional Search Space Reduction (Pre-AIC)
    effective_config = _run_search_space_reduction(feature_config, data)

    # Phase 2: AIC Evaluation
    all_results, converged_results = _run_aic_evaluation(data, effective_config)
    if len(converged_results) == 0:
        return _create_error_result(
            "NO_CONVERGED_MODELS", "No models converged during AIC evaluation",
            len(data), feature_config, constraint_config, time.time() - start_time, all_results
        )

    # Phase 2.5: Optional Multiple Testing Correction
    converged_results = _run_multiple_testing_correction(converged_results)

    # Phase 3: Economic Constraints
    valid_results, constraint_violations = _run_constraint_validation(
        converged_results, constraint_config
    )

    # Phase 3.5: Optional Out-of-Sample Validation
    oos_results = _run_oos_validation(
        data, valid_results, feature_config['target_variable']
    )

    # Phase 3.7: Optional Statistical Constraints
    stat_constraint_results = _run_statistical_constraints(valid_results)

    # Phase 4: Bootstrap Analysis (Block or Standard)
    # Try block bootstrap first if enabled, fall back to standard
    bootstrap_results = _run_block_bootstrap(
        data, valid_results, bootstrap_config, feature_config['target_variable']
    )
    if bootstrap_results is None:
        # Fall back to standard bootstrap
        bootstrap_results = _run_bootstrap_analysis(
            data, valid_results, bootstrap_config, feature_config['target_variable']
        )

    # Select best model
    best_model = _select_best_model(valid_results, converged_results)

    # Phase 5: Optional Regression Diagnostics (post-selection)
    diagnostics_results = _run_regression_diagnostics(
        data, best_model, feature_config['target_variable']
    )

    execution_time = time.time() - start_time

    # Compile results (core fields only - enhancements stored separately)
    results = _compile_final_results(
        best_model, all_results, valid_results, converged_results,
        constraint_violations, feature_config, constraint_config,
        bootstrap_results, bootstrap_config, experiment_config, execution_time
    )

    # Store enhancement results as additional attributes if available
    # Note: These are optional extensions, not breaking the core dataclass
    if oos_results is not None:
        results.oos_validation = oos_results  # type: ignore[attr-defined]
    if stat_constraint_results is not None:
        results.statistical_constraints = stat_constraint_results  # type: ignore[attr-defined]
    if diagnostics_results is not None:
        results.regression_diagnostics = diagnostics_results  # type: ignore[attr-defined]

    return _finalize_and_report(results, execution_time, bootstrap_results is not None)


def run_feature_selection_pipeline(
    data: pd.DataFrame,
    feature_config: FeatureSelectionConfig,
    constraint_config: EconomicConstraintConfig,
    bootstrap_config: Optional[BootstrapAnalysisConfig] = None,
    experiment_config: Optional[ExperimentConfig] = None
) -> FeatureSelectionResults:
    """
    Run complete feature selection pipeline with all atomic functions.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing features and target variable
    feature_config : FeatureSelectionConfig
        Type-safe feature selection configuration
    constraint_config : EconomicConstraintConfig
        Economic constraints configuration
    bootstrap_config : Optional[BootstrapAnalysisConfig]
        Bootstrap analysis configuration (None to disable)
    experiment_config : Optional[ExperimentConfig]
        MLflow experiment configuration (None to disable)

    Returns
    -------
    FeatureSelectionResults
        Complete pipeline results with structured analysis
    """
    start_time = time.time()
    print("Starting Feature Selection Pipeline...")
    print(f"Dataset: {len(data)} rows x {len(data.columns)} columns")

    try:
        # Phase 1: Input Validation
        validation_error = _check_critical_validation_failures(
            data, feature_config, constraint_config, start_time
        )
        if validation_error:
            return validation_error

        # Execute phases 2-4 and return results
        return _execute_pipeline_phases(
            data, feature_config, constraint_config,
            bootstrap_config, experiment_config, start_time
        )

    except Exception as e:
        print(f"ERROR: Pipeline failed: {str(e)}")
        return _create_error_result(
            "PIPELINE_ERROR", f"Pipeline execution failed: {str(e)}",
            len(data) if not data.empty else 0,
            feature_config, constraint_config, time.time() - start_time
        )