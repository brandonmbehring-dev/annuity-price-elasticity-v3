"""
Statistical Constraints Engine for Feature Selection.

This module addresses Issue #6 from the mathematical analysis report:
Economic constraints without statistical uncertainty, where hard thresholds
at zero ignore coefficient significance and confidence intervals, leading
to economically meaningless constraint validation.

Key Functions:
- apply_statistical_constraints: CI-based constraint validation
- calculate_coefficient_significance: Statistical significance testing
- compare_constraint_methods: Hard vs statistical threshold comparison
- generate_constraint_recommendations: Business-interpretable guidance

Critical Statistical Issues Addressed:
- Issue #6: Economic Constraints Without Statistical Uncertainty (SEVERITY: MODERATE)
- Hard Threshold Problems: coef = -0.0001 vs +0.0001 treated completely differently
- Missing Significance: coef = 0.01 ± 5.0 (not significant) still passes hard constraints
- No Uncertainty Assessment: Should use confidence intervals, not point estimates

Mathematical Foundation:
- Confidence Intervals: β̂ ± t_(α/2,df) × SE(β̂)
- Statistical Significance: Test H₀: β = 0 vs H₁: β ≠ 0
- Economic Constraints: Test if coefficient significantly different from zero with expected sign
- Power Analysis: Assess Type II error (missing true economic relationships)

Design Principles:
- Statistical significance incorporated in economic validation
- Confidence interval-based constraint checking
- Comparative analysis with existing hard threshold approach
- Business-interpretable uncertainty quantification

Module Architecture (Phase 6.3c Split):
- constraint_types.py: Shared dataclasses and enums
- statistical_validators.py: Validation and calculation functions
- constraint_analyzers.py: Analysis, interpretation, and recommendations
- statistical_constraints_engine.py: Orchestrator + public API (this file)
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# RE-EXPORTS FROM SPLIT MODULES
# =============================================================================

# Shared types
from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
    ComprehensiveConstraintAnalysis,
)

# Validators
from src.features.selection.enhancements.statistical_constraints.statistical_validators import (
    _validate_model_inputs,
    _determine_constraint_type,
    _calculate_confidence_interval,
    _assess_constraint_strength,
    _evaluate_sign_constraint,
    _build_hard_threshold_comparison,
    _validate_single_statistical_constraint,
)

# Analyzers
from src.features.selection.enhancements.statistical_constraints.constraint_analyzers import (
    _interpret_positive_constraint,
    _interpret_negative_constraint,
    _count_constraint_outcomes,
    _categorize_features_by_strength,
    _determine_compliance_level,
    _assess_overall_constraint_compliance,
    _compute_methodology_summary,
    _identify_method_advantages,
    _build_disagreement_analysis,
    _compare_constraint_methodologies,
    _build_feature_recommendations,
    _generate_constraint_recommendations,
    _compute_average_precision,
    _estimate_power_metrics,
    _interpret_power,
    _analyze_constraint_detection_power,
)


# =============================================================================
# ORCHESTRATION UTILITIES
# =============================================================================


def _extract_model_specification(model: Any) -> Dict[str, Any]:
    """
    Extract model specification metadata for constraint analysis context.

    Parameters
    ----------
    model : statsmodels regression model
        Fitted model with standard attributes

    Returns
    -------
    Dict[str, Any]
        Model specification dictionary
    """
    return {
        'n_observations': int(model.nobs),
        'n_parameters': len(model.params),
        'model_r_squared': model.rsquared,
        'model_aic': model.aic,
        'degrees_freedom': int(model.df_resid),
        'constraint_validation_timestamp': datetime.now().isoformat()
    }


def _process_constraint_for_feature(
    feature_name: str,
    constraint_spec: Dict[str, Any],
    model: Any,
    confidence_level: float,
    significance_level: float
) -> Optional[StatisticalConstraintResult]:
    """
    Process a single constraint specification for a feature.

    Parameters
    ----------
    feature_name : str
        Feature to validate
    constraint_spec : Dict[str, Any]
        Constraint specification
    model : statsmodels regression model
        Fitted model
    confidence_level : float
        Confidence level for CI
    significance_level : float
        Statistical significance threshold

    Returns
    -------
    Optional[StatisticalConstraintResult]
        Constraint result or None if feature not in model
    """
    if feature_name not in model.params.index:
        logger.warning(f"Feature {feature_name} not found in model coefficients")
        return None

    return _validate_single_statistical_constraint(
        feature_name=feature_name,
        coefficient=model.params[feature_name],
        standard_error=model.bse[feature_name],
        t_statistic=model.tvalues[feature_name],
        p_value=model.pvalues[feature_name],
        constraint_spec=constraint_spec,
        confidence_level=confidence_level,
        significance_level=significance_level,
        degrees_freedom=model.df_resid,
        interpret_positive_fn=_interpret_positive_constraint,
        interpret_negative_fn=_interpret_negative_constraint
    )


# =============================================================================
# PUBLIC API
# =============================================================================


def apply_statistical_constraints(
    model: Any,
    constraint_specifications: Dict[str, Dict[str, Any]],
    confidence_level: float = 0.95,
    significance_level: float = 0.05
) -> ComprehensiveConstraintAnalysis:
    """Apply statistical constraint validation using confidence intervals."""
    try:
        _validate_model_inputs(model, constraint_specifications)
        model_spec = _extract_model_specification(model)

        # Process each constraint
        constraint_results = [
            result for result in (
                _process_constraint_for_feature(
                    feature_name, constraint_spec, model,
                    confidence_level, significance_level
                )
                for feature_name, constraint_spec in constraint_specifications.items()
            )
            if result is not None
        ]

        # Synthesize analysis components
        overall_assessment = _assess_overall_constraint_compliance(constraint_results)
        methodology_comparison = _compare_constraint_methodologies(
            constraint_results, constraint_specifications
        )
        business_recommendations = _generate_constraint_recommendations(
            constraint_results, overall_assessment
        )
        power_analysis = _analyze_constraint_detection_power(
            constraint_results, model_spec['n_observations']
        )

        return ComprehensiveConstraintAnalysis(
            model_specification=model_spec,
            constraint_results=constraint_results,
            overall_assessment=overall_assessment,
            methodology_comparison=methodology_comparison,
            business_recommendations=business_recommendations,
            power_analysis=power_analysis
        )

    except Exception as e:
        raise ValueError(
            f"Statistical constraint validation failed: {e}. "
            f"Check model specification and constraint rules."
        ) from e


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Types
    'ConstraintType',
    'StatisticalConstraintResult',
    'ComprehensiveConstraintAnalysis',

    # Public API
    'apply_statistical_constraints',

    # Re-exported validators (for backward compatibility)
    '_validate_model_inputs',
    '_determine_constraint_type',
    '_calculate_confidence_interval',
    '_assess_constraint_strength',
    '_evaluate_sign_constraint',
    '_build_hard_threshold_comparison',
    '_validate_single_statistical_constraint',

    # Re-exported analyzers (for backward compatibility)
    '_interpret_positive_constraint',
    '_interpret_negative_constraint',
    '_count_constraint_outcomes',
    '_categorize_features_by_strength',
    '_determine_compliance_level',
    '_assess_overall_constraint_compliance',
    '_compute_methodology_summary',
    '_identify_method_advantages',
    '_build_disagreement_analysis',
    '_compare_constraint_methodologies',
    '_build_feature_recommendations',
    '_generate_constraint_recommendations',
    '_compute_average_precision',
    '_estimate_power_metrics',
    '_interpret_power',
    '_analyze_constraint_detection_power',
]
