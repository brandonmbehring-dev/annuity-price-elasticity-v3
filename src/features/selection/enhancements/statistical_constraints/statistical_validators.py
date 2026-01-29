"""
Statistical Validators for Constraint Engine.

This module contains validation and calculation functions for statistical
constraint checking. Extracted from statistical_constraints_engine.py.

Functions:
- _validate_model_inputs: Validate model and constraint inputs
- _determine_constraint_type: Parse constraint type from specification
- _calculate_confidence_interval: Compute CI for coefficient
- _evaluate_sign_constraint: Evaluate constraint based on sign expectation
- _build_hard_threshold_comparison: Compare hard vs statistical thresholds
- _validate_single_statistical_constraint: Core constraint validator
- _assess_constraint_strength: Assess strength of constraint satisfaction

Design Principles:
- Single responsibility: validation and calculation only
- No interpretation or recommendation generation
- Pure functions with clear inputs/outputs
"""

import logging
from typing import Any, Callable, Dict, Tuple, Optional

import numpy as np

from src.features.selection.enhancements.statistical_constraints.constraint_types import (
    ConstraintType,
    StatisticalConstraintResult,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def _validate_model_inputs(
    model: Any, constraint_specifications: Dict[str, Dict[str, Any]]
) -> None:
    """
    Validate model and constraint specification inputs.

    Parameters
    ----------
    model : statsmodels regression model
        Fitted model to validate
    constraint_specifications : Dict[str, Dict[str, Any]]
        Constraint rules to validate

    Raises
    ------
    ValueError
        If model or constraint specifications are invalid
    """
    if not hasattr(model, 'params') or not hasattr(model, 'bse'):
        raise ValueError(
            f"CRITICAL: Model does not have required coefficient attributes. "
            f"Business impact: Cannot validate economic constraints statistically. "
            f"Required action: Ensure model is fitted statsmodels regression."
        )

    if not constraint_specifications:
        raise ValueError(
            f"CRITICAL: No constraint specifications provided. "
            f"Business impact: Cannot validate business theory compliance. "
            f"Required action: Provide economic constraint rules."
        )


# =============================================================================
# CONSTRAINT TYPE DETERMINATION
# =============================================================================


def _determine_constraint_type(expected_sign: str) -> ConstraintType:
    """
    Determine constraint type from expected sign specification.

    Parameters
    ----------
    expected_sign : str
        Expected sign ('positive', 'negative', or other for magnitude)

    Returns
    -------
    ConstraintType
        The appropriate constraint type enum value
    """
    if expected_sign == 'positive':
        return ConstraintType.POSITIVE
    elif expected_sign == 'negative':
        return ConstraintType.NEGATIVE
    return ConstraintType.MAGNITUDE


# =============================================================================
# CONFIDENCE INTERVAL CALCULATION
# =============================================================================


def _calculate_confidence_interval(
    coefficient: float,
    standard_error: float,
    confidence_level: float,
    degrees_freedom: int
) -> Tuple[float, float]:
    """
    Calculate confidence interval for coefficient estimate.

    Parameters
    ----------
    coefficient : float
        Point estimate
    standard_error : float
        Standard error of coefficient
    confidence_level : float
        Confidence level (e.g., 0.95)
    degrees_freedom : int
        Degrees of freedom for t-distribution

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound) confidence interval
    """
    from scipy.stats import t
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha / 2, degrees_freedom)
    ci_lower = coefficient - t_critical * standard_error
    ci_upper = coefficient + t_critical * standard_error
    return (ci_lower, ci_upper)


# =============================================================================
# CONSTRAINT STRENGTH ASSESSMENT
# =============================================================================


def _assess_constraint_strength(
    coefficient: float,
    ci_lower: float,
    ci_upper: float,
    expected_sign: str
) -> str:
    """
    Assess strength of constraint satisfaction based on CI position.

    Single responsibility: Constraint strength assessment only.

    Parameters
    ----------
    coefficient : float
        Point estimate
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    expected_sign : str
        Expected coefficient sign ('positive' or 'negative')

    Returns
    -------
    str
        Constraint strength ('STRONG', 'MODERATE', 'WEAK', 'VIOLATED')
    """
    if expected_sign == 'positive':
        if ci_lower > 0.01:  # Strongly positive
            return "STRONG"
        elif ci_lower > 0:  # Marginally positive
            return "MODERATE"
        elif coefficient > 0 and ci_upper > 0:  # Point positive but uncertain
            return "WEAK"
        else:  # Negative or zero
            return "VIOLATED"
    else:  # negative expected
        if ci_upper < -0.01:  # Strongly negative
            return "STRONG"
        elif ci_upper < 0:  # Marginally negative
            return "MODERATE"
        elif coefficient < 0 and ci_lower < 0:  # Point negative but uncertain
            return "WEAK"
        else:  # Positive or zero
            return "VIOLATED"


# =============================================================================
# SIGN CONSTRAINT EVALUATION
# =============================================================================


def _evaluate_sign_constraint(
    constraint_type: ConstraintType,
    coefficient: float,
    ci_lower: float,
    ci_upper: float,
    business_rationale: str,
    statistically_significant: bool,
    minimum_magnitude: float,
    interpret_positive_fn: Callable[..., str],
    interpret_negative_fn: Callable[..., str]
) -> Tuple[bool, str, str]:
    """
    Evaluate constraint satisfaction based on sign expectation.

    Parameters
    ----------
    constraint_type : ConstraintType
        Type of constraint (POSITIVE, NEGATIVE, MAGNITUDE)
    coefficient : float
        Coefficient point estimate
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    business_rationale : str
        Business rationale for constraint
    statistically_significant : bool
        Whether coefficient is statistically significant
    minimum_magnitude : float
        Minimum magnitude threshold (for MAGNITUDE constraints)
    interpret_positive_fn : callable
        Function to interpret positive constraints
    interpret_negative_fn : callable
        Function to interpret negative constraints

    Returns
    -------
    Tuple[bool, str, str]
        (constraint_satisfied, constraint_strength, business_interpretation)
    """
    if constraint_type == ConstraintType.POSITIVE:
        constraint_satisfied = ci_lower > 0
        constraint_strength = _assess_constraint_strength(
            coefficient, ci_lower, ci_upper, expected_sign='positive'
        )
        business_interpretation = interpret_positive_fn(
            coefficient, ci_lower, ci_upper, business_rationale, statistically_significant
        )
    elif constraint_type == ConstraintType.NEGATIVE:
        constraint_satisfied = ci_upper < 0
        constraint_strength = _assess_constraint_strength(
            coefficient, ci_lower, ci_upper, expected_sign='negative'
        )
        business_interpretation = interpret_negative_fn(
            coefficient, ci_lower, ci_upper, business_rationale, statistically_significant
        )
    else:  # MAGNITUDE constraint
        constraint_satisfied = statistically_significant and abs(coefficient) >= minimum_magnitude
        constraint_strength = "MODERATE" if constraint_satisfied else "WEAK"
        business_interpretation = f"Magnitude constraint assessment: |{coefficient:.4f}| vs {minimum_magnitude}"

    return constraint_satisfied, constraint_strength, business_interpretation


# =============================================================================
# HARD THRESHOLD COMPARISON
# =============================================================================


def _build_hard_threshold_comparison(
    expected_sign: str,
    coefficient: float,
    constraint_satisfied: bool,
    minimum_magnitude: float
) -> Dict[str, bool]:
    """
    Build comparison between hard threshold and statistical approaches.

    Parameters
    ----------
    expected_sign : str
        Expected coefficient sign
    coefficient : float
        Coefficient point estimate
    constraint_satisfied : bool
        Whether statistical constraint is satisfied
    minimum_magnitude : float
        Minimum magnitude threshold

    Returns
    -------
    Dict[str, bool]
        Comparison dictionary with hard_threshold_passes,
        statistical_approach_passes, methods_agree
    """
    if expected_sign == 'positive':
        hard_passes = coefficient > 0
    elif expected_sign == 'negative':
        hard_passes = coefficient < 0
    else:
        hard_passes = abs(coefficient) >= minimum_magnitude

    return {
        'hard_threshold_passes': hard_passes,
        'statistical_approach_passes': constraint_satisfied,
        'methods_agree': hard_passes == constraint_satisfied
    }


# =============================================================================
# CORE CONSTRAINT VALIDATOR
# =============================================================================


def _validate_single_statistical_constraint(
    feature_name: str,
    coefficient: float,
    standard_error: float,
    t_statistic: float,
    p_value: float,
    constraint_spec: Dict[str, Any],
    confidence_level: float,
    significance_level: float,
    degrees_freedom: int,
    interpret_positive_fn: Callable[..., str],
    interpret_negative_fn: Callable[..., str]
) -> StatisticalConstraintResult:
    """
    Validate single constraint using statistical significance and CIs.

    Parameters
    ----------
    feature_name : str
        Feature being validated
    coefficient, standard_error, t_statistic, p_value : float
        Coefficient statistics
    constraint_spec : Dict[str, Any]
        Constraint specification
    confidence_level, significance_level : float
        Statistical thresholds
    degrees_freedom : int
        Degrees of freedom for t-distribution
    interpret_positive_fn, interpret_negative_fn : callable
        Interpretation functions

    Returns
    -------
    StatisticalConstraintResult
    """
    # Extract constraint specification
    expected_sign = constraint_spec.get('expected_sign', 'positive')
    minimum_magnitude = constraint_spec.get('minimum_magnitude', 0.0)
    business_rationale = constraint_spec.get('business_rationale', 'Economic theory expectation')

    constraint_type = _determine_constraint_type(expected_sign)
    ci_lower, ci_upper = _calculate_confidence_interval(
        coefficient, standard_error, confidence_level, degrees_freedom
    )
    statistically_significant = p_value < significance_level

    constraint_satisfied, constraint_strength, business_interpretation = _evaluate_sign_constraint(
        constraint_type, coefficient, ci_lower, ci_upper,
        business_rationale, statistically_significant, minimum_magnitude,
        interpret_positive_fn, interpret_negative_fn
    )

    hard_threshold_comparison = _build_hard_threshold_comparison(
        expected_sign, coefficient, constraint_satisfied, minimum_magnitude
    )

    return StatisticalConstraintResult(
        feature_name=feature_name,
        constraint_type=constraint_type,
        coefficient_estimate=coefficient,
        standard_error=standard_error,
        confidence_interval=(ci_lower, ci_upper),
        t_statistic=t_statistic,
        p_value=p_value,
        statistically_significant=statistically_significant,
        constraint_satisfied=constraint_satisfied,
        constraint_strength=constraint_strength,
        business_interpretation=business_interpretation,
        hard_threshold_comparison=hard_threshold_comparison
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    '_validate_model_inputs',
    '_determine_constraint_type',
    '_calculate_confidence_interval',
    '_assess_constraint_strength',
    '_evaluate_sign_constraint',
    '_build_hard_threshold_comparison',
    '_validate_single_statistical_constraint',
]
