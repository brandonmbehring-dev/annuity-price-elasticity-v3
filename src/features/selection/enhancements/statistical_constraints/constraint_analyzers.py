"""
Constraint Analyzers for Statistical Constraints Engine.

This module contains analysis, interpretation, and recommendation functions
for statistical constraint validation. Extracted from statistical_constraints_engine.py.

Functions:
- Interpretation: _interpret_positive_constraint, _interpret_negative_constraint
- Aggregation: _count_constraint_outcomes, _categorize_features_by_strength
- Compliance: _determine_compliance_level, _assess_overall_constraint_compliance
- Methodology: _compute_methodology_summary, _identify_method_advantages,
               _build_disagreement_analysis, _compare_constraint_methodologies
- Recommendations: _build_feature_recommendations, _generate_constraint_recommendations
- Power Analysis: _compute_average_precision, _estimate_power_metrics,
                  _interpret_power, _analyze_constraint_detection_power

Design Principles:
- Single responsibility: analysis and interpretation only
- No validation or calculation logic
- Clear separation from core validators
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.features.selection.enhancements.statistical_constraints.constraint_types import StatisticalConstraintResult

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# BUSINESS INTERPRETATION
# =============================================================================


def _interpret_positive_constraint(
    coefficient: float,
    ci_lower: float,
    ci_upper: float,
    business_rationale: str,
    statistically_significant: bool
) -> str:
    """
    Generate business interpretation for positive constraint validation.

    Single responsibility: Business interpretation only.

    Parameters
    ----------
    coefficient : float
        Coefficient estimate
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    business_rationale : str
        Business rationale for constraint
    statistically_significant : bool
        Statistical significance of coefficient

    Returns
    -------
    str
        Business-friendly interpretation
    """
    if ci_lower > 0:
        return (f"STRONG evidence for positive effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"{business_rationale} is statistically supported.")
    elif coefficient > 0 and statistically_significant:
        return (f"MODERATE evidence for positive effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"{business_rationale} has some support but uncertainty remains.")
    elif coefficient > 0:
        return (f"WEAK evidence for positive effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"Point estimate suggests {business_rationale.lower()}, but not statistically significant.")
    else:
        return (f"VIOLATED: Negative effect detected ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"Contradicts expectation: {business_rationale.lower()}.")


def _interpret_negative_constraint(
    coefficient: float,
    ci_lower: float,
    ci_upper: float,
    business_rationale: str,
    statistically_significant: bool
) -> str:
    """
    Generate business interpretation for negative constraint validation.

    Single responsibility: Business interpretation only.

    Parameters
    ----------
    coefficient : float
        Coefficient estimate
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    business_rationale : str
        Business rationale for constraint
    statistically_significant : bool
        Statistical significance of coefficient

    Returns
    -------
    str
        Business-friendly interpretation
    """
    if ci_upper < 0:
        return (f"STRONG evidence for negative effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"{business_rationale} is statistically supported.")
    elif coefficient < 0 and statistically_significant:
        return (f"MODERATE evidence for negative effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"{business_rationale} has some support but uncertainty remains.")
    elif coefficient < 0:
        return (f"WEAK evidence for negative effect ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"Point estimate suggests {business_rationale.lower()}, but not statistically significant.")
    else:
        return (f"VIOLATED: Positive effect detected ({coefficient:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
                f"Contradicts expectation: {business_rationale.lower()}.")


# =============================================================================
# CONSTRAINT OUTCOME AGGREGATION
# =============================================================================


def _count_constraint_outcomes(
    constraint_results: List[StatisticalConstraintResult]
) -> Dict[str, int]:
    """
    Count constraint outcomes by strength category.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint validation results

    Returns
    -------
    Dict[str, int]
        Counts for each strength category
    """
    return {
        'total': len(constraint_results),
        'strong': sum(1 for r in constraint_results if r.constraint_strength == "STRONG"),
        'moderate': sum(1 for r in constraint_results if r.constraint_strength == "MODERATE"),
        'weak': sum(1 for r in constraint_results if r.constraint_strength == "WEAK"),
        'violated': sum(1 for r in constraint_results if r.constraint_strength == "VIOLATED"),
        'satisfied': sum(1 for r in constraint_results if r.constraint_satisfied),
        'significant': sum(1 for r in constraint_results if r.statistically_significant)
    }


def _categorize_features_by_strength(
    constraint_results: List[StatisticalConstraintResult]
) -> Dict[str, List[str]]:
    """
    Categorize features by constraint strength.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint validation results

    Returns
    -------
    Dict[str, List[str]]
        Feature names grouped by strength category
    """
    return {
        'violated': [r.feature_name for r in constraint_results if r.constraint_strength == "VIOLATED"],
        'weak': [r.feature_name for r in constraint_results if r.constraint_strength == "WEAK"],
        'strong': [r.feature_name for r in constraint_results if r.constraint_strength == "STRONG"]
    }


# =============================================================================
# COMPLIANCE ASSESSMENT
# =============================================================================


def _determine_compliance_level(counts: Dict[str, int]) -> Tuple[str, str, bool, str]:
    """
    Determine compliance level based on constraint outcome counts.

    Parameters
    ----------
    counts : Dict[str, int]
        Constraint outcome counts

    Returns
    -------
    Tuple[str, str, bool, str]
        (overall_compliance, confidence_level, production_readiness, recommendation)
    """
    total = counts['total']
    if counts['violated'] > 0:
        return (
            "POOR", "LOW", False,
            f"{counts['violated']} constraint(s) violated - economic theory not supported"
        )
    elif counts['strong'] >= total * 0.7:
        return (
            "EXCELLENT", "HIGH", True,
            "Strong statistical support for all major economic relationships"
        )
    elif counts['moderate'] + counts['strong'] >= total * 0.6:
        return (
            "GOOD", "MODERATE", True,
            "Most economic relationships supported with reasonable confidence"
        )
    elif counts['weak'] + counts['moderate'] + counts['strong'] >= total * 0.5:
        return (
            "ACCEPTABLE", "MODERATE", True,
            "Economic relationships have mixed statistical support"
        )
    else:
        return (
            "CONCERNING", "LOW", False,
            "Insufficient statistical evidence for economic relationships"
        )


def _assess_overall_constraint_compliance(
    constraint_results: List[StatisticalConstraintResult]
) -> Dict[str, Any]:
    """
    Assess overall model compliance with economic constraints.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Individual constraint validation results

    Returns
    -------
    Dict[str, Any]
        Overall compliance assessment
    """
    if not constraint_results:
        return {'no_constraints_tested': True}

    counts = _count_constraint_outcomes(constraint_results)
    compliance, confidence, ready, recommendation = _determine_compliance_level(counts)

    return {
        'overall_compliance': compliance,
        'confidence_level': confidence,
        'production_readiness': ready,
        'primary_recommendation': recommendation,
        'compliance_statistics': {
            'total_constraints': counts['total'],
            'satisfied_constraints': counts['satisfied'],
            'compliance_rate': counts['satisfied'] / counts['total'],
            'significance_rate': counts['significant'] / counts['total'],
            'strength_distribution': {
                'strong': counts['strong'],
                'moderate': counts['moderate'],
                'weak': counts['weak'],
                'violated': counts['violated']
            }
        }
    }


# =============================================================================
# METHODOLOGY COMPARISON
# =============================================================================


def _compute_methodology_summary(
    constraint_results: List[StatisticalConstraintResult]
) -> Dict[str, Any]:
    """
    Compute summary statistics comparing hard threshold and statistical methodologies.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint results to summarize

    Returns
    -------
    Dict[str, Any]
        Summary statistics dictionary
    """
    hard_passes = sum(1 for r in constraint_results if r.hard_threshold_comparison['hard_threshold_passes'])
    stat_passes = sum(1 for r in constraint_results if r.hard_threshold_comparison['statistical_approach_passes'])
    agrees = sum(1 for r in constraint_results if r.hard_threshold_comparison['methods_agree'])
    total = len(constraint_results)

    return {
        'hard_threshold_passes': hard_passes,
        'statistical_approach_passes': stat_passes,
        'agreement_rate': agrees / total,
        'total_disagreements': total - agrees
    }


def _identify_method_advantages(
    constraint_results: List[StatisticalConstraintResult]
) -> Tuple[List[str], List[str]]:
    """
    Identify advantages of each constraint methodology.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint results to analyze

    Returns
    -------
    Tuple[List[str], List[str]]
        (statistical_advantages, hard_threshold_advantages)
    """
    statistical_advantages = []
    hard_threshold_advantages = []

    for result in constraint_results:
        hard_passes = result.hard_threshold_comparison['hard_threshold_passes']
        stat_passes = result.hard_threshold_comparison['statistical_approach_passes']

        if hard_passes and not stat_passes:
            statistical_advantages.append(f"{result.feature_name}: Rejected non-significant effect")

    return statistical_advantages, hard_threshold_advantages


def _build_disagreement_analysis(
    constraint_results: List[StatisticalConstraintResult]
) -> List[Dict[str, Any]]:
    """
    Build detailed analysis of methodology disagreements.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint results to analyze

    Returns
    -------
    List[Dict[str, Any]]
        List of disagreement analysis entries
    """
    disagreements = [r for r in constraint_results if not r.hard_threshold_comparison['methods_agree']]

    return [
        {
            'feature': r.feature_name,
            'hard_threshold_result': r.hard_threshold_comparison['hard_threshold_passes'],
            'statistical_result': r.constraint_satisfied,
            'reason_for_disagreement': (
                "Statistical method considers uncertainty" if not r.statistically_significant
                else "Different threshold criteria"
            )
        }
        for r in disagreements
    ]


def _compare_constraint_methodologies(
    constraint_results: List[StatisticalConstraintResult],
    constraint_specifications: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare statistical vs hard threshold constraint methodologies.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Statistical constraint results
    constraint_specifications : Dict[str, Dict[str, Any]]
        Original constraint specifications (unused, kept for API compatibility)

    Returns
    -------
    Dict[str, Any]
        Methodology comparison analysis
    """
    if not constraint_results:
        return {'no_results_for_comparison': True}

    summary = _compute_methodology_summary(constraint_results)
    statistical_advantages, hard_threshold_advantages = _identify_method_advantages(constraint_results)
    disagreement_analysis = _build_disagreement_analysis(constraint_results)

    recommendation = (
        "Statistical approach recommended - incorporates coefficient uncertainty"
        if statistical_advantages
        else "Both methods show similar results - statistical approach still preferred for rigor"
    )

    return {
        'methodology_comparison_summary': summary,
        'statistical_method_advantages': statistical_advantages,
        'hard_threshold_advantages': hard_threshold_advantages,
        'disagreement_analysis': disagreement_analysis,
        'methodology_recommendation': recommendation
    }


# =============================================================================
# RECOMMENDATION GENERATION
# =============================================================================


def _build_feature_recommendations(categories: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build feature-specific recommendations based on strength categories.

    Parameters
    ----------
    categories : Dict[str, List[str]]
        Feature names grouped by strength

    Returns
    -------
    Dict[str, str]
        Recommendations for each category with features
    """
    recommendations = {}

    if categories['violated']:
        recommendations['violations'] = (
            f"Investigate features with violated constraints: {', '.join(categories['violated'])}. "
            "Consider data quality issues, model specification, or revised economic theory."
        )

    if categories['weak']:
        recommendations['weak_relationships'] = (
            f"Monitor features with weak statistical support: {', '.join(categories['weak'])}. "
            "Consider larger sample size or additional control variables."
        )

    if categories['strong']:
        recommendations['strong_relationships'] = (
            f"Leverage strongly supported relationships: {', '.join(categories['strong'])}. "
            "These features have robust statistical and economic justification."
        )

    return recommendations


def _generate_constraint_recommendations(
    constraint_results: List[StatisticalConstraintResult],
    overall_assessment: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate actionable business recommendations based on constraint analysis.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint validation results
    overall_assessment : Dict[str, Any]
        Overall compliance assessment

    Returns
    -------
    Dict[str, str]
        Business recommendations by category
    """
    recommendations = {
        'primary': overall_assessment.get('primary_recommendation', 'Assessment incomplete')
    }

    categories = _categorize_features_by_strength(constraint_results)
    recommendations.update(_build_feature_recommendations(categories))

    recommendations['production'] = (
        "Model passes economic constraint validation and is suitable for production deployment."
        if overall_assessment.get('production_readiness', False)
        else "Model requires constraint violation resolution before production deployment."
    )

    recommendations['methodology'] = (
        "Statistical constraint validation provides more reliable assessment than hard thresholds. "
        "Continue using confidence interval-based validation for future models."
    )

    return recommendations


# =============================================================================
# POWER ANALYSIS
# =============================================================================


def _compute_average_precision(
    constraint_results: List[StatisticalConstraintResult]
) -> float:
    """
    Compute average coefficient precision across constraints.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint results with standard errors

    Returns
    -------
    float
        Average precision (inverse of standard error)
    """
    return np.mean([
        1 / result.standard_error if result.standard_error > 0 else 0
        for result in constraint_results
    ])


def _estimate_power_metrics(
    n_observations: int,
    n_parameters: int
) -> Tuple[float, float, float]:
    """
    Estimate power-related metrics based on sample size.

    Parameters
    ----------
    n_observations : int
        Sample size
    n_parameters : int
        Number of parameters in model

    Returns
    -------
    Tuple[float, float, float]
        (t_critical, min_detectable_effect, estimated_power)
    """
    from scipy.stats import t
    df = max(1, n_observations - n_parameters - 1)
    t_critical = t.ppf(0.975, df=df)
    min_detectable = t_critical / np.sqrt(n_observations) if n_observations > 0 else 0

    medium_effect_size = 0.5
    power = min(1.0, max(0.0, (medium_effect_size * np.sqrt(n_observations) - t_critical) / 3.0))

    return t_critical, min_detectable, power


def _interpret_power(estimated_power: float) -> Tuple[str, str]:
    """
    Interpret power level and generate recommendation.

    Parameters
    ----------
    estimated_power : float
        Estimated statistical power

    Returns
    -------
    Tuple[str, str]
        (interpretation, recommendation)
    """
    if estimated_power > 0.8:
        interpretation = "HIGH"
    elif estimated_power > 0.6:
        interpretation = "MODERATE"
    else:
        interpretation = "LOW"

    recommendation = (
        "Adequate power for constraint detection"
        if estimated_power > 0.7
        else "Consider increasing sample size for better constraint detection power"
    )

    return interpretation, recommendation


def _analyze_constraint_detection_power(
    constraint_results: List[StatisticalConstraintResult],
    n_observations: int
) -> Dict[str, float]:
    """
    Analyze statistical power for detecting constraint violations.

    Parameters
    ----------
    constraint_results : List[StatisticalConstraintResult]
        Constraint validation results
    n_observations : int
        Sample size for power calculation

    Returns
    -------
    Dict[str, float]
        Power analysis results
    """
    if not constraint_results:
        return {}

    avg_precision = _compute_average_precision(constraint_results)
    _, min_detectable, estimated_power = _estimate_power_metrics(
        n_observations, len(constraint_results)
    )
    interpretation, recommendation = _interpret_power(estimated_power)

    return {
        'average_precision': avg_precision,
        'minimum_detectable_effect': min_detectable,
        'estimated_power_medium_effect': estimated_power,
        'sample_size': n_observations,
        'power_interpretation': interpretation,
        'recommendation': recommendation
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Interpretation
    '_interpret_positive_constraint',
    '_interpret_negative_constraint',

    # Aggregation
    '_count_constraint_outcomes',
    '_categorize_features_by_strength',

    # Compliance
    '_determine_compliance_level',
    '_assess_overall_constraint_compliance',

    # Methodology
    '_compute_methodology_summary',
    '_identify_method_advantages',
    '_build_disagreement_analysis',
    '_compare_constraint_methodologies',

    # Recommendations
    '_build_feature_recommendations',
    '_generate_constraint_recommendations',

    # Power Analysis
    '_compute_average_precision',
    '_estimate_power_metrics',
    '_interpret_power',
    '_analyze_constraint_detection_power',
]
