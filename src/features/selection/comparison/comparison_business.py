"""
Comparison Business Analysis Module for Methodology Comparison.

This module contains all business impact analysis, recommendation generation,
and performance summary functions extracted from comparative_analysis.py
for maintainability.

Module Responsibilities:
- Business impact analysis (risk, confidence, operational, cost-benefit)
- Methodology adoption recommendations
- Performance comparison summaries
- Key findings generation

Used by: comparative_analysis.py (imports for orchestration)
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BUSINESS IMPACT ANALYSIS
# =============================================================================


def _analyze_business_impact(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any],
    performance_comparison: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze business impact of methodology enhancement. Returns impact analysis dict."""
    business_impact: Dict[str, Any] = {
        'risk_reduction': {},
        'decision_confidence': {},
        'operational_impact': {},
        'cost_benefit_analysis': {}
    }

    try:
        # Risk reduction analysis
        business_impact['risk_reduction'] = _analyze_risk_reduction(enhanced_results)

        # Decision confidence analysis
        business_impact['decision_confidence'] = _analyze_decision_confidence()

        # Operational impact
        business_impact['operational_impact'] = _analyze_operational_impact(
            enhanced_results, performance_comparison
        )

        # Cost-benefit analysis
        business_impact['cost_benefit_analysis'] = _analyze_cost_benefit()

    except Exception as e:
        logger.warning(f"Business impact analysis failed: {e}")
        business_impact['analysis_failed'] = True

    return business_impact


def _analyze_risk_reduction(enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze risk reduction from enhanced methodology.

    Single responsibility: Risk reduction analysis only.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, Any]
        Risk reduction analysis
    """
    baseline_risks = [
        "No generalization evidence - model may fail in production",
        "Multiple testing bias - likely selecting noise as signal",
        "Unknown assumption violations - invalid statistical inference"
    ]

    enhanced_mitigations: List[str] = []
    if enhanced_results.get('temporal_validation'):
        enhanced_mitigations.append("Out-of-sample validation provides generalization confidence")

    if enhanced_results.get('multiple_testing_correction'):
        enhanced_mitigations.append("Multiple testing correction controls false discovery rate")

    if enhanced_results.get('regression_diagnostics'):
        enhanced_mitigations.append("Comprehensive diagnostics validate model assumptions")

    return {
        'baseline_risks': baseline_risks,
        'enhanced_mitigations': enhanced_mitigations,
        'risk_reduction_level': 'HIGH' if len(enhanced_mitigations) >= 3 else 'MODERATE'
    }


def _analyze_decision_confidence() -> Dict[str, Any]:
    """
    Analyze decision confidence factors.

    Single responsibility: Decision confidence analysis only.

    Returns
    -------
    Dict[str, Any]
        Decision confidence analysis
    """
    baseline_confidence_factors = ["AIC-based selection"]
    enhanced_confidence_factors = [
        "Out-of-sample performance evidence",
        "Statistical significance validation",
        "Economic theory compliance",
        "Assumption validation"
    ]

    return {
        'baseline_confidence_factors': baseline_confidence_factors,
        'enhanced_confidence_factors': enhanced_confidence_factors,
        'confidence_improvement': 'SUBSTANTIAL'
    }


def _analyze_operational_impact(
    enhanced_results: Dict[str, Any],
    performance_comparison: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze operational impact of methodology enhancement.

    Single responsibility: Operational impact analysis only.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results
    performance_comparison : Dict[str, Any]
        Performance comparison results

    Returns
    -------
    Dict[str, Any]
        Operational impact analysis
    """
    exclusive_benefits = performance_comparison.get('enhanced_exclusive_benefits', {})

    expected_production_performance = (
        "30-40% lower than reported (literature expectation)"
        if not exclusive_benefits.get('out_of_sample_validation')
        else "Well-estimated through out-of-sample validation"
    )

    model_maintenance = (
        "Reactive - fix issues as they arise in production"
        if not enhanced_results.get('regression_diagnostics')
        else "Proactive - identify and address issues before deployment"
    )

    business_trust = (
        "LOW - lack of generalization evidence undermines confidence"
        if not enhanced_results.get('temporal_validation')
        else "HIGH - comprehensive validation builds stakeholder confidence"
    )

    return {
        'expected_production_performance': expected_production_performance,
        'model_maintenance': model_maintenance,
        'business_trust': business_trust
    }


def _analyze_cost_benefit() -> Dict[str, Any]:
    """
    Analyze cost-benefit of methodology enhancement.

    Single responsibility: Cost-benefit analysis only.

    Returns
    -------
    Dict[str, Any]
        Cost-benefit analysis
    """
    implementation_cost = "MODERATE"  # Additional statistical analysis
    validation_cost = "LOW"          # One-time setup
    ongoing_cost = "MINIMAL"         # Automated validation

    business_benefits = [
        "Reduced model failure risk",
        "Improved production performance prediction",
        "Enhanced stakeholder confidence",
        "Regulatory compliance readiness",
        "Faster model iteration cycles"
    ]

    return {
        'implementation_costs': {
            'statistical_analysis': implementation_cost,
            'validation_setup': validation_cost,
            'ongoing_maintenance': ongoing_cost
        },
        'business_benefits': business_benefits,
        'roi_assessment': 'POSITIVE - Benefits significantly outweigh costs',
        'payback_period': '1-2 model deployments'
    }


# =============================================================================
# RECOMMENDATION GENERATION
# =============================================================================


def _assemble_recommendations(
    statistical_validation_comparison: Dict[str, Any],
    production_readiness_comparison: Dict[str, Any],
    business_impact_analysis: Dict[str, Any]
) -> Dict[str, str]:
    """
    Assemble all recommendation categories.

    Parameters
    ----------
    statistical_validation_comparison : Dict[str, Any]
        Statistical validation comparison
    production_readiness_comparison : Dict[str, Any]
        Production readiness comparison
    business_impact_analysis : Dict[str, Any]
        Business impact analysis

    Returns
    -------
    Dict[str, str]
        All assembled recommendations
    """
    return {
        'primary': _generate_primary_recommendation(
            statistical_validation_comparison, production_readiness_comparison
        ),
        'implementation': _generate_implementation_recommendation(),
        'risk_management': _generate_risk_recommendation(business_impact_analysis),
        'business_case': _generate_business_recommendation(business_impact_analysis),
        'technical': _generate_technical_recommendation(),
        'stakeholder': _generate_stakeholder_recommendation()
    }


def _generate_methodology_recommendations(
    performance_comparison: Dict[str, Any],
    statistical_validation_comparison: Dict[str, Any],
    production_readiness_comparison: Dict[str, Any],
    business_impact_analysis: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate comprehensive methodology adoption recommendations.

    Orchestrates recommendation generation by delegating to helper functions.

    Parameters
    ----------
    performance_comparison : Dict[str, Any]
        Performance comparison results (unused, retained for interface compatibility)
    statistical_validation_comparison : Dict[str, Any]
        Statistical validation comparison
    production_readiness_comparison : Dict[str, Any]
        Production readiness comparison
    business_impact_analysis : Dict[str, Any]
        Business impact analysis

    Returns
    -------
    Dict[str, str]
        Comprehensive recommendations by category
    """
    try:
        return _assemble_recommendations(
            statistical_validation_comparison, production_readiness_comparison, business_impact_analysis
        )
    except Exception as e:
        logger.warning(f"Recommendation generation failed: {e}")
        return {'generation_failed': f"Error: {e}"}


def _generate_primary_recommendation(
    statistical_validation_comparison: Dict[str, Any],
    production_readiness_comparison: Dict[str, Any]
) -> str:
    """
    Generate primary methodology adoption recommendation.

    Single responsibility: Primary recommendation only.

    Parameters
    ----------
    statistical_validation_comparison : Dict[str, Any]
        Statistical validation comparison
    production_readiness_comparison : Dict[str, Any]
        Production readiness comparison

    Returns
    -------
    str
        Primary recommendation
    """
    rigor_improvement = statistical_validation_comparison.get(
        'statistical_rigor_assessment', {}
    ).get('rigor_improvement', 0)

    confidence_improvement = production_readiness_comparison.get(
        'deployment_confidence', {}
    ).get('confidence_improvement', 0)

    if rigor_improvement >= 40 and confidence_improvement >= 30:
        return (
            "STRONGLY RECOMMEND adopting enhanced methodology - "
            "substantial improvements in statistical rigor and production readiness"
        )
    elif rigor_improvement >= 20:
        return (
            "RECOMMEND adopting enhanced methodology - "
            "significant statistical improvements justify implementation effort"
        )
    else:
        return (
            "CONSIDER adopting enhanced methodology - "
            "moderate improvements may be valuable for critical applications"
        )


def _generate_implementation_recommendation() -> str:
    """
    Generate implementation strategy recommendation.

    Single responsibility: Implementation recommendation only.

    Returns
    -------
    str
        Implementation recommendation
    """
    return (
        "Implement in phases: (1) Temporal validation, (2) Multiple testing correction, "
        "(3) Block bootstrap and diagnostics. Validate each phase maintains mathematical equivalence."
    )


def _generate_risk_recommendation(business_impact_analysis: Dict[str, Any]) -> str:
    """
    Generate risk management recommendation.

    Single responsibility: Risk recommendation only.

    Parameters
    ----------
    business_impact_analysis : Dict[str, Any]
        Business impact analysis

    Returns
    -------
    str
        Risk management recommendation
    """
    baseline_risks = len(
        business_impact_analysis.get('risk_reduction', {}).get('baseline_risks', [])
    )
    enhanced_mitigations = len(
        business_impact_analysis.get('risk_reduction', {}).get('enhanced_mitigations', [])
    )

    if enhanced_mitigations >= baseline_risks * 0.8:
        return (
            "Enhanced methodology significantly reduces production deployment risks. "
            "Recommended for all critical model deployments."
        )
    else:
        return (
            "Enhanced methodology provides moderate risk reduction. "
            "Consider for high-stakes applications."
        )


def _generate_business_recommendation(business_impact_analysis: Dict[str, Any]) -> str:
    """
    Generate business case recommendation.

    Single responsibility: Business recommendation only.

    Parameters
    ----------
    business_impact_analysis : Dict[str, Any]
        Business impact analysis

    Returns
    -------
    str
        Business case recommendation
    """
    roi_assessment = business_impact_analysis.get(
        'cost_benefit_analysis', {}
    ).get('roi_assessment', '')

    if 'POSITIVE' in roi_assessment:
        return (
            "Strong business case for adoption - benefits significantly outweigh costs. "
            "Expected payback within 1-2 model deployments."
        )
    else:
        return (
            "Moderate business case - evaluate based on specific use case requirements."
        )


def _generate_technical_recommendation() -> str:
    """
    Generate technical considerations recommendation.

    Single responsibility: Technical recommendation only.

    Returns
    -------
    str
        Technical recommendation
    """
    return (
        "Enhanced methodology requires additional statistical libraries and computational resources. "
        "Ensure team has time series analysis and bootstrap methodology expertise."
    )


def _generate_stakeholder_recommendation() -> str:
    """
    Generate stakeholder communication recommendation.

    Single responsibility: Stakeholder recommendation only.

    Returns
    -------
    str
        Stakeholder recommendation
    """
    return (
        "Emphasize improved confidence in model performance and reduced production risks. "
        "Highlight statistical rigor improvements and regulatory compliance benefits."
    )


# =============================================================================
# PERFORMANCE SUMMARY
# =============================================================================


def _calculate_improvement_counts(
    improvements: Dict[str, Any],
    exclusive_benefits: Dict[str, bool]
) -> Tuple[int, int]:
    """
    Calculate counts of positive improvements and exclusive benefits.

    Parameters
    ----------
    improvements : Dict[str, Any]
        Metric improvements
    exclusive_benefits : Dict[str, bool]
        Enhanced methodology exclusive benefits

    Returns
    -------
    Tuple[int, int]
        (positive_improvements_count, exclusive_benefits_count)
    """
    positive_improvements = sum(
        1 for imp in improvements.values()
        if imp.get('improved', False)
    )
    exclusive_count = sum(1 for benefit in exclusive_benefits.values() if benefit)
    return positive_improvements, exclusive_count


def _determine_overall_assessment(
    positive_improvements: int,
    exclusive_count: int,
    total_improvements: int
) -> Tuple[str, str]:
    """
    Determine overall assessment and performance advantage.

    Parameters
    ----------
    positive_improvements : int
        Count of positive metric improvements
    exclusive_count : int
        Count of exclusive benefits
    total_improvements : int
        Total number of improvements evaluated

    Returns
    -------
    Tuple[str, str]
        (overall_assessment, performance_advantage)
    """
    if exclusive_count >= 2:
        return 'ENHANCED METHODOLOGY STRONGLY PREFERRED', 'SUBSTANTIAL'
    elif total_improvements > 0 and positive_improvements >= total_improvements * 0.6:
        return 'ENHANCED METHODOLOGY PREFERRED', 'MODERATE'
    else:
        return 'MIXED RESULTS', 'MINIMAL'


def _generate_key_findings(
    exclusive_benefits: Dict[str, bool],
    enhanced_performance: Dict[str, float],
    positive_improvements: int
) -> List[str]:
    """
    Generate key findings from performance comparison.

    Parameters
    ----------
    exclusive_benefits : Dict[str, bool]
        Enhanced methodology exclusive benefits
    enhanced_performance : Dict[str, float]
        Enhanced methodology performance
    positive_improvements : int
        Count of positive improvements

    Returns
    -------
    List[str]
        List of key findings
    """
    key_findings: List[str] = []

    if exclusive_benefits.get('out_of_sample_validation'):
        key_findings.append("Enhanced methodology provides crucial out-of-sample validation")

    if exclusive_benefits.get('generalization_assessment'):
        gap = enhanced_performance.get('generalization_gap', np.nan)
        if not np.isnan(gap):
            key_findings.append(f"Generalization gap quantified: {gap:.3f} R² points")

    if positive_improvements >= 1:
        key_findings.append(f"{positive_improvements} performance metrics improved")

    return key_findings


def _summarize_performance_comparison(
    improvements: Dict[str, Any],
    exclusive_benefits: Dict[str, bool],
    enhanced_performance: Dict[str, float]
) -> Dict[str, Any]:
    """
    Summarize performance comparison results.

    Orchestrates summary generation by delegating to helper functions.

    Parameters
    ----------
    improvements : Dict[str, Any]
        Metric improvements
    exclusive_benefits : Dict[str, bool]
        Enhanced methodology exclusive benefits
    enhanced_performance : Dict[str, float]
        Enhanced methodology performance

    Returns
    -------
    Dict[str, Any]
        Performance comparison summary
    """
    summary: Dict[str, Any] = {
        'overall_assessment': 'UNKNOWN',
        'key_findings': [],
        'performance_advantage': 'UNKNOWN'
    }

    try:
        positive_improvements, exclusive_count = _calculate_improvement_counts(
            improvements, exclusive_benefits
        )
        assessment, advantage = _determine_overall_assessment(
            positive_improvements, exclusive_count, len(improvements)
        )
        summary['overall_assessment'] = assessment
        summary['performance_advantage'] = advantage
        summary['key_findings'] = _generate_key_findings(
            exclusive_benefits, enhanced_performance, positive_improvements
        )
    except Exception as e:
        logger.warning(f"Performance summary failed: {e}")
        summary['summary_failed'] = True

    return summary
