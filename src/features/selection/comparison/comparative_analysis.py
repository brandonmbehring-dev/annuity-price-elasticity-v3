"""
Comparative Analysis Framework for Feature Selection Methodologies.

This module provides comprehensive comparison between baseline (current) and
enhanced (statistically rigorous) feature selection methodologies, enabling
stakeholders to understand the impact of statistical improvements and make
informed decisions about methodology adoption.

Key Functions:
- compare_methodologies: Side-by-side baseline vs enhanced comparison
- analyze_performance_differences: Statistical significance of improvements
- generate_methodology_recommendations: Business-oriented guidance
- create_comparison_visualizations: Comprehensive comparison dashboards

Comparative Analysis Features:
- Performance metric comparison (R², MAPE, generalization gaps)
- Model selection consistency analysis
- Statistical validation comparison
- Production readiness assessment comparison
- Business impact evaluation

Design Principles:
- Objective comparison without bias toward either methodology
- Statistical significance testing of performance differences
- Business-interpretable results and recommendations
- Comprehensive visualization support
- Integration with existing reporting framework

Module Architecture (Phase 6.2 Split):
- comparative_analysis.py: Orchestration + dataclass + validation (this file)
- comparison_metrics.py: Performance/selection/validation/readiness metrics
- comparison_business.py: Business impact analysis + recommendations
"""

import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Import metric comparison functions from comparison_metrics.py
from src.features.selection.comparison.comparison_metrics import (
    _compare_performance_metrics,
    _compare_model_selection,
    _compare_statistical_validation,
    _compare_production_readiness,
    _extract_baseline_performance,
    _extract_enhanced_performance,
    _compute_metric_improvements,
    _identify_enhanced_exclusive_benefits,
    _extract_baseline_selection,
    _extract_enhanced_selection,
    _analyze_selection_consistency,
    _get_selection_criteria_comparison,
    _calculate_feature_overlap,
    _get_default_validation_comparison,
    _extract_baseline_validation,
    _extract_enhanced_validation,
    _identify_validation_improvements,
    _assess_statistical_rigor,
    _calculate_statistical_rigor_score,
    _get_baseline_readiness,
    _extract_enhanced_readiness,
    _identify_readiness_improvements,
    _assess_deployment_confidence,
    _calculate_confidence_score,
)

# Import business analysis functions from comparison_business.py
from src.features.selection.comparison.comparison_business import (
    _analyze_business_impact,
    _analyze_risk_reduction,
    _analyze_decision_confidence,
    _analyze_operational_impact,
    _analyze_cost_benefit,
    _generate_methodology_recommendations,
    _assemble_recommendations,
    _generate_primary_recommendation,
    _generate_implementation_recommendation,
    _generate_risk_recommendation,
    _generate_business_recommendation,
    _generate_technical_recommendation,
    _generate_stakeholder_recommendation,
    _summarize_performance_comparison,
    _calculate_improvement_counts,
    _determine_overall_assessment,
    _generate_key_findings,
)


# =============================================================================
# DATACLASS DEFINITION
# =============================================================================


@dataclass
class MethodologyComparison:
    """
    Container for comprehensive methodology comparison results.

    Attributes
    ----------
    baseline_results : Dict[str, Any]
        Results from baseline (current) methodology
    enhanced_results : Dict[str, Any]
        Results from enhanced (rigorous) methodology
    performance_comparison : Dict[str, Any]
        Performance metric comparisons
    model_selection_comparison : Dict[str, Any]
        Model selection consistency analysis
    statistical_validation_comparison : Dict[str, Any]
        Statistical validation methodology comparison
    production_readiness_comparison : Dict[str, Any]
        Production deployment readiness comparison
    business_impact_analysis : Dict[str, Any]
        Business-oriented impact assessment
    recommendations : Dict[str, str]
        Methodology adoption recommendations
    """
    baseline_results: Dict[str, Any]
    enhanced_results: Dict[str, Any]
    performance_comparison: Dict[str, Any]
    model_selection_comparison: Dict[str, Any]
    statistical_validation_comparison: Dict[str, Any]
    production_readiness_comparison: Dict[str, Any]
    business_impact_analysis: Dict[str, Any]
    recommendations: Dict[str, str]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _validate_comparison_inputs(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> None:
    """
    Validate inputs for methodology comparison.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if baseline_results is None:
        raise ValueError("baseline_results cannot be None")
    if enhanced_results is None:
        raise ValueError("enhanced_results cannot be None")
    if not isinstance(baseline_results, dict):
        raise ValueError("baseline_results must be a dictionary")
    if not isinstance(enhanced_results, dict):
        raise ValueError("enhanced_results must be a dictionary")


def _get_default_comparison_metrics() -> List[str]:
    """Return default comparison metrics for methodology analysis."""
    return [
        'model_selection_consistency',
        'performance_metrics',
        'statistical_validation',
        'production_readiness',
        'business_impact'
    ]


# =============================================================================
# ORCHESTRATION FUNCTIONS
# =============================================================================


def _run_all_comparisons(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run all comparison analyses and return results tuple.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Tuple[Dict[str, Any], ...]
        (performance, model_selection, statistical_validation,
         production_readiness, business_impact)
    """
    performance = _compare_performance_metrics(baseline_results, enhanced_results)
    model_selection = _compare_model_selection(baseline_results, enhanced_results)
    statistical_validation = _compare_statistical_validation(baseline_results, enhanced_results)
    production_readiness = _compare_production_readiness(baseline_results, enhanced_results)
    business_impact = _analyze_business_impact(baseline_results, enhanced_results, performance)
    return performance, model_selection, statistical_validation, production_readiness, business_impact


def _build_methodology_comparison(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any],
    comparisons: Tuple[Dict[str, Any], ...],
    recommendations: Dict[str, str]
) -> MethodologyComparison:
    """
    Build MethodologyComparison dataclass from comparison results.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results
    comparisons : Tuple[Dict[str, Any], ...]
        Tuple of (performance, model_selection, statistical_validation,
        production_readiness, business_impact)
    recommendations : Dict[str, str]
        Generated recommendations

    Returns
    -------
    MethodologyComparison
        Assembled comparison result
    """
    performance, model_selection, statistical_validation, production_readiness, business_impact = comparisons
    return MethodologyComparison(
        baseline_results=baseline_results,
        enhanced_results=enhanced_results,
        performance_comparison=performance,
        model_selection_comparison=model_selection,
        statistical_validation_comparison=statistical_validation,
        production_readiness_comparison=production_readiness,
        business_impact_analysis=business_impact,
        recommendations=recommendations
    )


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================


def compare_methodologies(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any],
    comparison_metrics: List[str] = None
) -> MethodologyComparison:
    """Compare baseline vs enhanced feature selection methodologies."""
    try:
        logger.info("Starting comprehensive methodology comparison analysis")
        _validate_comparison_inputs(baseline_results, enhanced_results)

        # Run all comparison analyses
        comparisons = _run_all_comparisons(baseline_results, enhanced_results)
        performance, _, statistical_validation, production_readiness, business_impact = comparisons

        # Generate recommendations based on comparison results
        recommendations = _generate_methodology_recommendations(
            performance, statistical_validation, production_readiness, business_impact
        )

        logger.info("Methodology comparison completed successfully")
        return _build_methodology_comparison(
            baseline_results, enhanced_results, comparisons, recommendations
        )

    except Exception as e:
        raise ValueError(
            f"CRITICAL: Methodology comparison failed. "
            f"Business impact: Cannot assess improvement from statistical rigor. "
            f"Original error: {e}"
        ) from e


# =============================================================================
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

# These re-exports maintain backward compatibility for any code that imports
# private functions directly from comparative_analysis.py

__all__ = [
    # Main public API
    'compare_methodologies',
    'MethodologyComparison',

    # Orchestration helpers (internal use)
    '_validate_comparison_inputs',
    '_get_default_comparison_metrics',
    '_run_all_comparisons',
    '_build_methodology_comparison',

    # Re-exports from comparison_metrics.py
    '_compare_performance_metrics',
    '_compare_model_selection',
    '_compare_statistical_validation',
    '_compare_production_readiness',

    # Re-exports from comparison_business.py
    '_analyze_business_impact',
    '_generate_methodology_recommendations',
    '_summarize_performance_comparison',
]
