"""
Comparison Metrics Module for Methodology Comparison.

This module contains all performance metric extraction, model selection comparison,
statistical validation comparison, and production readiness functions extracted
from comparative_analysis.py for maintainability.

Module Responsibilities:
- Performance metric extraction (baseline and enhanced)
- Model selection comparison and consistency analysis
- Statistical validation comparison and rigor assessment
- Production readiness comparison and confidence scoring

Used by: comparative_analysis.py (imports for orchestration)
"""

import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE METRIC EXTRACTION
# =============================================================================


def _extract_baseline_performance(baseline_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract performance metrics from baseline results.

    Single responsibility: Baseline performance extraction only.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results

    Returns
    -------
    Dict[str, float]
        Baseline performance metrics
    """
    performance: Dict[str, float] = {}

    # Extract AIC if available
    if 'valid_results_sorted' in baseline_results:
        valid_results = baseline_results['valid_results_sorted']
        if len(valid_results) > 0:
            best_model = valid_results.iloc[0]
            performance['aic'] = float(best_model.get('aic', np.nan))
            performance['r_squared'] = float(best_model.get('r_squared', np.nan))

    # Extract from selected model if available
    if 'selected_model' in baseline_results:
        selected = baseline_results['selected_model']
        if 'aic' not in performance or np.isnan(performance.get('aic', np.nan)):
            performance['aic'] = float(selected.get('aic', np.nan))
        if 'r_squared' not in performance or np.isnan(performance.get('r_squared', np.nan)):
            performance['r_squared'] = float(selected.get('r_squared', np.nan))

    return performance


def _extract_enhanced_performance(enhanced_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract performance metrics from enhanced results.

    Single responsibility: Enhanced performance extraction only.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, float]
        Enhanced performance metrics including validation metrics
    """
    performance: Dict[str, float] = {}

    # Extract base metrics
    if 'valid_results_sorted' in enhanced_results:
        valid_results = enhanced_results['valid_results_sorted']
        if len(valid_results) > 0:
            best_model = valid_results.iloc[0]
            performance['aic'] = float(best_model.get('aic', np.nan))
            performance['r_squared'] = float(best_model.get('r_squared', np.nan))

    # Extract from selected model
    if 'selected_model' in enhanced_results:
        selected = enhanced_results['selected_model']
        if 'aic' not in performance or np.isnan(performance.get('aic', np.nan)):
            performance['aic'] = float(selected.get('aic', np.nan))
        if 'r_squared' not in performance or np.isnan(performance.get('r_squared', np.nan)):
            performance['r_squared'] = float(selected.get('r_squared', np.nan))

    # Extract temporal validation metrics (enhanced-only)
    if 'temporal_validation' in enhanced_results:
        temporal = enhanced_results['temporal_validation']
        if 'test_performance' in temporal:
            test_perf = temporal['test_performance']
            performance['test_r_squared'] = float(test_perf.get('r_squared', np.nan))
            performance['test_mse'] = float(test_perf.get('mse', np.nan))

        # Calculate generalization gap
        if 'train_performance' in temporal and 'test_performance' in temporal:
            train_r2 = temporal['train_performance'].get('r_squared', np.nan)
            test_r2 = temporal['test_performance'].get('r_squared', np.nan)
            if not np.isnan(train_r2) and not np.isnan(test_r2):
                performance['generalization_gap'] = float(train_r2 - test_r2)

    return performance


def _compute_metric_improvements(
    baseline_performance: Dict[str, float],
    enhanced_performance: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute improvements between baseline and enhanced metrics.

    Single responsibility: Metric improvement computation only.

    Parameters
    ----------
    baseline_performance : Dict[str, float]
        Baseline performance metrics
    enhanced_performance : Dict[str, float]
        Enhanced performance metrics

    Returns
    -------
    Dict[str, Any]
        Metric-by-metric improvement analysis
    """
    improvements: Dict[str, Any] = {}

    # AIC comparison (lower is better)
    if 'aic' in baseline_performance and 'aic' in enhanced_performance:
        baseline_aic = baseline_performance['aic']
        enhanced_aic = enhanced_performance['aic']
        if not np.isnan(baseline_aic) and not np.isnan(enhanced_aic):
            improvements['aic'] = {
                'baseline': baseline_aic,
                'enhanced': enhanced_aic,
                'difference': enhanced_aic - baseline_aic,
                'improved': enhanced_aic < baseline_aic
            }

    # R-squared comparison (higher is better)
    if 'r_squared' in baseline_performance and 'r_squared' in enhanced_performance:
        baseline_r2 = baseline_performance['r_squared']
        enhanced_r2 = enhanced_performance['r_squared']
        if not np.isnan(baseline_r2) and not np.isnan(enhanced_r2):
            improvements['r_squared'] = {
                'baseline': baseline_r2,
                'enhanced': enhanced_r2,
                'difference': enhanced_r2 - baseline_r2,
                'improved': enhanced_r2 > baseline_r2
            }

    return improvements


def _identify_enhanced_exclusive_benefits(enhanced_results: Dict[str, Any]) -> Dict[str, bool]:
    """
    Identify benefits exclusive to enhanced methodology.

    Single responsibility: Exclusive benefit identification only.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, bool]
        Dictionary of exclusive benefits with boolean indicators
    """
    return {
        'out_of_sample_validation': 'temporal_validation' in enhanced_results,
        'multiple_testing_correction': 'multiple_testing_correction' in enhanced_results,
        'block_bootstrap': 'block_bootstrap' in enhanced_results,
        'regression_diagnostics': 'regression_diagnostics' in enhanced_results,
        'statistical_constraints': 'statistical_constraints' in enhanced_results,
        'generalization_assessment': 'generalization_gap' in enhanced_results.get('temporal_validation', {})
    }


def _compare_performance_metrics(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare performance metrics between methodologies.

    Orchestrates performance comparison by delegating to helper functions.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, Any]
        Comprehensive performance comparison
    """
    performance_comparison: Dict[str, Any] = {
        'baseline_performance': {},
        'enhanced_performance': {},
        'improvements': {},
        'enhanced_exclusive_benefits': {}
    }

    try:
        baseline_perf = _extract_baseline_performance(baseline_results)
        enhanced_perf = _extract_enhanced_performance(enhanced_results)

        performance_comparison['baseline_performance'] = baseline_perf
        performance_comparison['enhanced_performance'] = enhanced_perf
        performance_comparison['improvements'] = _compute_metric_improvements(
            baseline_perf, enhanced_perf
        )
        performance_comparison['enhanced_exclusive_benefits'] = _identify_enhanced_exclusive_benefits(
            enhanced_results
        )

    except Exception as e:
        logger.warning(f"Performance metric comparison failed: {e}")
        performance_comparison['comparison_failed'] = True

    return performance_comparison


# =============================================================================
# MODEL SELECTION COMPARISON
# =============================================================================


def _extract_baseline_selection(baseline_results: Dict[str, Any]) -> tuple:
    """
    Extract baseline model selection details.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results

    Returns
    -------
    tuple
        (features_string, selection_dict)
    """
    features = "Unknown"
    selection: Dict[str, Any] = {'method': 'AIC minimization', 'criteria': 'In-sample AIC only'}

    if 'selected_model' in baseline_results:
        selected = baseline_results['selected_model']
        features = selected.get('features', 'Unknown')
        selection['selected_features'] = features
        selection['aic'] = selected.get('aic', np.nan)
        selection['n_features'] = selected.get('n_features', 0)

    return features, selection


def _extract_enhanced_selection(enhanced_results: Dict[str, Any]) -> tuple:
    """
    Extract enhanced model selection details.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    tuple
        (features_string, selection_dict)
    """
    features = "Unknown"
    selection: Dict[str, Any] = {
        'method': 'Statistical validation',
        'criteria': 'Out-of-sample + constraints + diagnostics'
    }

    if 'selected_model' in enhanced_results:
        selected = enhanced_results['selected_model']
        features = selected.get('features', 'Unknown')
        selection['selected_features'] = features
        selection['aic'] = selected.get('aic', np.nan)
        selection['n_features'] = selected.get('n_features', 0)

    if 'temporal_validation' in enhanced_results:
        selection['validation_method'] = 'Temporal train/test split'

    return features, selection


def _analyze_selection_consistency(
    baseline_features: str,
    enhanced_features: str
) -> Dict[str, Any]:
    """
    Analyze consistency between baseline and enhanced selections.

    Parameters
    ----------
    baseline_features : str
        Baseline selected features
    enhanced_features : str
        Enhanced selected features

    Returns
    -------
    Dict[str, Any]
        Consistency analysis
    """
    feature_overlap = _calculate_feature_overlap(baseline_features, enhanced_features)

    if feature_overlap >= 0.8:
        consistency_level = 'HIGH'
        interpretation = 'Enhanced validation confirms baseline selection'
    elif feature_overlap >= 0.5:
        consistency_level = 'MODERATE'
        interpretation = 'Partial overlap - enhanced methodology refines selection'
    else:
        consistency_level = 'LOW'
        interpretation = 'Significant divergence - validation changes selection substantially'

    return {
        'feature_overlap': feature_overlap,
        'consistency_level': consistency_level,
        'interpretation': interpretation,
        'same_model_selected': baseline_features == enhanced_features
    }


def _get_selection_criteria_comparison() -> Dict[str, Any]:
    """
    Get comparison of selection criteria between methodologies.

    Returns
    -------
    Dict[str, Any]
        Selection criteria comparison
    """
    return {
        'baseline_criteria': "AIC minimization (in-sample)",
        'enhanced_criteria': "Statistical validation with out-of-sample evidence",
        'enhanced_advantages': [
            "Out-of-sample validation prevents overfitting",
            "Multiple testing correction controls false discoveries",
            "Statistical constraint validation ensures economic validity",
            "Regression diagnostics validate model assumptions"
        ]
    }


def _compare_model_selection(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare model selection consistency between methodologies.

    Orchestrates model selection comparison by delegating to helper functions.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline results
    enhanced_results : Dict[str, Any]
        Enhanced results

    Returns
    -------
    Dict[str, Any]
        Model selection comparison analysis
    """
    model_comparison: Dict[str, Any] = {
        'baseline_selection': {},
        'enhanced_selection': {},
        'consistency_analysis': {},
        'selection_criteria_comparison': {}
    }

    try:
        baseline_features, baseline_selection = _extract_baseline_selection(baseline_results)
        enhanced_features, enhanced_selection = _extract_enhanced_selection(enhanced_results)

        model_comparison['baseline_selection'] = baseline_selection
        model_comparison['enhanced_selection'] = enhanced_selection
        model_comparison['consistency_analysis'] = _analyze_selection_consistency(
            baseline_features, enhanced_features
        )
        model_comparison['selection_criteria_comparison'] = _get_selection_criteria_comparison()

    except Exception as e:
        logger.warning(f"Model selection comparison failed: {e}")
        model_comparison['comparison_failed'] = True

    return model_comparison


def _calculate_feature_overlap(baseline_features: str, enhanced_features: str) -> float:
    """
    Calculate feature overlap between two feature sets.

    Single responsibility: Feature overlap calculation only.

    Parameters
    ----------
    baseline_features : str
        Baseline feature string (e.g., "feat1 + feat2")
    enhanced_features : str
        Enhanced feature string

    Returns
    -------
    float
        Overlap percentage (0.0 to 1.0)
    """
    try:
        # Parse feature sets
        baseline_set = set(feat.strip() for feat in baseline_features.split('+'))
        enhanced_set = set(feat.strip() for feat in enhanced_features.split('+'))

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(baseline_set.intersection(enhanced_set))
        union = len(baseline_set.union(enhanced_set))

        return intersection / union if union > 0 else 0.0

    except (AttributeError, TypeError) as e:
        # Input not a string or split() failed
        logger.warning(f"Feature overlap calculation failed: {e}. Inputs: baseline='{baseline_features}', enhanced='{enhanced_features}'")
        return 0.0


# =============================================================================
# STATISTICAL VALIDATION COMPARISON
# =============================================================================


def _get_default_validation_comparison() -> Dict[str, Any]:
    """Return default validation comparison structure."""
    return {
        'baseline_validation': {},
        'enhanced_validation': {},
        'validation_improvements': [],
        'statistical_rigor_assessment': {}
    }


def _compare_statistical_validation(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare statistical validation methodologies.

    Orchestrates validation comparison by delegating to helper functions.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline results
    enhanced_results : Dict[str, Any]
        Enhanced results

    Returns
    -------
    Dict[str, Any]
        Statistical validation comparison
    """
    comparison = _get_default_validation_comparison()
    try:
        comparison['baseline_validation'] = _extract_baseline_validation(baseline_results)
        comparison['enhanced_validation'] = _extract_enhanced_validation(enhanced_results)
        comparison['validation_improvements'] = _identify_validation_improvements(
            baseline_results, enhanced_results
        )
        comparison['statistical_rigor_assessment'] = _assess_statistical_rigor(
            baseline_results, enhanced_results
        )
    except Exception as e:
        logger.warning(f"Statistical validation comparison failed: {e}")
        comparison['comparison_failed'] = True
    return comparison


def _extract_baseline_validation(baseline_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract baseline validation characteristics.

    Single responsibility: Baseline validation extraction only.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results

    Returns
    -------
    Dict[str, Any]
        Baseline validation characteristics
    """
    validation_level = "BASIC"
    validation_methods: List[str] = []

    if baseline_results.get('bootstrap_results'):
        validation_methods.append("Standard Bootstrap (100 samples)")
        validation_level = "MODERATE"

    if baseline_results.get('constraint_validation'):
        validation_methods.append("Hard Threshold Economic Constraints")

    return {
        'validation_level': validation_level,
        'methods_used': validation_methods,
        'out_of_sample_validation': False,
        'multiple_testing_correction': False,
        'assumption_validation': False
    }


def _extract_enhanced_validation(enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract enhanced validation characteristics.

    Single responsibility: Enhanced validation extraction only.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, Any]
        Enhanced validation characteristics
    """
    validation_methods: List[str] = []
    validation_level = "BASIC"

    if enhanced_results.get('temporal_validation'):
        validation_methods.append("Temporal Train/Test Split")
        validation_level = "HIGH"

    if enhanced_results.get('multiple_testing_correction'):
        validation_methods.append("Multiple Testing Correction")
        validation_level = "HIGH"

    if enhanced_results.get('block_bootstrap'):
        validation_methods.append("Block Bootstrap (1000+ samples)")

    if enhanced_results.get('statistical_constraints'):
        validation_methods.append("Statistical Constraint Validation (CI-based)")

    if enhanced_results.get('regression_diagnostics'):
        validation_methods.append("Comprehensive Regression Diagnostics")
        validation_level = "RIGOROUS"

    return {
        'validation_level': validation_level,
        'methods_used': validation_methods,
        'out_of_sample_validation': True,
        'multiple_testing_correction': True,
        'assumption_validation': True
    }


def _identify_validation_improvements(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> List[str]:
    """
    Identify validation improvements from baseline to enhanced.

    Single responsibility: Improvement identification only.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    List[str]
        List of validation improvements
    """
    improvements: List[str] = []

    if not baseline_results.get('temporal_validation') and enhanced_results.get('temporal_validation'):
        improvements.append("Added out-of-sample validation - provides generalization evidence")

    if not baseline_results.get('multiple_testing_correction') and enhanced_results.get('multiple_testing_correction'):
        improvements.append("Added multiple testing correction - controls false discovery rate")

    if enhanced_results.get('block_bootstrap') and baseline_results.get('bootstrap_results'):
        improvements.append("Upgraded to block bootstrap - proper time series methodology")

    if enhanced_results.get('regression_diagnostics'):
        improvements.append("Added regression diagnostics - validates statistical assumptions")

    return improvements


def _assess_statistical_rigor(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess statistical rigor improvement between methodologies.

    Single responsibility: Rigor assessment only.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline methodology results
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, Any]
        Statistical rigor assessment
    """
    rigor_score_baseline = _calculate_statistical_rigor_score(baseline_results)
    rigor_score_enhanced = _calculate_statistical_rigor_score(enhanced_results)

    improvement_pct = (
        ((rigor_score_enhanced - rigor_score_baseline) / rigor_score_baseline * 100)
        if rigor_score_baseline > 0 else 100
    )

    return {
        'baseline_rigor_score': rigor_score_baseline,
        'enhanced_rigor_score': rigor_score_enhanced,
        'rigor_improvement': rigor_score_enhanced - rigor_score_baseline,
        'rigor_improvement_pct': improvement_pct
    }


def _calculate_statistical_rigor_score(results: Dict[str, Any]) -> int:
    """
    Calculate statistical rigor score (0-100).

    Single responsibility: Rigor scoring only.

    Parameters
    ----------
    results : Dict[str, Any]
        Methodology results

    Returns
    -------
    int
        Statistical rigor score (0-100)
    """
    score = 0

    # Base modeling (20 points)
    if results.get('selected_model'):
        score += 20

    # Out-of-sample validation (25 points)
    if results.get('temporal_validation'):
        score += 25

    # Multiple testing correction (20 points)
    if results.get('multiple_testing_correction'):
        score += 20

    # Bootstrap analysis (15 points)
    if results.get('bootstrap_results') or results.get('block_bootstrap'):
        score += 10  # Basic bootstrap
        if results.get('block_bootstrap'):
            score += 5   # Time series appropriate

    # Constraint validation (10 points)
    if results.get('constraint_validation') or results.get('statistical_constraints'):
        score += 5   # Basic constraints
        if results.get('statistical_constraints'):
            score += 5   # Statistical constraints

    # Regression diagnostics (10 points)
    if results.get('regression_diagnostics'):
        score += 10

    return min(score, 100)


# =============================================================================
# PRODUCTION READINESS COMPARISON
# =============================================================================


def _get_baseline_readiness() -> Dict[str, Any]:
    """
    Get baseline production readiness assessment (static, limited evidence).

    Returns
    -------
    Dict[str, Any]
        Baseline readiness assessment
    """
    return {
        'generalization_evidence': False,
        'assumption_validation': False,
        'statistical_significance': False,
        'confidence_level': 'LOW',
        'deployment_recommendation': 'HIGH RISK - No generalization evidence'
    }


def _extract_enhanced_readiness(enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract enhanced production readiness assessment.

    Parameters
    ----------
    enhanced_results : Dict[str, Any]
        Enhanced methodology results

    Returns
    -------
    Dict[str, Any]
        Enhanced readiness assessment
    """
    enhanced_temporal = enhanced_results.get('temporal_validation', {})
    enhanced_diagnostics = enhanced_results.get('regression_diagnostics', {})

    production_assessment = enhanced_temporal.get('production_assessment', {})
    recommendations = production_assessment.get('recommendations', ['Unknown'])
    deployment_rec = recommendations[0] if production_assessment else 'Assessment needed'

    return {
        'generalization_evidence': bool(enhanced_temporal.get('test_performance')),
        'assumption_validation': bool(enhanced_diagnostics.get('overall_assessment')),
        'statistical_significance': bool(enhanced_results.get('multiple_testing_correction')),
        'confidence_level': production_assessment.get('confidence_level', 'UNKNOWN'),
        'deployment_recommendation': deployment_rec
    }


def _identify_readiness_improvements(
    baseline_readiness: Dict[str, Any],
    enhanced_readiness: Dict[str, Any]
) -> List[str]:
    """
    Identify improvements in production readiness.

    Parameters
    ----------
    baseline_readiness : Dict[str, Any]
        Baseline readiness assessment
    enhanced_readiness : Dict[str, Any]
        Enhanced readiness assessment

    Returns
    -------
    List[str]
        List of improvement descriptions
    """
    improvements: List[str] = []

    if enhanced_readiness['generalization_evidence'] and not baseline_readiness['generalization_evidence']:
        improvements.append("Provides actual generalization evidence through out-of-sample validation")

    if enhanced_readiness['assumption_validation'] and not baseline_readiness['assumption_validation']:
        improvements.append("Validates regression assumptions through comprehensive diagnostics")

    if enhanced_readiness['statistical_significance'] and not baseline_readiness['statistical_significance']:
        improvements.append("Controls statistical significance through multiple testing correction")

    return improvements


def _assess_deployment_confidence(
    baseline_confidence_score: int,
    enhanced_confidence_score: int
) -> Dict[str, Any]:
    """
    Assess deployment confidence improvement.

    Parameters
    ----------
    baseline_confidence_score : int
        Baseline confidence score
    enhanced_confidence_score : int
        Enhanced confidence score

    Returns
    -------
    Dict[str, Any]
        Deployment confidence assessment
    """
    confidence_improvement = enhanced_confidence_score - baseline_confidence_score

    if confidence_improvement >= 40:
        interpretation = "SIGNIFICANT IMPROVEMENT"
    elif confidence_improvement >= 20:
        interpretation = "MODERATE IMPROVEMENT"
    else:
        interpretation = "MINIMAL IMPROVEMENT"

    return {
        'baseline_confidence_score': baseline_confidence_score,
        'enhanced_confidence_score': enhanced_confidence_score,
        'confidence_improvement': confidence_improvement,
        'confidence_interpretation': interpretation
    }


def _compare_production_readiness(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare production readiness between methodologies.

    Orchestrates production readiness comparison by delegating to helper functions.

    Parameters
    ----------
    baseline_results : Dict[str, Any]
        Baseline results
    enhanced_results : Dict[str, Any]
        Enhanced results

    Returns
    -------
    Dict[str, Any]
        Production readiness comparison
    """
    readiness_comparison: Dict[str, Any] = {
        'baseline_readiness': {},
        'enhanced_readiness': {},
        'readiness_improvements': [],
        'deployment_confidence': {}
    }

    try:
        baseline_readiness = _get_baseline_readiness()
        enhanced_readiness = _extract_enhanced_readiness(enhanced_results)

        readiness_comparison['baseline_readiness'] = baseline_readiness
        readiness_comparison['enhanced_readiness'] = enhanced_readiness

        readiness_comparison['readiness_improvements'] = _identify_readiness_improvements(
            baseline_readiness, enhanced_readiness
        )

        baseline_confidence_score = 30  # Low due to lack of validation
        enhanced_confidence_score = _calculate_confidence_score(enhanced_readiness)
        readiness_comparison['deployment_confidence'] = _assess_deployment_confidence(
            baseline_confidence_score, enhanced_confidence_score
        )

    except Exception as e:
        logger.warning(f"Production readiness comparison failed: {e}")
        readiness_comparison['comparison_failed'] = True

    return readiness_comparison


def _calculate_confidence_score(readiness: Dict[str, Any]) -> int:
    """
    Calculate deployment confidence score (0-100).

    Single responsibility: Confidence scoring only.

    Parameters
    ----------
    readiness : Dict[str, Any]
        Production readiness assessment

    Returns
    -------
    int
        Confidence score (0-100)
    """
    score = 30  # Base score

    if readiness.get('generalization_evidence'):
        score += 30  # Major factor

    if readiness.get('assumption_validation'):
        score += 20  # Important factor

    if readiness.get('statistical_significance'):
        score += 15  # Significance control

    confidence_level = readiness.get('confidence_level', 'LOW')
    if confidence_level == 'HIGH':
        score += 5
    elif confidence_level == 'MODERATE':
        score += 3

    return min(score, 100)
