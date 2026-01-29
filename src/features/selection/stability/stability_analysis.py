"""
Stability Analysis Module for Feature Selection Pipeline.

This module provides atomic functions for bootstrap analysis coordination,
win rate analysis, information ratio analysis, and stability assessment
following CODING_STANDARDS.md Section 3.1 requirements.

Purpose: Decompose notebook_interface.py stability functions (400-500 lines)
Status: MANDATORY (decomposition of 2,274-line module)
Priority: HIGH (code organization and maintainability)

Key Functions:
- run_bootstrap_stability_analysis(): Main stability analysis coordination
- calculate_win_rates(): Win rate calculations
- analyze_information_ratios(): Information ratio analysis
- evaluate_feature_consistency(): Feature consistency evaluation
- generate_stability_metrics(): Stability metric generation
- validate_bootstrap_results(): Bootstrap result validation
- aggregate_stability_insights(): Insights aggregation
- format_stability_outputs(): Output formatting

Mathematical Equivalence: All functions maintain identical results to original
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings


# =============================================================================
# Helper Functions (Private)
# =============================================================================

def _validate_bootstrap_results_input(bootstrap_results: List[Any], operation: str) -> None:
    """Validate bootstrap results input for stability operations.

    Raises ValueError if bootstrap_results is empty with business context.
    """
    if not bootstrap_results:
        raise ValueError(
            f"CRITICAL: No bootstrap results for {operation}. "
            f"Business impact: Cannot complete {operation}. "
            "Required action: Complete bootstrap analysis first."
        )


def _classify_information_ratio(information_ratio: float) -> str:
    """Classify information ratio into performance category."""
    if information_ratio <= -0.5:
        return "Excellent"
    elif information_ratio <= -0.2:
        return "Good"
    elif information_ratio <= 0.2:
        return "Average"
    return "Poor"


def _calculate_benchmark_aic(bootstrap_results: List[Any]) -> float:
    """Calculate benchmark AIC as population median across all bootstrap AICs."""
    all_bootstrap_aics = []
    for result in bootstrap_results:
        all_bootstrap_aics.extend(result.bootstrap_aics)
    return np.median(all_bootstrap_aics)


def _compute_single_ir(result: Any, benchmark_aic: float, model_index: int) -> Dict[str, Any]:
    """Compute information ratio metrics for a single model."""
    model_aics = np.array(result.bootstrap_aics)
    mean_aic = np.mean(model_aics)
    std_aic = np.std(model_aics)

    information_ratio = (mean_aic - benchmark_aic) / std_aic if std_aic > 0 else 0.0

    return {
        'model_name': f"Model {model_index + 1}",
        'features': result.model_features,
        'information_ratio': information_ratio,
        'mean_aic': mean_aic,
        'aic_volatility': std_aic,
        'ir_assessment': _classify_information_ratio(information_ratio),
        'benchmark_aic': benchmark_aic
    }


def _analyze_feature_usage(bootstrap_results: List[Any]) -> Tuple[Dict[str, int], List[List[str]]]:
    """Analyze feature usage patterns across bootstrap results."""
    feature_usage = {}
    model_features = []

    for result in bootstrap_results:
        features = [f.strip() for f in result.model_features.split('+')]
        model_features.append(features)
        for feature in features:
            feature_usage[feature] = feature_usage.get(feature, 0) + 1

    return feature_usage, model_features


def _calculate_feature_consistency(feature_usage: Dict[str, int], total_models: int) -> Dict[str, Dict[str, Any]]:
    """Calculate consistency metrics for each feature."""
    feature_consistency = {}
    for feature, usage_count in feature_usage.items():
        consistency_pct = (usage_count / total_models) * 100
        stability_class = 'High' if consistency_pct >= 70 else 'Moderate' if consistency_pct >= 30 else 'Low'
        feature_consistency[feature] = {
            'usage_count': usage_count,
            'consistency_pct': consistency_pct,
            'stability_class': stability_class
        }
    return feature_consistency


def _build_consistency_summary(feature_consistency: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Build consistency summary and identify high-consistency features."""
    high_consistency = [f for f, info in feature_consistency.items() if info['consistency_pct'] >= 70]
    moderate_consistency = [f for f, info in feature_consistency.items() if 30 <= info['consistency_pct'] < 70]

    summary = {
        'total_unique_features': len(feature_consistency),
        'high_consistency_features': len(high_consistency),
        'moderate_consistency_features': len(moderate_consistency),
        'overall_stability': 'High' if len(high_consistency) >= 3 else 'Moderate'
    }
    return summary, high_consistency


def _calculate_aic_stability_stats(bootstrap_results: List[Any]) -> Dict[str, float]:
    """Calculate AIC stability statistics from bootstrap results."""
    aic_cvs = [result.aic_stability_coefficient for result in bootstrap_results]
    return {
        'mean_cv': np.mean(aic_cvs),
        'median_cv': np.median(aic_cvs),
        'min_cv': np.min(aic_cvs),
        'max_cv': np.max(aic_cvs),
        'std_cv': np.std(aic_cvs)
    }


def _calculate_r2_stability_stats(bootstrap_results: List[Any]) -> Dict[str, float]:
    """Calculate R-squared stability statistics from bootstrap results."""
    r2_cvs = [result.r2_stability_coefficient for result in bootstrap_results]
    return {
        'mean_cv': np.mean(r2_cvs),
        'median_cv': np.median(r2_cvs),
        'min_cv': np.min(r2_cvs),
        'max_cv': np.max(r2_cvs),
        'std_cv': np.std(r2_cvs)
    }


def _classify_overall_stability(bootstrap_results: List[Any]) -> str:
    """Classify overall stability based on assessment distribution."""
    assessments = [result.stability_assessment for result in bootstrap_results]
    stable_models = sum(1 for a in assessments if a == 'STABLE')
    moderate_models = sum(1 for a in assessments if a == 'MODERATE')
    total = len(bootstrap_results)

    if stable_models >= total * 0.6:
        return 'HIGH'
    elif (stable_models + moderate_models) >= total * 0.8:
        return 'MODERATE'
    return 'LOW'


def _collect_stability_distribution(bootstrap_results: List[Any]) -> Dict[str, int]:
    """Collect stability assessment distribution."""
    assessments = [result.stability_assessment for result in bootstrap_results]
    distribution = {}
    for assessment in set(assessments):
        distribution[assessment] = assessments.count(assessment)
    return distribution


def _validate_single_result(result: Any, model_index: int) -> List[str]:
    """Validate a single bootstrap result and return validation errors."""
    errors = []
    model_name = f"Model {model_index + 1}"
    required_attrs = ['model_features', 'bootstrap_aics', 'original_aic',
                      'stability_assessment', 'aic_stability_coefficient']

    for attr in required_attrs:
        if not hasattr(result, attr):
            errors.append(f"{model_name}: Missing attribute '{attr}'")

    if hasattr(result, 'bootstrap_aics'):
        if not result.bootstrap_aics or len(result.bootstrap_aics) == 0:
            errors.append(f"{model_name}: Empty bootstrap AIC values")
        elif any(np.isnan(aic) or np.isinf(aic) for aic in result.bootstrap_aics):
            errors.append(f"{model_name}: Invalid AIC values (NaN or Inf)")

    if hasattr(result, 'aic_stability_coefficient'):
        if np.isnan(result.aic_stability_coefficient) or result.aic_stability_coefficient < 0:
            errors.append(f"{model_name}: Invalid stability coefficient")

    return errors


def _build_aic_matrix_and_model_info(bootstrap_results: List[Any], n_samples: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build AIC matrix and model info for win rate calculation."""
    n_models = len(bootstrap_results)
    bootstrap_aic_matrix = np.zeros((n_samples, n_models))
    model_info = []

    for i, result in enumerate(bootstrap_results):
        bootstrap_aic_matrix[:, i] = result.bootstrap_aics
        model_info.append({
            'model_name': f"Model {i+1}",
            'features': result.model_features,
            'original_aic': result.original_aic
        })

    return bootstrap_aic_matrix, model_info


def _compute_win_counts(aic_matrix: np.ndarray, n_samples: int) -> np.ndarray:
    """Compute win counts from AIC matrix."""
    n_models = aic_matrix.shape[1]
    win_counts = np.zeros(n_models)
    for sample_idx in range(n_samples):
        winner_idx = np.argmin(aic_matrix[sample_idx, :])
        win_counts[winner_idx] += 1
    return win_counts


def _compile_win_rate_results(model_info: List[Dict[str, Any]], win_counts: np.ndarray,
                               bootstrap_results: List[Any], n_samples: int) -> List[Dict[str, Any]]:
    """Compile win rate results from computed counts."""
    win_rate_results = []
    for i in range(len(model_info)):
        win_rate_results.append({
            **model_info[i],
            'win_rate_pct': (win_counts[i] / n_samples) * 100,
            'win_count': int(win_counts[i]),
            'median_bootstrap_aic': np.median(bootstrap_results[i].bootstrap_aics)
        })
    win_rate_results.sort(key=lambda x: x['win_rate_pct'], reverse=True)
    return win_rate_results


def _build_executive_summary(win_rates: List[Dict[str, Any]], ir_results: List[Dict[str, Any]],
                              consistency_results: Dict[str, Any]) -> str:
    """Build executive summary string for stability insights."""
    top_win_rate = win_rates[0]['model_name'] if win_rates else 'Unknown'
    top_ir_model = ir_results[0]['model_name'] if ir_results else 'Unknown'
    win_pct = win_rates[0]['win_rate_pct'] if win_rates else 0.0
    ir_value = ir_results[0]['information_ratio'] if ir_results else 0.0
    overall_stability = consistency_results['consistency_summary']['overall_stability']

    return f"""
Stability Analysis Complete:
- Models Analyzed: {len(win_rates)}
- Top Win Rate: {top_win_rate} ({win_pct:.1f}%)
- Top Information Ratio: {top_ir_model} (IR: {ir_value:.3f})
- Overall Feature Consistency: {overall_stability}
"""


def _build_top_performers(win_rates: List[Dict[str, Any]], ir_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build top performers dictionary for insights."""
    top_win = win_rates[0]['model_name'] if win_rates else None
    top_ir = ir_results[0]['model_name'] if ir_results else None
    return {
        'win_rate_leader': win_rates[0] if win_rates else None,
        'ir_leader': ir_results[0] if ir_results else None,
        'consensus_winner': top_win if top_win == top_ir else 'Mixed'
    }


# =============================================================================
# Public API Functions
# =============================================================================


def run_bootstrap_stability_analysis(data: pd.DataFrame,
                                   valid_models: pd.DataFrame,
                                   config: Dict[str, Any],
                                   target_variable: str) -> List[Any]:
    """Main stability analysis coordination. Returns bootstrap results with uncertainty quantification."""
    if not config.get('enabled', False):
        print("Bootstrap stability analysis disabled - skipping")
        return []

    if valid_models.empty:
        raise ValueError(
            "CRITICAL: No valid models for stability analysis. "
            "Business impact: Cannot assess model reliability. "
            "Required action: Ensure constraint validation produces valid models."
        )

    from src.features.selection.engines.bootstrap_engine import run_bootstrap_stability

    models_to_analyze = min(config.get('models_to_analyze', 15), len(valid_models))
    n_samples = config.get('n_samples', 100)
    print(f"Bootstrap stability analysis: {models_to_analyze} models, {n_samples} samples each")

    bootstrap_results = run_bootstrap_stability(data, valid_models, config, target_variable)

    if not bootstrap_results:
        raise ValueError(
            "CRITICAL: Bootstrap stability analysis produced no results. "
            "Business impact: Model reliability assessment failed. "
            "Required action: Check bootstrap configuration and data quality."
        )

    return bootstrap_results


# TODO: Remove alias after v2.1 validation - added for backward compatibility
run_stability_analysis = run_bootstrap_stability_analysis


def calculate_win_rates(bootstrap_results: List[Any],
                       n_samples: int) -> List[Dict[str, Any]]:
    """Calculate win rates from bootstrap analysis results.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Win rate calculations with statistical analysis.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results with AIC distributions
    n_samples : int
        Number of bootstrap samples per model

    Returns
    -------
    List[Dict[str, Any]]
        Win rate results sorted by performance
    """
    _validate_bootstrap_results_input(bootstrap_results, "win rate calculation")

    aic_matrix, model_info = _build_aic_matrix_and_model_info(bootstrap_results, n_samples)
    win_counts = _compute_win_counts(aic_matrix, n_samples)

    return _compile_win_rate_results(model_info, win_counts, bootstrap_results, n_samples)


def analyze_information_ratios(bootstrap_results: List[Any]) -> List[Dict[str, Any]]:
    """Analyze information ratios for risk-adjusted model performance.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Information ratio analysis with risk adjustment.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results with AIC distributions

    Returns
    -------
    List[Dict[str, Any]]
        Information ratio analysis results sorted by IR (lower is better)
    """
    _validate_bootstrap_results_input(bootstrap_results, "information ratio analysis")

    benchmark_aic = _calculate_benchmark_aic(bootstrap_results)

    ir_results = [
        _compute_single_ir(result, benchmark_aic, i)
        for i, result in enumerate(bootstrap_results)
    ]

    ir_results.sort(key=lambda x: x['information_ratio'])
    return ir_results


def evaluate_feature_consistency(bootstrap_results: List[Any]) -> Dict[str, Any]:
    """Evaluate feature consistency across bootstrap samples.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Feature consistency evaluation with stability metrics.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results for consistency evaluation

    Returns
    -------
    Dict[str, Any]
        Feature consistency evaluation results
    """
    _validate_bootstrap_results_input(bootstrap_results, "consistency evaluation")

    feature_usage, model_features = _analyze_feature_usage(bootstrap_results)
    total_models = len(bootstrap_results)

    feature_consistency = _calculate_feature_consistency(feature_usage, total_models)
    consistency_summary, high_consistency_features = _build_consistency_summary(feature_consistency)

    return {
        'feature_consistency': feature_consistency,
        'consistency_summary': consistency_summary,
        'high_consistency_features': high_consistency_features,
        'model_features_list': model_features
    }


def generate_stability_metrics(bootstrap_results: List[Any]) -> Dict[str, Any]:
    """Generate comprehensive stability metrics from bootstrap analysis.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Stability metric generation with statistical summaries.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results for metric generation

    Returns
    -------
    Dict[str, Any]
        Comprehensive stability metrics
    """
    _validate_bootstrap_results_input(bootstrap_results, "stability metrics")

    samples_per_model = len(bootstrap_results[0].bootstrap_aics) if bootstrap_results else 0

    return {
        'models_analyzed': len(bootstrap_results),
        'bootstrap_samples_per_model': samples_per_model,
        'stability_distribution': _collect_stability_distribution(bootstrap_results),
        'aic_stability_stats': _calculate_aic_stability_stats(bootstrap_results),
        'r2_stability_stats': _calculate_r2_stability_stats(bootstrap_results),
        'overall_stability_assessment': _classify_overall_stability(bootstrap_results)
    }


def validate_bootstrap_results(bootstrap_results: List[Any]) -> Tuple[bool, List[str]]:
    """Validate bootstrap results for consistency and completeness.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Bootstrap result validation with comprehensive checks.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results to validate

    Returns
    -------
    Tuple[bool, List[str]]
        (validation_passed, validation_errors)
    """
    if not bootstrap_results:
        return False, ["No bootstrap results provided"]

    validation_errors = []

    for i, result in enumerate(bootstrap_results):
        validation_errors.extend(_validate_single_result(result, i))

    # Cross-validation: check consistent sample sizes
    sample_sizes = set(len(result.bootstrap_aics) for result in bootstrap_results)
    if len(sample_sizes) > 1:
        validation_errors.append("Inconsistent bootstrap sample sizes across models")

    return len(validation_errors) == 0, validation_errors


def aggregate_stability_insights(win_rates: List[Dict[str, Any]],
                               ir_results: List[Dict[str, Any]],
                               consistency_results: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate stability insights from multiple analyses.

    Atomic function following CODING_STANDARDS.md Section 3.1 (30-50 lines).
    Single responsibility: Insights aggregation with executive summary.

    Parameters
    ----------
    win_rates : List[Dict[str, Any]]
        Win rate analysis results
    ir_results : List[Dict[str, Any]]
        Information ratio analysis results
    consistency_results : Dict[str, Any]
        Feature consistency evaluation results

    Returns
    -------
    Dict[str, Any]
        Aggregated stability insights with recommendations
    """
    top_performers = _build_top_performers(win_rates, ir_results)
    is_consensus = top_performers['consensus_winner'] != 'Mixed'

    return {
        'executive_summary': _build_executive_summary(win_rates, ir_results, consistency_results),
        'top_performers': top_performers,
        'consistency_insights': consistency_results['consistency_summary'],
        'recommendation': 'High confidence' if is_consensus else 'Moderate confidence',
        'analysis_metadata': {
            'win_rate_models': len(win_rates),
            'ir_models': len(ir_results),
            'features_analyzed': consistency_results['consistency_summary']['total_unique_features']
        }
    }


def format_stability_outputs(insights: Dict[str, Any],
                           metrics: Dict[str, Any]) -> str:
    """Format stability outputs for display and reporting.

    Atomic function following CODING_STANDARDS.md Section 3.1 (35-45 lines).
    Single responsibility: Output formatting with professional presentation.

    Parameters
    ----------
    insights : Dict[str, Any]
        Aggregated stability insights
    metrics : Dict[str, Any]
        Comprehensive stability metrics

    Returns
    -------
    str
        Formatted stability analysis report
    """
    try:
        report = f"""
=== COMPREHENSIVE STABILITY ANALYSIS REPORT ===

{insights.get('executive_summary', 'Executive summary not available')}

STABILITY METRICS:
  Models Analyzed: {metrics.get('models_analyzed', 'Unknown')}
  Bootstrap Samples: {metrics.get('bootstrap_samples_per_model', 'Unknown')}
  Overall Assessment: {metrics.get('overall_stability_assessment', 'Unknown')}

DISTRIBUTION:
"""
        # Add stability distribution
        for assessment, count in metrics.get('stability_distribution', {}).items():
            report += f"  {assessment}: {count} models\n"

        report += f"""
RECOMMENDATIONS:
  Selection Confidence: {insights.get('recommendation', 'Unknown')}
  Consensus Model: {insights.get('top_performers', {}).get('consensus_winner', 'Unknown')}

Analysis complete. See detailed results for model-specific insights.
"""

        return report.strip()

    except Exception as e:
        return f"Stability report formatting failed: {e}"