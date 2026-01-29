"""
Model Comparison Helper Functions.

This module provides shared utility functions for model comparison visualizations:
- Pareto frontier calculation
- Bootstrap metrics extraction
- Constraint summary computation
- Model selection summary generation

Part of Phase 6.3 module split.

Module Architecture:
- comparison_helpers.py: Utility functions (this file)
- comparison_scatter_plots.py: Scatter plot functions
- comparison_coefficient_analysis.py: Coefficient analysis functions
- comparison_bootstrap_plots.py: Bootstrap distribution functions
- model_comparison.py: Orchestrator class
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# PARETO FRONTIER
# =============================================================================


def find_pareto_frontier(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """
    Find Pareto optimal points (minimize x, maximize y).

    Args:
        x_values: X-axis values (to minimize, e.g., AIC)
        y_values: Y-axis values (to maximize, e.g., R²)

    Returns:
        Boolean array indicating Pareto optimal points
    """
    pareto_mask = np.ones(len(x_values), dtype=bool)

    for i in range(len(x_values)):
        if pareto_mask[i]:
            # Point i is dominated if there exists j such that:
            # x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i])
            dominated = ((x_values <= x_values[i]) &
                       (y_values >= y_values[i]) &
                       ((x_values < x_values[i]) | (y_values > y_values[i])))

            # Remove point i if it's dominated by any other point
            if np.any(dominated & (np.arange(len(x_values)) != i)):
                pareto_mask[i] = False

    return pareto_mask


# =============================================================================
# BOOTSTRAP METRICS EXTRACTION
# =============================================================================


def extract_bootstrap_metrics(
    bootstrap_results: List[Dict[str, Any]]
) -> Tuple[List[str], List[float], List[float]]:
    """Extract model names, stability CVs, and success rates from bootstrap results."""
    model_names = []
    stability_cvs = []
    success_rates = []

    for result in bootstrap_results:
        if isinstance(result, dict):
            model_names.append(result.get('features', 'Unknown'))
            stability_cvs.append(result.get('aic_stability_cv', 0))
            success_rates.append(result.get('successful_fits', 100) / 100)
        elif hasattr(result, 'model_features'):
            model_names.append(result.model_features)
            stability_cvs.append(result.stability_metrics.get('aic_cv', 0))
            success_rates.append(result.stability_metrics.get('successful_fit_rate', 1))

    return model_names, stability_cvs, success_rates


def compute_stability_ranking(
    top_models: pd.DataFrame,
    bootstrap_results: List[Dict[str, Any]]
) -> Optional[np.ndarray]:
    """Compute stability ranking from bootstrap results."""
    bootstrap_dict = {}
    for result in bootstrap_results:
        if isinstance(result, dict):
            key = result.get('features')
            value = result.get('aic_stability_cv', 999)
        else:
            key = result.model_features
            value = result.stability_metrics.get('aic_cv', 999)
        bootstrap_dict[key] = value

    stability_values = [
        bootstrap_dict.get(model['features'], 999)
        for _, model in top_models.iterrows()
    ]

    return pd.Series(stability_values).rank().values


def sort_bootstrap_by_stability(
    bootstrap_results: List[Dict[str, Any]],
    top_n: int
) -> List[Tuple[float, Any]]:
    """Sort bootstrap results by stability (lowest CV first)."""
    sorted_results = []
    for result in bootstrap_results[:top_n]:
        if isinstance(result, dict):
            cv = result.get('aic_stability_cv', float('inf'))
        elif hasattr(result, 'stability_metrics'):
            cv = result.stability_metrics.get('aic_cv', float('inf'))
        else:
            cv = float('inf')
        sorted_results.append((cv, result))

    sorted_results.sort(key=lambda x: x[0])
    return sorted_results


# =============================================================================
# CONSTRAINT SUMMARY
# =============================================================================


def compute_constraint_summary(top_models: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute economic constraint summary for models."""
    constraint_summary = []

    for _, model in top_models.iterrows():
        coefficients = model.get('coefficients', {})
        features = model['features']

        violations = 0
        total_checks = 0

        for coef_name, coef_value in coefficients.items():
            if coef_name == 'const':
                continue
            total_checks += 1

            # Economic constraint rules
            if 'competitor' in coef_name.lower() and coef_value > 0:
                violations += 1
            elif 'prudential' in coef_name.lower() and coef_value < 0:
                violations += 1

        constraint_pass_rate = (total_checks - violations) / max(1, total_checks)
        constraint_summary.append({
            'model': features[:20],
            'pass_rate': constraint_pass_rate,
            'violations': violations,
            'total_checks': total_checks
        })

    return constraint_summary


# =============================================================================
# SIGN CONSISTENCY EXTRACTION
# =============================================================================


def extract_sign_consistency_data(
    coefficient_stability: Dict[str, Any],
    top_models: pd.DataFrame
) -> Tuple[List[str], List[float], List[float]]:
    """Extract sign consistency data from coefficient stability."""
    features_analyzed = []
    sign_consistency = []
    mean_values = []

    for model_features, feature_stats in coefficient_stability.items():
        if model_features in top_models['features'].values:
            for feature_name, stats in feature_stats.items():
                features_analyzed.append(f"{model_features.split('+')[0]}...\n{feature_name}")
                sign_consistency.append(stats.get('sign_consistency', 0))
                mean_values.append(abs(stats.get('mean', 0)))

    return features_analyzed, sign_consistency, mean_values


def extract_uncertainty_data(
    coefficient_stability: Dict[str, Any],
    top_models: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Extract uncertainty data from coefficient stability."""
    uncertainty_data = []

    for model_features, feature_stats in coefficient_stability.items():
        if model_features in top_models['features'].values:
            for feature_name, stats in feature_stats.items():
                uncertainty_data.append({
                    'feature': feature_name,
                    'model': model_features.split('+')[0] + '...',
                    'cv': stats.get('cv', 0),
                    'mean': stats.get('mean', 0),
                    'uncertainty_score': min(stats.get('cv', 0), 2.0)
                })

    return uncertainty_data


# =============================================================================
# MODEL SELECTION SUMMARY
# =============================================================================


def create_model_selection_summary(
    aic_results: pd.DataFrame,
    information_criteria_results: List[Dict[str, Any]],
    bootstrap_results: List[Dict[str, Any]]
) -> str:
    """
    Create text summary of model selection decision process.

    Args:
        aic_results: AIC evaluation results
        information_criteria_results: Information criteria results
        bootstrap_results: Bootstrap analysis results

    Returns:
        Formatted text summary
    """
    summary_lines = []
    summary_lines.append("MODEL SELECTION DECISION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append("")

    # Analysis scope
    summary_lines.append(f"Analysis Scope:")
    summary_lines.append(f"  • Total Models Evaluated: {len(aic_results)}")
    summary_lines.append(f"  • Information Criteria Models: {len(information_criteria_results) if information_criteria_results else 0}")
    summary_lines.append(f"  • Bootstrap Stability Models: {len(bootstrap_results) if bootstrap_results else 0}")
    summary_lines.append("")

    # Top performers
    if not aic_results.empty:
        best_aic_model = aic_results.iloc[0]
        summary_lines.append(f"Best AIC Performance:")
        summary_lines.append(f"  • Model: {best_aic_model['features']}")
        summary_lines.append(f"  • AIC Score: {best_aic_model['aic']:.4f}")
        summary_lines.append(f"  • R-squared: {best_aic_model['r_squared']:.4f}")
        summary_lines.append("")

    # Stability champion
    if bootstrap_results:
        most_stable = min(bootstrap_results,
                        key=lambda x: x.get('aic_stability_cv', 999) if isinstance(x, dict)
                                    else x.stability_metrics.get('aic_cv', 999))

        if isinstance(most_stable, dict):
            stable_name = most_stable.get('features', 'Unknown')
            stable_cv = most_stable.get('aic_stability_cv', 0)
        else:
            stable_name = most_stable.model_features
            stable_cv = most_stable.stability_metrics.get('aic_cv', 0)

        summary_lines.append(f"Most Stable Model:")
        summary_lines.append(f"  • Model: {stable_name}")
        summary_lines.append(f"  • Stability CV: {stable_cv:.6f}")
        summary_lines.append("")

    # Selection criteria
    summary_lines.append("Selection Methodology:")
    summary_lines.append("  ✓ AIC-based model quality assessment")
    summary_lines.append("  ✓ Economic constraint validation")
    summary_lines.append("  ✓ Bootstrap stability quantification")
    summary_lines.append("  ✓ Multi-criteria robustness evaluation")
    summary_lines.append("")

    summary_lines.append("Recommendation: Proceed with integrated selection approach")
    summary_lines.append("considering performance, stability, and business constraints.")

    return "\\n".join(summary_lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Pareto frontier
    'find_pareto_frontier',
    # Bootstrap metrics
    'extract_bootstrap_metrics',
    'compute_stability_ranking',
    'sort_bootstrap_by_stability',
    # Constraint summary
    'compute_constraint_summary',
    # Sign consistency
    'extract_sign_consistency_data',
    'extract_uncertainty_data',
    # Model selection summary
    'create_model_selection_summary',
]
