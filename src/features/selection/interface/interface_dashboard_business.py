"""
Final Business Recommendation Functions.

This module contains business logic for generating final recommendations
based on comprehensive stability scoring.

Module Responsibilities:
- Determine empty recommendation fallback
- Calculate optimal balance recommendations
- Compare stability vs AIC trade-offs
- Generate alternative model suggestions
- Build grade distribution summaries
- Generate final business recommendations

Used by: interface_dashboard.py (orchestrator)
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def get_empty_recommendation() -> Dict[str, Any]:
    """
    Return empty recommendation when no models available.

    Returns
    -------
    Dict[str, Any]
        Empty recommendation dictionary
    """
    return {
        "primary_recommendation": "No models available for recommendation",
        "confidence_level": "None",
        "alternative_models": [],
    }


def determine_recommendation_when_same(
    best_overall: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Determine recommendation when best overall equals best AIC model.

    Parameters
    ----------
    best_overall : Dict[str, Any]
        Best overall model data

    Returns
    -------
    Tuple[str, str, str]
        (primary_recommendation, confidence_level, rationale)
    """
    return (
        f"{best_overall['model_name']} (Optimal Balance)",
        "Very High",
        "Achieves optimal balance - best AIC performance with highest stability",
    )


def determine_recommendation_when_different(
    best_overall: Dict[str, Any], best_aic_model: Dict[str, Any]
) -> Tuple[str, str, str]:
    """
    Determine recommendation when best overall differs from best AIC model.

    Parameters
    ----------
    best_overall : Dict[str, Any]
        Best overall stability model
    best_aic_model : Dict[str, Any]
        Best AIC model

    Returns
    -------
    Tuple[str, str, str]
        (primary_recommendation, confidence_level, rationale)
    """
    stability_advantage = (
        best_overall["composite_score"] - best_aic_model["composite_score"]
    )
    aic_cost = best_overall["original_aic"] - best_aic_model["original_aic"]

    if stability_advantage > 10 and aic_cost < 5:
        return (
            f"{best_overall['model_name']} (Stability Leader)",
            "High",
            f"Superior stability ({stability_advantage:.1f} points) justifies modest AIC cost ({aic_cost:.1f})",
        )
    elif aic_cost > 10:
        return (
            f"{best_aic_model['model_name']} (AIC Leader)",
            "Moderate",
            f"AIC advantage ({aic_cost:.1f}) outweighs stability difference ({stability_advantage:.1f})",
        )
    else:
        return (
            f"{best_overall['model_name']} (Balanced Choice)",
            "High",
            "Close performance across win rate and risk-adjusted metrics",
        )


def build_alternative_models_list(
    comprehensive_scores: List[Dict[str, Any]]
) -> List[str]:
    """
    Build list of alternative model recommendations.

    Parameters
    ----------
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive scoring results

    Returns
    -------
    List[str]
        List of alternative model descriptions
    """
    alternatives = []
    for model in comprehensive_scores[1:4]:
        if model["composite_score"] >= 60:
            alternatives.append(
                f"{model['model_name']} ({model['composite_score']:.1f}/100)"
            )
    return alternatives


def build_grade_distribution(
    comprehensive_scores: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Build grade distribution from comprehensive scores.

    Parameters
    ----------
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive scoring results

    Returns
    -------
    Dict[str, int]
        Grade counts by letter grade
    """
    return {
        "A+": sum(1 for r in comprehensive_scores if r["composite_score"] >= 80),
        "A": sum(1 for r in comprehensive_scores if 70 <= r["composite_score"] < 80),
        "B+": sum(1 for r in comprehensive_scores if 60 <= r["composite_score"] < 70),
        "B": sum(1 for r in comprehensive_scores if 50 <= r["composite_score"] < 60),
        "C": sum(1 for r in comprehensive_scores if r["composite_score"] < 50),
    }


def build_best_model_summary(best_overall: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build summary for best overall model.

    Parameters
    ----------
    best_overall : Dict[str, Any]
        Best overall model data

    Returns
    -------
    Dict[str, Any]
        Best model summary dictionary
    """
    return {
        "model_name": best_overall["model_name"],
        "composite_score": best_overall["composite_score"],
        "stability_grade": best_overall["stability_grade"],
        "win_rate_score": best_overall["win_rate_score"],
        "ir_score": best_overall["ir_score"],
        "model_features": best_overall["features"],
    }


def generate_final_recommendations(
    comprehensive_scores: List[Dict[str, Any]], bootstrap_results: List[Any]
) -> Dict[str, Any]:
    """
    Generate final business recommendations based on comprehensive scoring.

    Orchestrator function: combines multiple recommendation logic helpers
    following CODING_STANDARDS.md.

    Parameters
    ----------
    comprehensive_scores : List[Dict[str, Any]]
        Comprehensive scoring results
    bootstrap_results : List[Any]
        Bootstrap results (for validation)

    Returns
    -------
    Dict[str, Any]
        Final recommendation dictionary with primary, alternatives, and summary
    """
    if not comprehensive_scores:
        return get_empty_recommendation()

    best_overall = comprehensive_scores[0]
    best_aic_idx = np.argmin(
        [result["original_aic"] for result in comprehensive_scores]
    )
    best_aic_model = comprehensive_scores[best_aic_idx]

    if best_overall["model_name"] == best_aic_model["model_name"]:
        primary_recommendation, confidence_level, rationale = (
            determine_recommendation_when_same(best_overall)
        )
    else:
        primary_recommendation, confidence_level, rationale = (
            determine_recommendation_when_different(best_overall, best_aic_model)
        )

    excellent_count = sum(
        1 for r in comprehensive_scores if r["composite_score"] >= 80
    )
    good_count = sum(
        1 for r in comprehensive_scores if 60 <= r["composite_score"] < 80
    )

    return {
        "primary_recommendation": primary_recommendation,
        "confidence_level": confidence_level,
        "rationale": rationale,
        "alternative_models": build_alternative_models_list(comprehensive_scores),
        "performance_summary": {
            "best_overall_model": build_best_model_summary(best_overall),
            "grade_distribution": build_grade_distribution(comprehensive_scores),
            "top_5_models": comprehensive_scores[:5],
            "total_high_quality": excellent_count + good_count,
        },
    }
