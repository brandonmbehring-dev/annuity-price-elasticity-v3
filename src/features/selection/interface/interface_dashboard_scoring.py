"""
Win Rate and Information Ratio Scoring Functions.

This module contains scoring system functions for integrating
win rate and information ratio metrics.

Module Responsibilities:
- Base model information extraction
- Information ratio normalization
- Stability grade computation
- Comprehensive result construction
- Unified scoring system integration

Used by: interface_dashboard.py (orchestrator)
"""

from typing import Dict, List, Any


def extract_base_model_info(win_rate_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract base model information from win rate result.

    Parameters
    ----------
    win_rate_result : Dict[str, Any]
        Win rate analysis result for a model

    Returns
    -------
    Dict[str, Any]
        Base model information dictionary
    """
    return {
        "model_name": win_rate_result["model"],
        "features": win_rate_result["features"],
        "original_aic": win_rate_result["original_aic"],
        "median_bootstrap_aic": win_rate_result["median_bootstrap_aic"],
    }


def normalize_ir_to_score(ir_value: float) -> float:
    """
    Normalize Information Ratio to 0-100 scale.

    Maps IR values from [-2, 2] to [0, 100] with clamping.

    Parameters
    ----------
    ir_value : float
        Information Ratio value

    Returns
    -------
    float
        Normalized score in [0, 100]
    """
    return max(0, min(100, (ir_value + 2) * 25))


def compute_stability_grade(composite_score: float) -> str:
    """
    Determine stability grade from composite score.

    Parameters
    ----------
    composite_score : float
        Composite score in [0, 100]

    Returns
    -------
    str
        Grade string (e.g., "A+ (Excellent)")
    """
    if composite_score >= 80:
        return "A+ (Excellent)"
    elif composite_score >= 70:
        return "A (Very Good)"
    elif composite_score >= 60:
        return "B+ (Good)"
    elif composite_score >= 50:
        return "B (Average)"
    else:
        return "C (Below Average)"


def build_comprehensive_result(
    base_info: Dict[str, Any],
    win_rate_score: float,
    ir_score: float,
    ir_value: float,
    composite_score: float,
    stability_grade: str,
    ir_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build comprehensive result dictionary for a single model.

    Parameters
    ----------
    base_info : Dict[str, Any]
        Base model information
    win_rate_score : float
        Win rate percentage score
    ir_score : float
        Normalized information ratio score
    ir_value : float
        Raw information ratio value
    composite_score : float
        Weighted composite score
    stability_grade : str
        Letter grade string
    ir_data : Dict[str, Any]
        Full information ratio data

    Returns
    -------
    Dict[str, Any]
        Comprehensive model result
    """
    return {
        **base_info,
        "win_rate_score": win_rate_score,
        "ir_score": ir_score,
        "composite_score": composite_score,
        "stability_grade": stability_grade,
        "in_sample_win_rate": win_rate_score,
        "out_sample_win_rate": win_rate_score,
        "individual_metrics": {
            "win_rate_pct": win_rate_score,
            "information_ratio": ir_value,
            "success_rate": ir_data.get("success_rate", 50.0),
        },
    }


def create_comprehensive_scoring_system(
    win_rate_results: List[Dict[str, Any]],
    ir_results: List[Dict[str, Any]],
    integration_weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Create unified scoring integrating Win Rate and IR.

    Orchestrator function: combines win rate and information ratio metrics
    into a comprehensive scoring system following CODING_STANDARDS.md.

    Parameters
    ----------
    win_rate_results : List[Dict[str, Any]]
        Win rate analysis results
    ir_results : List[Dict[str, Any]]
        Information ratio results
    integration_weights : Dict[str, float]
        Weights for combining metrics (win_rate_weight, information_ratio_weight)

    Returns
    -------
    List[Dict[str, Any]]
        Comprehensive scores sorted by composite score (descending)
    """
    ir_lookup = {result["model_name"]: result for result in ir_results}
    comprehensive_scores = []

    for win_rate_result in win_rate_results:
        model_name = win_rate_result["model"]
        base_info = extract_base_model_info(win_rate_result)

        win_rate_score = win_rate_result["win_rate_pct"]

        ir_data = ir_lookup.get(model_name, {})
        ir_value = ir_data.get("information_ratio", 0)
        ir_score = normalize_ir_to_score(ir_value)

        composite_score = (
            integration_weights["win_rate_weight"] * win_rate_score
            + integration_weights["information_ratio_weight"] * ir_score
        )

        stability_grade = compute_stability_grade(composite_score)

        comprehensive_result = build_comprehensive_result(
            base_info,
            win_rate_score,
            ir_score,
            ir_value,
            composite_score,
            stability_grade,
            ir_data,
        )
        comprehensive_scores.append(comprehensive_result)

    return sorted(comprehensive_scores, key=lambda x: x["composite_score"], reverse=True)
