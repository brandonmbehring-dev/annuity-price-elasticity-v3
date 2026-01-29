"""
Dashboard and Stability Analysis Functions for Feature Selection.

This module orchestrates comprehensive stability dashboard generation,
integrating win rate and information ratio scoring with visualizations
and business recommendations.

Module Responsibilities:
- Main orchestration of dashboard generation workflow
- Re-export all public functions for backward compatibility
- Integration of validation, scoring, visualization, business, and DVC modules

Architecture (Post-Refactoring):
- Validation: interface_dashboard_validation.py (input validation, config extraction)
- Scoring: interface_dashboard_scoring.py (win rate + IR integration)
- Visualization: interface_dashboard_viz.py (6 plot types)
- Business: interface_dashboard_business.py (final recommendations)
- DVC: interface_dashboard_dvc.py (checkpoint creation)

Used by: notebook_interface.py (re-exports public functions)
"""

from typing import Dict, List, Any

# Import atomic operations from specialized modules
from .interface_dashboard_validation import (
    validate_dashboard_inputs,
    import_advanced_stability_analysis,
    extract_dashboard_config,
    process_advanced_stability_results,
)

from .interface_dashboard_scoring import (
    extract_base_model_info,
    normalize_ir_to_score,
    compute_stability_grade,
    build_comprehensive_result,
    create_comprehensive_scoring_system,
)

from .interface_dashboard_viz import (
    plot_winrate_vs_ir_scatter,
    plot_composite_score_distribution,
    plot_winrate_rankings,
    plot_ir_rankings,
    plot_aic_vs_stability,
    plot_recommendation_summary,
    create_comprehensive_dashboard_visualizations,
    create_dashboard_visualizations_safe,
)

from .interface_dashboard_business import (
    get_empty_recommendation,
    determine_recommendation_when_same,
    determine_recommendation_when_different,
    build_alternative_models_list,
    build_grade_distribution,
    build_best_model_summary,
    generate_final_recommendations,
)

from .interface_dashboard_dvc import (
    convert_numpy_types_for_checkpoint,
    build_dashboard_checkpoint_data,
    create_dashboard_dvc_checkpoint,
)


def _execute_dashboard_analysis_pipeline(
    bootstrap_results: List[Any],
    config: Dict[str, Any],
    integration_weights: Dict[str, float],
    run_advanced_stability_analysis: Any,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Execute dashboard analysis pipeline (stability + IR + scoring).

    Helper function: extracts steps 2-4 from main orchestrator to reduce
    complexity per CODING_STANDARDS.md §3.1 (30-50 line function limit).

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results from feature selection
    config : Dict[str, Any]
        Dashboard configuration with models_to_analyze
    integration_weights : Dict[str, float]
        Weights for win_rate and information_ratio in unified scoring
    run_advanced_stability_analysis : Any
        Imported advanced stability analysis function

    Returns
    -------
    tuple[List[Dict], List[Dict], List[Dict]]
        (win_rate_results, ir_results, comprehensive_scores)

    Raises
    ------
    RuntimeError
        If analysis pipeline execution fails
    """
    # Step 2: Run advanced stability analysis (win rate + IR)
    advanced_config = config.copy()
    advanced_config["models_to_analyze"] = config.get("models_to_analyze", 15)
    advanced_results = run_advanced_stability_analysis(
        bootstrap_results, advanced_config
    )

    # Step 3: Process stability results
    win_rate_results = advanced_results.get("win_rate_results", [])
    ir_results = advanced_results.get("information_ratio_results", [])

    # Step 4: Integrate into unified scoring system
    print("  Integrating results into unified scoring system...")
    comprehensive_scores = create_comprehensive_scoring_system(
        win_rate_results, ir_results, integration_weights
    )
    print(f"  Comprehensive scoring complete: {len(comprehensive_scores)} models scored")

    return win_rate_results, ir_results, comprehensive_scores


# Re-export all public functions for backward compatibility
__all__ = [
    # Main public function
    "generate_comprehensive_stability_dashboard",
    # Validation functions
    "validate_dashboard_inputs",
    "import_advanced_stability_analysis",
    "extract_dashboard_config",
    "process_advanced_stability_results",
    # Scoring functions
    "extract_base_model_info",
    "normalize_ir_to_score",
    "compute_stability_grade",
    "build_comprehensive_result",
    "create_comprehensive_scoring_system",
    # Visualization functions
    "plot_winrate_vs_ir_scatter",
    "plot_composite_score_distribution",
    "plot_winrate_rankings",
    "plot_ir_rankings",
    "plot_aic_vs_stability",
    "plot_recommendation_summary",
    "create_comprehensive_dashboard_visualizations",
    "create_dashboard_visualizations_safe",
    # Business functions
    "get_empty_recommendation",
    "determine_recommendation_when_same",
    "determine_recommendation_when_different",
    "build_alternative_models_list",
    "build_grade_distribution",
    "build_best_model_summary",
    "generate_final_recommendations",
    # DVC functions
    "convert_numpy_types_for_checkpoint",
    "build_dashboard_checkpoint_data",
    "create_dashboard_dvc_checkpoint",
]


def generate_comprehensive_stability_dashboard(
    bootstrap_results: List[Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive stability dashboard with win rate and IR integration.

    Main orchestrator function: coordinates validation, scoring, visualization,
    business recommendations, and DVC checkpointing following CODING_STANDARDS.md.

    Parameters
    ----------
    bootstrap_results : List[Any]
        Bootstrap analysis results from feature selection
    config : Dict[str, Any]
        Dashboard configuration with keys:
        - models_to_analyze: int (default 15)
        - create_visualizations: bool (default True)
        - integration_weights: Dict[str, float] (win_rate_weight, information_ratio_weight)
        - create_dvc_checkpoint: bool (default False)
        - fig_width: int (default 16)
        - fig_height: int (default 12)

    Returns
    -------
    Dict[str, Any]
        Dashboard results with keys:
        - win_rate_results: List[Dict] (win rate analysis for each model)
        - information_ratio_results: List[Dict] (IR analysis for each model)
        - comprehensive_scores: List[Dict] (integrated scores sorted by composite)
        - visualizations: Dict[str, plt.Figure] (dashboard plots)
        - final_recommendations: Dict (business recommendations)

    Raises
    ------
    ValueError
        If inputs are invalid (empty bootstrap_results, invalid config)
    RuntimeError
        If dashboard generation fails at any stage

    Examples
    --------
    >>> config = {
    ...     "models_to_analyze": 15,
    ...     "create_visualizations": True,
    ...     "integration_weights": {"win_rate_weight": 0.5, "information_ratio_weight": 0.5}
    ... }
    >>> results = generate_comprehensive_stability_dashboard(bootstrap_results, config)
    >>> print(results["comprehensive_scores"][0])  # Best model
    """
    # Step 1: Validate inputs
    validate_dashboard_inputs(bootstrap_results, config)
    run_advanced_stability_analysis = import_advanced_stability_analysis()
    n_models, create_visualizations, integration_weights = extract_dashboard_config(
        config, bootstrap_results
    )

    results: Dict[str, Any] = {}

    try:
        # Steps 2-4: Execute analysis pipeline (extracted for code brevity)
        win_rate, ir, scores = _execute_dashboard_analysis_pipeline(
            bootstrap_results, config, integration_weights, run_advanced_stability_analysis
        )
        results["win_rate_results"] = win_rate
        results["information_ratio_results"] = ir
        results["comprehensive_scores"] = scores

        # Step 5: Create visualizations if requested
        if create_visualizations:
            create_dashboard_visualizations_safe(results, scores, config)

        # Step 6: Generate business recommendations
        recommendations = generate_final_recommendations(scores, bootstrap_results)
        results["final_recommendations"] = recommendations
        print("  Business recommendations generated")

        # Step 7: Create DVC checkpoint if requested
        if config.get("create_dvc_checkpoint", False):
            try:
                create_dashboard_dvc_checkpoint(results, config)
                print("  DVC checkpoint created successfully")
            except Exception as e:
                print(f"WARNING: DVC checkpoint creation failed: {str(e)}")

    except Exception as e:
        raise RuntimeError(
            f"Comprehensive stability dashboard generation failed: {str(e)}"
        ) from e

    print("SUCCESS: Comprehensive stability dashboard complete")
    return results
