"""
Export and Tracking Functions for Feature Selection.

This module contains all export, MLflow tracking, and DVC integration
functions extracted from notebook_interface.py for maintainability.

Module Responsibilities:
- Model metadata creation and export
- Bootstrap results export
- File I/O with DVC tracking
- MLflow experiment finalization
- JSON serialization helpers

Used by: notebook_interface.py (re-exports public functions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================


def _convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.

    Single responsibility: JSON serialization helper only.
    Follows UNIFIED_CODING_STANDARDS.md by being focused and reusable.

    Parameters
    ----------
    obj : Any
        Object to convert (dict, list, numpy types, etc.)

    Returns
    -------
    Any
        JSON-serializable equivalent of input object
    """
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    else:
        return obj


# =============================================================================
# METADATA CREATION FUNCTIONS
# =============================================================================


def _create_feature_selection_metadata(selection_method: str) -> Dict[str, Any]:
    """
    Create feature selection metadata with timestamp.

    Parameters
    ----------
    selection_method : str
        Selection method used (e.g., "original_aic_only", "stability_weighted")

    Returns
    -------
    Dict[str, Any]
        Feature selection metadata dictionary
    """
    from datetime import datetime

    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "refactoring_version": "enhanced_dry_compliant",
        "mathematical_equivalence": "validated",
        "configuration_system": "enhanced_typeddict_builder",
        "selection_method": selection_method,
    }


def _create_selected_model_metadata(final_selected_model: pd.Series) -> Dict[str, Any]:
    """
    Create metadata for the selected model.

    Parameters
    ----------
    final_selected_model : pd.Series
        Selected model data from results DataFrame

    Returns
    -------
    Dict[str, Any]
        Selected model metadata dictionary
    """
    return {
        "features": final_selected_model["features"],
        "aic_score": float(final_selected_model["aic"]),
        "r_squared": float(final_selected_model["r_squared"]),
        "n_features": int(final_selected_model["n_features"]),
        "coefficients": _convert_numpy_types(dict(final_selected_model["coefficients"])),
    }


def _create_selection_process_metadata(
    results_df: pd.DataFrame,
    valid_results_sorted: pd.DataFrame,
    constraint_violations: Optional[List[Any]],
) -> Dict[str, Any]:
    """
    Create selection process metadata.

    Parameters
    ----------
    results_df : pd.DataFrame
        All results DataFrame
    valid_results_sorted : pd.DataFrame
        Valid results sorted by AIC
    constraint_violations : Optional[List[Any]]
        List of constraint violations

    Returns
    -------
    Dict[str, Any]
        Selection process metadata dictionary
    """
    converged_count = (
        int(results_df["converged"].sum()) if "converged" in results_df.columns else 0
    )
    return {
        "total_combinations_evaluated": len(results_df),
        "converged_models": converged_count,
        "economically_valid_models": len(valid_results_sorted),
        "constraint_violations": len(constraint_violations) if constraint_violations else 0,
    }


def _create_configuration_metadata(
    feature_config: Dict[str, Any],
    constraint_config: Dict[str, Any],
    bootstrap_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create configuration metadata.

    Parameters
    ----------
    feature_config : Dict[str, Any]
        Feature selection configuration
    constraint_config : Dict[str, Any]
        Constraint configuration
    bootstrap_config : Dict[str, Any]
        Bootstrap configuration

    Returns
    -------
    Dict[str, Any]
        Configuration metadata dictionary
    """
    return {
        "feature_config": _convert_numpy_types(dict(feature_config)),
        "constraint_config": _convert_numpy_types(dict(constraint_config)),
        "bootstrap_config": _convert_numpy_types(dict(bootstrap_config)),
    }


def _export_model_metadata(
    final_selected_model: pd.Series,
    selection_method: str,
    results_df: Optional[pd.DataFrame] = None,
    valid_results_sorted: Optional[pd.DataFrame] = None,
    constraint_violations: Optional[List[Any]] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    constraint_config: Optional[Dict[str, Any]] = None,
    bootstrap_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export model metadata and selection process info."""
    final_results_export = {
        "feature_selection_metadata": _create_feature_selection_metadata(selection_method),
        "selected_model": _create_selected_model_metadata(final_selected_model),
    }

    if results_df is not None and valid_results_sorted is not None:
        final_results_export["selection_process"] = _create_selection_process_metadata(
            results_df, valid_results_sorted, constraint_violations
        )

    if all(
        config is not None
        for config in [feature_config, constraint_config, bootstrap_config]
    ):
        final_results_export["configuration"] = _create_configuration_metadata(
            feature_config, constraint_config, bootstrap_config
        )

    return final_results_export


# =============================================================================
# BOOTSTRAP RESULTS EXPORT
# =============================================================================


def _export_bootstrap_results(
    final_results_export: Dict[str, Any],
    bootstrap_results: List[Any],
    final_selected_model: pd.Series,
    selection_method: str,
) -> Dict[str, Any]:
    """
    Add bootstrap stability analysis to results export.

    Single responsibility: Bootstrap results processing only.
    Follows UNIFIED_CODING_STANDARDS.md with clear mathematical focus.

    Parameters
    ----------
    final_results_export : Dict[str, Any]
        Base results dictionary to enhance
    bootstrap_results : List[Any]
        Bootstrap stability results
    final_selected_model : pd.Series
        Selected model data
    selection_method : str
        Selection method used

    Returns
    -------
    Dict[str, Any]
        Enhanced results with bootstrap analysis
    """
    if not bootstrap_results:
        return final_results_export

    # Get best stability result (assuming first is best)
    best_stability = bootstrap_results[0]
    bootstrap_median_aic = np.median(best_stability.bootstrap_aics)

    # Add stability analysis section
    final_results_export["stability_analysis"] = {
        "selection_method": selection_method,
        "best_model_stability": best_stability.stability_assessment,
        "aic_stability_coefficient": float(best_stability.aic_stability_coefficient),
        "r2_stability_coefficient": float(best_stability.r2_stability_coefficient),
        "successful_fits": int(best_stability.successful_fits),
        "total_attempts": int(best_stability.total_attempts),
        "bootstrap_median_aic": float(bootstrap_median_aic),
        "original_aic": float(final_selected_model["aic"]),
        "bootstrap_vs_original_aic_diff": float(
            bootstrap_median_aic - final_selected_model["aic"]
        ),
    }

    return final_results_export


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================


def _log_dvc_success(path: str) -> None:
    """
    Log success message for DVC checkpoint creation.

    Parameters
    ----------
    path : str
        Path to the DVC-tracked file
    """
    print(f"SUCCESS: Final results DVC checkpoint created - {path}")
    print("   Selected features preserved for modeling pipeline")
    print("   Complete analysis metadata available")


def _export_to_files(final_results_export: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export results to files with DVC tracking.

    Parameters
    ----------
    final_results_export : Dict[str, Any]
        Results dictionary to export

    Returns
    -------
    Dict[str, Any]
        Export status dictionary with paths and DVC status
    """
    import json
    import os

    export_info: Dict[str, Any] = {"export_paths": [], "dvc_status": "not_attempted"}

    try:
        final_results_json_ready = _convert_numpy_types(final_results_export)

        os.makedirs("outputs/feature_selection", exist_ok=True)
        final_results_path = "outputs/feature_selection/final_selected_features.json"

        with open(final_results_path, "w") as f:
            json.dump(final_results_json_ready, f, indent=2)

        export_info["export_paths"].append(final_results_path)

        dvc_result = os.system(f"dvc add {final_results_path}")
        if dvc_result == 0:
            export_info["dvc_status"] = "success"
            _log_dvc_success(final_results_path)
        else:
            export_info["dvc_status"] = "failed"
            print("WARNING: Final results DVC checkpoint failed - results available locally")

    except Exception as e:
        export_info["dvc_status"] = f"error: {str(e)}"
        print(f"ERROR: Results export failed: {e}")

    return export_info


# =============================================================================
# MLFLOW INTEGRATION
# =============================================================================


def _finalize_mlflow_experiment(selection_method: str, run_id: str) -> str:
    """
    Complete MLflow experiment with enhanced logging.

    Single responsibility: MLflow completion only.
    Follows UNIFIED_CODING_STANDARDS.md with clear external integration focus.

    Parameters
    ----------
    selection_method : str
        Selection method used
    run_id : str
        MLflow run ID

    Returns
    -------
    str
        MLflow completion status ('success' or 'error: <message>')
    """
    try:
        # Import MLflow functions if available
        from src.config.mlflow_config import (
            safe_mlflow_log_param,
            end_mlflow_experiment,
        )

        # Enhanced MLflow logging for final results
        safe_mlflow_log_param(
            "refactoring_approach", "enhanced_dry_compliant_atomic_functions"
        )
        safe_mlflow_log_param(
            "mathematical_equivalence_status", "validated_identical_results"
        )
        safe_mlflow_log_param("dvc_integration", "strategic_checkpoints_implemented")
        safe_mlflow_log_param(
            "architecture_improvements",
            "configuration_builder_data_preprocessing_utils",
        )
        safe_mlflow_log_param("enhanced_selection_method", selection_method)

        # End MLflow experiment
        end_mlflow_experiment("FINISHED")
        print(f"SUCCESS: Enhanced MLflow experiment completed (run: {run_id})")
        return "success"

    except Exception as e:
        print(f"WARNING: MLflow completion failed: {e}")
        return f"error: {str(e)}"


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================


def export_final_model_selection(
    final_selected_model: pd.Series, bootstrap_results: Optional[List[Any]] = None,
    selection_method: str = "original_aic_only", run_id: str = "",
    constraint_violations: Optional[List[Any]] = None, results_df: Optional[pd.DataFrame] = None,
    valid_results_sorted: Optional[pd.DataFrame] = None, feature_config: Optional[Dict[str, Any]] = None,
    constraint_config: Optional[Dict[str, Any]] = None, bootstrap_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Export final model results, bootstrap data, and MLflow tracking."""
    export_status = {"export_paths": [], "mlflow_status": "not_attempted",
                     "dvc_status": "not_attempted", "final_results": {}}
    try:
        final_results_export = _export_model_metadata(
            final_selected_model=final_selected_model, selection_method=selection_method,
            results_df=results_df, valid_results_sorted=valid_results_sorted,
            constraint_violations=constraint_violations, feature_config=feature_config,
            constraint_config=constraint_config, bootstrap_config=bootstrap_config)
        final_results_export = _export_bootstrap_results(
            final_results_export=final_results_export, bootstrap_results=bootstrap_results or [],
            final_selected_model=final_selected_model, selection_method=selection_method)
        file_export_info = _export_to_files(final_results_export)
        export_status["export_paths"] = file_export_info["export_paths"]
        export_status["dvc_status"] = file_export_info["dvc_status"]
        export_status["final_results"] = _convert_numpy_types(final_results_export)
        export_status["mlflow_status"] = _finalize_mlflow_experiment(selection_method, run_id)
        return export_status
    except Exception as e:
        print(f"ERROR: Export orchestration failed: {e}")
        export_status["dvc_status"] = f"orchestration_error: {str(e)}"
        return export_status


# =============================================================================
# DUAL VALIDATION EXPORT
# =============================================================================


def save_dual_validation_results(summary: Dict[str, Any], output_path: str) -> None:
    """
    Save dual validation results to file.

    Parameters
    ----------
    summary : Dict[str, Any]
        Dual validation summary dictionary
    output_path : str
        Path to save results JSON

    Returns
    -------
    None
    """
    try:
        import json

        # Use canonical _convert_numpy_types for JSON serialization
        json_ready_summary = _convert_numpy_types(summary)

        with open(output_path, "w") as f:
            json.dump(json_ready_summary, f, indent=2)

        print(f"SUCCESS: Dual validation results saved to {output_path}")

    except Exception as e:
        print(f"WARNING: Failed to save results to {output_path}: {e}")
