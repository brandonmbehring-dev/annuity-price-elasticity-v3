"""
DVC Checkpoint Creation for Dashboard Results.

This module contains functions for creating DVC checkpoints
of comprehensive dashboard results for version control.

Module Responsibilities:
- Convert numpy types to JSON-serializable formats
- Build checkpoint data structures
- Write checkpoint files with metadata
- Create DVC version control entries

Used by: interface_dashboard.py (orchestrator)
"""

import json
import os
import numpy as np
from typing import Dict, Any
from datetime import datetime


def convert_numpy_types_for_checkpoint(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.

    Single responsibility: JSON serialization helper only.

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    Any
        JSON-serializable equivalent
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types_for_checkpoint(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_for_checkpoint(item) for item in obj]
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


def build_dashboard_checkpoint_data(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build checkpoint data structure for dashboard results.

    Parameters
    ----------
    results : Dict[str, Any]
        Dashboard results
    config : Dict[str, Any]
        Dashboard configuration

    Returns
    -------
    Dict[str, Any]
        Checkpoint data dictionary
    """
    return {
        "dashboard_metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "dashboard_version": "comprehensive_v1",
            "analysis_type": "win_rate_information_ratio_integration",
        },
        "comprehensive_scores": convert_numpy_types_for_checkpoint(
            results.get("comprehensive_scores", [])
        ),
        "win_rate_results": convert_numpy_types_for_checkpoint(
            results.get("win_rate_results", [])
        ),
        "information_ratio_results": convert_numpy_types_for_checkpoint(
            results.get("information_ratio_results", [])
        ),
        "final_recommendations": convert_numpy_types_for_checkpoint(
            results.get("final_recommendations", {})
        ),
        "configuration": convert_numpy_types_for_checkpoint(config),
    }


def create_dashboard_dvc_checkpoint(
    results: Dict[str, Any], config: Dict[str, Any]
) -> None:
    """
    Create DVC checkpoint for comprehensive dashboard results.

    Parameters
    ----------
    results : Dict[str, Any]
        Dashboard results
    config : Dict[str, Any]
        Dashboard configuration

    Raises
    ------
    RuntimeError
        If DVC checkpoint creation fails
    """
    checkpoint_data = build_dashboard_checkpoint_data(results, config)

    os.makedirs("outputs/feature_selection", exist_ok=True)
    checkpoint_path = "outputs/feature_selection/comprehensive_dashboard_results.json"

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    dvc_result = os.system(f"dvc add {checkpoint_path}")
    if dvc_result == 0:
        print(f"DVC checkpoint created: {checkpoint_path}")
    else:
        raise RuntimeError(f"DVC checkpoint creation failed for {checkpoint_path}")
