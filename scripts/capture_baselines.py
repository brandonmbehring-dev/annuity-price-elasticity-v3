#!/usr/bin/env python3
"""
Capture Baselines - Capture pre-refactoring outputs for equivalence testing.

Captures outputs from the current (pre-refactoring) implementation to enable
mathematical equivalence verification at 1e-12 precision.

Usage:
    python scripts/capture_baselines.py --output tests/baselines/pre_refactoring/
    python scripts/capture_baselines.py --output tests/baselines/golden/ --type golden
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def capture_feature_selection_baseline(
    data: pd.DataFrame,
    target_column: str = "sales_target_current",
    candidate_features: Optional[list] = None,
) -> Dict[str, Any]:
    """Capture feature selection results from direct pipeline call.

    Parameters
    ----------
    data : pd.DataFrame
        Input data for feature selection
    target_column : str
        Target column name
    candidate_features : list, optional
        Candidate features (auto-detected if not provided)

    Returns
    -------
    Dict[str, Any]
        Feature selection results
    """
    try:
        from src.features.selection.notebook_interface import (
            production_feature_selection,
        )

        # Get candidate features if not provided
        if candidate_features is None:
            # Use columns that look like rate features
            candidate_features = [
                col for col in data.columns
                if any(x in col.lower() for x in ['rate', 'competitor', 'lag'])
                and col != target_column
            ]

        if target_column not in data.columns:
            print(f"  Warning: Target column '{target_column}' not found in data")
            return {"error": f"Target column '{target_column}' not found"}

        # production_feature_selection uses: target, features (not target_column, candidate_features)
        results = production_feature_selection(
            data=data,
            target=target_column,
            features=candidate_features[:20],  # Limit for speed
            max_features=3,
        )

        # Convert to serializable dict
        if hasattr(results, 'to_dict'):
            return results.to_dict()
        elif hasattr(results, '__dict__'):
            return {k: v for k, v in results.__dict__.items() if not k.startswith('_')}
        else:
            return dict(results)

    except Exception as e:
        print(f"  Warning: Feature selection failed: {e}")
        return {"error": str(e)}


def capture_inference_baseline(
    data: pd.DataFrame,
    features: list,
    target_column: str = "sales_target_current",
) -> Dict[str, Any]:
    """Capture inference results from direct pipeline call.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    features : list
        Selected features
    target_column : str
        Target column

    Returns
    -------
    Dict[str, Any]
        Inference results
    """
    try:
        from src.models.inference_scenarios import center_baseline

        if not features:
            return {"error": "No features provided for inference"}

        # Check required columns exist
        missing = [f for f in features + [target_column] if f not in data.columns]
        if missing:
            return {"error": f"Missing columns: {missing}"}

        # Run center_baseline
        predictions, model = center_baseline(
            sales_df=data,
            rates_df=data,  # Rates embedded for RILA
            features=features,
            target_variable=target_column,
        )

        # Extract model coefficients
        coefficients = {}
        if hasattr(model, 'coef_'):
            coefficients = dict(zip(features, model.coef_.tolist()))

        return {
            "coefficients": coefficients,
            "intercept": float(model.intercept_) if hasattr(model, 'intercept_') else 0.0,
            "n_observations": len(data),
            "features_used": features,
        }

    except Exception as e:
        print(f"  Warning: Inference failed: {e}")
        return {"error": str(e)}


def capture_forecasting_baseline(
    data: pd.DataFrame,
    target_column: str = "sales_target_current",
) -> Dict[str, Any]:
    """Capture forecasting results from direct pipeline call.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    target_column : str
        Target column

    Returns
    -------
    Dict[str, Any]
        Forecasting results (metadata only - full results too large)
    """
    try:
        from src.models.forecasting_orchestrator import run_forecasting_pipeline

        # Just capture metadata, not full predictions
        return {
            "data_shape": list(data.shape),
            "target_column": target_column,
            "available": True,
            "note": "Full forecasting baseline requires dedicated capture",
        }

    except ImportError:
        return {"available": False, "note": "Forecasting module not available"}
    except Exception as e:
        print(f"  Warning: Forecasting check failed: {e}")
        return {"error": str(e)}


def load_fixture_data() -> Optional[pd.DataFrame]:
    """Load data from test fixtures.

    Returns
    -------
    pd.DataFrame or None
        Loaded fixture data
    """
    fixture_paths = [
        Path("tests/fixtures/rila/final_weekly_dataset.parquet"),
        Path("tests/fixtures/rila/competitive_features_engineered.parquet"),
        Path("tests/baselines/aws_mode/10_final_dataset.parquet"),
    ]

    for path in fixture_paths:
        if path.exists():
            print(f"  Loading fixture: {path}")
            return pd.read_parquet(path)

    print("  Warning: No fixture data found")
    return None


def run_capture(
    output_dir: str,
    capture_type: str = "pre_refactoring",
) -> bool:
    """Run baseline capture.

    Parameters
    ----------
    output_dir : str
        Output directory for baselines
    capture_type : str
        Type of capture (pre_refactoring, golden)

    Returns
    -------
    bool
        True if successful
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BASELINE CAPTURE - Type: {capture_type}")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")

    # Load data
    print("\n[1/4] Loading fixture data...")
    data = load_fixture_data()

    if data is None:
        print("ERROR: Could not load fixture data")
        return False

    print(f"  Loaded {len(data)} rows, {len(data.columns)} columns")

    # Detect target column
    target_candidates = ['sales_target_current', 'target', 'y', 'sales']
    target_column = None
    for col in target_candidates:
        if col in data.columns:
            target_column = col
            break

    if target_column is None:
        # Use first numeric column as fallback
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_column = numeric_cols[0]
            print(f"  Warning: Using '{target_column}' as target (fallback)")

    # Capture feature selection
    print("\n[2/4] Capturing feature selection baseline...")
    fs_results = capture_feature_selection_baseline(data, target_column)

    fs_path = output_path / "feature_selection_results.json"
    with open(fs_path, 'w') as f:
        json.dump(fs_results, f, indent=2, default=str)
    print(f"  Saved: {fs_path}")

    # Capture inference
    print("\n[3/4] Capturing inference baseline...")
    features = fs_results.get('selected_features', [])
    if not features and 'error' not in fs_results:
        # Try to get features from the results structure
        features = list(fs_results.get('coefficients', {}).keys())[:3]

    if features:
        inf_results = capture_inference_baseline(data, features, target_column)
    else:
        inf_results = {"error": "No features available from feature selection"}

    inf_path = output_path / "inference_results.json"
    with open(inf_path, 'w') as f:
        json.dump(inf_results, f, indent=2, default=str)
    print(f"  Saved: {inf_path}")

    # Capture forecasting metadata
    print("\n[4/4] Capturing forecasting baseline...")
    fc_results = capture_forecasting_baseline(data, target_column)

    fc_path = output_path / "forecasting_results.json"
    with open(fc_path, 'w') as f:
        json.dump(fc_results, f, indent=2, default=str)
    print(f"  Saved: {fc_path}")

    # Write metadata
    metadata = {
        "capture_date": datetime.datetime.now().isoformat(),
        "capture_type": capture_type,
        "python_version": sys.version,
        "data_shape": list(data.shape),
        "target_column": target_column,
        "files_captured": [
            "feature_selection_results.json",
            "inference_results.json",
            "forecasting_results.json",
        ],
        "notes": [
            "Baselines captured from current implementation",
            "Use equivalence_guard.py to verify refactoring preserves outputs",
        ],
    }

    meta_path = output_path / "capture_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {meta_path}")

    print(f"\n{'='*60}")
    print("CAPTURE COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    print(f"Files: {len(metadata['files_captured'])} + metadata")

    return True


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Capture pre-refactoring baselines for equivalence testing."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/baselines/pre_refactoring/",
        help="Output directory for baselines",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="pre_refactoring",
        choices=["pre_refactoring", "golden"],
        help="Type of capture",
    )

    args = parser.parse_args()

    success = run_capture(args.output, args.type)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
