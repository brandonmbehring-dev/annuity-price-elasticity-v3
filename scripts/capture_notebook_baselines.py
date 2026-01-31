"""Capture notebook outputs as baselines for mathematical equivalence testing.

This script captures notebook outputs with fixed random seed (42) to enable
bit-identical validation at 1e-12 precision.

Usage:
    # Capture all products and notebooks
    python scripts/capture_notebook_baselines.py --all

    # Capture specific product
    python scripts/capture_notebook_baselines.py --product 6Y20B

    # Capture specific notebook for a product
    python scripts/capture_notebook_baselines.py --product 6Y20B --notebook nb01
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def get_baseline_dir(product: str, notebook: str) -> Path:
    """Get baseline directory for a product and notebook.

    Args:
        product: Product code (e.g., "6Y20B", "1Y10B")
        notebook: Notebook identifier (e.g., "nb00", "nb01", "nb02")

    Returns:
        Path to baseline directory
    """
    product_dir = f"rila_{product.lower()}"
    notebook_dir = {
        "nb00": "nb00_data_pipeline",
        "nb01": "nb01_price_elasticity",
        "nb02": "nb02_forecasting",
    }[notebook]

    return PROJECT_ROOT / "tests" / "baselines" / "notebooks" / product_dir / notebook_dir


def get_output_dir() -> Path:
    """Get notebook output directory."""
    return PROJECT_ROOT / "outputs"


def create_capture_metadata(
    notebook: str, product: str, outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Create capture metadata JSON.

    Args:
        notebook: Notebook identifier
        product: Product code
        outputs: Dictionary of output names to their metadata

    Returns:
        Metadata dictionary
    """
    return {
        "notebook": notebook,
        "product": product,
        "capture_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "outputs": outputs,
    }


def capture_dataframe_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Extract metadata from a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Metadata dictionary with shape, columns, dtypes
    """
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def capture_nb00_baselines(product: str) -> dict[str, str]:
    """Capture NB00 data pipeline outputs.

    Args:
        product: Product code (e.g., "6Y20B")

    Returns:
        Dictionary mapping output names to file paths
    """
    baseline_dir = get_baseline_dir(product, "nb00")
    baseline_dir.mkdir(parents=True, exist_ok=True)

    output_dir = get_output_dir()
    captured = {}
    outputs_metadata = {}

    # Files to capture from NB00
    files_to_capture = {
        "final_dataset.parquet": output_dir / "datasets" / "final_dataset.parquet",
        "weekly_aggregated.parquet": output_dir
        / "datasets"
        / "weekly_aggregated_features.parquet",
        "FlexGuard_Sales.parquet": output_dir / "datasets" / "FlexGuard_Sales.parquet",
        "WINK_competitive_rates.parquet": output_dir
        / "datasets"
        / "WINK_competitive_rates.parquet",
    }

    for dest_name, source_path in files_to_capture.items():
        if source_path.exists():
            dest_path = baseline_dir / dest_name
            shutil.copy2(source_path, dest_path)
            captured[dest_name] = str(dest_path)

            # Capture metadata
            df = pd.read_parquet(source_path)
            outputs_metadata[dest_name.replace(".parquet", "")] = (
                capture_dataframe_metadata(df)
            )
            print(f"  Captured: {dest_name} ({df.shape[0]} rows, {df.shape[1]} cols)")
        else:
            print(f"  WARNING: {source_path} not found")

    # Save metadata
    metadata = create_capture_metadata("nb00_data_pipeline", product, outputs_metadata)
    metadata_path = baseline_dir / "capture_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    captured["capture_metadata.json"] = str(metadata_path)

    return captured


def capture_nb01_baselines(product: str) -> dict[str, str]:
    """Capture NB01 price elasticity inference outputs.

    Args:
        product: Product code (e.g., "6Y20B")

    Returns:
        Dictionary mapping output names to file paths
    """
    baseline_dir = get_baseline_dir(product, "nb01")
    baseline_dir.mkdir(parents=True, exist_ok=True)

    captured = {}
    outputs_metadata = {}

    # Check for existing baseline structure subdirectories
    subdirs = ["01_data_prep", "02_bootstrap_model", "03_rate_scenarios",
               "04_confidence_intervals", "05_export"]

    for subdir in subdirs:
        subdir_path = baseline_dir / subdir
        if subdir_path.exists() and any(subdir_path.iterdir()):
            # Already captured - record existing files
            for f in subdir_path.glob("*.parquet"):
                df = pd.read_parquet(f)
                key = f"{subdir}/{f.name}".replace(".parquet", "")
                outputs_metadata[key] = capture_dataframe_metadata(df)
                captured[f"{subdir}/{f.name}"] = str(f)
                print(f"  Existing: {subdir}/{f.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Also capture top-level files if they exist
    for f in baseline_dir.glob("*.parquet"):
        df = pd.read_parquet(f)
        outputs_metadata[f.stem] = capture_dataframe_metadata(df)
        captured[f.name] = str(f)
        print(f"  Existing: {f.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Check for numpy arrays
    for f in baseline_dir.glob("*.npy"):
        arr = np.load(f)
        outputs_metadata[f.stem] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        captured[f.name] = str(f)
        print(f"  Existing: {f.name} (shape {arr.shape})")

    # Update metadata
    metadata = create_capture_metadata("nb01_price_elasticity", product, outputs_metadata)
    metadata_path = baseline_dir / "capture_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    captured["capture_metadata.json"] = str(metadata_path)

    return captured


def capture_nb02_baselines(product: str) -> dict[str, str]:
    """Capture NB02 forecasting outputs.

    Args:
        product: Product code (e.g., "6Y20B")

    Returns:
        Dictionary mapping output names to file paths
    """
    baseline_dir = get_baseline_dir(product, "nb02")
    baseline_dir.mkdir(parents=True, exist_ok=True)

    output_dir = get_output_dir()
    captured = {}
    outputs_metadata = {}

    # Files to capture from NB02
    results_file = output_dir / "results" / "flexguard_forecasting_results_atomic.csv"
    metrics_file = output_dir / "results" / "flexguard_performance_summary_atomic.json"

    if results_file.exists():
        # Convert CSV to parquet for consistency
        df = pd.read_csv(results_file)
        dest_path = baseline_dir / "forecast_results.parquet"
        df.to_parquet(dest_path, index=False)
        captured["forecast_results.parquet"] = str(dest_path)
        outputs_metadata["forecast_results"] = capture_dataframe_metadata(df)
        print(f"  Captured: forecast_results.parquet ({df.shape[0]} rows, {df.shape[1]} cols)")
    else:
        print(f"  WARNING: {results_file} not found")

    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        dest_path = baseline_dir / "performance_metrics.json"
        with open(dest_path, "w") as f:
            json.dump(metrics, f, indent=2)
        captured["performance_metrics.json"] = str(dest_path)
        outputs_metadata["performance_metrics"] = metrics
        print(f"  Captured: performance_metrics.json")
    else:
        print(f"  WARNING: {metrics_file} not found")

    # Also capture existing baseline files
    for f in baseline_dir.glob("*.parquet"):
        if f.name not in captured:
            df = pd.read_parquet(f)
            outputs_metadata[f.stem] = capture_dataframe_metadata(df)
            captured[f.name] = str(f)
            print(f"  Existing: {f.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

    for f in baseline_dir.glob("*.json"):
        if f.name not in ["capture_metadata.json"] and f.name not in captured:
            with open(f) as file:
                outputs_metadata[f.stem] = json.load(file)
            captured[f.name] = str(f)
            print(f"  Existing: {f.name}")

    # Update metadata
    metadata = create_capture_metadata("nb02_forecasting", product, outputs_metadata)
    metadata_path = baseline_dir / "capture_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    captured["capture_metadata.json"] = str(metadata_path)

    return captured


def capture_all_baselines(product: str) -> dict[str, dict[str, str]]:
    """Capture all notebook baselines for a product.

    Args:
        product: Product code (e.g., "6Y20B")

    Returns:
        Dictionary mapping notebook to captured files
    """
    print(f"\n{'='*60}")
    print(f"Capturing baselines for product: {product}")
    print(f"{'='*60}")

    results = {}

    print(f"\n--- NB00: Data Pipeline ---")
    results["nb00"] = capture_nb00_baselines(product)

    print(f"\n--- NB01: Price Elasticity ---")
    results["nb01"] = capture_nb01_baselines(product)

    print(f"\n--- NB02: Forecasting ---")
    results["nb02"] = capture_nb02_baselines(product)

    return results


def main():
    """Main entry point for baseline capture."""
    parser = argparse.ArgumentParser(
        description="Capture notebook outputs as baselines for equivalence testing"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Capture baselines for all products",
    )
    parser.add_argument(
        "--product",
        choices=["6Y20B", "1Y10B"],
        help="Product to capture baselines for",
    )
    parser.add_argument(
        "--notebook",
        choices=["nb00", "nb01", "nb02"],
        help="Specific notebook to capture (requires --product)",
    )
    args = parser.parse_args()

    if args.all:
        products = ["6Y20B", "1Y10B"]
    elif args.product:
        products = [args.product]
    else:
        parser.error("Either --all or --product is required")

    all_results = {}
    for product in products:
        if args.notebook:
            print(f"\nCapturing {args.notebook} for {product}...")
            capture_func = {
                "nb00": capture_nb00_baselines,
                "nb01": capture_nb01_baselines,
                "nb02": capture_nb02_baselines,
            }[args.notebook]
            all_results[product] = {args.notebook: capture_func(product)}
        else:
            all_results[product] = capture_all_baselines(product)

    # Summary
    print(f"\n{'='*60}")
    print("CAPTURE SUMMARY")
    print(f"{'='*60}")
    for product, notebooks in all_results.items():
        print(f"\n{product}:")
        for notebook, files in notebooks.items():
            print(f"  {notebook}: {len(files)} files captured")


if __name__ == "__main__":
    main()
