#!/usr/bin/env python3
"""
Equivalence Guard - Mathematical equivalence verification at 1e-12 precision.

Run after EVERY code change to verify mathematical equivalence with baselines.

Usage:
    python scripts/equivalence_guard.py --baseline tests/baselines/pre_refactoring/
    python scripts/equivalence_guard.py --baseline tests/baselines/pre_refactoring/ --current outputs/
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Difference:
    """Record of a difference between baseline and current."""

    location: str
    baseline_value: Any
    current_value: Any
    magnitude: Optional[float] = None
    details: str = ""


@dataclass
class EquivalenceReport:
    """Report of equivalence comparison."""

    status: Literal["PASS", "FAIL", "SKIP"]
    precision: float
    differences: List[Difference] = field(default_factory=list)
    files_compared: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "precision": self.precision,
            "differences": [
                {
                    "location": d.location,
                    "baseline_value": str(d.baseline_value),
                    "current_value": str(d.current_value),
                    "magnitude": d.magnitude,
                    "details": d.details,
                }
                for d in self.differences
            ],
            "files_compared": self.files_compared,
            "message": self.message,
        }


def compare_numeric_values(
    baseline: float, current: float, precision: float = 1e-12
) -> Tuple[bool, Optional[float]]:
    """Compare two numeric values at specified precision.

    Returns
    -------
    Tuple[bool, Optional[float]]
        (is_equal, magnitude_of_difference if not equal)
    """
    if np.isnan(baseline) and np.isnan(current):
        return True, None
    if np.isnan(baseline) or np.isnan(current):
        return False, float("inf")

    diff = abs(baseline - current)
    if diff <= precision:
        return True, None
    return False, diff


def compare_dataframes(
    baseline: pd.DataFrame, current: pd.DataFrame, precision: float = 1e-12
) -> EquivalenceReport:
    """Compare two DataFrames at specified precision.

    Parameters
    ----------
    baseline : pd.DataFrame
        Baseline DataFrame
    current : pd.DataFrame
        Current DataFrame to compare
    precision : float
        Numeric tolerance (default 1e-12)

    Returns
    -------
    EquivalenceReport
        Comparison results
    """
    differences: List[Difference] = []

    # Check shape
    if baseline.shape != current.shape:
        differences.append(
            Difference(
                location="shape",
                baseline_value=baseline.shape,
                current_value=current.shape,
                details=f"Shape mismatch: {baseline.shape} vs {current.shape}",
            )
        )
        return EquivalenceReport(
            status="FAIL", precision=precision, differences=differences
        )

    # Check columns
    if list(baseline.columns) != list(current.columns):
        differences.append(
            Difference(
                location="columns",
                baseline_value=list(baseline.columns),
                current_value=list(current.columns),
                details="Column mismatch",
            )
        )
        return EquivalenceReport(
            status="FAIL", precision=precision, differences=differences
        )

    # Compare numeric columns
    for col in baseline.columns:
        if pd.api.types.is_numeric_dtype(baseline[col]):
            for idx in range(len(baseline)):
                baseline_val = baseline[col].iloc[idx]
                current_val = current[col].iloc[idx]

                is_equal, magnitude = compare_numeric_values(
                    baseline_val, current_val, precision
                )

                if not is_equal:
                    differences.append(
                        Difference(
                            location=f"row {idx}, col '{col}'",
                            baseline_value=baseline_val,
                            current_value=current_val,
                            magnitude=magnitude,
                            details=f"Numeric difference exceeds {precision}",
                        )
                    )

                    # Stop after 10 differences to avoid spam
                    if len(differences) >= 10:
                        differences.append(
                            Difference(
                                location="...",
                                baseline_value="...",
                                current_value="...",
                                details=f"(truncated - more differences exist)",
                            )
                        )
                        return EquivalenceReport(
                            status="FAIL", precision=precision, differences=differences
                        )
        else:
            # Non-numeric: exact comparison
            mismatches = baseline[col] != current[col]
            if mismatches.any():
                first_mismatch = mismatches.idxmax()
                differences.append(
                    Difference(
                        location=f"row {first_mismatch}, col '{col}'",
                        baseline_value=baseline[col].iloc[first_mismatch],
                        current_value=current[col].iloc[first_mismatch],
                        details="Non-numeric value mismatch",
                    )
                )

    status = "PASS" if len(differences) == 0 else "FAIL"
    return EquivalenceReport(status=status, precision=precision, differences=differences)


def compare_json(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    precision: float = 1e-12,
    path: str = "",
) -> List[Difference]:
    """Compare nested JSON/dict structures with numeric tolerance.

    Parameters
    ----------
    baseline : dict
        Baseline dictionary
    current : dict
        Current dictionary
    precision : float
        Numeric tolerance
    path : str
        Current path in nested structure

    Returns
    -------
    List[Difference]
        List of differences found
    """
    differences: List[Difference] = []

    # Check keys
    baseline_keys = set(baseline.keys())
    current_keys = set(current.keys())

    missing_keys = baseline_keys - current_keys
    extra_keys = current_keys - baseline_keys

    for key in missing_keys:
        differences.append(
            Difference(
                location=f"{path}.{key}" if path else key,
                baseline_value=baseline[key],
                current_value="<missing>",
                details="Key missing in current",
            )
        )

    for key in extra_keys:
        differences.append(
            Difference(
                location=f"{path}.{key}" if path else key,
                baseline_value="<missing>",
                current_value=current[key],
                details="Extra key in current",
            )
        )

    # Compare common keys
    for key in baseline_keys & current_keys:
        b_val = baseline[key]
        c_val = current[key]
        key_path = f"{path}.{key}" if path else key

        if isinstance(b_val, dict) and isinstance(c_val, dict):
            differences.extend(compare_json(b_val, c_val, precision, key_path))
        elif isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            is_equal, magnitude = compare_numeric_values(b_val, c_val, precision)
            if not is_equal:
                differences.append(
                    Difference(
                        location=key_path,
                        baseline_value=b_val,
                        current_value=c_val,
                        magnitude=magnitude,
                        details=f"Numeric difference exceeds {precision}",
                    )
                )
        elif isinstance(b_val, list) and isinstance(c_val, list):
            if len(b_val) != len(c_val):
                differences.append(
                    Difference(
                        location=key_path,
                        baseline_value=f"list[{len(b_val)}]",
                        current_value=f"list[{len(c_val)}]",
                        details="List length mismatch",
                    )
                )
            else:
                for i, (b_item, c_item) in enumerate(zip(b_val, c_val)):
                    if isinstance(b_item, dict) and isinstance(c_item, dict):
                        differences.extend(
                            compare_json(b_item, c_item, precision, f"{key_path}[{i}]")
                        )
                    elif b_item != c_item:
                        differences.append(
                            Difference(
                                location=f"{key_path}[{i}]",
                                baseline_value=b_item,
                                current_value=c_item,
                                details="List item mismatch",
                            )
                        )
        elif b_val != c_val:
            differences.append(
                Difference(
                    location=key_path,
                    baseline_value=b_val,
                    current_value=c_val,
                    details="Value mismatch",
                )
            )

    return differences


def run_guard(
    baseline_dir: str,
    current_dir: Optional[str] = None,
    precision: float = 1e-12,
) -> EquivalenceReport:
    """Main entry point for equivalence guard.

    Parameters
    ----------
    baseline_dir : str
        Directory containing baseline outputs
    current_dir : str, optional
        Directory containing current outputs (uses baseline_dir if not provided)
    precision : float
        Numeric tolerance (default 1e-12)

    Returns
    -------
    EquivalenceReport
        Comparison results
    """
    baseline_path = Path(baseline_dir)

    if not baseline_path.exists():
        return EquivalenceReport(
            status="SKIP",
            precision=precision,
            message=f"Baseline directory not found: {baseline_dir}. "
            "Run capture_baselines.py first.",
        )

    current_path = Path(current_dir) if current_dir else baseline_path

    all_differences: List[Difference] = []
    files_compared: List[str] = []

    # Compare JSON files
    for json_file in baseline_path.glob("*.json"):
        if json_file.name == "capture_metadata.json":
            continue

        current_file = current_path / json_file.name
        if not current_file.exists():
            all_differences.append(
                Difference(
                    location=str(json_file.name),
                    baseline_value="<file exists>",
                    current_value="<missing>",
                    details="File missing in current outputs",
                )
            )
            continue

        with open(json_file) as f:
            baseline_data = json.load(f)
        with open(current_file) as f:
            current_data = json.load(f)

        diffs = compare_json(baseline_data, current_data, precision)
        for diff in diffs:
            diff.location = f"{json_file.name}:{diff.location}"
        all_differences.extend(diffs)
        files_compared.append(json_file.name)

    # Compare parquet files
    for pq_file in baseline_path.glob("*.parquet"):
        current_file = current_path / pq_file.name
        if not current_file.exists():
            all_differences.append(
                Difference(
                    location=str(pq_file.name),
                    baseline_value="<file exists>",
                    current_value="<missing>",
                    details="File missing in current outputs",
                )
            )
            continue

        baseline_df = pd.read_parquet(pq_file)
        current_df = pd.read_parquet(current_file)

        report = compare_dataframes(baseline_df, current_df, precision)
        for diff in report.differences:
            diff.location = f"{pq_file.name}:{diff.location}"
        all_differences.extend(report.differences)
        files_compared.append(pq_file.name)

    if not files_compared:
        return EquivalenceReport(
            status="SKIP",
            precision=precision,
            message=f"No baseline files found in {baseline_dir}",
        )

    status = "PASS" if len(all_differences) == 0 else "FAIL"
    return EquivalenceReport(
        status=status,
        precision=precision,
        differences=all_differences,
        files_compared=files_compared,
        message=f"Compared {len(files_compared)} files at precision {precision}",
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Verify mathematical equivalence at 1e-12 precision."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="tests/baselines/pre_refactoring/",
        help="Directory containing baseline outputs",
    )
    parser.add_argument(
        "--current",
        type=str,
        default=None,
        help="Directory containing current outputs (optional)",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=1e-12,
        help="Numeric tolerance (default: 1e-12)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    report = run_guard(args.baseline, args.current, args.precision)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"EQUIVALENCE GUARD - Precision: {args.precision}")
        print(f"{'='*60}")
        print(f"Status: {report.status}")
        print(f"Message: {report.message}")

        if report.files_compared:
            print(f"\nFiles compared: {', '.join(report.files_compared)}")

        if report.differences:
            print(f"\nDifferences ({len(report.differences)}):")
            for diff in report.differences[:10]:
                print(f"  - {diff.location}")
                print(f"    Baseline: {diff.baseline_value}")
                print(f"    Current:  {diff.current_value}")
                if diff.magnitude:
                    print(f"    Magnitude: {diff.magnitude}")
                print(f"    Details: {diff.details}")

        print(f"{'='*60}\n")

    sys.exit(0 if report.status in ("PASS", "SKIP") else 1)


if __name__ == "__main__":
    main()
