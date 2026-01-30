#!/usr/bin/env python3
"""
Mathematical Equivalence Validation Script

Validates that refactored code maintains 1e-12 precision equivalence
with original implementation across all pipeline stages.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pytest


def run_validation():
    """Run comprehensive mathematical equivalence validation."""

    print("=" * 80)
    print("MATHEMATICAL EQUIVALENCE VALIDATION")
    print("=" * 80)
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    # Stage-by-stage validation
    print("1. Stage-by-Stage Validation (1e-12 precision)")
    print("-" * 80)

    stage_result = pytest.main([
        "tests/integration/test_pipeline_stage_equivalence.py",
        "-v",
        "--tb=short"
    ])

    results["checks"]["stage_by_stage"] = "PASSED" if stage_result == 0 else "FAILED"
    print()

    # Bootstrap statistical validation
    print("2. Bootstrap Statistical Validation (CV < 5%)")
    print("-" * 80)

    bootstrap_result = pytest.main([
        "tests/integration/test_bootstrap_statistical_equivalence.py",
        "-v",
        "--tb=short"
    ])

    results["checks"]["bootstrap_statistical"] = "PASSED" if bootstrap_result == 0 else "FAILED"
    print()

    # End-to-end validation
    print("3. End-to-End Pipeline Validation")
    print("-" * 80)

    e2e_result = pytest.main([
        "tests/e2e/test_full_pipeline_offline.py",
        "-v",
        "--tb=short"
    ])

    results["checks"]["e2e_pipeline"] = "PASSED" if e2e_result == 0 else "FAILED"
    print()

    # Economic constraints
    print("4. Economic Constraint Validation")
    print("-" * 80)

    constraint_result = pytest.main([
        "tests/property_based/test_economic_constraints.py",
        "-v",
        "--tb=short"
    ])

    results["checks"]["economic_constraints"] = "PASSED" if constraint_result == 0 else "FAILED"
    print()

    # Final summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_passed = all(v == "PASSED" for v in results["checks"].values())

    for check, status in results["checks"].items():
        symbol = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"{symbol} {check.replace('_', ' ').title()}: {status}")

    print()

    if all_passed:
        print("RESULT: Mathematical Equivalence MAINTAINED [PASS]")
        print("Your refactored code is ready for reintegration.")
        results["ready_for_reintegration"] = True
        exit_code = 0
    else:
        print("RESULT: Mathematical Equivalence BROKEN [FAIL]")
        print("Review failures above before reintegration.")
        results["ready_for_reintegration"] = False
        exit_code = 1

    # Save results
    with open("validation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Detailed report saved to: validation_report.json")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(run_validation())
