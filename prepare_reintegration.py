#!/usr/bin/env python3
"""
Reintegration Preparation Script

Generates comprehensive report for safe reintegration back to AWS environment.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd):
    """Run shell command and capture output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def check_test_suite():
    """Run full test suite and capture results."""
    print("Running full test suite...")
    code, stdout, stderr = run_command("pytest -v --tb=short -q")

    # Parse pytest output for summary
    passed = 0
    failed = 0
    total = 0

    lines = (stdout + stderr).split("\n")
    for line in lines:
        # Look for pytest summary line like "2500 passed in 120.45s"
        if "passed" in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if "passed" in part.lower() and i > 0:
                    try:
                        passed = int(parts[i - 1])
                        total = passed
                    except:
                        pass
        if "failed" in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if "failed" in part.lower() and i > 0:
                    try:
                        failed = int(parts[i - 1])
                        total += failed
                    except:
                        pass

    return {
        "total_tests": total if total > 0 else "unknown",
        "passed": passed if passed > 0 else "unknown",
        "failed": failed,
        "exit_code": code,
        "all_passed": code == 0
    }


def check_equivalence():
    """Run mathematical equivalence validation."""
    print("Validating mathematical equivalence...")
    code, stdout, stderr = run_command("python validate_equivalence.py")

    # Load validation report
    report_path = Path("validation_report.json")
    if report_path.exists():
        try:
            with open(report_path) as f:
                report = json.load(f)
            return report
        except Exception as e:
            return {"error": f"Could not load validation report: {e}"}
    else:
        return {
            "error": "Validation report not found",
            "exit_code": code
        }


def check_performance():
    """Run performance baseline tests."""
    print("Checking performance baselines...")
    code, stdout, stderr = run_command("pytest tests/performance/ -m 'not slow' -v --tb=short -q")

    return {
        "passed": code == 0,
        "exit_code": code
    }


def check_changelog():
    """Check if changelog has been updated."""
    changelog_path = Path("CHANGELOG_REFACTORING.md")

    if not changelog_path.exists():
        return {
            "exists": False,
            "updated": False,
            "warning": "CHANGELOG_REFACTORING.md not found"
        }

    with open(changelog_path) as f:
        content = f.read()

    # Check if changelog has been modified from template
    has_changes = "## Changes Made" in content and len(content) > 500

    return {
        "exists": True,
        "updated": has_changes,
        "warning": None if has_changes else "Changelog appears to be template only"
    }


def generate_report():
    """Generate comprehensive reintegration report."""

    print("=" * 80)
    print("REINTEGRATION PREPARATION")
    print("=" * 80)
    print()

    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "checks": {}
    }

    # Run checks
    print("\n" + "=" * 80)
    print("RUNNING VALIDATION CHECKS")
    print("=" * 80)
    print()

    report["checks"]["test_suite"] = check_test_suite()
    print()

    report["checks"]["equivalence"] = check_equivalence()
    print()

    report["checks"]["performance"] = check_performance()
    print()

    print("Checking changelog...")
    report["checks"]["changelog"] = check_changelog()
    print("  Changelog check complete")
    print()

    # Determine readiness
    test_passed = report["checks"]["test_suite"].get("all_passed", False)
    equiv_passed = report["checks"]["equivalence"].get("ready_for_reintegration", False)
    perf_passed = report["checks"]["performance"]["passed"]
    changelog_updated = report["checks"]["changelog"].get("updated", False)

    report["ready_for_reintegration"] = (
        test_passed and equiv_passed and perf_passed and changelog_updated
    )

    # Print summary
    print("=" * 80)
    print("REINTEGRATION READINESS REPORT")
    print("=" * 80)
    print()

    # Test Suite
    print(f"Test Suite: {'[PASS] PASSED' if test_passed else '[FAIL] FAILED'}")
    if report['checks']['test_suite']['total_tests'] != "unknown":
        print(f"  - Tests run: {report['checks']['test_suite']['total_tests']}")
        print(f"  - Tests passed: {report['checks']['test_suite']['passed']}")
        if report['checks']['test_suite']['failed'] > 0:
            print(f"  - Tests failed: {report['checks']['test_suite']['failed']}")
    else:
        print(f"  - Could not parse test results (exit code: {report['checks']['test_suite']['exit_code']})")
    print()

    # Mathematical Equivalence
    print(f"Mathematical Equivalence: {'[PASS] PASSED' if equiv_passed else '[FAIL] FAILED'}")
    if "checks" in report["checks"]["equivalence"]:
        for check, status in report["checks"]["equivalence"]["checks"].items():
            symbol = "[PASS]" if status == "PASSED" else "[FAIL]"
            print(f"  {symbol} {check.replace('_', ' ').title()}: {status}")
    elif "error" in report["checks"]["equivalence"]:
        print(f"  [FAIL] Error: {report['checks']['equivalence']['error']}")
    print()

    # Performance
    print(f"Performance Baselines: {'[PASS] PASSED' if perf_passed else '[FAIL] FAILED'}")
    print()

    # Changelog
    changelog_status = report["checks"]["changelog"]
    print(f"Changelog: {'[PASS] UPDATED' if changelog_updated else '[WARN] NOT UPDATED'}")
    if changelog_status.get("warning"):
        print(f"  Warning: {changelog_status['warning']}")
    print()

    # Final verdict
    print("=" * 80)
    if report["ready_for_reintegration"]:
        print("[PASS] READY FOR REINTEGRATION")
        print()
        print("Next steps:")
        print("1. Review CHANGELOG_REFACTORING.md")
        print("2. Run: ./create_reintegration_package.sh")
        print("3. Transfer package to AWS environment")
        print("4. Follow REINTEGRATION_GUIDE.md")
        exit_code = 0
    else:
        print("[FAIL] NOT READY FOR REINTEGRATION")
        print()
        print("Issues found:")
        if not test_passed:
            print("  - Test suite has failures")
        if not equiv_passed:
            print("  - Mathematical equivalence broken or not validated")
        if not perf_passed:
            print("  - Performance regressions detected")
        if not changelog_updated:
            print("  - Changelog not updated")
        print()
        print("Fix issues above before reintegration.")
        exit_code = 1

    # Save report
    report_path = Path("reintegration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Full report saved to: {report_path}")
    print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(generate_report())
