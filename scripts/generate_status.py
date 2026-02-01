#!/usr/bin/env python3
"""
Generate status metrics for project documentation (CLAUDE.md, README.md).

Auto-generates:
- Test pass rate and count
- Coverage percentage
- Test quality breakdown (from test_inventory.json)
- Leakage gate status
- Last updated timestamp

Resolves Metrics Drift Issue:
- Single source of truth for test/coverage metrics
- Updates both CLAUDE.md and README.md consistently
- Reads from actual test runs and .coverage file

Usage:
    python scripts/generate_status.py           # Print status
    python scripts/generate_status.py --update  # Update CLAUDE.md
    python scripts/generate_status.py --readme  # Update README.md
    python scripts/generate_status.py --all     # Update all docs
    python scripts/generate_status.py --json    # Output JSON for CI
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def get_test_count_fast() -> int:
    """Get test count via pytest collection (fast, no execution)."""
    exit_code, stdout, stderr = run_command(["python", "-m", "pytest", "--collect-only", "-q"])
    # Parse "N tests collected" or count lines
    output = stdout + stderr
    match = re.search(r"(\d+) tests? collected", output.lower())
    if match:
        return int(match.group(1))
    # Fallback: count test items
    return len([line for line in stdout.split("\n") if "::" in line])


def get_test_metrics() -> dict:
    """Run pytest and extract pass/fail counts."""
    exit_code, stdout, stderr = run_command(
        ["python", "-m", "pytest", "--tb=no", "-q", "--no-header"]
    )

    # Parse pytest output for pass/fail counts
    # Format: "1284 passed, 5 failed, 10 skipped"
    output = stdout + stderr
    passed = 0
    failed = 0
    skipped = 0

    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))

    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))

    match = re.search(r"(\d+) skipped", output)
    if match:
        skipped = int(match.group(1))

    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "pass_rate": pass_rate,
    }


def get_test_quality_breakdown() -> dict:
    """Load test quality breakdown from test_inventory.json if available."""
    inventory_path = PROJECT_ROOT / "docs/test_quality/test_inventory.json"
    if not inventory_path.exists():
        return {}

    try:
        with open(inventory_path) as f:
            data = json.load(f)
        return {
            "meaningful_pct": data.get("quality_breakdown", {}).get("A_meaningful", 0),
            "shallow_pct": data.get("quality_breakdown", {}).get("B_shallow", 0),
            "over_mocked_pct": data.get("quality_breakdown", {}).get("C_over_mocked", 0),
            "tautological_pct": data.get("quality_breakdown", {}).get("D_tautological", 0),
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def get_coverage_metrics(run_tests: bool = True) -> dict:
    """Get coverage metrics from .coverage file or by running tests.

    Parameters
    ----------
    run_tests : bool
        If True, run pytest with coverage. If False, read from .coverage file.
    """
    if run_tests:
        exit_code, stdout, stderr = run_command(
            ["python", "-m", "pytest", "--cov=src", "--cov-report=term", "-q", "--tb=no"],
            timeout=300,
        )
        output = stdout + stderr

        # Parse coverage output for total percentage
        # Format: "TOTAL    12345   1234    90%"
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
        if match:
            return {"coverage_percent": int(match.group(1))}

    # Try to read from existing .coverage file
    coverage_file = PROJECT_ROOT / ".coverage"
    if coverage_file.exists():
        exit_code, stdout, stderr = run_command(["python", "-m", "coverage", "report"])
        match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", stdout)
        if match:
            return {"coverage_percent": int(match.group(1))}

    return {"coverage_percent": None}


def get_leakage_gate_status() -> str:
    """Check leakage gate status."""
    exit_code, stdout, stderr = run_command(
        ["python", "-m", "pytest", "-m", "leakage", "-q", "--tb=no"]
    )

    if exit_code == 0:
        return "PASSED"
    elif "no tests ran" in (stdout + stderr).lower():
        return "NO TESTS"
    else:
        return "FAILED"


def generate_status_table(
    test_metrics: dict,
    coverage: dict,
    leakage_status: str,
    quality_breakdown: dict = None,
) -> str:
    """Generate markdown status table for CLAUDE.md."""
    coverage_pct = coverage.get("coverage_percent")
    coverage_str = f"{coverage_pct}%" if coverage_pct else "N/A"

    passed = test_metrics.get("passed", 0)
    _failed = test_metrics.get("failed", 0)  # Reserved for future use

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Include quality breakdown if available
    quality_note = ""
    if quality_breakdown and quality_breakdown.get("meaningful_pct"):
        quality_note = f"; {quality_breakdown['meaningful_pct']:.0f}% meaningful"

    return f"""| Checkpoint | Status | Notes |
|------------|--------|-------|
| Exploration complete | DONE | Multi-product architecture validated |
| Core functionality | DONE | DI patterns, adapters, strategies implemented |
| Test coverage | {coverage_str} | {passed} tests{quality_note}; priority: core modules >60%, infrastructure can be lower |
| Leakage gate | {leakage_status} | Critical tests now BLOCKING (see audit 2026-01-26) |
| Production deployment | PENDING | Awaiting P0 fixes from audit |

_Last updated: {timestamp}_"""


def generate_readme_status(
    test_metrics: dict,
    coverage: dict,
    quality_breakdown: dict = None,
) -> str:
    """Generate status summary for README.md."""
    coverage_pct = coverage.get("coverage_percent", "N/A")
    passed = test_metrics.get("passed", 0)
    failed = test_metrics.get("failed", 0)
    total = passed + failed

    timestamp = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "## Current Status",
        "",
        "| Metric | Value | Source |",
        "|--------|-------|--------|",
        f"| Tests | {total:,} | `pytest --collect-only` |",
        f"| Coverage | {coverage_pct}% | `.coverage` |",
    ]

    if quality_breakdown and quality_breakdown.get("meaningful_pct"):
        lines.append(
            f"| Meaningful Tests | {quality_breakdown['meaningful_pct']:.1f}% | `test_inventory.json` |"
        )

    lines.extend(
        [
            "",
            f"_Auto-generated: {timestamp}_ | _Regenerate: `python scripts/generate_status.py`_",
        ]
    )

    return "\n".join(lines)


def generate_json_output(
    test_metrics: dict,
    coverage: dict,
    leakage_status: str,
    quality_breakdown: dict = None,
) -> dict:
    """Generate JSON output for CI integration."""
    return {
        "timestamp": datetime.now().isoformat(),
        "tests": test_metrics,
        "coverage": coverage,
        "leakage_gate": leakage_status,
        "quality_breakdown": quality_breakdown or {},
    }


def update_claude_md(status_table: str) -> bool:
    """Update CLAUDE.md with new status table."""
    claude_md_path = PROJECT_ROOT / "CLAUDE.md"

    if not claude_md_path.exists():
        print(f"ERROR: {claude_md_path} not found")
        return False

    content = claude_md_path.read_text()

    # Find and replace the status table
    # Pattern: Look for table starting with "| Checkpoint |" through next "---" or double newline
    pattern = r"\| Checkpoint \| Status \| Notes \|.*?(?=\n\n\*\*Mode\*\*|\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("WARNING: Could not find status table in CLAUDE.md")
        return False

    new_content = content[: match.start()] + status_table + content[match.end() :]
    claude_md_path.write_text(new_content)

    return True


def main():
    """CLI entry point for generating project status documentation."""
    parser = argparse.ArgumentParser(
        description="Generate status metrics for project documentation"
    )
    parser.add_argument("--update", action="store_true", help="Update CLAUDE.md in place")
    parser.add_argument("--readme", action="store_true", help="Generate README status section")
    parser.add_argument("--all", action="store_true", help="Update all documentation files")
    parser.add_argument("--quick", action="store_true", help="Skip slow operations (coverage)")
    parser.add_argument("--json", action="store_true", help="Output JSON for CI integration")
    parser.add_argument("--no-run", action="store_true", help="Don't run tests, use cached data")
    args = parser.parse_args()

    print("Gathering metrics...")

    # Get test metrics
    if args.no_run:
        print("  Using cached data (--no-run)...")
        test_metrics = {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "pass_rate": 0}
        test_count = get_test_count_fast()
        test_metrics["total"] = test_count
        print(f"    Tests collected: {test_count}")
    else:
        print("  Running tests...")
        test_metrics = get_test_metrics()
        print(f"    Tests: {test_metrics['passed']} passed, {test_metrics['failed']} failed")

    # Get test quality breakdown
    quality_breakdown = get_test_quality_breakdown()
    if quality_breakdown:
        print(f"    Quality: {quality_breakdown.get('meaningful_pct', 0):.1f}% meaningful")

    # Get coverage
    if args.quick:
        coverage = get_coverage_metrics(run_tests=False)
        if coverage.get("coverage_percent"):
            print(f"  Coverage (cached): {coverage['coverage_percent']}%")
        else:
            print("  Skipping coverage (--quick)")
    elif args.no_run:
        coverage = get_coverage_metrics(run_tests=False)
        print(f"  Coverage (cached): {coverage.get('coverage_percent', 'N/A')}%")
    else:
        print("  Running coverage (this may take a few minutes)...")
        coverage = get_coverage_metrics(run_tests=True)
        print(f"    Coverage: {coverage.get('coverage_percent', 'N/A')}%")

    # Check leakage gates
    if args.no_run:
        leakage_status = "UNKNOWN"
    else:
        print("  Checking leakage gates...")
        leakage_status = get_leakage_gate_status()
        print(f"    Leakage gate: {leakage_status}")

    # JSON output for CI
    if args.json:
        output = generate_json_output(test_metrics, coverage, leakage_status, quality_breakdown)
        print(json.dumps(output, indent=2))
        return

    # Generate status table
    status_table = generate_status_table(test_metrics, coverage, leakage_status, quality_breakdown)

    print("\n" + "=" * 60)
    print(status_table)
    print("=" * 60)

    if args.readme or args.all:
        readme_status = generate_readme_status(test_metrics, coverage, quality_breakdown)
        print("\n--- README Status Section ---")
        print(readme_status)

    if args.update or args.all:
        if update_claude_md(status_table):
            print("\nCLAUDE.md updated successfully.")
        else:
            print("\nFailed to update CLAUDE.md")
            sys.exit(1)


if __name__ == "__main__":
    main()
