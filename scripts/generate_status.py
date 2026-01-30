#!/usr/bin/env python3
"""
Generate status metrics for CLAUDE.md header.

Auto-generates:
- Test pass rate
- Coverage percentage
- Leakage gate status
- Last updated timestamp

Usage:
    python scripts/generate_status.py           # Print status
    python scripts/generate_status.py --update  # Update CLAUDE.md
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


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


def get_coverage_metrics() -> dict:
    """Run coverage and extract percentage."""
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

    return {"coverage_percent": None}


def get_leakage_gate_status() -> str:
    """Check leakage gate status."""
    exit_code, stdout, stderr = run_command(
        ["python", "-m", "pytest", "tests/test_leakage_gates.py", "-v", "--tb=short"]
    )

    if exit_code == 0:
        return "PASSED"
    elif "no tests ran" in (stdout + stderr).lower():
        return "NO TESTS"
    else:
        return "FAILED"


def generate_status_table(test_metrics: dict, coverage: dict, leakage_status: str) -> str:
    """Generate markdown status table."""
    coverage_pct = coverage.get("coverage_percent")
    coverage_str = f"{coverage_pct}%" if coverage_pct else "N/A"

    pass_rate = test_metrics.get("pass_rate", 0)
    passed = test_metrics.get("passed", 0)
    failed = test_metrics.get("failed", 0)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    return f"""| Checkpoint | Status | Notes |
|------------|--------|-------|
| Exploration complete | DONE | Multi-product architecture validated |
| Core functionality | DONE | DI patterns, adapters, strategies implemented |
| Test coverage | {coverage_str} | {passed} tests; priority: core modules >60%, infrastructure can be lower |
| Leakage gate | {leakage_status} | Critical tests now BLOCKING (see audit 2026-01-26) |
| Production deployment | PENDING | Awaiting P0 fixes from audit |

_Last updated: {timestamp}_"""


def update_claude_md(status_table: str) -> bool:
    """Update CLAUDE.md with new status table."""
    claude_md_path = Path(__file__).parent.parent / "CLAUDE.md"

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
    parser = argparse.ArgumentParser(description="Generate status metrics for CLAUDE.md")
    parser.add_argument("--update", action="store_true", help="Update CLAUDE.md in place")
    parser.add_argument("--quick", action="store_true", help="Skip slow operations (coverage)")
    args = parser.parse_args()

    print("Gathering metrics...")

    # Always run test metrics
    print("  Running tests...")
    test_metrics = get_test_metrics()
    print(f"    Tests: {test_metrics['passed']} passed, {test_metrics['failed']} failed")

    # Skip coverage if --quick
    if args.quick:
        coverage = {"coverage_percent": None}
        print("  Skipping coverage (--quick)")
    else:
        print("  Running coverage (this may take a few minutes)...")
        coverage = get_coverage_metrics()
        print(f"    Coverage: {coverage.get('coverage_percent', 'N/A')}%")

    # Check leakage gates
    print("  Checking leakage gates...")
    leakage_status = get_leakage_gate_status()
    print(f"    Leakage gate: {leakage_status}")

    # Generate status table
    status_table = generate_status_table(test_metrics, coverage, leakage_status)

    print("\n" + "=" * 60)
    print(status_table)
    print("=" * 60)

    if args.update:
        if update_claude_md(status_table):
            print("\nCLAUDE.md updated successfully.")
        else:
            print("\nFailed to update CLAUDE.md")
            sys.exit(1)


if __name__ == "__main__":
    main()
