#!/usr/bin/env python3
"""
Stub Hunter - Find placeholder implementations that should be wired.

Scans source files for patterns indicating stub implementations.

Usage:
    python scripts/stub_hunter.py --path src/
    python scripts/stub_hunter.py --path src/ --allowlist scripts/stub_allowlist.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# Patterns indicating stub implementations
STUB_PATTERNS = {
    "empty_return_list": r"return\s+\[\]",
    "empty_return_dict": r"return\s+\{\}",
    "zero_numeric_return": r"return\s+0\.0",
    "none_return": r"return\s+None(?:\s*#|$|\s)",
    "placeholder_comment": r"#\s*[Pp]laceholder",
    "todo_marker": r"#\s*TODO",
    "would_call_comment": r"#\s*[Ww]ould\s+call",
    "actual_impl_comment": r"#\s*[Aa]ctual\s+implementation",
    "not_implemented": r"raise\s+NotImplementedError",
}


@dataclass
class StubFinding:
    """A potential stub implementation found in code."""

    file_path: str
    line_number: int
    pattern_name: str
    line_content: str
    context: str = ""
    severity: str = "warning"  # warning, info, error


@dataclass
class StubReport:
    """Summary report of stub findings."""

    findings: List[StubFinding] = field(default_factory=list)
    files_scanned: int = 0
    files_with_stubs: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "files_scanned": self.files_scanned,
                "files_with_stubs": len(self.files_with_stubs),
                "total_findings": len(self.findings),
            },
            "findings": [
                {
                    "file": f.file_path,
                    "line": f.line_number,
                    "pattern": f.pattern_name,
                    "content": f.line_content.strip(),
                    "severity": f.severity,
                }
                for f in self.findings
            ],
        }


def load_allowlist(allowlist_path: Optional[str]) -> Dict[str, Dict]:
    """Load allowlist of intentional stubs.

    Parameters
    ----------
    allowlist_path : str, optional
        Path to allowlist JSON file

    Returns
    -------
    Dict[str, Dict]
        Allowlist keyed by file path
    """
    if not allowlist_path:
        return {}

    path = Path(allowlist_path)
    if not path.exists():
        return {}

    with open(path) as f:
        data = json.load(f)

    # Remove _comment key if present
    return {k: v for k, v in data.items() if not k.startswith("_")}


def is_allowed(
    filepath: str, pattern_name: str, allowlist: Dict[str, Dict]
) -> bool:
    """Check if this stub is intentionally allowed.

    Parameters
    ----------
    filepath : str
        Relative file path
    pattern_name : str
        Pattern that matched
    allowlist : Dict
        Allowlist dictionary

    Returns
    -------
    bool
        True if allowed
    """
    # Normalize path
    normalized = filepath.replace("\\", "/")

    for allowed_path, config in allowlist.items():
        if allowed_path in normalized:
            patterns = config.get("patterns", [])
            # Check if pattern matches any allowed pattern
            for allowed_pattern in patterns:
                if allowed_pattern.lower() in pattern_name.lower():
                    return True
    return False


def get_context_lines(lines: List[str], line_num: int, context: int = 2) -> str:
    """Get surrounding context lines.

    Parameters
    ----------
    lines : List[str]
        All lines in file
    line_num : int
        Target line number (0-indexed)
    context : int
        Number of lines before/after

    Returns
    -------
    str
        Context string
    """
    start = max(0, line_num - context)
    end = min(len(lines), line_num + context + 1)

    context_lines = []
    for i in range(start, end):
        prefix = ">>> " if i == line_num else "    "
        context_lines.append(f"{prefix}{i+1:4d}: {lines[i].rstrip()}")

    return "\n".join(context_lines)


def scan_file(
    file_path: Path, allowlist: Dict[str, Dict], base_path: Path
) -> List[StubFinding]:
    """Scan a single file for stub patterns.

    Parameters
    ----------
    file_path : Path
        File to scan
    allowlist : Dict
        Allowlist of intentional stubs
    base_path : Path
        Base path for relative paths

    Returns
    -------
    List[StubFinding]
        Findings in this file
    """
    findings: List[StubFinding] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IOError):
        return findings

    lines = content.split("\n")
    relative_path = str(file_path.relative_to(base_path))

    for line_num, line in enumerate(lines):
        for pattern_name, pattern in STUB_PATTERNS.items():
            if re.search(pattern, line):
                # Check allowlist
                if is_allowed(relative_path, pattern_name, allowlist):
                    continue

                # Determine severity
                severity = "warning"
                if pattern_name in ("placeholder_comment", "would_call_comment"):
                    severity = "error"  # These definitely indicate stubs
                elif pattern_name == "not_implemented":
                    severity = "info"  # Explicit stubs are OK if allowed

                findings.append(
                    StubFinding(
                        file_path=relative_path,
                        line_number=line_num + 1,
                        pattern_name=pattern_name,
                        line_content=line,
                        context=get_context_lines(lines, line_num),
                        severity=severity,
                    )
                )

    return findings


def scan_directory(
    path: str,
    exclude: Optional[List[str]] = None,
    allowlist_path: Optional[str] = None,
) -> StubReport:
    """Scan directory for stub implementations.

    Parameters
    ----------
    path : str
        Directory to scan
    exclude : List[str], optional
        Patterns to exclude
    allowlist_path : str, optional
        Path to allowlist JSON

    Returns
    -------
    StubReport
        Scan results
    """
    base_path = Path(path).resolve()
    if not base_path.exists():
        return StubReport()

    exclude_patterns = exclude or ["test", "__pycache__", ".git", "venv"]
    allowlist = load_allowlist(allowlist_path)

    report = StubReport()
    files_with_stubs: Set[str] = set()

    for py_file in base_path.rglob("*.py"):
        # Check exclusions
        relative_path = str(py_file.relative_to(base_path))
        if any(excl in relative_path for excl in exclude_patterns):
            continue

        report.files_scanned += 1
        findings = scan_file(py_file, allowlist, base_path)

        if findings:
            files_with_stubs.add(relative_path)
            report.findings.extend(findings)

    report.files_with_stubs = files_with_stubs
    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Find placeholder implementations that should be wired."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="src/",
        help="Directory to scan",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["test", "__pycache__", ".git", "venv"],
        help="Patterns to exclude",
    )
    parser.add_argument(
        "--allowlist",
        type=str,
        default=None,
        help="Path to allowlist JSON",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error code if error-severity stubs found",
    )

    args = parser.parse_args()

    report = scan_directory(args.path, args.exclude, args.allowlist)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print("STUB HUNTER - Finding placeholder implementations")
        print(f"{'='*60}")
        print(f"Files scanned: {report.files_scanned}")
        print(f"Files with stubs: {len(report.files_with_stubs)}")
        print(f"Total findings: {len(report.findings)}")

        if report.findings:
            # Group by severity
            errors = [f for f in report.findings if f.severity == "error"]
            warnings = [f for f in report.findings if f.severity == "warning"]
            infos = [f for f in report.findings if f.severity == "info"]

            if errors:
                print(f"\n[ERRORS] ({len(errors)}):")
                for finding in errors:
                    print(f"  {finding.file_path}:{finding.line_number}")
                    print(f"    Pattern: {finding.pattern_name}")
                    print(f"    Content: {finding.line_content.strip()}")

            if warnings:
                print(f"\n[WARNINGS] ({len(warnings)}):")
                for finding in warnings[:10]:  # Limit output
                    print(f"  {finding.file_path}:{finding.line_number}")
                    print(f"    Pattern: {finding.pattern_name}")
                if len(warnings) > 10:
                    print(f"  ... and {len(warnings) - 10} more warnings")

            if infos:
                print(f"\n[INFO] ({len(infos)}):")
                for finding in infos[:5]:
                    print(f"  {finding.file_path}:{finding.line_number}")
                    print(f"    Pattern: {finding.pattern_name}")

        print(f"{'='*60}\n")

    # Exit code
    if args.fail_on_error:
        errors = [f for f in report.findings if f.severity == "error"]
        sys.exit(1 if errors else 0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
