#!/usr/bin/env python3
"""
Hardcode Scanner - Find product-specific hardcoded strings.

Identifies hardcoded company names, product codes, and column references
that should be configuration-driven for multi-product support.

Usage:
    python scripts/hardcode_scanner.py --path src/models/
    python scripts/hardcode_scanner.py --path src/ --output hardcodes.json
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Company names that should be config-driven
COMPANY_NAMES = [
    "Prudential",
    "Fidelity",
    "Lincoln",
    "Nationwide",
    "Allianz",
    "Athene",
    "AIG",
    "Corebridge",
    "Equitable",
    "Jackson",
    "Principal",
    "Transamerica",
]

# Product codes
PRODUCT_CODES = [
    "6Y20B",
    "6Y10B",
    "10Y20B",
    "1Y10B",
    "RILA",
    "FIA",
    "MYGA",
]


@dataclass
class HardcodeFinding:
    """A hardcoded string finding."""

    file_path: str
    line_number: int
    matched_value: str
    category: str  # company_name, product_code, column_access
    line_content: str
    suggested_fix: str = ""


@dataclass
class HardcodeReport:
    """Summary of hardcode findings."""

    findings: List[HardcodeFinding] = field(default_factory=list)
    files_scanned: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        by_category: Dict[str, List] = {}
        for f in self.findings:
            by_category.setdefault(f.category, []).append(
                {
                    "file": f.file_path,
                    "line": f.line_number,
                    "value": f.matched_value,
                    "content": f.line_content.strip(),
                    "suggested_fix": f.suggested_fix,
                }
            )

        return {
            "summary": {
                "files_scanned": self.files_scanned,
                "total_findings": len(self.findings),
                "by_category": {k: len(v) for k, v in by_category.items()},
            },
            "findings": by_category,
        }


def scan_file(file_path: Path, base_path: Path) -> List[HardcodeFinding]:
    """Scan a single file for hardcoded patterns.

    Parameters
    ----------
    file_path : Path
        File to scan
    base_path : Path
        Base path for relative paths

    Returns
    -------
    List[HardcodeFinding]
        Findings in this file
    """
    findings: List[HardcodeFinding] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IOError):
        return findings

    lines = content.split("\n")
    relative_path = str(file_path.relative_to(base_path))

    # Skip test files and fixtures
    if "test" in relative_path.lower() or "fixture" in relative_path.lower():
        return findings

    # Skip config files where hardcoding is expected
    if "config" in relative_path.lower() and "builder" not in relative_path.lower():
        return findings

    for line_num, line in enumerate(lines):
        # Skip comments and docstrings (simple heuristic)
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Check company names in code
        for company in COMPANY_NAMES:
            # Pattern: df["Company"] or df['Company'] or 'Company' in assignment
            patterns = [
                rf'\bdf\[["\']' + company + rf'["\']\]',  # df["Prudential"]
                rf'\bdf_\w+\[["\']' + company + rf'["\']\]',  # df_rates["Prudential"]
                rf'["\']' + company + rf'["\']',  # String literal (broader)
            ]

            for pattern in patterns:
                if re.search(pattern, line):
                    # Check if it's not in a config definition
                    if "=" in line and ":" not in line:
                        # Assignment context - likely hardcoded
                        findings.append(
                            HardcodeFinding(
                                file_path=relative_path,
                                line_number=line_num + 1,
                                matched_value=company,
                                category="company_name",
                                line_content=line,
                                suggested_fix=f"Use config.own_rate_column or similar config field",
                            )
                        )
                        break

        # Check product codes in logic (not in config)
        for code in PRODUCT_CODES:
            # Look for product codes in conditionals or comparisons
            if re.search(rf'==\s*["\']' + code + rf'["\']', line):
                findings.append(
                    HardcodeFinding(
                        file_path=relative_path,
                        line_number=line_num + 1,
                        matched_value=code,
                        category="product_code",
                        line_content=line,
                        suggested_fix="Use config.product_code comparison or product registry lookup",
                    )
                )

        # Check for column access patterns that hardcode company names
        column_pattern = r'df(?:_\w+)?\[["\']([A-Z][a-z]+)["\']\]'
        matches = re.finditer(column_pattern, line)
        for match in matches:
            col_name = match.group(1)
            if col_name in COMPANY_NAMES:
                findings.append(
                    HardcodeFinding(
                        file_path=relative_path,
                        line_number=line_num + 1,
                        matched_value=f'df["{col_name}"]',
                        category="column_access",
                        line_content=line,
                        suggested_fix=f"Use df[config.own_rate_column] instead",
                    )
                )

    return findings


def scan_directory(
    path: str,
    exclude: Optional[List[str]] = None,
) -> HardcodeReport:
    """Scan directory for hardcoded patterns.

    Parameters
    ----------
    path : str
        Directory to scan
    exclude : List[str], optional
        Patterns to exclude

    Returns
    -------
    HardcodeReport
        Scan results
    """
    base_path = Path(path).resolve()
    if not base_path.exists():
        return HardcodeReport()

    exclude_patterns = exclude or ["__pycache__", ".git", "venv", "test"]

    report = HardcodeReport()

    for py_file in base_path.rglob("*.py"):
        # Check exclusions
        relative_path = str(py_file.relative_to(base_path))
        if any(excl in relative_path for excl in exclude_patterns):
            continue

        report.files_scanned += 1
        findings = scan_file(py_file, base_path)
        report.findings.extend(findings)

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Find product-specific hardcoded strings that should be config-driven."
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
        default=["__pycache__", ".git", "venv"],
        help="Patterns to exclude",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (optional)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON to stdout",
    )

    args = parser.parse_args()

    report = scan_directory(args.path, args.exclude)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Report written to {args.output}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.output:
        print(f"\n{'='*60}")
        print("HARDCODE SCANNER - Finding config-worthy hardcodes")
        print(f"{'='*60}")
        print(f"Files scanned: {report.files_scanned}")
        print(f"Total findings: {len(report.findings)}")

        if report.findings:
            # Group by category
            by_category: Dict[str, List[HardcodeFinding]] = {}
            for finding in report.findings:
                by_category.setdefault(finding.category, []).append(finding)

            for category, findings in by_category.items():
                print(f"\n[{category.upper()}] ({len(findings)}):")
                for finding in findings[:5]:
                    print(f"  {finding.file_path}:{finding.line_number}")
                    print(f"    Value: {finding.matched_value}")
                    print(f"    Suggested: {finding.suggested_fix}")
                if len(findings) > 5:
                    print(f"  ... and {len(findings) - 5} more")

        print(f"{'='*60}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
