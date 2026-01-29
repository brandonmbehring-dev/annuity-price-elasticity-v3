#!/usr/bin/env python3
"""
Column Lineage Tracker for Annuity Price Elasticity v2.

Analyzes how columns flow through the pipeline:
- Source columns from data files
- Transformations applied
- Derived columns created
- Where each column is used

Usage:
    # Analyze column usage in a file
    python scripts/column_lineage.py analyze src/features/

    # Track specific column
    python scripts/column_lineage.py track "prudential_cap"

    # Generate lineage report
    python scripts/column_lineage.py report --output lineage_report.md
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnUsage:
    """Single column usage instance."""

    column_name: str
    file_path: str
    line_number: int
    usage_type: str  # 'read', 'write', 'transform', 'filter'
    context: str  # Code snippet


@dataclass
class ColumnLineage:
    """Complete lineage for a column."""

    column_name: str
    sources: List[ColumnUsage] = field(default_factory=list)
    transformations: List[ColumnUsage] = field(default_factory=list)
    consumers: List[ColumnUsage] = field(default_factory=list)
    derived_from: Set[str] = field(default_factory=set)
    derives: Set[str] = field(default_factory=set)


@dataclass
class LineageReport:
    """Complete lineage report for codebase."""

    columns: Dict[str, ColumnLineage] = field(default_factory=dict)
    files_analyzed: int = 0

    def get_orphan_columns(self) -> List[str]:
        """Columns that are written but never read."""
        orphans = []
        for name, lineage in self.columns.items():
            if lineage.sources and not lineage.consumers:
                orphans.append(name)
        return orphans

    def get_undefined_columns(self) -> List[str]:
        """Columns that are read but never defined."""
        undefined = []
        for name, lineage in self.columns.items():
            if lineage.consumers and not lineage.sources:
                undefined.append(name)
        return undefined


# =============================================================================
# COLUMN PATTERNS
# =============================================================================

# Known rate column patterns
RATE_COLUMN_PATTERNS = [
    r"prudential",
    r"pru_cap",
    r"P_lag_\d+",
    r"C_lag_\d+",
    r"competitor",
    r"DGS\d+",
    r"rate",
    r"cap",
]

# DataFrame column access patterns
COLUMN_ACCESS_PATTERNS = [
    # df['column']
    r'(\w+)\[[\'"]([\w_]+)[\'"]\]',
    # df.column
    r'(\w+)\.([\w_]+)',
    # df[["col1", "col2"]]
    r'\[\[[\'"]([\w_]+)[\'"]',
]

# Assignment patterns
ASSIGNMENT_PATTERNS = [
    # df['new_col'] = ...
    r'(\w+)\[[\'"]([\w_]+)[\'"]\]\s*=',
    # df.assign(new_col=...)
    r'\.assign\(([\w_]+)\s*=',
]


# =============================================================================
# AST ANALYSIS
# =============================================================================

class ColumnVisitor(ast.NodeVisitor):
    """AST visitor to find column usages."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.usages: List[ColumnUsage] = []
        self.current_function = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Subscript(self, node: ast.Subscript):
        """Detect df['column'] patterns."""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            col_name = node.slice.value

            # Determine if it's a read or write based on context
            usage_type = "read"
            if isinstance(node.ctx, ast.Store):
                usage_type = "write"

            self.usages.append(ColumnUsage(
                column_name=col_name,
                file_path=self.file_path,
                line_number=node.lineno,
                usage_type=usage_type,
                context=self.current_function or "module",
            ))

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect method calls that use column names."""
        # Check for .assign() calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
            for keyword in node.keywords:
                self.usages.append(ColumnUsage(
                    column_name=keyword.arg,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    usage_type="write",
                    context=self.current_function or "module",
                ))

        # Check for column names in various DataFrame methods
        method_names = ["drop", "rename", "groupby", "sort_values", "merge"]
        if isinstance(node.func, ast.Attribute) and node.func.attr in method_names:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.usages.append(ColumnUsage(
                        column_name=arg.value,
                        file_path=self.file_path,
                        line_number=node.lineno,
                        usage_type="read",
                        context=self.current_function or "module",
                    ))
                elif isinstance(arg, ast.List):
                    for elt in arg.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.usages.append(ColumnUsage(
                                column_name=elt.value,
                                file_path=self.file_path,
                                line_number=node.lineno,
                                usage_type="read",
                                context=self.current_function or "module",
                            ))

        self.generic_visit(node)


# =============================================================================
# REGEX-BASED ANALYSIS
# =============================================================================

def extract_columns_regex(file_path: Path) -> List[ColumnUsage]:
    """Extract column usages using regex (catches things AST misses)."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    usages = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith("#"):
            continue

        # Check for string column references
        for pattern in COLUMN_ACCESS_PATTERNS:
            for match in re.finditer(pattern, line):
                col_name = match.group(2) if len(match.groups()) > 1 else match.group(1)
                # Filter to likely column names
                if _is_likely_column(col_name):
                    usage_type = "read"
                    if "=" in line and line.index("=") > match.start():
                        usage_type = "write"

                    usages.append(ColumnUsage(
                        column_name=col_name,
                        file_path=str(file_path),
                        line_number=line_num,
                        usage_type=usage_type,
                        context=line.strip()[:80],
                    ))

    return usages


def _is_likely_column(name: str) -> bool:
    """Heuristic to filter out non-column names."""
    # Skip short names and common non-columns
    if len(name) < 2:
        return False
    if name in {"df", "pd", "np", "self", "cls", "args", "kwargs", "data", "result", "value", "key", "item"}:
        return False
    # Skip method names
    if name.startswith("_") or name.endswith("_"):
        return False
    # Check for rate column patterns
    for pattern in RATE_COLUMN_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            return True
    # Check for snake_case names (likely columns)
    if "_" in name:
        return True
    return False


# =============================================================================
# LINEAGE BUILDING
# =============================================================================

def analyze_file(file_path: Path) -> List[ColumnUsage]:
    """Analyze a single file for column usages."""
    usages = []

    # AST analysis
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        visitor = ColumnVisitor(str(file_path))
        visitor.visit(tree)
        usages.extend(visitor.usages)
    except (SyntaxError, UnicodeDecodeError):
        pass

    # Regex analysis (catches additional patterns)
    usages.extend(extract_columns_regex(file_path))

    # Deduplicate
    seen = set()
    unique = []
    for usage in usages:
        key = (usage.column_name, usage.file_path, usage.line_number, usage.usage_type)
        if key not in seen:
            seen.add(key)
            unique.append(usage)

    return unique


def build_lineage(base_path: Path, dirs: List[str] = None) -> LineageReport:
    """Build complete lineage report."""
    if dirs is None:
        dirs = ["src/", "notebooks/"]

    report = LineageReport()

    for dir_name in dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            continue

        for file_path in dir_path.rglob("*.py"):
            if "__pycache__" in str(file_path):
                continue

            usages = analyze_file(file_path)
            report.files_analyzed += 1

            for usage in usages:
                if usage.column_name not in report.columns:
                    report.columns[usage.column_name] = ColumnLineage(usage.column_name)

                lineage = report.columns[usage.column_name]

                if usage.usage_type == "write":
                    lineage.sources.append(usage)
                elif usage.usage_type in ("read", "filter"):
                    lineage.consumers.append(usage)
                else:
                    lineage.transformations.append(usage)

    return report


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(report: LineageReport) -> str:
    """Generate markdown report."""
    lines = [
        "# Column Lineage Report",
        "",
        f"Files analyzed: {report.files_analyzed}",
        f"Columns tracked: {len(report.columns)}",
        "",
        "---",
        "",
    ]

    # Orphan columns (written but not read)
    orphans = report.get_orphan_columns()
    if orphans:
        lines.extend([
            "## Orphan Columns",
            "",
            "Columns that are written but never read:",
            "",
        ])
        for col in sorted(orphans)[:20]:
            lines.append(f"- `{col}`")
        lines.append("")

    # Undefined columns (read but not written)
    undefined = report.get_undefined_columns()
    if undefined:
        lines.extend([
            "## Undefined Columns",
            "",
            "Columns that are read but never defined (may be from external sources):",
            "",
        ])
        for col in sorted(undefined)[:20]:
            lines.append(f"- `{col}`")
        lines.append("")

    # Rate columns detail
    lines.extend([
        "## Rate Column Usage",
        "",
        "| Column | Writers | Readers |",
        "|--------|---------|---------|",
    ])

    for col_name in sorted(report.columns.keys()):
        lineage = report.columns[col_name]
        # Only show rate-related columns
        if any(re.match(p, col_name, re.IGNORECASE) for p in RATE_COLUMN_PATTERNS):
            writers = len(lineage.sources)
            readers = len(lineage.consumers)
            lines.append(f"| `{col_name}` | {writers} | {readers} |")

    return "\n".join(lines)


def track_column(report: LineageReport, column_name: str) -> str:
    """Generate detailed tracking for a specific column."""
    if column_name not in report.columns:
        return f"Column '{column_name}' not found in codebase."

    lineage = report.columns[column_name]

    lines = [
        f"# Column Lineage: {column_name}",
        "",
        "## Sources (where column is created/written)",
        "",
    ]

    if lineage.sources:
        for usage in lineage.sources[:10]:
            lines.append(f"- `{usage.file_path}:{usage.line_number}` in {usage.context}")
    else:
        lines.append("- Not created in analyzed files (may be from data source)")

    lines.extend([
        "",
        "## Consumers (where column is read)",
        "",
    ])

    if lineage.consumers:
        for usage in lineage.consumers[:20]:
            lines.append(f"- `{usage.file_path}:{usage.line_number}` in {usage.context}")
    else:
        lines.append("- Not consumed in analyzed files")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Column lineage tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze column usage")
    analyze_parser.add_argument("path", type=Path, help="Path to analyze")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track specific column")
    track_parser.add_argument("column", type=str, help="Column name to track")
    track_parser.add_argument("--path", type=Path, default=Path("."), help="Base path")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate lineage report")
    report_parser.add_argument("--path", type=Path, default=Path("."), help="Base path")
    report_parser.add_argument("--output", type=Path, help="Output file")

    args = parser.parse_args()

    if args.command == "analyze":
        report = build_lineage(args.path)
        print(f"Analyzed {report.files_analyzed} files")
        print(f"Found {len(report.columns)} unique columns")

        # Show summary
        print("\nTop 10 most-used columns:")
        sorted_cols = sorted(
            report.columns.items(),
            key=lambda x: len(x[1].consumers),
            reverse=True,
        )
        for col_name, lineage in sorted_cols[:10]:
            print(f"  {col_name}: {len(lineage.consumers)} reads, {len(lineage.sources)} writes")

    elif args.command == "track":
        report = build_lineage(args.path)
        result = track_column(report, args.column)
        print(result)

    elif args.command == "report":
        report = build_lineage(args.path)
        content = generate_report(report)

        if args.output:
            args.output.write_text(content)
            print(f"Report written to {args.output}")
        else:
            print(content)


if __name__ == "__main__":
    main()
