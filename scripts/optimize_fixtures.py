#!/usr/bin/env python3
"""
Fixture Optimization for Annuity Price Elasticity v2.

Analyzes and optimizes test fixtures:
1. Remove unused columns
2. Reduce row counts while preserving coverage
3. Verify fixture completeness
4. Generate minimal fixture sets

Usage:
    # Analyze fixtures
    python scripts/optimize_fixtures.py analyze

    # Verify fixtures cover required schemas
    python scripts/optimize_fixtures.py verify

    # Generate optimized fixtures
    python scripts/optimize_fixtures.py optimize --output tests/fixtures/optimized/
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

FIXTURE_DIR = Path("tests/fixtures")
REQUIRED_COLUMNS = {
    "rila": [
        "date", "sales", "prudential_rate", "competitor_rate",
        "buffer_level", "term_years",
    ],
    "fia": [
        "date", "sales", "own_rate", "competitor_rate",
        "crediting_method",
    ],
    "common": [
        "date", "product_code",
    ],
}

# Columns that can be dropped to reduce fixture size
DROPPABLE_COLUMNS = [
    "detailed_notes", "raw_data", "metadata",
]

# Maximum rows for optimized fixtures
MAX_ROWS = {
    "unit": 100,
    "integration": 500,
    "baseline": 1000,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FixtureInfo:
    """Information about a single fixture file."""

    path: Path
    rows: int
    columns: List[str]
    size_bytes: int
    dtypes: Dict[str, str]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class FixtureReport:
    """Report on fixture analysis."""

    fixtures: List[FixtureInfo] = field(default_factory=list)
    missing_columns: Dict[str, List[str]] = field(default_factory=dict)
    unused_columns: Dict[str, List[str]] = field(default_factory=dict)
    total_size_mb: float = 0.0


# =============================================================================
# ANALYSIS
# =============================================================================

def load_fixture(path: Path) -> Optional[pd.DataFrame]:
    """Load fixture file (CSV, Parquet, or JSON)."""
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def analyze_fixture(path: Path) -> Optional[FixtureInfo]:
    """Analyze a single fixture file."""
    df = load_fixture(path)
    if df is None:
        return None

    return FixtureInfo(
        path=path,
        rows=len(df),
        columns=list(df.columns),
        size_bytes=path.stat().st_size,
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
    )


def analyze_fixtures(fixture_dir: Path) -> FixtureReport:
    """Analyze all fixtures in directory."""
    report = FixtureReport()

    for ext in [".csv", ".parquet", ".json"]:
        for path in fixture_dir.rglob(f"*{ext}"):
            info = analyze_fixture(path)
            if info:
                report.fixtures.append(info)
                report.total_size_mb += info.size_mb

    return report


def verify_schema_coverage(
    report: FixtureReport,
    required: Dict[str, List[str]] = REQUIRED_COLUMNS,
) -> Dict[str, List[str]]:
    """Verify fixtures cover required schemas."""
    all_columns: Set[str] = set()
    for fixture in report.fixtures:
        all_columns.update(fixture.columns)

    missing = {}
    for schema_name, required_cols in required.items():
        missing_cols = [col for col in required_cols if col not in all_columns]
        if missing_cols:
            missing[schema_name] = missing_cols

    return missing


def find_unused_columns(
    report: FixtureReport,
    droppable: List[str] = DROPPABLE_COLUMNS,
) -> Dict[str, List[str]]:
    """Find columns that can potentially be dropped."""
    unused = {}

    for fixture in report.fixtures:
        droppable_in_fixture = [
            col for col in fixture.columns
            if any(d in col.lower() for d in droppable)
        ]
        if droppable_in_fixture:
            unused[str(fixture.path)] = droppable_in_fixture

    return unused


# =============================================================================
# OPTIMIZATION
# =============================================================================

def optimize_fixture(
    path: Path,
    output_path: Path,
    max_rows: int = 500,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """Optimize a single fixture, return (original_size, new_size)."""
    df = load_fixture(path)
    if df is None:
        return (0, 0)

    original_size = path.stat().st_size

    # Drop specified columns
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    # Reduce rows if needed
    if len(df) > max_rows:
        # Stratified sampling if possible
        df = df.sample(n=max_rows, random_state=42)

    # Save optimized fixture
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif path.suffix == ".json":
        df.to_json(output_path, orient="records")

    new_size = output_path.stat().st_size
    return (original_size, new_size)


def optimize_all_fixtures(
    fixture_dir: Path,
    output_dir: Path,
    fixture_type: str = "integration",
) -> Dict[str, Tuple[int, int]]:
    """Optimize all fixtures."""
    max_rows = MAX_ROWS.get(fixture_type, 500)
    results = {}

    for ext in [".csv", ".parquet"]:
        for path in fixture_dir.rglob(f"*{ext}"):
            relative_path = path.relative_to(fixture_dir)
            output_path = output_dir / relative_path

            original, optimized = optimize_fixture(
                path, output_path, max_rows=max_rows
            )
            if original > 0:
                results[str(path)] = (original, optimized)

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_analysis_report(report: FixtureReport) -> None:
    """Print analysis report to console."""
    print("=" * 60)
    print("Fixture Analysis Report")
    print("=" * 60)
    print(f"Total fixtures: {len(report.fixtures)}")
    print(f"Total size: {report.total_size_mb:.2f} MB")
    print()

    print("Fixtures by size:")
    sorted_fixtures = sorted(report.fixtures, key=lambda x: x.size_bytes, reverse=True)
    for fixture in sorted_fixtures[:10]:
        print(f"  {fixture.path.name}: {fixture.rows} rows, {fixture.size_mb:.2f} MB")

    print()
    print("Column coverage:")
    all_columns = set()
    for fixture in report.fixtures:
        all_columns.update(fixture.columns)
    print(f"  Total unique columns: {len(all_columns)}")


def generate_markdown_report(report: FixtureReport) -> str:
    """Generate markdown report."""
    lines = [
        "# Fixture Analysis Report",
        "",
        f"Total fixtures: {len(report.fixtures)}",
        f"Total size: {report.total_size_mb:.2f} MB",
        "",
        "## Fixtures by Size",
        "",
        "| Fixture | Rows | Columns | Size (MB) |",
        "|---------|------|---------|-----------|",
    ]

    sorted_fixtures = sorted(report.fixtures, key=lambda x: x.size_bytes, reverse=True)
    for fixture in sorted_fixtures:
        lines.append(
            f"| {fixture.path.name} | {fixture.rows} | {len(fixture.columns)} | {fixture.size_mb:.2f} |"
        )

    # Schema coverage
    missing = verify_schema_coverage(report)
    if missing:
        lines.extend([
            "",
            "## Missing Schema Columns",
            "",
        ])
        for schema, cols in missing.items():
            lines.append(f"**{schema}**: {', '.join(cols)}")
    else:
        lines.extend([
            "",
            "## Schema Coverage",
            "",
            "All required columns present.",
        ])

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixture optimization tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze fixtures")
    analyze_parser.add_argument(
        "--path", type=Path, default=FIXTURE_DIR, help="Fixtures directory"
    )
    analyze_parser.add_argument(
        "--output", type=Path, help="Output report file"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify schema coverage")
    verify_parser.add_argument(
        "--path", type=Path, default=FIXTURE_DIR, help="Fixtures directory"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize fixtures")
    optimize_parser.add_argument(
        "--path", type=Path, default=FIXTURE_DIR, help="Source fixtures directory"
    )
    optimize_parser.add_argument(
        "--output", type=Path, required=True, help="Output directory"
    )
    optimize_parser.add_argument(
        "--type", type=str, choices=["unit", "integration", "baseline"],
        default="integration", help="Fixture type (determines row limit)"
    )

    args = parser.parse_args()

    if args.command == "analyze":
        report = analyze_fixtures(args.path)
        print_analysis_report(report)

        if args.output:
            content = generate_markdown_report(report)
            args.output.write_text(content)
            print(f"\nReport written to {args.output}")

    elif args.command == "verify":
        report = analyze_fixtures(args.path)
        missing = verify_schema_coverage(report)

        if missing:
            print("Missing schema columns:")
            for schema, cols in missing.items():
                print(f"  {schema}: {', '.join(cols)}")
            sys.exit(1)
        else:
            print("All required schema columns present.")

    elif args.command == "optimize":
        print(f"Optimizing fixtures from {args.path} to {args.output}")
        print(f"Max rows: {MAX_ROWS[args.type]}")

        results = optimize_all_fixtures(args.path, args.output, args.type)

        total_original = sum(r[0] for r in results.values())
        total_optimized = sum(r[1] for r in results.values())
        reduction = (1 - total_optimized / total_original) * 100 if total_original > 0 else 0

        print(f"\nOptimized {len(results)} fixtures")
        print(f"Original size: {total_original / 1024 / 1024:.2f} MB")
        print(f"Optimized size: {total_optimized / 1024 / 1024:.2f} MB")
        print(f"Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
