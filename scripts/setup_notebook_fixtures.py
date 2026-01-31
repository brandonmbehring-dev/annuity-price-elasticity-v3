#!/usr/bin/env python3
"""Setup notebook fixtures for CI testing.

Copies fixture data to outputs/datasets/ and applies legacy column renaming
so that NB01/NB02 can run without AWS access.

The mapping transforms internal naming (competitor_weighted_*) to legacy
output naming (competitor_mid_*) expected by the production notebooks.
"""

import sys
from pathlib import Path

import pandas as pd

# Project root (assuming script is in scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Source fixtures
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "rila"

# Output location (where notebooks expect data)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "datasets"

# Legacy column mapping: internal → legacy output names
# The fixture uses competitor_weighted_*, notebooks expect competitor_mid_*


def build_column_renaming() -> dict:
    """Build column renaming map dynamically for all competitor_weighted columns."""
    renaming = {}
    # Handle _current suffix
    renaming["competitor_weighted_current"] = "competitor_mid_current"
    # Handle _t{N} suffixes
    for i in range(1, 19):
        renaming[f"competitor_weighted_t{i}"] = f"competitor_mid_t{i}"
    return renaming


COLUMN_RENAMING = build_column_renaming()


def setup_fixtures() -> None:
    """Copy and transform fixture files for notebook CI testing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Transform final_dataset.parquet (rename competitor columns)
    final_src = FIXTURE_DIR / "final_weekly_dataset.parquet"
    final_dst = OUTPUT_DIR / "final_dataset.parquet"

    if not final_src.exists():
        print(f"ERROR: Source fixture not found: {final_src}", file=sys.stderr)
        sys.exit(1)

    df_final = pd.read_parquet(final_src)

    # Apply column renaming (only columns that exist)
    rename_map = {k: v for k, v in COLUMN_RENAMING.items() if k in df_final.columns}
    if rename_map:
        df_final = df_final.rename(columns=rename_map)
        print(f"  Renamed {len(rename_map)} columns for legacy compatibility")

    df_final.to_parquet(final_dst, index=False)
    print(f"  {final_dst.name} <- {final_src.name} ({len(df_final)} rows)")

    # 2. Copy WINK competitive rates (no transformation needed)
    wink_src = FIXTURE_DIR / "market_weighted_competitive_rates.parquet"
    wink_dst = OUTPUT_DIR / "WINK_competitive_rates.parquet"

    if not wink_src.exists():
        print(f"ERROR: Source fixture not found: {wink_src}", file=sys.stderr)
        sys.exit(1)

    df_wink = pd.read_parquet(wink_src)
    df_wink.to_parquet(wink_dst, index=False)
    print(f"  {wink_dst.name} <- {wink_src.name} ({len(df_wink)} rows)")


if __name__ == "__main__":
    print("Setting up notebook fixtures for CI...")
    setup_fixtures()
    print("✓ Notebook fixtures ready")
