#!/usr/bin/env python
"""
Fixture Refresh Script.

This script refreshes fixture data from AWS S3, capturing current production
data for offline development. It should be run quarterly or after major data
source changes.

Usage:
    python tests/fixtures/refresh_fixtures.py

Environment Variables Required:
    - STS_ENDPOINT_URL: AWS STS endpoint
    - ROLE_ARN: IAM role ARN for S3 access
    - XID: User identifier for role assumption
    - BUCKET_NAME: S3 bucket name

What This Script Does:
    1. Loads current production data from AWS S3
    2. Saves raw data fixtures (sales, WINK, weights, macro)
    3. Runs full pipeline and captures all 10 stage outputs
    4. Runs inference and captures baseline results
    5. Saves metadata with refresh timestamp and data shapes

Output:
    - tests/fixtures/rila/*.parquet (raw and processed data)
    - tests/fixtures/rila/economic_indicators/*.parquet (macro data)
    - tests/fixtures/rila/refresh_metadata.json (metadata)

Refresh Schedule:
    - Quarterly (every 90 days) recommended
    - After major data source changes
    - Before production deployment validation

Example:
    $ export STS_ENDPOINT_URL="https://sts.us-east-1.amazonaws.com"
    $ export ROLE_ARN="arn:aws:iam::123456789:role/MyRole"
    $ export XID="user123"
    $ export BUCKET_NAME="my-bucket"
    $ python tests/fixtures/refresh_fixtures.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def get_aws_config() -> Dict[str, str]:
    """
    Load AWS configuration from environment variables.

    Returns
    -------
    dict
        AWS configuration for S3Adapter

    Raises
    ------
    ValueError
        If required environment variables are missing
    """
    required_vars = ["STS_ENDPOINT_URL", "ROLE_ARN", "XID", "BUCKET_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(
            f"Missing required AWS environment variables: {missing}\n"
            f"Required: {required_vars}\n\n"
            f"Set them with:\n"
            f"  export STS_ENDPOINT_URL='https://sts.us-east-1.amazonaws.com'\n"
            f"  export ROLE_ARN='arn:aws:iam::123456789:role/MyRole'\n"
            f"  export XID='user123'\n"
            f"  export BUCKET_NAME='my-bucket'"
        )

    return {
        "sts_endpoint_url": os.getenv("STS_ENDPOINT_URL"),
        "role_arn": os.getenv("ROLE_ARN"),
        "xid": os.getenv("XID"),
        "bucket_name": os.getenv("BUCKET_NAME"),
    }


def refresh_fixtures_from_aws(output_dir: Path, aws_config: Dict[str, str], verbose: bool = True):
    """
    Refresh fixture data from AWS S3.

    Parameters
    ----------
    output_dir : Path
        Directory to save fixtures (e.g., tests/fixtures/rila)
    aws_config : dict
        AWS configuration for S3Adapter
    verbose : bool
        Print progress messages

    Raises
    ------
    ImportError
        If required modules are not available
    Exception
        If data loading or processing fails
    """
    # Import required modules
    try:
        from src.data.adapters.s3_adapter import S3Adapter
        from src.pipelines.data_pipeline import DataPipeline
        from src.notebooks import create_interface
    except ImportError as e:
        raise ImportError(
            f"Required modules not available: {e}\n"
            f"Ensure src package is properly installed."
        )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "economic_indicators").mkdir(exist_ok=True)

    if verbose:
        print("=" * 70)
        print("FIXTURE REFRESH FROM AWS")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"AWS bucket: {aws_config['bucket_name']}")
        print()

    # Initialize AWS adapter
    if verbose:
        print("[1/6] Initializing AWS S3 adapter...")

    try:
        adapter = S3Adapter(aws_config)
        adapter._ensure_connection()
        if verbose:
            print("[PASS] AWS S3 connection established")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to AWS S3: {e}")

    # Load sales data
    if verbose:
        print("\n[2/6] Loading sales data from S3...")

    try:
        sales = adapter.load_sales_data()
        sales_path = output_dir / "raw_sales_data.parquet"
        sales.to_parquet(sales_path)

        if verbose:
            print(f"[PASS] Sales data: {sales.shape} → {sales_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load sales data: {e}")

    # Load competitive rates
    if verbose:
        print("\n[3/6] Loading competitive rates from S3...")

    try:
        rates = adapter.load_competitive_rates(start_date="2020-01-01")
        rates_path = output_dir / "raw_wink_data.parquet"
        rates.to_parquet(rates_path)

        if verbose:
            print(f"[PASS] Competitive rates: {rates.shape} → {rates_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load competitive rates: {e}")

    # Load market weights
    if verbose:
        print("\n[4/6] Loading market weights from S3...")

    try:
        weights = adapter.load_market_weights()
        weights_path = output_dir / "market_share_weights.parquet"
        weights.to_parquet(weights_path)

        if verbose:
            print(f"[PASS] Market weights: {weights.shape} → {weights_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load market weights: {e}")

    # Load macro/economic indicators
    if verbose:
        print("\n[5/6] Loading economic indicators from S3...")

    try:
        macro = adapter.load_macro_data()

        for indicator_name, indicator_df in macro.items():
            indicator_path = output_dir / f"economic_indicators/{indicator_name}.parquet"
            indicator_df.to_parquet(indicator_path)

            if verbose:
                print(f"  - {indicator_name}: {indicator_df.shape}")

        if verbose:
            print(f"[PASS] Economic indicators: {len(macro)} files")
    except Exception as e:
        if verbose:
            print(f"[WARN] Warning: Could not load economic indicators: {e}")

    # Run full pipeline and capture stage outputs
    if verbose:
        print("\n[6/6] Running full pipeline to capture stage outputs...")

    try:
        pipeline = DataPipeline(adapter=adapter)

        # Capture each stage output
        for stage_num in range(1, 11):
            if verbose:
                print(f"  - Running stage {stage_num:02d}...", end=" ")

            stage_output = pipeline.run_stage(stage_num)
            stage_path = output_dir / f"stage_{stage_num:02d}_output.parquet"
            stage_output.to_parquet(stage_path)

            if verbose:
                print(f"[PASS] {stage_output.shape}")

        if verbose:
            print(f"[PASS] Pipeline: All 10 stages captured")

    except Exception as e:
        if verbose:
            print(f"\n[WARN] Warning: Pipeline execution failed: {e}")
            print("  Continuing with partial fixtures...")

    # Run inference and capture baseline (optional)
    if verbose:
        print("\nRunning inference to capture baseline...")

    try:
        interface = create_interface("6Y20B", environment="aws", adapter_kwargs={'adapter': adapter})
        data = interface.load_data()
        inference_result = interface.run_inference(data)

        # Save inference baseline
        if 'predictions' in inference_result:
            baseline_df = pd.DataFrame({
                'predictions': inference_result['predictions']
            })
            baseline_path = output_dir / "inference_baseline.parquet"
            baseline_df.to_parquet(baseline_path)

            if verbose:
                print(f"[PASS] Inference baseline: {baseline_df.shape}")

    except Exception as e:
        if verbose:
            print(f"[WARN] Warning: Inference failed: {e}")
            print("  Continuing without inference baseline...")

    # Save metadata
    if verbose:
        print("\nSaving metadata...")

    metadata = {
        'refresh_date': datetime.now().isoformat(),
        'aws_config': {
            'bucket_name': aws_config['bucket_name'],
            'role_arn': aws_config['role_arn'],
        },
        'data_shape': {
            'sales': list(sales.shape),
            'rates': list(rates.shape),
            'weights': list(weights.shape),
        },
        'files_created': len(list(output_dir.glob("*.parquet"))),
    }

    metadata_path = output_dir / "refresh_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[PASS] Metadata: {metadata_path}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("FIXTURE REFRESH COMPLETE")
        print("=" * 70)
        print(f"Refresh date: {metadata['refresh_date']}")
        print(f"Files created: {metadata['files_created']}")
        print(f"Output directory: {output_dir}")
        print()
        print("Next steps:")
        print("  1. Validate fixtures: pytest tests/fixtures/test_fixture_validity.py -v")
        print("  2. Test equivalence: pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v")
        print("  3. Commit fixtures: git add tests/fixtures/rila/ && git commit -m 'Refresh fixtures'")
        print("=" * 70)


def main():
    """Main entry point for fixture refresh script."""
    parser = argparse.ArgumentParser(
        description="Refresh fixture data from AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Refresh fixtures with default output directory
    python tests/fixtures/refresh_fixtures.py

    # Refresh fixtures with custom output directory
    python tests/fixtures/refresh_fixtures.py --output /path/to/fixtures

    # Quiet mode (minimal output)
    python tests/fixtures/refresh_fixtures.py --quiet

Environment Variables Required:
    STS_ENDPOINT_URL    AWS STS endpoint
    ROLE_ARN            IAM role ARN for S3 access
    XID                 User identifier for role assumption
    BUCKET_NAME         S3 bucket name
        """
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path("tests/fixtures/rila"),
        help="Output directory for fixtures (default: tests/fixtures/rila)"
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Minimal output (quiet mode)"
    )

    args = parser.parse_args()

    # Get AWS config
    try:
        aws_config = get_aws_config()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Run fixture refresh
    try:
        refresh_fixtures_from_aws(
            output_dir=args.output,
            aws_config=aws_config,
            verbose=not args.quiet
        )
        return 0

    except Exception as e:
        print(f"\nERROR: Fixture refresh failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
