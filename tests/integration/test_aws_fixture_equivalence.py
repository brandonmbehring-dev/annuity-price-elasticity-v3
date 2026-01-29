"""
AWS vs Fixture Equivalence Tests.

These tests validate that fixture-based development produces results identical
to AWS execution. They ensure that offline development with fixtures maintains
mathematical equivalence with production AWS data.

**IMPORTANT**: These tests require AWS credentials and are marked with @pytest.mark.aws.
They are skipped by default in offline mode and should only be run when:
- Validating fixture-AWS equivalence
- Before production deployment
- After fixture refresh

Usage:
    # Run all AWS equivalence tests (requires AWS credentials)
    pytest tests/integration/test_aws_fixture_equivalence.py -m aws -v

    # Skip AWS tests (default for offline development)
    pytest tests/integration/ -m "not aws" -v

    # Run only AWS tests
    pytest -m aws -v

Environment Variables Required:
    - STS_ENDPOINT_URL: AWS STS endpoint
    - ROLE_ARN: IAM role ARN for S3 access
    - XID: User identifier for role assumption
    - BUCKET_NAME: S3 bucket name

Mathematical Equivalence:
    All comparisons use 1e-12 precision tolerance to ensure bit-for-bit
    equivalence between AWS and fixture data.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import adapters
try:
    from src.data.adapters.s3_adapter import S3Adapter
    from src.data.adapters.fixture_adapter import FixtureAdapter
    S3_ADAPTER_AVAILABLE = True
except ImportError:
    S3Adapter = None
    FixtureAdapter = None
    S3_ADAPTER_AVAILABLE = False

# Import pipeline and interface
try:
    from src.pipelines.data_pipeline import DataPipeline
    from src.notebooks import create_interface
    PIPELINE_AVAILABLE = True
except ImportError:
    DataPipeline = None
    create_interface = None
    PIPELINE_AVAILABLE = False


def validate_dataframe_equivalence(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    tolerance: float = 1e-12,
    stage: str = "unknown"
) -> None:
    """
    Validate two DataFrames are mathematically equivalent.

    Parameters
    ----------
    actual : pd.DataFrame
        Actual DataFrame (fixture-based)
    expected : pd.DataFrame
        Expected DataFrame (AWS-based)
    tolerance : float
        Numerical tolerance for floating point comparisons
    stage : str
        Stage name for error messages

    Raises
    ------
    AssertionError
        If DataFrames are not equivalent at specified tolerance
    """
    # Check shapes match
    assert actual.shape == expected.shape, (
        f"Stage {stage}: Shape mismatch. "
        f"Fixture: {actual.shape}, AWS: {expected.shape}"
    )

    # Check columns match
    assert set(actual.columns) == set(expected.columns), (
        f"Stage {stage}: Column mismatch. "
        f"Fixture-only: {set(actual.columns) - set(expected.columns)}, "
        f"AWS-only: {set(expected.columns) - set(actual.columns)}"
    )

    # Check index matches
    pd.testing.assert_index_equal(
        actual.index,
        expected.index,
        exact=False,
        check_names=True,
        obj=f"Stage {stage} index"
    )

    # Check values for each column
    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]):
            # Numeric columns: use allclose with tolerance
            np.testing.assert_allclose(
                actual[col].values,
                expected[col].values,
                rtol=tolerance,
                atol=tolerance,
                err_msg=f"Stage {stage}, column '{col}' values differ"
            )
        else:
            # Non-numeric columns: exact equality
            pd.testing.assert_series_equal(
                actual[col],
                expected[col],
                check_exact=True,
                obj=f"Stage {stage}, column '{col}'"
            )


def get_aws_config() -> dict:
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
            f"Missing required AWS environment variables: {missing}. "
            f"Required: {required_vars}"
        )

    return {
        "sts_endpoint_url": os.getenv("STS_ENDPOINT_URL"),
        "role_arn": os.getenv("ROLE_ARN"),
        "xid": os.getenv("XID"),
        "bucket_name": os.getenv("BUCKET_NAME"),
    }


@pytest.fixture(scope="module")
def aws_config():
    """AWS configuration fixture."""
    return get_aws_config()


@pytest.fixture(scope="module")
def aws_adapter(aws_config):
    """AWS S3 adapter fixture."""
    if not S3_ADAPTER_AVAILABLE:
        pytest.skip("S3Adapter not available")
    return S3Adapter(aws_config)


@pytest.fixture(scope="module")
def fixture_adapter():
    """Fixture adapter fixture."""
    if not S3_ADAPTER_AVAILABLE:
        pytest.skip("FixtureAdapter not available")
    return FixtureAdapter()


@pytest.mark.aws
@pytest.mark.slow
class TestAWSFixtureEquivalence:
    """
    Test suite for AWS vs Fixture equivalence validation.

    These tests ensure that fixture-based development produces results
    identical to AWS execution at 1e-12 precision.
    """

    def test_aws_credentials_available(self, aws_config):
        """
        Verify AWS credentials are available.

        This is a sanity check before running expensive AWS tests.
        """
        assert aws_config is not None
        assert "sts_endpoint_url" in aws_config
        assert "role_arn" in aws_config
        assert "xid" in aws_config
        assert "bucket_name" in aws_config

        print(f"✓ AWS credentials configured: {aws_config['bucket_name']}")

    def test_data_loading_sales_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test that sales data loaded from AWS matches fixture data.

        Validates that raw sales data is identical between AWS S3 and
        local fixtures.
        """
        # Load from AWS
        aws_sales = aws_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")

        # Load from fixture
        fixture_sales = fixture_adapter.load_sales_data(product_filter="FlexGuard 6Y20B")

        # Validate equivalence
        validate_dataframe_equivalence(
            actual=fixture_sales,
            expected=aws_sales,
            tolerance=1e-12,
            stage="sales_data_loading"
        )

        print(f"✓ Sales data equivalence: {aws_sales.shape}")

    def test_data_loading_rates_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test that competitive rates loaded from AWS match fixture data.

        Validates that WINK competitive rates are identical between
        AWS S3 and local fixtures.
        """
        # Load from AWS
        aws_rates = aws_adapter.load_competitive_rates(start_date="2020-01-01")

        # Load from fixture
        fixture_rates = fixture_adapter.load_competitive_rates(start_date="2020-01-01")

        # Validate equivalence
        validate_dataframe_equivalence(
            actual=fixture_rates,
            expected=aws_rates,
            tolerance=1e-12,
            stage="competitive_rates_loading"
        )

        print(f"✓ Competitive rates equivalence: {aws_rates.shape}")

    def test_data_loading_weights_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test that market weights loaded from AWS match fixture data.

        Validates that market share weights are identical between
        AWS S3 and local fixtures.
        """
        # Load from AWS
        aws_weights = aws_adapter.load_market_weights()

        # Load from fixture
        fixture_weights = fixture_adapter.load_market_weights()

        # Validate equivalence
        validate_dataframe_equivalence(
            actual=fixture_weights,
            expected=aws_weights,
            tolerance=1e-12,
            stage="market_weights_loading"
        )

        print(f"✓ Market weights equivalence: {aws_weights.shape}")

    @pytest.mark.slow
    def test_pipeline_stage_01_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test pipeline stage 1 (product filtering) equivalence.

        Validates that product filtering produces identical results
        between AWS and fixture data sources.
        """
        if not PIPELINE_AVAILABLE:
            pytest.skip("DataPipeline not available")

        # Run with AWS data
        aws_pipeline = DataPipeline(adapter=aws_adapter)
        aws_stage_01 = aws_pipeline.run_stage(1)

        # Run with fixture data
        fixture_pipeline = DataPipeline(adapter=fixture_adapter)
        fixture_stage_01 = fixture_pipeline.run_stage(1)

        # Validate equivalence
        validate_dataframe_equivalence(
            actual=fixture_stage_01,
            expected=aws_stage_01,
            tolerance=1e-12,
            stage="pipeline_stage_01_product_filtering"
        )

        print(f"✓ Pipeline stage 1 equivalence: {aws_stage_01.shape}")

    @pytest.mark.slow
    def test_pipeline_full_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test full pipeline (all 10 stages) equivalence.

        Validates that the complete data pipeline produces identical
        results between AWS and fixture data sources.

        This is the most comprehensive equivalence test, ensuring that
        all pipeline transformations maintain mathematical equivalence.
        """
        if not PIPELINE_AVAILABLE:
            pytest.skip("DataPipeline not available")

        # Run with AWS data
        aws_pipeline = DataPipeline(adapter=aws_adapter)
        aws_result = aws_pipeline.run_full_pipeline()

        # Run with fixture data
        fixture_pipeline = DataPipeline(adapter=fixture_adapter)
        fixture_result = fixture_pipeline.run_full_pipeline()

        # Validate equivalence
        validate_dataframe_equivalence(
            actual=fixture_result,
            expected=aws_result,
            tolerance=1e-12,
            stage="pipeline_full_execution"
        )

        print(f"✓ Full pipeline equivalence: {aws_result.shape}")

    @pytest.mark.slow
    def test_inference_equivalence(self, aws_adapter, fixture_adapter):
        """
        Test inference results equivalence.

        Validates that model inference produces identical results
        (coefficients, predictions, metrics) between AWS and fixture
        data sources.

        This is the ultimate validation - if inference matches at 1e-12
        precision, all upstream transformations are proven equivalent.
        """
        if not PIPELINE_AVAILABLE:
            pytest.skip("Pipeline and inference not available")

        # Run inference with AWS data
        aws_interface = create_interface(
            "6Y20B",
            environment="aws",
            adapter_kwargs={'adapter': aws_adapter}
        )
        aws_data = aws_interface.load_data()
        aws_inference = aws_interface.run_inference(aws_data)

        # Run inference with fixture data
        fixture_interface = create_interface(
            "6Y20B",
            environment="fixture",
            adapter_kwargs={'adapter': fixture_adapter}
        )
        fixture_data = fixture_interface.load_data()
        fixture_inference = fixture_interface.run_inference(fixture_data)

        # Validate coefficients equivalence
        if 'coefficients' in aws_inference and 'coefficients' in fixture_inference:
            np.testing.assert_allclose(
                fixture_inference['coefficients'],
                aws_inference['coefficients'],
                rtol=1e-12,
                atol=1e-12,
                err_msg="Inference coefficients differ between AWS and fixture"
            )
            print(f"✓ Inference coefficients equivalence")

        # Validate predictions equivalence
        if 'predictions' in aws_inference and 'predictions' in fixture_inference:
            np.testing.assert_allclose(
                fixture_inference['predictions'],
                aws_inference['predictions'],
                rtol=1e-12,
                atol=1e-12,
                err_msg="Inference predictions differ between AWS and fixture"
            )
            print(f"✓ Inference predictions equivalence")

        # Validate metrics equivalence
        if 'metrics' in aws_inference and 'metrics' in fixture_inference:
            for metric_name, aws_value in aws_inference['metrics'].items():
                fixture_value = fixture_inference['metrics'][metric_name]
                assert abs(fixture_value - aws_value) < 1e-12, (
                    f"Metric '{metric_name}' differs: "
                    f"Fixture={fixture_value}, AWS={aws_value}"
                )
            print(f"✓ Inference metrics equivalence")


class TestAWSConnection:
    """
    Tests for AWS connection and access validation.

    These tests verify that AWS credentials work and S3 access is
    functional before running expensive equivalence tests.
    """

    @pytest.mark.aws
    def test_aws_s3_connection(self, aws_config):
        """
        Test that AWS S3 connection can be established.

        Validates credentials and network access to S3.
        """
        if not S3_ADAPTER_AVAILABLE:
            pytest.skip("S3Adapter not available")

        # Try to create adapter
        adapter = S3Adapter(aws_config)

        # Try to establish connection
        try:
            adapter._ensure_connection()
            print("✓ AWS S3 connection successful")
        except Exception as e:
            pytest.fail(f"AWS S3 connection failed: {e}")

    @pytest.mark.aws
    def test_aws_s3_bucket_access(self, aws_adapter):
        """
        Test that S3 bucket is accessible.

        Validates read permissions on configured S3 bucket.
        """
        # Try to list objects (just check access)
        try:
            # This should work if credentials and bucket access are correct
            aws_adapter._ensure_connection()
            bucket = aws_adapter._bucket

            # Try to get bucket metadata
            bucket.load()
            print(f"✓ AWS S3 bucket accessible: {bucket.name}")

        except Exception as e:
            pytest.fail(f"AWS S3 bucket access failed: {e}")


# Utility functions for fixture refresh validation

def load_fixture_metadata() -> dict:
    """
    Load fixture metadata from refresh_metadata.json.

    Returns
    -------
    dict
        Fixture metadata with refresh_date, data_shape, etc.

    Raises
    ------
    FileNotFoundError
        If metadata file doesn't exist
    """
    import json

    metadata_path = Path("tests/fixtures/rila/refresh_metadata.json")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Fixture metadata not found: {metadata_path}. "
            f"Run fixture refresh script to create metadata."
        )

    with open(metadata_path, 'r') as f:
        return json.load(f)


def check_fixture_freshness(max_days: int = 90) -> bool:
    """
    Check if fixtures are fresh enough.

    Parameters
    ----------
    max_days : int
        Maximum age in days before fixtures are considered stale

    Returns
    -------
    bool
        True if fixtures are fresh, False if stale
    """
    from datetime import datetime

    try:
        metadata = load_fixture_metadata()
        refresh_date = datetime.fromisoformat(metadata['refresh_date'])
        days_old = (datetime.now() - refresh_date).days

        return days_old < max_days

    except (FileNotFoundError, KeyError, ValueError):
        return False
