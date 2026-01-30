"""
AWS Mocking Layer for Offline Notebook Development

Provides drop-in replacements for AWS operations that load from local fixtures
instead of making AWS API calls. This enables notebook development and refactoring
on non-AWS machines using previously captured data.

Usage in notebooks:
    from src.validation_support.aws_mock_layer import setup_offline_environment
    setup_offline_environment(fixture_path='tests/fixtures/aws_captured_data/')

Key Features:
    - Mock S3 resource that reads from local parquet files
    - Mock STS client that returns fake credentials
    - Automatic environment configuration
    - Backward compatible with AWS mode (controlled by environment variable)

Author: Claude Code
Date: 2026-01-09
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class OfflineS3Resource:
    """
    Mock S3 resource that reads from local fixtures instead of AWS S3.

    Maps S3 bucket paths to local parquet files captured during AWS data capture.
    """

    def __init__(self, fixture_path: Path):
        """
        Initialize offline S3 resource.

        Args:
            fixture_path: Path to directory containing captured fixtures
        """
        self.fixture_path = Path(fixture_path)
        self.metadata = self._load_metadata()

        # Validate fixture directory
        if not self.fixture_path.exists():
            raise FileNotFoundError(
                f"OFFLINE MODE ERROR: Fixture directory not found: {self.fixture_path}. "
                f"Required Action: Run 'python scripts/capture_aws_data_once.py' in AWS environment first."
            )

        logger.info(f"Offline S3 resource initialized with fixtures from {self.fixture_path}")

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load capture metadata if available."""
        metadata_file = self.fixture_path / 'capture_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None

    def Bucket(self, bucket_name: str) -> "OfflineS3Resource":
        """
        Mock Bucket() method to match boto3 interface.

        Args:
            bucket_name: S3 bucket name (ignored in offline mode)

        Returns:
            Self (acts as mock bucket object)
        """
        logger.debug(f"Offline S3: Bucket({bucket_name}) called")
        return self

    def objects(self) -> "OfflineS3Resource":
        """
        Mock objects() method to match boto3 interface.

        Returns:
            Self (acts as mock objects collection)
        """
        return self

    def filter(self, Prefix: str) -> list:
        """
        Mock filter() method to match boto3 interface.

        Args:
            Prefix: S3 prefix to filter (used to determine fixture file)

        Returns:
            List of mock S3 objects
        """
        logger.debug(f"Offline S3: filter(Prefix={Prefix}) called")

        # Map S3 prefix to local fixture file
        fixture_file = self._map_s3_prefix_to_fixture(Prefix)

        if fixture_file and fixture_file.exists():
            # Return mock object with key attribute
            class MockS3Object:
                def __init__(self, key):
                    self.key = key

            return [MockS3Object(str(fixture_file))]
        else:
            return []

    def _map_s3_prefix_to_fixture(self, s3_prefix: str) -> Optional[Path]:
        """
        Map S3 prefix to local fixture file.

        Args:
            s3_prefix: S3 key prefix (e.g., 'access/ierpt/tde_sales_by_product_by_fund/')

        Returns:
            Path to local fixture file if mapping exists, None otherwise
        """
        # Define S3 prefix to fixture file mappings
        prefix_mappings = {
            'access/ierpt/tde_sales_by_product_by_fund': 'raw_sales_data.parquet',
            'access/ierpt/wink_ann_product_rates': 'raw_wink_data.parquet',
            'MACRO_ECONOMIC_DATA/DGS5_index': 'economic_indicators/dgs5.parquet',
            'MACRO_ECONOMIC_DATA/VIXCLS_index': 'economic_indicators/vixcls.parquet',
            'MACRO_ECONOMIC_DATA/cpi_scaled': 'economic_indicators/cpi.parquet',
            'flex_guard_market_share': 'market_share_weights.parquet'
        }

        # Find matching prefix
        for prefix_pattern, fixture_file in prefix_mappings.items():
            if prefix_pattern in s3_prefix:
                return self.fixture_path / fixture_file

        # No mapping found
        logger.warning(f"No fixture mapping for S3 prefix: {s3_prefix}")
        return None

    def load_parquet_from_s3_prefix(self, bucket_name: str, s3_prefix: str) -> pd.DataFrame:
        """
        Load parquet data from S3 prefix (offline mode).

        This is the main method used by data extraction functions.

        Args:
            bucket_name: S3 bucket name (ignored in offline mode)
            s3_prefix: S3 prefix path

        Returns:
            DataFrame loaded from local fixture

        Raises:
            FileNotFoundError: If no fixture exists for the given prefix
        """
        logger.info(f"Offline S3: Loading data for prefix {s3_prefix} (bucket={bucket_name})")

        # Map prefix to fixture file
        fixture_file = self._map_s3_prefix_to_fixture(s3_prefix)

        if not fixture_file or not fixture_file.exists():
            raise FileNotFoundError(
                f"OFFLINE MODE ERROR: No fixture found for S3 prefix: {s3_prefix}. "
                f"Expected fixture: {fixture_file}. "
                f"Required Action: Run data capture script in AWS environment first."
            )

        # Load parquet file
        logger.info(f"Loading fixture: {fixture_file}")
        df = pd.read_parquet(fixture_file)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        return df


class OfflineSTSClient:
    """
    Mock STS client that returns fake credentials.

    In offline mode, credentials are never actually used since no AWS API calls are made.
    """

    def assume_role(self, RoleArn: str, RoleSessionName: str) -> Dict[str, Any]:
        """
        Mock assume_role() method.

        Args:
            RoleArn: IAM role ARN (ignored in offline mode)
            RoleSessionName: Session name (ignored in offline mode)

        Returns:
            Mock credentials dict
        """
        logger.info(f"Offline STS: assume_role called (role={RoleArn})")
        logger.info("Offline STS: Returning mock credentials (not used in offline mode)")

        return {
            'Credentials': {
                'AccessKeyId': 'OFFLINE_MODE_NO_REAL_CREDENTIALS',
                'SecretAccessKey': 'OFFLINE_MODE_NO_REAL_CREDENTIALS',
                'SessionToken': 'OFFLINE_MODE_NO_REAL_CREDENTIALS',
                'Expiration': '2099-12-31T23:59:59Z'
            },
            'AssumedRoleUser': {
                'AssumedRoleId': 'OFFLINE:offline-session',
                'Arn': RoleArn
            }
        }


def _resolve_fixture_path(fixture_path: str) -> Path:
    """
    Resolve fixture path, auto-detecting project root if relative.

    Args:
        fixture_path: Path to fixtures directory (absolute or relative)

    Returns:
        Resolved absolute Path to fixtures directory
    """
    fixture_path = Path(fixture_path)
    if fixture_path.is_absolute():
        return fixture_path

    # Try to find project root by looking for src/ directory
    search_dirs = [
        Path.cwd(),  # Current directory
        Path.cwd().parent,  # Parent (if in notebooks/)
        Path(__file__).parent.parent.parent,  # Relative to this file
    ]
    for search_dir in search_dirs:
        candidate = search_dir / fixture_path
        if candidate.exists():
            return candidate

    return fixture_path  # Return as-is, will fail later with clear error


def _create_mock_data_loaders(mock_s3_resource: OfflineS3Resource, fixture_path: Path):
    """
    Create mock data loading functions for extraction module patching.

    Args:
        mock_s3_resource: Offline S3 resource for loading data
        fixture_path: Path to fixtures directory

    Returns:
        Dict of mock functions keyed by their target names
    """
    def mock_discover_and_load_sales_data(bucket: Any, s3_resource: Any, bucket_name: str) -> pd.DataFrame:
        """Load sales data from fixture instead of S3."""
        return mock_s3_resource.load_parquet_from_s3_prefix(
            bucket_name, 'access/ierpt/tde_sales_by_product_by_fund/'
        )

    def mock_discover_and_load_wink_data(bucket: Any, s3_resource: Any, bucket_name: str) -> pd.DataFrame:
        """Load WINK data from fixture instead of S3."""
        return mock_s3_resource.load_parquet_from_s3_prefix(
            bucket_name, 'access/ierpt/wink_ann_product_rates/'
        )

    def mock_load_market_share_weights_from_s3(s3_path: str) -> pd.DataFrame:
        """Load market share weights from fixture."""
        return pd.read_parquet(fixture_path / 'market_share_weights.parquet')

    def mock_download_s3_parquet_with_optional_date_suffix(bucket_name: str, s3_prefix: str, date_suffix: Optional[str]) -> pd.DataFrame:
        """Load economic indicator from fixture."""
        indicator_map = {
            'DGS5': 'dgs5.parquet',
            'VIXCLS': 'vixcls.parquet',
            'cpi': 'cpi.parquet',
        }
        for key, filename in indicator_map.items():
            if key in s3_prefix:
                return pd.read_parquet(fixture_path / 'economic_indicators' / filename)
        raise FileNotFoundError(f"No fixture for S3 prefix: {s3_prefix}")

    return {
        'discover_and_load_sales_data': mock_discover_and_load_sales_data,
        'discover_and_load_wink_data': mock_discover_and_load_wink_data,
        'load_market_share_weights_from_s3': mock_load_market_share_weights_from_s3,
        'download_s3_parquet_with_optional_date_suffix': mock_download_s3_parquet_with_optional_date_suffix,
    }


def _patch_extraction_module(mock_sts_client: OfflineSTSClient,
                             mock_s3_resource: OfflineS3Resource,
                             mock_loaders: Dict[str, Any]) -> None:
    """
    Patch extraction module with mock AWS functions.

    Args:
        mock_sts_client: Mock STS client
        mock_s3_resource: Mock S3 resource
        mock_loaders: Dict of mock data loading functions

    Raises:
        ImportError: If extraction module cannot be imported
    """
    try:
        from src.data import extraction
    except ImportError as e:
        logger.error(f"Cannot import src.data.extraction: {e}")
        raise ImportError(
            f"OFFLINE MODE ERROR: Cannot import src.data.extraction module. "
            f"Ensure you are running from project root directory."
        )

    # Patch AWS setup functions
    extraction.setup_aws_sts_client_with_validation = lambda aws_config: mock_sts_client
    extraction.assume_iam_role_with_validation = lambda sts_client, aws_config: {
        'Credentials': mock_sts_client.assume_role('mock-role', 'mock-session')['Credentials']
    }
    extraction.setup_s3_resource_with_validation = lambda credentials, bucket_name: (
        mock_s3_resource, mock_s3_resource.Bucket(bucket_name)
    )

    # Patch data loading functions
    for func_name, mock_func in mock_loaders.items():
        setattr(extraction, func_name, mock_func)


def _print_offline_confirmation(fixture_path: Path, mock_s3_resource: OfflineS3Resource) -> None:
    """Print confirmation message for offline mode setup."""
    print("=" * 80)
    print("[OK] Offline environment configured")
    print(f"Path: Using fixtures from: {fixture_path.absolute()}")
    print("Note: AWS API calls will NOT be made")
    print("=" * 80)

    if mock_s3_resource.metadata:
        print(f"\nFixture metadata:")
        print(f"  Capture date: {mock_s3_resource.metadata.get('capture_date', 'Unknown')}")
        print(f"  Pipeline stages: {len(mock_s3_resource.metadata.get('stages', {}))}")

    print(f"\nOffline mode ready for notebook execution.")
    print("=" * 80)


def setup_offline_environment(
    fixture_path: str = 'tests/fixtures/aws_complete/',
    force: bool = False
) -> OfflineS3Resource:
    """
    Configure notebooks to run in offline mode using fixtures.

    Call this at the start of any notebook to enable offline operation.
    This function patches the src.data.extraction module to use local fixtures
    instead of making AWS API calls.

    Args:
        fixture_path: Path to fixtures directory (default: tests/fixtures/aws_complete/)
        force: Force offline mode even if AWS credentials are available

    Returns:
        Mock S3 resource for manual operations if needed
    """
    os.environ['OFFLINE_MODE'] = '1'

    resolved_path = _resolve_fixture_path(fixture_path)
    mock_sts_client = OfflineSTSClient()
    mock_s3_resource = OfflineS3Resource(resolved_path)

    mock_loaders = _create_mock_data_loaders(mock_s3_resource, resolved_path)
    _patch_extraction_module(mock_sts_client, mock_s3_resource, mock_loaders)
    _print_offline_confirmation(resolved_path, mock_s3_resource)

    return mock_s3_resource


def verify_fixture_availability(fixture_path: str = 'tests/fixtures/aws_complete/') -> bool:
    """
    Check that all required fixtures are present.

    Args:
        fixture_path: Path to fixtures directory

    Returns:
        True if all required fixtures are present, False otherwise

    Example:
        >>> from src.validation_support.aws_mock_layer import verify_fixture_availability
        >>> verify_fixture_availability()
        [PASS] All required fixtures present
        True
    """
    required_files = [
        'raw_sales_data.parquet',
        'raw_wink_data.parquet',
        'economic_indicators/dgs5.parquet',
        'economic_indicators/vixcls.parquet',
        'economic_indicators/cpi.parquet',
        'market_share_weights.parquet',
        'capture_metadata.json'
    ]

    fixture_path = Path(fixture_path)
    missing = []

    for file in required_files:
        if not (fixture_path / file).exists():
            missing.append(file)

    if missing:
        print(f"[ERROR] Missing fixtures: {missing}")
        print(f"\nRequired action:")
        print(f"  1. Run data capture in AWS environment:")
        print(f"     python scripts/capture_aws_data_once.py")
        print(f"  2. Verify fixtures:")
        print(f"     python scripts/verify_fixture_completeness.py")
        return False

    print(f"[OK] All required fixtures present ({len(required_files)} files)")
    return True


def is_offline_mode() -> bool:
    """
    Check if offline mode is currently enabled.

    Returns:
        True if OFFLINE_MODE environment variable is set, False otherwise
    """
    return os.environ.get('OFFLINE_MODE', '0') == '1'


# Module-level initialization message
if __name__ == '__main__':
    print(__doc__)
    print("\nTo use this module in a notebook:")
    print("  from src.validation_support.aws_mock_layer import setup_offline_environment")
    print("  setup_offline_environment()")
