"""
Tests for AWS Mock Layer for offline notebook development.

Tests cover:
- OfflineS3Resource initialization and methods
- OfflineSTSClient mock credentials
- Path resolution utilities
- Fixture availability verification
- Offline mode detection
"""

import pytest
import pandas as pd
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.validation_support.aws_mock_layer import (
    OfflineS3Resource,
    OfflineSTSClient,
    _resolve_fixture_path,
    verify_fixture_availability,
    is_offline_mode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_fixture_dir():
    """Create temporary fixture directory with sample files."""
    temp_dir = tempfile.mkdtemp()
    fixture_path = Path(temp_dir)

    # Create fixture files
    (fixture_path / 'economic_indicators').mkdir()

    # Create sample parquet files
    sample_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    sample_df.to_parquet(fixture_path / 'raw_sales_data.parquet')
    sample_df.to_parquet(fixture_path / 'raw_wink_data.parquet')
    sample_df.to_parquet(fixture_path / 'market_share_weights.parquet')
    sample_df.to_parquet(fixture_path / 'economic_indicators' / 'dgs5.parquet')
    sample_df.to_parquet(fixture_path / 'economic_indicators' / 'vixcls.parquet')
    sample_df.to_parquet(fixture_path / 'economic_indicators' / 'cpi.parquet')

    # Create metadata file
    metadata = {
        'capture_date': '2026-01-30',
        'stages': {'stage1': 'complete', 'stage2': 'complete'}
    }
    with open(fixture_path / 'capture_metadata.json', 'w') as f:
        json.dump(metadata, f)

    yield fixture_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def incomplete_fixture_dir():
    """Create fixture directory missing some required files."""
    temp_dir = tempfile.mkdtemp()
    fixture_path = Path(temp_dir)

    # Only create some files
    sample_df = pd.DataFrame({'col1': [1, 2, 3]})
    sample_df.to_parquet(fixture_path / 'raw_sales_data.parquet')

    yield fixture_path

    shutil.rmtree(temp_dir)


# =============================================================================
# Tests: OfflineS3Resource
# =============================================================================


class TestOfflineS3Resource:
    """Tests for OfflineS3Resource class."""

    def test_init_with_valid_path(self, temp_fixture_dir):
        """Test initialization with valid fixture directory."""
        resource = OfflineS3Resource(temp_fixture_dir)

        assert resource.fixture_path == temp_fixture_dir
        assert resource.metadata is not None
        assert resource.metadata['capture_date'] == '2026-01-30'

    def test_init_with_invalid_path(self):
        """Test initialization raises error for missing directory."""
        with pytest.raises(FileNotFoundError, match="Fixture directory not found"):
            OfflineS3Resource('/nonexistent/path')

    def test_load_metadata_returns_dict(self, temp_fixture_dir):
        """Test metadata loading returns parsed JSON."""
        resource = OfflineS3Resource(temp_fixture_dir)

        assert isinstance(resource.metadata, dict)
        assert 'stages' in resource.metadata

    def test_load_metadata_returns_none_if_missing(self, incomplete_fixture_dir):
        """Test metadata loading returns None if file missing."""
        resource = OfflineS3Resource(incomplete_fixture_dir)

        assert resource.metadata is None

    def test_bucket_returns_self(self, temp_fixture_dir):
        """Test Bucket() method returns self for chaining."""
        resource = OfflineS3Resource(temp_fixture_dir)
        result = resource.Bucket('any-bucket-name')

        assert result is resource

    def test_objects_returns_self(self, temp_fixture_dir):
        """Test objects() method returns self for chaining."""
        resource = OfflineS3Resource(temp_fixture_dir)
        result = resource.objects()

        assert result is resource

    def test_filter_returns_mock_objects_for_known_prefix(self, temp_fixture_dir):
        """Test filter() returns mock objects for mapped prefixes."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource.filter(Prefix='access/ierpt/tde_sales_by_product_by_fund/')

        assert len(result) == 1
        assert hasattr(result[0], 'key')

    def test_filter_returns_empty_for_unknown_prefix(self, temp_fixture_dir):
        """Test filter() returns empty list for unmapped prefixes."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource.filter(Prefix='unknown/prefix/')

        assert result == []

    def test_map_s3_prefix_to_fixture_sales(self, temp_fixture_dir):
        """Test prefix mapping for sales data."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource._map_s3_prefix_to_fixture(
            'access/ierpt/tde_sales_by_product_by_fund/some_file'
        )

        assert result == temp_fixture_dir / 'raw_sales_data.parquet'

    def test_map_s3_prefix_to_fixture_wink(self, temp_fixture_dir):
        """Test prefix mapping for WINK data."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource._map_s3_prefix_to_fixture(
            'access/ierpt/wink_ann_product_rates/some_file'
        )

        assert result == temp_fixture_dir / 'raw_wink_data.parquet'

    def test_map_s3_prefix_to_fixture_economic(self, temp_fixture_dir):
        """Test prefix mapping for economic indicators."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource._map_s3_prefix_to_fixture(
            'MACRO_ECONOMIC_DATA/DGS5_index/data'
        )

        assert result == temp_fixture_dir / 'economic_indicators' / 'dgs5.parquet'

    def test_map_s3_prefix_returns_none_for_unknown(self, temp_fixture_dir):
        """Test prefix mapping returns None for unknown prefixes."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource._map_s3_prefix_to_fixture('totally/unknown/prefix')

        assert result is None

    def test_load_parquet_from_s3_prefix_success(self, temp_fixture_dir):
        """Test loading parquet data from fixture."""
        resource = OfflineS3Resource(temp_fixture_dir)

        df = resource.load_parquet_from_s3_prefix(
            'any-bucket',
            'access/ierpt/tde_sales_by_product_by_fund/'
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'col1' in df.columns

    def test_load_parquet_from_s3_prefix_error_unknown(self, temp_fixture_dir):
        """Test loading raises error for unknown prefix."""
        resource = OfflineS3Resource(temp_fixture_dir)

        with pytest.raises(FileNotFoundError, match="No fixture found"):
            resource.load_parquet_from_s3_prefix(
                'any-bucket',
                'unknown/prefix/'
            )


# =============================================================================
# Tests: OfflineSTSClient
# =============================================================================


class TestOfflineSTSClient:
    """Tests for OfflineSTSClient class."""

    def test_assume_role_returns_mock_credentials(self):
        """Test assume_role returns properly structured mock credentials."""
        client = OfflineSTSClient()

        result = client.assume_role(
            RoleArn='arn:aws:iam::123456:role/test-role',
            RoleSessionName='test-session'
        )

        assert 'Credentials' in result
        assert 'AccessKeyId' in result['Credentials']
        assert 'SecretAccessKey' in result['Credentials']
        assert 'SessionToken' in result['Credentials']
        assert 'Expiration' in result['Credentials']

    def test_assume_role_credentials_are_fake(self):
        """Test credentials clearly indicate offline mode."""
        client = OfflineSTSClient()

        result = client.assume_role(
            RoleArn='arn:aws:iam::123456:role/test-role',
            RoleSessionName='test-session'
        )

        assert 'OFFLINE_MODE' in result['Credentials']['AccessKeyId']

    def test_assume_role_preserves_role_arn(self):
        """Test role ARN is preserved in response."""
        client = OfflineSTSClient()
        test_arn = 'arn:aws:iam::123456:role/specific-role'

        result = client.assume_role(
            RoleArn=test_arn,
            RoleSessionName='test-session'
        )

        assert result['AssumedRoleUser']['Arn'] == test_arn


# =============================================================================
# Tests: _resolve_fixture_path
# =============================================================================


class TestResolveFixturePath:
    """Tests for fixture path resolution."""

    def test_absolute_path_returned_unchanged(self):
        """Test absolute paths are returned as-is."""
        absolute_path = '/absolute/path/to/fixtures'

        result = _resolve_fixture_path(absolute_path)

        assert result == Path(absolute_path)

    def test_relative_path_searched_in_cwd(self, temp_fixture_dir):
        """Test relative paths are searched from current directory."""
        # Get the relative path from cwd
        with patch('pathlib.Path.cwd', return_value=temp_fixture_dir.parent):
            relative_name = temp_fixture_dir.name
            result = _resolve_fixture_path(relative_name)

            assert result == temp_fixture_dir

    def test_nonexistent_relative_path_returned_for_later_error(self):
        """Test nonexistent relative path is returned for later validation."""
        result = _resolve_fixture_path('nonexistent/relative/path')

        # Should return as Path object even if doesn't exist
        assert isinstance(result, Path)


# =============================================================================
# Tests: verify_fixture_availability
# =============================================================================


class TestVerifyFixtureAvailability:
    """Tests for fixture availability verification."""

    def test_all_fixtures_present_returns_true(self, temp_fixture_dir, capsys):
        """Test returns True when all fixtures present."""
        result = verify_fixture_availability(str(temp_fixture_dir))

        assert result is True
        captured = capsys.readouterr()
        assert '[OK]' in captured.out

    def test_missing_fixtures_returns_false(self, incomplete_fixture_dir, capsys):
        """Test returns False when fixtures missing."""
        result = verify_fixture_availability(str(incomplete_fixture_dir))

        assert result is False
        captured = capsys.readouterr()
        assert '[ERROR]' in captured.out
        assert 'Missing fixtures' in captured.out

    def test_missing_fixtures_lists_required_action(self, incomplete_fixture_dir, capsys):
        """Test missing fixtures shows required action."""
        verify_fixture_availability(str(incomplete_fixture_dir))

        captured = capsys.readouterr()
        assert 'Required action' in captured.out


# =============================================================================
# Tests: is_offline_mode
# =============================================================================


class TestIsOfflineMode:
    """Tests for offline mode detection."""

    def test_returns_true_when_env_set(self):
        """Test returns True when OFFLINE_MODE=1."""
        with patch.dict(os.environ, {'OFFLINE_MODE': '1'}):
            assert is_offline_mode() is True

    def test_returns_false_when_env_not_set(self):
        """Test returns False when OFFLINE_MODE not set."""
        env_backup = os.environ.pop('OFFLINE_MODE', None)
        try:
            assert is_offline_mode() is False
        finally:
            if env_backup:
                os.environ['OFFLINE_MODE'] = env_backup

    def test_returns_false_when_env_zero(self):
        """Test returns False when OFFLINE_MODE=0."""
        with patch.dict(os.environ, {'OFFLINE_MODE': '0'}):
            assert is_offline_mode() is False


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_s3_resource_handles_empty_prefix(self, temp_fixture_dir):
        """Test S3 resource handles empty prefix gracefully."""
        resource = OfflineS3Resource(temp_fixture_dir)

        result = resource._map_s3_prefix_to_fixture('')

        assert result is None

    def test_s3_resource_chaining_works(self, temp_fixture_dir):
        """Test method chaining works as expected."""
        resource = OfflineS3Resource(temp_fixture_dir)

        # This pattern matches boto3 usage
        result = resource.Bucket('bucket').objects().filter(
            Prefix='access/ierpt/tde_sales_by_product_by_fund/'
        )

        assert isinstance(result, list)

    def test_sts_client_handles_special_characters(self):
        """Test STS client handles special characters in role ARN."""
        client = OfflineSTSClient()

        result = client.assume_role(
            RoleArn='arn:aws:iam::123456:role/my-role-with-special_chars.v2',
            RoleSessionName='session-with-dashes'
        )

        assert 'Credentials' in result
