"""
Unit tests for src/data/adapters/s3_adapter.py.

Tests validate S3Adapter uses boto3 correctly with mocked AWS calls.
"""

import io
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.data.adapters.s3_adapter import S3Adapter


class TestS3AdapterInit:
    """Tests for S3Adapter initialization."""

    def test_init_with_valid_config(self, aws_config_dict):
        """S3Adapter should initialize with valid config."""
        adapter = S3Adapter(aws_config_dict)

        assert adapter._config == aws_config_dict
        assert adapter.source_type == "aws"

    def test_init_sets_default_paths(self, aws_config_dict):
        """S3Adapter should set default paths when none provided."""
        adapter = S3Adapter(aws_config_dict)

        assert 'sales' in adapter._paths
        assert 'rates' in adapter._paths
        assert 'weights' in adapter._paths
        assert 'macro' in adapter._paths
        assert 'output' in adapter._paths

    def test_init_with_custom_paths(self, aws_config_dict):
        """S3Adapter should accept custom paths."""
        custom_paths = {'sales': 'custom/sales/'}
        adapter = S3Adapter(aws_config_dict, paths=custom_paths)

        assert adapter._paths == custom_paths

    def test_lazy_connection(self, aws_config_dict):
        """S3Adapter should not connect on init (lazy connection)."""
        adapter = S3Adapter(aws_config_dict)

        assert adapter._s3_resource is None
        assert adapter._bucket is None


class TestS3AdapterSourceType:
    """Tests for source_type property."""

    def test_source_type_is_aws(self, aws_config_dict):
        """source_type should return 'aws'."""
        adapter = S3Adapter(aws_config_dict)
        assert adapter.source_type == "aws"


class TestS3AdapterCreateResource:
    """Tests for _create_s3_resource method."""

    def test_create_s3_resource_calls_sts(self, aws_config_dict):
        """_create_s3_resource should call STS assume_role."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            # Setup mocks
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts
            mock_resource.return_value = MagicMock()

            adapter = S3Adapter(aws_config_dict)
            adapter._create_s3_resource()

            mock_client.assert_called_once_with(
                'sts', endpoint_url=aws_config_dict['sts_endpoint_url']
            )
            mock_sts.assume_role.assert_called_once()

    def test_create_s3_resource_uses_credentials(self, aws_config_dict):
        """_create_s3_resource should use assumed role credentials."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts
            mock_resource.return_value = MagicMock()

            adapter = S3Adapter(aws_config_dict)
            adapter._create_s3_resource()

            mock_resource.assert_called_once_with(
                's3',
                aws_access_key_id='AKID',
                aws_secret_access_key='SECRET',
                aws_session_token='TOKEN',
            )

    def test_create_s3_resource_handles_role_failure(self, aws_config_dict):
        """_create_s3_resource should raise ValueError on role assumption failure."""
        with patch('boto3.client') as mock_client:
            mock_sts = MagicMock()
            mock_sts.assume_role.side_effect = Exception("Access denied")
            mock_client.return_value = mock_sts

            adapter = S3Adapter(aws_config_dict)

            with pytest.raises(ValueError, match="Failed to assume IAM role"):
                adapter._create_s3_resource()


class TestS3AdapterLoadData:
    """Tests for load data methods."""

    def test_load_sales_data_calls_s3(self, aws_config_dict, sample_sales_df):
        """load_sales_data should call S3 correctly."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            # Setup mocks
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts

            # Mock S3 resource
            mock_s3 = MagicMock()
            mock_bucket = MagicMock()
            mock_obj = MagicMock()

            # Create parquet buffer
            buffer = io.BytesIO()
            sample_sales_df.to_parquet(buffer, engine='pyarrow')
            buffer.seek(0)

            # Setup object iteration
            mock_bucket.objects.filter.return_value = [MagicMock(key='data/sales/file.parquet')]
            mock_obj.download_fileobj = lambda b: b.write(buffer.getvalue())
            mock_s3.Object.return_value = mock_obj
            mock_s3.Bucket.return_value = mock_bucket
            mock_resource.return_value = mock_s3

            adapter = S3Adapter(aws_config_dict)
            df = adapter.load_sales_data()

            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_load_sales_data_raises_on_empty(self, aws_config_dict):
        """load_sales_data should raise ValueError when no files found."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            # Setup mocks
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts

            mock_s3 = MagicMock()
            mock_bucket = MagicMock()
            mock_bucket.objects.filter.return_value = []  # No files
            mock_s3.Bucket.return_value = mock_bucket
            mock_resource.return_value = mock_s3

            adapter = S3Adapter(aws_config_dict)

            with pytest.raises(ValueError, match="No parquet files found"):
                adapter.load_sales_data()


class TestS3AdapterSaveOutput:
    """Tests for save_output method."""

    def test_save_output_parquet(self, aws_config_dict, sample_sales_df):
        """save_output should save DataFrame as parquet."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            # Setup mocks
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts

            mock_s3 = MagicMock()
            mock_bucket = MagicMock()
            mock_s3.Bucket.return_value = mock_bucket
            mock_resource.return_value = mock_s3

            adapter = S3Adapter(aws_config_dict)
            uri = adapter.save_output(sample_sales_df, "test_output", format="parquet")

            assert uri.startswith("s3://")
            assert uri.endswith(".parquet")
            mock_bucket.put_object.assert_called_once()

    def test_save_output_invalid_format(self, aws_config_dict, sample_sales_df):
        """save_output should raise ValueError for unsupported format."""
        with patch('boto3.client') as mock_client, patch('boto3.resource') as mock_resource:
            mock_sts = MagicMock()
            mock_sts.assume_role.return_value = {
                'Credentials': {
                    'AccessKeyId': 'AKID',
                    'SecretAccessKey': 'SECRET',
                    'SessionToken': 'TOKEN',
                }
            }
            mock_client.return_value = mock_sts
            mock_resource.return_value = MagicMock()

            adapter = S3Adapter(aws_config_dict)

            with pytest.raises(ValueError, match="Unsupported format"):
                adapter.save_output(sample_sales_df, "test", format="json")


class TestS3AdapterProtocol:
    """Tests verifying S3Adapter implements DataAdapterBase."""

    def test_implements_abstract_methods(self, aws_config_dict):
        """S3Adapter should implement all abstract methods."""
        adapter = S3Adapter(aws_config_dict)

        assert callable(adapter.load_sales_data)
        assert callable(adapter.load_competitive_rates)
        assert callable(adapter.load_market_weights)
        assert callable(adapter.load_macro_data)
        assert callable(adapter.save_output)
        assert hasattr(adapter, 'source_type')
