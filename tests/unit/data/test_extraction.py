"""
Unit tests for src/data/extraction.py.

Tests validate AWS data extraction functions work correctly with mocked AWS calls.
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.extraction import (
    # Core AWS functions
    create_sts_client,
    assume_iam_role,
    create_s3_resource_with_credentials,
    create_s3_client,
    # S3 operations
    list_parquet_objects_with_prefix,
    download_parquet_from_s3_object,
    concatenate_dataframe_list,
    create_s3_file_path,
    # Buffer operations
    convert_dataframe_to_parquet_buffer,
    upload_parquet_buffer_to_s3,
    # Error classes
    AWSConnectionError,
    DataLoadingError,
    DataValidationError,
    # Validation wrappers
    setup_aws_sts_client_with_validation,
    assume_iam_role_with_validation,
    setup_s3_resource_with_validation,
)


class TestCreateStsClient:
    """Tests for create_sts_client function."""

    @patch('src.data.extraction.boto3')
    def test_create_sts_client_returns_client(self, mock_boto3):
        """create_sts_client should return boto3 STS client."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        config = {'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com'}
        result = create_sts_client(config)

        assert result == mock_client
        mock_boto3.client.assert_called_once_with(
            'sts', endpoint_url='https://sts.us-east-1.amazonaws.com'
        )

    def test_create_sts_client_invalid_url_raises(self):
        """create_sts_client should raise ValueError for invalid URL."""
        config = {'sts_endpoint_url': 'http://invalid.com'}  # Must be https

        with pytest.raises(ValueError, match="Invalid STS endpoint URL"):
            create_sts_client(config)

    def test_create_sts_client_empty_url_raises(self):
        """create_sts_client should raise ValueError for empty URL."""
        config = {'sts_endpoint_url': ''}

        with pytest.raises(ValueError, match="Invalid STS endpoint URL"):
            create_sts_client(config)


class TestAssumeIamRole:
    """Tests for assume_iam_role function."""

    def test_assume_role_returns_credentials(self):
        """assume_iam_role should return assumed role object."""
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'AKID',
                'SecretAccessKey': 'SECRET',
                'SessionToken': 'TOKEN',
            }
        }

        config = {
            'role_arn': 'arn:aws:iam::123:role/TestRole',
            'xid': 'test_session',
        }

        result = assume_iam_role(mock_sts, config)

        assert 'Credentials' in result
        mock_sts.assume_role.assert_called_once_with(
            RoleArn='arn:aws:iam::123:role/TestRole',
            RoleSessionName='test_session'
        )

    def test_assume_role_empty_arn_raises(self):
        """assume_iam_role should raise ValueError for empty role_arn."""
        mock_sts = MagicMock()
        config = {'role_arn': '', 'xid': 'test'}

        with pytest.raises(ValueError, match="role_arn and session_name cannot be empty"):
            assume_iam_role(mock_sts, config)

    def test_assume_role_failure_raises(self):
        """assume_iam_role should raise ValueError on STS failure."""
        mock_sts = MagicMock()
        mock_sts.assume_role.side_effect = Exception("Access denied")

        config = {
            'role_arn': 'arn:aws:iam::123:role/TestRole',
            'xid': 'test_session',
        }

        with pytest.raises(ValueError, match="Failed to assume role"):
            assume_iam_role(mock_sts, config)


class TestCreateS3ResourceWithCredentials:
    """Tests for create_s3_resource_with_credentials function."""

    @patch('src.data.extraction.boto3')
    def test_create_s3_resource_returns_resource(self, mock_boto3):
        """create_s3_resource_with_credentials should return S3 resource."""
        mock_resource = MagicMock()
        mock_boto3.resource.return_value = mock_resource

        credentials = {
            'AccessKeyId': 'AKID',
            'SecretAccessKey': 'SECRET',
            'SessionToken': 'TOKEN',
        }

        result = create_s3_resource_with_credentials(credentials)

        assert result == mock_resource
        mock_boto3.resource.assert_called_once_with(
            's3',
            aws_access_key_id='AKID',
            aws_secret_access_key='SECRET',
            aws_session_token='TOKEN'
        )

    def test_create_s3_resource_missing_keys_raises(self):
        """create_s3_resource_with_credentials should raise for missing keys."""
        credentials = {'AccessKeyId': 'AKID'}  # Missing SecretAccessKey and SessionToken

        with pytest.raises(ValueError, match="Credentials missing required keys"):
            create_s3_resource_with_credentials(credentials)


class TestCreateS3Client:
    """Tests for create_s3_client function."""

    @patch('src.data.extraction.boto3')
    def test_create_s3_client_returns_client(self, mock_boto3):
        """create_s3_client should return S3 client."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        result = create_s3_client()

        assert result == mock_client
        mock_boto3.client.assert_called_once_with('s3')


class TestListParquetObjectsWithPrefix:
    """Tests for list_parquet_objects_with_prefix function."""

    def test_list_objects_returns_keys(self):
        """list_parquet_objects_with_prefix should return sorted parquet keys."""
        mock_bucket = MagicMock()

        # Mock object iteration
        mock_obj1 = MagicMock()
        mock_obj1.key = 'data/file1.parquet'
        mock_obj2 = MagicMock()
        mock_obj2.key = 'data/file2.parquet'
        mock_obj3 = MagicMock()
        mock_obj3.key = 'data/file.csv'  # Should be filtered out

        mock_bucket.objects.filter.return_value = [mock_obj1, mock_obj2, mock_obj3]

        result = list_parquet_objects_with_prefix(mock_bucket, 'data/')

        assert len(result) == 2
        assert all(key.endswith('.parquet') for key in result)
        assert result == sorted(result)  # Should be sorted

    def test_list_objects_empty_prefix_raises(self):
        """list_parquet_objects_with_prefix should raise for empty prefix."""
        mock_bucket = MagicMock()

        with pytest.raises(ValueError, match="Prefix cannot be empty"):
            list_parquet_objects_with_prefix(mock_bucket, '')

    def test_list_objects_no_files_raises(self):
        """list_parquet_objects_with_prefix should raise when no files found."""
        mock_bucket = MagicMock()
        mock_bucket.objects.filter.return_value = []

        with pytest.raises(ValueError, match="No parquet files found"):
            list_parquet_objects_with_prefix(mock_bucket, 'empty/')


class TestDownloadParquetFromS3Object:
    """Tests for download_parquet_from_s3_object function."""

    def test_download_parquet_returns_dataframe(self, sample_sales_df):
        """download_parquet_from_s3_object should return DataFrame."""
        # Create parquet buffer
        buffer = io.BytesIO()
        sample_sales_df.to_parquet(buffer, engine='pyarrow')
        buffer.seek(0)

        mock_s3 = MagicMock()
        mock_obj = MagicMock()
        mock_obj.get.return_value = {'Body': MagicMock(read=lambda: buffer.getvalue())}
        mock_s3.Object.return_value = mock_obj

        result = download_parquet_from_s3_object(mock_s3, 'bucket', 'key.parquet')

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_sales_df)

    def test_download_parquet_empty_params_raises(self):
        """download_parquet_from_s3_object should raise for empty params."""
        mock_s3 = MagicMock()

        with pytest.raises(ValueError, match="bucket_name and key cannot be empty"):
            download_parquet_from_s3_object(mock_s3, '', 'key')

        with pytest.raises(ValueError, match="bucket_name and key cannot be empty"):
            download_parquet_from_s3_object(mock_s3, 'bucket', '')


class TestConcatenateDataframeList:
    """Tests for concatenate_dataframe_list function."""

    def test_concatenate_returns_combined(self):
        """concatenate_dataframe_list should combine DataFrames."""
        df1 = pd.DataFrame({'A': [1, 2]})
        df2 = pd.DataFrame({'A': [3, 4]})

        result = concatenate_dataframe_list([df1, df2])

        assert len(result) == 4
        assert list(result['A']) == [1, 2, 3, 4]

    def test_concatenate_empty_list_raises(self):
        """concatenate_dataframe_list should raise for empty list."""
        with pytest.raises(ValueError, match="DataFrame list cannot be empty"):
            concatenate_dataframe_list([])


class TestCreateS3FilePath:
    """Tests for create_s3_file_path function."""

    def test_create_path_with_date(self):
        """create_s3_file_path should create path with date suffix."""
        result = create_s3_file_path('data/sales', 'flexguard', '2023-01-01')

        assert result == 'data/sales/flexguard_2023-01-01.parquet'

    def test_create_path_without_date(self):
        """create_s3_file_path should create path without date suffix."""
        result = create_s3_file_path('data/sales', 'flexguard', '')

        assert result == 'data/sales/flexguard.parquet'

    def test_create_path_empty_params_raises(self):
        """create_s3_file_path should raise for empty params."""
        with pytest.raises(ValueError, match="base_path and filename cannot be empty"):
            create_s3_file_path('', 'filename', '2023-01-01')


class TestConvertDataframeToParquetBuffer:
    """Tests for convert_dataframe_to_parquet_buffer function."""

    def test_convert_returns_buffer(self, sample_sales_df):
        """convert_dataframe_to_parquet_buffer should return BytesIO buffer."""
        result = convert_dataframe_to_parquet_buffer(sample_sales_df)

        assert isinstance(result, io.BytesIO)
        assert result.tell() == 0  # Should be at start

        # Should be readable as parquet
        loaded = pd.read_parquet(result)
        assert len(loaded) == len(sample_sales_df)

    def test_convert_empty_df_raises(self):
        """convert_dataframe_to_parquet_buffer should raise for empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot convert empty DataFrame"):
            convert_dataframe_to_parquet_buffer(empty_df)


class TestUploadParquetBufferToS3:
    """Tests for upload_parquet_buffer_to_s3 function."""

    def test_upload_calls_s3_client(self, sample_sales_df):
        """upload_parquet_buffer_to_s3 should call S3 client."""
        mock_client = MagicMock()
        buffer = convert_dataframe_to_parquet_buffer(sample_sales_df)

        upload_parquet_buffer_to_s3(mock_client, buffer, 'bucket', 'key.parquet')

        mock_client.upload_fileobj.assert_called_once()

    def test_upload_empty_params_raises(self):
        """upload_parquet_buffer_to_s3 should raise for empty params."""
        mock_client = MagicMock()
        buffer = io.BytesIO(b'data')

        with pytest.raises(ValueError, match="bucket_name and key cannot be empty"):
            upload_parquet_buffer_to_s3(mock_client, buffer, '', 'key')

    def test_upload_empty_buffer_raises(self):
        """upload_parquet_buffer_to_s3 should raise for empty buffer."""
        mock_client = MagicMock()
        empty_buffer = io.BytesIO()

        with pytest.raises(ValueError, match="Buffer appears to be empty"):
            upload_parquet_buffer_to_s3(mock_client, empty_buffer, 'bucket', 'key')


class TestErrorClasses:
    """Tests for custom error classes."""

    def test_aws_connection_error(self):
        """AWSConnectionError should be Exception subclass."""
        error = AWSConnectionError("Connection failed")
        assert isinstance(error, Exception)
        assert str(error) == "Connection failed"

    def test_data_loading_error(self):
        """DataLoadingError should be Exception subclass."""
        error = DataLoadingError("Loading failed")
        assert isinstance(error, Exception)
        assert str(error) == "Loading failed"

    def test_data_validation_error(self):
        """DataValidationError should be Exception subclass."""
        error = DataValidationError("Validation failed")
        assert isinstance(error, Exception)
        assert str(error) == "Validation failed"


class TestValidationWrappers:
    """Tests for validation wrapper functions."""

    @patch('src.data.extraction.create_sts_client')
    def test_setup_aws_sts_client_with_validation_success(self, mock_create):
        """setup_aws_sts_client_with_validation should return client on success."""
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        config = {'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com'}
        result = setup_aws_sts_client_with_validation(config)

        assert result == mock_client

    @patch('src.data.extraction.create_sts_client')
    def test_setup_aws_sts_client_with_validation_failure(self, mock_create):
        """setup_aws_sts_client_with_validation should raise AWSConnectionError on failure."""
        mock_create.side_effect = Exception("Connection failed")

        config = {'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com'}

        with pytest.raises(AWSConnectionError, match="STS client creation failed"):
            setup_aws_sts_client_with_validation(config)

    @patch('src.data.extraction.assume_iam_role')
    def test_assume_iam_role_with_validation_success(self, mock_assume):
        """assume_iam_role_with_validation should return credentials on success."""
        mock_response = {'Credentials': {'AccessKeyId': 'AKID'}}
        mock_assume.return_value = mock_response

        mock_sts = MagicMock()
        config = {'role_arn': 'arn:aws:iam::123:role/Test', 'xid': 'test'}

        result = assume_iam_role_with_validation(mock_sts, config)

        assert result == mock_response

    @patch('src.data.extraction.assume_iam_role')
    def test_assume_iam_role_with_validation_failure(self, mock_assume):
        """assume_iam_role_with_validation should raise AWSConnectionError on failure."""
        mock_assume.side_effect = Exception("Role assumption failed")

        mock_sts = MagicMock()
        config = {'role_arn': 'arn:aws:iam::123:role/Test', 'xid': 'test'}

        with pytest.raises(AWSConnectionError, match="IAM role assumption failed"):
            assume_iam_role_with_validation(mock_sts, config)
