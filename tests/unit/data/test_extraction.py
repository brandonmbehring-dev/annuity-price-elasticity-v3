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
    # S3 operations with optional date suffix
    download_s3_parquet_with_optional_date_suffix,
    upload_parquet_to_s3_with_optional_date_suffix,
    # Error classes
    AWSConnectionError,
    DataLoadingError,
    DataValidationError,
    # Validation wrappers
    setup_aws_sts_client_with_validation,
    assume_iam_role_with_validation,
    setup_s3_resource_with_validation,
    # Sales/WINK data discovery and loading
    _discover_sales_parquet_files,
    _load_sales_dataframes_from_keys,
    _validate_sales_dataset_structure,
    discover_and_load_sales_data,
    _discover_wink_parquet_files,
    _load_wink_dataframes_from_keys,
    _validate_wink_dataset_structure,
    discover_and_load_wink_data,
    load_market_share_weights_from_s3,
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
        """create_sts_client should raise ValueError for invalid URL.

        Business impact: Using non-HTTPS STS endpoints exposes credentials
        in transit. Fail-fast prevents security misconfiguration that could
        compromise production data access.
        """
        config = {'sts_endpoint_url': 'http://invalid.com'}  # Must be https

        with pytest.raises(ValueError, match="Invalid STS endpoint URL"):
            create_sts_client(config)

    def test_create_sts_client_empty_url_raises(self):
        """create_sts_client should raise ValueError for empty URL.

        Business impact: Empty endpoint would cause silent failures in
        authentication, preventing access to sales and competitive rate data
        needed for elasticity estimation.
        """
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
        """assume_iam_role should raise ValueError for empty role_arn.

        Business impact: Missing role ARN prevents data access entirely.
        Failing fast with clear error enables rapid diagnosis rather than
        cryptic AWS errors downstream in the pipeline.
        """
        mock_sts = MagicMock()
        config = {'role_arn': '', 'xid': 'test'}

        with pytest.raises(ValueError, match="role_arn and session_name cannot be empty"):
            assume_iam_role(mock_sts, config)

    def test_assume_role_failure_raises(self):
        """assume_iam_role should raise ValueError on STS failure.

        Business impact: Authentication failures must be surfaced immediately.
        Silent fallback would cause missing data, leading to biased elasticity
        estimates or complete pipeline failure.
        """
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

    @patch('src.data.extraction.create_s3_resource_with_credentials')
    def test_setup_s3_resource_with_validation_success(self, mock_create):
        """setup_s3_resource_with_validation should return resource and bucket on success."""
        mock_resource = MagicMock()
        mock_bucket = MagicMock()
        mock_resource.Bucket.return_value = mock_bucket
        mock_create.return_value = mock_resource

        credentials = {'AccessKeyId': 'AKID', 'SecretAccessKey': 'SECRET', 'SessionToken': 'TOKEN'}
        result = setup_s3_resource_with_validation(credentials, 'test-bucket')

        assert result == (mock_resource, mock_bucket)
        mock_resource.Bucket.assert_called_once_with('test-bucket')

    @patch('src.data.extraction.create_s3_resource_with_credentials')
    def test_setup_s3_resource_with_validation_failure(self, mock_create):
        """setup_s3_resource_with_validation should raise AWSConnectionError on failure."""
        mock_create.side_effect = Exception("S3 resource creation failed")

        credentials = {'AccessKeyId': 'AKID', 'SecretAccessKey': 'SECRET', 'SessionToken': 'TOKEN'}

        with pytest.raises(AWSConnectionError, match="S3 resource creation failed"):
            setup_s3_resource_with_validation(credentials, 'test-bucket')


class TestCreateS3ClientException:
    """Tests for create_s3_client exception handling."""

    @patch('src.data.extraction.boto3')
    def test_create_s3_client_exception_raises_valueerror(self, mock_boto3):
        """create_s3_client should raise ValueError when boto3.client fails."""
        mock_boto3.client.side_effect = Exception("AWS connection failed")

        with pytest.raises(ValueError, match="Failed to create S3 client"):
            create_s3_client()


class TestDownloadParquetException:
    """Tests for download_parquet_from_s3_object exception handling."""

    def test_download_parquet_failure_raises(self):
        """download_parquet_from_s3_object should raise ValueError on S3 failure."""
        mock_s3 = MagicMock()
        mock_obj = MagicMock()
        mock_obj.get.side_effect = Exception("S3 access denied")
        mock_s3.Object.return_value = mock_obj

        with pytest.raises(ValueError, match="Failed to download parquet"):
            download_parquet_from_s3_object(mock_s3, 'bucket', 'key.parquet')


class TestConcatenateDataframeListException:
    """Tests for concatenate_dataframe_list exception handling."""

    def test_concatenate_empty_result_raises(self):
        """concatenate_dataframe_list should raise when result is empty."""
        # This happens when all DataFrames have 0 rows
        df1 = pd.DataFrame({'A': pd.Series([], dtype=int)})
        df2 = pd.DataFrame({'A': pd.Series([], dtype=int)})

        with pytest.raises(ValueError, match="Concatenated DataFrame is empty"):
            concatenate_dataframe_list([df1, df2])

    def test_concatenate_general_exception_raises(self):
        """concatenate_dataframe_list should wrap general exceptions."""
        # Pass incompatible types to force exception
        with pytest.raises(ValueError, match="Failed to concatenate DataFrames"):
            concatenate_dataframe_list([None])


class TestDownloadS3ParquetWithOptionalDateSuffix:
    """Tests for download_s3_parquet_with_optional_date_suffix function."""

    @patch('src.data.extraction.boto3')
    def test_download_with_date_suffix(self, mock_boto3, sample_sales_df):
        """download_s3_parquet_with_optional_date_suffix should append date suffix to key."""
        buffer = io.BytesIO()
        sample_sales_df.to_parquet(buffer, engine='pyarrow')
        buffer.seek(0)

        mock_client = MagicMock()
        mock_client.get_object.return_value = {'Body': MagicMock(read=lambda: buffer.getvalue())}
        mock_boto3.client.return_value = mock_client

        result = download_s3_parquet_with_optional_date_suffix(
            'test-bucket', 'data/sales', '2023-01-01'
        )

        assert isinstance(result, pd.DataFrame)
        mock_client.get_object.assert_called_once_with(
            Bucket='test-bucket', Key='data/sales_2023-01-01.parquet'
        )

    @patch('src.data.extraction.boto3')
    def test_download_without_date_suffix(self, mock_boto3, sample_sales_df):
        """download_s3_parquet_with_optional_date_suffix should omit date suffix when None."""
        buffer = io.BytesIO()
        sample_sales_df.to_parquet(buffer, engine='pyarrow')
        buffer.seek(0)

        mock_client = MagicMock()
        mock_client.get_object.return_value = {'Body': MagicMock(read=lambda: buffer.getvalue())}
        mock_boto3.client.return_value = mock_client

        result = download_s3_parquet_with_optional_date_suffix(
            'test-bucket', 'data/sales', None
        )

        assert isinstance(result, pd.DataFrame)
        mock_client.get_object.assert_called_once_with(
            Bucket='test-bucket', Key='data/sales.parquet'
        )

    @patch('src.data.extraction.boto3')
    def test_download_failure_raises(self, mock_boto3):
        """download_s3_parquet_with_optional_date_suffix should raise on failure."""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("S3 access denied")
        mock_boto3.client.return_value = mock_client

        with pytest.raises(ValueError, match="Failed to download S3 parquet"):
            download_s3_parquet_with_optional_date_suffix(
                'test-bucket', 'data/sales', '2023-01-01'
            )


class TestUploadParquetToS3WithOptionalDateSuffix:
    """Tests for upload_parquet_to_s3_with_optional_date_suffix function."""

    def test_upload_with_date_suffix(self, sample_sales_df):
        """upload_parquet_to_s3_with_optional_date_suffix should append date suffix."""
        mock_client = MagicMock()
        buffer = convert_dataframe_to_parquet_buffer(sample_sales_df)

        upload_parquet_to_s3_with_optional_date_suffix(
            mock_client, buffer, 'test-bucket', 'data/sales', '2023-01-01'
        )

        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs['Key'] == 'data/sales_2023-01-01.parquet'

    def test_upload_without_date_suffix(self, sample_sales_df):
        """upload_parquet_to_s3_with_optional_date_suffix should omit date suffix when None."""
        mock_client = MagicMock()
        buffer = convert_dataframe_to_parquet_buffer(sample_sales_df)

        upload_parquet_to_s3_with_optional_date_suffix(
            mock_client, buffer, 'test-bucket', 'data/sales', None
        )

        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs['Key'] == 'data/sales.parquet'

    def test_upload_failure_raises(self, sample_sales_df):
        """upload_parquet_to_s3_with_optional_date_suffix should raise on failure."""
        mock_client = MagicMock()
        mock_client.put_object.side_effect = Exception("S3 upload denied")
        buffer = convert_dataframe_to_parquet_buffer(sample_sales_df)

        with pytest.raises(ValueError, match="Failed to upload parquet"):
            upload_parquet_to_s3_with_optional_date_suffix(
                mock_client, buffer, 'test-bucket', 'data/sales', '2023-01-01'
            )


class TestConvertDataframeToParquetBufferException:
    """Tests for convert_dataframe_to_parquet_buffer exception handling."""

    def test_convert_exception_raises(self):
        """convert_dataframe_to_parquet_buffer should wrap parquet conversion errors."""
        # Create DataFrame that will fail parquet conversion
        df = pd.DataFrame({'A': [1, 2, 3]})

        with patch('pandas.DataFrame.to_parquet', side_effect=Exception("Pyarrow error")):
            with pytest.raises(ValueError, match="Failed to convert DataFrame to parquet"):
                convert_dataframe_to_parquet_buffer(df)


class TestUploadParquetBufferException:
    """Tests for upload_parquet_buffer_to_s3 exception handling."""

    def test_upload_exception_raises(self, sample_sales_df):
        """upload_parquet_buffer_to_s3 should wrap upload errors."""
        mock_client = MagicMock()
        mock_client.upload_fileobj.side_effect = Exception("Network error")
        buffer = convert_dataframe_to_parquet_buffer(sample_sales_df)

        with pytest.raises(ValueError, match="Failed to upload to s3://"):
            upload_parquet_buffer_to_s3(mock_client, buffer, 'bucket', 'key.parquet')


class TestDiscoverSalesParquetFiles:
    """Tests for _discover_sales_parquet_files function."""

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_sales_success(self, mock_list):
        """_discover_sales_parquet_files should return parquet keys."""
        mock_list.return_value = ['access/ierpt/tde_sales_by_product_by_fund/file1.parquet']
        mock_bucket = MagicMock()

        result = _discover_sales_parquet_files(mock_bucket)

        assert len(result) == 1
        mock_list.assert_called_once_with(mock_bucket, "access/ierpt/tde_sales_by_product_by_fund/")

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_sales_empty_raises(self, mock_list):
        """_discover_sales_parquet_files should raise when no files found."""
        mock_list.return_value = []
        mock_bucket = MagicMock()

        with pytest.raises(DataLoadingError, match="No sales data parquet files found"):
            _discover_sales_parquet_files(mock_bucket)

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_sales_exception_raises(self, mock_list):
        """_discover_sales_parquet_files should wrap exceptions."""
        mock_list.side_effect = Exception("S3 bucket access denied")
        mock_bucket = MagicMock()

        with pytest.raises(DataLoadingError, match="Sales data discovery failed"):
            _discover_sales_parquet_files(mock_bucket)


class TestLoadSalesDataframesFromKeys:
    """Tests for _load_sales_dataframes_from_keys function."""

    @patch('src.data.extraction.download_parquet_from_s3_object')
    @patch('src.data.extraction.concatenate_dataframe_list')
    def test_load_sales_success(self, mock_concat, mock_download):
        """_load_sales_dataframes_from_keys should load and concatenate DataFrames."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        mock_download.return_value = df
        mock_concat.return_value = df

        mock_s3 = MagicMock()
        result = _load_sales_dataframes_from_keys(mock_s3, 'bucket', ['key1.parquet'])

        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once()
        mock_concat.assert_called_once()

    @patch('src.data.extraction.download_parquet_from_s3_object')
    def test_load_sales_file_failure_raises(self, mock_download):
        """_load_sales_dataframes_from_keys should raise on file load failure."""
        mock_download.side_effect = Exception("File corrupted")
        mock_s3 = MagicMock()

        with pytest.raises(DataLoadingError, match="Failed to load parquet file"):
            _load_sales_dataframes_from_keys(mock_s3, 'bucket', ['bad_file.parquet'])

    @patch('src.data.extraction.download_parquet_from_s3_object')
    def test_load_sales_all_empty_raises(self, mock_download):
        """_load_sales_dataframes_from_keys should raise when all DataFrames are empty."""
        mock_download.return_value = pd.DataFrame()
        mock_s3 = MagicMock()

        with pytest.raises(DataLoadingError, match="No valid sales data loaded"):
            _load_sales_dataframes_from_keys(mock_s3, 'bucket', ['empty.parquet'])


class TestValidateSalesDatasetStructure:
    """Tests for _validate_sales_dataset_structure function."""

    def test_validate_sales_empty_raises(self):
        """_validate_sales_dataset_structure should raise for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError, match="Combined sales dataset is empty"):
            _validate_sales_dataset_structure(df)

    def test_validate_sales_missing_columns_raises(self):
        """_validate_sales_dataset_structure should raise for missing columns."""
        df = pd.DataFrame({'other_column': [1, 2, 3]})

        with pytest.raises(DataValidationError, match="Missing required columns"):
            _validate_sales_dataset_structure(df)

    def test_validate_sales_valid_passes(self):
        """_validate_sales_dataset_structure should pass for valid DataFrame."""
        df = pd.DataFrame({
            'application_signed_date': pd.date_range('2023-01-01', periods=3),
            'contract_issue_date': pd.date_range('2023-01-01', periods=3),
            'product_name': ['Product A'] * 3,
        })

        # Should not raise
        _validate_sales_dataset_structure(df)


class TestDiscoverWinkParquetFiles:
    """Tests for _discover_wink_parquet_files function."""

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_wink_success(self, mock_list):
        """_discover_wink_parquet_files should return parquet keys."""
        mock_list.return_value = ['access/ierpt/wink_ann_product_rates/file1.parquet']
        mock_bucket = MagicMock()

        result = _discover_wink_parquet_files(mock_bucket)

        assert len(result) == 1
        mock_list.assert_called_once_with(mock_bucket, "access/ierpt/wink_ann_product_rates/")

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_wink_empty_raises(self, mock_list):
        """_discover_wink_parquet_files should raise when no files found."""
        mock_list.return_value = []
        mock_bucket = MagicMock()

        with pytest.raises(DataLoadingError, match="No WINK competitive rate files found"):
            _discover_wink_parquet_files(mock_bucket)

    @patch('src.data.extraction.list_parquet_objects_with_prefix')
    def test_discover_wink_exception_raises(self, mock_list):
        """_discover_wink_parquet_files should wrap exceptions."""
        mock_list.side_effect = Exception("S3 bucket access denied")
        mock_bucket = MagicMock()

        with pytest.raises(DataLoadingError, match="WINK data discovery failed"):
            _discover_wink_parquet_files(mock_bucket)


class TestLoadWinkDataframesFromKeys:
    """Tests for _load_wink_dataframes_from_keys function."""

    @patch('src.data.extraction.download_parquet_from_s3_object')
    @patch('src.data.extraction.concatenate_dataframe_list')
    def test_load_wink_success(self, mock_concat, mock_download):
        """_load_wink_dataframes_from_keys should load and concatenate DataFrames."""
        df = pd.DataFrame({'date': ['2023-01-01'], 'rate_1': [5.0]})
        mock_download.return_value = df
        mock_concat.return_value = df

        mock_s3 = MagicMock()
        result = _load_wink_dataframes_from_keys(mock_s3, 'bucket', ['key1.parquet'])

        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once()
        mock_concat.assert_called_once()

    @patch('src.data.extraction.download_parquet_from_s3_object')
    def test_load_wink_file_failure_raises(self, mock_download):
        """_load_wink_dataframes_from_keys should raise on file load failure."""
        mock_download.side_effect = Exception("File corrupted")
        mock_s3 = MagicMock()

        with pytest.raises(DataLoadingError, match="Failed to load WINK parquet file"):
            _load_wink_dataframes_from_keys(mock_s3, 'bucket', ['bad_file.parquet'])

    @patch('src.data.extraction.download_parquet_from_s3_object')
    def test_load_wink_all_empty_raises(self, mock_download):
        """_load_wink_dataframes_from_keys should raise when all DataFrames are empty."""
        mock_download.return_value = pd.DataFrame()
        mock_s3 = MagicMock()

        with pytest.raises(DataLoadingError, match="No valid WINK data loaded"):
            _load_wink_dataframes_from_keys(mock_s3, 'bucket', ['empty.parquet'])


class TestValidateWinkDatasetStructure:
    """Tests for _validate_wink_dataset_structure function."""

    def test_validate_wink_empty_raises(self):
        """_validate_wink_dataset_structure should raise for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError, match="Combined WINK dataset is empty"):
            _validate_wink_dataset_structure(df)

    def test_validate_wink_insufficient_rate_columns_raises(self):
        """_validate_wink_dataset_structure should raise for insufficient rate columns."""
        df = pd.DataFrame({
            'date': ['2023-01-01'],
            'rate_1': [5.0],
            'rate_2': [5.0],
            # Only 2 rate columns, need at least 5
        })

        with pytest.raises(DataValidationError, match="Insufficient competitive rate columns"):
            _validate_wink_dataset_structure(df)

    def test_validate_wink_missing_date_column_raises(self):
        """_validate_wink_dataset_structure should raise for missing date column."""
        df = pd.DataFrame({
            'rate_1': [5.0], 'rate_2': [5.0], 'rate_3': [5.0],
            'rate_4': [5.0], 'rate_5': [5.0],
        })

        with pytest.raises(DataValidationError, match="No 'date' column found"):
            _validate_wink_dataset_structure(df)

    def test_validate_wink_valid_passes(self):
        """_validate_wink_dataset_structure should pass for valid DataFrame."""
        df = pd.DataFrame({
            'date': ['2023-01-01'],
            'rate_1': [5.0], 'rate_2': [5.0], 'rate_3': [5.0],
            'rate_4': [5.0], 'rate_5': [5.0],
        })

        # Should not raise
        _validate_wink_dataset_structure(df)


class TestDiscoverAndLoadSalesData:
    """Tests for discover_and_load_sales_data orchestration function."""

    @patch('src.validation.pipeline_validation_helpers.validate_extraction_output')
    @patch('src.data.extraction._validate_sales_dataset_structure')
    @patch('src.data.extraction._load_sales_dataframes_from_keys')
    @patch('src.data.extraction._discover_sales_parquet_files')
    def test_discover_and_load_sales_success(self, mock_discover, mock_load, mock_validate, mock_prod_validate):
        """discover_and_load_sales_data should orchestrate full pipeline."""
        df = pd.DataFrame({
            'application_signed_date': pd.date_range('2023-01-01', periods=3),
            'contract_issue_date': pd.date_range('2023-01-01', periods=3),
            'product_name': ['Product A'] * 3,
        })
        mock_discover.return_value = ['file1.parquet']
        mock_load.return_value = df
        mock_prod_validate.return_value = df

        mock_bucket = MagicMock()
        mock_s3 = MagicMock()

        result = discover_and_load_sales_data(mock_bucket, mock_s3, 'test-bucket')

        assert isinstance(result, pd.DataFrame)
        mock_discover.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()
        mock_prod_validate.assert_called_once()


class TestDiscoverAndLoadWinkData:
    """Tests for discover_and_load_wink_data orchestration function."""

    @patch('src.validation.pipeline_validation_helpers.validate_extraction_output')
    @patch('src.data.extraction._validate_wink_dataset_structure')
    @patch('src.data.extraction._load_wink_dataframes_from_keys')
    @patch('src.data.extraction._discover_wink_parquet_files')
    def test_discover_and_load_wink_success(self, mock_discover, mock_load, mock_validate, mock_prod_validate):
        """discover_and_load_wink_data should orchestrate full pipeline."""
        df = pd.DataFrame({
            'date': ['2023-01-01'],
            'product_name': ['Product A'],
            'rate_1': [5.0], 'rate_2': [5.0], 'rate_3': [5.0],
            'rate_4': [5.0], 'rate_5': [5.0],
        })
        mock_discover.return_value = ['file1.parquet']
        mock_load.return_value = df
        mock_prod_validate.return_value = df

        mock_bucket = MagicMock()
        mock_s3 = MagicMock()

        result = discover_and_load_wink_data(mock_bucket, mock_s3, 'test-bucket')

        assert isinstance(result, pd.DataFrame)
        mock_discover.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()
        mock_prod_validate.assert_called_once()


class TestLoadMarketShareWeightsFromS3:
    """Tests for load_market_share_weights_from_s3 function."""

    @patch('pandas.read_parquet')
    def test_load_market_share_weights_success(self, mock_read):
        """load_market_share_weights_from_s3 should return DataFrame."""
        df = pd.DataFrame({'company': ['A', 'B'], 'weight': [0.6, 0.4]})
        mock_read.return_value = df

        result = load_market_share_weights_from_s3('s3://bucket/weights.parquet')

        assert isinstance(result, pd.DataFrame)
        mock_read.assert_called_once_with('s3://bucket/weights.parquet')

    @patch('pandas.read_parquet')
    def test_load_market_share_weights_failure_raises(self, mock_read):
        """load_market_share_weights_from_s3 should raise on failure."""
        mock_read.side_effect = Exception("S3 access denied")

        with pytest.raises(ValueError, match="Failed to load market share weights"):
            load_market_share_weights_from_s3('s3://bucket/weights.parquet')
