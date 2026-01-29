"""
Data loading and AWS operations for clean_v0 pipeline.

This module handles all data extraction operations used in 00_clean_v0.ipynb:
- AWS S3 operations (STS, resource creation, data loading/saving)
- Data loading and concatenation
- S3 upload operations

Following CODING_STANDARDS.md principles:
- Single responsibility functions (10-30 lines max)
- Full type hints with TypedDict configurations
- Immutable operations (return new DataFrames)
- Explicit error handling with clear messages
"""

import boto3
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from src.config.pipeline_config import AWSConfig

# =============================================================================
# CONTEXT ANCHOR: DATA EXTRACTION OBJECTIVES
# =============================================================================
# PURPOSE: Secure, reliable data extraction from AWS S3 with comprehensive error handling
# USED BY: notebooks/00_data_pipeline_refactored.ipynb (primary client), all data operations
# DEPENDENCIES: boto3, pandas, src.config.pipeline_config (AWSConfig)
# LAST VALIDATED: 2025-11-12 (v3.0 cleanup - import patterns cleaned)
# PATTERN STATUS: CANONICAL (single data extraction approach)
#
# ARCHITECTURAL FLOW: AWS config → STS credentials → S3 operations → validated DataFrames → pipeline
# SUCCESS CRITERIA: All extractions complete without errors, business rule validation passes
# INTEGRATION: Works with config_builder AWS configurations seamlessly
# MAINTENANCE: Keep functions atomic (10-30 lines), maintain full type hints and error handling


def create_sts_client(config: AWSConfig) -> boto3.client:
    """Create AWS STS client for role assumption.

    Parameters
    ----------
    config : AWSConfig
        AWS configuration containing STS endpoint URL and other connection parameters

    Returns
    -------
    boto3.client
        Configured STS client for role assumption operations

    Raises
    ------
    ValueError
        If STS endpoint URL is invalid or missing

    Examples
    --------
    >>> aws_config = {'sts_endpoint_url': 'https://sts.us-east-1.amazonaws.com', ...}
    >>> sts_client = create_sts_client(aws_config)
    >>> assert sts_client.meta.service_model.service_name == 'sts'
    """
    endpoint_url = config['sts_endpoint_url']
    if not endpoint_url or not endpoint_url.startswith('https://'):
        raise ValueError(f"Invalid STS endpoint URL: {endpoint_url}")
    return boto3.client('sts', endpoint_url=endpoint_url)


def assume_iam_role(sts_client: boto3.client, config: AWSConfig) -> Dict[str, Any]:
    """Assume IAM role and return temporary credentials.

    Parameters
    ----------
    sts_client : boto3.client
        AWS STS client for role assumption operations
    config : AWSConfig
        AWS configuration containing role ARN and session identifiers

    Returns
    -------
    Dict[str, Any]
        AWS response containing temporary credentials and role information

    Raises
    ------
    ValueError
        If role ARN or session name is empty, or if role assumption fails

    Examples
    --------
    >>> sts_client = boto3.client('sts')
    >>> config = {'role_arn': 'arn:aws:iam::123:role/DataRole', 'xid': 'user123', ...}
    >>> credentials = assume_iam_role(sts_client, config)
    >>> assert 'Credentials' in credentials
    """
    role_arn = config['role_arn']
    session_name = config['xid']

    if not role_arn or not session_name:
        raise ValueError("role_arn and session_name cannot be empty")

    try:
        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name
        )
        return assumed_role_object
    except Exception as e:
        raise ValueError(f"Failed to assume role {role_arn}: {e}") from e


def create_s3_resource_with_credentials(credentials: Dict[str, str]) -> boto3.resource:
    """Create S3 resource using temporary credentials.

    Parameters
    ----------
    credentials : Dict[str, str]
        AWS temporary credentials containing AccessKeyId, SecretAccessKey, and SessionToken

    Returns
    -------
    boto3.resource
        Configured S3 resource for data operations

    Raises
    ------
    ValueError
        If required credential keys are missing

    Examples
    --------
    >>> creds = {'AccessKeyId': 'ASIA...', 'SecretAccessKey': '...', 'SessionToken': '...'}
    >>> s3 = create_s3_resource_with_credentials(creds)
    >>> assert s3.meta.service_name == 's3'
    """
    required_keys = ['AccessKeyId', 'SecretAccessKey', 'SessionToken']
    if not all(key in credentials for key in required_keys):
        raise ValueError(f"Credentials missing required keys: {required_keys}")

    return boto3.resource(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )


def create_s3_client() -> boto3.client:
    """Create S3 client using default credentials.

    Returns
    -------
    boto3.client
        S3 client configured with default AWS credentials

    Raises
    ------
    ValueError
        If S3 client creation fails due to missing or invalid credentials

    Examples
    --------
    >>> s3_client = create_s3_client()
    >>> assert s3_client.meta.service_model.service_name == 's3'
    """
    try:
        return boto3.client('s3')
    except Exception as e:
        raise ValueError(f"Failed to create S3 client: {e}") from e


def list_parquet_objects_with_prefix(bucket: boto3.resource, prefix: str) -> List[str]:
    """List all parquet objects in S3 bucket with given prefix.

    Parameters
    ----------
    bucket : boto3.resource
        S3 bucket resource to search for parquet files
    prefix : str
        S3 key prefix to filter objects (e.g., 'data/sales/')

    Returns
    -------
    List[str]
        Sorted list of S3 keys for parquet files matching the prefix

    Raises
    ------
    ValueError
        If prefix is empty or no parquet files are found

    Examples
    --------
    >>> bucket = s3.Bucket('my-bucket')
    >>> keys = list_parquet_objects_with_prefix(bucket, 'data/sales/')
    >>> assert all(key.endswith('.parquet') for key in keys)
    """
    if not prefix:
        raise ValueError("Prefix cannot be empty")

    try:
        parquet_keys = []
        for obj in bucket.objects.filter(Prefix=prefix):
            if obj.key.endswith('.parquet'):
                parquet_keys.append(obj.key)

        if not parquet_keys:
            raise ValueError(f"No parquet files found with prefix: {prefix}")

        return sorted(parquet_keys)
    except Exception as e:
        raise ValueError(f"Failed to list objects with prefix {prefix}: {e}") from e


def download_parquet_from_s3_object(s3_resource: boto3.resource, bucket_name: str, key: str) -> pd.DataFrame:
    """Download and read parquet file from S3.

    Parameters
    ----------
    s3_resource : boto3.resource
        S3 resource for accessing objects
    bucket_name : str
        Name of the S3 bucket containing the parquet file
    key : str
        S3 object key path to the parquet file

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the parquet file

    Raises
    ------
    ValueError
        If bucket_name or key is empty, or if download/parsing fails

    Examples
    --------
    >>> s3 = boto3.resource('s3')
    >>> df = download_parquet_from_s3_object(s3, 'my-bucket', 'data/sales.parquet')
    >>> assert isinstance(df, pd.DataFrame)
    """
    if not bucket_name or not key:
        raise ValueError("bucket_name and key cannot be empty")

    try:
        obj = s3_resource.Object(bucket_name, key)
        buffer = io.BytesIO(obj.get()['Body'].read())
        return pd.read_parquet(buffer)
    except Exception as e:
        raise ValueError(f"Failed to download parquet from {bucket_name}/{key}: {e}") from e


def concatenate_dataframe_list(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate list of DataFrames with validation.

    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of pandas DataFrames to concatenate

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with reset index

    Raises
    ------
    ValueError
        If dataframe list is empty or concatenation results in empty DataFrame

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2]})
    >>> df2 = pd.DataFrame({'A': [3, 4]})
    >>> result = concatenate_dataframe_list([df1, df2])
    >>> assert len(result) == 4
    """
    if not dataframes:
        raise ValueError("DataFrame list cannot be empty")

    try:
        result = pd.concat(dataframes, ignore_index=True)
        if len(result) == 0:
            raise ValueError("Concatenated DataFrame is empty")
        return result
    except Exception as e:
        raise ValueError(f"Failed to concatenate DataFrames: {e}") from e


def download_s3_parquet_with_optional_date_suffix(bucket_name: str, base_path: str,
                                                 date_suffix: Optional[str]) -> pd.DataFrame:
    """Download parquet file from S3 with optional date suffix.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket
    base_path : str
        Base S3 key path without date suffix
    date_suffix : Optional[str]
        Optional date suffix to append to filename (format: YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the parquet file

    Raises
    ------
    ValueError
        If S3 download or parquet parsing fails

    Examples
    --------
    >>> df = download_s3_parquet_with_optional_date_suffix(
    ...     'my-bucket', 'data/sales', '2023-01-01'
    ... )
    >>> assert isinstance(df, pd.DataFrame)
    """
    try:
        s3 = boto3.client('s3')

        if date_suffix:
            s3_key = f"{base_path}_{date_suffix}.parquet"
        else:
            s3_key = f"{base_path}.parquet"

        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        parquet_data = response['Body'].read()

        buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(buffer)

        return df
    except Exception as e:
        raise ValueError(f"Failed to download S3 parquet from s3://{bucket_name}/{s3_key}: {e}")


def create_s3_file_path(base_path: str, filename: str, current_date: str) -> str:
    """Create standardized S3 file path.

    Parameters
    ----------
    base_path : str
        Base S3 path for file storage
    filename : str
        Base filename without extension
    current_date : str
        Date string to append (format: YYYY-MM-DD)

    Returns
    -------
    str
        Complete S3 key path with .parquet extension

    Raises
    ------
    ValueError
        If base_path or filename is empty

    Examples
    --------
    >>> path = create_s3_file_path('data/sales', 'flexguard', '2023-01-01')
    >>> assert path == 'data/sales/flexguard_2023-01-01.parquet'
    """
    if not all([base_path, filename]):
        raise ValueError("base_path and filename cannot be empty")

    if current_date:
        return f"{base_path}/{filename}_{current_date}.parquet"
    else:
        return f"{base_path}/{filename}.parquet"


def convert_dataframe_to_parquet_buffer(df: pd.DataFrame) -> io.BytesIO:
    """Convert DataFrame to parquet format in memory buffer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert to parquet format

    Returns
    -------
    io.BytesIO
        In-memory buffer containing parquet data, positioned at start

    Raises
    ------
    ValueError
        If DataFrame is empty or parquet conversion fails

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]})
    >>> buffer = convert_dataframe_to_parquet_buffer(df)
    >>> assert buffer.tell() == 0  # Buffer positioned at start
    """
    if len(df) == 0:
        raise ValueError("Cannot convert empty DataFrame to parquet")

    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine="pyarrow")
        buffer.seek(0)
        return buffer
    except Exception as e:
        raise ValueError(f"Failed to convert DataFrame to parquet: {e}") from e


def upload_parquet_buffer_to_s3(client: boto3.client, buffer: io.BytesIO, bucket_name: str, key: str) -> None:
    """Upload parquet buffer to S3.

    Parameters
    ----------
    client : boto3.client
        S3 client for upload operations
    buffer : io.BytesIO
        In-memory buffer containing parquet data
    bucket_name : str
        Name of the destination S3 bucket
    key : str
        S3 object key for the uploaded file

    Raises
    ------
    ValueError
        If bucket_name or key is empty, buffer is empty, or upload fails

    Examples
    --------
    >>> s3_client = boto3.client('s3')
    >>> buffer = io.BytesIO(b'parquet_data')
    >>> upload_parquet_buffer_to_s3(s3_client, buffer, 'my-bucket', 'data/file.parquet')
    """
    if not bucket_name or not key:
        raise ValueError("bucket_name and key cannot be empty")

    buffer.seek(0, 2)  # Seek to end
    buffer_size = buffer.tell()
    if buffer_size == 0:
        raise ValueError("Buffer appears to be empty")

    try:
        buffer.seek(0)  # Reset to start
        client.upload_fileobj(buffer, bucket_name, key)
    except Exception as e:
        raise ValueError(f"Failed to upload to s3://{bucket_name}/{key}: {e}") from e


def upload_parquet_to_s3_with_optional_date_suffix(s3_client: boto3.client, parquet_buffer: io.BytesIO,
                                                   bucket_name: str, base_path: str,
                                                   date_suffix: Optional[str]) -> None:
    """Upload parquet buffer to S3 with optional date suffix.

    Parameters
    ----------
    s3_client : boto3.client
        S3 client for upload operations
    parquet_buffer : io.BytesIO
        In-memory buffer containing parquet data
    bucket_name : str
        Name of the destination S3 bucket
    base_path : str
        Base S3 key path without date suffix
    date_suffix : Optional[str]
        Optional date suffix to append to filename

    Raises
    ------
    ValueError
        If upload to S3 fails

    Examples
    --------
    >>> s3_client = boto3.client('s3')
    >>> buffer = io.BytesIO()
    >>> upload_parquet_to_s3_with_optional_date_suffix(
    ...     s3_client, buffer, 'bucket', 'data/sales', '2023-01-01'
    ... )
    """
    try:
        if date_suffix:
            s3_key = f"{base_path}_{date_suffix}.parquet"
        else:
            s3_key = f"{base_path}.parquet"

        parquet_buffer.seek(0)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=parquet_buffer.getvalue(),
            ContentType='application/octet-stream'
        )
    except Exception as e:
        raise ValueError(f"Failed to upload parquet to s3://{bucket_name}/{s3_key}: {e}")


# =============================================================================
# ENHANCED ERROR HANDLING FUNCTIONS - GRANULAR OPERATIONS
# =============================================================================

# Error message constants preserving exact notebook messages
ERROR_MESSAGES = {
    'sts_client_failed': "CRITICAL: STS client creation failed. Business Impact: Cannot assume IAM role for S3 data access. Required Action: Verify AWS credentials and STS endpoint configuration.",
    'role_assumption_failed': "CRITICAL: IAM role assumption failed. Business Impact: Cannot access Prudential data lake for analysis. Required Action: Verify IAM role permissions and trust relationship.",
    's3_resource_failed': "CRITICAL: S3 resource creation failed. Business Impact: Cannot access data lake for pipeline execution. Required Action: Verify S3 permissions and bucket access rights.",
    'sales_discovery_failed': "CRITICAL: Sales data discovery failed in S3. Business Impact: Cannot access sales data for analysis pipeline. Required Action: Verify S3 bucket permissions and data availability.",
    'no_sales_files': "CRITICAL: No sales data parquet files found in S3. Business Impact: Cannot proceed with sales analysis without source data. Required Action: Verify S3 data pipeline has executed and files exist.",
    'sales_file_failed': "CRITICAL: Failed to load parquet file {key}. Business Impact: Incomplete sales dataset will affect analysis accuracy. Required Action: Verify file integrity and S3 access permissions.",
    'no_valid_sales': "CRITICAL: No valid sales data loaded from any parquet files. Business Impact: Cannot proceed with sales analysis. Required Action: Verify data quality and S3 file integrity.",
    'sales_concat_failed': "CRITICAL: Sales data loading and concatenation failed. Business Impact: Cannot construct complete sales dataset for analysis. Required Action: Review S3 data integrity and loading procedures.",
    'empty_sales_dataset': "CRITICAL: Combined sales dataset is empty. Business Impact: No sales data available for pipeline processing. Required Action: Verify source data exists and loading procedures.",
    'missing_sales_columns': "CRITICAL: Missing required columns in sales data: {columns}. Business Impact: Cannot perform product filtering and time series creation. Required Action: Verify sales data schema completeness.",
    'wink_discovery_failed': "CRITICAL: WINK data discovery failed in S3. Business Impact: Cannot access competitive rates for market analysis. Required Action: Verify S3 bucket permissions and WINK data availability.",
    'no_wink_files': "CRITICAL: No WINK competitive rate files found in S3. Business Impact: Cannot perform competitive analysis without rate data. Required Action: Verify WINK data pipeline has executed and files exist.",
    'wink_file_failed': "CRITICAL: Failed to load WINK parquet file {key}. Business Impact: Incomplete competitive rate dataset will affect market analysis. Required Action: Verify WINK file integrity and S3 access permissions.",
    'no_valid_wink': "CRITICAL: No valid WINK data loaded from any parquet files. Business Impact: Cannot proceed with competitive rate analysis. Required Action: Verify WINK data quality and S3 file integrity.",
    'wink_concat_failed': "CRITICAL: WINK data loading and concatenation failed. Business Impact: Cannot construct complete competitive rate dataset. Required Action: Review WINK S3 data integrity and loading procedures.",
    'empty_wink_dataset': "CRITICAL: Combined WINK dataset is empty. Business Impact: No competitive rate data available for market analysis. Required Action: Verify WINK source data exists and loading procedures.",
    'insufficient_rate_columns': "CRITICAL: Insufficient competitive rate columns ({count} found). Business Impact: Cannot perform comprehensive competitive analysis. Required Action: Verify WINK data includes major competitor rates.",
    'no_wink_date_column': "CRITICAL: No 'date' column found in WINK data. Business Impact: Cannot perform temporal competitive analysis. Required Action: Verify WINK data schema includes date information."
}

class AWSConnectionError(Exception):
    """AWS connection and authentication errors."""
    pass

class DataLoadingError(Exception):
    """Data loading and processing errors."""
    pass

class DataValidationError(Exception):
    """Data validation and schema errors."""
    pass

def setup_aws_sts_client_with_validation(aws_config: Dict[str, Any]) -> boto3.client:
    """Create AWS STS client with comprehensive error handling.

    Parameters
    ----------
    aws_config : Dict[str, Any]
        AWS configuration containing STS endpoint URL

    Returns
    -------
    boto3.client
        Configured STS client for role assumption operations

    Raises
    ------
    AWSConnectionError
        If STS client creation fails with detailed business context
    """
    try:
        return create_sts_client(aws_config)
    except Exception as e:
        raise AWSConnectionError(
            f"{ERROR_MESSAGES['sts_client_failed']} "
            f"AWS Config: {aws_config['sts_endpoint_url']}. "
            f"Original error: {e}"
        ) from e

def assume_iam_role_with_validation(sts_client: boto3.client, aws_config: Dict[str, Any]) -> Dict[str, Any]:
    """Assume IAM role with comprehensive error handling.

    Parameters
    ----------
    sts_client : boto3.client
        AWS STS client for role assumption operations
    aws_config : Dict[str, Any]
        AWS configuration containing role ARN

    Returns
    -------
    Dict[str, Any]
        AWS response containing temporary credentials and role information

    Raises
    ------
    AWSConnectionError
        If IAM role assumption fails with detailed business context
    """
    try:
        return assume_iam_role(sts_client, aws_config)
    except Exception as e:
        raise AWSConnectionError(
            f"{ERROR_MESSAGES['role_assumption_failed']} "
            f"Role ARN: {aws_config['role_arn']}. "
            f"Original error: {e}"
        ) from e

def setup_s3_resource_with_validation(credentials: Dict[str, Any], source_bucket_name: str) -> tuple:
    """Create S3 resource and bucket with comprehensive error handling.

    Parameters
    ----------
    credentials : Dict[str, Any]
        AWS temporary credentials from role assumption
    source_bucket_name : str
        Name of the source S3 bucket

    Returns
    -------
    tuple
        (s3_resource, bucket) for S3 operations

    Raises
    ------
    AWSConnectionError
        If S3 resource creation fails with detailed business context
    """
    try:
        s3_resource = create_s3_resource_with_credentials(credentials)
        bucket = s3_resource.Bucket(source_bucket_name)
        return s3_resource, bucket
    except Exception as e:
        raise AWSConnectionError(
            f"{ERROR_MESSAGES['s3_resource_failed']} "
            f"Source bucket: {source_bucket_name}. "
            f"Original error: {e}"
        ) from e

def _discover_sales_parquet_files(bucket: Any) -> List[str]:
    """Discover sales parquet files from S3 bucket.

    Parameters
    ----------
    bucket : S3.Bucket
        S3 bucket resource for file discovery

    Returns
    -------
    List[str]
        List of discovered parquet file keys

    Raises
    ------
    DataLoadingError
        If discovery fails or no files found
    """
    try:
        parquet_keys = list_parquet_objects_with_prefix(bucket, "access/ierpt/tde_sales_by_product_by_fund/")
        if not parquet_keys:
            raise DataLoadingError(
                f"{ERROR_MESSAGES['no_sales_files']} "
                f"S3 Path: access/ierpt/tde_sales_by_product_by_fund/"
            )
        return parquet_keys
    except DataLoadingError:
        raise
    except Exception as e:
        raise DataLoadingError(
            f"{ERROR_MESSAGES['sales_discovery_failed']} "
            f"Expected path: access/ierpt/tde_sales_by_product_by_fund/. "
            f"Original error: {e}"
        ) from e


def _load_sales_dataframes_from_keys(s3_resource: Any, source_bucket_name: str, parquet_keys: List[str]) -> pd.DataFrame:
    """Load and concatenate sales DataFrames from S3 keys.

    Parameters
    ----------
    s3_resource : boto3.resource
        S3 resource for data operations
    source_bucket_name : str
        Name of the source S3 bucket
    parquet_keys : List[str]
        List of S3 keys to load

    Returns
    -------
    pd.DataFrame
        Combined DataFrame from all loaded files

    Raises
    ------
    DataLoadingError
        If loading or concatenation fails
    """
    try:
        dataframes = []
        for i, key in enumerate(parquet_keys):
            try:
                df_temp = download_parquet_from_s3_object(s3_resource, source_bucket_name, key)
                if not df_temp.empty:
                    dataframes.append(df_temp)
            except Exception as e:
                raise DataLoadingError(
                    ERROR_MESSAGES['sales_file_failed'].format(key=key) + f" "
                    f"File path: {key}. Original error: {e}"
                ) from e

        if not dataframes:
            raise DataLoadingError(ERROR_MESSAGES['no_valid_sales'])

        return concatenate_dataframe_list(dataframes)

    except DataLoadingError:
        raise
    except Exception as e:
        raise DataLoadingError(
            f"{ERROR_MESSAGES['sales_concat_failed']} "
            f"Files processed: {len(parquet_keys)}. "
            f"Original error: {e}"
        ) from e


def _validate_sales_dataset_structure(df_combined: pd.DataFrame) -> None:
    """Validate sales dataset structure and required columns.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Combined sales DataFrame to validate

    Raises
    ------
    DataValidationError
        If validation fails (empty dataset or missing columns)
    """
    if df_combined.empty:
        raise DataValidationError(ERROR_MESSAGES['empty_sales_dataset'])

    required_columns = ['application_signed_date', 'contract_issue_date', 'product_name']
    missing_columns = [col for col in required_columns if col not in df_combined.columns]
    if missing_columns:
        raise DataValidationError(
            ERROR_MESSAGES['missing_sales_columns'].format(columns=missing_columns) + f" "
            f"Available columns: {len(df_combined.columns)} total"
        )


def discover_and_load_sales_data(bucket: Any, s3_resource: Any, source_bucket_name: str) -> pd.DataFrame:
    """Discover and load sales data from S3 with comprehensive validation.

    Orchestrates the complete sales data loading pipeline:
    1. Discovery: Find all parquet files in S3
    2. Loading: Download and concatenate DataFrames
    3. Validation: Verify structure and required columns
    4. Production Validation: Quality gates with fail-fast error handling

    Parameters
    ----------
    bucket : S3.Bucket
        S3 bucket resource for operations
    s3_resource : boto3.resource
        S3 resource for data operations
    source_bucket_name : str
        Name of the source S3 bucket

    Returns
    -------
    pd.DataFrame
        Combined sales DataFrame with validation

    Raises
    ------
    DataLoadingError
        If sales data discovery or loading fails
    DataValidationError
        If validation fails
    ValueError
        If production validation fails (data quality gates)
    """
    from src.validation.pipeline_validation_helpers import validate_extraction_output

    # Step 1: Discover sales data files
    parquet_keys = _discover_sales_parquet_files(bucket)

    # Step 2: Load and concatenate sales data
    df_combined = _load_sales_dataframes_from_keys(s3_resource, source_bucket_name, parquet_keys)

    # Step 3: Validate combined dataset (existing validation)
    _validate_sales_dataset_structure(df_combined)

    # Step 4: Production validation with corrected column names
    df_validated = validate_extraction_output(
        df=df_combined,
        stage_name="sales_data_extraction",
        date_column="application_signed_date",
        critical_columns=['application_signed_date', 'contract_issue_date', 'product_name'],
        allow_shrinkage=False  # Sales data should only grow
    )

    return df_validated

def _discover_wink_parquet_files(bucket: Any) -> List[str]:
    """Discover WINK parquet files from S3 bucket.

    Parameters
    ----------
    bucket : S3.Bucket
        S3 bucket resource for file discovery

    Returns
    -------
    List[str]
        List of discovered parquet file keys

    Raises
    ------
    DataLoadingError
        If discovery fails or no files found
    """
    try:
        parquet_keys = list_parquet_objects_with_prefix(bucket, "access/ierpt/wink_ann_product_rates/")
        if not parquet_keys:
            raise DataLoadingError(
                f"{ERROR_MESSAGES['no_wink_files']} "
                f"S3 Path: access/ierpt/wink_ann_product_rates/"
            )
        return parquet_keys
    except DataLoadingError:
        raise
    except Exception as e:
        raise DataLoadingError(
            f"{ERROR_MESSAGES['wink_discovery_failed']} "
            f"Expected path: access/ierpt/wink_ann_product_rates/. "
            f"Original error: {e}"
        ) from e


def _load_wink_dataframes_from_keys(s3_resource: Any, source_bucket_name: str, parquet_keys: List[str]) -> pd.DataFrame:
    """Load and concatenate WINK DataFrames from S3 keys.

    Parameters
    ----------
    s3_resource : boto3.resource
        S3 resource for data operations
    source_bucket_name : str
        Name of the source S3 bucket
    parquet_keys : List[str]
        List of S3 keys to load

    Returns
    -------
    pd.DataFrame
        Combined DataFrame from all loaded files

    Raises
    ------
    DataLoadingError
        If loading or concatenation fails
    """
    try:
        dataframes = []
        for key in parquet_keys:
            try:
                df_temp = download_parquet_from_s3_object(s3_resource, source_bucket_name, key)
                if not df_temp.empty:
                    dataframes.append(df_temp)
            except Exception as e:
                raise DataLoadingError(
                    ERROR_MESSAGES['wink_file_failed'].format(key=key) + f" "
                    f"File path: {key}. Original error: {e}"
                ) from e

        if not dataframes:
            raise DataLoadingError(ERROR_MESSAGES['no_valid_wink'])

        return concatenate_dataframe_list(dataframes)

    except DataLoadingError:
        raise
    except Exception as e:
        raise DataLoadingError(
            f"{ERROR_MESSAGES['wink_concat_failed']} "
            f"Files processed: {len(parquet_keys)}. "
            f"Original error: {e}"
        ) from e


def _validate_wink_dataset_structure(df_wink: pd.DataFrame) -> None:
    """Validate WINK dataset structure and required columns.

    Parameters
    ----------
    df_wink : pd.DataFrame
        Combined WINK DataFrame to validate

    Raises
    ------
    DataValidationError
        If validation fails (empty dataset, insufficient rate columns, missing date)
    """
    if df_wink.empty:
        raise DataValidationError(ERROR_MESSAGES['empty_wink_dataset'])

    # Validate rate columns - business minimum for competitive analysis
    rate_columns = [col for col in df_wink.columns if 'rate' in col.lower()]
    if len(rate_columns) < 5:
        raise DataValidationError(
            ERROR_MESSAGES['insufficient_rate_columns'].format(count=len(rate_columns)) + f" "
            f"Rate columns found: {rate_columns}"
        )

    # Validate date column presence
    if 'date' not in df_wink.columns:
        date_like_columns = [col for col in df_wink.columns if 'date' in col.lower()]
        raise DataValidationError(
            f"{ERROR_MESSAGES['no_wink_date_column']} "
            f"Date-like columns found: {date_like_columns}"
        )


def discover_and_load_wink_data(bucket: Any, s3_resource: Any, source_bucket_name: str) -> pd.DataFrame:
    """Discover and load WINK competitive data from S3 with comprehensive validation.

    Orchestrates the complete WINK data loading pipeline:
    1. Discovery: Find all parquet files in S3
    2. Loading: Download and concatenate DataFrames
    3. Validation: Verify structure and required columns
    4. Production Validation: Quality gates with fail-fast error handling

    Parameters
    ----------
    bucket : S3.Bucket
        S3 bucket resource for operations
    s3_resource : boto3.resource
        S3 resource for data operations
    source_bucket_name : str
        Name of the source S3 bucket

    Returns
    -------
    pd.DataFrame
        Combined WINK DataFrame with validation

    Raises
    ------
    DataLoadingError
        If WINK data discovery or loading fails
    DataValidationError
        If validation fails
    ValueError
        If production validation fails (data quality gates)
    """
    from src.validation.pipeline_validation_helpers import validate_extraction_output

    # Step 1: Discover WINK data files
    parquet_keys = _discover_wink_parquet_files(bucket)

    # Step 2: Load and concatenate WINK data
    df_wink = _load_wink_dataframes_from_keys(s3_resource, source_bucket_name, parquet_keys)

    # Step 3: Validate combined dataset (existing validation)
    _validate_wink_dataset_structure(df_wink)

    # Step 4: Production validation (correct for WINK data which has 'date' column)
    df_validated = validate_extraction_output(
        df=df_wink,
        stage_name="wink_data_extraction",
        date_column="date",
        critical_columns=['date', 'product_name'],
        allow_shrinkage=False  # WINK data should only grow
    )

    return df_validated


def load_market_share_weights_from_s3(s3_path: str) -> pd.DataFrame:
    """Load market share weights from S3.

    Parameters
    ----------
    s3_path : str
        Complete S3 path to the market share weights parquet file

    Returns
    -------
    pd.DataFrame
        DataFrame containing market share weights by company and quarter

    Raises
    ------
    ValueError
        If S3 path is invalid or file loading fails

    Examples
    --------
    >>> df = load_market_share_weights_from_s3('s3://bucket/weights.parquet')
    >>> assert 'current_quarter' in df.columns
    """
    try:
        df = pd.read_parquet(s3_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load market share weights from {s3_path}: {e}")


# ===================================================================
# AWS UTILITY FUNCTIONS (Migrated from helpers/AWS_*_tools.py)
# ===================================================================











