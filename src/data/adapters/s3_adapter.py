"""
AWS S3 Data Source Adapter.

Production adapter for loading data from AWS S3 using assumed role credentials.

Usage:
    from src.data.adapters.s3_adapter import S3Adapter
    from src.core.types import AWSConfig

    config: AWSConfig = {
        "sts_endpoint_url": "https://sts.us-east-1.amazonaws.com",
        "role_arn": "arn:aws:iam::...",
        "xid": "user123",
        "bucket_name": "my-bucket"
    }
    adapter = S3Adapter(config)
    df = adapter.load_sales_data()
"""

from typing import Optional, Dict, Any
from pathlib import Path
import io
import pandas as pd

from src.data.adapters.base import DataAdapterBase
from src.core.types import AWSConfig


class S3Adapter(DataAdapterBase):
    """AWS S3 data source adapter.

    Loads data from S3 using STS assumed role credentials.
    Production-ready with comprehensive error handling.

    Parameters
    ----------
    config : AWSConfig
        AWS configuration with STS endpoint, role ARN, and bucket info
    paths : Optional[Dict[str, str]]
        S3 key prefixes for different data types

    Examples
    --------
    >>> adapter = S3Adapter(config)
    >>> sales_df = adapter.load_sales_data()
    >>> rates_df = adapter.load_competitive_rates("2022-01-01")
    """

    # Default S3 key prefixes
    DEFAULT_PATHS = {
        "sales": "data/tde_sales/",
        "rates": "data/wink_rates/",
        "weights": "data/market_weights/",
        "macro": "data/macro/",
        "output": "outputs/",
    }

    def __init__(
        self,
        config: AWSConfig,
        paths: Optional[Dict[str, str]] = None,
    ):
        self._config = config
        self._paths = paths or self.DEFAULT_PATHS
        self._s3_resource = None  # Lazy initialization
        self._bucket = None

    @property
    def source_type(self) -> str:
        """Return 'aws' as the data source identifier."""
        return "aws"

    def _ensure_connection(self) -> None:
        """Establish S3 connection if not already connected."""
        if self._s3_resource is None:
            self._s3_resource = self._create_s3_resource()
            self._bucket = self._s3_resource.Bucket(self._config["bucket_name"])

    def _create_s3_resource(self) -> Any:
        """Create S3 resource using assumed role credentials.

        Returns
        -------
        boto3.resource
            Configured S3 resource

        Raises
        ------
        ImportError
            If boto3 is not installed
        ValueError
            If role assumption fails
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3Adapter. "
                "Install with: pip install boto3"
            )

        # Create STS client
        sts_client = boto3.client(
            "sts", endpoint_url=self._config["sts_endpoint_url"]
        )

        # Assume role
        try:
            assumed_role = sts_client.assume_role(
                RoleArn=self._config["role_arn"],
                RoleSessionName=self._config["xid"],
            )
        except Exception as e:
            raise ValueError(
                f"CRITICAL: Failed to assume IAM role. "
                f"Role ARN: {self._config['role_arn']}. "
                f"Error: {e}. "
                f"Required action: Verify IAM role permissions and trust policy."
            ) from e

        credentials = assumed_role["Credentials"]

        # Create S3 resource with temporary credentials
        return boto3.resource(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    def _load_parquet_from_s3(
        self,
        prefix: str,
        product_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and concatenate parquet files from S3 prefix.

        Parameters
        ----------
        prefix : str
            S3 key prefix to search
        product_filter : Optional[str]
            Filter by product_name column if present

        Returns
        -------
        pd.DataFrame
            Concatenated data from all parquet files
        """
        self._ensure_connection()

        dfs = []
        for obj in self._bucket.objects.filter(Prefix=prefix):
            if not obj.key.endswith(".parquet"):
                continue

            # Download and read parquet
            buffer = io.BytesIO()
            self._s3_resource.Object(
                self._config["bucket_name"], obj.key
            ).download_fileobj(buffer)

            df_temp = pd.read_parquet(buffer, engine="pyarrow")

            # Apply product filter if specified
            if product_filter and "product_name" in df_temp.columns:
                df_temp = df_temp[
                    df_temp["product_name"] == product_filter
                ].reset_index(drop=True)

            dfs.append(df_temp)

        if not dfs:
            raise ValueError(
                f"CRITICAL: No parquet files found at prefix '{prefix}'. "
                f"Business impact: Cannot proceed without data. "
                f"Required action: Verify S3 path and bucket permissions."
            )

        return pd.concat(dfs, ignore_index=True)

    def load_sales_data(
        self, product_filter: Optional[str] = None
    ) -> pd.DataFrame:
        """Load TDE sales data from S3.

        Parameters
        ----------
        product_filter : Optional[str]
            Filter by product name

        Returns
        -------
        pd.DataFrame
            Sales data with required columns
        """
        df = self._load_parquet_from_s3(
            self._paths["sales"], product_filter=product_filter
        )

        # Validate required columns
        required = ["application_signed_date", "premium_amount"]
        self._validate_dataframe(df, required, "sales_data")

        return df

    def load_competitive_rates(self, start_date: str) -> pd.DataFrame:
        """Load WINK competitive rate data from S3.

        Parameters
        ----------
        start_date : str
            Start date for rate data (YYYY-MM-DD)

        Returns
        -------
        pd.DataFrame
            Competitive rates by company and date
        """
        df = self._load_parquet_from_s3(self._paths["rates"])

        # Filter by start date
        if "effective_date" in df.columns:
            df = df[df["effective_date"] >= start_date].reset_index(drop=True)

        return df

    def load_market_weights(self) -> pd.DataFrame:
        """Load market share weights from S3.

        Returns
        -------
        pd.DataFrame
            Market weights by company
        """
        return self._load_parquet_from_s3(self._paths["weights"])

    def load_macro_data(self) -> pd.DataFrame:
        """Load macroeconomic indicator data from S3.

        Returns
        -------
        pd.DataFrame
            Macro indicators (interest rates, indices)
        """
        return self._load_parquet_from_s3(self._paths["macro"])

    def save_output(
        self, df: pd.DataFrame, name: str, format: str = "parquet"
    ) -> str:
        """Save output to S3.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        name : str
            Output name (becomes part of S3 key)
        format : str
            Output format: "parquet", "csv"

        Returns
        -------
        str
            S3 URI where data was saved
        """
        self._ensure_connection()

        # Build S3 key
        key = f"{self._paths['output']}{name}.{format}"

        # Serialize to buffer
        buffer = io.BytesIO()
        if format == "parquet":
            df.to_parquet(buffer, engine="pyarrow", index=False)
        elif format == "csv":
            df.to_csv(buffer, index=False)
        else:
            raise ValueError(f"Unsupported format for S3: {format}")

        buffer.seek(0)

        # Upload to S3
        self._bucket.put_object(Key=key, Body=buffer)

        return f"s3://{self._config['bucket_name']}/{key}"


__all__ = ["S3Adapter"]
