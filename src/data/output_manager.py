"""
Output Manager - Unified Output Destination Control

This module provides a unified interface for saving datasets to either local
filesystem or S3, controlled by a simple toggle. This allows the same notebook
code to write outputs to different destinations without code changes.

CANONICAL FUNCTIONS:
- save_output(): Save DataFrame to local or S3 based on configuration
- load_output(): Load DataFrame from local or S3
- configure_output_mode(): Set output destination preferences

CONFIGURATION:
- Local mode: Writes to outputs/datasets/ directory (default)
- S3 mode: Writes to configured S3 bucket and path

Usage Pattern (from notebooks):
    from src.data.output_manager import save_output, configure_output_mode

    # Configure once at start of notebook
    configure_output_mode(write_to_s3=True, s3_config=aws_config)

    # Use same save command throughout
    save_output(df, "FlexGuard_Sales")
"""

import os
import io
import pandas as pd
import boto3
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output destinations for datasets (local vs S3)."""

    def __init__(self, base_dir: str = None):
        """
        Initialize output manager.

        Args:
            base_dir: Project root directory (auto-detected if not provided)
        """
        # Auto-detect project root if not specified
        if base_dir is None:
            cwd = Path.cwd()
            if cwd.name == 'notebooks':
                base_dir = str(cwd.parent)
            elif (cwd / 'src').exists():
                base_dir = str(cwd)
            else:
                base_dir = str(Path(__file__).parent.parent.parent)

        self.base_dir = Path(base_dir)
        self.local_output_dir = self.base_dir / "outputs" / "datasets"
        self.local_output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.write_to_s3 = False
        self.s3_config = None
        self.s3_client = None

    def configure(
        self,
        write_to_s3: bool = False,
        s3_config: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Configure output destination.

        Args:
            write_to_s3: If True, write to S3; if False, write to local
            s3_config: Configuration dictionary with S3 bucket and path info
                Required keys if write_to_s3=True:
                - output_bucket_name: S3 bucket name
                - output_base_path: Base path within bucket
        """
        self.write_to_s3 = write_to_s3

        if write_to_s3:
            if not s3_config:
                raise ValueError("s3_config required when write_to_s3=True")

            required_keys = ['output_bucket_name', 'output_base_path']
            missing_keys = [k for k in required_keys if k not in s3_config]
            if missing_keys:
                raise ValueError(f"s3_config missing required keys: {missing_keys}")

            self.s3_config = s3_config

            # Initialize S3 client if not already done
            if self.s3_client is None:
                self.s3_client = boto3.client('s3')

            logger.info(f"Output mode: S3 (bucket={s3_config['output_bucket_name']})")
        else:
            logger.info(f"Output mode: Local (dir={self.local_output_dir})")

    def save_output(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        subdirectory: str = ""
    ) -> str:
        """
        Save DataFrame to configured destination (local or S3).

        Args:
            df: DataFrame to save
            dataset_name: Name for the dataset file (without extension)
            subdirectory: Optional subdirectory within output location

        Returns:
            Path or S3 URI where data was saved

        Raises:
            ValueError: If dataset_name is invalid or DataFrame is empty
            RuntimeError: If save operation fails
        """
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame for dataset: {dataset_name}")

        if not dataset_name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(
                f"Invalid dataset name: {dataset_name}. "
                "Use alphanumeric, underscore, and hyphen only."
            )

        filename = f"{dataset_name}.parquet"

        try:
            if self.write_to_s3:
                return self._save_to_s3(df, filename, subdirectory)
            else:
                return self._save_to_local(df, filename, subdirectory)

        except Exception as e:
            logger.error(f"Failed to save dataset {dataset_name}: {str(e)}")
            raise RuntimeError(f"Dataset save operation failed: {str(e)}")

    def load_output(
        self,
        dataset_name: str,
        subdirectory: str = "",
        try_s3_if_local_missing: bool = True
    ) -> pd.DataFrame:
        """
        Load DataFrame from configured location.

        Args:
            dataset_name: Name of the dataset (without extension)
            subdirectory: Optional subdirectory within output location
            try_s3_if_local_missing: If True and local file missing, try S3

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If dataset not found
            RuntimeError: If load operation fails
        """
        filename = f"{dataset_name}.parquet"

        # Try local first
        try:
            return self._load_from_local(filename, subdirectory)
        except FileNotFoundError:
            if try_s3_if_local_missing and self.s3_config:
                logger.info(f"Local file not found, attempting S3 load for: {dataset_name}")
                return self._load_from_s3(filename, subdirectory)
            raise

    def _save_to_local(
        self,
        df: pd.DataFrame,
        filename: str,
        subdirectory: str
    ) -> str:
        """Save DataFrame to local filesystem."""
        if subdirectory:
            output_dir = self.local_output_dir / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.local_output_dir

        file_path = output_dir / filename

        logger.info(f"Saving to local: {file_path} ({len(df)} rows)")
        df.to_parquet(file_path)

        logger.info(f"Successfully saved: {file_path}")
        return str(file_path)

    def _save_to_s3(
        self,
        df: pd.DataFrame,
        filename: str,
        subdirectory: str
    ) -> str:
        """Save DataFrame to S3."""
        if not self.s3_client or not self.s3_config:
            raise RuntimeError("S3 not configured. Call configure() first.")

        # Construct S3 key
        base_path = self.s3_config['output_base_path']
        if subdirectory:
            s3_key = f"{base_path}/{subdirectory}/{filename}"
        else:
            s3_key = f"{base_path}/{filename}"

        bucket_name = self.s3_config['output_bucket_name']
        s3_uri = f"s3://{bucket_name}/{s3_key}"

        logger.info(f"Saving to S3: {s3_uri} ({len(df)} rows)")

        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        buffer.seek(0)

        # Upload to S3
        self.s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType='application/octet-stream'
        )

        logger.info(f"Successfully saved: {s3_uri}")
        return s3_uri

    def _load_from_local(
        self,
        filename: str,
        subdirectory: str
    ) -> pd.DataFrame:
        """Load DataFrame from local filesystem."""
        if subdirectory:
            output_dir = self.local_output_dir / subdirectory
        else:
            output_dir = self.local_output_dir

        file_path = output_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        logger.info(f"Loading from local: {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded dataset: {filename} ({len(df)} rows)")
        return df

    def _load_from_s3(
        self,
        filename: str,
        subdirectory: str
    ) -> pd.DataFrame:
        """Load DataFrame from S3."""
        if not self.s3_client or not self.s3_config:
            raise RuntimeError("S3 not configured. Call configure() first.")

        # Construct S3 key
        base_path = self.s3_config['output_base_path']
        if subdirectory:
            s3_key = f"{base_path}/{subdirectory}/{filename}"
        else:
            s3_key = f"{base_path}/{filename}"

        bucket_name = self.s3_config['output_bucket_name']
        s3_uri = f"s3://{bucket_name}/{s3_key}"

        logger.info(f"Loading from S3: {s3_uri}")

        # Download from S3
        buffer = io.BytesIO()
        self.s3_client.download_fileobj(bucket_name, s3_key, buffer)
        buffer.seek(0)

        # Read parquet
        df = pd.read_parquet(buffer)
        logger.info(f"Loaded dataset: {filename} ({len(df)} rows)")
        return df

    def get_output_location_info(self) -> Dict[str, Any]:
        """
        Get information about current output configuration.

        Returns:
            Dictionary with configuration details
        """
        if self.write_to_s3 and self.s3_config:
            return {
                'mode': 'S3',
                'bucket': self.s3_config['output_bucket_name'],
                'base_path': self.s3_config['output_base_path'],
                'full_path': f"s3://{self.s3_config['output_bucket_name']}/{self.s3_config['output_base_path']}"
            }
        else:
            return {
                'mode': 'Local',
                'directory': str(self.local_output_dir),
                'full_path': str(self.local_output_dir)
            }


# =============================================================================
# DEPENDENCY INJECTION PATTERN - ARCHITECTURAL DECISION
# =============================================================================
# PATTERN: Dual-mode singleton for notebook convenience + DI for testability
# NOTEBOOK USAGE: Convenience functions use singleton (simple imports)
# TEST USAGE: Factory functions for isolated testing (no shared state)
# INTERNAL MODULES: Should avoid singletons, use explicit parameters
#
# RATIONALE: Singletons provide ergonomic API for notebooks while DI enables
# comprehensive testing. This is an intentional architectural choice, not
# technical debt. Notebooks are a presentation layer where convenience matters.
#
# USAGE GUIDANCE:
# - Notebooks: Use convenience functions (save_output, configure_output_mode, etc.)
# - Tests: Use create_output_manager() for test isolation
# - Internal src/: Pass managers explicitly when needed

# Global instance for notebook convenience
_output_manager = None


def get_output_manager() -> OutputManager:
    """Get global OutputManager instance.

    MIGRATION NOTE: This function now returns the singleton for backward compatibility.
    New code should create managers explicitly:
        manager = OutputManager()

    Returns:
        OutputManager instance (singleton for backward compatibility)
    """
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager


def create_output_manager(base_dir: str = None) -> OutputManager:
    """Create a new output manager instance (DI pattern).

    This is the preferred pattern for dependency injection.
    Use this when you need an isolated manager instance (e.g., in tests).

    Args:
        base_dir: Project root directory (auto-detected if not provided)

    Returns:
        New OutputManager instance

    Example:
        manager = create_output_manager()
        manager.configure(write_to_s3=True, s3_config=config)
        manager.save_output(df, "results")
    """
    return OutputManager(base_dir=base_dir)


# Convenience functions for notebook use (backward-compatible with DI support)
def configure_output_mode(
    write_to_s3: bool = False,
    s3_config: Optional[Dict[str, str]] = None,
    manager: Optional[OutputManager] = None
) -> None:
    """
    Configure output destination (local or S3).

    Args:
        write_to_s3: If True, write to S3; if False, write to local (default)
        s3_config: S3 configuration dict (required if write_to_s3=True)
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Example:
        # Local mode (default)
        configure_output_mode(write_to_s3=False)

        # S3 mode
        configure_output_mode(
            write_to_s3=True,
            s3_config={
                'output_bucket_name': 'my-bucket',
                'output_base_path': 'my/path'
            }
        )
    """
    if manager is None:
        manager = get_output_manager()  # Fallback to singleton
    manager.configure(write_to_s3=write_to_s3, s3_config=s3_config)


def save_output(
    df: pd.DataFrame,
    dataset_name: str,
    subdirectory: str = "",
    manager: Optional[OutputManager] = None
) -> str:
    """
    Save DataFrame to configured destination.

    Args:
        df: DataFrame to save
        dataset_name: Dataset name (without .parquet extension)
        subdirectory: Optional subdirectory
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        Path or S3 URI where data was saved
    """
    if manager is None:
        manager = get_output_manager()  # Fallback to singleton
    return manager.save_output(df, dataset_name, subdirectory)


def load_output(
    dataset_name: str,
    subdirectory: str = "",
    manager: Optional[OutputManager] = None
) -> pd.DataFrame:
    """
    Load DataFrame from configured location.

    Args:
        dataset_name: Dataset name (without .parquet extension)
        subdirectory: Optional subdirectory
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        Loaded DataFrame
    """
    if manager is None:
        manager = get_output_manager()  # Fallback to singleton
    return manager.load_output(dataset_name, subdirectory)


def get_output_info(manager: Optional[OutputManager] = None) -> Dict[str, Any]:
    """
    Get information about current output configuration.

    Args:
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        Dictionary with mode and location details
    """
    if manager is None:
        manager = get_output_manager()  # Fallback to singleton
    return manager.get_output_location_info()
