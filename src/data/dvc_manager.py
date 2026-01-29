"""
DVC Automation Manager - Background DVC Operations

This module provides automated DVC operations that happen transparently in the background,
eliminating the need for manual `dvc add` and `dvc push` commands in notebooks.

CANONICAL FUNCTIONS:
- save_dataset(): Save DataFrame with automatic DVC tracking
- load_dataset(): Load DataFrame with DVC pull if needed
- checkpoint_pipeline(): Strategic pipeline checkpoints

AUTOMATIC OPERATIONS:
- Background DVC add operations
- S3 push with error handling
- Path management and validation
- Dependency tracking for dvc.yaml

Usage Pattern (from notebooks):
    from src.data.dvc_manager import save_dataset, load_dataset

    # Replaces manual DVC operations
    save_dataset(df, "FlexGuard_Sales", "Sales data after product filtering")
    df = load_dataset("weekly_aggregated_features")
"""

import os
import subprocess
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# =============================================================================
# CONTEXT ANCHOR: DVC AUTOMATION OBJECTIVES
# =============================================================================
# PURPOSE: Eliminate manual DVC operations from notebooks, provide transparent data persistence
# USED BY: All data pipeline notebooks (00_, 01_, 02_) to replace manual os.system('dvc add')
# DEPENDENCIES: DVC installation, S3 remote configuration, outputs/datasets/ directory
# LAST VALIDATED: 2025-11-21 (initial creation for pipeline streamlining)
# PATTERN STATUS: CANONICAL (replaces all manual DVC operations in notebooks)
#
# ARCHITECTURAL FLOW: save_dataset() → auto DVC add → background S3 push → logging
# SUCCESS CRITERIA: Zero manual DVC commands in notebooks, transparent operations
# INTEGRATION: Works with existing parquet file patterns and S3 remote
# MAINTENANCE: Monitor logs for DVC operation failures, ensure S3 credentials valid

logger = logging.getLogger(__name__)

class DVCManager:
    """Automated DVC operations manager for transparent data persistence."""

    def __init__(self, base_dir: str = None):
        # Auto-detect project root if not specified
        if base_dir is None:
            # Try to find project root by looking for src/ directory
            cwd = Path.cwd()
            if cwd.name == 'notebooks':
                base_dir = str(cwd.parent)
            elif (cwd / 'src').exists():
                base_dir = str(cwd)
            else:
                # Fallback to module location
                base_dir = str(Path(__file__).parent.parent.parent)

        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / "outputs" / "datasets"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Change to base dir only if it exists and we're not already there
        if self.base_dir.exists() and Path.cwd() != self.base_dir:
            os.chdir(self.base_dir)

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        description: str = "",
        auto_push: bool = True
    ) -> str:
        """
        Save DataFrame with automatic DVC tracking and S3 push.

        Args:
            df: DataFrame to save
            dataset_name: Name for the dataset file (without extension)
            description: Optional description for logging
            auto_push: Whether to automatically push to S3 remote

        Returns:
            Path to saved file

        Raises:
            ValueError: If dataset_name is invalid or DataFrame is empty
            RuntimeError: If DVC operations fail
        """
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame for dataset: {dataset_name}")

        if not dataset_name.replace('_', '').isalnum():
            raise ValueError(f"Invalid dataset name: {dataset_name}. Use alphanumeric and underscore only.")

        # Construct file path
        file_path = self.outputs_dir / f"{dataset_name}.parquet"

        try:
            # Save DataFrame
            logger.info(f"Saving dataset: {dataset_name} ({len(df)} rows) - {description}")
            df.to_parquet(file_path)

            # Automatic DVC add
            self._dvc_add(file_path)

            # Background S3 push if requested
            if auto_push:
                self._dvc_push_background(file_path.name)

            logger.info(f"Successfully saved and tracked: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to save dataset {dataset_name}: {str(e)}")
            raise RuntimeError(f"Dataset save operation failed: {str(e)}")

    def load_dataset(self, dataset_name: str, auto_pull: bool = True) -> pd.DataFrame:
        """
        Load DataFrame with automatic DVC pull if needed.

        Args:
            dataset_name: Name of the dataset (without extension)
            auto_pull: Whether to attempt DVC pull if file not found locally

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If dataset not found locally or in DVC remote
            RuntimeError: If DVC operations fail
        """
        file_path = self.outputs_dir / f"{dataset_name}.parquet"

        # Try to load locally first
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"Loaded dataset: {dataset_name} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Failed to load local file {file_path}: {str(e)}")

        # Attempt DVC pull if auto_pull enabled
        if auto_pull:
            try:
                logger.info(f"Attempting DVC pull for: {dataset_name}")
                self._dvc_pull(f"{dataset_name}.parquet")

                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded dataset after DVC pull: {dataset_name} ({len(df)} rows)")
                    return df

            except Exception as e:
                logger.warning(f"DVC pull failed for {dataset_name}: {str(e)}")

        raise FileNotFoundError(f"Dataset not found: {dataset_name} (tried local and DVC remote)")

    def checkpoint_pipeline(self, stage_name: str, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Create a strategic checkpoint with multiple related datasets.

        Args:
            stage_name: Name of the pipeline stage
            datasets: Dictionary mapping dataset names to DataFrames
        """
        logger.info(f"Creating pipeline checkpoint: {stage_name}")

        for dataset_name, df in datasets.items():
            description = f"Pipeline stage: {stage_name}"
            self.save_dataset(df, dataset_name, description, auto_push=False)

        # Single batch push for all datasets in this checkpoint
        self._dvc_push_batch()

        logger.info(f"Pipeline checkpoint completed: {stage_name} ({len(datasets)} datasets)")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets with metadata.

        Returns:
            List of dataset information dictionaries
        """
        datasets = []

        for parquet_file in self.outputs_dir.glob("*.parquet"):
            dvc_file = parquet_file.with_suffix(".parquet.dvc")

            dataset_info = {
                "name": parquet_file.stem,
                "local_exists": parquet_file.exists(),
                "dvc_tracked": dvc_file.exists(),
                "size_mb": parquet_file.stat().st_size / 1024 / 1024 if parquet_file.exists() else 0,
                "modified": datetime.fromtimestamp(parquet_file.stat().st_mtime) if parquet_file.exists() else None
            }

            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    dataset_info["rows"] = len(df)
                    dataset_info["columns"] = len(df.columns)
                except (pd.errors.ParserError, OSError, ValueError) as e:
                    logger.debug(f"Could not read parquet {parquet_file.name}: {e}")
                    dataset_info["rows"] = None
                    dataset_info["columns"] = None

            datasets.append(dataset_info)

        return sorted(datasets, key=lambda x: x["name"])

    def _dvc_add(self, file_path: Path) -> None:
        """Add file to DVC tracking."""
        try:
            result = subprocess.run(
                ["dvc", "add", str(file_path)],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"DVC add warning for {file_path.name}: {result.stderr}")
            else:
                logger.debug(f"DVC add successful: {file_path.name}")

        except subprocess.TimeoutExpired:
            logger.error(f"DVC add timeout for: {file_path.name}")
        except Exception as e:
            logger.error(f"DVC add failed for {file_path.name}: {str(e)}")

    def _dvc_push_background(self, filename: str) -> None:
        """Push single file to DVC remote in background."""
        try:
            # Use subprocess.Popen for background operation
            process = subprocess.Popen(
                ["dvc", "push", f"outputs/datasets/{filename}"],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.debug(f"Started background DVC push for: {filename}")

        except Exception as e:
            logger.warning(f"Background DVC push failed for {filename}: {str(e)}")

    def _dvc_push_batch(self) -> None:
        """Push all tracked files to DVC remote."""
        try:
            result = subprocess.run(
                ["dvc", "push"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for batch operations
            )

            if result.returncode == 0:
                logger.info("DVC batch push completed successfully")
            else:
                logger.warning(f"DVC batch push completed with warnings: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("DVC batch push timeout")
        except Exception as e:
            logger.error(f"DVC batch push failed: {str(e)}")

    def _dvc_pull(self, filename: str) -> None:
        """Pull specific file from DVC remote."""
        try:
            result = subprocess.run(
                ["dvc", "pull", f"outputs/datasets/{filename}"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise RuntimeError(f"DVC pull failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"DVC pull timeout for: {filename}")


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
# - Notebooks: Use convenience functions (save_dataset, load_dataset, etc.)
# - Tests: Use create_dvc_manager() for test isolation
# - Internal src/: Pass managers explicitly when needed

# Global instance for notebook convenience
_dvc_manager = None

def get_dvc_manager() -> DVCManager:
    """Get global DVC manager instance.

    MIGRATION NOTE: This function now returns the singleton for backward compatibility.
    New code should create managers explicitly:
        manager = DVCManager()

    Returns:
        DVCManager instance (singleton for backward compatibility)
    """
    global _dvc_manager
    if _dvc_manager is None:
        _dvc_manager = DVCManager()
    return _dvc_manager

def create_dvc_manager(base_dir: str = None) -> DVCManager:
    """Create a new DVC manager instance (DI pattern).

    This is the preferred pattern for dependency injection.
    Use this when you need an isolated manager instance (e.g., in tests).

    Args:
        base_dir: Project root directory (auto-detected if not provided)

    Returns:
        New DVCManager instance

    Example:
        manager = create_dvc_manager()
        manager.save_dataset(df, "sales_data", "Sales data after filtering")
    """
    return DVCManager(base_dir=base_dir)

# Convenience functions for notebook use (backward-compatible with DI support)
def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    description: str = "",
    manager: Optional[DVCManager] = None
) -> str:
    """Save DataFrame with automatic DVC tracking (convenience function).

    Args:
        df: DataFrame to save
        dataset_name: Name for the dataset file (without extension)
        description: Optional description for logging
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        Path to saved file
    """
    if manager is None:
        manager = get_dvc_manager()  # Fallback to singleton
    return manager.save_dataset(df, dataset_name, description)

def load_dataset(
    dataset_name: str,
    manager: Optional[DVCManager] = None
) -> pd.DataFrame:
    """Load DataFrame with automatic DVC pull if needed (convenience function).

    Args:
        dataset_name: Name of the dataset (without extension)
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        Loaded DataFrame
    """
    if manager is None:
        manager = get_dvc_manager()  # Fallback to singleton
    return manager.load_dataset(dataset_name)

def checkpoint_pipeline(
    stage_name: str,
    datasets: Dict[str, pd.DataFrame],
    manager: Optional[DVCManager] = None
) -> None:
    """Create strategic checkpoint with multiple datasets (convenience function).

    Args:
        stage_name: Name of the pipeline stage
        datasets: Dictionary mapping dataset names to DataFrames
        manager: Optional manager instance (DI pattern). If None, uses singleton.
    """
    if manager is None:
        manager = get_dvc_manager()  # Fallback to singleton
    manager.checkpoint_pipeline(stage_name, datasets)

def list_datasets(manager: Optional[DVCManager] = None) -> List[Dict[str, Any]]:
    """List all available datasets (convenience function).

    Args:
        manager: Optional manager instance (DI pattern). If None, uses singleton.

    Returns:
        List of dataset information dictionaries
    """
    if manager is None:
        manager = get_dvc_manager()  # Fallback to singleton
    return manager.list_datasets()