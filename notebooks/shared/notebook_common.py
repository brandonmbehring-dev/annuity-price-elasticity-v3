"""
Shared Notebook Utilities - Setup, Paths, Reproducibility.

Centralizes common boilerplate that appears in every production notebook:
- sys.path configuration for project imports
- Random seed initialization for reproducibility
- Standard output paths for products
- Section display helpers for notebook organization

Design Notes:
    This module handles technical setup only. Business logic and
    educational explanations remain explicit in the notebooks themselves.
"""

import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

RANDOM_SEED = 42  # Fixed seed for reproducible bootstrap results


# =============================================================================
# PROJECT ROOT DETECTION
# =============================================================================


def get_project_root(cwd: Optional[str] = None) -> Path:
    """
    Auto-detect project root from current working directory.

    Handles various notebook execution contexts:
    - notebooks/production/rila_6y20b/
    - notebooks/production/rila_1y10b/
    - notebooks/production/fia/
    - notebooks/eda/rila_6y20b/
    - notebooks/archive/*/
    - notebooks/
    - project root (fallback)

    Parameters
    ----------
    cwd : str, optional
        Current working directory. If None, uses os.getcwd().

    Returns
    -------
    Path
        Absolute path to project root directory.

    Raises
    ------
    RuntimeError
        If project root cannot be determined (src/ not found).

    Examples
    --------
    >>> root = get_project_root()
    >>> (root / 'src').exists()
    True
    """
    if cwd is None:
        cwd = os.getcwd()

    # Pattern matching for notebook directory structures
    if "notebooks/production/rila" in cwd:
        project_root = Path(cwd).parents[2]
    elif "notebooks/production/fia" in cwd:
        project_root = Path(cwd).parents[2]
    elif "notebooks/eda/rila" in cwd:
        project_root = Path(cwd).parents[2]
    elif "notebooks/archive" in cwd:
        project_root = Path(cwd).parents[2]
    elif os.path.basename(cwd) == "notebooks":
        project_root = Path(cwd).parent
    else:
        # Fallback: assume running from project root
        project_root = Path(cwd)

    # Validate detection worked
    src_path = project_root / "src"
    if not src_path.exists():
        raise RuntimeError(
            f"sys.path setup failed: 'src' package not found at {project_root}/src\n"
            f"Current directory: {cwd}\n"
            "This indicates the sys.path detection logic needs adjustment "
            "for your directory structure."
        )

    return project_root


def setup_sys_path(project_root: Optional[Path] = None) -> Path:
    """
    Configure sys.path for project imports.

    Parameters
    ----------
    project_root : Path, optional
        Project root path. If None, auto-detects.

    Returns
    -------
    Path
        Project root path used.
    """
    if project_root is None:
        project_root = get_project_root()

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return project_root


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def initialize_reproducibility(seed: int = RANDOM_SEED) -> None:
    """
    Initialize all random number generators for reproducibility.

    Sets seeds for:
    - numpy.random
    - Python's random module

    Parameters
    ----------
    seed : int
        Random seed value. Default is 42.

    Notes
    -----
    This must be called BEFORE any stochastic operations (bootstrap sampling,
    data shuffling, etc.) to ensure reproducible results across runs.
    """
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# SETUP ORCHESTRATION
# =============================================================================


@dataclass
class NotebookEnvironment:
    """Container for notebook environment information."""

    project_root: Path
    seed: int
    current_date: str
    current_time: datetime
    warnings_suppressed: bool


def setup_notebook(
    seed: int = RANDOM_SEED,
    suppress_warnings: bool = True,
) -> NotebookEnvironment:
    """
    Standard notebook initialization.

    Performs all common setup steps:
    1. Suppress warnings (optional)
    2. Detect and configure project root
    3. Set up sys.path for imports
    4. Initialize reproducibility seeds

    Parameters
    ----------
    seed : int
        Random seed for reproducibility. Default 42.
    suppress_warnings : bool
        Whether to suppress all warnings. Default True.

    Returns
    -------
    NotebookEnvironment
        Dataclass containing environment information.

    Examples
    --------
    >>> env = setup_notebook()
    >>> print(f"Project root: {env.project_root}")
    >>> print(f"Seed: {env.seed}")
    """
    # Suppress warnings if requested
    if suppress_warnings:
        warnings.filterwarnings("ignore")

    # Setup project path
    project_root = setup_sys_path()

    # Initialize reproducibility
    initialize_reproducibility(seed)

    # Capture time information
    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")

    return NotebookEnvironment(
        project_root=project_root,
        seed=seed,
        current_date=current_date,
        current_time=current_time,
        warnings_suppressed=suppress_warnings,
    )


# =============================================================================
# OUTPUT PATHS
# =============================================================================


def get_output_paths(
    product_code: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Get standard output paths for a product.

    Parameters
    ----------
    product_code : str
        Product identifier (e.g., "6Y20B", "1Y10B").
    project_root : Path, optional
        Project root path. If None, auto-detects.

    Returns
    -------
    Dict[str, Path]
        Dictionary with paths:
        - datasets: Path to datasets directory
        - bi_team: Path to BI team outputs
        - visualizations: Path to visualization outputs

    Examples
    --------
    >>> paths = get_output_paths("6Y20B")
    >>> paths["datasets"]
    PosixPath('/path/to/outputs/datasets')
    """
    if project_root is None:
        project_root = get_project_root()

    # Determine dataset subdirectory based on product
    if product_code.upper() == "6Y20B":
        datasets_dir = project_root / "outputs" / "datasets"
        bi_team_dir = project_root / "outputs" / "rila_6y20b" / "bi_team"
    elif product_code.upper() == "1Y10B":
        datasets_dir = project_root / "outputs" / "datasets_1y10b"
        bi_team_dir = project_root / "outputs" / "rila_1y10b" / "bi_team"
    elif product_code.upper() == "10Y20B":
        datasets_dir = project_root / "outputs" / "datasets_10y20b"
        bi_team_dir = project_root / "outputs" / "rila_10y20b" / "bi_team"
    else:
        # Generic fallback
        datasets_dir = project_root / "outputs" / f"datasets_{product_code.lower()}"
        bi_team_dir = project_root / "outputs" / f"rila_{product_code.lower()}" / "bi_team"

    return {
        "datasets": datasets_dir,
        "bi_team": bi_team_dir,
        "visualizations": bi_team_dir,
        "project_root": project_root,
    }


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def display_section(title: str, level: int = 2) -> None:
    """
    Display a markdown section header.

    For use in notebooks to provide clear section breaks.

    Parameters
    ----------
    title : str
        Section title text.
    level : int
        Markdown heading level (1-6). Default 2.

    Examples
    --------
    >>> display_section("Data Loading")
    ## Data Loading
    >>> display_section("Validation", level=3)
    ### Validation
    """
    prefix = "#" * level
    print(f"\n{prefix} {title}\n")


def print_environment_summary(env: NotebookEnvironment) -> None:
    """
    Print a summary of the notebook environment.

    Parameters
    ----------
    env : NotebookEnvironment
        Environment dataclass from setup_notebook().
    """
    print(f"Project root: {env.project_root}")
    print(f"Random seed initialized: {env.seed}")
    print(f"  All bootstrap operations will be reproducible across runs.")
    print(f"Date: {env.current_date}")
    if env.warnings_suppressed:
        print("  Warnings suppressed for clean output.")


def print_data_summary(
    name: str,
    df,
    date_col: str = "date",
) -> None:
    """
    Print a standard data loading summary.

    Parameters
    ----------
    name : str
        Dataset name for display.
    df : pd.DataFrame
        Loaded DataFrame.
    date_col : str
        Name of date column. Default "date".
    """
    print(f"{name} loading completed:")
    print(f"   Total records: {df.shape[0]:,}")
    print(f"   Total columns: {df.shape[1]}")

    if date_col in df.columns:
        print(f"   Date range: {df[date_col].min()} to {df[date_col].max()}")
